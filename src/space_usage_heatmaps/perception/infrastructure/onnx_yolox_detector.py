"""PersonDetector over the official YOLOX ONNX models (Megvii, Apache-2.0) with ONNX Runtime.

Pre- and post-processing are the official demo's (YOLOX
demo/ONNXRuntime/onnx_inference.py with yolox/data/data_augment.py preproc and
yolox/utils/demo_utils.py), reimplemented in numpy so the YOLOX package is not
a dependency:

- letterbox: resize by r = min(S / h, S / w) with bilinear interpolation, paste
  at the top left of an S x S image filled with 114; BGR as decoded, no mean or
  std (the release models take raw 0-255 pixels), CHW float32;
- decode: per stride 8, 16, 32, xy = (out + grid) * stride, wh = exp(out) * stride;
  class scores = objectness * class probability; cx cy w h -> x1 y1 x2 y2, / r
  (source pixels, before NMS);
- class-agnostic NMS (the demo's default): each box takes its best class,
  score > 0.1, IoU > 0.45 suppressed (areas with the demo's +1); then only
  boxes whose best class is person (COCO 0) are kept;
- then clipped to the image (not in the demo, which draws them unclipped).

Only the model file goes through ONNX Runtime; nothing else in it is run.
"""

import logging
import re

import cv2
import numpy as np

from space_usage_heatmaps.perception.domain.frame import Detections

SCORE_THR = 0.1  # the official ONNX demo's multiclass_nms score_thr
NMS_THR = 0.45  # and its nms_thr
PERSON = 0  # COCO class id
STRIDES = (8, 16, 32)
CUDA, CPU = "CUDAExecutionProvider", "CPUExecutionProvider"
# What ONNX Runtime's messages say when the GPU runs out of memory during run:
# the BFC arena, a CUDA allocation, cuDNN or cuBLAS workspaces.
OUT_OF_MEMORY = re.compile(r"Failed to allocate memory|out of memory|_ALLOC_FAILED")

log = logging.getLogger(__name__)


def preprocess(image, size):
    """(3, size, size) float32 letterboxed input and the resize ratio."""
    padded = np.full((size, size, 3), 114, np.uint8)
    r = min(size / image.shape[0], size / image.shape[1])
    w, h = int(image.shape[1] * r), int(image.shape[0] * r)
    padded[:h, :w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), r


def _grids(size):
    grids, strides = [], []
    for s in STRIDES:
        xv, yv = np.meshgrid(np.arange(size // s), np.arange(size // s))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((len(grids[-1]), 1), s))
    return np.concatenate(grids), np.concatenate(strides)


def decode(output, size, grids=None):
    """Raw (1, N, 85) output -> (N, 4) x1 y1 x2 y2 boxes in input pixels, (N, 80) scores."""
    grid, stride = grids or _grids(size)
    p = output[0]
    xy = (p[:, :2] + grid) * stride
    wh = np.exp(p[:, 2:4]) * stride
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
    return boxes, p[:, 4:5] * p[:, 5:]


def nms(boxes, scores, thr):
    """The demo's single-class NMS: indices kept, best first."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        w = np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1)
        inter = w * h
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(overlap <= thr)[0] + 1]
    return keep


def engine_of(providers):
    """The provider a session runs on; anything but CUDA is logged as a fallback."""
    if providers[0] != CUDA:
        log.warning("ONNX Runtime is running YOLOX on %s, not CUDA: slow", providers[0])
    return providers[0]


def recipe():
    """What the boxes depend on besides the model file, for the track cache."""
    return {
        "preprocess": "yolox letterbox, top-left, pad 114, BGR 0-255",
        "postprocess": "boxes clipped to the image",
        "score_thr": SCORE_THR,
        "nms_thr": NMS_THR,
        "nms": "class-agnostic",
        "classes": [PERSON],
    }


def load(path):
    return OnnxYoloxDetector.load(path)


class OnnxYoloxDetector:
    def __init__(self, session, cpu_session=None, engine=None):
        """session: an onnxruntime.InferenceSession of a YOLOX model, or a test double.
        cpu_session(): the same model on CPUExecutionProvider, built only if the
        GPU runs out of memory during a run (spec 4). engine: session's provider.
        providers: every provider this detector has run on, in order."""
        self._session, self._cpu_session = session, cpu_session
        self.engine = engine
        self.providers = [engine] if engine else []
        inp = session.get_inputs()[0]
        self._input, self.input_size = inp.name, int(inp.shape[-1])
        self._grids = _grids(self.input_size)

    @classmethod
    def load(cls, path):
        """ONNX Runtime on CUDA, falling back to CPU (logged), at load or on out of memory."""
        import onnxruntime

        onnxruntime.preload_dlls()  # the CUDA and cuDNN libraries of the nvidia-* wheels
        onnxruntime.disable_telemetry_events()  # privacy R1: no usage events leave the process
        session = onnxruntime.InferenceSession(str(path), providers=[CUDA, CPU])
        return cls(
            session,
            cpu_session=lambda: onnxruntime.InferenceSession(str(path), providers=[CPU]),
            engine=engine_of(session.get_providers()),
        )

    def _run(self, x):
        feed = {self._input: x[None]}
        try:
            return self._session.run(None, feed)[0]
        except Exception as e:  # onnxruntime's RuntimeException is not a RuntimeError
            if self.engine == CPU or self._cpu_session is None or not OUT_OF_MEMORY.search(str(e)):
                raise
        log.warning("ONNX Runtime ran out of GPU memory on %s; retrying on CPU: slow", self.engine)
        self._session, self.engine = self._cpu_session(), CPU
        self.providers.append(CPU)
        return self._session.run(None, feed)[0]

    def detect(self, frame):
        x, r = preprocess(frame.image, self.input_size)
        out = self._run(x)
        boxes, scores = decode(out, self.input_size, self._grids)
        boxes = boxes / r  # to source pixels before NMS, as the demo
        cls = scores.argmax(1)
        best = scores[np.arange(len(cls)), cls]
        valid = best > SCORE_THR
        boxes, best, cls = boxes[valid], best[valid], cls[valid]
        keep = [i for i in nms(boxes, best, NMS_THR) if cls[i] == PERSON]
        h, w = frame.image.shape[:2]
        # Clipped after NMS, so the NMS stays the demo's.
        kept = np.clip(boxes[keep].reshape(-1, 4), 0, [w, h, w, h])
        return Detections(kept, best[keep].astype(float))
