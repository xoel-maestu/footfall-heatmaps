"""PersonDetector over RF-DETR Nano (Roboflow rfdetr, Apache-2.0) as ONNX, on ONNX Runtime.

Pre- and post-processing are rfdetr's ONNX inference helpers' (rfdetr 1.11.2,
rfdetr/export/_onnx/inference.py with export/_runtime/preprocess.py and
decode.py), which match RFDETR.predict(); reimplemented in numpy and OpenCV:

- BGR -> RGB, 0-1 float, stretched to the model's S x S input (no letterbox)
  with bilinear half-pixel-centre interpolation and no antialiasing (OpenCV's
  INTER_LINEAR on float32, what torchvision's resize(antialias=False) does),
  then ImageNet mean and std, CHW float32;
- outputs dets (Q, 4) normalised cx cy w h and labels (Q, 91) logits; per-class
  sigmoid over every slot (the COCO checkpoint's sparse category ids, no
  background slot dropped, as predict()), the top Q query/class pairs by
  score, then score > SCORE_THR; boxes scaled to the source size and clipped.

Only label person (COCO category id 1) is kept. SCORE_THR is the YOLOX
recipe's 0.1 (the helpers default to 0.3), so ByteTrack's second association
gets low boxes. A DETR has no NMS.
"""

import cv2
import numpy as np

from space_usage_heatmaps.perception.domain.frame import Detections
from space_usage_heatmaps.perception.infrastructure.onnx_session import OnnxDetector

SCORE_THR = 0.1
PERSON = 1  # COCO category id: the checkpoint's 91 slots are the sparse ids
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(image, size):
    """(3, size, size) float32 input."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(((resized - MEAN) / STD).transpose(2, 0, 1), dtype=np.float32)


def postprocess(dets, logits, shape):
    """The model's outputs for one image -> person Detections in source pixels."""
    boxes, logits = dets[0], logits[0]
    q, c = logits.shape
    scores = 1 / (1 + np.exp(-logits.clip(-88, 88)))
    flat = scores.reshape(-1)
    top = np.lexsort((np.arange(flat.size), -flat))[:q]  # by score, ties by index
    top = top[(flat[top] > SCORE_THR) & (top % c == PERSON)]
    cx, cy, w, h = boxes[top // c].T
    hh, ww = shape[:2]
    scale = np.array([ww, hh, ww, hh], float)
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1) * scale
    return Detections(np.clip(xyxy, 0, scale).reshape(-1, 4), flat[top].astype(float))


def recipe():
    """What the boxes depend on besides the model file, for the track cache."""
    return {
        "preprocess": "rfdetr stretch to square, RGB 0-1, bilinear half-pixel, ImageNet mean/std",
        "postprocess": "sigmoid, top Q query/class pairs, boxes scaled and clipped",
        "score_thr": SCORE_THR,
        "nms": "none",
        "classes": [PERSON],
    }


def load(path):
    return OnnxRfdetrDetector.load(path)


class OnnxRfdetrDetector(OnnxDetector):
    def detect(self, frame):
        x = preprocess(frame.image, self.input_size)
        dets, logits = self._run({self._input: x[None]})[:2]
        return postprocess(dets, logits, frame.image.shape)
