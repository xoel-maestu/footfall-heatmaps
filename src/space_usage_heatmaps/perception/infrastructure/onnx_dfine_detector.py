"""PersonDetector over D-FINE (Peterande/D-FINE, Apache-2.0) exported to ONNX, with ONNX Runtime.

Pre- and post-processing are the official ONNX example's (D-FINE
tools/inference/onnx_inf.py), in numpy and Pillow so torch is not needed:

- letterbox: BGR -> RGB, resize by r = min(S / w, S / h) to (int(w r), int(h r))
  with Pillow's bilinear filter, paste centred on an S x S black image
  (pad_w = (S - w r) // 2, pad_h likewise); to 0-1 (ToTensor), CHW float32;
- the model takes the image and orig_target_sizes = [[S, S]] (the letterboxed
  size, as the example) and returns its top 300 (labels, x1 y1 x2 y2 boxes in
  input pixels, scores); a DETR has no NMS;
- boxes back to source pixels: (box - pad) / r, as the example's draw().

Then only label person (COCO index 0: the exported postprocessor returns
contiguous 0-79 labels, before any category-id remap) with score > SCORE_THR
(the example draws at 0.4; the pipeline keeps low scores for ByteTrack's
second association, as for YOLOX), clipped to the image.
"""

import cv2
import numpy as np
from PIL import Image

from space_usage_heatmaps.perception.domain.frame import Detections
from space_usage_heatmaps.perception.infrastructure.onnx_session import OnnxDetector, clip_to_image

SCORE_THR = 0.1  # the YOLOX recipe's: low boxes feed ByteTrack's second association
PERSON = 0  # COCO contiguous index


def preprocess(image, size):
    """(3, size, size) float32 input, the resize ratio and the (left, top) padding."""
    h, w = image.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(w * r), int(h * r)
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    resized = np.asarray(rgb.resize((nw, nh), Image.BILINEAR))
    pad_w, pad_h = (size - nw) // 2, (size - nh) // 2
    padded = np.zeros((size, size, 3), np.float32)
    padded[pad_h : pad_h + nh, pad_w : pad_w + nw] = resized / np.float32(255)
    return np.ascontiguousarray(padded.transpose(2, 0, 1)), r, (pad_w, pad_h)


def postprocess(labels, boxes, scores, ratio, pad, shape):
    """The model's outputs for one image -> person Detections in source pixels."""
    labels, boxes, scores = labels[0], boxes[0], scores[0]
    keep = (labels == PERSON) & (scores > SCORE_THR)
    pad_w, pad_h = pad
    out = (boxes[keep].astype(float) - [pad_w, pad_h, pad_w, pad_h]) / ratio
    return Detections(clip_to_image(out, shape), scores[keep].astype(float))


def recipe():
    """What the boxes depend on besides the model file, for the track cache."""
    return {
        "preprocess": "dfine letterbox, centred, pad 0, RGB 0-1, Pillow bilinear",
        "postprocess": "top 300, (box - pad) / ratio, clipped to the image",
        "pillow": _version("pillow"),
        "score_thr": SCORE_THR,
        "nms": "none",
        "classes": [PERSON],
    }


def _version(dist):
    from importlib import metadata

    return metadata.version(dist)


def load(path):
    return OnnxDfineDetector.load(path)


class OnnxDfineDetector(OnnxDetector):
    def detect(self, frame):
        x, r, pad = preprocess(frame.image, self.input_size)
        size = np.array([[self.input_size, self.input_size]], np.int64)
        labels, boxes, scores = self._run({self._input: x[None], "orig_target_sizes": size})
        return postprocess(labels, boxes, scores, r, pad, frame.image.shape)
