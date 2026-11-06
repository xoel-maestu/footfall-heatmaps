"""A PersonDetector double that returns the tiny scene's ground-truth boxes of
each frame: the pipeline runs end to end with no model and no GPU."""

import numpy as np

from space_usage_heatmaps.perception.domain.frame import Detections


class GroundTruthDetector:
    input_size = 640  # what the test configs ask for

    def __init__(self, rows, camera=7):
        self.by_frame = {}
        for c, _, frame, x, y, w, h, *_ in rows:
            if int(c) == camera:
                self.by_frame.setdefault(int(frame), []).append([x, y, x + w, y + h])

    def detect(self, frame):
        boxes = np.array(self.by_frame.get(frame.index, []), float).reshape(-1, 4)
        return Detections(boxes, np.full(len(boxes), 0.9))
