"""Test doubles of the perception ports: frames, detections and tracks in memory."""

import numpy as np

from space_usage_heatmaps.perception.domain.frame import Detections, Frame


class ListFrameSource:
    def __init__(self, images, fps=10.0):
        self.images = images
        self.fps = fps

    def frames(self):
        for i, image in enumerate(self.images):
            yield Frame(i, i / self.fps, image)


class ScriptedDetector:
    """One person walking right, 4 px a frame, whatever the image shows."""

    def detect(self, frame):
        x = 20.0 + 4 * frame.index
        return Detections(np.array([[x, 30.0, x + 20, 90.0]]), np.array([0.9]))


class EchoTracker:
    """Track id = 7 for every box: enough to see what the use case passes through."""

    def __init__(self):
        self.calls = []

    def update(self, boxes, scores, frame_index):
        self.calls.append((frame_index, boxes, scores))
        return np.full(len(scores), 7), boxes
