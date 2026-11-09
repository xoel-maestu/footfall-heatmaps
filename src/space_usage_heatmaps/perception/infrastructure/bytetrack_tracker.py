"""Tracker port over ByteTrack from roboflow `trackers` (Apache-2.0), pinned in pyproject.toml.

Motion and IoU only: ByteTrack associates boxes by Kalman-predicted IoU in two
stages (high-score boxes, then low-score ones), with no appearance features
(privacy rule R3); it is never given the image. Parameters are the library's
defaults, read from its constructor, except frame_rate, which scales the lost
track buffer (30 frames at 30 fps) to the source:

- track_activation_threshold 0.7: a new track needs a box this confident
- high_conf_det_threshold 0.6: first-stage boxes; the rest go to the second stage
- minimum_iou_threshold 0.1: weakest IoU that still matches a box to a track
- minimum_consecutive_frames 2: a track is reported from its second matched frame
- lost_track_buffer 30: frames (at 30 fps) a lost track waits to be matched again

Track ids are this instance's counter, from 0: a new run starts again.
"""

import inspect

import numpy as np
import supervision as sv
import trackers

DEFAULTS = {
    name: p.default
    for name, p in inspect.signature(trackers.ByteTrackTracker).parameters.items()
    if name in {
        "lost_track_buffer", "track_activation_threshold", "minimum_consecutive_frames",
        "minimum_iou_threshold", "high_conf_det_threshold",
    }
}  # fmt: skip


class ByteTrackTracker:
    def __init__(self, frame_rate, **params):
        if unknown := sorted(set(params) - set(DEFAULTS)):
            raise ValueError(f"bytetrack has no parameters {unknown}; it takes {sorted(DEFAULTS)}")
        self.parameters = {**DEFAULTS, **params, "frame_rate": frame_rate}
        self._tracker = trackers.ByteTrackTracker(**self.parameters)

    def update(self, boxes, scores, frame_index):
        detections = sv.Detections(
            xyxy=np.asarray(boxes, float).reshape(-1, 4), confidence=np.asarray(scores, float)
        )
        out = self._tracker.update(detections)
        confirmed = out.tracker_id >= 0
        return out.tracker_id[confirmed].astype(int), out.xyxy[confirmed].reshape(-1, 4)
