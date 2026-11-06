"""Tracker port over OC-SORT from roboflow `trackers` (Apache-2.0), the same
pinned package as ByteTrack.

Motion and IoU only: a Kalman filter on the box, IoU plus a direction
consistency term (observation-centric momentum), and a second pass against each
lost track's last observed box; no appearance features (spec 3, R3) and never
the image. Parameters are the library's defaults, which are the OC-SORT paper's
(det_thresh 0.6, max_age 30, min_hits 3, iou 0.3, delta_t 3, inertia 0.2),
except frame_rate, which scales the lost track buffer to the source, and the
overrides a config names (tracker_params):

- high_conf_det_threshold 0.6: lower boxes are dropped (no low-score stage)
- minimum_iou_threshold 0.3: weakest IoU that still matches a box to a track
- direction_consistency_weight 0.2: weight of the motion direction term
- delta_t 3: frames back for that direction
- minimum_consecutive_frames 3: a track is reported from its third matched frame
- lost_track_buffer 30: frames (at 30 fps) a lost track waits to be matched again

Track ids are this instance's counter, from 0: a new run starts again.
"""

import inspect

import numpy as np
import supervision as sv
import trackers

DEFAULTS = {
    name: p.default
    for name, p in inspect.signature(trackers.OCSORTTracker).parameters.items()
    if name in {
        "lost_track_buffer", "minimum_consecutive_frames", "minimum_iou_threshold",
        "direction_consistency_weight", "high_conf_det_threshold", "delta_t",
    }
}  # fmt: skip


class OcSortTracker:
    def __init__(self, frame_rate, **params):
        if unknown := sorted(set(params) - set(DEFAULTS)):
            raise ValueError(f"ocsort has no parameters {unknown}; it takes {sorted(DEFAULTS)}")
        self.parameters = {**DEFAULTS, **params, "frame_rate": frame_rate}
        self._tracker = trackers.OCSORTTracker(**self.parameters)

    def update(self, boxes, scores, frame_index):
        detections = sv.Detections(
            xyxy=np.asarray(boxes, float).reshape(-1, 4), confidence=np.asarray(scores, float)
        )
        out = self._tracker.update(detections)
        confirmed = out.tracker_id >= 0
        return out.tracker_id[confirmed].astype(int), out.xyxy[confirmed].reshape(-1, 4)
