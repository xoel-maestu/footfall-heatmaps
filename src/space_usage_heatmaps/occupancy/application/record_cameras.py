"""Use case: many cameras of a site through one shared detector (improvements phase 5).

The feed yields (camera id, frame) in the order the cameras take turns (round
robin, decided by the infrastructure that reads the sources), and (camera id,
None) when that camera's source dropped. Per camera, as in record_camera:
privacy mask before its own tracker (R6), its own calibration and track
heights, floor points into the recorder (RecordSite or CalibrateSite). The
frame goes no further than the detector (R1). The recorder is closed when the
feed ends, until() turns true, or on an error.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from space_usage_heatmaps.evaluation.application.evaluate_oracle import camera_floor_points


@dataclass
class CameraPipe:
    mask: object  # PrivacyMask
    tracker: object  # Tracker
    cam: object  # CameraCalibration
    time_of: Callable  # frame -> UTC epoch seconds
    seconds: float  # dwell per tracked box: stride / fps
    heights: object = None  # auto-track: this camera's TrackHeights, made on its first frame


def record_cameras(feed, detector, pipes, config, recorder, until=None, on_frame=None):
    """pipes: camera id -> CameraPipe; on_frame(camera id) after each processed
    frame. How many frames ran, every camera together."""
    count = 0
    try:
        for camera_id, frame in feed:
            if frame is None:
                recorder.mark_incomplete(camera_id)
                continue
            p = pipes[camera_id]
            if p.heights is None:
                p.heights = config.track_heights()
            d = p.mask.apply(detector.detect(frame))
            ids, boxes = p.tracker.update(d.boxes, d.scores, frame.index)
            ids, boxes = np.asarray(ids, int), np.asarray(boxes, float).reshape(-1, 4)
            xs, ys, _ = camera_floor_points(
                p.cam,
                boxes,
                config.foot_point,
                config.person_height_m,
                config.min_implied_height_m,
                ids,
                np.full(len(ids), frame.index),
                p.heights,
            )
            recorder.add(camera_id, p.time_of(frame), ids, xs, ys, p.seconds)
            count += 1
            if on_frame is not None:
                on_frame(camera_id)
            if until is not None and until():
                break
    finally:
        recorder.close()
    return count
