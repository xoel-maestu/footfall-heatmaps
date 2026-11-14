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

from footfall_heatmaps.floorplan.domain.foot_point import camera_floor_points
from footfall_heatmaps.perception.application.track_video import fill_between
from footfall_heatmaps.perception.domain.frame import TrackedBoxes


@dataclass
class CameraPipe:
    mask: object  # PrivacyMask
    tracker: object  # Tracker
    cam: object  # CameraCalibration
    time_of: Callable  # frame -> UTC epoch seconds
    seconds: float  # dwell per tracked box: stride / fps
    heights: object = None  # auto-track: this camera's TrackHeights, made on its first frame
    processed: int = 0  # frames of this camera through the detector so far


def record_cameras(feed, detector, pipes, config, recorder, until=None, on_frame=None):
    """pipes: camera id -> CameraPipe; on_frame(camera id, frame) after each
    processed frame, and with None for a drop. How many frames ran, every
    camera together.

    With a stride, each camera's skipped frames get their boxes back as the
    evaluation's fill_stride gives them (a box moves linearly to its next
    processed box, or stays), each worth one source frame of dwell. So the
    recorded dwell and passage are the evaluated recipe's. A camera's samples
    lag one processed frame; its last one is flushed when the feed ends."""
    stride = config.stride
    pending = {}  # camera id -> (its last processed TrackedBoxes, the frame's time)
    count = 0

    def emit(camera_id, a, b):
        p, (boxes_a, t_a) = pipes[camera_id], a
        frame_s = p.seconds / stride
        for t in fill_between(boxes_a, b, stride):
            k = t.frame_index - boxes_a.frame_index
            xs, ys, _ = camera_floor_points(
                p.cam,
                t.boxes.reshape(-1, 4),
                config.foot_point,
                config.person_height_m,
                config.min_implied_height_m,
                t.track_ids,
                np.full(len(t.track_ids), t.frame_index),
                p.heights,
            )
            recorder.add(camera_id, t_a + k * frame_s, t.track_ids, xs, ys, frame_s)

    try:
        for camera_id, frame in feed:
            if frame is None:
                recorder.mark_incomplete(camera_id)
                if on_frame is not None:
                    on_frame(camera_id, None)
                continue
            p = pipes[camera_id]
            if p.heights is None:
                p.heights = config.track_heights()
            d = p.mask.apply(detector.detect(frame))
            ids, boxes = p.tracker.update(d.boxes, d.scores, frame.index)
            ids, boxes = np.asarray(ids, int), np.asarray(boxes, float).reshape(-1, 4)
            # processed frames are stride source frames apart, whatever the source numbers them
            n = p.processed
            p.processed += 1
            current = TrackedBoxes(n * stride, ids, boxes)
            if camera_id in pending:
                emit(camera_id, pending[camera_id], current)
            pending[camera_id] = (current, p.time_of(frame))
            count += 1
            if on_frame is not None:
                on_frame(camera_id, frame)
            if until is not None and until():
                break
        for camera_id, a in pending.items():
            emit(camera_id, a, None)
    finally:
        recorder.close()
    return count
