"""Use case: record one camera into slots, frame by frame, as the frames arrive.

Each frame's detections go through the privacy mask before the tracker (R6);
the confirmed tracks' foot points go through the camera's calibration to the
floor, and from there into RecordSlots. The frame itself goes no further than
the detector (R1). The open slot is written as incomplete when the frames end,
or stop on an error.
"""

import numpy as np

from footfall_heatmaps.floorplan.domain.foot_point import camera_floor_points


def record_camera(frames, detector, mask, tracker, cam, config, recorder, time_of, seconds):
    """frames: an iterator of Frame; time_of(frame): its UTC epoch seconds;
    seconds: the dwell each tracked box adds (stride / fps). How many frames ran."""
    heights = config.track_heights()  # auto-track: each track's height so far
    count = 0
    try:
        for frame in frames:
            d = mask.apply(detector.detect(frame))
            ids, boxes = tracker.update(d.boxes, d.scores, frame.index)
            ids, boxes = np.asarray(ids, int), np.asarray(boxes, float).reshape(-1, 4)
            xs, ys, _ = camera_floor_points(
                cam,
                boxes,
                config.foot_point,
                config.person_height_m,
                config.min_implied_height_m,
                ids,
                np.full(len(ids), frame.index),
                heights,
            )
            recorder.add(time_of(frame), ids, xs, ys, seconds)
            count += 1
    finally:
        recorder.close()
    return count
