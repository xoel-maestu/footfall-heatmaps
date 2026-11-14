import numpy as np

from footfall_heatmaps.perception.infrastructure.ocsort_tracker import OcSortTracker


def walk(x0, frame):
    x = x0 + 3.0 * frame
    return [x, 50.0, x + 30, 150.0]


def run(tracker, boxes_per_frame, score=0.9):
    out = []
    for i, boxes in enumerate(boxes_per_frame):
        b = np.array(boxes, float).reshape(-1, 4)
        out.append(tracker.update(b, np.full(len(b), score), i))
    return out


def test_a_walking_box_keeps_one_id_once_confirmed():
    # Appearing after the first frames (SORT reports every match during the
    # first minimum_consecutive_frames of a run).
    frames = [[] for _ in range(5)] + [[walk(10, f)] for f in range(10)]

    out = run(OcSortTracker(frame_rate=30), frames)

    # Not reported on the frame that starts it; trackers 2.6.1 counts that
    # frame as the first of minimum_consecutive_frames.
    assert all(len(ids) == 0 for ids, _ in out[:6])
    ids = np.concatenate([ids for ids, _ in out[6:]])
    assert len(ids) == 9 and set(ids.tolist()) == {0}
    np.testing.assert_allclose(out[10][1], [walk(10, 5)], atol=3)


def test_two_people_get_two_ids():
    out = run(OcSortTracker(frame_rate=30), [[walk(10, f), walk(300, f)] for f in range(5)])

    assert sorted(out[4][0].tolist()) == [0, 1]


def test_a_box_below_the_detection_threshold_starts_no_track():
    out = run(OcSortTracker(frame_rate=30), [[walk(10, f)] for f in range(5)], score=0.5)

    assert all(len(ids) == 0 for ids, _ in out)


def test_a_short_gap_keeps_the_id():
    frames = [[walk(10, f)] if f not in (5, 6) else [] for f in range(12)]

    out = run(OcSortTracker(frame_rate=30), frames)

    assert {i for ids, _ in out for i in ids.tolist()} == {0}


def test_no_boxes_gives_empty_arrays():
    ids, boxes = OcSortTracker(frame_rate=30).update(np.empty((0, 4)), np.empty(0), 0)

    assert ids.shape == (0,) and boxes.shape == (0, 4)


def test_parameters_are_the_published_defaults_but_the_frame_rate():
    p = OcSortTracker(frame_rate=25).parameters

    assert p == {
        "lost_track_buffer": 30,
        "frame_rate": 25,
        "minimum_consecutive_frames": 3,
        "minimum_iou_threshold": 0.3,
        "direction_consistency_weight": 0.2,
        "high_conf_det_threshold": 0.6,
        "delta_t": 3,
    }


def test_a_lower_detection_threshold_is_passed_to_the_library():
    tracker = OcSortTracker(frame_rate=30, high_conf_det_threshold=0.1)
    frames = [[] for _ in range(5)] + [[walk(10, f)] for f in range(6)]

    out = run(tracker, frames, score=0.5)

    assert tracker.parameters["high_conf_det_threshold"] == 0.1
    assert {i for ids, _ in out for i in ids.tolist()} == {0}
