import numpy as np

from footfall_heatmaps.perception.infrastructure.bytetrack_tracker import ByteTrackTracker


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
    out = run(ByteTrackTracker(frame_rate=30), [[walk(10, f)] for f in range(10)])

    assert len(out[0][0]) == 0  # unconfirmed on its first frame: not reported
    ids = np.concatenate([ids for ids, _ in out[1:]])
    assert len(ids) == 9 and set(ids.tolist()) == {0}
    np.testing.assert_allclose(out[5][1], [walk(10, 5)], atol=3)


def test_two_people_get_two_ids():
    out = run(ByteTrackTracker(frame_rate=30), [[walk(10, f), walk(300, f)] for f in range(5)])

    assert sorted(out[4][0].tolist()) == [0, 1]


def test_a_box_below_the_activation_score_starts_no_track():
    out = run(ByteTrackTracker(frame_rate=30), [[walk(10, f)] for f in range(5)], score=0.5)

    assert all(len(ids) == 0 for ids, _ in out)


def test_no_boxes_gives_empty_arrays():
    ids, boxes = ByteTrackTracker(frame_rate=30).update(np.empty((0, 4)), np.empty(0), 0)

    assert ids.shape == (0,) and boxes.shape == (0, 4)


def test_parameters_are_the_trackers_defaults_but_the_frame_rate():
    p = ByteTrackTracker(frame_rate=25).parameters

    assert p == {
        "lost_track_buffer": 30,
        "frame_rate": 25,
        "track_activation_threshold": 0.7,
        "minimum_consecutive_frames": 2,
        "minimum_iou_threshold": 0.1,
        "high_conf_det_threshold": 0.6,
    }
