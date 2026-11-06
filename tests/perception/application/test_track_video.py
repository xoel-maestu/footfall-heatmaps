import numpy as np

from space_usage_heatmaps.perception.application.track_video import collect_tracks, track_video
from tests.perception.fakes import EchoTracker, ListFrameSource, ScriptedDetector

IMAGES = [np.zeros((120, 160, 3), np.uint8) for _ in range(4)]


def test_each_frame_goes_through_detector_then_tracker():
    tracker = EchoTracker()

    out = list(track_video(ListFrameSource(IMAGES), ScriptedDetector(), tracker))

    assert [t.frame_index for t in out] == [0, 1, 2, 3]
    assert [c[0] for c in tracker.calls] == [0, 1, 2, 3]
    assert out[2].track_ids.tolist() == [7]
    np.testing.assert_allclose(out[2].boxes, [[28, 30, 48, 90]])


def test_the_tracker_never_receives_the_image():
    tracker = EchoTracker()

    list(track_video(ListFrameSource(IMAGES), ScriptedDetector(), tracker))

    for _, boxes, scores in tracker.calls:
        assert (boxes.shape, scores.shape) == ((1, 4), (1,))


def test_collect_tracks_gives_one_row_per_tracked_box():
    ids, frames, boxes = collect_tracks(
        track_video(ListFrameSource(IMAGES), ScriptedDetector(), EchoTracker())
    )

    assert ids.tolist() == [7, 7, 7, 7]
    assert frames.tolist() == [0, 1, 2, 3]
    assert boxes.shape == (4, 4)


def test_collect_tracks_of_no_boxes_is_empty_with_the_right_shapes():
    ids, frames, boxes = collect_tracks(iter([]))

    assert (ids.shape, frames.shape, boxes.shape) == ((0,), (0,), (0, 4))


def test_detect_then_track_is_track_video():
    """The two halves, run apart (as when detections are replayed from a cache),
    give what track_video gives."""
    from space_usage_heatmaps.perception.application.track_video import (
        detect_video,
        track_detections,
    )

    detections = list(detect_video(ListFrameSource(IMAGES), ScriptedDetector()))
    apart = list(track_detections(detections, EchoTracker()))
    whole = list(track_video(ListFrameSource(IMAGES), ScriptedDetector(), EchoTracker()))

    assert [i for i, _ in detections] == [0, 1, 2, 3]
    assert [t.frame_index for t in apart] == [t.frame_index for t in whole]
    for a, b in zip(apart, whole, strict=True):
        np.testing.assert_array_equal(a.boxes, b.boxes)


def _detections(scores_by_frame):
    from space_usage_heatmaps.perception.domain.frame import Detections

    return [
        (i, Detections(np.tile([[0.0, 0.0, 10.0, 20.0]], (len(s), 1)), np.array(s, float)))
        for i, s in enumerate(scores_by_frame)
    ]


def test_select_detections_keeps_every_stride_th_frame_and_scores_above_the_threshold():
    from space_usage_heatmaps.perception.application.track_video import select_detections

    cached = _detections([[0.05, 0.1, 0.3], [0.9], [0.2, 0.1], [0.5], []])

    out = list(select_detections(cached, stride=2, min_score=0.1))

    assert [i for i, _ in out] == [0, 2, 4]  # frames 1 and 3 are skipped by the stride
    assert [d.scores.tolist() for _, d in out] == [[0.3], [0.2], []]  # strictly above, as NMS
    assert all(d.boxes.shape == (len(d.scores), 4) for _, d in out)


def test_select_detections_with_no_change_is_the_identity():
    from space_usage_heatmaps.perception.application.track_video import select_detections

    cached = _detections([[0.15, 0.3], [0.9]])

    out = list(select_detections(cached, stride=1, min_score=0.1))

    assert [i for i, _ in out] == [0, 1]
    assert [d.scores.tolist() for _, d in out] == [[0.15, 0.3], [0.9]]


def _tracked(rows):
    """rows: frame index -> {track id: x1}; boxes 10 x 20 px."""
    from space_usage_heatmaps.perception.domain.frame import TrackedBoxes

    return [
        TrackedBoxes(
            i,
            np.array(list(tracks), int),
            np.array([[x, 0.0, x + 10, 20.0] for x in tracks.values()]).reshape(-1, 4),
        )
        for i, tracks in rows
    ]


def test_fill_stride_interpolates_between_two_processed_frames_of_a_track():
    from space_usage_heatmaps.perception.application.track_video import fill_stride

    out = list(fill_stride(_tracked([(0, {1: 0.0}), (2, {1: 8.0}), (4, {1: 12.0})]), 2))

    assert [t.frame_index for t in out] == [0, 1, 2, 3, 4, 5]
    assert [t.boxes[:, 0].tolist() for t in out] == [[0], [4], [8], [10], [12], [12]]
    assert all(t.track_ids.tolist() == [1] for t in out)


def test_fill_stride_holds_a_track_that_is_not_seen_next():
    from space_usage_heatmaps.perception.application.track_video import fill_stride

    # Track 2 ends at frame 0, track 1 starts at 3: each holds for its own stride.
    out = list(fill_stride(_tracked([(0, {2: 5.0}), (3, {1: 0.0}), (6, {})]), 3))

    assert [t.frame_index for t in out] == list(range(9))
    assert [t.track_ids.tolist() for t in out] == [[2]] * 3 + [[1]] * 3 + [[]] * 3
    assert out[2].boxes[:, 0].tolist() == [5.0]


def test_fill_stride_keeps_dwell_time_correct():
    """Every processed sample stands for stride frames, so the filled samples
    at 1 / fps give the dwell of the processed ones at stride / fps."""
    from space_usage_heatmaps.perception.application.track_video import fill_stride

    rng = np.random.default_rng(0)
    rows = [
        (i, {k: float(rng.uniform(0, 50)) for k in range(5) if rng.random() < 0.7})
        for i in range(0, 60, 3)
    ]
    processed = sum(len(r) for _, r in rows)

    filled = sum(len(t.track_ids) for t in fill_stride(_tracked(rows), 3))

    assert filled == 3 * processed


def test_fill_stride_of_one_is_the_identity():
    from space_usage_heatmaps.perception.application.track_video import fill_stride

    rows = _tracked([(0, {1: 0.0}), (1, {1: 2.0, 4: 9.0})])

    out = list(fill_stride(rows, 1))

    assert [t.frame_index for t in out] == [0, 1]
    np.testing.assert_array_equal(out[1].boxes, rows[1].boxes)


def test_fill_stride_never_fills_past_the_last_frame_of_the_video():
    from space_usage_heatmaps.perception.application.track_video import fill_stride

    # A 6-frame video at stride 4: processed frames 0 and 4, frame 5 is the last.
    out = list(fill_stride(_tracked([(0, {1: 0.0}), (4, {1: 8.0})]), 4, end=6))

    assert [t.frame_index for t in out] == [0, 1, 2, 3, 4, 5]
