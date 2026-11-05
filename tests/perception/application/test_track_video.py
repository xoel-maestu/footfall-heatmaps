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
