import numpy as np
import pytest

from footfall_heatmaps.evaluation.domain.position_error import position_error


def test_errors_are_matched_by_track_and_frame_whatever_the_order():
    truth_ids, truth_frames = np.array([1, 1, 2, 2]), np.array([0, 1, 0, 1])
    truth_xy = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]])
    ids, frames = np.array([2, 1, 1, 3]), np.array([1, 1, 0, 0])  # (3, 0) has no truth
    xy = np.array([[6.0, 5.0], [1.0, 3.0], [0.0, 4.0], [9.0, 9.0]])

    e = position_error(ids, frames, xy, truth_ids, truth_frames, truth_xy)

    assert e.matched == 3 and e.unmatched == 1
    assert e.median == pytest.approx(3.0)
    assert e.p90 == pytest.approx(np.percentile([0.0, 3.0, 4.0], 90))


def test_no_match_gives_no_error():
    e = position_error(
        np.array([1]),
        np.array([0]),
        np.zeros((1, 2)),
        np.array([2]),
        np.array([0]),
        np.zeros((1, 2)),
    )

    assert (e.matched, e.median, e.p90) == (0, None, None)
