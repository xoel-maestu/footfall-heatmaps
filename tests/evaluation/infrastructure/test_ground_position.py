"""Ground position error of tracked boxes against annotated floor positions."""

import numpy as np
import pytest

from space_usage_heatmaps.evaluation.domain.dataset_reader import Samples
from space_usage_heatmaps.evaluation.domain.tracks import Tracks
from space_usage_heatmaps.evaluation.infrastructure.ground_position import matched_ground_error


def truth():
    # Two people in two frames; boxes x, y, w, h.
    return Samples(
        track_ids=np.array([1, 2, 1, 2]),
        frames=np.array([0, 0, 1, 1]),
        boxes=np.array([[0, 0, 10, 20], [100, 0, 10, 20], [2, 0, 10, 20], [102, 0, 10, 20.0]]),
        world=np.array([[0, 0], [5, 5], [0.1, 0], [5, 5.1]]),
    )


def test_each_truth_box_is_paired_with_the_overlapping_tracked_box():
    # Tracked boxes listed in another order, with other ids; floor points 0.1 m
    # and 0.3 m off for person 1, 0.2 m for person 2.
    tracks = Tracks(
        track_ids=np.array([9, 8, 9, 8]),
        frames=np.array([0, 0, 1, 1]),
        boxes=np.array([[100, 0, 110, 20], [0, 0, 10, 20], [101, 0, 111, 20], [2, 0, 12, 20.0]]),
    )
    xy = np.array([[5, 5.2], [0.1, 0], [5.2, 5.1], [0.1, 0.3]])

    e = matched_ground_error(truth(), tracks, xy)

    assert e["matched"] == 4 and e["truth_boxes"] == 4 and e["tracked_boxes"] == 4
    d = sorted([0.2, 0.1, 0.2, 0.3])
    assert e["median_m"] == pytest.approx(np.median(d))
    assert e["p90_m"] == pytest.approx(np.percentile(d, 90))


def test_boxes_below_the_iou_threshold_are_misses_not_errors():
    tracks = Tracks(np.array([9]), np.array([0]), np.array([[50, 0, 60, 20.0]]))

    e = matched_ground_error(truth(), tracks, np.array([[0.0, 0.0]]))

    assert e["matched"] == 0 and e["median_m"] is None and e["recall"] == 0.0


def test_errors_split_by_a_grouping_of_the_tracked_boxes():
    tracks = Tracks(
        track_ids=np.array([9, 8, 9, 8]),
        frames=np.array([0, 0, 1, 1]),
        boxes=np.array([[100, 0, 110, 20], [0, 0, 10, 20], [101, 0, 111, 20], [2, 0, 12, 20.0]]),
    )
    xy = np.array([[5, 5.2], [0.1, 0], [5.2, 5.1], [0.1, 0.3]])
    head = np.array([True, False, True, False])

    e = matched_ground_error(truth(), tracks, xy, groups={"head": head, "bottom": ~head})

    assert e["groups"]["head"]["matched"] == 2
    assert e["groups"]["head"]["median_m"] == pytest.approx(0.2)
    assert e["groups"]["bottom"]["median_m"] == pytest.approx(0.2)
    assert e["groups"]["bottom"]["p90_m"] == pytest.approx(np.percentile([0.1, 0.3], 90))
