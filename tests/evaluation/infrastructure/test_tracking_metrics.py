"""HOTA and IDF1 of tracked boxes against the camera's ground-truth boxes."""

import numpy as np
import pytest

from space_usage_heatmaps.evaluation.domain.dataset_reader import Samples
from space_usage_heatmaps.evaluation.domain.tracks import Tracks
from space_usage_heatmaps.evaluation.infrastructure.tracking_metrics import (
    pooled,
    tracking_metrics,
)

N = 20  # frames


def truth():
    """Two people, 20 frames, boxes x, y, w, h (the dataset's format)."""
    rows = [(p, f, 10.0 + 100 * p + f, 20.0, 30.0, 80.0) for f in range(N) for p in (1, 2)]
    ids, frames, *box = map(np.array, zip(*rows, strict=True))
    return Samples(ids, frames, np.column_stack(box), np.zeros((len(ids), 2)))


def as_tracks(s, ids):
    xyxy = s.boxes.copy()
    xyxy[:, 2:] += xyxy[:, :2]
    return Tracks(np.asarray(ids), s.frames, xyxy)


def test_tracks_equal_to_the_truth_score_one():
    s = truth()

    m = tracking_metrics(s, as_tracks(s, s.track_ids + 40))

    assert m["hota"] == pytest.approx(1.0) and m["idf1"] == pytest.approx(1.0)


def test_an_id_switch_halfway_costs_identity_but_not_detection():
    s = truth()
    ids = np.where((s.track_ids == 2) & (s.frames >= N // 2), 9, s.track_ids)

    m = tracking_metrics(s, as_tracks(s, ids))

    assert m["deta"] == pytest.approx(1.0)
    assert m["idf1"] == pytest.approx(0.75)  # half of one of the two people is mis-identified
    assert m["hota"] < 0.95


def test_missing_boxes_lower_detection_and_boxes_outside_the_truth_frames_are_ignored():
    s = truth()
    t = as_tracks(s, s.track_ids)
    keep = t.track_ids == 1
    late = Tracks(np.array([5]), np.array([N + 3]), np.array([[0.0, 0, 10, 10]]))
    t = Tracks(
        np.concatenate([t.track_ids[keep], late.track_ids]),
        np.concatenate([t.frames[keep], late.frames]),
        np.concatenate([t.boxes[keep], late.boxes]),
    )

    m = tracking_metrics(s, t)

    assert m["deta"] == pytest.approx(0.5) and m["idf1"] == pytest.approx(2 / 3)


def test_pooled_is_the_mean_over_scenes():
    assert pooled([{"hota": 0.4, "idf1": 0.5}, {"hota": 0.6, "idf1": 0.7}]) == pytest.approx(
        {"hota": 0.5, "idf1": 0.6}
    )
