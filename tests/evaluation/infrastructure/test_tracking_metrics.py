"""HOTA and IDF1 of tracked boxes against the camera's ground-truth boxes."""

import numpy as np
import pytest

from footfall_heatmaps.evaluation.domain.dataset_reader import Samples
from footfall_heatmaps.evaluation.domain.tracks import Tracks
from footfall_heatmaps.evaluation.infrastructure.tracking_metrics import (
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


def test_a_track_on_a_distractor_is_neither_rewarded_nor_penalised():
    # MOT17 marks static people, reflections and the like as distractors
    # (TrackEval's MOT17 preprocessing): a tracked box that matches one is
    # removed, and the distractor itself is never a missed person.
    s = truth()
    distractor = s.track_ids == 2
    marked = Samples(s.track_ids, s.frames, s.boxes, s.world, distractor=distractor)
    only_one = Samples(
        s.track_ids[~distractor], s.frames[~distractor], s.boxes[~distractor], s.world[~distractor]
    )

    with_box = tracking_metrics(marked, as_tracks(s, s.track_ids))
    without = tracking_metrics(marked, as_tracks(only_one, only_one.track_ids))

    assert with_box["hota"] == pytest.approx(1.0) and with_box["idf1"] == pytest.approx(1.0)
    assert without == pytest.approx(with_box)


def test_a_box_off_every_distractor_still_counts_as_a_false_positive():
    s = truth()
    marked = Samples(s.track_ids, s.frames, s.boxes, s.world, distractor=s.track_ids == 2)
    t = as_tracks(s, s.track_ids)
    far = t.boxes.copy()
    far[s.track_ids == 2] += [500.0, 0, 500.0, 0]

    m = tracking_metrics(marked, Tracks(t.track_ids, t.frames, far))

    assert m["deta"] == pytest.approx(0.5)


def test_a_box_on_a_car_beside_a_distractor_stays_a_false_positive():
    # TrackEval's MOT17 preprocessing matches tracked boxes against every truth
    # row: a box that overlaps a car (IoU 0.95) more than a distractor (IoU
    # 0.81) goes to the car, so it is kept and costs precision.
    s = truth()
    car = s.track_ids == 1  # person 1's rows play an unscored car
    distractor_boxes = s.boxes[car] + [0.0, 6.4, 0.0, 0.0]
    n, k = len(s.frames), int(car.sum())
    rows = Samples(
        np.r_[s.track_ids, np.full(k, 7)],
        np.r_[s.frames, s.frames[car]],
        np.r_[s.boxes, distractor_boxes],
        np.zeros((n + k, 2)),
        distractor=np.r_[np.zeros(n, bool), np.ones(k, bool)],
        scored=np.r_[~car, np.zeros(k, bool)],
    )
    t = as_tracks(s, s.track_ids)
    boxes = t.boxes.copy()
    boxes[car, 3] -= 4.0  # the car's box, 4 px shorter

    m = tracking_metrics(rows, Tracks(t.track_ids, t.frames, boxes))

    assert m["deta"] == pytest.approx(0.5)  # person 2 found, the car box a false positive
