import math

import numpy as np
import pytest

from space_usage_heatmaps.evaluation.domain.label_free import (
    label_free_blocks,
    label_free_metrics,
    pearson,
    spearman,
)

W, H = 1000, 500
EDGE = 0.02 * H  # 10 px


def _track(tid, frames, box):
    n = len(frames)
    return np.full(n, tid), np.asarray(frames), np.tile(np.asarray(box, float), (n, 1))


def _stack(*tracks):
    ids, frames, boxes = zip(*tracks, strict=True)
    return np.concatenate(ids), np.concatenate(frames), np.vstack(boxes)


LEFT = [0.0, 100.0, 40.0, 300.0]  # touches the left border
MIDDLE = [400.0, 100.0, 440.0, 300.0]  # far from every border


def test_tracks_that_enter_and_leave_at_the_edge_conserve_occupancy():
    # Present from the first frame, or born and dead at the border.
    ids, frames, boxes = _stack(
        _track(1, range(0, 10), MIDDLE),  # present at the start, ends at the last frame
        _track(2, range(3, 8), LEFT),
    )

    m = label_free_metrics(ids, frames, boxes, W, H, fps=10.0)

    assert m.conservation_error == 0.0
    assert m.interior_birth_fraction == 0.0  # track 2 is born at the border
    assert m.tracks == 2
    assert m.mean_track_length_s == pytest.approx((1.0 + 0.5) / 2)


def test_a_track_born_and_lost_inside_the_view_breaks_conservation():
    # Track 2 appears in the middle at frame 4 and vanishes there at frame 5:
    # edge entries minus exits stay at 1 while 2 people are active for 2 frames.
    ids, frames, boxes = _stack(_track(1, range(0, 10), MIDDLE), _track(2, [4, 5], MIDDLE))

    m = label_free_metrics(ids, frames, boxes, W, H, fps=10.0)

    assert m.conservation_error == pytest.approx(2 / 12)
    assert m.interior_birth_fraction == 1.0


def test_an_interior_death_keeps_the_balance_high_until_the_end():
    # Born at the border at frame 2, lost in the middle at frame 4: from frame 5
    # to 9 the balance says one person is there and nobody is active.
    ids, frames, boxes = _stack(
        _track(1, range(0, 10), MIDDLE), _track(2, [2, 3], LEFT), _track(2, [4], MIDDLE)
    )

    m = label_free_metrics(ids, frames, boxes, W, H, fps=10.0)

    assert m.conservation_error == pytest.approx(5 / 13)


def test_block_metrics_cut_the_frames_and_births_by_block():
    ids, frames, boxes = _stack(_track(1, range(0, 10), MIDDLE), _track(2, [6, 7], MIDDLE))

    b = label_free_blocks(ids, frames, boxes, W, H, fps=10.0, block_frames=5, n_blocks=2)

    assert b.conservation_error.tolist() == [0.0, pytest.approx(2 / 7)]
    assert math.isnan(b.interior_birth_fraction[0])  # no birth after the first frame
    assert b.interior_birth_fraction[1] == 1.0
    assert b.mean_track_length_s[0] == pytest.approx(1.0)  # tracks born in the block
    assert b.mean_track_length_s[1] == pytest.approx(0.2)


def test_spearman_ranks_with_ties_and_pearson_is_linear():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486833, abs=1e-6)
    assert pearson(x, 2 * x + 1) == pytest.approx(1.0)
    assert pearson(x, x**3) < 1.0


def test_correlations_skip_nan_pairs_and_need_variance():
    assert spearman([1, 2, np.nan, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert math.isnan(pearson([1, 1, 1], [1, 2, 3]))
