"""Identity-free passage by chord length (improvements plan phase 6, H2).

Each tracklet adds min(1, path length inside the cell / (pi * a / 4)) to a
cell: pi * a / 4 is the mean chord of a square of side a (Cauchy-Crofton),
0.39 m at a = 0.5 m.
"""

import math

import numpy as np
import pytest

from space_usage_heatmaps.occupancy.domain.chord import chord_passage, mean_chord
from space_usage_heatmaps.occupancy.domain.grid import GridSpec

SPEC = GridSpec(0.0, 0.0, 0.5, nx=8, ny=8)  # 4 m x 4 m
CHORD = math.pi * 0.5 / 4


def grid_of(spec, track_ids, frames, xs, ys):
    cells, _, weights = chord_passage(spec, track_ids, frames, xs, ys)
    out = np.zeros(spec.ny * spec.nx)
    np.add.at(out, cells, weights)
    return out.reshape(spec.ny, spec.nx)


def path(points, track=1, start=0):
    xy = np.asarray(points, float)
    n = len(xy)
    return np.full(n, track), np.arange(start, start + n), xy[:, 0], xy[:, 1]


def test_the_mean_chord_of_a_half_metre_cell_is_0_39_m():
    assert mean_chord(0.5) == pytest.approx(CHORD)
    assert round(mean_chord(0.5), 2) == 0.39


def test_a_straight_crossing_of_a_cell_counts_one():
    # Along row 2 (y = 1.25), from x = 0.5 to x = 1.0: exactly cell (2, 1), 0.5 m.
    g = grid_of(SPEC, *path([(0.5, 1.25), (0.75, 1.25), (1.0, 1.25)]))

    assert g[2, 1] == 1.0  # min(1, 0.5 / 0.39)
    assert g.sum() == 1.0


def test_a_path_is_clipped_exactly_at_the_cell_boundaries():
    # One segment from (0.25, 0.25) to (1.25, 0.25): 0.25 m in cell 0, 0.5 in
    # cell 1, 0.25 in cell 2, with no sample inside cell 1.
    g = grid_of(SPEC, *path([(0.25, 0.25), (1.25, 0.25)]))

    assert g[0, 0] == pytest.approx(0.25 / CHORD)
    assert g[0, 1] == 1.0
    assert g[0, 2] == pytest.approx(0.25 / CHORD)


def test_a_corner_clip_adds_a_fraction():
    # From (0.4, 0.6) to (0.6, 0.4), through the shared corner (0.5, 0.5):
    # cell (1, 0) holds (0.4, 0.6)-(0.5, 0.5), cell (0, 1) holds
    # (0.5, 0.5)-(0.6, 0.4), each 0.1 * sqrt(2) m; cell (0, 0) only its corner.
    g = grid_of(SPEC, *path([(0.4, 0.6), (0.6, 0.4)]))

    piece = 0.1 * math.sqrt(2) / CHORD
    assert g[1, 0] == pytest.approx(piece)
    assert g[0, 1] == pytest.approx(piece)
    assert g[0, 0] == pytest.approx(0.0)
    assert 0 < piece < 1


def test_a_diagonal_clip_of_one_cell_corner_is_its_length_over_the_chord():
    # From (0.4, 0.0) to (0.6, 0.2) passes the corner of cell (0, 0) and (0, 1):
    # (0, 0) holds (0.4, 0.0)-(0.5, 0.1), (0, 1) holds (0.5, 0.1)-(0.6, 0.2).
    g = grid_of(SPEC, *path([(0.4, 0.0), (0.6, 0.2)]))

    assert g[0, 0] == pytest.approx(0.1 * math.sqrt(2) / CHORD)
    assert g[0, 1] == pytest.approx(0.1 * math.sqrt(2) / CHORD)


def test_a_wandering_track_is_capped_at_one_per_cell():
    zigzag = [(0.1, 0.1), (0.4, 0.4), (0.1, 0.4), (0.4, 0.1), (0.1, 0.1)]
    g = grid_of(SPEC, *path(zigzag))

    assert g[0, 0] == 1.0
    assert g.sum() == 1.0


def test_a_stationary_track_adds_zero():
    # No path length, no passage: the formula gives min(1, 0 / 0.39) = 0. A
    # single-sample track has no segment and adds 0 too.
    g = grid_of(SPEC, *path([(1.1, 1.1)] * 50))
    single = grid_of(SPEC, *path([(2.1, 2.1)], track=2))

    assert g.sum() == 0.0
    assert single.sum() == 0.0


def test_a_track_counts_only_its_own_segments():
    # Two tracks, interleaved in frames: never joined to each other.
    a = path([(0.1, 0.25), (0.4, 0.25)], track=1)
    b = path([(3.1, 3.25), (3.4, 3.25)], track=2)
    ids, frames, xs, ys = (np.r_[u, v] for u, v in zip(a, b, strict=True))

    g = grid_of(SPEC, ids, frames, xs, ys)

    assert g[0, 0] == pytest.approx(0.3 / CHORD)
    assert g[6, 6] == pytest.approx(0.3 / CHORD)
    assert g.sum() == pytest.approx(0.6 / CHORD)


def test_samples_are_joined_in_frame_order():
    ids = np.array([1, 1, 1])
    frames = np.array([2, 0, 1])
    xs, ys = np.array([0.4, 0.1, 0.25]), np.full(3, 0.25)

    g = grid_of(SPEC, ids, frames, xs, ys)

    assert g[0, 0] == pytest.approx(0.3 / CHORD)


def test_a_path_cut_at_a_cell_boundary_sums_exactly_as_the_unbroken_path():
    xs = np.linspace(0.25, 3.75, 15)  # 0.25 m steps along row 0, through cell edges
    whole = grid_of(SPEC, *path(np.column_stack([xs, np.full(15, 0.25)])))
    # Cut into two tracklets at x = 2.0 (a cell edge), the sample there in both.
    first = path(np.column_stack([xs[:8], np.full(8, 0.25)]), track=1)
    second = path(np.column_stack([xs[7:], np.full(8, 0.25)]), track=2, start=7)
    parts = grid_of(SPEC, *(np.r_[u, v] for u, v in zip(first, second, strict=True)))

    np.testing.assert_allclose(parts, whole)


def test_a_fragmented_path_sums_about_as_the_unbroken_path():
    # A 3.5 m walk cut into 4 tracklets mid-cell, with one frame lost at each
    # cut. Distinct ids would count the cut cells twice; chord length adds at
    # most a fraction there.
    xs = np.linspace(0.25, 3.75, 71)  # 5 cm steps
    ys = np.full(71, 1.25)
    whole = grid_of(SPEC, *path(np.column_stack([xs, ys])))
    cuts = [(0, 20), (21, 40), (41, 60), (61, 71)]  # frames 20, 40, 60 lost, mid-cell
    ids, frames = [], []
    for track, (a, b) in enumerate(cuts):
        ids += [track] * (b - a)
        frames += list(range(a, b))
    ids, frames = np.array(ids), np.array(frames)
    parts = grid_of(SPEC, ids, frames, xs[frames], ys[frames])

    assert whole.sum() == pytest.approx(3.5 / 0.5 - 1 + 2 * 0.25 / CHORD)
    assert abs(parts.sum() - whole.sum()) <= 0.1 * whole.sum()
    distinct = sum(len({int(x // 0.5) for x in xs[a:b]}) for a, b in cuts)
    assert distinct - (whole > 0).sum() == 3  # distinct ids overcount each cut cell


def test_path_outside_the_grid_is_clipped_and_far_points_cost_nothing():
    # From x = -1e9 to the centre of row 0: only the part inside counts.
    g = grid_of(SPEC, *path([(-1e9, 0.25), (0.25, 0.25)]))

    assert g[0, 0] == pytest.approx(0.25 / CHORD)
    assert g.sum() == pytest.approx(0.25 / CHORD)


def test_non_finite_points_are_skipped():
    g = grid_of(SPEC, *path([(0.1, 0.25), (np.nan, 0.25), (0.4, 0.25)]))

    assert g.sum() == 0.0


def test_each_track_cell_pair_carries_the_frame_it_entered_the_cell():
    cells, frames, _ = chord_passage(SPEC, *path([(0.25, 0.25), (0.75, 0.25), (1.25, 0.25)]))

    entered = dict(zip(cells.tolist(), frames.tolist(), strict=True))
    assert entered == {0: 0, 1: 0, 2: 1}
