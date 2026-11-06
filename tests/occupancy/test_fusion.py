import math

import numpy as np
import pytest

from space_usage_heatmaps.occupancy.domain.fusion import (
    CameraGrid,
    Fusion,
    fuse_grids,
    fuse_layers,
    fusion_weights,
    joint_coverage,
    overlap,
)
from space_usage_heatmaps.occupancy.domain.grid import Grid, GridSpec

SPEC = GridSpec(origin_x=0.0, origin_y=0.0, cell_m=0.5, nx=4, ny=4)


def camera(name, dwell, coverage, passage=None):
    dwell = np.asarray(dwell, float)
    passage = dwell / 10 if passage is None else np.asarray(passage, float)
    return CameraGrid(name, Grid(SPEC, dwell, passage, 0), np.asarray(coverage, bool))


def full(value):
    return np.full((4, 4), float(value))


def test_two_cameras_that_see_the_same_people_count_them_once():
    a, b = camera("a", full(10), full(1)), camera("b", full(10), full(1))

    for weighting in ("mean", "visibility"):
        fused = fuse_grids([a, b], Fusion(weighting))
        np.testing.assert_allclose(fused.dwell, full(10))
        np.testing.assert_allclose(fused.passage, full(1))
    np.testing.assert_allclose(fuse_grids([a, b], Fusion("sum")).dwell, full(20))


def test_an_occluded_camera_dilutes_the_mean_but_not_the_visibility_weighting():
    seen, hidden = camera("a", full(10), full(1)), camera("b", full(0), full(1))

    np.testing.assert_allclose(fuse_grids([seen, hidden], Fusion("mean")).dwell, full(5))
    np.testing.assert_allclose(fuse_grids([seen, hidden], Fusion("visibility", 1)).dwell, full(10))
    np.testing.assert_allclose(
        fuse_grids([seen, hidden], Fusion("visibility", math.inf)).dwell, full(10)
    )


def test_visibility_power_moves_from_the_mean_to_the_larger_camera():
    a, b = camera("a", full(8), full(1)), camera("b", full(2), full(1))

    def fused(p):
        return fuse_grids([a, b], Fusion("visibility", p)).dwell[0, 0]

    assert fused(0) == pytest.approx(5)  # the mean
    assert fused(1) == pytest.approx((8 * 8 + 2 * 2) / 10)  # weights 8/8 and 2/8
    assert fused(math.inf) == pytest.approx(8)
    assert fused(0) < fused(1) < fused(2) < fused(math.inf)


def test_a_cell_one_camera_covers_takes_its_value_and_uncovered_cells_are_empty():
    cov_a = np.zeros((4, 4), bool)
    cov_a[:, :2] = True
    cov_b = np.zeros((4, 4), bool)
    cov_b[:, 1:3] = True
    a, b = camera("a", full(4), cov_a), camera("b", full(6), cov_b)

    for weighting in ("mean", "visibility"):
        fused = fuse_grids([a, b], Fusion(weighting)).dwell
        np.testing.assert_allclose(fused[:, 0], 4)  # a only
        np.testing.assert_allclose(fused[:, 2], 6)  # b only
        np.testing.assert_allclose(fused[:, 3], 0)  # nobody
    np.testing.assert_array_equal(joint_coverage([a, b]), cov_a | cov_b)
    np.testing.assert_array_equal(overlap([a, b]), cov_a & cov_b)


def test_weights_sum_to_one_on_the_joint_coverage_and_zero_outside():
    cov = np.ones((4, 4), bool)
    cov[3] = False
    a = camera("a", np.arange(16).reshape(4, 4), cov)
    b = camera("b", np.zeros((4, 4)), cov)

    for f in (Fusion("mean"), Fusion("visibility", 1), Fusion("visibility", math.inf)):
        w = fusion_weights([a, b], f)
        np.testing.assert_allclose(w.sum(axis=0)[:3], 1)
        np.testing.assert_allclose(w[:, 3], 0)


def test_whole_run_weights_fuse_time_blocks_additively():
    rng = np.random.default_rng(0)
    blocks = [rng.random((5, 4, 4)) for _ in range(3)]  # 3 cameras, 5 blocks each
    grids = [camera(str(i), b.sum(axis=0), full(1)) for i, b in enumerate(blocks)]
    w = fusion_weights(grids, Fusion("visibility", 2))

    per_block = fuse_layers(w, blocks)

    assert per_block.shape == (5, 4, 4)
    np.testing.assert_allclose(
        per_block.sum(axis=0), fuse_grids(grids, Fusion("visibility", 2)).dwell
    )


def test_visibility_on_zones_gives_one_weight_per_zone():
    a = np.zeros((4, 4))
    a[0, 0] = 10  # all of a's mass in one cell of the first 2 m zone (4 x 4 cells)
    b = full(1)
    cams = [camera("a", a, full(1)), camera("b", b, full(1))]
    w = fusion_weights(cams, Fusion("visibility", 1, 2.0))

    assert np.ptp(w[0]) == 0 and np.ptp(w[1]) == 0  # one zone: constant weights
    assert w[0, 0, 0] == pytest.approx(10 / 26)  # zone mass 10 vs 16: ratios 10/16, 1


def test_cameras_on_another_grid_or_unknown_weighting_are_errors():
    other = GridSpec(origin_x=0.0, origin_y=0.0, cell_m=0.5, nx=2, ny=2)
    a = camera("a", full(1), full(1))
    b = CameraGrid("b", Grid(other, np.zeros((2, 2)), np.zeros((2, 2)), 0), np.ones((2, 2), bool))

    with pytest.raises(ValueError, match="grid"):
        fuse_grids([a, b])
    with pytest.raises(ValueError, match="weighting"):
        Fusion("median")
    with pytest.raises(ValueError, match="no camera"):
        fuse_grids([])
