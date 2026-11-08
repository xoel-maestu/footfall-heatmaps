import hashlib
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


def _left_half():
    cov = np.zeros((4, 4), bool)
    cov[:, :2] = True
    return cov


def test_covered_cells_does_not_penalise_a_camera_that_covers_part_of_a_zone():
    # One 2 m zone (4 x 4 cells). Both cameras see the same 1 person-second per
    # cell; a covers only the left half, b all of it. The zone sum halves a's
    # ratio (8 vs 16); the mean over covered cells makes them alike (1 vs 1).
    partial, whole = (
        camera("a", full(1) * _left_half(), _left_half()),
        camera("b", full(1), full(1)),
    )

    plain = fusion_weights([partial, whole], Fusion("visibility", 1, 2.0))
    normalised = fusion_weights([partial, whole], Fusion("visibility", 1, 2.0, covered_cells=True))

    assert plain[0, 0, 0] == pytest.approx(0.5 / 1.5)  # ratios 8/16 and 1
    np.testing.assert_allclose(normalised[0][_left_half()], 0.5)  # ratios 1 and 1
    np.testing.assert_allclose(normalised[1][~_left_half()], 1)  # b alone where a cannot see
    np.testing.assert_allclose(normalised.sum(axis=0), 1)


def test_covered_cells_still_follows_visibility_and_power():
    # a sees half as much per covered cell as b: ratio 1/2 whatever its coverage.
    a, b = camera("a", 0.5 * full(1) * _left_half(), _left_half()), camera("b", full(1), full(1))

    for p in (1, 2):
        w = fusion_weights([a, b], Fusion("visibility", p, 2.0, covered_cells=True))
        r = 0.5**p
        np.testing.assert_allclose(w[0][_left_half()], r / (1 + r))
    w = fusion_weights([a, b], Fusion("visibility", math.inf, 2.0, covered_cells=True))
    np.testing.assert_allclose(w[1], 1)


# sha256 of the weights before covered_cells existed (main b765c5e).
GOLDEN = {
    ("sum", 1.0, None): "72b669efc00fbf525034f111074c18ba00b08d1094e3b5ad5b01010fe0934112",
    ("mean", 1.0, None): "1c19c7732cd945143f29e87f468a9eaacfa8752e0b5df853020cefc4c67cc862",
    ("visibility", 1, None): "be43af3495d8ccbdce69c4b5dcfb594463f42145d337960041fd92f2f98fdf94",
    ("visibility", 2, 1.0): "097fc6a0534fd78ad2416ec6cc2abf7c803372a8d417766ed0e34e336a1b5a35",
    ("visibility", 1, 2.0): "d2ed2af48df0d6deec3f3ffb73d097c0954a6abf7f6caf450880ab64046c30dd",
    ("visibility", 2, 2.0): "0a6dedf384afc87334da92695412115bb3096091df9af3af73b3081d936636a3",
    (
        "visibility",
        math.inf,
        2.0,
    ): "3a7b0634ade69c53f314344d24dbe5017e2a824fb350c2933d49699b3c299828",
}


def test_existing_weightings_are_byte_identical_with_covered_cells_off():
    rng = np.random.default_rng(1)
    cams = [camera(str(i), rng.random((4, 4)), rng.random((4, 4)) > 0.3) for i in range(3)]

    for args, digest in GOLDEN.items():
        f = Fusion(*args)
        assert f.covered_cells is False
        assert hashlib.sha256(fusion_weights(cams, f).tobytes()).hexdigest() == digest, args


def test_covered_cells_needs_visibility_on_zones():
    with pytest.raises(ValueError, match="covered_cells"):
        Fusion("visibility", 1, None, covered_cells=True)
    with pytest.raises(ValueError, match="covered_cells"):
        Fusion("mean", 1, 2.0, covered_cells=True)
