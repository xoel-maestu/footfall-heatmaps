import numpy as np
import pytest

from footfall_heatmaps.evaluation.domain.bootstrap import (
    DOMAINS,
    BlockLayers,
    SceneBlocks,
    block_bootstrap,
    block_layers,
    estimate,
)
from footfall_heatmaps.occupancy.domain.grid import Grid, GridSpec

SPEC = GridSpec(0.0, 0.0, 0.5, nx=8, ny=8)  # 4 m x 4 m
MASK = np.ones((8, 8), dtype=bool)
FPS = 10.0
BLOCK = 300  # frames in 30 s at 10 fps


def walkers(seed, n_blocks=6, people=4):
    """Random walks on the grid: (track_ids, frames, xs, ys), one sample per frame."""
    rng = np.random.default_rng(seed)
    ids, frames, xs, ys = [], [], [], []
    for track in range(people):
        start = rng.integers(0, n_blocks * BLOCK - 400)
        n = rng.integers(100, 400)
        x = np.clip(2 + np.cumsum(rng.normal(0, 0.05, n)), 0, 3.99)
        y = np.clip(2 + np.cumsum(rng.normal(0, 0.05, n)), 0, 3.99)
        ids += [track] * n
        frames += list(range(start, start + n))
        xs += list(x)
        ys += list(y)
    return np.array(ids), np.array(frames), np.array(xs), np.array(ys)


def test_blocks_add_up_to_the_whole_run():
    ids, frames, xs, ys = walkers(0)

    blocks = block_layers(SPEC, ids, frames, xs, ys, FPS, BLOCK, n_blocks=6)
    whole = Grid.from_samples(SPEC, ids, xs, ys, FPS)

    assert blocks.dwell.shape == (6, 8, 8)
    np.testing.assert_allclose(blocks.dwell.sum(axis=0), whole.dwell)
    # Passage: a track that stays in a cell across blocks counts in its first one only.
    np.testing.assert_array_equal(blocks.passage.sum(axis=0), whole.passage)


def test_passage_counts_a_track_in_the_block_where_it_entered_the_cell():
    blocks = block_layers(
        SPEC, np.array([7, 7, 7]), np.array([10, 299, 300]), np.full(3, 1.1),
        np.full(3, 1.1), FPS, BLOCK, n_blocks=2,
    )  # fmt: skip

    assert blocks.passage[:, 2, 2].tolist() == [1, 0]
    assert blocks.dwell[:, 2, 2].tolist() == pytest.approx([0.2, 0.1])


def test_samples_past_the_last_block_are_an_error():
    with pytest.raises(ValueError, match="block"):
        block_layers(SPEC, np.array([1]), np.array([600]), [1.0], [1.0], FPS, BLOCK, n_blocks=2)


def scene(truth_seed, *preds):
    def layers(seed):
        return block_layers(SPEC, *walkers(seed), FPS, BLOCK, n_blocks=6)

    truth = layers(truth_seed)
    return SceneBlocks(
        truth=truth, preds=tuple(p(truth, layers) for p in preds), mask=MASK, cell_m=SPEC.cell_m
    )


def same(truth, layers):
    return truth


def doubled(truth, layers):
    return BlockLayers(truth.dwell * 2, truth.passage * 2)


def other_walk(truth, layers):
    return layers(99)


def test_a_perfect_prediction_has_zero_error_with_a_zero_width_interval():
    result = block_bootstrap([scene(0, same)], n_resamples=200, seed=1)

    for layer in ("dwell", "passage"):
        for metric in ("mass_error", "tvd"):
            e = result.scenes[0][0][layer][metric]
            assert (e.value, e.lo, e.hi) == pytest.approx((0, 0, 0), abs=1e-12)
    assert result.scenes[0][0]["dwell"]["pearson"].value == pytest.approx(1)


def test_double_the_mass_is_a_mass_error_of_one_in_every_resample():
    result = block_bootstrap([scene(0, doubled)], n_resamples=200, seed=1)

    e = result.scenes[0][0]["dwell"]["mass_error"]
    assert (e.value, e.lo, e.hi) == pytest.approx((1, 1, 1))
    assert result.scenes[0][0]["dwell"]["tvd"].hi == pytest.approx(0, abs=1e-12)


def test_the_interval_contains_the_estimate_and_has_width_under_noise():
    result = block_bootstrap([scene(0, other_walk)], n_resamples=300, seed=1)

    for metric in ("mass_error", "tvd", "wape_zone", "emd_m"):
        e = result.scenes[0][0]["dwell"][metric]
        assert e.lo <= e.value <= e.hi and e.lo < e.hi, metric
    # WAPE is a dwell metric: undefined for passage.
    assert result.scenes[0][0]["passage"]["wape_zone"].value is None


def test_pooled_is_the_mean_over_scenes():
    result = block_bootstrap([scene(0, same), scene(1, doubled)], n_resamples=100, seed=1)

    e = result.pooled[0]["dwell"]["mass_error"]
    assert (e.value, e.lo, e.hi) == pytest.approx((0.5, 0.5, 0.5))


def test_the_difference_is_paired_on_the_same_resampled_blocks():
    # Two identical noisy predictions: each has a wide interval, their paired
    # difference is exactly zero in every resample.
    result = block_bootstrap([scene(0, other_walk, other_walk)], n_resamples=200, seed=1)

    assert result.scenes[0][0]["dwell"]["tvd"].hi > result.scenes[0][0]["dwell"]["tvd"].lo
    d = result.delta_pooled["dwell"]["tvd"]
    assert (d.value, d.lo, d.hi) == pytest.approx((0, 0, 0), abs=1e-12)


def test_the_difference_is_first_minus_second():
    result = block_bootstrap([scene(0, doubled, same)], n_resamples=100, seed=1)

    d = result.delta_scenes[0]["dwell"]["mass_error"]
    assert (d.value, d.lo, d.hi) == pytest.approx((1, 1, 1))
    assert block_bootstrap([scene(0, same)], n_resamples=10).delta_pooled is None


def test_the_same_seed_gives_the_same_intervals():
    a = block_bootstrap([scene(0, other_walk)], n_resamples=100, seed=3)
    b = block_bootstrap([scene(0, other_walk)], n_resamples=100, seed=3)

    assert a.scenes[0][0]["dwell"]["tvd"] == b.scenes[0][0]["dwell"]["tvd"]


def shifted(truth, layers):
    # The truth with its blocks shifted by one: identical over the whole run
    # (TVD 0), different in any resample that repeats blocks.
    return BlockLayers(np.roll(truth.dwell, 1, axis=0), np.roll(truth.passage, 1, axis=0))


def test_recentred_intervals_are_clipped_to_the_metric_domain():
    # Every resampled TVD is above 0: recentred on the whole-run value 0, the
    # lower bound would be negative, an impossible TVD. It is clipped to 0.
    e = block_bootstrap([scene(0, shifted)], n_resamples=300, seed=1).scenes[0][0]["dwell"]["tvd"]

    assert e.value == pytest.approx(0, abs=1e-12)
    assert e.lo == 0 < e.hi


def test_paired_differences_are_not_clipped():
    r = block_bootstrap([scene(0, shifted, same)], n_resamples=300, seed=1)

    d = r.delta_scenes[0]["dwell"]["tvd"]
    assert d.value == pytest.approx(0, abs=1e-12)
    assert d.lo < 0 < d.hi


@pytest.mark.parametrize(
    ("metric", "draws", "bounds"),
    [
        ("tvd", [0.98, 0.9, 1.0, 1.0, 1.0], (None, 1.0)),
        ("pearson", [0.99, 0.9, 1.0, 1.0, 1.0], (None, 1.0)),
        ("pearson", [-0.99, -0.9, -1.0, -1.0, -1.0], (-1.0, None)),
        ("mass_error", [0.01, 0.1, 0.0, 0.0, 0.0], (0.0, None)),
        ("wape_zone", [0.01, 0.1, 0.0, 0.0, 0.0], (0.0, None)),
        ("emd_m", [0.01, 0.1, 0.0, 0.0, 0.0], (0.0, None)),
    ],
)
def test_each_metric_is_clipped_to_its_own_domain(metric, draws, bounds):
    # draws[0] is the whole-run value; the recentred percentiles fall outside.
    e = estimate(np.array(draws), 2.5, DOMAINS[metric])
    unclipped = estimate(np.array(draws), 2.5)

    lo, hi = bounds
    assert (unclipped.lo < lo) if lo is not None else (unclipped.hi > hi)
    assert (e.lo == lo) if lo is not None else (e.hi == hi)
    assert e.value == draws[0]


def test_scenes_can_be_scored_in_parallel_with_the_same_result():
    from concurrent.futures import ThreadPoolExecutor

    scenes = [scene(0, other_walk), scene(1, doubled)]
    serial = block_bootstrap(scenes, n_resamples=50, seed=3)
    with ThreadPoolExecutor(2) as pool:
        parallel = block_bootstrap(scenes, n_resamples=50, seed=3, map=pool.map)

    assert parallel == serial
