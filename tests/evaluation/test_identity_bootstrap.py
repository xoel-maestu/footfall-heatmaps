import numpy as np
import pytest

from space_usage_heatmaps.evaluation.domain.bootstrap import IdentityBlocks, identity_bootstrap


def blocks(idtp, n_truth, n_pred):
    return IdentityBlocks(*(np.asarray(v, float) for v in (idtp, n_truth, n_pred)))


def test_the_value_is_the_whole_run_idf1_pooled_as_the_mean_over_scenes():
    a = blocks([5, 5, 5], [10, 10, 10], [10, 10, 10])  # .5
    b = blocks([8, 0], [10, 10], [10, 10])  # 16 / 40 = .4

    r = identity_bootstrap([(a,), (b,)], n_resamples=200, seed=0)

    assert r.scenes[0][0].value == pytest.approx(0.5)
    assert r.scenes[1][0].value == pytest.approx(0.4)
    assert r.pooled[0].value == pytest.approx(0.45)
    assert r.delta_pooled is None


def test_the_paired_difference_excludes_zero_when_one_wins_every_block():
    rng = np.random.default_rng(1)
    nt, n_pred = rng.integers(50, 100, 20), rng.integers(50, 100, 20)
    better = blocks(0.6 * np.minimum(nt, n_pred), nt, n_pred)
    worse = blocks(0.3 * np.minimum(nt, n_pred), nt, n_pred)

    r = identity_bootstrap([(better, worse)], n_resamples=300, seed=0)

    d = r.delta_pooled
    assert d.value > 0 and d.lo > 0
    assert r.delta_scenes[0].value == pytest.approx(d.value)


def test_identical_predictions_differ_by_exactly_zero():
    a = blocks([3, 4, 5], [10, 10, 10], [9, 9, 9])

    d = identity_bootstrap([(a, a)], n_resamples=100, seed=0).delta_pooled

    assert (d.value, d.lo, d.hi) == (0.0, 0.0, 0.0)
