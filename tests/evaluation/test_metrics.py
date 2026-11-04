import math

import numpy as np
import pytest

from space_usage_heatmaps.evaluation.domain.metrics import compare_layer

# 2 rows x 4 columns of 1 m cells: two 2 m zones, left and right.
TRUTH = np.array([[2.0, 0, 0, 0], [0, 2.0, 0, 0]])
PRED = np.array([[1.0, 0, 1.0, 0], [0, 1.0, 0, 1.0]])
ALL = np.ones_like(TRUTH, dtype=bool)


def test_identical_maps_score_zero_error_and_full_correlation():
    m = compare_layer(TRUTH, TRUTH, ALL, cell_m=1.0)

    assert (m.mass_error, m.tvd, m.wape_zone, m.emd_m) == (0, 0, 0, 0)
    assert m.pearson == pytest.approx(1.0)


def test_hand_computed_metrics():
    m = compare_layer(PRED, TRUTH, ALL, cell_m=1.0)

    assert m.mass_error == 0  # 4 vs 4
    # p = .5 at two cells; p_hat = .25 at four cells: 0.5 * (.25 + .25 + .25 + .25)
    assert m.tvd == pytest.approx(0.5)
    # right zone has no truth (< 1 %) and is ignored; left: |2 - 4| / 4
    assert m.wape_zone == pytest.approx(0.5)
    # .25 of mass moves 2 m from each truth cell: (0.5,0.5)->(2.5,0.5), (1.5,1.5)->(3.5,1.5)
    assert m.emd_m == pytest.approx(1.0)
    assert m.pearson == pytest.approx(2 / math.sqrt(12))


def test_mass_error_is_relative_to_truth():
    m = compare_layer(TRUTH * 1.5, TRUTH, ALL, cell_m=1.0)

    assert m.mass_error == pytest.approx(0.5)
    assert m.tvd == pytest.approx(0)


def test_cells_outside_the_coverage_mask_are_ignored():
    mask = ALL.copy()
    mask[:, 2:] = False

    m = compare_layer(PRED + np.array([[0, 0, 9, 9], [0, 0, 9, 9]]), PRED, mask, cell_m=1.0)

    assert (m.mass_error, m.tvd, m.wape_zone, m.emd_m) == (0, 0, 0, 0)


def test_emd_is_measured_on_a_one_metre_grid():
    # 0.5 m cells: both masses fall into the same 1 m block, so no transport.
    truth = np.array([[1.0, 0], [0, 0]])
    pred = np.array([[0, 0], [0, 1.0]])

    m = compare_layer(pred, truth, np.ones((2, 2), bool), cell_m=0.5)

    assert m.tvd == pytest.approx(1.0)
    assert m.emd_m == 0


def test_zones_are_two_metre_blocks_at_any_cell_size():
    # 0.5 m cells, 4 x 4 = one 2 m zone: mass moved inside the zone is no zone error.
    truth = np.zeros((4, 4))
    truth[0, 0] = 4
    pred = np.zeros((4, 4))
    pred[3, 3] = 4

    m = compare_layer(pred, truth, np.ones((4, 4), bool), cell_m=0.5)

    assert m.wape_zone == 0
    assert m.tvd == pytest.approx(1.0)


def test_empty_prediction_against_truth_is_total_error():
    m = compare_layer(np.zeros_like(TRUTH), TRUTH, ALL, cell_m=1.0)

    assert (m.mass_error, m.tvd, m.wape_zone) == (1, 1, 1)
    assert math.isnan(m.emd_m)
    assert math.isnan(m.pearson)


def test_empty_truth_is_an_error_not_a_score():
    with pytest.raises(ValueError, match="truth is empty"):
        compare_layer(TRUTH, np.zeros_like(TRUTH), ALL, cell_m=1.0)
    with pytest.raises(ValueError, match="truth is empty"):
        compare_layer(np.zeros_like(TRUTH), np.zeros_like(TRUTH), ALL, cell_m=1.0)


def test_empty_mask_is_an_error_not_a_score():
    with pytest.raises(ValueError, match="mask is empty"):
        compare_layer(TRUTH, TRUTH, np.zeros_like(ALL), cell_m=1.0)


def test_zones_with_truth_below_one_percent_are_ignored():
    # Left zone holds 99.5 % of the truth, right zone 0.5 %: the right zone's
    # error (|50 - 0.5|) is not counted, only the left one's |99.5 - 99.5|.
    truth = np.array([[99.5, 0, 0.5, 0], [0, 0, 0, 0]])
    pred = np.array([[99.5, 0, 50.0, 0], [0, 0, 0, 0]])

    m = compare_layer(pred, truth, ALL, cell_m=1.0)

    assert m.wape_zone == 0
    assert m.wape_zone_reason is None


def test_wape_is_none_with_a_reason_when_no_zone_reaches_one_percent(recwarn):
    # 0.5 m cells, 12 x 12 zones of 2 m with uniform truth: each holds 1/144 < 1 %.
    truth = np.ones((48, 48))

    m = compare_layer(truth, truth, np.ones_like(truth, bool), cell_m=0.5)

    assert m.wape_zone is None
    assert "1 %" in m.wape_zone_reason
    assert m.tvd == 0
    assert not recwarn.list


def test_wape_is_computed_for_dwell_only():
    m = compare_layer(PRED, TRUTH, ALL, cell_m=1.0, zones=False)

    assert m.wape_zone is None
    assert "dwell" in m.wape_zone_reason
    assert m.tvd == pytest.approx(0.5)
