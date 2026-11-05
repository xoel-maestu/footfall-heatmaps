import numpy as np
import pytest

from space_usage_heatmaps.evaluation.application.evaluate_oracle import (
    TRUTHS,
    build_ground_truth_grid,
    evaluate_oracle,
)
from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
from tests.conftest import gt_row, write_scene

FULL, VISIBLE, PRIMARY = "full@coverage", "visible@visibility", "visible@coverage"


def test_ground_truth_holds_every_person_and_the_visible_ones_on_the_scene_extent(tmp_path, rows):
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows))

    gt = build_ground_truth_grid(reader, "001", "7", 0.5)

    assert gt.full.dwell.sum() == pytest.approx(16 / 10)  # 15 + track 9, at 10 fps
    assert gt.visible.dwell.sum() == pytest.approx(15 / 10)
    assert gt.full.spec.nx * 0.5 >= 6.5 and gt.full.spec.ny * 0.5 >= 7.0
    assert 0 < gt.coverage.sum() < gt.coverage.size
    assert np.all(gt.full.dwell[~gt.coverage] == 0)
    # Visibility: covered cells where the camera sees someone at least once.
    np.testing.assert_array_equal(gt.visibility, gt.coverage & (gt.visible.dwell > 0))


def test_gt_world_reproduces_the_visible_truth_exactly_and_misses_the_hidden(tmp_path, rows):
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows))

    report = evaluate_oracle(reader, "gt-world", "001", "7", 0.5)

    for truth in (PRIMARY, VISIBLE):
        for layer in ("dwell", "passage"):
            m = report.results[truth][layer]
            assert (m.mass_error, m.tvd, m.emd_m) == (0, 0, 0)
        assert report.results[truth]["dwell"].wape_zone == 0
    assert report.results[VISIBLE]["passage"].wape_zone is None
    # Track 9 stands in camera 7's view for one frame but camera 7 never sees it.
    assert report.occlusion_fraction == pytest.approx(0.1 / 1.6)
    assert report.results[FULL]["dwell"].mass_error == pytest.approx(0.1 / 1.6)
    # Passages: track 1 crosses 4 cells, track 2 and track 9 one each.
    assert report.results[FULL]["passage"].mass_error == pytest.approx(1 / 6)


def test_gt_boxes_with_exact_feet_reproduce_the_visible_truth(tmp_path, rows):
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows))

    report = evaluate_oracle(reader, "gt-boxes", "001", "7", 0.25)

    assert report.results[VISIBLE]["dwell"].tvd == pytest.approx(0, abs=1e-9)
    assert report.results[VISIBLE]["passage"].mass_error == pytest.approx(0, abs=1e-9)


def test_gt_boxes_moves_mass_when_feet_are_hidden(tmp_path):
    # The visible box ends 20 px above the real foot: the point lands at y = 3.6
    # instead of 2.1, farther from the camera than the truth.
    rows = [gt_row(7, 1, f, 0.1, 2.1, foot_shift_px=-20) for f in range(10)]
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows))

    report = evaluate_oracle(reader, "gt-boxes", "001", "7", 0.25)

    assert report.results[FULL]["dwell"].tvd == pytest.approx(1.0)
    assert report.results[FULL]["dwell"].emd_m > 0


def test_a_camera_that_sees_no_floor_is_an_error(tmp_path, rows):
    # A negative homography left unfixed (no extrinsics, no ground truth to
    # orient it) puts the floor behind the camera: never a perfect score.
    root = write_scene(tmp_path, rows, cameras=(7, 5), flipped=(5,), extrinsics=False)

    with pytest.raises(ValueError, match="coverage mask is empty"):
        evaluate_oracle(NvidiaMtmcReader(root), "gt-world", "001", "5", 0.5)


def test_a_camera_that_sees_nobody_is_an_error(tmp_path, rows):
    root = write_scene(tmp_path, rows, cameras=(7, 5))

    with pytest.raises(ValueError, match="visibility mask is empty"):
        evaluate_oracle(NvidiaMtmcReader(root), "gt-world", "001", "5", 0.5)


def test_unknown_source_is_rejected(tmp_path, rows):
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows))

    with pytest.raises(ValueError, match="source"):
        evaluate_oracle(reader, "detector", "001", "7", 0.5)


def test_blocks_are_ordered_primary_first():
    assert TRUTHS == (PRIMARY, FULL, VISIBLE)


def test_mass_where_nobody_is_visible_is_penalised_on_coverage_not_on_visibility(tmp_path):
    # Track 1 stands still; in half the frames its visible box ends 20 px above
    # the foot, so gt-boxes puts that half in a covered cell with no visible
    # truth. The visibility mask, derived from the truth, hides that error.
    rows = [gt_row(7, 1, f, 0.1, 2.1, foot_shift_px=-20 if f >= 5 else 0) for f in range(10)]
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows))

    report = evaluate_oracle(reader, "gt-boxes", "001", "7", 0.25)

    assert report.results[VISIBLE]["dwell"].tvd == pytest.approx(0, abs=1e-9)
    assert report.results[PRIMARY]["dwell"].tvd == pytest.approx(0.5)


def test_gt_boxes_reports_the_ground_position_error_against_the_json_truth(tmp_path, rows):
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows))

    exact = evaluate_oracle(reader, "gt-boxes", "001", "7", 0.5)
    centre = evaluate_oracle(reader, "gt-boxes", "001", "7", 0.5, foot_point="centroid")

    assert exact.position_error.matched == 15
    assert (exact.position_error.median, exact.position_error.p90) == pytest.approx(
        (0, 0), abs=1e-6
    )
    assert centre.foot_point == "centroid" and centre.position_error.median > 0.1
    assert exact.head_samples == 0


def test_hidden_feet_show_as_position_error(tmp_path):
    rows = [gt_row(7, 1, f, 0.1, 2.1, foot_shift_px=-20) for f in range(10)]
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows))

    report = evaluate_oracle(reader, "gt-boxes", "001", "7", 0.25)

    assert report.position_error.median > 1.0


def test_gt_world_position_error_compares_the_two_truth_files(tmp_path, rows):
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows))

    report = evaluate_oracle(reader, "gt-world", "001", "7", 0.5)

    assert report.position_error.matched == 15 and report.position_error.p90 == pytest.approx(
        0, abs=1e-9
    )
