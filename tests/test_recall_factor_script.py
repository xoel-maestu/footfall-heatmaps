"""scripts/recall-factor.py (improvements plan, phase 7, H7), on hand-made tracks."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from footfall_heatmaps.evaluation.domain.tracks import Tracks
from footfall_heatmaps.occupancy.domain.grid import GridSpec

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recall-factor.py"
T0 = 1_793_955_600  # 2026-11-06T09:00:00Z
SPEC = GridSpec(origin_x=0.0, origin_y=0.0, cell_m=0.5, nx=4, ny=2)


def script():
    spec = importlib.util.spec_from_file_location("recall_factor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def boxes_at(xs):
    """Boxes whose x1 is the floor x, so the test's floor points are trivial."""
    xs = np.asarray(xs, float)
    return np.column_stack([xs, np.zeros_like(xs), xs + 1, np.ones_like(xs)])


def identity_floor(boxes, ids, frames):
    return boxes[:, 0], np.full(len(boxes), 0.1)


def test_replay_reads_the_slot_as_the_recorder_does_from_processed_frames_only():
    fps, stride = 10.0, 10
    # 6 people in cell 0 on every processed frame for 120 s; filled frames in between
    frames = np.repeat(np.arange(0, 1200), 6)
    ids = np.tile(np.arange(6), 1200)
    tracks = Tracks(ids, frames, boxes_at(np.full(len(ids), 0.1)))

    (row,) = script().replay(SPEC, tracks, identity_floor, 1200, stride, fps, T0)

    assert row["observed_s"] == pytest.approx(120.0)
    assert row["mean_occupancy"] == pytest.approx(6.0)  # filled rows would read 60
    assert row["slot_start_utc"] == T0


def test_replay_gates_a_minute_below_k_ids_and_keeps_its_observed_time():
    fps, stride = 10.0, 10
    # 6 people in minute 0, 2 people in minute 1
    f0 = np.arange(0, 600, 10)
    f1 = np.arange(600, 1200, 10)
    frames = np.concatenate([np.repeat(f0, 6), np.repeat(f1, 2)])
    ids = np.concatenate([np.tile(np.arange(6), len(f0)), np.tile(np.arange(2), len(f1))])
    tracks = Tracks(ids, frames, boxes_at(np.full(len(ids), 0.1)))

    (row,) = script().replay(SPEC, tracks, identity_floor, 1200, stride, fps, T0)

    assert row["observed_s"] == pytest.approx(120.0)
    assert row["mean_occupancy"] == pytest.approx(6 * 60 / 120)


def test_factor_is_the_sum_of_truth_over_the_sum_of_shown():
    m = script()
    cams = [{"shown": 2.0, "truth": 3.0}, {"shown": 6.0, "truth": 7.0}]

    assert m.fit_factor(cams) == pytest.approx(10 / 8)


def test_errors_median_and_p90_of_the_absolute_relative_error():
    m = script()
    cams = [{"shown": s, "truth": 10.0} for s in (9.0, 8.0, 12.0, 5.0)]

    e = m.error_summary(cams, 1.0)

    assert e["errors"] == pytest.approx([-0.1, -0.2, 0.2, -0.5])
    assert e["median_abs"] == pytest.approx(0.2)
    assert e["p90_abs"] == pytest.approx(np.percentile([0.1, 0.2, 0.2, 0.5], 90))
    assert m.error_summary(cams, 2.0)["errors"] == pytest.approx([0.8, 0.6, 1.4, 0.0])


def test_a_camera_without_truth_is_left_out_of_the_errors():
    m = script()
    cams = [{"shown": 1.0, "truth": 0.0}, {"shown": 9.0, "truth": 10.0}]

    e = m.error_summary(cams, 1.0)

    assert e["n"] == 1 and e["left_out"] == 1


def sealed_run(m, tmp_path, monkeypatch, measure):
    monkeypatch.setattr(m, "REPORTS", tmp_path)
    monkeypatch.setattr(m, "measure", measure)
    monkeypatch.setattr("sys.argv", ["recall-factor.py", "sealed"])
    (tmp_path / "recall-factor-tune.json").write_text('{"factor": 1.0}')
    m.main()


def test_sealed_scoring_leaves_a_started_marker_even_when_it_crashes(tmp_path, monkeypatch):
    m = script()

    def crash(split):
        assert (tmp_path / "recall-factor-sealed.started").exists()  # written before scoring
        raise RuntimeError("crash halfway")

    with pytest.raises(RuntimeError):
        sealed_run(m, tmp_path, monkeypatch, crash)
    assert (tmp_path / "recall-factor-sealed.started").exists()


def test_sealed_scoring_refuses_to_run_after_a_started_marker(tmp_path, monkeypatch):
    m = script()
    (tmp_path / "recall-factor-sealed.started").write_text("")
    called = []

    with pytest.raises(SystemExit, match="scored once"):
        sealed_run(m, tmp_path, monkeypatch, called.append)
    assert called == []
