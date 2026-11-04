import json

from space_usage_heatmaps.cli import main
from tests.conftest import write_scene

METRICS = {"mass_error", "tvd", "wape_zone", "emd_m", "pearson", "wape_zone_reason"}


def test_eval_prints_both_truths_and_writes_the_report(tmp_path, rows, monkeypatch, capsys):
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(write_scene(tmp_path / "data", rows)))
    monkeypatch.chdir(tmp_path)

    code = main(["eval", "--source", "gt-world", "--scene", "001", "--camera", "7"])

    assert code == 0
    out = capsys.readouterr().out
    blocks = ["visible@coverage", "full@coverage", "visible@visibility"]
    assert all(b in out for b in blocks)
    assert [out.index(b) for b in blocks] == sorted(out.index(b) for b in blocks)
    assert "occlusion fraction 0.0625" in out
    passage = next(ln for ln in out.splitlines() if ln.startswith("passage"))
    assert "dwell only" in passage and "n/a" not in passage
    report = json.loads((tmp_path / "reports" / "oracle-gt-world-s001-c7-0.5m.json").read_text())
    assert report["cell_m"] == 0.5
    assert {report["scene"], report["camera"], report["source"]} == {"001", "7", "gt-world"}
    assert report["occlusion_fraction"] == 0.0625
    assert list(report["results"]) == blocks
    visible = report["results"]["visible@visibility"]
    assert set(visible["dwell"]) == METRICS
    assert visible["passage"]["tvd"] == 0
    # WAPE is a dwell metric: null with its reason for passage.
    assert visible["passage"]["wape_zone"] is None
    assert "dwell" in visible["passage"]["wape_zone_reason"]


def test_all_cameras_prints_one_row_per_calibrated_camera(tmp_path, rows, monkeypatch, capsys):
    monkeypatch.setenv(
        "SPACE_USAGE_DATA_DIR", str(write_scene(tmp_path / "data", rows, cameras=(7, 8)))
    )
    monkeypatch.chdir(tmp_path)

    code = main(["eval", "--source", "gt-world", "--scene", "001", "--camera", "all"])

    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert [ln.split()[0] for ln in lines[-2:]] == ["7", "8"]
    header = next(ln.split() for ln in lines if ln.startswith("camera"))
    tvd = [h for h in header if h.endswith("d.tvd")]
    assert tvd == ["vcov.d.tvd", "fcov.d.tvd", "vvis.d.tvd"]
    assert (tmp_path / "reports" / "oracle-gt-world-s001-c8-0.5m.json").exists()


def test_a_camera_without_visible_people_is_an_error_not_a_score(
    tmp_path, rows, monkeypatch, capsys
):
    monkeypatch.setenv(
        "SPACE_USAGE_DATA_DIR", str(write_scene(tmp_path / "data", rows, cameras=(7, 5)))
    )
    monkeypatch.chdir(tmp_path)

    code = main(["eval", "--source", "gt-world", "--scene", "001", "--camera", "5"])

    assert code == 1
    assert "visibility mask is empty" in capsys.readouterr().err
    assert not (tmp_path / "reports").exists()


def test_all_cameras_reports_a_failing_camera_and_scores_the_rest(
    tmp_path, rows, monkeypatch, capsys
):
    root = write_scene(tmp_path / "data", rows, cameras=(7, 8))
    path = root / "nvidia" / "MTMC_Tracking_2024" / "test" / "scene_001"
    calib = json.loads((path / "calibration_2025_format.json").read_text())
    calib["sensors"][0]["attributes"] = [
        a for a in calib["sensors"][0]["attributes"] if a["name"] != "fps"
    ]  # camera 7 lacks its fps: a missing-data error, not a crash
    (path / "calibration_2025_format.json").write_text(json.dumps(calib))
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(root))
    monkeypatch.chdir(tmp_path)

    code = main(["eval", "--source", "gt-world", "--scene", "001", "--camera", "all"])

    assert code == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2].split()[:2] == ["7", "error:"] and "fps" in lines[-2]
    assert lines[-1].split()[0] == "8" and "error" not in lines[-1]
    assert (tmp_path / "reports" / "oracle-gt-world-s001-c8-0.5m.json").exists()
