"""space-usage calibrate/run/serve --site: every camera of a site in one process,
on video files with a ground-truth detector in place of YOLOX (as test_cli_run)."""

import functools
import json
import sqlite3

import numpy as np
import pytest

from space_usage_heatmaps import cli, site_recording
from space_usage_heatmaps.perception.infrastructure import container
from tests.conftest import write_scene
from tests.perception.oracle_detector import GroundTruthDetector
from tests.perception.synthetic_video import write_video
from tests.test_cli_serve import get_json, parse, serving

RECIPE = """detector: yolox-s
tracker: bytetrack
input_size: 640
engine: onnx-cuda
stride: 1
foot_point: bottom
"""

SITE = """site: shop
recipe: {recipe}
stride: 1
extent_m: [-4.0, 0.0, 4.0, 8.0]
cell_m: 0.5
retention_days: 90
start_utc: 2026-11-06T09:00:00Z
fusion:
  weighting: visibility
  power: 2
  zone_m: 2.0
  weights: sites/shop/weights.npz
calibration:
  minutes: 15
cameras:
  - id: "7"
    source: nvidia/MTMC_Tracking_2024/test/scene_001/camera_0007/video.mp4
    calibration: nvidia-mtmc/001/7
  - id: "8"
    source: nvidia/MTMC_Tracking_2024/test/scene_001/camera_0008/video.mp4
    calibration: nvidia-mtmc/001/8
    masks:
      image: []
      floor: []
"""


@pytest.fixture
def site(tmp_path, rows, monkeypatch):
    data = write_scene(tmp_path / "data", rows, cameras=(7, 8))
    scene = data / "nvidia/MTMC_Tracking_2024/test/scene_001"
    write_video(scene / "camera_0007/video.mp4")
    write_video(scene / "camera_0008/video.mp4", seed=1)
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(data))
    monkeypatch.setattr(container, "ensure_weights", lambda d, name: d / name)
    monkeypatch.setattr(container, "load_detector", lambda *_: GroundTruthDetector(rows))
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(RECIPE)
    path = tmp_path / "site.yaml"
    path.write_text(SITE.format(recipe=recipe))
    return path, data


def calibrate(path):
    return cli.main(["calibrate", "--site", str(path)])


def test_calibrate_writes_the_frozen_weights_of_the_site(site, capsys):
    path, data = site

    assert calibrate(path) == 0

    with np.load(data / "sites/shop/weights.npz") as f:
        assert list(f["cameras"]) == ["7", "8"]
        assert f["weights"].shape == (2, 20, 20)
        w = f["weights"]
        assert np.allclose(w.sum(0)[w.sum(0) > 0], 1.0)
        assert json.loads(str(f["fusion"])) == {
            "weighting": "visibility", "power": 2, "zone_m": 2.0, "covered_cells": False,
        }  # fmt: skip
    assert "weights" in capsys.readouterr().out


def test_calibrate_never_overwrites_frozen_weights(site, capsys):
    path, _ = site
    assert calibrate(path) == 0

    assert calibrate(path) == 1
    assert "already" in capsys.readouterr().err


def test_run_needs_the_calibrated_weights(site, tmp_path, capsys):
    path, _ = site

    assert cli.main(["run", "--site", str(path), "--db", str(tmp_path / "x.db")]) == 1
    assert "space-usage calibrate --site" in capsys.readouterr().err


def test_run_records_the_fused_slots_of_every_camera(site, tmp_path, monkeypatch, capsys):
    path, _ = site
    assert calibrate(path) == 0
    monkeypatch.setattr(
        site_recording, "RecordSite", functools.partial(site_recording.RecordSite, k=1)
    )
    db = tmp_path / "x.db"
    stats = tmp_path / "stats.json"

    assert cli.main(["run", "--site", str(path), "--db", str(db), "--stats", str(stats)]) == 0

    with sqlite3.connect(db) as c:
        tables = c.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        rows = c.execute("SELECT camera_id, start_utc, layer FROM slots ORDER BY 3").fetchall()
    assert tables == [("slots",)]
    assert rows == [
        ("shop", "2026-11-06T09:00:00Z", "dwell"),
        ("shop", "2026-11-06T09:00:00Z", "passage"),
    ]
    s = json.loads(stats.read_text())
    assert [c["camera"] for c in s["cameras"]] == ["7", "8"]
    assert all(c["processed_frames"] == 12 for c in s["cameras"])
    assert "2 cameras" in capsys.readouterr().out


def test_the_weights_must_match_the_site_cameras(site, tmp_path, capsys):
    path, _ = site
    assert calibrate(path) == 0
    path.write_text(path.read_text().split('  - id: "8"')[0])

    assert cli.main(["run", "--site", str(path), "--db", str(tmp_path / "x.db")]) == 1
    assert "calibrated for cameras" in capsys.readouterr().err


def test_a_recipe_with_stitching_is_refused(site, tmp_path, capsys):
    path, _ = site
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(RECIPE + "stitch:\n  gap_s: 10.0\n  radius_m: 1.0\n")

    assert calibrate(path) == 1
    assert "stitch" in capsys.readouterr().err


def test_run_takes_either_a_site_or_a_config_and_source(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["run", "--db", str(tmp_path / "x.db")])
    assert "--site" in capsys.readouterr().err


def test_serve_site_records_in_process_and_serves_fused_live_and_slots(site, tmp_path, monkeypatch):
    path, _ = site
    path.write_text(path.read_text().replace("start_utc: 2026-11-06T09:00:00Z\n", ""))
    assert calibrate(path) == 0  # the live layer decays to now: the video starts now
    monkeypatch.setattr(
        site_recording, "RecordSite", functools.partial(site_recording.RecordSite, k=1)
    )
    args = parse(
        ["serve", "--site", str(path), "--db", str(tmp_path / "x.db"), "--record", "--port", "0"]
    )

    server, recording, stop = cli.start_serve(args)
    serving(server)
    try:
        recording.join(timeout=30)
        assert not recording.is_alive()
        live = get_json(server, "/api/live?camera=shop")
        slots = get_json(server, "/api/slots?camera=shop")
        info = get_json(server, "/api/site")
    finally:
        stop.set()
        server.shutdown()
        server.server_close()

    assert live["shape"] == [20, 20] and np.asarray(live["dwell"]).sum() > 0
    assert live["ended"] is not None
    assert len(slots["slots"]) in (1, 2)  # 2 across a quarter
    assert all(np.asarray(s["dwell"]).sum() == 0 for s in slots["slots"])  # read with k=5
    assert info["site"] == "shop" and [c["id"] for c in info["cameras"]] == ["7", "8"]
    assert info["fusion"]["power"] == 2
    assert all(c["processed_frames"] == 12 for c in info["cameras"])


def test_serve_site_without_record_reads_slots_only(site, tmp_path):
    path, _ = site
    assert calibrate(path) == 0
    args = parse(["serve", "--site", str(path), "--db", str(tmp_path / "x.db"), "--port", "0"])

    server, recording, stop = cli.start_serve(args)
    serving(server)
    try:
        slots = get_json(server, "/api/slots?camera=shop")
        info = get_json(server, "/api/site")
    finally:
        server.shutdown()
        server.server_close()

    assert recording is None and slots["slots"] == []
    assert [c["state"] for c in info["cameras"]] == ["not recording", "not recording"]


def test_the_meter_reports_per_second_rates_after_the_warm_up():
    now = [0.0]
    meter = site_recording.CameraMeter(["a", "b"], clock=lambda: now[0])
    for second in range(20):
        for k in range(4 if second != 12 else 1):  # camera a stalls in second 12
            now[0] = second + k / 4
            meter.tick("a")
            meter.tick("b")
    now[0] = 20.0

    r = meter.report()

    assert r["timed_seconds"] == 15  # seconds 5..19
    a, b = r["cameras"]
    assert a["processed_frames"] == 77 and a["processed_fps_p5"] == pytest.approx(
        3.1
    )  # linear 5th pct of 14 x 4 and one 1
    assert r["processed_fps_p5_min"] == a["processed_fps_p5"]
    assert a["processed_fps_p5_10s"] == pytest.approx(3.7)  # one 10 s window, 5..14: 37 frames
