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


class ScriptedStream:
    """A live source (OpenCvStreamSource's shape) driven by a script per URL:
    each connection takes the next entry, "down" (opening raises SourceDropped),
    or (frames, then) where then is "end" or "error" (a non-drop error)."""

    scripts = {}

    def __init__(self, url, stride):
        from space_usage_heatmaps.perception.domain.ports import SourceDropped

        step = self.scripts[url].pop(0)
        if step == "down":
            raise SourceDropped(f"cannot open stream {url}")
        self.n, self.then, self.fps = *step, 25.0

    def frames(self):
        import time

        from space_usage_heatmaps.perception.domain.frame import Frame

        for i in range(self.n):
            time.sleep(0.002)
            yield Frame(i, i / self.fps, np.zeros((4, 4, 3), np.uint8))
        if self.then == "error":
            raise RuntimeError("decoder error")


def test_a_stream_down_at_start_does_not_stop_the_site_and_connects_later():
    ScriptedStream.scripts = {"rtsp://x/c": ["down", "down", (3, "end")]}
    waits = []
    make, fps, up = site_recording._stream_maker("rtsp://x/c", 10, ScriptedStream, waits.append)

    assert not up and fps == site_recording.DEFAULT_STREAM_FPS
    drops = []
    got = [f.index for f in make(lambda: drops.append(1))]
    assert got == [0, 1, 2] and drops == [1] and waits == [1]  # one failed reconnect


def test_one_failing_camera_leaves_the_others_recording_and_its_slots_incomplete():
    from space_usage_heatmaps.perception.infrastructure.round_robin_feed import round_robin_feed
    from tests.occupancy.test_record_camera import CONFIG, Crowd, IdPerBox
    from tests.occupancy.test_record_cameras import Log, pipe

    ScriptedStream.scripts = {
        "rtsp://x/a": [(40, "end")],
        "rtsp://x/b": [(3, "error"), (3, "end")],  # fails while recording
        "rtsp://x/c": ["down", "down", (3, "end")],  # down at start
    }
    sources, ups = {}, {}
    for c in "abc":
        sources[c], _, ups[c] = site_recording._stream_maker(
            f"rtsp://x/{c}", 1, ScriptedStream, lambda s: None
        )
    stats, log = {}, Log()
    feed = round_robin_feed(sources, paced=False, stats=stats, sleep=lambda s: None)

    n = record_cameras_of(feed, {c: pipe(IdPerBox()) for c in "abc"}, CONFIG, log, Crowd())

    assert ups == {"a": True, "b": True, "c": False}
    assert sorted(set(log.dropped)) == ["b", "c"] and "a" not in log.dropped
    assert stats["b"]["errors"] == 1 and stats["c"]["drops"] == 1
    added = {c: sum(1 for a in log.added if a[0] == c) for c in "abc"}
    assert added["a"] >= 30 and added["b"] >= 4 and added["c"] >= 1
    assert n == sum(added.values()) and log.closed


def record_cameras_of(feed, pipes, config, log, detector):
    from space_usage_heatmaps.occupancy.application.record_cameras import record_cameras

    return record_cameras(feed, detector, pipes, config, log)


def test_the_meter_measures_stream_lag_and_lost_frames():
    now = [0.0]
    meter = site_recording.CameraMeter(["a", "b"], clock=lambda: now[0])
    meter.streams({"a": 30.0, "b": 30.0}, stride=10)
    meter.feed_stats = {"a": {"dropped": 0}, "b": {"dropped": 0}}
    ts = 100.0
    for k in range(3 * 60):  # 60 s at 3 processed fps, keeping pace: no lag
        now[0] = k / 3
        meter.tick("a", ts + k / 3)
    ts = 50.0
    for k in range(3 * 60):  # each frame comes 10 % late: OpenCV's buffer grows
        now[0] = 1.1 * k / 3
        meter.tick("b", ts)
        ts += 1 / 3
        if k == 90:
            ts += 1.0  # 30 source frames lost
        if k == 120:
            meter.feed_stats["b"]["dropped"] += 1  # one frame replaced unread: not lost
            ts += 1 / 3
    meter.restart("a")  # a reconnect: a new connection, its own stream time
    for k in range(3):
        now[0] = 60 + k / 3
        meter.tick("a", k / 3)

    a, b = meter.report()["cameras"]

    assert a["lost_frames"] == 0 and a["stream_restarts"] == 1
    assert a["lag_p95_s"] == pytest.approx(0, abs=1e-6)
    assert a["lag_slope_s_per_min"] == pytest.approx(0, abs=1e-6)
    assert b["lost_frames"] == 30 and b["stream_restarts"] == 0
    assert b["lag_slope_s_per_min"] > 3  # grows: the lag is not bounded
    r = meter.report()
    assert r["lag_slope_max_s_per_min"] == b["lag_slope_s_per_min"]


def test_a_site_counts_k_as_its_config_says(site, tmp_path, monkeypatch):
    from space_usage_heatmaps.occupancy.domain.slots import DEFAULT_K_COUNT
    from space_usage_heatmaps.occupancy.domain.stitching import K_STITCH

    path, _ = site
    assert calibrate(path) == 0
    seen = []
    real = site_recording.RecordSite

    def spy(*args, **kwargs):
        seen.append(kwargs.get("stitch"))
        return real(*args, **kwargs)

    monkeypatch.setattr(site_recording, "RecordSite", spy)
    text = path.read_text()
    assert cli.main(["run", "--site", str(path), "--db", str(tmp_path / "a.db")]) == 0
    for k_count in ("ids", "chains"):
        path.write_text(text + f"k_count: {k_count}\n")
        assert cli.main(["run", "--site", str(path), "--db", str(tmp_path / f"{k_count}.db")]) == 0

    default = K_STITCH if DEFAULT_K_COUNT == "chains" else None
    assert seen == [default, None, K_STITCH]


def test_a_site_with_an_unknown_k_count_is_refused(site, capsys):
    path, _ = site
    path.write_text(path.read_text() + "k_count: people\n")

    assert calibrate(path) == 1
    assert "k_count" in capsys.readouterr().err
