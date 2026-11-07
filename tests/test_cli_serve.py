"""space-usage serve: the viewer's server with the recorder in the same process, on a
video file with a ground-truth detector in place of YOLOX (as tests/test_cli_run.py)."""

import argparse
import functools
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from space_usage_heatmaps import cli
from space_usage_heatmaps.perception.infrastructure import container
from tests.conftest import write_scene
from tests.perception.oracle_detector import GroundTruthDetector
from tests.perception.synthetic_video import write_video
from tests.test_cli_run import CONFIG


@pytest.fixture
def setup(tmp_path, rows, monkeypatch):
    data = write_scene(tmp_path / "data", rows)
    video = write_video(data / "nvidia/MTMC_Tracking_2024/test/scene_001/camera_0007/video.mp4")
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(data))
    monkeypatch.setattr(container, "ensure_weights", lambda d, name: d / name)
    monkeypatch.setattr(container, "load_detector", lambda *_: GroundTruthDetector(rows))
    config = tmp_path / "c.yaml"
    config.write_text(CONFIG)
    return config, video


def parse(argv):
    """The serve arguments as main() parses them."""
    captured = {}

    def keep(args):
        captured["args"] = args
        return 0

    original = cli.run_serve
    cli.run_serve = keep
    try:
        assert cli.main(argv) == 0
    finally:
        cli.run_serve = original
    return captured["args"]


def serving(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def get_json(server, path):
    host, port = server.server_address[:2]
    with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=5) as r:
        return json.loads(r.read())


def test_serve_binds_to_localhost_by_default(tmp_path, setup):
    config, _ = setup
    args = parse(["serve", "--config", str(config), "--db", str(tmp_path / "x.db")])

    assert isinstance(args, argparse.Namespace)
    assert args.host == "127.0.0.1" and args.source is None


def test_serve_records_in_process_and_serves_live_and_slots(tmp_path, setup):
    config, video = setup
    db = tmp_path / "x.db"
    args = parse(
        ["serve", "--config", str(config), "--db", str(db), "--source", str(video), "--port", "0"]
    )

    server, recording, stop = cli.start_serve(args)
    serving(server)
    try:
        recording.join(timeout=30)
        assert not recording.is_alive()
        live = get_json(server, "/api/live?camera=7")
        slots = get_json(server, "/api/slots?camera=7")
    finally:
        stop.set()
        server.shutdown()
        server.server_close()

    assert live["shape"] == [20, 20]
    assert np.asarray(live["dwell"]).sum() == 0  # 2 people on this video: under k (R5)
    assert slots["shape"] == [20, 20] and len(slots["slots"]) in (1, 2)  # 2 across a quarter
    assert live["ended"] is not None  # the file was read to its end


def test_serve_live_carries_the_recorded_people_with_k_1(tmp_path, setup, monkeypatch):
    config, video = setup
    monkeypatch.setattr(cli, "RecordSlots", functools.partial(cli.RecordSlots, k=1))
    args = parse(
        ["serve", "--config", str(config), "--db", str(tmp_path / "x.db"), "--source", str(video)]
        + ["--port", "0"]
    )

    server, recording, stop = cli.start_serve(args)
    serving(server)
    try:
        recording.join(timeout=30)
        live = get_json(server, "/api/live?camera=7")
    finally:
        stop.set()
        server.shutdown()
        server.server_close()

    assert np.asarray(live["dwell"]).sum() > 0 and np.asarray(live["passage"]).sum() > 0


def test_serve_without_a_source_has_no_live_layer(tmp_path, setup):
    config, _ = setup
    args = parse(["serve", "--config", str(config), "--db", str(tmp_path / "x.db"), "--port", "0"])

    server, recording, stop = cli.start_serve(args)
    serving(server)
    try:
        host, port = server.server_address[:2]
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://{host}:{port}/api/live?camera=7", timeout=5)
    finally:
        server.shutdown()
        server.server_close()

    assert recording is None and e.value.code == 404


@pytest.mark.parametrize("error", [RuntimeError("bug"), OSError("cannot open the source")])
def test_a_recorder_that_dies_turns_live_into_a_503(tmp_path, setup, monkeypatch, error):
    config, video = setup
    args = parse(
        [
            "serve",
            "--config",
            str(config),
            "--db",
            str(tmp_path / "x.db"),
            "--source",
            str(video),
            "--port",
            "0",
        ]
    )

    def crash(args, live, stop):
        raise error

    monkeypatch.setattr(cli, "run_record", crash)
    monkeypatch.setattr(threading, "excepthook", lambda a: None)  # the traceback is expected
    server, recording, stop = cli.start_serve(args)
    serving(server)
    try:
        recording.join(timeout=30)
        host, port = server.server_address[:2]
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://{host}:{port}/api/live?camera=7", timeout=5)
    finally:
        stop.set()
        server.shutdown()
        server.server_close()

    assert e.value.code == 503 and "recorder stopped" in json.loads(e.value.read())["error"]


@pytest.mark.parametrize(
    "host, warned", [("127.0.0.1", False), ("localhost", False), ("0.0.0.0", True)]
)
def test_serve_warns_when_bound_beyond_this_machine(tmp_path, setup, capsys, host, warned):
    config, _ = setup
    args = parse(
        [
            "serve",
            "--config",
            str(config),
            "--db",
            str(tmp_path / "x.db"),
            "--port",
            "0",
            "--host",
            host,
        ]
    )

    server, _, _ = cli.start_serve(args)
    server.server_close()

    assert ("warning" in capsys.readouterr().err) is warned


def test_serve_shows_a_fused_export(tmp_path, setup):
    config, _ = setup
    export = tmp_path / "fused.json"
    export.write_text(
        json.dumps(
            {
                "scene": "001",
                "cameras": ["7", "8"],
                "fusion": {"weighting": "mean", "power": 0, "zone_m": None},
                "period": "the whole video",
                "cell_m": 0.5,
                "origin": [0, 0],
                "shape": [1, 2],
                "dwell": [[3.0, 9.0]],
                "passage": [[2.0, 7.0]],
            }
        )
    )
    args = parse(
        ["serve", "--config", str(config), "--db", str(tmp_path / "x.db"), "--port", "0"]
        + ["--fused", str(export)]
    )

    server, recording, _ = cli.start_serve(args)
    serving(server)
    try:
        fused = get_json(server, "/api/fused")
    finally:
        server.shutdown()
        server.server_close()

    assert recording is None
    assert fused["dwell"] == [[0.0, 9.0]] and fused["k"] == 5  # 2 people: suppressed


def test_serve_refuses_a_fused_export_with_global_id_passage(tmp_path, setup):
    # The E2b export's passage counts cross-camera ids, less accurate than E2a's:
    # the viewer's passage layer stays E2a.
    config, _ = setup
    export = tmp_path / "fused.json"
    export.write_text(json.dumps({"passage_ids": "global", "dwell": [[0]], "passage": [[0]]}))

    args = parse(
        ["serve", "--config", str(config), "--db", str(tmp_path / "x.db"), "--port", "0"]
        + ["--fused", str(export)]
    )

    with pytest.raises(ValueError, match="global.*E2a"):  # main prints it, exit 1
        server, _, _ = cli.start_serve(args)
        server.server_close()
