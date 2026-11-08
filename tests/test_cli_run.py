"""space-usage run on a video file, with a ground-truth detector in place of YOLOX:
decoding, ByteTrack, the foot point, the slots and SQLite are the real ones."""

import sqlite3

import pytest

from space_usage_heatmaps import cli
from space_usage_heatmaps.perception.infrastructure import container
from tests.conftest import write_scene
from tests.perception.oracle_detector import GroundTruthDetector
from tests.perception.synthetic_video import write_video

CONFIG = """detector: yolox-s
tracker: bytetrack
input_size: 640
engine: onnx-cuda
stride: 1
foot_point: bottom
run:
  camera_id: "7"
  calibration: nvidia-mtmc/001/7
  extent_m: [-4.0, 0.0, 4.0, 8.0]
  cell_m: 0.5
  retention_days: 90
  masks:
    image: []
    floor: []
"""


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


def test_run_writes_the_slots_of_the_video(tmp_path, setup, capsys):
    config, video = setup
    db = tmp_path / "x.db"

    args = ["run", "--config", str(config), "--source", str(video), "--db", str(db)]
    assert cli.main([*args, "--start", "2026-11-06T09:14:59Z"]) == 0

    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT start_utc, layer, incomplete FROM slots ORDER BY 1, 2").fetchall()
    assert rows == [
        ("2026-11-06T09:00:00Z", "dwell", 1),
        ("2026-11-06T09:00:00Z", "passage", 1),
        ("2026-11-06T09:15:00Z", "dwell", 1),
        ("2026-11-06T09:15:00Z", "passage", 1),
    ]
    out = capsys.readouterr().out
    assert "12 frames" in out and "2 slots" in out


def test_the_start_can_come_from_the_config(tmp_path, setup):
    config, video = setup
    config.write_text(CONFIG + "  start_utc: 2026-11-06T10:00:00Z\n")
    db = tmp_path / "x.db"

    assert cli.main(["run", "--config", str(config), "--source", str(video), "--db", str(db)]) == 0

    with sqlite3.connect(db) as c:
        assert c.execute("SELECT DISTINCT start_utc FROM slots").fetchall() == [
            ("2026-11-06T10:00:00Z",)
        ]


def test_a_config_without_a_run_section_is_an_error(tmp_path, setup, capsys):
    config, video = setup
    config.write_text(CONFIG.split("run:")[0])

    args = ["run", "--config", str(config), "--source", str(video), "--db", str(tmp_path / "x.db")]
    assert cli.main(args) == 1
    assert "run section" in capsys.readouterr().err


def test_the_accepted_config_records_scene_071_camera_645():
    settings = cli.load_run_settings("configs/h5-best.yaml")

    assert (settings.camera_id, settings.calibration) == ("645", "nvidia-mtmc/071/645")
    assert cli.load_config("configs/h5-best.yaml").detector == "yolox-s"


def test_a_start_past_the_retention_fails_instead_of_reporting_slots(tmp_path, setup, capsys):
    config, video = setup
    db = tmp_path / "x.db"

    args = ["run", "--config", str(config), "--source", str(video), "--db", str(db)]
    assert cli.main([*args, "--start", "2025-01-01T00:00:00Z"]) != 0

    captured = capsys.readouterr()
    assert "past retention" in captured.err
    assert "0 slots" in captured.out
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM slots").fetchone() == (0,)


def test_run_counts_k_as_the_run_section_says(tmp_path, setup, monkeypatch):
    from space_usage_heatmaps.occupancy.domain.slots import DEFAULT_K_COUNT
    from space_usage_heatmaps.occupancy.domain.stitching import K_STITCH

    config, video = setup
    seen = []
    real = cli.RecordSlots

    def spy(*args, **kwargs):
        seen.append(kwargs.get("stitch"))
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "RecordSlots", spy)
    base = ["run", "--config", str(config), "--source", str(video)]
    assert cli.main([*base, "--db", str(tmp_path / "a.db")]) == 0
    for k_count in ("ids", "chains"):
        config.write_text(CONFIG + f"  k_count: {k_count}\n")
        assert cli.main([*base, "--db", str(tmp_path / f"{k_count}.db")]) == 0

    default = K_STITCH if DEFAULT_K_COUNT == "chains" else None
    assert seen == [default, None, K_STITCH]


def test_an_unknown_k_count_is_an_error(tmp_path, setup, capsys):
    config, video = setup
    config.write_text(CONFIG + "  k_count: people\n")

    args = ["run", "--config", str(config), "--source", str(video), "--db", str(tmp_path / "x.db")]
    assert cli.main(args) == 1
    assert "k_count" in capsys.readouterr().err
