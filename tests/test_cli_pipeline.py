"""space-usage eval --config on the perception pipeline, with a ground-truth
detector in place of YOLOX: video decoding, ByteTrack, the track cache and the
scoring are the real ones."""

import json

import numpy as np

from space_usage_heatmaps import cli
from space_usage_heatmaps.perception.infrastructure import container
from tests.conftest import write_scene
from tests.perception.oracle_detector import GroundTruthDetector
from tests.perception.synthetic_video import write_video

CONFIG = (
    "detector: yolox-s\ntracker: bytetrack\ninput_size: 640\nengine: onnx-cuda\n"
    "stride: 1\nfoot_point: bottom\n"
)


def test_eval_config_runs_the_pipeline_and_caches_its_recipe(tmp_path, rows, monkeypatch, capsys):
    data = write_scene(tmp_path / "data", rows)
    write_video(data / "nvidia/MTMC_Tracking_2024/test/scene_001/camera_0007/video.mp4", n=12)
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(data))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(cli.SPLITS, "tiny", (("001", "7"),))
    loaded = []
    monkeypatch.setattr(container, "ensure_weights", lambda d, name: d / name)
    monkeypatch.setattr(
        container, "load_detector", lambda path: loaded.append(path) or GroundTruthDetector(rows)
    )
    config = tmp_path / "h1.yaml"
    config.write_text(CONFIG)

    args = ["eval", "--config", str(config), "--split", "tiny", "--resamples", "20"]
    assert cli.main(args) == 0
    assert cli.main(args) == 0  # second run from the cache: the detector is not loaded again

    assert loaded == [data / "models" / "yolox-s"]
    out = capsys.readouterr().out
    first, second = [ln.split()[4] for ln in out.splitlines() if ln.startswith("scene 001")]
    path = data / "cache/tracks/yolox-s-bytetrack-640-onnx-cuda-stride1/scene_001_camera_7.npz"
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        assert sorted(f.files) == ["boxes", "frames", "meta", "track_ids"]
        assert int(first) == int(second) == len(f["track_ids"]) > 0
    assert meta["weights_sha256"] == container.WEIGHTS["yolox-s"].sha256
    assert (meta["score_thr"], meta["nms_thr"], meta["input_size"]) == (0.1, 0.45, 640)
    assert meta["tracker"] == "trackers.ByteTrackTracker" and meta["trackers"] == "2.6.1"


def test_the_h1_config_parses():
    c = cli.load_config("configs/h1-yolox-s.yaml")

    assert (c.detector, c.engine, c.tracker, c.input_size, c.foot_point) == (
        "yolox-s", "onnx-cuda", "bytetrack", 640, "auto-track",
    )  # fmt: skip
