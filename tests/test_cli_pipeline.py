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
        container,
        "load_detector",
        lambda path, *_: loaded.append(path) or GroundTruthDetector(rows),
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


def test_the_cache_records_the_provider_each_video_ran_on(tmp_path, rows, monkeypatch):
    class OnCpuAfterOom(GroundTruthDetector):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    data = write_scene(tmp_path / "data", rows)
    write_video(data / "nvidia/MTMC_Tracking_2024/test/scene_001/camera_0007/video.mp4", n=12)
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(data))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(cli.SPLITS, "tiny", (("001", "7"),))
    monkeypatch.setattr(container, "ensure_weights", lambda d, name: d / name)
    monkeypatch.setattr(container, "load_detector", lambda path, *_: OnCpuAfterOom(rows))
    (tmp_path / "h1.yaml").write_text(CONFIG)

    assert cli.main(["eval", "--config", "h1.yaml", "--split", "tiny", "--resamples", "5"]) == 0

    path = data / "cache/tracks/yolox-s-bytetrack-640-onnx-cuda-stride1/scene_001_camera_7.npz"
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
    assert meta["run"] == {"providers": OnCpuAfterOom.providers}
    (report,) = (tmp_path / "reports").rglob("*.json")
    provenance = json.loads(report.read_text())["tracks_provenance"]["h1"]["001/7"]
    assert provenance["stored"]["run"] == {"providers": OnCpuAfterOom.providers}


def test_the_recipe_names_the_opencv_build_that_decodes_and_resizes():
    from importlib import metadata

    meta = container.pipeline_metadata("yolox-s", "bytetrack", 640, 1)

    assert meta["opencv"] == metadata.version("opencv-python-headless")


def test_the_best_config_is_the_accepted_h2_one():
    best, h2 = (
        cli.load_config("configs/h1-best.yaml"),
        cli.load_config("configs/h2-yolox-s-960.yaml"),
    )

    assert best == h2 and (best.detector, best.input_size) == ("yolox-s", 960)


def test_a_second_tracker_replays_the_cached_detections(tmp_path, rows, monkeypatch, capsys):
    data = write_scene(tmp_path / "data", rows)
    write_video(data / "nvidia/MTMC_Tracking_2024/test/scene_001/camera_0007/video.mp4", n=12)
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(data))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(cli.SPLITS, "tiny", (("001", "7"),))
    loaded = []
    monkeypatch.setattr(container, "ensure_weights", lambda d, name: d / name)
    monkeypatch.setattr(
        container,
        "load_detector",
        lambda path, *_: loaded.append(path) or GroundTruthDetector(rows),
    )
    for tracker in ("bytetrack", "ocsort", "ucmc"):
        (tmp_path / f"{tracker}.yaml").write_text(CONFIG.replace("bytetrack", tracker))
        args = ["eval", "--config", f"{tracker}.yaml", "--split", "tiny", "--resamples", "5"]
        assert cli.main(args) == 0

    assert len(loaded) == 1  # the detector ran for the first tracker only
    detections = data / "cache/detections/yolox-s-640-onnx-cuda-stride1/scene_001_camera_7.npz"
    with np.load(detections) as f:
        assert sorted(f.files) == ["box_frames", "boxes", "frames", "meta", "scores"]
        assert len(f["frames"]) == 12
        assert "tracker" not in json.loads(str(f["meta"]))
    for tracker in ("bytetrack", "ocsort", "ucmc"):
        tracks = data / f"cache/tracks/yolox-s-{tracker}-640-onnx-cuda-stride1"
        assert (tracks / "scene_001_camera_7.npz").exists()


def test_the_h3_configs_change_only_the_tracker_of_h1_best():
    best = cli.load_config("configs/h1-best.yaml")
    for tracker in ("ocsort", "ucmc"):
        c = cli.load_config(f"configs/h3-{tracker}.yaml")
        assert c.tracker == tracker and c == type(best)(**{**vars(best), "tracker": tracker})


def test_the_caches_check_the_video_and_ucmc_tracks_the_calibration(tmp_path, rows, monkeypatch):
    data = write_scene(tmp_path / "data", rows)
    video = data / "nvidia/MTMC_Tracking_2024/test/scene_001/camera_0007/video.mp4"
    write_video(video, n=12)
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(data))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(cli.SPLITS, "tiny", (("001", "7"),))
    loaded = []
    monkeypatch.setattr(container, "ensure_weights", lambda d, name: d / name)
    monkeypatch.setattr(
        container,
        "load_detector",
        lambda path, *_: loaded.append(path) or GroundTruthDetector(rows),
    )
    (tmp_path / "ucmc.yaml").write_text(CONFIG.replace("bytetrack", "ucmc"))
    args = ["eval", "--config", "ucmc.yaml", "--split", "tiny", "--resamples", "5"]

    assert cli.main(args) == 0
    tracks = data / "cache/tracks/yolox-s-ucmc-640-onnx-cuda-stride1/scene_001_camera_7.npz"
    detections = data / "cache/detections/yolox-s-640-onnx-cuda-stride1/scene_001_camera_7.npz"
    with np.load(tracks) as t, np.load(detections) as d:
        track_meta, detection_meta = json.loads(str(t["meta"])), json.loads(str(d["meta"]))
    assert detection_meta["video_bytes"] == track_meta["video_bytes"] == video.stat().st_size
    assert len(track_meta["homography_sha256"]) == 64
    assert "homography_sha256" not in detection_meta

    write_video(video, n=10)  # another video under the same name
    assert cli.main(args) == 0
    assert len(loaded) == 2


def test_tracker_params_get_their_own_track_cache(tmp_path):
    base = cli.RunConfig.from_mapping(
        {"detector": "yolox-s", "tracker": "ucmc", "input_size": 960, "engine": "onnx-cuda",
         "stride": 1, "foot_point": "bottom"}
    )  # fmt: skip
    low = cli.RunConfig(**{**vars(base), "tracker_params": {"conf_thresh": 0.1}})

    assert cli.cache_key(base) == "yolox-s-ucmc-960-onnx-cuda-stride1"
    assert cli.cache_key(low).startswith(cli.cache_key(base) + "-params")
    assert cli.cache_key(low) == cli.cache_key(
        cli.RunConfig(**{**vars(base), "tracker_params": {"conf_thresh": 0.1}})
    )
    assert cli.detection_cache_key(low) == cli.detection_cache_key(base)


def test_a_broken_tracking_guard_fails_the_eval(tmp_path, rows, monkeypatch, capsys):
    data = write_scene(tmp_path / "data", rows)
    write_video(data / "nvidia/MTMC_Tracking_2024/test/scene_001/camera_0007/video.mp4", n=12)
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(data))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(cli.SPLITS, "tiny", (("001", "7"),))
    monkeypatch.setattr(container, "ensure_weights", lambda d, name: d / name)
    monkeypatch.setattr(container, "load_detector", lambda path, *_: GroundTruthDetector(rows))
    real = cli.score_tracking

    def worse(*a, **k):
        t = real(*a, **k)
        t["delta"]["pooled"]["hota"] = -0.02  # two points lower
        return t

    monkeypatch.setattr(cli, "score_tracking", worse)
    (tmp_path / "a.yaml").write_text(CONFIG)
    (tmp_path / "b.yaml").write_text(CONFIG)
    args = ["eval", "--config", "a.yaml", "--compare", "b.yaml", "--split", "tiny"]

    assert cli.main([*args, "--resamples", "5"]) == cli.GUARD_BROKEN
    assert "BROKEN" in capsys.readouterr().out
    (report,) = (tmp_path / "reports").rglob("*.json")
    assert json.loads(report.read_text())["tracking"]["guard_holds"] is False
