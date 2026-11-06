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


def test_the_low_threshold_h3_configs_change_only_the_tracker_params():
    for tracker, params in (
        ("ocsort", {"high_conf_det_threshold": 0.1}),
        (
            "ucmc",
            {"conf_thresh": 0.1, "high_score": 0.6, "wx": 0.2, "wy": 0.4, "a1": 10.0, "a2": 10.0},
        ),
    ):
        h3 = cli.load_config(f"configs/h3-{tracker}.yaml")
        low = cli.load_config(f"configs/h3-{tracker}-lowthr.yaml")
        assert low == type(h3)(**{**vars(h3), "tracker_params": params})
        container.tracker_metadata(tracker, params)  # every name is the tracker's


def _tiny(tmp_path, rows, monkeypatch, n=12):
    """The tiny scene and video; the detector is the truth, scored 0.3 on frame 5."""

    class Scored(GroundTruthDetector):
        def detect(self, frame):
            d = super().detect(frame)
            return type(d)(d.boxes, np.full(len(d.scores), 0.3 if frame.index == 5 else 0.9))

    data = write_scene(tmp_path / "data", rows)
    write_video(data / "nvidia/MTMC_Tracking_2024/test/scene_001/camera_0007/video.mp4", n=n)
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(data))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(cli.SPLITS, "tiny", (("001", "7"),))
    monkeypatch.setattr(container, "ensure_weights", lambda d, name: d / name)
    loaded = []
    monkeypatch.setattr(
        container,
        "load_detector",
        lambda path, detector, params=None, engine=None: (
            loaded.append((params, engine)) or Scored(rows)
        ),
    )
    return data, loaded


def _eval(tmp_path, name, extra):
    (tmp_path / f"{name}.yaml").write_text(CONFIG + extra)
    return cli.main(["eval", "--config", f"{name}.yaml", "--split", "tiny", "--resamples", "5"])


def _tracks(data, key):
    with np.load(data / f"cache/tracks/{key}/scene_001_camera_7.npz") as f:
        return f["frames"].copy(), json.loads(str(f["meta"]))


def test_a_higher_score_threshold_filters_the_cached_detections(tmp_path, rows, monkeypatch):
    data, loaded = _tiny(tmp_path, rows, monkeypatch)

    assert _eval(tmp_path, "base", "") == 0
    assert _eval(tmp_path, "strict", "detector_params:\n  score_thr: 0.5\n") == 0

    assert len(loaded) == 1  # the second config replayed the first one's detections
    assert sorted(p.name for p in (data / "cache/detections").iterdir()) == [
        "yolox-s-640-onnx-cuda-stride1"
    ]
    base = cli.load_config("base.yaml")
    strict = cli.load_config("strict.yaml")
    base_frames, _ = _tracks(data, cli.cache_key(base))
    strict_frames, meta = _tracks(data, cli.cache_key(strict))
    assert meta["score_thr"] == 0.5 and cli.cache_key(strict) != cli.cache_key(base)
    assert 0 < len(strict_frames) < len(base_frames)


def test_a_lower_score_threshold_or_another_nms_runs_the_detector_again(
    tmp_path, rows, monkeypatch
):
    data, loaded = _tiny(tmp_path, rows, monkeypatch)

    assert _eval(tmp_path, "low", "detector_params:\n  score_thr: 0.05\n") == 0
    assert _eval(tmp_path, "nms", "detector_params:\n  nms_thr: 0.6\n") == 0

    assert [p for p, _ in loaded] == [{"score_thr": 0.05}, {"nms_thr": 0.6}]
    assert len(list((data / "cache/detections").iterdir())) == 2


def test_a_stride_replays_every_frame_detections_and_fills_the_skipped_frames(
    tmp_path, rows, monkeypatch
):
    data, loaded = _tiny(tmp_path, rows, monkeypatch)

    assert _eval(tmp_path, "every", "") == 0
    (tmp_path / "half.yaml").write_text(CONFIG.replace("stride: 1", "stride: 2"))
    assert cli.main(["eval", "--config", "half.yaml", "--split", "tiny", "--resamples", "5"]) == 0

    assert len(loaded) == 1
    every, _ = _tracks(data, cli.cache_key(cli.load_config("every.yaml")))
    half, meta = _tracks(data, cli.cache_key(cli.load_config("half.yaml")))
    assert meta["stride"] == 2 and "stride_fill" in meta
    assert set(half) & {1, 3, 5}  # odd frames come back, filled


def test_the_tensorrt_engine_has_its_own_detections(tmp_path, rows, monkeypatch):
    data, loaded = _tiny(tmp_path, rows, monkeypatch)

    assert _eval(tmp_path, "cuda", "") == 0
    (tmp_path / "trt.yaml").write_text(CONFIG.replace("onnx-cuda", "onnx-trt"))
    assert cli.main(["eval", "--config", "trt.yaml", "--split", "tiny", "--resamples", "5"]) == 0

    assert [e for _, e in loaded] == ["onnx-cuda", "onnx-trt"]
    assert (data / "cache/detections/yolox-s-640-onnx-trt-stride1").is_dir()


def test_the_bench_names_the_engine_and_precision_that_ran():
    cuda = cli.engine_label("onnx-cuda", ["CUDAExecutionProvider"], "1.31.1")
    trt = cli.engine_label("onnx-trt", ["TensorrtExecutionProvider"], "1.31.1")
    fell = cli.engine_label("onnx-trt", ["CUDAExecutionProvider"], "1.31.1")

    assert cuda == "onnx-cuda (ONNX Runtime 1.31.1, CUDAExecutionProvider, fp32)"
    assert trt == "onnx-trt (ONNX Runtime 1.31.1, TensorrtExecutionProvider, fp16)"
    assert fell.endswith("CUDAExecutionProvider, fp32)")


def test_detector_params_get_their_own_track_cache():
    base = cli.load_config("configs/h1-best.yaml")
    low = cli.RunConfig(**{**vars(base), "detector_params": {"score_thr": 0.05}})

    assert cli.cache_key(base) == "yolox-s-bytetrack-960-onnx-cuda-stride1"
    assert cli.cache_key(low).startswith(cli.cache_key(base) + "-params")
    assert cli.detection_cache_key(base) == "yolox-s-960-onnx-cuda-stride1"
    assert cli.detection_cache_key(low).startswith("yolox-s-960-onnx-cuda-stride1-params")
