import pytest

from space_usage_heatmaps.perception.infrastructure import container


def test_detector_and_input_size_pick_the_model_file():
    assert container.check("yolox-s", "bytetrack", 640) == "yolox-s"
    assert container.check("yolox-s", "bytetrack", 960) == "yolox-s-960"


def test_a_size_without_a_model_file_is_rejected():
    with pytest.raises(ValueError, match="yolox-s at 640"):
        container.check("yolox-s", "bytetrack", 1024)


def test_the_metadata_names_the_model_file_and_its_sha256():
    meta = container.pipeline_metadata("yolox-s", "bytetrack", 960, 1)

    assert meta["weights"] == "yolox_s_960.onnx"
    assert meta["weights_sha256"] == container.WEIGHTS["yolox-s-960"].sha256
    assert meta["input_size"] == 960


def test_the_yolox_640_recipe_is_unchanged_so_its_cached_tracks_stay_valid():
    meta = container.pipeline_metadata("yolox-s", "bytetrack", 640, 1)

    assert {k: meta[k] for k in ("preprocess", "postprocess", "nms", "classes")} == {
        "preprocess": "yolox letterbox, top-left, pad 114, BGR 0-255",
        "postprocess": "boxes clipped to the image",
        "nms": "class-agnostic",
        "classes": [0],
    }
    assert meta["weights"] == "yolox_s.onnx"


def test_dfine_has_its_own_recipe():
    meta = container.pipeline_metadata("dfine-s", "bytetrack", 640, 1)

    assert meta["weights"] == "dfine_s_obj2coco.onnx"
    assert meta["nms"] == "none" and meta["classes"] == [0]
    assert meta["preprocess"].startswith("dfine letterbox")


def test_rfdetr_nano_runs_at_its_384_input():
    meta = container.pipeline_metadata("rfdetr-n", "bytetrack", 384, 1)

    assert meta["weights"] == "rf_detr_nano.onnx" and meta["classes"] == [1]
    with pytest.raises(ValueError):
        container.check("rfdetr-n", "bytetrack", 640)


def test_every_model_file_is_registered_with_an_adapter():
    for detector, _ in container.MODELS:
        assert detector in container.FAMILIES
    assert set(container.MODELS.values()) <= set(container.WEIGHTS)


def test_a_model_whose_input_differs_from_the_config_is_an_error(monkeypatch, tmp_path):
    source = type("Source", (), {"fps": 30.0})()
    monkeypatch.setattr(container, "OpenCvFileSource", lambda *a: source)
    monkeypatch.setattr(container, "ensure_weights", lambda d, name: d / name)
    model = type("Model", (), {"input_size": 640})()
    monkeypatch.setattr(container, "load_detector", lambda path, detector: model)

    with pytest.raises(ValueError, match="960.*640"):
        container.build_pipeline(tmp_path / "v.mp4", "yolox-s", "bytetrack", 960, 1, tmp_path)

    model.input_size = 960
    assert container.build_pipeline(tmp_path / "v.mp4", "yolox-s", "bytetrack", 960, 1, tmp_path)


def test_the_bytetrack_recipe_is_unchanged_so_the_h1_best_tracks_stay_valid():
    meta = container.pipeline_metadata("yolox-s", "bytetrack", 960, 1)

    assert meta["tracker"] == "trackers.ByteTrackTracker"
    assert meta["tracker_params"] == container.BYTETRACK_DEFAULTS | {
        "frame_rate": "video fps / stride"
    }
    assert set(meta) == set(container.detection_metadata("yolox-s", 960, 1)) | {
        "tracker", "trackers", "tracker_params",
    }  # fmt: skip


def test_each_tracker_has_its_own_track_recipe_on_the_same_detections():
    metas = {t: container.pipeline_metadata("yolox-s", t, 960, 1) for t in container.TRACKERS}
    detections = container.detection_metadata("yolox-s", 960, 1)

    assert len({m["tracker"] for m in metas.values()}) == len(container.TRACKERS)
    for m in metas.values():
        assert {k: m[k] for k in detections} == detections
    assert metas["ocsort"]["tracker_params"]["minimum_iou_threshold"] == 0.3
    assert metas["ucmc"]["tracker_params"]["a1"] == 10.0


def test_build_tracker_picks_the_adapter_and_ucmc_needs_the_homography():
    from space_usage_heatmaps.perception.infrastructure.ucmc_tracker import UcmcTracker

    assert type(container.build_tracker("ocsort", 30.0)).__name__ == "OcSortTracker"
    assert isinstance(
        container.build_tracker("ucmc", 30.0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), UcmcTracker
    )
    with pytest.raises(ValueError, match="homography"):
        container.build_tracker("ucmc", 30.0)
    with pytest.raises(ValueError, match="sort"):
        container.build_tracker("sort", 30.0)
