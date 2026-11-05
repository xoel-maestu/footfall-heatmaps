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
