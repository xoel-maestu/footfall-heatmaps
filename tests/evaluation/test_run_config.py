from pathlib import Path

import pytest
import yaml

from space_usage_heatmaps.evaluation.domain.run_config import RunConfig

ITER0 = {
    "detector": "yolov8n",
    "tracker": "bytetrack",
    "input_size": 640,
    "foot_point": "centroid",
    "engine": "ultralytics",
    "stride": 1,
}


def test_the_iter0_file_parses_to_the_ultralytics_recipe():
    cfg = RunConfig.from_mapping(yaml.safe_load(Path("configs/iter0.yaml").read_text()))

    assert cfg == RunConfig(**ITER0)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("foot_point", "head", "foot_point"),
        ("engine", "tensorrt-int4", "engine"),
        ("stride", 0, "stride"),
        ("stride", 1.5, "stride"),
        ("input_size", 650, "input_size"),
        ("detector", "", "detector"),
    ],
)
def test_an_invalid_value_names_the_key(key, value, message):
    with pytest.raises(ValueError, match=message):
        RunConfig.from_mapping(ITER0 | {key: value})


def test_missing_and_unknown_keys_are_errors():
    with pytest.raises(ValueError, match="missing.*stride"):
        RunConfig.from_mapping({k: v for k, v in ITER0.items() if k != "stride"})
    with pytest.raises(ValueError, match="unknown.*half"):
        RunConfig.from_mapping(ITER0 | {"half": True})
