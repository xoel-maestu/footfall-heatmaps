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
        ("foot_point", "feet", "foot_point"),
        ("person_height_m", 0, "person_height_m"),
        ("min_implied_height_m", "1.4", "min_implied_height_m"),
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


def test_the_h4_auto_file_is_iter0_with_the_auto_foot_point():
    cfg = RunConfig.from_mapping(yaml.safe_load(Path("configs/h4-auto.yaml").read_text()))

    assert cfg == RunConfig(**(ITER0 | {"foot_point": "auto"}))
    assert (cfg.person_height_m, cfg.min_implied_height_m) == (1.70, 1.40)
