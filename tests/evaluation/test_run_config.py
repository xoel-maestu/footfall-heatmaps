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
        ("track_height_percentile", 1.5, "track_height_percentile"),
        ("hidden_ratio", 0, "hidden_ratio"),
        ("track_height_min_m", 2.5, "track_height_min_m"),
        ("engine", "tensorrt-int4", "engine"),
        ("stride", 0, "stride"),
        ("stride", 1.5, "stride"),
        ("input_size", 650, "input_size"),
        ("detector", "", "detector"),
    ],
)
def test_an_invalid_value_names_the_key(key, value, message):
    mode = {"person_height_m": "auto", "min_implied_height_m": "auto"}.get(key, "auto-track")
    base = ITER0 if key not in MODE_KEYS else ITER0 | {"foot_point": mode}
    with pytest.raises(ValueError, match=message):
        RunConfig.from_mapping(base | {key: value})


MODE_KEYS = (
    "person_height_m",
    "min_implied_height_m",
    "track_height_percentile",
    "hidden_ratio",
    "track_height_min_m",
)


@pytest.mark.parametrize(
    ("foot_point", "key", "value"),
    [
        ("auto-track", "person_height_m", 1.70),
        ("auto-track", "min_implied_height_m", 1.40),
        ("auto", "hidden_ratio", 0.88),
        ("head", "min_implied_height_m", 1.40),
        ("bottom", "person_height_m", 1.70),
        ("centroid", "track_height_percentile", 0.9),
    ],
)
def test_a_parameter_of_another_foot_point_is_an_error(foot_point, key, value):
    with pytest.raises(ValueError, match=f"{key} does not apply to foot_point {foot_point}"):
        RunConfig.from_mapping(ITER0 | {"foot_point": foot_point, key: value})


def test_auto_track_heights_need_min_below_max():
    with pytest.raises(ValueError, match="track_height_min_m"):
        RunConfig.from_mapping(
            ITER0
            | {"foot_point": "auto-track", "track_height_min_m": 1.9, "track_height_max_m": 1.5}
        )


def test_missing_and_unknown_keys_are_errors():
    with pytest.raises(ValueError, match="missing.*stride"):
        RunConfig.from_mapping({k: v for k, v in ITER0.items() if k != "stride"})
    with pytest.raises(ValueError, match="unknown.*half"):
        RunConfig.from_mapping(ITER0 | {"half": True})


def test_the_h4_auto_file_is_iter0_with_the_auto_foot_point():
    cfg = RunConfig.from_mapping(yaml.safe_load(Path("configs/h4-auto.yaml").read_text()))

    assert cfg == RunConfig(**(ITER0 | {"foot_point": "auto"}))
    assert (cfg.person_height_m, cfg.min_implied_height_m) == (1.70, 1.40)


def test_the_h4_auto_track_file_names_its_parameters_with_the_tuned_values():
    cfg = RunConfig.from_mapping(yaml.safe_load(Path("configs/h4-auto-track.yaml").read_text()))

    assert cfg == RunConfig(**(ITER0 | {"foot_point": "auto-track"}))
    heights = cfg.track_heights()
    assert (heights.percentile, heights.hidden_ratio, heights.min_m, heights.max_m) == (
        0.9,
        0.88,
        1.0,
        2.0,
    )
