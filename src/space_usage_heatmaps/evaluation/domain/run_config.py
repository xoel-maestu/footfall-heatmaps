"""A run configuration (configs/*.yaml): which detector, tracker and foot point."""

from dataclasses import MISSING, dataclass, fields

from space_usage_heatmaps.floorplan.domain.foot_point import (
    FOOT_POINTS,
    MIN_IMPLIED_HEIGHT_M,
    PERSON_HEIGHT_M,
)

ENGINES = ("ultralytics",)


@dataclass(frozen=True)
class RunConfig:
    detector: str  # model name, e.g. yolov8n
    tracker: str  # tracker name, e.g. bytetrack
    input_size: int  # detector input side in pixels, a multiple of 32
    foot_point: str  # image point of a box that goes through the homography
    engine: str  # inference runtime
    stride: int  # process one frame in stride
    person_height_m: float = PERSON_HEIGHT_M  # head foot point: height of the box top
    min_implied_height_m: float = MIN_IMPLIED_HEIGHT_M  # auto: shorter boxes use the head

    @classmethod
    def from_mapping(cls, d):
        names = [f.name for f in fields(cls)]
        required = [f.name for f in fields(cls) if f.default is MISSING]
        if missing := [n for n in required if n not in d]:
            raise ValueError(f"config: missing keys {missing}")
        if unknown := sorted(set(d) - set(names)):
            raise ValueError(f"config: unknown keys {unknown}")
        cfg = cls(**{n: d[n] for n in names if n in d})
        for name in ("detector", "tracker"):
            if not isinstance(getattr(cfg, name), str) or not getattr(cfg, name):
                raise ValueError(f"config: {name} must be a non-empty string")
        if cfg.foot_point not in FOOT_POINTS:
            raise ValueError(f"config: foot_point must be one of {FOOT_POINTS}")
        for name in ("person_height_m", "min_implied_height_m"):
            v = getattr(cfg, name)
            if isinstance(v, bool) or not isinstance(v, int | float) or not 0 < v < 3:
                raise ValueError(f"config: {name} must be a height in metres, 0 to 3")
        if cfg.engine not in ENGINES:
            raise ValueError(f"config: engine must be one of {ENGINES}")
        if not _positive_int(cfg.stride):
            raise ValueError("config: stride must be a positive integer")
        if not _positive_int(cfg.input_size) or cfg.input_size % 32:
            raise ValueError("config: input_size must be a positive multiple of 32")
        return cfg


def _positive_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0
