"""A run configuration (configs/*.yaml): which detector, tracker and foot point."""

from dataclasses import dataclass, fields

FOOT_POINTS = ("centroid", "bottom-centre")
ENGINES = ("ultralytics",)


@dataclass(frozen=True)
class RunConfig:
    detector: str  # model name, e.g. yolov8n
    tracker: str  # tracker name, e.g. bytetrack
    input_size: int  # detector input side in pixels, a multiple of 32
    foot_point: str  # image point of a box that goes through the homography
    engine: str  # inference runtime
    stride: int  # process one frame in stride

    @classmethod
    def from_mapping(cls, d):
        names = [f.name for f in fields(cls)]
        if missing := [n for n in names if n not in d]:
            raise ValueError(f"config: missing keys {missing}")
        if unknown := sorted(set(d) - set(names)):
            raise ValueError(f"config: unknown keys {unknown}")
        cfg = cls(**{n: d[n] for n in names})
        for name in ("detector", "tracker"):
            if not isinstance(getattr(cfg, name), str) or not getattr(cfg, name):
                raise ValueError(f"config: {name} must be a non-empty string")
        if cfg.foot_point not in FOOT_POINTS:
            raise ValueError(f"config: foot_point must be one of {FOOT_POINTS}")
        if cfg.engine not in ENGINES:
            raise ValueError(f"config: engine must be one of {ENGINES}")
        if not _positive_int(cfg.stride):
            raise ValueError("config: stride must be a positive integer")
        if not _positive_int(cfg.input_size) or cfg.input_size % 32:
            raise ValueError("config: input_size must be a positive multiple of 32")
        return cfg


def _positive_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0
