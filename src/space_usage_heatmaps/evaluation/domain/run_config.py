"""A run configuration (configs/*.yaml): which detector, tracker and foot point."""

from dataclasses import MISSING, dataclass, field, fields

from space_usage_heatmaps.floorplan.domain.foot_point import (
    FOOT_POINTS,
    HIDDEN_RATIO,
    MIN_IMPLIED_HEIGHT_M,
    PERSON_HEIGHT_M,
    TRACK_HEIGHT_MAX_M,
    TRACK_HEIGHT_MIN_M,
    TRACK_HEIGHT_PERCENTILE,
    TrackHeights,
)

BASELINE = "ultralytics"  # the iteration 0 recipe
# The perception pipeline on ONNX Runtime: CUDA (fp32) or TensorRT (fp16).
PIPELINE_ENGINES = ("onnx-cuda", "onnx-trt")
ENGINES = (BASELINE, *PIPELINE_ENGINES)
HEIGHTS_M = ("person_height_m", "min_implied_height_m", "track_height_min_m", "track_height_max_m")
FRACTIONS = ("track_height_percentile", "hidden_ratio")
# The optional keys each foot point reads; any other one in a file is an error.
FOOT_POINT_KEYS = {
    "centroid": (),
    "bottom": (),
    "head": ("person_height_m",),
    "auto": ("person_height_m", "min_implied_height_m"),
    "auto-track": FRACTIONS + ("track_height_min_m", "track_height_max_m"),
}


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
    track_height_percentile: float = TRACK_HEIGHT_PERCENTILE  # auto-track: quantile, 0 to 1
    hidden_ratio: float = HIDDEN_RATIO  # auto-track: below this share of its track, hidden
    track_height_min_m: float = TRACK_HEIGHT_MIN_M  # auto-track: clip of a track's height
    track_height_max_m: float = TRACK_HEIGHT_MAX_M
    # Overrides of the tracker's defaults by name; the tracker adapter checks the names.
    tracker_params: dict = field(default_factory=dict)
    # Overrides of the detector's thresholds (score_thr, nms_thr), fractions;
    # the detector adapter checks the names.
    detector_params: dict = field(default_factory=dict)

    def track_heights(self):
        """A fresh auto-track height estimator with this config's parameters."""
        return TrackHeights(
            self.track_height_percentile,
            self.hidden_ratio,
            self.track_height_min_m,
            self.track_height_max_m,
        )

    @classmethod
    def from_mapping(cls, d):
        names = [f.name for f in fields(cls)]
        required = [
            f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING
        ]
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
        params = cfg.tracker_params
        if not isinstance(params, dict) or not all(
            isinstance(k, str) and _number(v, 0, float("inf")) for k, v in params.items()
        ):
            raise ValueError("config: tracker_params must map parameter names to positive numbers")
        detector_params = cfg.detector_params
        if not isinstance(detector_params, dict) or not all(
            isinstance(k, str) and _number(v, 0, 1) for k, v in detector_params.items()
        ):
            raise ValueError(
                "config: detector_params must map parameter names to numbers in (0, 1)"
            )
        optional = {n for n in names if n not in required} - {"tracker_params", "detector_params"}
        for name in sorted(optional & set(d) - set(FOOT_POINT_KEYS[cfg.foot_point])):
            raise ValueError(
                f"config: {name} does not apply to foot_point {cfg.foot_point} "
                f"(it reads {list(FOOT_POINT_KEYS[cfg.foot_point]) or 'no parameters'})"
            )
        for name in HEIGHTS_M:
            if not _number(getattr(cfg, name), 0, 3):
                raise ValueError(f"config: {name} must be a height in metres, 0 to 3")
        for name in FRACTIONS:
            if not _number(getattr(cfg, name), 0, 1, closed_high=True):
                raise ValueError(f"config: {name} must be a number in (0, 1]")
        if cfg.track_height_min_m >= cfg.track_height_max_m:
            raise ValueError("config: track_height_min_m must be below track_height_max_m")
        if cfg.engine not in ENGINES:
            raise ValueError(f"config: engine must be one of {ENGINES}")
        if cfg.engine == BASELINE:
            for name in ("tracker_params", "detector_params"):
                if getattr(cfg, name):
                    raise ValueError(f"config: {name} apply to {PIPELINE_ENGINES} only")
        if not _positive_int(cfg.stride):
            raise ValueError("config: stride must be a positive integer")
        if cfg.stride > 1 and cfg.engine == BASELINE:
            raise ValueError(
                f"config: stride above 1 needs {PIPELINE_ENGINES}, which fill the skipped frames"
            )
        if not _positive_int(cfg.input_size) or cfg.input_size % 32:
            raise ValueError("config: input_size must be a positive multiple of 32")
        return cfg


def _number(v, low, high, closed_high=False):
    if isinstance(v, bool) or not isinstance(v, int | float):
        return False
    return low < v <= high if closed_high else low < v < high


def _positive_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0
