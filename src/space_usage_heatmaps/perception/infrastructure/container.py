"""Wiring of the perception pipeline: video file -> a detector on ONNX Runtime -> a tracker.

The config's detector and input_size pick the model file (MODELS); the
detector's family picks the adapter, with its own pre- and post-processing.
The config's tracker picks the tracker adapter (TRACKERS).

detection_metadata names everything the detections depend on, for the
detection cache; pipeline_metadata adds the tracker, for the track cache. Both
read package metadata only, so they work without the runtime group.
"""

from importlib import metadata

from space_usage_heatmaps.perception.infrastructure import onnx_dfine_detector as dfine
from space_usage_heatmaps.perception.infrastructure import onnx_rfdetr_detector as rfdetr
from space_usage_heatmaps.perception.infrastructure import onnx_yolox_detector as yolox
from space_usage_heatmaps.perception.infrastructure.bytetrack_tracker import (
    DEFAULTS as BYTETRACK_DEFAULTS,
)
from space_usage_heatmaps.perception.infrastructure.bytetrack_tracker import ByteTrackTracker
from space_usage_heatmaps.perception.infrastructure.model_weights import WEIGHTS, ensure_weights
from space_usage_heatmaps.perception.infrastructure.ocsort_tracker import (
    DEFAULTS as OCSORT_DEFAULTS,
)
from space_usage_heatmaps.perception.infrastructure.ocsort_tracker import OcSortTracker
from space_usage_heatmaps.perception.infrastructure.opencv_file_source import OpenCvFileSource
from space_usage_heatmaps.perception.infrastructure.ucmc_tracker import (
    DEFAULTS as UCMC_DEFAULTS,
)
from space_usage_heatmaps.perception.infrastructure.ucmc_tracker import (
    VERSION as UCMC_VERSION,
)
from space_usage_heatmaps.perception.infrastructure.ucmc_tracker import UcmcTracker

TRACKERS = ("bytetrack", "ocsort", "ucmc")
ENGINE = "onnx-cuda"
# (config detector, input size) -> model file (model_weights.WEIGHTS key)
MODELS = {
    ("yolox-s", 640): "yolox-s",
    ("yolox-s", 960): "yolox-s-960",
    ("yolox-s", 1280): "yolox-s-1280",
    ("dfine-s", 640): "dfine-s",
    ("rfdetr-n", 384): "rfdetr-n",
}
# config detector -> adapter module: Detector.load(path) and recipe()
FAMILIES = {"yolox-s": yolox, "dfine-s": dfine, "rfdetr-n": rfdetr}


def _version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def model_file(detector, input_size):
    """The model file's WEIGHTS key."""
    if (detector, input_size) not in MODELS:
        known = ", ".join(f"{d} at {s}" for d, s in MODELS)
        raise ValueError(f"engine {ENGINE} runs {known}; not {detector} at {input_size}")
    return MODELS[detector, input_size]


def check(detector, tracker, input_size):
    """The model file's WEIGHTS key, once the tracker is known too."""
    if tracker not in TRACKERS:
        raise ValueError(f"engine {ENGINE} runs trackers {list(TRACKERS)}, not {tracker!r}")
    return model_file(detector, input_size)


def detection_metadata(detector, input_size, stride):
    model = model_file(detector, input_size)
    return {
        "detector": detector,
        "weights": WEIGHTS[model].file,
        "weights_sha256": WEIGHTS[model].sha256,
        "engine": ENGINE,
        "onnxruntime": _version("onnxruntime-gpu"),
        "opencv": _version("opencv-python-headless"),  # decoding and the resize
        **FAMILIES[detector].recipe(),
        "input_size": input_size,
        "stride": stride,
    }


def tracker_metadata(tracker, params=None):
    """The tracker's recipe: its defaults with the config's overrides (params)."""
    defaults = {"bytetrack": BYTETRACK_DEFAULTS, "ocsort": OCSORT_DEFAULTS, "ucmc": UCMC_DEFAULTS}
    if tracker not in defaults:
        raise ValueError(f"engine {ENGINE} runs trackers {list(TRACKERS)}, not {tracker!r}")
    params = dict(params or {})
    if unknown := sorted(set(params) - set(defaults[tracker])):
        raise ValueError(
            f"{tracker} has no parameters {unknown}; it takes {sorted(defaults[tracker])}"
        )
    tracker_params = defaults[tracker] | params | {"frame_rate": "video fps / stride"}
    if tracker == "ucmc":
        return {
            "tracker": f"space_usage_heatmaps UcmcTracker v{UCMC_VERSION}",
            "scipy": _version("scipy"),
            "tracker_params": tracker_params,
        }
    name = {"bytetrack": "trackers.ByteTrackTracker", "ocsort": "trackers.OCSORTTracker"}[tracker]
    return {"tracker": name, "trackers": _version("trackers"), "tracker_params": tracker_params}


def pipeline_metadata(detector, tracker, input_size, stride, params=None):
    check(detector, tracker, input_size)
    return detection_metadata(detector, input_size, stride) | tracker_metadata(tracker, params)


def load_detector(path, detector):
    return FAMILIES[detector].load(path)


def video_fps(video):
    return OpenCvFileSource(video).fps


def build_detection(video, detector, input_size, stride, models_dir, loop=False):
    """(frame source, person detector) for one video."""
    weights = model_file(detector, input_size)
    source = OpenCvFileSource(video, stride, loop)
    model = load_detector(ensure_weights(models_dir, weights), detector)
    if model.input_size != input_size:
        raise ValueError(
            f"config input_size {input_size}, but {WEIGHTS[weights].file} takes {model.input_size}"
        )
    return source, model


def build_tracker(tracker, frame_rate, world_to_image=None, params=None):
    """A fresh tracker. world_to_image: the camera's floor homography (metres ->
    pixels), for trackers that associate on the floor. params: overrides of the
    tracker's defaults by name (the config's tracker_params)."""
    params = dict(params or {})
    if tracker == "bytetrack":
        return ByteTrackTracker(frame_rate=frame_rate, **params)
    if tracker == "ocsort":
        return OcSortTracker(frame_rate=frame_rate, **params)
    if tracker == "ucmc":
        if world_to_image is None:
            raise ValueError("tracker ucmc associates on the floor: it needs the camera homography")
        return UcmcTracker(frame_rate, world_to_image, params)
    raise ValueError(f"engine {ENGINE} runs trackers {list(TRACKERS)}, not {tracker!r}")


def build_pipeline(
    video,
    detector,
    tracker,
    input_size,
    stride,
    models_dir,
    loop=False,
    world_to_image=None,
    tracker_params=None,
):
    """(frame source, person detector, tracker) for one video; a fresh tracker each time."""
    check(detector, tracker, input_size)
    source, model = build_detection(video, detector, input_size, stride, models_dir, loop)
    tracker = build_tracker(tracker, source.fps / stride, world_to_image, tracker_params)
    return source, model, tracker
