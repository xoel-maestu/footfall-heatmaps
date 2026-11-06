"""Wiring of the perception pipeline: video file -> a detector on ONNX Runtime -> ByteTrack.

The config's detector and input_size pick the model file (MODELS); the
detector's family picks the adapter, with its own pre- and post-processing.

pipeline_metadata names everything the tracks depend on, for the track cache;
it reads package metadata only, so it works without the runtime group.
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
from space_usage_heatmaps.perception.infrastructure.opencv_file_source import OpenCvFileSource

TRACKERS = ("bytetrack",)
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


def check(detector, tracker, input_size):
    """The model file's WEIGHTS key."""
    if tracker not in TRACKERS:
        raise ValueError(f"engine {ENGINE} runs trackers {list(TRACKERS)}, not {tracker!r}")
    if (detector, input_size) not in MODELS:
        known = ", ".join(f"{d} at {s}" for d, s in MODELS)
        raise ValueError(f"engine {ENGINE} runs {known}; not {detector} at {input_size}")
    return MODELS[detector, input_size]


def pipeline_metadata(detector, tracker, input_size, stride):
    model = check(detector, tracker, input_size)
    return {
        "detector": detector,
        "weights": WEIGHTS[model].file,
        "weights_sha256": WEIGHTS[model].sha256,
        "engine": ENGINE,
        "onnxruntime": _version("onnxruntime-gpu"),
        "opencv": _version("opencv-python-headless"),  # decoding and the resize
        **FAMILIES[detector].recipe(),
        "input_size": input_size,
        "tracker": "trackers.ByteTrackTracker",
        "trackers": _version("trackers"),
        "tracker_params": BYTETRACK_DEFAULTS | {"frame_rate": "video fps / stride"},
        "stride": stride,
    }


def load_detector(path, detector):
    return FAMILIES[detector].load(path)


def build_pipeline(video, detector, tracker, input_size, stride, models_dir, loop=False):
    """(frame source, person detector, tracker) for one video; a fresh tracker each time."""
    weights = check(detector, tracker, input_size)
    source = OpenCvFileSource(video, stride, loop)
    model = load_detector(ensure_weights(models_dir, weights), detector)
    return source, model, ByteTrackTracker(frame_rate=source.fps / stride)
