"""Wiring of the perception pipeline: video file -> YOLOX-s on ONNX Runtime -> ByteTrack.

pipeline_metadata names everything the tracks depend on, for the track cache;
it reads package metadata only, so it works without the runtime group.
"""

from importlib import metadata

from space_usage_heatmaps.perception.infrastructure import onnx_yolox_detector as yolox
from space_usage_heatmaps.perception.infrastructure.bytetrack_tracker import (
    DEFAULTS as BYTETRACK_DEFAULTS,
)
from space_usage_heatmaps.perception.infrastructure.bytetrack_tracker import ByteTrackTracker
from space_usage_heatmaps.perception.infrastructure.model_weights import WEIGHTS, ensure_weights
from space_usage_heatmaps.perception.infrastructure.opencv_file_source import OpenCvFileSource

TRACKERS = ("bytetrack",)
ENGINE = "onnx-cuda"


def _version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def check(detector, tracker, input_size):
    if detector not in WEIGHTS:
        raise ValueError(f"engine {ENGINE} runs detectors {sorted(WEIGHTS)}, not {detector!r}")
    if tracker not in TRACKERS:
        raise ValueError(f"engine {ENGINE} runs trackers {list(TRACKERS)}, not {tracker!r}")
    if input_size != 640:
        raise ValueError("the YOLOX release models take a 640 x 640 input only")


def pipeline_metadata(detector, tracker, input_size, stride):
    check(detector, tracker, input_size)
    return {
        "detector": detector,
        "weights": WEIGHTS[detector].file,
        "weights_sha256": WEIGHTS[detector].sha256,
        "engine": ENGINE,
        "onnxruntime": _version("onnxruntime-gpu"),
        "opencv": _version("opencv-python-headless"),  # decoding and the letterbox resize
        "preprocess": "yolox letterbox, top-left, pad 114, BGR 0-255",
        "postprocess": "boxes clipped to the image",
        "score_thr": yolox.SCORE_THR,
        "nms_thr": yolox.NMS_THR,
        "nms": "class-agnostic",
        "classes": [yolox.PERSON],
        "input_size": input_size,
        "tracker": "trackers.ByteTrackTracker",
        "trackers": _version("trackers"),
        "tracker_params": BYTETRACK_DEFAULTS | {"frame_rate": "video fps / stride"},
        "stride": stride,
    }


def load_detector(path):
    return yolox.OnnxYoloxDetector.load(path)


def build_pipeline(video, detector, tracker, input_size, stride, models_dir, loop=False):
    """(frame source, person detector, tracker) for one video; a fresh tracker each time."""
    check(detector, tracker, input_size)
    source = OpenCvFileSource(video, stride, loop)
    model = load_detector(ensure_weights(models_dir, detector))
    return source, model, ByteTrackTracker(frame_rate=source.fps / stride)
