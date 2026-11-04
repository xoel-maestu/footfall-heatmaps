"""The settings of the Ultralytics reference recipe and the metadata cached
tracks are checked against. Reads package metadata only: importing this
module never imports ultralytics, so it works without the baseline group."""

import hashlib
from importlib import metadata

WEIGHTS = {"yolov8n": "yolov8n.pt"}  # official COCO weights, ultralytics/assets release
TRACKERS = {"bytetrack": "bytetrack.yaml"}  # built into the ultralytics package
PERSON = 0  # COCO class id
CLASSES = [PERSON]
CONF = 0.1  # what model.track() uses when no conf is given; passed explicitly


def _installed_tracker(name):
    """(ultralytics version, sha256 of its built-in tracker yaml), or Nones when
    ultralytics is not installed."""
    try:
        dist = metadata.distribution("ultralytics")
    except metadata.PackageNotFoundError:
        return None, None
    path = dist.locate_file(f"ultralytics/cfg/trackers/{name}")
    return dist.version, hashlib.sha256(path.read_bytes()).hexdigest()


def recipe_metadata(config):
    """Everything a config's tracks depend on, except the foot point (applied
    when scoring)."""
    version, sha = _installed_tracker(TRACKERS.get(config.tracker, config.tracker))
    return {
        "ultralytics": version,
        "tracker_yaml_sha256": sha,
        "classes": CLASSES,
        "conf": CONF,
        "imgsz": config.input_size,
        "weights": WEIGHTS.get(config.detector, config.detector),
        "stride": config.stride,
    }
