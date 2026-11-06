"""Port: what the evaluation needs from a dataset with floor ground truth."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    # 3x3 ground-plane homography, metres -> pixels, scaled so that the third
    # coordinate is positive for floor points in front of the camera; None for
    # a dataset without floor calibration (image-space tracking only).
    world_to_image: np.ndarray | None
    width: int
    height: int
    fps: float
    # 3x4 camera matrix K [R | t], world metres with z up, scaled so that its
    # columns 0, 1 and 3 are world_to_image; None without a 3D calibration.
    projection: np.ndarray | None = None


@dataclass(frozen=True)
class Samples:
    """One row per (track, frame) the camera's ground truth lists."""

    track_ids: np.ndarray  # (N,)
    frames: np.ndarray  # (N,)
    boxes: np.ndarray  # (N, 4) x, y, w, h in pixels
    world: np.ndarray  # (N, 2) floor position in metres; NaN without floor truth
    # (N,) bool: a region the tracking guards neither score nor penalise (MOT17's
    # static people, reflections...); a tracked box matched to it is dropped.
    distractor: np.ndarray | None = None


@dataclass(frozen=True)
class FloorTruth:
    """Every person of the scene in every frame, whether this camera sees them or not."""

    track_ids: np.ndarray  # (N,)
    frames: np.ndarray  # (N,)
    world: np.ndarray  # (N, 2) floor position in metres
    visible: np.ndarray  # (N,) bool: this camera has a visible box of them


class DatasetReader(Protocol):
    """A dataset without floor truth (image-space tracking only) has no
    floor_truth or scene_extent: both raise ValueError."""

    name: str

    def cameras(self, scene: str) -> list[str]:
        """Ids of the scene's calibrated cameras."""
        ...

    def camera(self, scene: str, camera: str) -> CameraCalibration: ...

    def samples(self, scene: str, camera: str) -> Samples: ...

    def floor_truth(self, scene: str, camera: str) -> FloorTruth: ...

    def scene_extent(self, scene: str) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) in metres over every camera of the scene."""
        ...

    def video(self, scene: str, camera: str) -> Path:
        """The camera's video file, which the perception pipeline reads."""
        ...
