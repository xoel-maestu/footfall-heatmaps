"""Port: the tracked person boxes of one camera's video, positions only."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Tracks:
    """One row per tracked box: no image, no appearance, only where it was."""

    track_ids: np.ndarray  # (N,) int
    frames: np.ndarray  # (N,) int, 0-based video frame
    boxes: np.ndarray  # (N, 4) x1, y1, x2, y2 in pixels


class TrackSource(Protocol):
    def tracks(self, scene: str, camera: str) -> Tracks: ...
