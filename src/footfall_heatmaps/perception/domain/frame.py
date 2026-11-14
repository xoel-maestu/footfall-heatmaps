"""What perception passes between its ports: a frame in memory, person boxes, tracks.

A Frame's image never leaves perception and is never written anywhere (privacy
rule R1); Detections and TrackedBoxes carry positions and scores only (R3).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    index: int  # 0-based frame of the source video
    timestamp: float  # seconds from the start of the source
    image: np.ndarray  # (H, W, 3) uint8 BGR, in memory only


@dataclass(frozen=True)
class Detections:
    boxes: np.ndarray  # (N, 4) x1, y1, x2, y2 in source pixels
    scores: np.ndarray  # (N,) person confidence, 0 to 1


@dataclass(frozen=True)
class TrackedBoxes:
    """The confirmed tracks of one frame."""

    frame_index: int
    track_ids: np.ndarray  # (M,) int, run-local counters
    boxes: np.ndarray  # (M, 4) x1, y1, x2, y2 in source pixels
