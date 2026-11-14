"""Ports of the perception pipeline: frame source -> person detector -> tracker."""

from collections.abc import Iterator
from typing import Protocol

import numpy as np

from footfall_heatmaps.perception.domain.frame import Detections, Frame


class FrameSource(Protocol):
    def frames(self) -> Iterator[Frame]: ...


class PersonDetector(Protocol):
    def detect(self, frame: Frame) -> Detections: ...


class Tracker(Protocol):
    def update(
        self, boxes: np.ndarray, scores: np.ndarray, frame_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Boxes and scores of one frame -> (track ids, boxes) of its confirmed
        tracks. Positions only: a tracker never sees the image."""
        ...


class SourceDropped(OSError):
    """A live source stopped or could not be reached; it may come back."""
