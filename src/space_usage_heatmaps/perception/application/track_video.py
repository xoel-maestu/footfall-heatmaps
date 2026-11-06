"""Use case: run a video through the person detector and the tracker.

Each frame lives only inside detect_video: the detector reads its image, and
what comes out is boxes and scores per frame. track_detections hands those to
the tracker, so detections can also be replayed from a cache (positions only).
"""

from collections.abc import Iterable, Iterator

import numpy as np

from space_usage_heatmaps.perception.domain.frame import Detections, TrackedBoxes
from space_usage_heatmaps.perception.domain.ports import FrameSource, PersonDetector, Tracker


def detect_video(source: FrameSource, detector: PersonDetector) -> Iterator[tuple[int, Detections]]:
    for frame in source.frames():
        yield frame.index, detector.detect(frame)


def track_detections(
    detections: Iterable[tuple[int, Detections]], tracker: Tracker
) -> Iterator[TrackedBoxes]:
    """(frame index, detections) of every processed frame, empty ones included:
    a tracker ages its tracks on each call."""
    for index, d in detections:
        ids, boxes = tracker.update(d.boxes, d.scores, index)
        yield TrackedBoxes(index, np.asarray(ids, dtype=int), np.asarray(boxes, float))


def track_video(
    source: FrameSource, detector: PersonDetector, tracker: Tracker
) -> Iterator[TrackedBoxes]:
    return track_detections(detect_video(source, detector), tracker)


def collect_tracks(tracked: Iterable[TrackedBoxes]):
    """(track ids, frames, boxes): one row per tracked box of every frame."""
    ids, frames, boxes = [np.empty(0, int)], [np.empty(0, int)], [np.empty((0, 4))]
    for t in tracked:
        ids.append(t.track_ids)
        frames.append(np.full(len(t.track_ids), t.frame_index))
        boxes.append(t.boxes.reshape(-1, 4))
    return np.concatenate(ids), np.concatenate(frames), np.concatenate(boxes)
