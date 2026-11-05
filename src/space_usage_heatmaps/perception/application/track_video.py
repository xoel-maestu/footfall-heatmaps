"""Use case: run a video through the person detector and the tracker.

Each frame lives only inside this loop: the detector reads its image, the
tracker gets boxes and scores, and what comes out is positions per frame.
"""

from collections.abc import Iterable, Iterator

import numpy as np

from space_usage_heatmaps.perception.domain.frame import TrackedBoxes
from space_usage_heatmaps.perception.domain.ports import FrameSource, PersonDetector, Tracker


def track_video(
    source: FrameSource, detector: PersonDetector, tracker: Tracker
) -> Iterator[TrackedBoxes]:
    for frame in source.frames():
        d = detector.detect(frame)
        ids, boxes = tracker.update(d.boxes, d.scores, frame.index)
        yield TrackedBoxes(frame.index, np.asarray(ids, dtype=int), np.asarray(boxes, float))


def collect_tracks(tracked: Iterable[TrackedBoxes]):
    """(track ids, frames, boxes): one row per tracked box of every frame."""
    ids, frames, boxes = [np.empty(0, int)], [np.empty(0, int)], [np.empty((0, 4))]
    for t in tracked:
        ids.append(t.track_ids)
        frames.append(np.full(len(t.track_ids), t.frame_index))
        boxes.append(t.boxes.reshape(-1, 4))
    return np.concatenate(ids), np.concatenate(frames), np.concatenate(boxes)
