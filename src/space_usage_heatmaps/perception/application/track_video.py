"""Use case: run a video through the person detector and the tracker.

Each frame lives only inside detect_video: the detector reads its image, and
what comes out is boxes and scores per frame. track_detections hands those to
the tracker, so detections can also be replayed from a cache (positions only).

With a stride N, only one frame in N goes through the detector and the tracker;
fill_stride gives the skipped frames their boxes back, so every source frame
has its tracks and dwell stays time-correct.
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


def select_detections(
    detections: Iterable[tuple[int, Detections]], stride: int, min_score: float
) -> Iterator[tuple[int, Detections]]:
    """Detections of every frame (a cache), as a detector with a higher score
    threshold would give them on every stride-th frame. Exact for detectors whose
    NMS keeps a box only because of higher-scored boxes (greedy NMS): those boxes
    pass any threshold the kept box passes."""
    for index, d in detections:
        if index % stride:
            continue
        keep = d.scores > min_score  # strict, as the detectors' own threshold
        yield index, Detections(d.boxes.reshape(-1, 4)[keep], d.scores[keep])


def fill_stride(
    tracked: Iterable[TrackedBoxes], stride: int, end: int | None = None
) -> Iterator[TrackedBoxes]:
    """Tracks of every source frame from those of every stride-th frame: a box
    seen on processed frames i and i + stride moves linearly between them; one
    not seen on i + stride stays where it was. Each processed box stands for
    stride frames, so dwell counted per frame matches stride / fps per processed
    box. Holds one processed frame back (stride frames of delay). end: the
    video's frame count, if known; no frame at or past it is filled."""
    previous = None
    for current in tracked:
        if previous is not None:
            yield from _fill(previous, current, stride)
        previous = current
    if previous is not None:
        yield from _fill(previous, None, stride, end)


def _fill(
    a: TrackedBoxes, b: TrackedBoxes | None, stride: int, end: int | None = None
) -> Iterator[TrackedBoxes]:
    yield a
    gap = stride if b is None else min(stride, b.frame_index - a.frame_index)
    if end is not None:
        gap = max(1, min(gap, end - a.frame_index))
    following = {}
    if b is not None and b.frame_index - a.frame_index == stride:
        following = dict(zip(b.track_ids.tolist(), b.boxes.reshape(-1, 4), strict=True))
    boxes = a.boxes.reshape(-1, 4)
    ends = np.array(
        [following.get(t, box) for t, box in zip(a.track_ids.tolist(), boxes, strict=True)]
    )
    ends = ends.reshape(-1, 4)
    for k in range(1, gap):
        yield TrackedBoxes(a.frame_index + k, a.track_ids, boxes + (ends - boxes) * (k / stride))


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
