"""TrackSource over the perception pipeline (perception.application.track_video),
in two halves: the detector's detections of a video, which a detection cache
can stand in for, and a fresh tracker over them."""

import sys
import time

from space_usage_heatmaps.evaluation.domain.tracks import Tracks
from space_usage_heatmaps.perception.application.track_video import (
    collect_tracks,
    detect_video,
    track_detections,
)


def _progress(detections, label, every=1000):
    start = time.monotonic()
    for n, d in enumerate(detections, 1):
        if n % every == 0:
            fps = n / (time.monotonic() - start)
            print(f"{label}: {n} frames, {fps:.1f} fps", file=sys.stderr)
        yield d


class PipelineDetections:
    def __init__(self, detection_of):
        """detection_of(scene, camera) -> (frame source, person detector)."""
        self.detection_of = detection_of
        self._detector = None

    def detections(self, scene, camera):
        source, self._detector = self.detection_of(scene, camera)
        return _progress(detect_video(source, self._detector), f"scene {scene} camera {camera}")

    @property
    def last_run(self):
        """The providers the last detector ran on, in order (CUDA, then CPU
        after running out of GPU memory), once its video is consumed."""
        providers = getattr(self._detector, "providers", None)
        return {"providers": list(providers)} if providers else None


class PipelineTrackSource:
    def __init__(self, detections, tracker_of):
        """detections.detections(scene, camera) -> (frame index, Detections) of
        every processed frame, and detections.last_run; tracker_of(scene,
        camera) -> a fresh tracker. last_run: the detections' last_run, for
        the track cache."""
        self.detections = detections
        self.tracker_of = tracker_of
        self.last_run = None

    def tracks(self, scene, camera):
        tracker = self.tracker_of(scene, camera)
        tracked = track_detections(self.detections.detections(scene, camera), tracker)
        ids, frames, boxes = collect_tracks(tracked)
        self.last_run = self.detections.last_run
        return Tracks(ids, frames, boxes)
