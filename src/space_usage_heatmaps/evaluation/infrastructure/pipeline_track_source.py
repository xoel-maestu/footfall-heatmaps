"""TrackSource over the perception pipeline (perception.application.track_video)."""

import sys
import time

from space_usage_heatmaps.evaluation.domain.tracks import Tracks
from space_usage_heatmaps.perception.application.track_video import collect_tracks, track_video


def _progress(tracked, label, every=1000):
    start = time.monotonic()
    for n, t in enumerate(tracked, 1):
        if n % every == 0:
            fps = n / (time.monotonic() - start)
            print(f"{label}: {n} frames, {fps:.1f} fps", file=sys.stderr)
        yield t


class PipelineTrackSource:
    def __init__(self, pipeline_of):
        """pipeline_of(scene, camera) -> (frame source, person detector, tracker).
        last_run: the providers the detector of the last video ran on, in order
        (CUDA, then CPU after running out of GPU memory), for the track cache."""
        self.pipeline_of = pipeline_of
        self.last_run = None

    def tracks(self, scene, camera):
        source, detector, tracker = self.pipeline_of(scene, camera)
        tracked = track_video(source, detector, tracker)
        ids, frames, boxes = collect_tracks(_progress(tracked, f"scene {scene} camera {camera}"))
        providers = getattr(detector, "providers", None)
        self.last_run = {"providers": list(providers)} if providers else None
        return Tracks(ids, frames, boxes)
