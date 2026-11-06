import numpy as np

from space_usage_heatmaps.evaluation.infrastructure.pipeline_track_source import (
    PipelineTrackSource,
    SelectedDetections,
)
from space_usage_heatmaps.perception.domain.frame import Detections


class EveryFrame:
    """Cached detections of a 5-frame video: one box on every frame."""

    last_run = None

    def detections(self, scene, camera):
        box = np.array([[1.0, 2, 3, 4]])
        return [(i, Detections(box, np.array([0.9]))) for i in range(5)]


class SameId:
    def update(self, boxes, scores, index):
        return list(range(len(boxes))), boxes


def test_stride_tracks_end_on_the_last_frame_of_the_video():
    selected = SelectedDetections(EveryFrame(), 3, 0.5)

    tracks = PipelineTrackSource(selected, lambda s, c: SameId(), 3).tracks("071", "645")

    assert tracks.frames.max() == 4
    assert sorted(tracks.frames.tolist()) == [0, 1, 2, 3, 4]
