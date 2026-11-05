"""R2 (spec 3): track ids are ephemeral, run-local counters.

Ids count up from 0 within a run, a new run starts again from 0, and they
depend on where the boxes are, never on what the image shows.
"""

import numpy as np

from space_usage_heatmaps.perception.application.track_video import collect_tracks, track_video
from space_usage_heatmaps.perception.domain.frame import Detections
from space_usage_heatmaps.perception.infrastructure import container
from space_usage_heatmaps.perception.infrastructure.bytetrack_tracker import ByteTrackTracker
from space_usage_heatmaps.perception.infrastructure.opencv_file_source import OpenCvFileSource
from tests.perception.synthetic_video import write_video


class TwoPeople:
    """Two people walking apart, then a third one appearing, whatever the image shows."""

    def detect(self, frame):
        i = frame.index
        boxes = [[10 + 2 * i, 20, 30 + 2 * i, 80], [120 - 2 * i, 20, 140 - 2 * i, 80]]
        if i >= 6:
            boxes.append([60, 30, 80, 90])
        return Detections(np.array(boxes, float), np.full(len(boxes), 0.9))


def run(video):
    source = OpenCvFileSource(video)
    return collect_tracks(track_video(source, TwoPeople(), ByteTrackTracker(source.fps)))


def test_ids_are_a_counter_from_zero_within_a_run(tmp_path):
    ids, _, _ = run(write_video(tmp_path / "v.avi", n=12))

    assert sorted(set(ids.tolist())) == [0, 1, 2]


def test_a_new_run_starts_the_ids_again(tmp_path):
    video = write_video(tmp_path / "v.avi", n=12)

    first, second = run(video), run(video)

    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_ids_do_not_depend_on_the_image(tmp_path):
    a = run(write_video(tmp_path / "a.avi", n=12, seed=1))
    b = run(write_video(tmp_path / "b.avi", n=12, seed=2))

    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_each_pipeline_gets_a_fresh_tracker(tmp_path, monkeypatch):
    video = write_video(tmp_path / "v.avi", n=4)
    monkeypatch.setattr(container, "ensure_weights", lambda d, name: d / name)
    monkeypatch.setattr(container, "load_detector", lambda path, *_: TwoPeople())

    *_, first = container.build_pipeline(video, "yolox-s", "bytetrack", 640, 1, tmp_path)
    *_, second = container.build_pipeline(video, "yolox-s", "bytetrack", 640, 1, tmp_path)

    assert first is not second
