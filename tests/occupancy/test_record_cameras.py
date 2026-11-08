"""Use case record_cameras: one shared detector over the frames of many cameras.

Each camera keeps its own mask, tracker, calibration and track heights; the
feed decides the turn (round robin) and reports drops as (camera, None).
"""

import numpy as np

from space_usage_heatmaps.evaluation.domain.dataset_reader import CameraCalibration
from space_usage_heatmaps.occupancy.application.record_cameras import (
    CameraPipe,
    record_cameras,
)
from space_usage_heatmaps.perception.domain.frame import Frame
from space_usage_heatmaps.perception.domain.privacy_mask import PrivacyMask
from tests.conftest import H
from tests.occupancy.test_record_camera import CONFIG, Crowd, IdPerBox

T0 = 1_793_955_600
CAM = CameraCalibration(np.array(H), width=1000, height=1000, fps=10.0)


class Counting(Crowd):
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return super().detect(frame)


class Log:
    def __init__(self):
        self.added, self.dropped, self.closed = [], [], False

    def add(self, camera_id, t, ids, xs, ys, seconds):
        self.added.append((camera_id, t, len(ids), seconds))

    def mark_incomplete(self, camera_id):
        self.dropped.append(camera_id)

    def close(self):
        self.closed = True


def pipe(tracker):
    return CameraPipe(PrivacyMask(), tracker, CAM, lambda f: T0 + f.timestamp, 0.5)


def frame(i):
    return Frame(i, i / 10, np.zeros((4, 4, 3), np.uint8))


def test_one_detector_serves_every_camera_and_each_keeps_its_tracker():
    detector, log = Counting(), Log()
    trackers = {"a": IdPerBox(), "b": IdPerBox()}
    feed = [("a", frame(0)), ("b", frame(0)), ("a", frame(1)), ("b", frame(1)), ("a", frame(2))]

    n = record_cameras(feed, detector, {c: pipe(t) for c, t in trackers.items()}, CONFIG, log)

    assert n == 5 and detector.calls == 5
    assert len(trackers["a"].seen) == 3 and len(trackers["b"].seen) == 2
    assert [a[0] for a in log.added] == ["a", "b", "a", "b", "a"]
    assert log.added[0] == ("a", T0, 7, 0.5) and log.closed


def test_a_drop_marks_that_camera_incomplete():
    log = Log()
    feed = [("a", frame(0)), ("b", None), ("b", frame(0))]

    record_cameras(feed, Crowd(), {"a": pipe(IdPerBox()), "b": pipe(IdPerBox())}, CONFIG, log)

    assert log.dropped == ["b"] and len(log.added) == 2


def test_until_stops_the_frames_and_still_closes():
    log = Log()
    feed = (("a", frame(i)) for i in range(100))

    n = record_cameras(
        feed, Crowd(), {"a": pipe(IdPerBox())}, CONFIG, log, until=lambda: len(log.added) >= 3
    )

    assert n == 3 and log.closed


def test_on_frame_sees_each_processed_camera():
    seen = []
    feed = [("a", frame(0)), ("b", frame(0))]

    record_cameras(
        feed, Crowd(), {"a": pipe(IdPerBox()), "b": pipe(IdPerBox())}, CONFIG, Log(),
        on_frame=seen.append,
    )  # fmt: skip

    assert seen == ["a", "b"]
