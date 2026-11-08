"""Use case record_cameras: one shared detector over the frames of many cameras.

Each camera keeps its own mask, tracker, calibration and track heights; the
feed decides the turn (round robin) and reports drops as (camera, None).
"""

from dataclasses import replace

import numpy as np
import pytest

from space_usage_heatmaps.evaluation.domain.dataset_reader import CameraCalibration
from space_usage_heatmaps.occupancy.application.record_cameras import (
    CameraPipe,
    record_cameras,
)
from space_usage_heatmaps.perception.domain.frame import Detections, Frame
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


def pipe(tracker, seconds=0.5):
    return CameraPipe(PrivacyMask(), tracker, CAM, lambda f: T0 + f.timestamp, seconds)


def frame(i):
    return Frame(i, i / 10, np.zeros((4, 4, 3), np.uint8))


def test_one_detector_serves_every_camera_and_each_keeps_its_tracker():
    detector, log = Counting(), Log()
    trackers = {"a": IdPerBox(), "b": IdPerBox()}
    feed = [("a", frame(0)), ("b", frame(0)), ("a", frame(1)), ("b", frame(1)), ("a", frame(2))]

    n = record_cameras(feed, detector, {c: pipe(t) for c, t in trackers.items()}, CONFIG, log)

    assert n == 5 and detector.calls == 5
    assert len(trackers["a"].seen) == 3 and len(trackers["b"].seen) == 2
    assert sorted(a[0] for a in log.added) == ["a", "a", "a", "b", "b"]
    assert log.added[0] == ("a", T0, 7, 0.5) and log.closed


def test_a_drop_marks_that_camera_incomplete():
    log = Log()
    feed = [("a", frame(0)), ("b", None), ("b", frame(0))]

    record_cameras(feed, Crowd(), {"a": pipe(IdPerBox()), "b": pipe(IdPerBox())}, CONFIG, log)

    assert log.dropped == ["b"] and len(log.added) == 2


def test_until_stops_the_frames_and_still_closes():
    log, seen = Log(), []
    feed = (("a", frame(i)) for i in range(100))

    n = record_cameras(
        feed, Crowd(), {"a": pipe(IdPerBox())}, CONFIG, log,
        until=lambda: len(seen) >= 3, on_frame=lambda c, f: seen.append(c),
    )  # fmt: skip

    assert n == 3 and len(log.added) == 3 and log.closed  # the last frame is flushed


def test_on_frame_sees_each_processed_frame_and_each_drop():
    seen = []
    feed = [("a", frame(0)), ("b", None), ("b", frame(3))]

    record_cameras(
        feed, Crowd(), {"a": pipe(IdPerBox()), "b": pipe(IdPerBox())}, CONFIG, Log(),
        on_frame=lambda c, f: seen.append((c, None if f is None else f.index)),
    )  # fmt: skip

    assert seen == [("a", 0), ("b", None), ("b", 3)]


class Walker:
    """One person whose box moves 10 px right per processed frame."""

    def detect(self, frame):
        x = 100.0 + 10 * frame.index
        return Detections(np.array([[x, 100.0, x + 10, 140.0]]), np.array([0.9]))


class Recorded(Log):
    def __init__(self):
        super().__init__()
        self.points = []

    def add(self, camera_id, t, ids, xs, ys, seconds):
        super().add(camera_id, t, ids, xs, ys, seconds)
        self.points.append((t, float(xs[0]) if len(xs) else None))


def test_with_a_stride_the_skipped_frames_are_filled_as_the_evaluation_does():
    """Each processed box stands for stride frames, moving linearly to the next
    processed box: the same samples eval's fill_stride scores, so dwell and the
    cells a track passes through match the evaluated recipe."""
    config = replace(CONFIG, stride=2)
    log = Recorded()
    image = np.zeros((4, 4, 3), np.uint8)
    feed = [("a", Frame(i, i / 10, image)) for i in (0, 2, 4)]  # every 2nd source frame

    record_cameras(feed, Walker(), {"a": pipe(IdPerBox(), seconds=0.2)}, config, log)

    times = [round(t - T0, 3) for t, _ in log.points]
    assert times == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 6 source frames of 0.1 s
    assert all(a[3] == pytest.approx(0.1) for a in log.added)
    xs = [x for _, x in log.points]
    assert xs[1] == pytest.approx((xs[0] + xs[2]) / 2, abs=0.05)  # halfway on the floor
    assert xs[5] == xs[4]  # after the last processed frame the box stays
