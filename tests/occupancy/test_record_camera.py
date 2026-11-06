"""Use case record_camera: frames -> detector -> mask -> tracker -> floor -> slots."""

import numpy as np
import pytest

from space_usage_heatmaps.evaluation.domain.dataset_reader import CameraCalibration
from space_usage_heatmaps.evaluation.domain.run_config import RunConfig
from space_usage_heatmaps.occupancy.application.record_camera import record_camera
from space_usage_heatmaps.occupancy.application.record_slots import RecordSlots
from space_usage_heatmaps.occupancy.domain.grid import GridSpec
from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore
from space_usage_heatmaps.perception.domain.frame import Detections
from space_usage_heatmaps.perception.domain.privacy_mask import PrivacyMask
from tests.conftest import H
from tests.perception.fakes import ListFrameSource

T0 = 1_793_955_600
CAM = CameraCalibration(np.array(H), width=1000, height=1000, fps=10.0)
CONFIG = RunConfig.from_mapping(
    {"detector": "yolox-s", "tracker": "bytetrack", "input_size": 640,
     "engine": "onnx-cuda", "stride": 1, "foot_point": "bottom"}
)  # fmt: skip


def feet(x, y):
    """Image point of floor (x, y) under conftest.H."""
    return (100 * x + 50 * y) / y, (50 * y + 100) / y


class Crowd:
    """Six people standing at floor (1.2, 2.2), one at (3.2, 2.2), every frame."""

    def detect(self, frame):
        boxes = []
        for x in [1.2] * 6 + [3.2]:
            u, v = feet(x, 2.2)
            boxes.append([u - 5, v - 20, u + 5, v])
        return Detections(np.array(boxes), np.full(len(boxes), 0.9))


class IdPerBox:
    """Track id = position of the box in the frame: as many ids as people."""

    def __init__(self):
        self.seen = []

    def update(self, boxes, scores, frame_index):
        self.seen.append(len(boxes))
        return np.arange(len(boxes)), boxes


def run(mask, n=150):
    store = SqliteSlotStore(":memory:")
    spec = GridSpec(0.0, 0.0, 1.0, 4, 4)
    recorder = RecordSlots(store, "645", spec, now=lambda: T0)
    tracker = IdPerBox()
    frames = ListFrameSource([np.zeros((8, 8, 3), np.uint8)] * n, fps=10.0).frames()
    count = record_camera(
        frames, Crowd(), mask, tracker, CAM, CONFIG, recorder,
        time_of=lambda f: T0 + 890 + f.timestamp, seconds=0.1,
    )  # fmt: skip
    return store, tracker, count


def test_frames_become_two_slots_across_a_quarter():
    store, _, count = run(PrivacyMask())

    rows = store.read("645", T0, T0 + 1800)
    assert count == 150
    assert [(s.start_utc - T0, s.layer, s.incomplete) for s in rows] == [
        (0, "dwell", True), (0, "passage", True), (900, "dwell", True), (900, "passage", True),
    ]  # fmt: skip
    first = {s.layer: s.grid for s in rows if s.start_utc == T0}
    assert first["passage"][2, 1] == 6  # six people in cell x 1..2, y 2..3
    assert first["passage"][2, 3] == 0  # one person: suppressed
    assert first["dwell"][2, 1] == pytest.approx(60.0)  # 6 people x 100 frames x 0.1 s
    assert rows[0].n_unique == 7


def test_the_mask_applies_before_the_tracker():
    u, v = feet(3.2, 2.2)
    square = [[u - 5, v - 5], [u + 5, v - 5], [u + 5, v + 5], [u - 5, v + 5]]

    _, tracker, _ = run(PrivacyMask(image_polygons=[square]), n=3)

    assert tracker.seen == [6, 6, 6]
