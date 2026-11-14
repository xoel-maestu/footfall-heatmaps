"""Privacy rule R6: per-camera privacy masks drop detections before the tracker.

A detection is dropped when the bottom-centre of its box (where the feet
touch the floor) lies in an image polygon, or its floor point does in a floor
polygon. The tracker never receives it, so it never gets an id.
"""

import numpy as np
import pytest

from footfall_heatmaps.evaluation.domain.dataset_reader import CameraCalibration
from footfall_heatmaps.evaluation.domain.run_config import RunConfig
from footfall_heatmaps.occupancy.application.record_camera import record_camera
from footfall_heatmaps.occupancy.application.record_slots import RecordSlots
from footfall_heatmaps.occupancy.domain.grid import GridSpec
from footfall_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore
from footfall_heatmaps.perception.domain.frame import Detections
from footfall_heatmaps.perception.domain.privacy_mask import PrivacyMask
from tests.perception.fakes import EchoTracker, ListFrameSource

# Floor (x, y) metres -> image (u, v): u = 100 x, v = 100 y.
SCALE = np.diag([100.0, 100.0, 1.0])
SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]


def boxes(*feet):
    """Boxes 20 px wide and 60 px tall standing on each (u, v)."""
    return np.array([[u - 10, v - 60, u + 10, v] for u, v in feet], float)


def test_an_empty_mask_keeps_everything():
    assert PrivacyMask().keep(boxes((50, 50), (500, 500))).tolist() == [True, True]


def test_image_polygons_drop_boxes_standing_inside():
    mask = PrivacyMask(image_polygons=[SQUARE])

    keep = mask.keep(boxes((50, 50), (150, 50), (50, 99)))

    assert keep.tolist() == [False, True, False]


def test_only_the_feet_count_not_the_rest_of_the_box():
    mask = PrivacyMask(image_polygons=[SQUARE])

    assert mask.keep(boxes((50, 130))).tolist() == [True]  # head inside, feet below


def test_floor_polygons_go_through_the_homography():
    mask = PrivacyMask(floor_polygons=[[[2, 2], [3, 2], [3, 3], [2, 3]]], world_to_image=SCALE)

    keep = mask.keep(boxes((250, 250), (150, 250)))

    assert keep.tolist() == [False, True]


def test_any_of_several_polygons_drops():
    triangle = [[200, 0], [300, 0], [250, 100]]
    mask = PrivacyMask(image_polygons=[SQUARE, triangle])

    assert mask.keep(boxes((50, 50), (250, 20), (400, 50))).tolist() == [False, False, True]


def test_a_floor_mask_needs_the_homography():
    with pytest.raises(ValueError, match="homography"):
        PrivacyMask(floor_polygons=[[[0, 0], [1, 0], [1, 1]]])


def test_a_polygon_needs_three_points():
    with pytest.raises(ValueError, match="three"):
        PrivacyMask(image_polygons=[[[0, 0], [1, 1]]])


class TwoPeople:
    def detect(self, frame):
        return Detections(boxes((50, 50), (150, 50)), np.array([0.9, 0.8]))


def test_masked_detections_never_reach_the_tracker():
    """Through footfall run's use case, the one path that applies the masks."""
    tracker = EchoTracker()
    frames = ListFrameSource([np.zeros((200, 200, 3), np.uint8)] * 3).frames()
    mask = PrivacyMask(image_polygons=[SQUARE])
    config = RunConfig.from_mapping(
        {"detector": "yolox-s", "tracker": "bytetrack", "input_size": 640,
         "engine": "onnx-cuda", "stride": 1, "foot_point": "bottom"}
    )  # fmt: skip
    cam = CameraCalibration(SCALE, width=200, height=200, fps=10.0)
    recorder = RecordSlots(SqliteSlotStore(":memory:"), "cam", GridSpec(0.0, 0.0, 1.0, 2, 2))

    record_camera(frames, TwoPeople(), mask, tracker, cam, config, recorder, lambda f: 0.0, 0.1)

    assert len(tracker.calls) == 3
    for _, sent, scores in tracker.calls:
        np.testing.assert_array_equal(sent, boxes((150, 50)))
        np.testing.assert_array_equal(scores, [0.8])
