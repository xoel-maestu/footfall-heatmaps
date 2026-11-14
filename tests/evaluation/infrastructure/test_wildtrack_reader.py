"""WILDTRACK reader on a tiny synthetic copy of its layout."""

import json

import cv2
import numpy as np
import pytest

from footfall_heatmaps.evaluation.infrastructure.wildtrack_reader import (
    SCENE,
    WildtrackReader,
    position_to_world,
)

# Camera C5 (IDIAP1) 2 m above (0, -10) looking along +y: camera x = world x,
# camera y = -world z, camera z = world y. Calibration in centimetres.
R = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
T_CM = -R @ np.array([0.0, -1000.0, 200.0])
K = np.array([[1000.0, 0, 960], [0, 1000, 540], [0, 0, 1]])


def position(x, y):
    """The grid position id of floor point (x, y) in metres."""
    return round((y + 9.0) / 0.025) * 480 + round((x + 3.0) / 0.025)


def write_dataset(root, people_per_frame):
    d = root / "wildtrack" / "Wildtrack_dataset"
    (d / "calibrations" / "extrinsic").mkdir(parents=True)
    (d / "calibrations" / "intrinsic_zero").mkdir(parents=True)
    rvec = cv2.Rodrigues(R)[0].ravel()
    (d / "calibrations" / "extrinsic" / "extr_IDIAP1.xml").write_text(
        '<?xml version="1.0"?>\n<opencv_storage>\n  <rvec>\n    '
        + " ".join(map(repr, rvec.tolist()))
        + "\n   </rvec> \n  <tvec>\n    "
        + " ".join(map(repr, T_CM.tolist()))
        + "\n  </tvec>\n</opencv_storage>\n"
    )
    fs = cv2.FileStorage(
        str(d / "calibrations" / "intrinsic_zero" / "intr_IDIAP1.xml"), cv2.FILE_STORAGE_WRITE
    )
    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", np.zeros((5, 1)))
    fs.release()
    (d / "annotations_positions").mkdir()
    (d / "Image_subsets" / "C5").mkdir(parents=True)
    for i, people in enumerate(people_per_frame):
        name = f"{5 * i:08d}"
        rows = []
        for pid, (x, y), box in people:
            views = [
                {"viewNum": v, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1} for v in range(7)
            ]
            if box is not None:
                views[4] = dict(zip(("xmin", "ymin", "xmax", "ymax"), box, strict=True))
                views[4]["viewNum"] = 4
            rows.append({"personID": pid, "positionID": position(x, y), "views": views})
        (d / "annotations_positions" / f"{name}.json").write_text(json.dumps(rows))
        cv2.imwrite(
            str(d / "Image_subsets" / "C5" / f"{name}.png"), np.zeros((108, 192, 3), np.uint8)
        )
    return root


FRAMES = [
    [(7, (1.0, 0.0), (1040, 300, 1080, 740)), (8, (2.0, 5.0), None)],
    [(7, (1.0, 0.5), (1035, 300, 1075, 730))],  # bottom-centre on (1, 0.5) to 0.5 px
]


def test_position_ids_are_the_floor_grid_in_metres():
    assert position_to_world(np.array([0, 481])) == pytest.approx(
        np.array([[-3.0, -9.0], [-2.975, -8.975]])
    )


def test_calibration_projects_floor_metres_to_pixels(tmp_path):
    cam = WildtrackReader(write_dataset(tmp_path, FRAMES)).camera(SCENE, "C5")

    p = cam.world_to_image @ [1.0, 0.0, 1.0]
    assert p[:2] / p[2] == pytest.approx([1060.0, 740.0])
    assert p[2] > 0
    head = cam.projection @ [1.0, 0.0, 1.7, 1.0]
    assert head[1] / head[2] == pytest.approx(540 + 1000 * 0.3 / 10)
    assert (cam.width, cam.height, cam.fps) == (1920, 1080, 2.0)


def test_samples_are_the_people_this_camera_boxes(tmp_path):
    s = WildtrackReader(write_dataset(tmp_path, FRAMES)).samples(SCENE, "C5")

    assert s.track_ids.tolist() == [7, 7]
    assert s.frames.tolist() == [0, 1]
    assert s.boxes[0] == pytest.approx([1040, 300, 40, 440])
    assert s.world == pytest.approx(np.array([[1.0, 0.0], [1.0, 0.5]]))


def test_floor_truth_lists_everyone_and_who_the_camera_sees(tmp_path):
    t = WildtrackReader(write_dataset(tmp_path, FRAMES)).floor_truth(SCENE, "C5")

    assert sorted(
        zip(t.track_ids.tolist(), t.frames.tolist(), t.visible.tolist(), strict=True)
    ) == [
        (7, 0, True),
        (7, 1, True),
        (8, 0, False),
    ]


def test_cameras_are_the_calibrated_ones_and_extent_is_the_grid(tmp_path):
    reader = WildtrackReader(write_dataset(tmp_path, FRAMES))

    assert reader.cameras(SCENE) == ["C5"]
    assert reader.scene_extent(SCENE) == pytest.approx((-3.0, -9.0, 9.0, 27.0))


def test_video_holds_the_annotated_frames_at_two_per_second(tmp_path):
    video = WildtrackReader(write_dataset(tmp_path, FRAMES)).video(SCENE, "C5")

    cap = cv2.VideoCapture(str(video))
    assert cap.get(cv2.CAP_PROP_FPS) == 2.0
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 2


def test_depression_is_the_optical_axis_angle_below_the_horizon(tmp_path):
    reader = WildtrackReader(write_dataset(tmp_path, FRAMES))

    assert reader.depression_deg(SCENE, "C5") == pytest.approx(0.0)
    assert reader.most_oblique(SCENE) == "C5"
