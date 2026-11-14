import json
import shutil

import numpy as np
import pytest

from footfall_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
from tests.conftest import BOX_W, H, write_scene


def test_reads_one_camera_samples(tmp_path, rows):
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows))

    s = reader.samples("001", "7")

    assert len(s.track_ids) == 15
    assert set(s.track_ids) == {1, 2}
    np.testing.assert_allclose(s.world[0], [0.07, 2.03])
    np.testing.assert_allclose(s.boxes[0, 2], BOX_W)


def test_scene_extent_spans_every_camera(tmp_path, rows):
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows))

    assert reader.scene_extent("001") == pytest.approx((-0.4, 2.03, 6.0, 9.0))


def test_reads_camera_calibration(tmp_path, rows):
    cam = NvidiaMtmcReader(write_scene(tmp_path, rows)).camera("001", "0007")

    assert (cam.width, cam.height, cam.fps) == (200, 200, 10.0)
    assert cam.world_to_image.shape == (3, 3)


def test_unknown_camera_is_an_error(tmp_path, rows):
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows))

    with pytest.raises(KeyError, match="Camera_0099"):
        reader.camera("001", "99")


def test_homography_sign_comes_from_the_extrinsics(tmp_path, rows):
    # NVIDIA homographies are defined up to scale and some have a negative one,
    # which puts the whole floor behind the camera. Camera 5 has no ground truth:
    # only the extrinsics can tell.
    root = write_scene(tmp_path, rows, cameras=(7, 5), flipped=(5,))

    cam = NvidiaMtmcReader(root).camera("001", "5")

    np.testing.assert_allclose(cam.world_to_image, H)


def test_camera_matrix_is_scaled_onto_the_oriented_homography(tmp_path, rows):
    root = write_scene(tmp_path, rows, cameras=(7, 5), flipped=(5,))

    cam = NvidiaMtmcReader(root).camera("001", "5")

    assert cam.projection.shape == (3, 4)
    np.testing.assert_allclose(cam.projection[:, [0, 1, 3]], H)
    np.testing.assert_allclose(cam.projection[:, 2], [7.0, 7.0, 7.0])


def test_no_extrinsics_means_no_camera_matrix(tmp_path, rows):
    cam = NvidiaMtmcReader(write_scene(tmp_path, rows, extrinsics=False)).camera("001", "7")

    assert cam.projection is None


def test_homography_sign_falls_back_to_the_camera_ground_truth(tmp_path, rows):
    root = write_scene(tmp_path, rows, flipped=(7,), extrinsics=False)

    cam = NvidiaMtmcReader(root).camera("001", "7")

    np.testing.assert_allclose(cam.world_to_image, H)


def test_lists_the_calibrated_cameras(tmp_path, rows):
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows, cameras=(7, 5)))

    assert reader.cameras("001") == ["5", "7"]


def test_floor_truth_lists_every_person_and_flags_the_camera_seeing_them(tmp_path, rows):
    # Camera 5 has a folder and a calibration but no ground-truth rows: it is
    # not one of the JSON's Camera_<i>, and camera 7 stays Camera_0000.
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows, cameras=(7, 5)))

    t = reader.floor_truth("001", "7")

    assert len(t.track_ids) == 16
    assert set(t.track_ids[t.visible]) == {1, 2}
    assert set(t.track_ids[~t.visible]) == {9}
    np.testing.assert_allclose(t.world[t.track_ids == 9], [[6.0, 9.0]])
    assert not reader.floor_truth("001", "5").visible.any()


def test_json_camera_indices_do_not_depend_on_the_folders_on_disk(tmp_path, rows):
    # An incremental download: camera 7's folder is missing, camera 8 stays Camera_0001.
    root = write_scene(tmp_path, rows, cameras=(7, 8))
    shutil.rmtree(root / "nvidia" / "MTMC_Tracking_2024" / "test" / "scene_001" / "camera_0007")
    reader = NvidiaMtmcReader(root)

    assert set(
        reader.floor_truth("001", "8").track_ids[reader.floor_truth("001", "8").visible]
    ) == {9}
    assert set(
        reader.floor_truth("001", "7").track_ids[reader.floor_truth("001", "7").visible]
    ) == {1, 2}


def test_json_cameras_that_ground_truth_txt_does_not_list_are_an_error(tmp_path, rows):
    root = write_scene(tmp_path, rows)
    path = (
        root
        / "nvidia"
        / "MTMC_Tracking_2024"
        / "test"
        / "scene_001"
        / "ground_truth_2025_format.json"
    )
    frames = json.loads(path.read_text())
    frames["0"][0]["2d_bounding_box_visible"]["Camera_0002"] = [0, 0, 1, 1]
    path.write_text(json.dumps(frames))

    with pytest.raises(ValueError, match="Camera_"):
        NvidiaMtmcReader(root).floor_truth("001", "7")
