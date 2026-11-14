"""CAVIAR reader on a tiny synthetic corridor clip."""

import cv2
import numpy as np
import pytest

from footfall_heatmaps.evaluation.infrastructure.caviar_reader import CAMERA, CaviarReader

GT = """<?xml version="1.0" encoding="UTF-8"?><dataset name="Walk1cor">
<frame number="0"><objectlist>
  <object id="0"><orientation>90</orientation><box h="40" w="12" xc="100" yc="50"/></object>
  <object id="3"><box h="20" w="10" xc="30" yc="40"/></object>
</objectlist><grouplist/></frame>
<frame number="1"><objectlist>
  <object id="0"><box h="40" w="12" xc="102" yc="50"/></object>
</objectlist></frame>
<frame number="2"><objectlist/></frame>
</dataset>
"""


def write_clip(root, name="Walk1cor"):
    d = root / "caviar" / name
    d.mkdir(parents=True)
    (d / "cw1gt.xml").write_text(GT)
    w = cv2.VideoWriter(str(d / f"{name}.mpg"), cv2.VideoWriter_fourcc(*"PIM1"), 25.0, (64, 48))
    for _ in range(3):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    return root


def test_scenes_are_the_corridor_clips(tmp_path):
    write_clip(tmp_path, "Walk1cor")
    write_clip(tmp_path, "Walk2cor")
    (tmp_path / "caviar" / "Walk1front").mkdir()

    reader = CaviarReader(tmp_path)

    assert reader.scenes() == ["Walk1cor", "Walk2cor"]
    assert reader.cameras("Walk1cor") == [CAMERA]


def test_samples_are_the_boxes_from_their_centres(tmp_path):
    s = CaviarReader(write_clip(tmp_path)).samples("Walk1cor", CAMERA)

    assert s.track_ids.tolist() == [0, 3, 0]
    assert s.frames.tolist() == [0, 0, 1]
    assert s.boxes[0] == pytest.approx([94, 30, 12, 40])
    assert np.isnan(s.world).all()


def test_camera_comes_from_the_video_and_has_no_floor(tmp_path):
    reader = CaviarReader(write_clip(tmp_path))

    cam = reader.camera("Walk1cor", CAMERA)

    assert (cam.width, cam.height, cam.fps) == (64, 48, 25.0)
    assert cam.world_to_image is None and cam.projection is None
    assert reader.video("Walk1cor", CAMERA).name == "Walk1cor.mpg"
    with pytest.raises(ValueError, match="no floor truth"):
        reader.floor_truth("Walk1cor", CAMERA)
