"""MOT17 reader on a tiny synthetic sequence."""

import cv2
import numpy as np
import pytest

from space_usage_heatmaps.evaluation.infrastructure.mot17_reader import CAMERA, Mot17Reader

SEQINFO = """[Sequence]
name=MOT17-02-FRCNN
imDir=img1
frameRate=30
seqLength=3
imWidth=64
imHeight=48
imExt=.jpg
"""
# frame, id, x, y, w, h, consider, class, visibility (1-based frames)
GT = """1,1,10,20,30,40,1,1,1.0
1,2,40,5,10,20,0,7,0.5
2,1,11,20,30,40,1,1,1.0
2,3,0,0,5,5,0,3,1.0
2,4,1,1,5,5,0,1,1.0
3,5,30,10,8,16,1,12,0.9
"""


def write_sequence(root, seq="MOT17-02"):
    d = root / "mot17" / "MOT17" / "train" / f"{seq}-FRCNN"
    (d / "gt").mkdir(parents=True)
    (d / "img1").mkdir()
    (d / "seqinfo.ini").write_text(SEQINFO)
    (d / "gt" / "gt.txt").write_text(GT)
    for i in range(1, 4):
        cv2.imwrite(str(d / "img1" / f"{i:06d}.jpg"), np.full((48, 64, 3), 10 * i, np.uint8))
    return root


def test_scenes_are_the_fetched_sequences(tmp_path):
    reader = Mot17Reader(write_sequence(tmp_path))

    assert reader.scenes() == ["MOT17-02"]
    assert reader.cameras("MOT17-02") == [CAMERA]


def test_samples_keep_pedestrians_and_mark_distractors(tmp_path):
    s = Mot17Reader(write_sequence(tmp_path)).samples("MOT17-02", CAMERA)

    # Scored: considered pedestrians. Distractors: classes 2, 7, 8, 12 (TrackEval's
    # MOT17 list). Dropped: other classes (vehicles, occluders) and ignored pedestrians.
    assert s.track_ids.tolist() == [1, 2, 1, 5]
    assert s.frames.tolist() == [0, 0, 1, 2]
    assert s.distractor.tolist() == [False, True, False, True]
    assert s.boxes[0] == pytest.approx([10, 20, 30, 40])
    assert np.isnan(s.world).all()


def test_camera_and_video_follow_seqinfo(tmp_path):
    reader = Mot17Reader(write_sequence(tmp_path))

    cam = reader.camera("MOT17-02", CAMERA)
    video = reader.video("MOT17-02", CAMERA)

    assert (cam.width, cam.height, cam.fps) == (64, 48, 30.0)
    assert cam.world_to_image is None
    cap = cv2.VideoCapture(str(video))
    assert cap.get(cv2.CAP_PROP_FPS) == 30.0
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    with pytest.raises(ValueError, match="no floor truth"):
        reader.scene_extent("MOT17-02")
