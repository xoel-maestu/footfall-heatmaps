"""Reader for MOT17 train sequences, evaluation only (CC BY-NC-SA 3.0,
docs/licenses.md).

Layout under the data dir: mot17/MOT17/train/MOT17-<NN>-FRCNN/ as in the
official MOT17.zip, of which only the static-camera sequences 02, 04 and 09
are fetched (the frames are the same in the DPM, FRCNN and SDP copies; only
their public detections differ, which are not used): seqinfo.ini, gt/gt.txt
(frame, id, x, y, w, h, consider, class, visibility; 1-based frames) and
img1/<NNNNNN>.jpg. One scene per sequence, one camera ("1"). No floor truth:
the tracking guards only, with TrackEval's MOT17 rules: considered
pedestrians are scored, classes 2, 7, 8 and 12 are distractors, the rest is
left out.
"""

import configparser
import functools
from pathlib import Path

import numpy as np

from space_usage_heatmaps.evaluation.domain.dataset_reader import CameraCalibration, Samples
from space_usage_heatmaps.evaluation.infrastructure.frame_sequence_video import sequence_video

CAMERA = "1"
PEDESTRIAN = 1
DISTRACTORS = (2, 7, 8, 12)  # person on vehicle, static person, distractor, reflection


class Mot17Reader:
    name = "MOT17 train (static cameras)"

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "mot17" / "MOT17" / "train"
        self._samples = functools.cache(self._load_samples)

    def scenes(self):
        return sorted(d.name.removesuffix("-FRCNN") for d in self.root.glob("MOT17-*-FRCNN"))

    def _sequence(self, scene, camera):
        if camera != CAMERA:
            raise KeyError(f"MOT17 sequences have one camera, {CAMERA!r}, not {camera!r}")
        d = self.root / f"{scene}-FRCNN"
        if not d.is_dir():
            raise KeyError(f"no MOT17 sequence {scene!r} under {self.root}")
        return d

    def _info(self, scene):
        ini = configparser.ConfigParser()
        ini.read(self._sequence(scene, CAMERA) / "seqinfo.ini")
        return ini["Sequence"]

    def cameras(self, scene):
        self._sequence(scene, CAMERA)
        return [CAMERA]

    def camera(self, scene, camera):
        self._sequence(scene, camera)
        info = self._info(scene)
        return CameraCalibration(
            world_to_image=None,
            width=int(info["imWidth"]),
            height=int(info["imHeight"]),
            fps=float(info["frameRate"]),
        )

    def video(self, scene, camera):
        """The sequence's frames as a lossless video under the data dir."""
        d = self._sequence(scene, camera)
        info = self._info(scene)
        ext = info.get("imExt", ".jpg")
        frames = [d / info["imDir"] / f"{i:06d}{ext}" for i in range(1, int(info["seqLength"]) + 1)]
        out = self.data_dir / "mot17" / "videos" / f"{scene}.mkv"
        return sequence_video(frames, float(info["frameRate"]), out)

    def _load_samples(self, scene):
        gt = np.loadtxt(self._sequence(scene, CAMERA) / "gt" / "gt.txt", delimiter=",", ndmin=2)
        cls = gt[:, 7].astype(int)
        scored = (gt[:, 6] != 0) & (cls == PEDESTRIAN)
        distractor = np.isin(cls, DISTRACTORS)
        g = gt[scored | distractor]
        return Samples(
            g[:, 1].astype(int),
            g[:, 0].astype(int) - 1,
            g[:, 2:6],
            np.full((len(g), 2), np.nan),
            distractor=distractor[scored | distractor],
        )

    def samples(self, scene, camera):
        self._sequence(scene, camera)
        return self._samples(scene)

    def floor_truth(self, scene, camera):
        raise ValueError("MOT17 has no floor truth: tracking guards only")

    def scene_extent(self, scene):
        raise ValueError("MOT17 has no floor truth: tracking guards only")
