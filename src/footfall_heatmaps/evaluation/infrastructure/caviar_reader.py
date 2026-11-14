"""Reader for CAVIAR's second set (shopping centre corridor in Lisbon),
evaluation only (docs/licenses.md).

Layout under the data dir: caviar/<Clip>cor/ with the clip's corridor-view
MPEG-1 video (<Clip>cor.mpg, 384x288, 25 fps) and its CVML ground truth
(c*gt.xml): per frame, each person's id and box as centre, width and height
in pixels. One scene per clip, one camera ("cor"). No floor truth: the
tracking guards only.
"""

import functools
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from footfall_heatmaps.evaluation.domain.dataset_reader import CameraCalibration, Samples

CAMERA = "cor"


class CaviarReader:
    name = "CAVIAR (second set, corridor view)"

    def __init__(self, data_dir):
        self.root = Path(data_dir) / "caviar"
        self._samples = functools.cache(self._load_samples)

    def scenes(self):
        return sorted(d.name for d in self.root.iterdir() if d.is_dir() and d.name.endswith("cor"))

    def _clip(self, scene, camera):
        if camera != CAMERA:
            raise KeyError(f"CAVIAR clips have one camera, {CAMERA!r}, not {camera!r}")
        d = self.root / scene
        if not d.is_dir():
            raise KeyError(f"no CAVIAR clip {scene!r} under {self.root}")
        return d

    def cameras(self, scene):
        self._clip(scene, CAMERA)
        return [CAMERA]

    def video(self, scene, camera):
        return self._clip(scene, camera) / f"{scene}.mpg"

    def camera(self, scene, camera):
        cap = cv2.VideoCapture(str(self.video(scene, camera)))
        try:
            if not cap.isOpened():
                raise OSError(f"cannot open video {self.video(scene, camera)}")
            size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
        finally:
            cap.release()
        return CameraCalibration(world_to_image=None, width=size[0], height=size[1], fps=fps)

    def _load_samples(self, scene):
        (gt,) = self._clip(scene, CAMERA).glob("*gt.xml")
        rows = []
        for frame in ET.parse(gt).getroot().iter("frame"):
            n = int(frame.get("number"))
            for obj in frame.iter("object"):
                b = obj.find("box")
                w, h = float(b.get("w")), float(b.get("h"))
                xc, yc = float(b.get("xc")), float(b.get("yc"))
                rows.append((int(obj.get("id")), n, xc - w / 2, yc - h / 2, w, h))
        a = np.array(rows, dtype=float).reshape(-1, 6)
        return Samples(
            a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2:6], np.full((len(a), 2), np.nan)
        )

    def samples(self, scene, camera):
        self._clip(scene, camera)
        return self._samples(scene)

    def floor_truth(self, scene, camera):
        raise ValueError("CAVIAR has no floor truth: tracking guards only")

    def scene_extent(self, scene):
        raise ValueError("CAVIAR has no floor truth: tracking guards only")
