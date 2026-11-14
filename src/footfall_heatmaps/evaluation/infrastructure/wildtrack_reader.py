"""Reader for WILDTRACK (EPFL CVLab), evaluation only (docs/licenses.md).

Layout under the data dir: wildtrack/Wildtrack_dataset/ as in the official
Wildtrack_dataset_full.zip, of which only what is read is fetched:
- calibrations/intrinsic_zero/intr_<name>.xml: K of the undistorted frames;
- calibrations/extrinsic/extr_<name>.xml: rvec, tvec (world in centimetres,
  z up, floor at z = 0);
- annotations_positions/<NNNNNNNN>.json: 400 annotated frames at 2 fps, named
  by their 10 fps index (0, 5, ..., 1995); per person, its positionID on the
  480 x 1440 floor grid (2.5 cm cells, origin at (-3, -9) m) and its box in
  each of the 7 views (-1 when not visible);
- Image_subsets/C<k>/<NNNNNNNN>.png: the undistorted 1920x1080 frames.
One scene; cameras C1-C7 are views 0-6, CVLab1-4 then IDIAP1-3. Frames are
0-based indices into the annotated frames, which video() turns into a 2 fps
video (frame_sequence_video).
"""

import functools
import json
import re
from pathlib import Path

import cv2
import numpy as np

from footfall_heatmaps.evaluation.domain.dataset_reader import (
    CameraCalibration,
    FloorTruth,
    Samples,
)
from footfall_heatmaps.evaluation.infrastructure.frame_sequence_video import sequence_video

SCENE = "wildtrack"
CALIBRATION_NAMES = ("CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3")
GRID_COLUMNS, GRID_ROWS, CELL_M = 480, 1440, 0.025
ORIGIN_M = (-3.0, -9.0)
WIDTH, HEIGHT, FPS = 1920, 1080, 2.0


def position_to_world(position_ids):
    """(N, 2) floor metres of grid position ids."""
    p = np.asarray(position_ids, dtype=np.int64)
    return np.column_stack(
        [ORIGIN_M[0] + CELL_M * (p % GRID_COLUMNS), ORIGIN_M[1] + CELL_M * (p // GRID_COLUMNS)]
    )


def _vector(text, tag):
    found = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    if not found:
        raise ValueError(f"no <{tag}> in extrinsic calibration")
    return np.array(found.group(1).split(), dtype=float)


class WildtrackReader:
    name = "WILDTRACK"

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "wildtrack" / "Wildtrack_dataset"
        self._rows = functools.cache(self._load_rows)

    @staticmethod
    def _check(scene):
        if scene != SCENE:
            raise KeyError(f"WILDTRACK has one scene, {SCENE!r}, not {scene!r}")

    @staticmethod
    def _view(camera):
        if not re.fullmatch(r"C[1-7]", camera):
            raise KeyError(f"WILDTRACK cameras are C1-C7, not {camera!r}")
        return int(camera[1]) - 1

    def _calibration_files(self, camera):
        name = CALIBRATION_NAMES[self._view(camera)]
        c = self.root / "calibrations"
        return c / "intrinsic_zero" / f"intr_{name}.xml", c / "extrinsic" / f"extr_{name}.xml"

    def cameras(self, scene):
        self._check(scene)
        return [
            f"C{v + 1}"
            for v in range(len(CALIBRATION_NAMES))
            if all(f.exists() for f in self._calibration_files(f"C{v + 1}"))
        ]

    def camera(self, scene, camera):
        """K [R | t / 100]: the world in metres; its third row is the depth
        (positive in front of the camera), so the floor columns need no sign fix."""
        self._check(scene)
        intrinsic, extrinsic = self._calibration_files(camera)
        fs = cv2.FileStorage(str(intrinsic), cv2.FILE_STORAGE_READ)
        k = fs.getNode("camera_matrix").mat()
        fs.release()
        if k is None or k.shape != (3, 3):
            raise ValueError(f"{intrinsic}: no 3x3 camera_matrix")
        text = extrinsic.read_text()
        r = cv2.Rodrigues(_vector(text, "rvec"))[0]
        t = _vector(text, "tvec") / 100.0
        projection = k @ np.column_stack([r, t])
        return CameraCalibration(
            world_to_image=projection[:, [0, 1, 3]],
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            projection=projection,
        )

    def depression_deg(self, scene, camera):
        """Angle of the camera's optical axis below the horizon: the third row of
        R (the axis in world coordinates) is the third row of K [R | t]."""
        axis = self.camera(scene, camera).projection[2, :3]
        return float(np.degrees(np.arcsin(-axis[2] / np.linalg.norm(axis))))

    def most_oblique(self, scene):
        """The camera whose view is furthest from looking straight down: the
        smallest depression (C5, IDIAP1, 8.7 degrees, 1.7 m above the floor)."""
        return min(self.cameras(scene), key=lambda c: self.depression_deg(scene, c))

    def _annotation_files(self):
        files = sorted((self.root / "annotations_positions").glob("*.json"))
        if not files:
            raise OSError(f"no WILDTRACK annotations under {self.root}")
        return files

    def _load_rows(self):
        """(person, frame, position, view boxes (N, 7, 4) xyxy, -1 when unseen)."""
        ids, frames, positions, boxes = [], [], [], []
        for frame, f in enumerate(self._annotation_files()):
            for p in json.loads(f.read_text()):
                ids.append(int(p["personID"]))
                frames.append(frame)
                positions.append(int(p["positionID"]))
                views = sorted(p["views"], key=lambda v: v["viewNum"])
                boxes.append([[v["xmin"], v["ymin"], v["xmax"], v["ymax"]] for v in views])
        return (
            np.array(ids, dtype=int),
            np.array(frames, dtype=int),
            position_to_world(positions).reshape(-1, 2),
            np.array(boxes, dtype=float).reshape(-1, len(CALIBRATION_NAMES), 4),
        )

    def _seen(self, camera):
        ids, frames, world, boxes = self._rows()
        b = boxes[:, self._view(camera)]
        return b, (b >= 0).all(axis=1) & (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])

    def samples(self, scene, camera):
        self._check(scene)
        ids, frames, world, _ = self._rows()
        b, seen = self._seen(camera)
        xywh = np.column_stack([b[seen, :2], b[seen, 2:] - b[seen, :2]])
        return Samples(ids[seen], frames[seen], xywh, world[seen])

    def floor_truth(self, scene, camera):
        self._check(scene)
        ids, frames, world, _ = self._rows()
        return FloorTruth(ids, frames, world, self._seen(camera)[1])

    def scene_extent(self, scene):
        self._check(scene)
        x0, y0 = ORIGIN_M
        return (x0, y0, x0 + CELL_M * GRID_COLUMNS, y0 + CELL_M * GRID_ROWS)

    def video(self, scene, camera):
        """The annotated frames of the camera as a 2 fps lossless video under
        the data dir (derived data, never committed)."""
        self._check(scene)
        images = self.root / "Image_subsets" / camera
        frames = [images / f.with_suffix(".png").name for f in self._annotation_files()]
        out = self.data_dir / "wildtrack" / "videos" / f"{camera}.mkv"
        return sequence_video(frames, FPS, out)
