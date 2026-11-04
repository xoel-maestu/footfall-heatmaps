"""Reader for NVIDIA PhysicalAI-SmartSpaces MTMC_Tracking_2024.

Layout under the data dir: nvidia/MTMC_Tracking_2024/test/scene_<NNN>/ with
- ground_truth.txt: one row per camera, track and frame where the camera has a
  visible box (camera_id track_id frame xmin ymin width height xworld yworld);
- ground_truth_2025_format.json: per frame, every person with its 3d_location
  and 2d_bounding_box_visible keyed Camera_<i>, the i-th camera id of
  ground_truth.txt in sorted order (Camera_0010 is camera 645 in scene 071). The
  ids come from the data, not from the camera_<NNNN> folders, which an
  incremental download may lack; the JSON must index exactly those cameras;
- calibration_2025_format.json: per Camera_<NNNN>, fps, frame size, intrinsic and
  extrinsic matrices and the world-to-image homography (defined up to scale; its
  sign is normalised here, see _oriented).
The two ground truths are parsed independently, so scoring one against the
other is a check of both.
"""

import functools
import json
from pathlib import Path

import numpy as np

from space_usage_heatmaps.evaluation.domain.dataset_reader import (
    CameraCalibration,
    FloorTruth,
    Samples,
)


class NvidiaMtmcReader:
    name = "nvidia/PhysicalAI-SmartSpaces MTMC_Tracking_2024"

    def __init__(self, data_dir):
        self.root = Path(data_dir) / "nvidia" / "MTMC_Tracking_2024" / "test"
        # Per-instance caches: each file is parsed once per reader and scene.
        self._rows = functools.cache(self._load_rows)
        self._sensors = functools.cache(self._load_sensors)
        self._people = functools.cache(self._load_people)

    def _scene(self, scene):
        return self.root / f"scene_{int(scene):03d}"

    def _load_rows(self, scene):
        return np.loadtxt(self._scene(scene) / "ground_truth.txt", ndmin=2)

    def _load_sensors(self, scene):
        calib = json.loads((self._scene(scene) / "calibration_2025_format.json").read_text())
        return {s["id"]: s for s in calib["sensors"]}

    def cameras(self, scene):
        return sorted((str(int(k.removeprefix("Camera_"))) for k in self._sensors(scene)), key=int)

    def camera(self, scene, camera):
        key = f"Camera_{int(camera):04d}"
        sensors = self._sensors(scene)
        if key not in sensors:
            raise KeyError(f"{key} not in scene {scene} calibration")
        s = sensors[key]
        attrs = {a["name"]: a["value"] for a in s["attributes"]}
        return CameraCalibration(
            world_to_image=self._oriented(scene, camera, s),
            width=int(attrs["frameWidth"]),
            height=int(attrs["frameHeight"]),
            fps=float(attrs["fps"]),
        )

    def _oriented(self, scene, camera, sensor):
        """The homography times -1 if its third row is negative in front of the camera.

        The depth of a floor point (x, y, 0) is the third row of K [R | t] on
        columns 0, 1 and 3, which the homography equals up to scale: the sign of
        that scale is the sign to fix. Without extrinsics, the camera's own
        ground-truth floor points, which are in front of it, give it.
        """
        h = np.array(sensor["homography"], dtype=float)
        if "intrinsicMatrix" in sensor and "extrinsicMatrix" in sensor:
            p = np.array(sensor["intrinsicMatrix"]) @ np.array(sensor["extrinsicMatrix"])
            sign = np.sign(np.vdot(h, p[:, [0, 1, 3]]))
        else:
            w = self.samples(scene, camera).world
            sign = np.sign(np.median(h[2] @ np.vstack([w.T, np.ones(len(w))]))) if len(w) else 1
        return h * (sign or 1)

    def video(self, scene, camera):
        return self._scene(scene) / f"camera_{int(camera):04d}" / "video.mp4"

    def samples(self, scene, camera):
        r = self._rows(scene)
        r = r[r[:, 0] == int(camera)]
        return Samples(
            track_ids=r[:, 1].astype(int),
            frames=r[:, 2].astype(int),
            boxes=r[:, 3:7],
            world=r[:, 7:9],
        )

    def _load_people(self, scene):
        """Arrays of every Person row of the 2025 JSON; seen is a bit set of the
        Camera_<i> indices with a visible box. Also the camera ids of
        ground_truth.txt in sorted order, which Camera_<i> indexes."""
        d = self._scene(scene)
        frames = json.loads((d / "ground_truth_2025_format.json").read_text())
        ids, fr, xy, seen, keys = [], [], [], [], set()
        for frame, people in frames.items():
            for p in people:
                if p["object_type"] != "Person":
                    continue
                ids.append(int(p["object_id"]))
                fr.append(int(frame))
                xy.append(p["3d_location"][:2])
                bits = 0
                for k in p.get("2d_bounding_box_visible") or {}:
                    keys.add(int(k.removeprefix("Camera_")))
                    bits |= 1 << int(k.removeprefix("Camera_"))
                seen.append(bits)
        listed = sorted({int(c) for c in self._rows(scene)[:, 0]})
        if keys != set(range(len(listed))):
            raise ValueError(
                f"scene {scene}: ground_truth_2025_format.json has {len(keys)} Camera_<i> keys "
                f"(max index {max(keys, default=-1)}) but ground_truth.txt lists "
                f"{len(listed)} cameras: cannot map Camera_<i> to camera ids"
            )
        return (
            np.array(ids, dtype=int),
            np.array(fr, dtype=int),
            np.array(xy, dtype=float).reshape(-1, 2),
            np.array(seen, dtype=np.int64),
            listed,
        )

    def floor_truth(self, scene, camera):
        """A camera ground_truth.txt does not list has no visible box of anyone."""
        ids, frames, world, seen, listed = self._people(scene)
        if int(camera) not in listed:
            return FloorTruth(ids, frames, world, np.zeros(len(ids), dtype=bool))
        bit = np.int64(1) << listed.index(int(camera))
        return FloorTruth(ids, frames, world, (seen & bit) != 0)

    def scene_extent(self, scene):
        w = np.concatenate([self._rows(scene)[:, 7:9], self._people(scene)[2]])
        return (*map(float, w.min(axis=0)), *map(float, w.max(axis=0)))
