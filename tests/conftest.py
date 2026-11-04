"""A tiny dataset in the NVIDIA MTMC 2024 layout, written to a temp dir."""

import json

import numpy as np
import pytest

# Camera at the origin looking along +y: u = (100 x + 50 y) / y, v = (50 y + 100) / y.
H = [[100.0, 50.0, 0.0], [0.0, 50.0, 100.0], [0.0, 1.0, 0.0]]
# K [R | t] whose floor columns (0, 1, 3) are H: its third row is the depth.
K = np.eye(3).tolist()
E = [[100.0, 50.0, 7.0, 0.0], [0.0, 50.0, 7.0, 100.0], [0.0, 1.0, 7.0, 0.0]]
BOX_W, BOX_H = 10.0, 20.0


def gt_row(camera, track, frame, x, y, foot_shift_px=0.0):
    """One ground_truth.txt row whose box bottom-centre projects onto (x, y)."""
    u = (100 * x + 50 * y) / y
    v = (50 * y + 100) / y + foot_shift_px
    return (camera, track, frame, u - BOX_W / 2, v - BOX_H, BOX_W, BOX_H, x, y)


def write_scene(root, rows, scene="001", cameras=(7,), flipped=(), extrinsics=True):
    """ground_truth.txt from rows, the 2025 JSON with the same people (Camera_<i>
    is the i-th camera id of ground_truth.txt in sorted order), a camera_<NNNN>
    folder per camera and a calibration for each of cameras;
    flipped ones get the homography with the opposite sign, which is the same
    homography up to scale."""
    d = root / "nvidia" / "MTMC_Tracking_2024" / "test" / f"scene_{scene}"
    d.mkdir(parents=True)
    np.savetxt(d / "ground_truth.txt", np.array(rows), fmt="%.10g")
    for c in sorted({int(r[0]) for r in rows} | set(cameras)):
        (d / f"camera_{c:04d}").mkdir()
    listed = sorted({int(r[0]) for r in rows})

    people = {}
    for c, track, frame, x, y, w, h, wx, wy in rows:
        p = people.setdefault((int(frame), int(track)), {"loc": [wx, wy, 0.9], "boxes": {}})
        p["boxes"][f"Camera_{listed.index(int(c)):04d}"] = [x, y, x + w, y + h]
    frames = {}
    for (frame, track), p in sorted(people.items()):
        frames.setdefault(str(frame), []).append(
            {
                "object_type": "Person",
                "object_id": track,
                "3d_location": p["loc"],
                "2d_bounding_box_visible": p["boxes"],
            }
        )
    (d / "ground_truth_2025_format.json").write_text(json.dumps(frames))

    sensors = []
    for c in cameras:
        sign = -1.0 if c in flipped else 1.0
        sensor = {
            "id": f"Camera_{c:04d}",
            "attributes": [
                {"name": "fps", "value": "10"},
                {"name": "frameWidth", "value": "200"},
                {"name": "frameHeight", "value": "200"},
            ],
            "homography": (sign * np.array(H)).tolist(),
        }
        if extrinsics:
            sensor |= {"intrinsicMatrix": K, "extrinsicMatrix": E}
        sensors.append(sensor)
    (d / "calibration_2025_format.json").write_text(json.dumps({"sensors": sensors}))
    return root


@pytest.fixture
def rows():
    # Track 1 walks in front of camera 7; track 2 stands still; camera 8 sees
    # track 9 inside camera 7's view, hidden from camera 7 (occluded).
    walk = [gt_row(7, 1, f, 0.07 + 0.2 * f, 2.03 + 0.1 * f) for f in range(10)]
    stand = [gt_row(7, 2, f, -0.4, 3.1) for f in range(5)]
    return [*walk, *stand, gt_row(8, 9, 0, 6.0, 9.0)]
