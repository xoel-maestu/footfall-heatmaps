"""Oracle runs: score ground truth against ground truth to validate the harness.

Three blocks, from the dataset's per-frame floor positions of every person:
- visible@coverage: people this camera has a visible box of, on its coverage
  mask (E1 primary: mass the camera puts where nobody is visible counts);
- full@coverage: every person, on the coverage mask (spec 5, the E2a target);
  its gap to visible@coverage is the occlusion ceiling of a single camera;
- visible@visibility: the visible people on the covered cells where the camera
  sees someone at least once (diagnostic only: the mask comes from the truth,
  so it hides false positives in cells without visible truth).

gt-world puts the camera's own ground-truth floor points on the grid (a
separate file from the truths: must score exact zeros on both visible blocks);
gt-boxes places each true box with a foot point (floorplan.domain.foot_point:
bottom-centre by default, head or auto), the geometry ceiling of foot point +
calibration. Both report the per-sample ground position error against the
2025 JSON truth, matched by track id and frame (for gt-world: how far the two
truth files disagree).
"""

from dataclasses import dataclass

import numpy as np

from space_usage_heatmaps.evaluation.domain.metrics import LayerMetrics, compare_grids
from space_usage_heatmaps.evaluation.domain.position_error import PositionError, position_error
from space_usage_heatmaps.evaluation.domain.tracks import Tracks
from space_usage_heatmaps.floorplan.domain.foot_point import (
    MIN_IMPLIED_HEIGHT_M,
    PERSON_HEIGHT_M,
    floor_points,
)
from space_usage_heatmaps.floorplan.domain.homography import Homography
from space_usage_heatmaps.occupancy.domain.grid import Grid, GridSpec
from space_usage_heatmaps.occupancy.domain.stitching import stitch

SOURCES = ("gt-world", "gt-boxes")
TRUTHS = ("visible@coverage", "full@coverage", "visible@visibility")


@dataclass(frozen=True)
class GroundTruth:
    full: Grid
    visible: Grid
    coverage: np.ndarray  # cells whose centre projects inside the image
    visibility: np.ndarray  # covered cells with at least one visible sample


@dataclass(frozen=True)
class OracleReport:
    dataset: str
    source: str
    scene: str
    camera: str
    cell_m: float
    results: dict[str, dict[str, LayerMetrics]]  # truth -> layer -> metrics
    occlusion_fraction: float  # share of the full dwell on coverage the camera never sees
    samples: int
    dropped: int  # prediction samples that fell outside the grid
    coverage_cells: int
    visibility_cells: int
    position_error: PositionError  # per sample, metres, against the 2025 JSON truth
    foot_point: str | None = None  # gt-boxes only
    head_samples: int = 0  # gt-boxes samples placed from the box top


def build_ground_truth_grid(reader, scene, camera, cell):
    """Full and visible truth grids of one camera on the scene extent, with their masks."""
    spec = GridSpec.covering(*reader.scene_extent(scene), cell_m=cell)
    cam = reader.camera(scene, camera)
    t = reader.floor_truth(scene, camera)
    v = t.visible
    full = Grid.from_samples(spec, t.track_ids, t.world[:, 0], t.world[:, 1], cam.fps)
    visible = Grid.from_samples(spec, t.track_ids[v], t.world[v, 0], t.world[v, 1], cam.fps)
    coverage = Homography(cam.world_to_image).sees(*spec.cell_centres(), cam.width, cam.height)
    if not coverage.any():
        raise ValueError(
            f"camera {camera} of scene {scene}: coverage mask is empty, no floor cell "
            "projects inside the image (homography sign or calibration)"
        )
    visibility = coverage & (visible.dwell > 0)
    if not visibility.any():
        raise ValueError(
            f"camera {camera} of scene {scene}: visibility mask is empty, "
            "no person is ever visible on its covered floor"
        )
    return GroundTruth(full, visible, coverage, visibility)


def camera_floor_points(
    cam,
    boxes,
    foot_point,
    person_height_m=PERSON_HEIGHT_M,
    min_implied_height_m=MIN_IMPLIED_HEIGHT_M,
    track_ids=None,
    frames=None,
    track_heights=None,
):
    """floor_points of x1, y1, x2, y2 boxes with one camera's calibration; the
    image points (centroid, bottom) need only its ground homography."""
    projection = cam.projection
    if projection is None:
        if foot_point not in ("centroid", "bottom"):
            raise ValueError(f"foot point {foot_point!r} needs the 3x4 camera matrix")
        projection = np.insert(cam.world_to_image, 2, 0.0, axis=1)
    return floor_points(
        boxes,
        projection,
        cam.height,
        foot_point,
        person_height_m,
        min_implied_height_m,
        track_ids,
        frames,
        track_heights,
    )


def recipe_floor_points(cam, tracks, config):
    """(ids, xs, ys, head): the floor points of a recipe's tracks and the ids
    they are scored with. The foot point reads the tracker's own ids
    (auto-track's per-track height); the config's stitching, if any, then
    relabels the ids from those points, so the points, and dwell, are the
    same with and without it."""
    xs, ys, head = camera_floor_points(
        cam,
        tracks.boxes,
        config.foot_point,
        config.person_height_m,
        config.min_implied_height_m,
        tracks.track_ids,
        tracks.frames,
        config.track_heights(),
    )
    ids, s = tracks.track_ids, config.stitcher()
    if s is not None:
        ids = stitch(ids, tracks.frames, np.column_stack([xs, ys]), cam.fps, s)
    return ids, xs, ys, head


def stitched_tracks(cam, tracks, config):
    """The tracks with the ids recipe_floor_points scores them with (for the
    image-space tracking metrics); unchanged without a stitch section."""
    if config.stitcher() is None:
        return tracks
    ids, _, _, _ = recipe_floor_points(cam, tracks, config)
    return Tracks(ids, tracks.frames, tracks.boxes)


def evaluate_oracle(
    reader,
    source,
    scene,
    camera,
    cell,
    foot_point="bottom",
    person_height_m=PERSON_HEIGHT_M,
    min_implied_height_m=MIN_IMPLIED_HEIGHT_M,
):
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}, expected one of {SOURCES}")
    gt = build_ground_truth_grid(reader, scene, camera, cell)
    cam = reader.camera(scene, camera)
    s = reader.samples(scene, camera)
    head = np.zeros(len(s.track_ids), dtype=bool)
    if source == "gt-world":
        xs, ys = s.world[:, 0], s.world[:, 1]
    else:
        x, y, w, h = s.boxes.T
        boxes = np.stack([x, y, x + w, y + h], axis=1)
        xs, ys, head = camera_floor_points(
            cam, boxes, foot_point, person_height_m, min_implied_height_m, s.track_ids, s.frames
        )
    pred = Grid.from_samples(gt.full.spec, s.track_ids, xs, ys, cam.fps)
    truth = reader.floor_truth(scene, camera)
    error = position_error(
        s.track_ids, s.frames, np.column_stack([xs, ys]), truth.track_ids, truth.frames, truth.world
    )
    seen, total = gt.visible.dwell[gt.coverage].sum(), gt.full.dwell[gt.coverage].sum()
    return OracleReport(
        dataset=reader.name,
        source=source,
        scene=scene,
        camera=camera,
        cell_m=cell,
        results={
            "visible@coverage": compare_grids(pred, gt.visible, gt.coverage),
            "full@coverage": compare_grids(pred, gt.full, gt.coverage),
            "visible@visibility": compare_grids(pred, gt.visible, gt.visibility),
        },
        occlusion_fraction=float(1 - seen / total),
        samples=len(s.track_ids),
        dropped=pred.dropped,
        coverage_cells=int(np.count_nonzero(gt.coverage)),
        visibility_cells=int(np.count_nonzero(gt.visibility)),
        position_error=error,
        foot_point=None if source == "gt-world" else foot_point,
        head_samples=int(head.sum()),
    )
