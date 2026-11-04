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
gt-boxes projects each true box's bottom-centre through the homography, the
geometry ceiling of foot point + homography.
"""

from dataclasses import dataclass

import numpy as np

from space_usage_heatmaps.evaluation.domain.metrics import LayerMetrics, compare_grids
from space_usage_heatmaps.floorplan.domain.homography import Homography
from space_usage_heatmaps.occupancy.domain.grid import Grid, GridSpec

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


def evaluate_oracle(reader, source, scene, camera, cell):
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}, expected one of {SOURCES}")
    gt = build_ground_truth_grid(reader, scene, camera, cell)
    cam = reader.camera(scene, camera)
    s = reader.samples(scene, camera)
    if source == "gt-world":
        xs, ys = s.world[:, 0], s.world[:, 1]
    else:
        x, y, w, h = s.boxes.T
        xs, ys = Homography(cam.world_to_image).to_world(x + w / 2, y + h)
    pred = Grid.from_samples(gt.full.spec, s.track_ids, xs, ys, cam.fps)
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
    )
