"""Score a multi-camera fused floor map against the floor ground truth (E2a).

Every calibrated camera of a scene (or the ones named) runs the same recipe;
each one's tracks become dwell and passage per 30 s block on the scene grid,
as in evaluate_config. occupancy.domain.fusion weighs them per cell from
their whole-run grids and fuses each block with those weights, so the paired
block bootstrap resamples the fused map like any other prediction.

Predictions: the fused map, and the single-camera baseline (the scene's E1
camera, the anchor, with its own recipe); deltas are fused - single.

Truth blocks (spec 9, E2a):
- full@joint: every person, on the joint coverage (cells at least one camera
  covers); the E2a target;
- visible@joint: the people at least one fused camera has a visible box of,
  on the joint coverage;
- full@anchor, visible@anchor: the anchor camera's E1 blocks (full@coverage
  and visible@coverage), where fused and single camera meet on the same cells;
- full@overlap: every person on the cells two or more cameras cover, where a
  fusion that adds cameras counts people twice (its mass error is the
  double-counting check).
"""

from dataclasses import dataclass

import numpy as np

from space_usage_heatmaps.evaluation.application.evaluate_config import BLOCK_S
from space_usage_heatmaps.evaluation.application.evaluate_oracle import camera_floor_points
from space_usage_heatmaps.evaluation.domain.bootstrap import (
    BlockLayers,
    SceneBlocks,
    block_bootstrap,
    block_layers,
)
from space_usage_heatmaps.evaluation.domain.metrics import compare_layer
from space_usage_heatmaps.floorplan.domain.homography import Homography
from space_usage_heatmaps.occupancy.domain.fusion import (
    CameraGrid,
    Fusion,
    fuse_layers,
    fusion_weights,
    joint_coverage,
    overlap,
)
from space_usage_heatmaps.occupancy.domain.grid import Grid, GridSpec

FUSION_TRUTHS = ("full@joint", "visible@joint", "full@anchor", "visible@anchor", "full@overlap")


@dataclass(frozen=True)
class FusionScene:
    scene: str
    anchor: str  # the single-camera baseline's camera
    cameras: tuple[str, ...]  # fused
    blocks: int
    joint_cells: int
    overlap_cells: int
    anchor_cells: int
    samples: int  # tracked boxes of every fused camera in the truth frames
    dropped: int  # of those, outside the grid
    occlusion_anchor: float  # share of the full dwell on the anchor's coverage it never sees
    occlusion_joint: float  # share of the full dwell on the joint coverage no fused camera sees
    sum_overlap_mass_error: float  # dwell mass error on the overlap if the cameras were added


@dataclass(frozen=True)
class FusionReport:
    dataset: str
    cell_m: float
    block_s: float
    n_resamples: int
    fusion: Fusion
    scenes: list[FusionScene]
    results: dict[str, object]  # truth block -> BootstrapResult, preds (fused, single)
    # Per scene, the fused map of the whole run (for the viewer's export).
    fused: dict[str, Grid]


def _camera_blocks(reader, scene, camera, config, source, spec, first, last, bf, nb):
    cam = reader.camera(scene, camera)
    t = source.tracks(scene, camera)
    keep = (t.frames >= first) & (t.frames <= last)
    xs, ys, _ = camera_floor_points(
        cam,
        t.boxes,
        config.foot_point,
        config.person_height_m,
        config.min_implied_height_m,
        t.track_ids,
        t.frames,
        config.track_heights(),
    )
    layers = block_layers(
        spec, t.track_ids[keep], t.frames[keep], xs[keep], ys[keep], cam.fps, bf, nb
    )
    coverage = Homography(cam.world_to_image).sees(*spec.cell_centres(), cam.width, cam.height)
    return layers, coverage, int(keep.sum())


def evaluate_fusion(
    reader, scenes, fused_run, single_run, cell, fusion, n_resamples=1000, seed=0, cameras=None
):
    """scenes: [(scene, anchor camera)]; fused_run, single_run: (RunConfig,
    TrackSource); cameras: the cameras to fuse, default every calibrated one."""
    blocks = {truth: [] for truth in FUSION_TRUTHS}
    report_scenes, fused_grids = [], {}
    for scene, anchor in scenes:
        names = [str(c) for c in (cameras or reader.cameras(scene))]
        spec = GridSpec.covering(*reader.scene_extent(scene), cell_m=cell)
        truth = reader.floor_truth(scene, anchor)
        first, last = int(truth.frames.min()), int(truth.frames.max())
        fps = {reader.camera(scene, c).fps for c in [anchor, *names]}
        if len(fps) != 1:
            raise ValueError(f"scene {scene}: cameras run at different frame rates {sorted(fps)}")
        (rate,) = fps
        bf = round(BLOCK_S * rate)
        nb = last // bf + 1

        def truth_layers(rows, t=truth, spec=spec, rate=rate, bf=bf, nb=nb):
            return block_layers(
                spec, t.track_ids[rows], t.frames[rows], *t.world[rows].T, rate, bf, nb
            )

        seen_by_any = np.zeros(len(truth.track_ids), dtype=bool)
        per_camera, samples, dropped = [], 0, 0
        config, source = fused_run
        for c in names:
            layers, coverage, n = _camera_blocks(
                reader, scene, c, config, source, spec, first, last, bf, nb
            )
            per_camera.append((c, layers, coverage))
            samples, dropped = samples + n, dropped + layers.dropped
            seen_by_any |= reader.floor_truth(scene, c).visible
        grids = [
            CameraGrid(c, Grid(spec, b.dwell.sum(0), b.passage.sum(0), b.dropped), cov)
            for c, b, cov in per_camera
        ]
        w = fusion_weights(grids, fusion)
        fused = BlockLayers(
            fuse_layers(w, [b.dwell for _, b, _ in per_camera]),
            fuse_layers(w, [b.passage for _, b, _ in per_camera]),
            dropped,
        )
        fused_grids[scene] = Grid(spec, fused.dwell.sum(0), fused.passage.sum(0), dropped)
        single_config, single_source = single_run
        single, anchor_cov, _ = _camera_blocks(
            reader, scene, anchor, single_config, single_source, spec, first, last, bf, nb
        )
        joint, shared = joint_coverage(grids), overlap(grids)
        full = truth_layers(np.ones(len(truth.track_ids), dtype=bool))
        visible_any, visible_anchor = truth_layers(seen_by_any), truth_layers(truth.visible)
        preds = (fused, single)
        for name, tr, mask in (
            ("full@joint", full, joint),
            ("visible@joint", visible_any, joint),
            ("full@anchor", full, anchor_cov),
            ("visible@anchor", visible_anchor, anchor_cov),
            ("full@overlap", full, shared),
        ):
            blocks[name].append(SceneBlocks(tr, preds, mask, cell))
        full_dwell = full.dwell.sum(0)
        sum_overlap = None
        if shared.any():
            added = fuse_layers(fusion_weights(grids, Fusion("sum")), [g.grid.dwell for g in grids])
            sum_overlap = compare_layer(added, full_dwell, shared, cell).mass_error
        report_scenes.append(
            FusionScene(
                scene=scene,
                anchor=anchor,
                cameras=tuple(names),
                blocks=nb,
                joint_cells=int(joint.sum()),
                overlap_cells=int(shared.sum()),
                anchor_cells=int(anchor_cov.sum()),
                samples=samples,
                dropped=dropped,
                occlusion_anchor=_unseen(full_dwell, visible_anchor.dwell.sum(0), anchor_cov),
                occlusion_joint=_unseen(full_dwell, visible_any.dwell.sum(0), joint),
                sum_overlap_mass_error=sum_overlap,
            )
        )
    if any(s.overlap_cells == 0 for s in report_scenes):
        del blocks["full@overlap"]  # a scene without overlap: nothing to double count
    return FusionReport(
        dataset=reader.name,
        cell_m=cell,
        block_s=BLOCK_S,
        n_resamples=n_resamples,
        fusion=fusion,
        scenes=report_scenes,
        results={
            truth: block_bootstrap(b, n_resamples=n_resamples, seed=seed)
            for truth, b in blocks.items()
        },
        fused=fused_grids,
    )


def _unseen(full, seen, mask):
    total = full[mask].sum()
    return float(1 - seen[mask].sum() / total) if total else 0.0
