"""Validate the label-free metrics against the error the truth gives.

For each scene, the config's tracks give the label-free metrics (from the
tracks alone) and the true error of their dwell map on visible@coverage (TVD
and mass error, the E1 primary block). Both are also cut into the 30 s blocks
of the bootstrap. The correlations say how far a label-free metric could stand
in for the true error on a camera without truth: per scene, and per block
pooled over every scene (blocks of one scene are not independent, so the
block n overstates the evidence).
"""

from dataclasses import dataclass

import numpy as np

from footfall_heatmaps.evaluation.application.evaluate_config import BLOCK_S
from footfall_heatmaps.evaluation.application.evaluate_oracle import build_ground_truth_grid
from footfall_heatmaps.evaluation.domain.bootstrap import block_layers
from footfall_heatmaps.evaluation.domain.label_free import (
    LabelFree,
    correlation,
    label_free_blocks,
    label_free_metrics,
)
from footfall_heatmaps.evaluation.domain.metrics import compare_layer
from footfall_heatmaps.floorplan.domain.foot_point import camera_floor_points

LABEL_FREE = ("conservation_error", "interior_birth_fraction", "mean_track_length_s")
TRUE_ERRORS = ("tvd", "mass_error")


@dataclass(frozen=True)
class SceneLabelFree:
    scene: str
    camera: str
    label_free: LabelFree
    true_error: dict[str, float]  # visible@coverage dwell, whole run
    blocks: int  # blocks with truth on the coverage mask


@dataclass(frozen=True)
class LabelFreeValidation:
    scenes: list[SceneLabelFree]
    blocks: dict[str, list[float]]  # per scored block: each label-free metric and true error
    correlations: dict[str, dict[str, dict]]  # level -> "metric~error" -> spearman, pearson, n


def validate_label_free(reader, scenes, config, source, cell):
    """scenes: [(scene, camera)]; source: the config's TrackSource."""
    rows, pooled = [], {k: [] for k in (*LABEL_FREE, *TRUE_ERRORS)}
    for scene, camera in scenes:
        gt = build_ground_truth_grid(reader, scene, camera, cell)
        cam = reader.camera(scene, camera)
        truth = reader.floor_truth(scene, camera)
        first, last = int(truth.frames.min()), int(truth.frames.max())
        block_frames = round(BLOCK_S * cam.fps)
        n_blocks = last // block_frames + 1
        t = source.tracks(scene, camera)
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
        keep = (t.frames >= first) & (t.frames <= last)
        ids, frames, boxes = t.track_ids[keep], t.frames[keep], t.boxes[keep]
        spec, v = gt.full.spec, truth.visible
        pred = block_layers(spec, ids, frames, xs[keep], ys[keep], cam.fps, block_frames, n_blocks)
        seen = block_layers(
            spec, truth.track_ids[v], truth.frames[v], *truth.world[v].T, cam.fps,
            block_frames, n_blocks,
        )  # fmt: skip
        whole = compare_layer(pred.dwell.sum(0), seen.dwell.sum(0), gt.coverage, cell)
        lf = label_free_metrics(ids, frames, boxes, cam.width, cam.height, cam.fps, first, last)
        per_block = label_free_blocks(
            ids, frames, boxes, cam.width, cam.height, cam.fps, block_frames, n_blocks, first, last
        )
        scored = 0
        for k in range(n_blocks):
            if not seen.dwell[k][gt.coverage].any():
                continue  # nobody visible in this block: no true error
            m = compare_layer(pred.dwell[k], seen.dwell[k], gt.coverage, cell, zones=False)
            for name in TRUE_ERRORS:
                pooled[name].append(getattr(m, name))
            for name in LABEL_FREE:
                pooled[name].append(float(getattr(per_block, name)[k]))
            scored += 1
        error = {name: getattr(whole, name) for name in TRUE_ERRORS}
        rows.append(SceneLabelFree(scene, camera, lf, error, scored))
    by_scene = {
        k: [getattr(r.label_free, k) for r in rows] if k in LABEL_FREE else
        [r.true_error[k] for r in rows]
        for k in pooled
    }  # fmt: skip
    return LabelFreeValidation(
        scenes=rows,
        blocks=pooled,
        correlations={
            level: {
                f"{m}~{e}": correlation(np.array(data[m]), np.array(data[e]))
                for m in LABEL_FREE
                for e in TRUE_ERRORS
            }
            for level, data in (("scene", by_scene), ("block", pooled))
        },
    )
