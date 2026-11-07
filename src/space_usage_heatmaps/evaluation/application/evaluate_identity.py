"""Multi-camera IDF1 on the floor (E2b) and the passage layer of global ids.

Every calibrated camera of a scene runs the same recipe; its tracks become
floor points (the recipe's foot point). occupancy.domain.association turns
the cameras' track ids into global ids, and each global id's points of one
frame are merged into one (their mean): the multi-camera prediction.

Truth: the people at least one of the cameras has a visible box of, on the
joint coverage (cells whose centre at least one camera sees); predictions
are kept on the same cells and frames. The score is TrackEval's IDF1 with a
floor-distance gate (the injected scorer, evaluation.infrastructure.
floor_identity), per scene, with the paired block bootstrap of
identity_bootstrap; pooled = mean over scenes; delta = first - second
association.

oracle: a reference that joins every camera track to the true person it is
nearest to (within the gate) in most of its frames: what perfect association
of these same tracks would score. It reads the truth: never a method.

Passage side effect: the first association's merged points give a passage
layer of distinct global ids per cell. It is scored against the E2a fused map
(occupancy.domain.fusion, whose passage is a weighted mix of the cameras'
own track counts) on full@joint and visible@joint. Both share the E2a fused
dwell, which does not depend on ids.

Global ids live only in memory here: nothing per person is returned but
counts and the fused grid (R2, R4).
"""

from dataclasses import dataclass

import numpy as np

from space_usage_heatmaps.evaluation.application.evaluate_config import BLOCK_S
from space_usage_heatmaps.evaluation.application.evaluate_oracle import camera_floor_points
from space_usage_heatmaps.evaluation.domain.bootstrap import (
    BlockLayers,
    IdentityBlocks,
    SceneBlocks,
    block_bootstrap,
    block_layers,
    identity_bootstrap,
)
from space_usage_heatmaps.floorplan.domain.homography import Homography
from space_usage_heatmaps.occupancy.domain.association import (
    CameraTracks,
    associate,
    merge_points,
)
from space_usage_heatmaps.occupancy.domain.fusion import (
    CameraGrid,
    fuse_layers,
    fusion_weights,
    joint_coverage,
)
from space_usage_heatmaps.occupancy.domain.grid import Grid, GridSpec

GATE_M = 1.0
PASSAGE_TRUTHS = ("full@joint", "visible@joint")


@dataclass(frozen=True)
class IdentityScene:
    scene: str
    cameras: tuple[str, ...]
    blocks: int
    joint_cells: int
    truth_people: int  # distinct people in the scored truth
    truth_points: int  # scored truth detections
    camera_tracks: int  # tracks of every camera
    global_ids: dict[str, int]  # per prediction, ids in the scored frames and cells
    counts: dict[str, dict[str, float]]  # per prediction: idf1, idp, idr


@dataclass(frozen=True)
class IdentityReport:
    dataset: str
    names: tuple[str, ...]  # predictions, in order; delta = names[0] - names[1]
    gate_m: float
    cell_m: float
    block_s: float
    n_resamples: int
    scenes: list[IdentityScene]
    idf1: object  # IdentityResult
    passage: dict[str, object]  # truth -> BootstrapResult of (global-id map, E2a fused map)
    fused: dict[str, Grid]  # per scene: E2a fused dwell, global-id passage


def _camera(reader, scene, camera, config, source, spec):
    cam = reader.camera(scene, camera)
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
    coverage = Homography(cam.world_to_image).sees(*spec.cell_centres(), cam.width, cam.height)
    return (
        cam.fps,
        CameraTracks(str(camera), t.track_ids, t.frames, np.column_stack([xs, ys])),
        coverage,
    )


def _on(spec, mask, xy):
    """Which points fall on a cell of mask."""
    cell, inside = spec.cell_index(xy[:, 0], xy[:, 1])
    on = np.zeros(len(xy), dtype=bool)
    on[np.flatnonzero(inside)] = mask.ravel()[cell]
    return on


def oracle_ids(cameras, truth_ids, truth_frames, truth_xy, gate_m):
    """Global ids that join each camera track to the true person nearest to it
    (within gate_m) in most of its placed frames; a track without such a
    person keeps its own id."""
    out, offset = [], int(truth_ids.max()) + 1 if len(truth_ids) else 0
    order = np.argsort(truth_frames, kind="stable")
    tf, tid, txy = truth_frames[order], truth_ids[order], truth_xy[order]
    for cam in cameras:
        label = np.full(len(cam.frames), -1)
        lo = np.searchsorted(tf, cam.frames, "left")
        hi = np.searchsorted(tf, cam.frames, "right")
        for i in np.flatnonzero((hi > lo) & np.isfinite(cam.xy).all(axis=1)):
            d = np.linalg.norm(txy[lo[i] : hi[i]] - cam.xy[i], axis=1)
            k = np.argmin(d)
            if d[k] <= gate_m:
                label[i] = tid[lo[i] + k]
        gid = np.empty(len(cam.frames), dtype=np.int64)
        for track in np.unique(cam.track_ids):
            rows = cam.track_ids == track
            votes = label[rows][label[rows] >= 0]
            placed = np.isfinite(cam.xy[rows]).all(axis=1).sum()
            if len(votes):
                values, counts = np.unique(votes, return_counts=True)
                if counts.max() * 2 >= placed:
                    gid[rows] = values[np.argmax(counts)]
                    continue
            gid[rows] = offset
            offset += 1
        out.append(gid)
    return out


def evaluate_identity(
    reader,
    scenes,
    run,
    fusion,
    associations,
    scorer,
    cell=0.5,
    gate_m=GATE_M,
    block_s=BLOCK_S,
    n_resamples=1000,
    seed=0,
    cameras=None,
    oracle=False,
    passage_resamples=None,
    map=map,
):
    """scenes: scene ids; run: (RunConfig, TrackSource) of every camera;
    associations: name -> Association, at least two (delta = first - second);
    scorer: floor_identity's signature; cameras: default every calibrated one;
    passage_resamples: the passage bootstrap's (default n_resamples; its EMD
    makes each resample slow, so a sweep may score point values only)."""
    if len(associations) < 2:
        raise ValueError("two associations at least: the method and its baseline")
    names = (*associations, *(("oracle",) if oracle else ()))
    config, source = run
    report_scenes, idf1_blocks, fused_grids = [], [], {}
    passage_blocks = {t: [] for t in PASSAGE_TRUTHS}
    for scene in scenes:
        ids = [str(c) for c in (cameras or reader.cameras(scene))]
        spec = GridSpec.covering(*reader.scene_extent(scene), cell_m=cell)
        cams, covers, rates = [], [], set()
        for c in ids:
            fps, cam, coverage = _camera(reader, scene, c, config, source, spec)
            cams.append(cam)
            covers.append(coverage)
            rates.add(fps)
        if len(rates) != 1:
            raise ValueError(f"scene {scene}: cameras run at different frame rates {sorted(rates)}")
        (fps,) = rates
        truth = reader.floor_truth(scene, ids[0])
        seen = np.zeros(len(truth.track_ids), dtype=bool)
        for c in ids:
            seen |= reader.floor_truth(scene, c).visible
        first, last = int(truth.frames.min()), int(truth.frames.max())
        bf = max(1, round(block_s * fps))
        nb = last // bf + 1

        def block_of(frames, bf=bf):
            return np.asarray(frames) // bf

        joint = joint_coverage(
            [CameraGrid(c.camera_id, None, v) for c, v in zip(cams, covers, strict=True)]
        )
        scored = seen & _on(spec, joint, truth.world)
        t_ids, t_frames, t_xy = truth.track_ids[scored], truth.frames[scored], truth.world[scored]

        gids = {name: associate(cams, a, fps) for name, a in associations.items()}
        if oracle:
            gids["oracle"] = oracle_ids(cams, t_ids, t_frames, t_xy, gate_m)
        preds, counts, n_ids = [], {}, {}
        for name in names:
            p_ids, p_frames, p_xy = merge_points(cams, gids[name])
            keep = (p_frames >= first) & (p_frames <= last) & _on(spec, joint, p_xy)
            r = scorer(
                t_ids,
                t_frames,
                t_xy,
                p_ids[keep],
                p_frames[keep],
                p_xy[keep],
                gate_m=gate_m,
                blocks=nb,
                block_of=block_of,
            )
            preds.append(IdentityBlocks(r.idtp, r.n_truth, r.n_pred))
            counts[name] = {"idf1": r.idf1, "idp": r.idp, "idr": r.idr}
            n_ids[name] = int(len(np.unique(p_ids[keep])))
        idf1_blocks.append(tuple(preds))

        # Passage: global ids of the first association against the E2a fused map.
        per_camera = []
        for cam in cams:
            ok = np.isfinite(cam.xy).all(axis=1) & (cam.frames >= first) & (cam.frames <= last)
            per_camera.append(
                block_layers(spec, cam.track_ids[ok], cam.frames[ok], *cam.xy[ok].T, fps, bf, nb)
            )
        grids = [
            CameraGrid(c.camera_id, Grid(spec, b.dwell.sum(0), b.passage.sum(0), b.dropped), v)
            for c, b, v in zip(cams, per_camera, covers, strict=True)
        ]
        w = fusion_weights(grids, fusion)
        fused_dwell = fuse_layers(w, [b.dwell for b in per_camera])
        e2a = BlockLayers(fused_dwell, fuse_layers(w, [b.passage for b in per_camera]))
        g_ids, g_frames, g_xy = merge_points(cams, gids[names[0]])
        ok = (g_frames >= first) & (g_frames <= last)
        mine = block_layers(spec, g_ids[ok], g_frames[ok], *g_xy[ok].T, fps, bf, nb)
        e2b = BlockLayers(fused_dwell, mine.passage, mine.dropped)
        fused_grids[scene] = Grid(spec, fused_dwell.sum(0), mine.passage.sum(0), mine.dropped)

        def truth_layers(rows, t=truth, spec=spec, fps=fps, bf=bf, nb=nb):
            return block_layers(
                spec, t.track_ids[rows], t.frames[rows], *t.world[rows].T, fps, bf, nb
            )

        full = truth_layers(np.ones(len(truth.track_ids), dtype=bool))
        passage_blocks["full@joint"].append(SceneBlocks(full, (e2b, e2a), joint, cell))
        passage_blocks["visible@joint"].append(
            SceneBlocks(truth_layers(seen), (e2b, e2a), joint, cell)
        )

        report_scenes.append(
            IdentityScene(
                scene=scene,
                cameras=tuple(ids),
                blocks=nb,
                joint_cells=int(joint.sum()),
                truth_people=int(len(np.unique(t_ids))),
                truth_points=int(len(t_ids)),
                camera_tracks=int(sum(len(np.unique(c.track_ids)) for c in cams)),
                global_ids=n_ids,
                counts=counts,
            )
        )
    return IdentityReport(
        dataset=reader.name,
        names=names,
        gate_m=gate_m,
        cell_m=cell,
        block_s=block_s,
        n_resamples=n_resamples,
        scenes=report_scenes,
        idf1=identity_bootstrap(idf1_blocks, n_resamples=n_resamples, seed=seed),
        passage={
            t: block_bootstrap(
                b,
                n_resamples=n_resamples if passage_resamples is None else passage_resamples,
                seed=seed,
                map=map,
            )
            for t, b in passage_blocks.items()
        },
        fused=fused_grids,
    )
