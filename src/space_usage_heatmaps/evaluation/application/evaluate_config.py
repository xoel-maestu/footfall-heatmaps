"""Score the tracks of one or two configurations against the floor ground truth.

Each tracked box becomes one floor point (the config's foot point, through the
camera matrix; see floorplan.domain.foot_point) and is accumulated in the floor
grid: dwell adds stride/fps person-seconds per sample, passage counts distinct
track ids per cell. The three truth blocks of evaluate_oracle are scored with
a paired block bootstrap over 30 s blocks.
"""

from dataclasses import dataclass

from space_usage_heatmaps.evaluation.application.evaluate_oracle import (
    TRUTHS,
    build_ground_truth_grid,
    camera_floor_points,
)
from space_usage_heatmaps.evaluation.domain.bootstrap import (
    BootstrapResult,
    SceneBlocks,
    block_bootstrap,
    block_layers,
)

BLOCK_S = 30.0


@dataclass(frozen=True)
class SceneRun:
    scene: str
    camera: str
    samples: int  # tracked boxes scored, of the first configuration
    dropped: int  # of those, outside the grid
    outside_truth_frames: int  # tracked boxes in frames the truth does not cover
    blocks: int
    occlusion_fraction: float


@dataclass(frozen=True)
class ConfigReport:
    dataset: str
    cell_m: float
    block_s: float
    n_resamples: int
    scenes: list[SceneRun]
    results: dict[str, BootstrapResult]  # truth block -> bootstrap


def evaluate_config(reader, scenes, runs, cell, n_resamples=1000, seed=0):
    """scenes: [(scene, camera)]; runs: [(RunConfig, TrackSource)], one or two.
    With two, the results carry runs[0] - runs[1]."""
    blocks = {truth: [] for truth in TRUTHS}
    report_scenes = []
    for scene, camera in scenes:
        gt = build_ground_truth_grid(reader, scene, camera, cell)
        spec, cam = gt.full.spec, reader.camera(scene, camera)
        truth = reader.floor_truth(scene, camera)
        first, last = int(truth.frames.min()), int(truth.frames.max())
        block_frames = round(BLOCK_S * cam.fps)
        n_blocks = last // block_frames + 1

        def layers(ids, frames, xs, ys, fps, spec=spec, bf=block_frames, nb=n_blocks):
            return block_layers(spec, ids, frames, xs, ys, fps, bf, nb)

        v = truth.visible
        full = layers(truth.track_ids, truth.frames, *truth.world.T, cam.fps)
        visible = layers(truth.track_ids[v], truth.frames[v], *truth.world[v].T, cam.fps)
        preds, counts = [], []
        for config, source in runs:
            t = source.tracks(scene, camera)
            keep = (t.frames >= first) & (t.frames <= last)
            xs, ys, _ = camera_floor_points(
                cam,
                t.boxes[keep],
                config.foot_point,
                config.person_height_m,
                config.min_implied_height_m,
                t.track_ids[keep],
                t.frames[keep],
                config.track_heights(),
            )
            fps = cam.fps / config.stride
            preds.append(layers(t.track_ids[keep], t.frames[keep], xs, ys, fps))
            counts.append((int(keep.sum()), preds[-1].dropped, int((~keep).sum())))
        for name, tr, mask in (
            ("visible@coverage", visible, gt.coverage),
            ("full@coverage", full, gt.coverage),
            ("visible@visibility", visible, gt.visibility),
        ):
            blocks[name].append(SceneBlocks(tr, tuple(preds), mask, cell))
        seen = gt.visible.dwell[gt.coverage].sum()
        total = gt.full.dwell[gt.coverage].sum()
        samples, dropped, outside = counts[0]
        report_scenes.append(
            SceneRun(scene, camera, samples, dropped, outside, n_blocks, float(1 - seen / total))
        )
    return ConfigReport(
        dataset=reader.name,
        cell_m=cell,
        block_s=BLOCK_S,
        n_resamples=n_resamples,
        scenes=report_scenes,
        results={
            truth: block_bootstrap(blocks[truth], n_resamples=n_resamples, seed=seed)
            for truth in TRUTHS
        },
    )
