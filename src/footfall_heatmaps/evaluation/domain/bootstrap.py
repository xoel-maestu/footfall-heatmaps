"""Paired block bootstrap of the heatmap metrics.

The run is cut into blocks of consecutive frames (30 s), each with its own
dwell and passage grid, so that a resample is a sum of block grids. A resample
draws each scene's blocks with replacement, the same draw for every
prediction: differences between two predictions are paired. Pooled values are
the unweighted mean over scenes of the per-scene metric, with each scene's
blocks resampled within that scene.

TVD, WAPE, EMD and Pearson are not additive over blocks: a resample that
repeats blocks scores worse than the whole run, so the raw percentile interval
can sit entirely above the whole-run value (TVD 0.638 with [0.645, 0.708] on
scene 071). Intervals are therefore recentred percentile intervals: the
percentiles of the draws shifted by their mean offset from the whole-run value.
This is not a bias-corrected (BCa) interval, and the shift can carry a bound
outside the metric's range (a TVD below 0), so each bound of an absolute
interval is clipped to its metric's domain (DOMAINS). Paired differences are
not clipped: a difference can take either sign.
"""

import math
from dataclasses import dataclass

import numpy as np

from footfall_heatmaps.evaluation.domain.metrics import compare_layer

LAYERS = ("dwell", "passage")
METRICS = ("mass_error", "tvd", "wape_zone", "emd_m", "pearson")
# Valid range of each metric, for clipping absolute intervals.
DOMAINS = {
    "mass_error": (0.0, math.inf),
    "tvd": (0.0, 1.0),
    "wape_zone": (0.0, math.inf),
    "emd_m": (0.0, math.inf),
    "pearson": (-1.0, 1.0),
}


@dataclass(frozen=True)
class BlockLayers:
    dwell: np.ndarray  # (blocks, ny, nx) person-seconds
    passage: np.ndarray  # (blocks, ny, nx) tracks counted in the block they entered the cell
    dropped: int = 0  # samples that fell outside the grid


@dataclass(frozen=True)
class SceneBlocks:
    truth: BlockLayers
    preds: tuple[BlockLayers, ...]  # one per configuration, on the same blocks
    mask: np.ndarray  # (ny, nx) cells to score
    cell_m: float


@dataclass(frozen=True)
class Estimate:
    value: float | None  # on the whole run; None when the metric is undefined
    lo: float | None
    hi: float | None


@dataclass(frozen=True)
class BootstrapResult:
    # [scene][pred][layer][metric] and [pred][layer][metric]
    scenes: list[list[dict[str, dict[str, Estimate]]]]
    pooled: list[dict[str, dict[str, Estimate]]]
    # preds[0] - preds[1] when there are two predictions, else None
    delta_scenes: list[dict[str, dict[str, Estimate]]] | None
    delta_pooled: dict[str, dict[str, Estimate]] | None


def block_layers(spec, track_ids, frames, xs, ys, fps, block_frames, n_blocks):
    """Dwell and passage per block; summed over blocks they equal the whole
    run's Grid. Passage counts a track in a cell in the block of its first
    sample there, so a stay that crosses a block boundary is counted once."""
    frames = np.asarray(frames)
    block = frames // block_frames
    if np.any(block >= n_blocks) or np.any(block < 0):
        raise ValueError(f"sample frame outside the {n_blocks} blocks of {block_frames} frames")
    col = np.floor((np.asarray(xs, float) - spec.origin_x) / spec.cell_m).astype(int)
    row = np.floor((np.asarray(ys, float) - spec.origin_y) / spec.cell_m).astype(int)
    inside = (col >= 0) & (col < spec.nx) & (row >= 0) & (row < spec.ny)
    cells = spec.ny * spec.nx
    cell = row[inside] * spec.nx + col[inside]
    block, frames = block[inside], frames[inside]
    ids = np.asarray(track_ids)[inside]
    dwell = np.bincount(block * cells + cell, minlength=n_blocks * cells) / fps
    # First sample of each (track, cell): sort by frame, keep the first occurrence.
    order = np.argsort(frames, kind="stable")
    _, first = np.unique(np.stack([ids[order], cell[order]]), axis=1, return_index=True)
    entered = order[first]
    passage = np.bincount(block[entered] * cells + cell[entered], minlength=n_blocks * cells)
    shape = (n_blocks, spec.ny, spec.nx)
    dropped = int((~inside).sum())
    return BlockLayers(dwell.reshape(shape), passage.astype(float).reshape(shape), dropped)


def block_bootstrap(scenes, n_resamples=1000, seed=0, level=0.95, map=map):
    """Estimates with recentred percentile intervals at `level` for every scene, prediction,
    layer and metric, pooled over scenes, and the paired difference of the first
    two predictions. map: how scenes are scored (e.g. a process pool's map); the
    draws are made here first, so the result does not depend on it."""
    rng = np.random.default_rng(seed)
    n_preds = len(scenes[0].preds)
    if any(len(s.preds) != n_preds for s in scenes):
        raise ValueError("every scene needs the same predictions")
    draws = [_block_counts(rng, s.truth.dwell.shape[0], n_resamples) for s in scenes]
    # values[draw, scene, pred, layer, metric]; draw 0 is the whole run.
    values = np.stack(list(map(scene_values, scenes, draws)), axis=1)
    pooled = values.mean(axis=1)
    alpha = (1 - level) / 2 * 100
    result = BootstrapResult(
        scenes=[
            [_table(values[:, i, p], alpha, clip=True) for p in range(n_preds)]
            for i in range(len(scenes))
        ],
        pooled=[_table(pooled[:, p], alpha, clip=True) for p in range(n_preds)],
        delta_scenes=None,
        delta_pooled=None,
    )
    if n_preds < 2:
        return result
    delta = values[:, :, 0] - values[:, :, 1]
    return BootstrapResult(
        scenes=result.scenes,
        pooled=result.pooled,
        delta_scenes=[_table(delta[:, i], alpha, clip=False) for i in range(len(scenes))],
        delta_pooled=_table(delta.mean(axis=1), alpha, clip=False),
    )


def _block_counts(rng, n_blocks, n_resamples):
    """counts[draw, block]: how many times each block is drawn; draw 0 is the
    whole run (every block once)."""
    counts = np.zeros((n_resamples + 1, n_blocks))
    counts[0] = 1
    picks = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    np.add.at(counts, (np.arange(1, n_resamples + 1)[:, None], picks), 1)
    return counts


@dataclass(frozen=True)
class IdentityBlocks:
    """A prediction's identity counts per block (evaluation.infrastructure.floor_identity)."""

    idtp: np.ndarray  # (blocks,)
    n_truth: np.ndarray  # (blocks,)
    n_pred: np.ndarray  # (blocks,)


@dataclass(frozen=True)
class IdentityResult:
    scenes: list[list[Estimate]]  # [scene][pred] IDF1
    pooled: list[Estimate]  # [pred], mean over scenes
    delta_scenes: list[Estimate] | None  # preds[0] - preds[1]
    delta_pooled: Estimate | None


def identity_bootstrap(scenes, n_resamples=1000, seed=0, level=0.95):
    """IDF1 of each prediction per scene and pooled (mean over scenes), with the
    paired block bootstrap of block_bootstrap (same draws for a given seed and
    block counts). scenes: [(IdentityBlocks per prediction)]. Each resample
    sums the drawn blocks' counts: the truth-to-prediction assignment stays
    the whole run's, so the interval leaves out the assignment's own variance."""
    rng = np.random.default_rng(seed)
    alpha = (1 - level) / 2 * 100
    values = []  # [scene] (draws, preds)
    for preds in scenes:
        c = _block_counts(rng, len(preds[0].idtp), n_resamples)
        v = []
        for p in preds:
            total = c @ p.n_truth + c @ p.n_pred
            v.append(np.divide(2 * (c @ p.idtp), total, out=np.zeros(len(c)), where=total > 0))
        values.append(np.column_stack(v))
    values = np.stack(values, axis=1)  # (draws, scenes, preds)
    n_preds = values.shape[2]
    pooled = values.mean(axis=1)
    unit = (0.0, 1.0)
    result = IdentityResult(
        scenes=[
            [estimate(values[:, i, p], alpha, unit) for p in range(n_preds)]
            for i in range(values.shape[1])
        ],
        pooled=[estimate(pooled[:, p], alpha, unit) for p in range(n_preds)],
        delta_scenes=None,
        delta_pooled=None,
    )
    if n_preds < 2:
        return result
    delta = values[:, :, 0] - values[:, :, 1]
    return IdentityResult(
        result.scenes,
        result.pooled,
        [estimate(delta[:, i], alpha) for i in range(delta.shape[1])],
        estimate(delta.mean(axis=1), alpha),
    )


def scene_values(s, counts):
    """values[draw, pred, layer, metric] of one scene for block counts[draw, block]."""
    values = np.full((len(counts), len(s.preds), len(LAYERS), len(METRICS)), np.nan)
    for r, c in enumerate(counts):
        for li, layer in enumerate(LAYERS):
            truth = np.tensordot(c, getattr(s.truth, layer), axes=1)
            if not np.any(truth[s.mask]):
                continue  # nothing to score in this draw: NaN
            for p, pred in enumerate(s.preds):
                grid = np.tensordot(c, getattr(pred, layer), axes=1)
                m = compare_layer(grid, truth, s.mask, s.cell_m, zones=layer == "dwell")
                values[r, p, li] = [_number(getattr(m, k)) for k in METRICS]
    return values


def _number(v):
    return math.nan if v is None else v


def _table(v, alpha, clip):
    """v[draw, layer, metric] -> {layer: {metric: Estimate}}; draw 0 is the value.
    With clip, each interval is clipped to its metric's domain."""
    return {
        layer: {
            metric: estimate(v[:, li, mi], alpha, DOMAINS[metric] if clip else None)
            for mi, metric in enumerate(METRICS)
        }
        for li, layer in enumerate(LAYERS)
    }


def estimate(draws, alpha, domain=None):
    """Recentred percentile interval: draws[0] is the whole-run value, the rest
    are resamples, shifted so that their mean is the value. domain (lo, hi)
    clips the bounds."""
    value, rest = draws[0], draws[1:]
    rest = rest[np.isfinite(rest)]
    if not math.isfinite(value) or rest.size == 0:
        return Estimate(None if not math.isfinite(value) else float(value), None, None)
    offset = rest.mean() - value
    lo, hi = np.percentile(rest - offset, [alpha, 100 - alpha])
    if domain is not None:
        lo, hi = np.clip([lo, hi], *domain)
    return Estimate(float(value), float(lo), float(hi))
