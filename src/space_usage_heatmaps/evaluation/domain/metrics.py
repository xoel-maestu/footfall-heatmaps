"""Heatmap fidelity metrics of spec section 6, over a camera's coverage mask."""

import math
from dataclasses import asdict, dataclass

import numpy as np
import ot

ZONE_M = 2.0  # WAPE zones are 2 m blocks
ZONE_MIN_SHARE = 0.01  # zones holding less than 1 % of the truth are ignored
EMD_CELL_M = 1.0  # EMD is computed on a 1 m grid


@dataclass(frozen=True)
class LayerMetrics:
    mass_error: float
    tvd: float
    wape_zone: float | None  # None when undefined; wape_zone_reason says why
    emd_m: float
    pearson: float
    wape_zone_reason: str | None = None

    def as_dict(self):
        return asdict(self)


def compare_grids(pred, truth, mask):
    """Metrics per layer for two Grids on the same spec; WAPE for dwell only."""
    if pred.spec != truth.spec:
        raise ValueError("prediction and truth grids differ in extent or cell size")
    cell = truth.spec.cell_m
    return {
        layer: compare_layer(
            getattr(pred, layer), getattr(truth, layer), mask, cell, zones=layer == "dwell"
        )
        for layer in ("dwell", "passage")
    }


def compare_layer(pred, truth, mask, cell_m, zones=True):
    """An empty mask or an empty truth is an error: there is nothing to score.
    An empty prediction scores total error, with EMD and Pearson NaN."""
    if not np.any(mask):
        raise ValueError("mask is empty: no cell to score")
    p = np.where(mask, pred, 0.0).astype(float)
    t = np.where(mask, truth, 0.0).astype(float)
    sp, st = p.sum(), t.sum()
    if st == 0:
        raise ValueError("truth is empty on the mask: no mass to score against")
    wape, reason = _wape_zone(p, t, cell_m) if zones else (None, "computed for dwell only")
    if sp == 0:
        return LayerMetrics(1.0, 1.0, None if wape is None else 1.0, math.nan, math.nan, reason)
    return LayerMetrics(
        mass_error=float(abs(sp - st) / st),
        tvd=float(0.5 * np.abs(p / sp - t / st).sum()),
        wape_zone=wape,
        emd_m=_emd(_blocks(p, round(EMD_CELL_M / cell_m)), _blocks(t, round(EMD_CELL_M / cell_m))),
        pearson=_pearson(p[mask], t[mask]),
        wape_zone_reason=reason,
    )


def _wape_zone(p, t, cell_m):
    k = round(ZONE_M / cell_m)
    zp, zt = _blocks(p, k), _blocks(t, k)
    kept = zt >= ZONE_MIN_SHARE * t.sum()
    if not kept.any():
        return None, "no 2 m zone holds 1 % of the truth"
    return float(np.abs(zp - zt)[kept].sum() / zt[kept].sum()), None


def _blocks(a, k):
    """Sum k x k blocks of cells, padding the edges with zeros."""
    ny, nx = -(-a.shape[0] // k) * k, -(-a.shape[1] // k) * k
    a = np.pad(a, ((0, ny - a.shape[0]), (0, nx - a.shape[1])))
    return a.reshape(ny // k, k, nx // k, k).sum(axis=(1, 3))


def _emd(p, t):
    """Wasserstein-1 in metres between the normalised maps; cells are EMD_CELL_M."""
    rows, cols = np.nonzero((p > 0) | (t > 0))
    a, b = p[rows, cols] / p.sum(), t[rows, cols] / t.sum()
    xy = np.stack([cols, rows], axis=1) * EMD_CELL_M
    cost = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    return float(ot.emd2(a, b, cost))


def _pearson(p, t):
    if p.std() == 0 or t.std() == 0:
        return math.nan
    return float(np.corrcoef(p, t)[0, 1])
