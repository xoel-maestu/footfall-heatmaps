"""Identity-free passage by chord length (improvements plan phase 6, H2).

Distinct ids per cell count a whole crossing for every tracklet fragment that
touches a cell, so a track the tracker splits, or a foot point that jitters at
a cell edge, adds a full person. Here each tracklet adds
min(1, path length inside the cell / mean chord) to a cell. By Cauchy's
formula the mean chord of a convex cell under uniform random lines is
pi * area / perimeter, pi * a / 4 for a square of side a (0.39 m at 0.5 m).

The path is the tracklet's floor polyline: its samples joined in frame order
(filled frames included), each segment clipped exactly against the cell
edges. A stationary tracklet has no path length and adds 0; a single sample
too. The cap of 1 bounds a tracklet that wanders inside one cell. Nothing
joins tracklets: no identity is needed.
"""

import math

import numpy as np


def mean_chord(cell_m):
    """Mean chord of a square cell of side cell_m: pi * cell_m / 4."""
    return math.pi * cell_m / 4


def chord_passage(spec, track_ids, frames, xs, ys):
    """(cells, frames, weights): one row per (track, cell) the track's path
    crosses inside the grid, with the flat cell index, the frame of the sample
    that starts the first segment entering that cell, and
    min(1, length inside / mean_chord)."""
    ids = np.asarray(track_ids)
    frames = np.asarray(frames)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    order = np.lexsort((frames, ids))
    ids, frames, xs, ys = ids[order], frames[order], xs[order], ys[order]
    # Segments between consecutive samples of one track, in grid units.
    c = spec.cell_m
    u, v = (xs - spec.origin_x) / c, (ys - spec.origin_y) / c
    seg = np.flatnonzero(ids[1:] == ids[:-1])
    u0, v0, u1, v1 = u[seg], v[seg], u[seg + 1], v[seg + 1]
    ok = np.isfinite(u0) & np.isfinite(v0) & np.isfinite(u1) & np.isfinite(v1)
    seg, u0, v0, u1, v1 = seg[ok], u0[ok], v0[ok], u1[ok], v1[ok]
    # Liang-Barsky: the part of each segment inside [0, nx] x [0, ny].
    du, dv = u1 - u0, v1 - v0
    t0, t1 = np.zeros(len(seg)), np.ones(len(seg))
    for p, q in ((-du, u0), (du, spec.nx - u0), (-dv, v0), (dv, spec.ny - v0)):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = q / p
        t0 = np.where(p < 0, np.maximum(t0, r), t0)
        t1 = np.where(p > 0, np.minimum(t1, r), t1)
        t1 = np.where((p == 0) & (q < 0), -1.0, t1)  # parallel and outside
    keep = t1 > t0
    seg, du, dv, t0, t1 = seg[keep], du[keep], dv[keep], t0[keep], t1[keep]
    u0, v0 = u0[keep], v0[keep]
    # Cut points: the clipped ends and every grid line crossed in between.
    ts, owner = [t0, t1], [np.arange(len(seg))] * 2
    for d, start in ((du, u0), (dv, v0)):
        a, b = start + t0 * d, start + t1 * d
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        first = np.floor(lo).astype(np.int64) + 1
        count = np.maximum(np.ceil(hi).astype(np.int64) - first, 0)
        who = np.repeat(np.arange(len(seg)), count)
        offset = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        line = first[who] + offset
        ts.append((line - start[who]) / d[who])
        owner.append(who)
    ts, owner = np.concatenate(ts), np.concatenate(owner)
    by = np.lexsort((ts, owner))
    ts, owner = ts[by], owner[by]
    piece = np.flatnonzero((owner[1:] == owner[:-1]) & (ts[1:] > ts[:-1]))
    who, ta, tb = owner[piece], ts[piece], ts[piece + 1]
    mid = (ta + tb) / 2
    col = np.clip(np.floor(u0[who] + mid * du[who]).astype(np.int64), 0, spec.nx - 1)
    row = np.clip(np.floor(v0[who] + mid * dv[who]).astype(np.int64), 0, spec.ny - 1)
    length = (tb - ta) * np.hypot(du[who], dv[who]) * c
    moved = length > 0
    cell, who, length = (row * spec.nx + col)[moved], who[moved], length[moved]
    track, start_frame = ids[seg[who]], frames[seg[who]]
    if len(cell) == 0:
        empty = np.empty(0, np.int64)
        return empty, empty, np.empty(0)
    pairs, inverse = np.unique(np.stack([track, cell]), axis=1, return_inverse=True)
    inverse = inverse.reshape(-1)
    total = np.bincount(inverse, weights=length, minlength=pairs.shape[1])
    entered = np.full(pairs.shape[1], np.iinfo(np.int64).max)
    np.minimum.at(entered, inverse, start_frame.astype(np.int64))
    return pairs[1], entered, np.minimum(1.0, total / mean_chord(c))
