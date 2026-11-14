"""Join the tracklets of one camera across short gaps, on the floor (phase 2).

A tracker loses a person behind a shelf and starts a new track when they come
back: one person becomes several track ids, and passage (distinct ids per
cell) counts them all. Stitching joins a track that ends to one that starts
later in the same camera when:
- the gap from the last frame of the first to the first frame of the second
  is at most gap_s (so the two are never alive in the same frame);
- the second starts within radius_m + v_max_mps x gap of where the first
  would be at constant velocity: its last floor point moved on by its velocity
  over its last velocity_window_s, the speed capped at v_max_mps so that the
  foot-point jitter of a short track cannot throw the prediction far;
- the pair is the nearest left: candidate pairs are taken by increasing
  distance to the prediction, each track taking at most one successor and one
  predecessor.

It only relabels: every row keeps its frame, box and floor point, and each
chain of joined tracks takes its first track's id. Dwell does not read ids,
so it is unchanged. Only floor points and time are used: no image, no
appearance feature (R3); the ids live only in memory (R2).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Stitch:
    gap_s: float
    radius_m: float
    v_max_mps: float = 1.5
    velocity_window_s: float = 1.0

    def __post_init__(self):
        if not self.gap_s > 0:
            raise ValueError("stitch gap_s must be positive")
        if not self.velocity_window_s > 0:
            raise ValueError("stitch velocity_window_s must be positive")
        for name in ("radius_m", "v_max_mps"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"stitch {name} must be 0 or more")


def stitch(track_ids, frames, xy, fps, s):
    """New track ids, one per row: the id of the first track of each joined
    chain. xy: (N, 2) floor metres, NaN where a box could not be placed."""
    ids = np.asarray(track_ids)
    if not len(ids):
        return ids.copy()
    frames = np.asarray(frames)
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    tracks, inv = np.unique(ids, return_inverse=True)
    inv = inv.ravel()
    n = len(tracks)
    first = np.full(n, np.iinfo(np.int64).max)
    last = np.full(n, np.iinfo(np.int64).min)
    np.minimum.at(first, inv, frames)
    np.maximum.at(last, inv, frames)

    # Floor points in (track, frame) order: per track, its first and last
    # placed point and its velocity over the last window.
    placed = np.flatnonzero(np.isfinite(xy).all(axis=1))
    placed = placed[np.lexsort((frames[placed], inv[placed]))]
    pt, pf, pxy = inv[placed], frames[placed], xy[placed]
    has = np.zeros(n, dtype=bool)
    has[pt] = True
    cut = np.flatnonzero(np.diff(pt)) + 1
    head, tail = np.r_[0, cut], np.r_[cut, len(pt)] - 1
    start_f = np.zeros(n, dtype=np.int64)
    end_f = np.zeros(n, dtype=np.int64)
    start_xy, end_xy, velocity = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2))
    k = pt[head]
    start_f[k], end_f[k] = pf[head], pf[tail]
    start_xy[k], end_xy[k] = pxy[head], pxy[tail]
    # Earliest placed point of each track within the window before its last one.
    window = s.velocity_window_s * fps
    in_window = pf >= end_f[pt] - window
    earliest = np.full(n, len(pt))
    np.minimum.at(earliest, pt[in_window], np.flatnonzero(in_window))
    e = earliest[k]
    dt = (pf[tail] - pf[e]) / fps
    moving = dt > 0
    velocity[k[moving]] = (pxy[tail][moving] - pxy[e][moving]) / dt[moving, None]
    speed = np.linalg.norm(velocity, axis=1)
    fast = speed > s.v_max_mps
    velocity[fast] *= (s.v_max_mps / speed[fast])[:, None]

    # Candidate pairs [ender, starter]: a later start within gap_s, both placed.
    starters = np.flatnonzero(has)
    starters = starters[np.argsort(first[starters], kind="stable")]
    enders = np.flatnonzero(has)
    lo = np.searchsorted(first[starters], last[enders], "right")
    hi = np.searchsorted(first[starters], last[enders] + s.gap_s * fps, "right")
    count = hi - lo
    a = np.repeat(enders, count)
    offset = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    b = starters[np.repeat(lo, count) + offset]
    t = (start_f[b] - end_f[a]) / fps
    predicted = end_xy[a] + velocity[a] * t[:, None]
    d = np.linalg.norm(start_xy[b] - predicted, axis=1)
    ok = d <= s.radius_m + s.v_max_mps * t
    a, b, d = a[ok], b[ok], d[ok]

    successor = np.full(n, -1)
    has_predecessor = np.zeros(n, dtype=bool)
    for j in np.argsort(d, kind="stable"):
        if successor[a[j]] < 0 and not has_predecessor[b[j]]:
            successor[a[j]] = b[j]
            has_predecessor[b[j]] = True

    label = np.arange(n)
    for h in np.flatnonzero(~has_predecessor):
        k = successor[h]
        while k >= 0:
            label[k] = h
            k = successor[k]
    return tracks[label][inv]
