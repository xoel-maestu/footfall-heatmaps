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

from collections import deque
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


# The accepted stitch (improvements plan phase 2, G10 R1), which the k test of
# persisted slots counts chains with (phase 6, H3).
K_STITCH = Stitch(gap_s=10.0, radius_m=1.0, v_max_mps=1.5)


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
    # placed point and the earliest placed point within the velocity window.
    placed = np.flatnonzero(np.isfinite(xy).all(axis=1))
    placed = placed[np.lexsort((frames[placed], inv[placed]))]
    pt, pf, pxy = inv[placed], frames[placed], xy[placed]
    has = np.zeros(n, dtype=bool)
    has[pt] = True
    cut = np.flatnonzero(np.diff(pt)) + 1
    head, tail = np.r_[0, cut], np.r_[cut, len(pt)] - 1
    ends = _Ends.empty(n)
    k = pt[head]
    ends.first[:], ends.last[:], ends.has[:] = first, last, has
    ends.start_f[k], ends.end_f[k] = pf[head], pf[tail]
    ends.start_xy[k], ends.end_xy[k] = pxy[head], pxy[tail]
    # Earliest placed point of each track within the window before its last one.
    window = s.velocity_window_s * fps
    in_window = pf >= ends.end_f[pt] - window
    earliest = np.full(n, len(pt))
    np.minimum.at(earliest, pt[in_window], np.flatnonzero(in_window))
    e = earliest[k]
    ends.early_f[k], ends.early_xy[k] = pf[e], pxy[e]
    return tracks[_link(ends, fps, s)][inv]


@dataclass
class _Ends:
    """What stitching reads of each track: its first and last row frame, and
    of its placed rows the first, the last, and the earliest one within the
    velocity window before the last (has: it has a placed row)."""

    first: np.ndarray
    last: np.ndarray
    has: np.ndarray
    start_f: np.ndarray
    end_f: np.ndarray
    early_f: np.ndarray
    start_xy: np.ndarray
    end_xy: np.ndarray
    early_xy: np.ndarray

    @classmethod
    def empty(cls, n):
        ints = [np.zeros(n, dtype=np.int64) for _ in range(5)]
        return cls(ints[0], ints[1], np.zeros(n, dtype=bool), ints[2], ints[3], ints[4],
                   np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2)))  # fmt: skip


def _link(ends, fps, s):
    """Per track (in the order of ends), the index of its chain's first track."""
    n = len(ends.first)
    first, last, has = ends.first, ends.last, ends.has
    start_f, end_f, start_xy, end_xy = ends.start_f, ends.end_f, ends.start_xy, ends.end_xy
    velocity = np.zeros((n, 2))
    k = np.flatnonzero(has)
    dt = (end_f[k] - ends.early_f[k]) / fps
    moving = dt > 0
    velocity[k[moving]] = (end_xy[k][moving] - ends.early_xy[k][moving]) / dt[moving, None]
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
    return label


class ChainBuilder:
    """The stitch of one slot, built as its rows arrive (k on chains, phase 6 H3).

    Keeps per track only what stitch() reads: its first and last row frame,
    its first placed point, and its placed points of the last velocity window.
    chains() then gives the same chains as stitch() over all the rows, as long
    as each track's rows come in frame order. It lives in memory with the open
    slot and is dropped when the slot closes; nothing of it is written."""

    def __init__(self, s, fps):
        self.s, self.fps = s, fps
        self.window = s.velocity_window_s * fps
        self._tracks = {}  # id -> [first, last, start (frame, xy) or None, recent deque]

    def add(self, track_ids, frames, xy):
        """Rows: track ids, integer frames at fps, (N, 2) floor metres (NaN: not placed)."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        placed = np.isfinite(xy).all(axis=1)
        for i, f, p, ok in zip(np.asarray(track_ids).tolist(), np.asarray(frames).tolist(),
                               xy, placed.tolist(), strict=True):  # fmt: skip
            t = self._tracks.get(i)
            if t is None:
                t = self._tracks[i] = [f, f, None, deque()]
            t[0], t[1] = min(t[0], f), max(t[1], f)
            if ok:
                if t[2] is None:
                    t[2] = (f, p)
                recent = t[3]
                recent.append((f, p))
                while recent[0][0] < f - self.window:
                    recent.popleft()

    def chains(self):
        """Track id -> the id of its chain's first track."""
        tracks = sorted(self._tracks)
        if not tracks:
            return {}
        ends = _Ends.empty(len(tracks))
        for j, i in enumerate(tracks):
            first, last, start, recent = self._tracks[i]
            ends.first[j], ends.last[j] = first, last
            if start is not None:
                ends.has[j] = True
                ends.start_f[j], ends.start_xy[j] = start
                ends.end_f[j], ends.end_xy[j] = recent[-1]
                ends.early_f[j], ends.early_xy[j] = recent[0]
        label = _link(ends, self.fps, self.s)
        return {i: tracks[label[j]] for j, i in enumerate(tracks)}
