"""Cross-camera identities from floor geometry alone (E2b).

Each camera's tracker gives its own track ids. Two tracks of different cameras
that stand on the same floor spot, frame after frame, are the same person; a
global id joins them. Only floor points and time are used: no image, no
appearance feature (R3).

Methods:
- none: every camera track is its own global id (the baseline);
- colocate: two tracks of different cameras are linked when they are seen in
  the same frames for at least min_overlap_s and, in at least min_close of
  those frames, their floor points are at most gate_m apart;
- colocate-mnn: as colocate, but a frame counts as close only when the two
  points are each other's nearest point in the other camera (one-to-one per
  frame and camera pair, as a per-frame assignment would make them).
Links are taken best first (largest share of close frames, then the smallest
mean distance in them) and joined into clusters unless the join would put
together two tracks of one camera alive at the same time, or two tracks seen
in the same frames for min_overlap_s but close in fewer than min_close of
them (two people in different places).

handover_s (optional): after the co-location, a cluster that starts at most
handover_s after another one ended, within handover_radius_m plus
handover_speed_mps times the gap of where it ended, continues it (each end and
each start used once, nearest first), under the same rule against two tracks of
one camera alive at the same time. It joins handovers between cameras that
do not overlap and fragments of one camera's track.

Privacy: the global ids exist only in memory during evaluation; nothing here
is stored (R2, R4).
"""

import math
from dataclasses import dataclass

import numpy as np

METHODS = ("none", "colocate", "colocate-mnn")
_CHUNK_PAIRS = 4_000_000  # cross-camera pairs accumulated before each bincount


@dataclass(frozen=True)
class CameraTracks:
    """One camera's tracks on the floor: one row per tracked box."""

    camera_id: str
    track_ids: np.ndarray  # (N,)
    frames: np.ndarray  # (N,)
    xy: np.ndarray  # (N, 2) floor metres; NaN where the box could not be placed


@dataclass(frozen=True)
class Association:
    method: str = "colocate"
    gate_m: float = 1.0
    min_overlap_s: float = 1.0
    min_close: float = 0.5
    handover_s: float | None = None
    handover_speed_mps: float = 1.5
    handover_radius_m: float = 0.5

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"association method must be one of {METHODS}: {self.method!r}")
        for name in ("gate_m", "min_overlap_s", "handover_speed_mps", "handover_radius_m"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"association {name} must be 0 or more")
        if not 0 < self.min_close <= 1:
            raise ValueError("association min_close must be in (0, 1]")
        if self.handover_s is not None and not self.handover_s > 0:
            raise ValueError("association handover_s must be positive")


NONE = Association("none")


def _track_index(cameras):
    """Per camera, the global track index of each row; per track, its camera,
    first and last frame."""
    rows, cams, first, last, n = [], [], [], [], 0
    for c, cam in enumerate(cameras):
        uid, inv = np.unique(np.asarray(cam.track_ids), return_inverse=True)
        rows.append(inv + n)
        frames = np.asarray(cam.frames)
        f0 = np.full(len(uid), np.iinfo(np.int64).max)
        f1 = np.full(len(uid), np.iinfo(np.int64).min)
        np.minimum.at(f0, inv, frames)
        np.maximum.at(f1, inv, frames)
        cams.append(np.full(len(uid), c))
        first.append(f0)
        last.append(f1)
        n += len(uid)
    cat = np.concatenate
    return rows, cat(cams) if cams else np.zeros(0, int), cat(first), cat(last), n


def _pair_stats(cameras, rows, track_cam, n, association):
    """co[i, j]: frames both tracks have a floor point; close[i, j]: of those,
    the frames counted close; dist[i, j]: the sum of their distances there.
    Upper triangle (i < j), tracks of different cameras."""
    tk = np.concatenate(rows)
    fr = np.concatenate([np.asarray(c.frames) for c in cameras])
    xy = np.concatenate([np.asarray(c.xy, float).reshape(-1, 2) for c in cameras])
    ok = np.isfinite(xy).all(axis=1)
    tk, fr, xy = tk[ok], fr[ok], xy[ok]
    order = np.argsort(fr, kind="stable")
    tk, fr, xy = tk[order], fr[order], xy[order]
    cam = track_cam[tk]
    bounds = np.flatnonzero(np.diff(fr)) + 1
    starts, ends = np.r_[0, bounds], np.r_[bounds, len(fr)]
    size = n * n
    co = np.zeros(size, np.int32)
    close = np.zeros(size, np.int32)
    dist = np.zeros(size, np.float32)
    keys, near, ds, pending = [], [], [], 0
    mnn = association.method == "colocate-mnn"

    def flush():
        nonlocal keys, near, ds, pending
        if keys:
            k, m, d = np.concatenate(keys), np.concatenate(near), np.concatenate(ds)
            co[:] += np.bincount(k, minlength=size).astype(np.int32)
            close[:] += np.bincount(k[m], minlength=size).astype(np.int32)
            dist[:] += np.bincount(k[m], weights=d[m], minlength=size).astype(np.float32)
        keys, near, ds, pending = [], [], [], 0

    for a, b in zip(starts, ends, strict=True):
        if b - a < 2:
            continue
        t, c, p = tk[a:b], cam[a:b], xy[a:b]
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        other = c[:, None] != c[None, :]
        i, j = np.nonzero(np.triu(other, 1))
        if not len(i):
            continue
        hit = d[i, j] <= association.gate_m
        if mnn:
            hit &= mutual_nearest(d, c)[i, j]
        lo, hi = np.minimum(t[i], t[j]), np.maximum(t[i], t[j])
        keys.append(lo * n + hi)
        near.append(hit)
        ds.append(d[i, j])
        pending += len(i)
        if pending >= _CHUNK_PAIRS:
            flush()
    flush()
    return co.reshape(n, n), close.reshape(n, n), dist.reshape(n, n)


def mutual_nearest(d, cams):
    """m[i, j]: j is i's nearest point in j's camera and i is j's nearest in i's."""
    nearest = np.zeros_like(d, dtype=bool)  # nearest[i, j]: j nearest to i in cam j
    for c in np.unique(cams):
        cols = np.flatnonzero(cams == c)
        best = cols[np.argmin(d[:, cols], axis=1)]
        nearest[np.arange(len(d)), best] = True
    return nearest & nearest.T


class _Clusters:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.members = {i: [i] for i in range(n)}

    def find(self, i):
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra] += self.members.pop(rb)


def _compatible(xs, ys, track_cam, first, last, conflict):
    x, y = np.asarray(xs), np.asarray(ys)
    same_cam = track_cam[x][:, None] == track_cam[y][None, :]
    alive = (first[x][:, None] <= last[y][None, :]) & (first[y][None, :] <= last[x][:, None])
    if (same_cam & alive).any():
        return False
    return not conflict[x][:, y].any()


def associate(cameras, association, fps):
    """The global id of every row of every camera (a list of arrays in the
    order of cameras); every track's rows share one id."""
    rows, track_cam, first, last, n = _track_index(cameras)
    if association.method == "none" or n == 0:
        return rows
    co, close, dist = _pair_stats(cameras, rows, track_cam, n, association)
    co = co + co.T
    close = close + close.T
    dist = dist + dist.T
    min_frames = max(1, math.ceil(association.min_overlap_s * fps))
    enough = co >= min_frames
    share = np.divide(close, co, out=np.zeros(co.shape), where=co > 0)
    conflict = enough & (share < association.min_close)
    i, j = np.nonzero(np.triu(enough & ~conflict, 1))
    mean_d = dist[i, j] / np.maximum(close[i, j], 1)
    order = np.lexsort((mean_d, -share[i, j]))
    clusters = _Clusters(n)
    for a, b in zip(i[order], j[order], strict=True):
        ra, rb = clusters.find(a), clusters.find(b)
        if ra == rb:
            continue
        xs, ys = clusters.members[ra], clusters.members[rb]
        if _compatible(xs, ys, track_cam, first, last, conflict):
            clusters.join(a, b)
    if association.handover_s is not None:
        _handover(cameras, rows, clusters, association, fps, (track_cam, first, last, conflict))
    root = np.array([clusters.find(k) for k in range(n)])
    return [root[r] for r in rows]


def _handover(cameras, rows, clusters, association, fps, tracks):
    """Join a cluster that starts where and soon after another one ended. Its
    start and end are its first and last floor point; a join must still pass
    _compatible on tracks = (track_cam, first, last, conflict), whose first and
    last are every row's: a track with no floor point is still alive."""
    n = len(tracks[0])
    root = np.array([clusters.find(k) for k in range(n)])
    ids, frames, xy = merge_points(cameras, [root[r] for r in rows])
    if not len(ids):
        return
    order = np.lexsort((frames, ids))
    ids, frames, xy = ids[order], frames[order], xy[order]
    cut = np.flatnonzero(np.diff(ids)) + 1
    head, tail = np.r_[0, cut], np.r_[cut, len(ids)] - 1
    cid, f0, f1 = ids[head], frames[head], frames[tail]
    p0, p1 = xy[head], xy[tail]
    gap = f0[None, :] - f1[:, None]  # [ender, starter] frames
    max_gap = association.handover_s * fps
    d = np.linalg.norm(p1[:, None, :] - p0[None, :, :], axis=2)
    reach = association.handover_radius_m + association.handover_speed_mps * gap / fps
    e, s = np.nonzero((gap > 0) & (gap <= max_gap) & (d <= reach))
    used_end, used_start = set(), set()
    for k in np.argsort(d[e, s], kind="stable"):
        a, b = e[k], s[k]
        if a in used_end or b in used_start:
            continue
        ra, rb = clusters.find(cid[a]), clusters.find(cid[b])
        if ra == rb or not _compatible(clusters.members[ra], clusters.members[rb], *tracks):
            continue
        used_end.add(a)
        used_start.add(b)
        clusters.join(cid[a], cid[b])


def merge_points(cameras, global_ids):
    """(ids, frames, xy): one floor point per global id and frame, the mean of
    its cameras' points there; rows without a floor point are left out."""
    gid = np.concatenate([np.asarray(g) for g in global_ids])
    fr = np.concatenate([np.asarray(c.frames) for c in cameras])
    xy = np.concatenate([np.asarray(c.xy, float).reshape(-1, 2) for c in cameras])
    ok = np.isfinite(xy).all(axis=1)
    gid, fr, xy = gid[ok], fr[ok], xy[ok]
    if not len(gid):
        return gid, fr, xy.reshape(0, 2)
    keys, inv = np.unique(np.column_stack([gid, fr]), axis=0, return_inverse=True)
    inv = inv.ravel()
    count = np.bincount(inv)
    mean = np.column_stack([np.bincount(inv, weights=xy[:, k]) / count for k in range(2)])
    return keys[:, 0], keys[:, 1], mean
