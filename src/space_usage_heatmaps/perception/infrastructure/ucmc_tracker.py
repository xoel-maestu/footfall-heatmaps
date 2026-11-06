"""Tracker port that associates boxes on the floor, after UCMCTrack (Yi et al.,
AAAI 2024; github.com/corfyi/UCMCTrack at 7851a49, Copyright (c) 2023 Kefu Yi,
MIT licence; docs/licenses.md).

Own implementation of its tracker (tracker/ucmc.py, tracker/kalman.py,
detector/mapper.py), without camera motion compensation, which a fixed camera
does not need. Each box's bottom centre goes through the camera's floor
homography to a point in metres, with its pixel noise carried to the floor by
the Jacobian of that mapping. A constant-velocity Kalman filter per track on
the floor; the cost of a box for a track is its Mahalanobis distance plus
log det of the innovation covariance, and assignments above the threshold are
refused. Three stages as in the original: high-score boxes with confirmed and
coasting tracks, low-score boxes with the tracks left, then the high-score
boxes left with tentative tracks; the rest start tentative tracks. Positions
only (spec 3, R3): never the image.

Parameters are the defaults of the original's make_args (util/run_ucmc.py),
cdt being frames at 30 fps, scaled to the source like ByteTrack's buffer:

- wx, wy 0.1: process noise of the floor acceleration
- vmax 0.5: prior spread of a new track's speed, m/s
- a1, a2 10: highest cost of a match in the high- and low-score stages
- cdt 30: frames a coasting track waits to be matched again
- high_score 0.6: boxes below it only continue tracks
- conf_thresh 0.5: boxes below it are dropped
- pixel noise of the bottom centre: 5 % of the box height, clipped to 2-13 px
  across and 2-10 px down (detector/mapper.py getUVError)

Differences from the original, each because it breaks or does not apply here:

- a confirmed track left unmatched coasts in every frame. The original marks
  it coasted only inside the low-score stage, which it skips when the frame
  has no low-score box, so such a track is never deleted;
- a box whose bottom centre is above the horizon (no floor point in front of
  the camera) is left out, instead of being mapped behind the camera;
- ids are given when a track is confirmed, so they count from 0 like the
  other trackers' (the original numbers tentative tracks too).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULTS = {
    "wx": 0.1,
    "wy": 0.1,
    "vmax": 0.5,
    "a1": 10.0,
    "a2": 10.0,
    "cdt": 30,
    "high_score": 0.6,
    "conf_thresh": 0.5,
}
# Bump when the behaviour changes: the track cache checks it.
VERSION = 1
H_OBS = np.array([[1.0, 0, 0, 0], [0, 0, 1, 0]])  # state (x, vx, y, vy) -> (x, y)


def floor_points(boxes, image_to_world):
    """Bottom centres of (N, 4) xyxy boxes on the floor: (N, 2) points in
    metres, (N, 2, 2) covariances, and (N,) whether the point is in front of
    the camera. image_to_world: inverse of the oriented floor homography."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    uv1 = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3], np.ones(len(boxes))])
    b = uv1 @ image_to_world.T
    front = b[:, 2] > 0
    gamma = 1.0 / np.where(front, b[:, 2], 1.0)
    xy = b[:, :2] * gamma[:, None]
    # d(xy)/d(uv): gamma A[:2, :2] - gamma^2 b[:2] A[2, :2]
    jac = gamma[:, None, None] * image_to_world[None, :2, :2] - (gamma**2)[:, None, None] * (
        b[:, :2, None] * image_to_world[None, 2:3, :2]
    )
    height = boxes[:, 3] - boxes[:, 1]
    su, sv = np.clip(0.05 * height, 2, 13), np.clip(0.05 * height, 2, 10)
    cov_uv = np.zeros((len(boxes), 2, 2))
    cov_uv[:, 0, 0], cov_uv[:, 1, 1] = su**2, sv**2
    cov = jac @ cov_uv @ jac.transpose(0, 2, 1)
    return xy, cov, front


def assign(cost, limit):
    """Minimum-cost matches (row, column) whose cost is below limit, as lapjv's
    extend_cost with cost_limit: leaving a row and a column unmatched costs limit."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    big = 1e9
    padded = np.full((n + m, n + m), big)
    padded[:n, :m] = np.minimum(cost, big)
    padded[np.arange(n), m + np.arange(n)] = limit / 2
    padded[n + np.arange(m), np.arange(m)] = limit / 2
    padded[n:, m:] = 0
    rows, cols = linear_sum_assignment(padded)
    return [
        (r, c) for r, c in zip(rows, cols, strict=True) if r < n and c < m and cost[r, c] < limit
    ]


@dataclass
class _Track:
    x: np.ndarray  # (4,) x, vx, y, vy
    p: np.ndarray  # (4, 4)
    status: str = "tentative"  # or "confirmed", "coasted"
    track_id: int = -1  # given on confirmation
    misses: int = 0  # frames since the last match (the original's death_count)
    hits: int = 0  # matches while tentative (birth_count)
    box: int = -1  # index of this frame's matched box, or -1


class UcmcTracker:
    def __init__(self, frame_rate, world_to_image):
        self.parameters = {**DEFAULTS, "frame_rate": frame_rate}
        self._image_to_world = np.linalg.inv(np.asarray(world_to_image, float))
        self._max_misses = DEFAULTS["cdt"] * frame_rate / 30.0
        dt = 1.0 / frame_rate
        self._f = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1.0]])
        g = np.array([[0.5 * dt * dt, 0], [dt, 0], [0, 0.5 * dt * dt], [0, dt]])
        self._q = g @ np.diag([DEFAULTS["wx"], DEFAULTS["wy"]]) @ g.T
        self._tracks = []
        self._next_id = 0

    def update(self, boxes, scores, frame_index):
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float)
        xy, cov, front = floor_points(boxes, self._image_to_world)
        usable = front & (scores >= DEFAULTS["conf_thresh"])
        high = np.flatnonzero(usable & (scores >= DEFAULTS["high_score"])).tolist()
        low = np.flatnonzero(usable & (scores < DEFAULTS["high_score"])).tolist()

        for t in self._tracks:
            t.x, t.p = self._f @ t.x, self._f @ t.p @ self._f.T + self._q
            t.box = -1

        active = [t for t in self._tracks if t.status != "tentative"]
        tentative = [t for t in self._tracks if t.status == "tentative"]
        left, high_left = self._match(active, high, xy, cov, DEFAULTS["a1"], "confirmed")
        left, _ = self._match(left, low, xy, cov, DEFAULTS["a2"], "confirmed")
        for t in left:
            t.status = "coasted"
        _, new = self._match(tentative, high_left, xy, cov, DEFAULTS["a1"], "tentative")
        for i in new:
            x = np.array([xy[i, 0], 0.0, xy[i, 1], 0.0])
            vv = DEFAULTS["vmax"] ** 2 / 3.0
            self._tracks.append(_Track(x, np.diag([1.0, vv, 1.0, vv]), box=i))

        for t in self._tracks:
            t.misses += 1
        self._tracks = [
            t
            for t in self._tracks
            if not (t.status == "coasted" and t.misses >= self._max_misses)
            and not (t.status == "tentative" and t.misses >= 2)
        ]
        out = [t for t in self._tracks if t.status == "confirmed" and t.box >= 0]
        ids = np.array([t.track_id for t in out], dtype=int)
        return ids, boxes[[t.box for t in out]].reshape(-1, 4)

    def _match(self, tracks, rows, xy, cov, limit, status):
        """Match boxes rows to tracks; updates the matched tracks. Returns the
        tracks and the box rows left unmatched."""
        if not tracks or not rows:
            return list(tracks), list(rows)
        cost = np.column_stack([_cost(t, xy[rows], cov[rows]) for t in tracks])
        matches = assign(cost, limit)
        for r, c in matches:
            t, i = tracks[c], rows[r]
            _correct(t, xy[i], cov[i])
            t.misses, t.box = 0, i
            if status == "confirmed":
                t.status = "confirmed"
            else:
                t.hits += 1
                if t.hits >= 2:
                    t.status, t.hits = "confirmed", 0
            if t.status == "confirmed" and t.track_id < 0:
                t.track_id, self._next_id = self._next_id, self._next_id + 1
        matched_tracks = {c for _, c in matches}
        matched_rows = {r for r, _ in matches}
        return (
            [t for j, t in enumerate(tracks) if j not in matched_tracks],
            [i for k, i in enumerate(rows) if k not in matched_rows],
        )


def _cost(track, xy, cov):
    """(N,) Mahalanobis distance plus log det of the innovation covariance."""
    s = H_OBS @ track.p @ H_OBS.T + cov
    diff = xy - H_OBS @ track.x
    d2 = np.einsum("ni,nij,nj->n", diff, np.linalg.inv(s), diff)
    return d2 + np.log(np.linalg.det(s))


def _correct(track, z, r):
    """Kalman update with the Joseph form, as filterpy's."""
    s = H_OBS @ track.p @ H_OBS.T + r
    k = track.p @ H_OBS.T @ np.linalg.inv(s)
    track.x = track.x + k @ (z - H_OBS @ track.x)
    i_kh = np.eye(4) - k @ H_OBS
    track.p = i_kh @ track.p @ i_kh.T + k @ r @ k.T
