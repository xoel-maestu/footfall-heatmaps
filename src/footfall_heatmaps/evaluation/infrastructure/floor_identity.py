"""Multi-camera IDF1 on the floor (E2b, multi-camera identity): TrackEval's identity metric
with floor distance in place of box IoU.

A truth position and a predicted floor point of the same frame can match when
they are at most gate_m apart. This is TrackEval's own Euclidean similarity,
1 - d / zero_distance, at its threshold 0.5 with zero_distance = 2 gate_m, as
its 3D and MTMC benchmarks score floor points. The global assignment of truth
ids to predicted ids is the one that maximises the matched detections (IDTP),
over the whole run, solved once (Hungarian). IDF1 = 2 IDTP / (truth + predicted
detections).

The counts are also split into time blocks under that one assignment, so a
block bootstrap can resample them (evaluation.domain.identity_bootstrap):
summed over blocks they are the whole run's counts.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class IdentityCounts:
    idtp: np.ndarray  # (blocks,) matched detections under the whole-run assignment
    n_truth: np.ndarray  # (blocks,) truth detections
    n_pred: np.ndarray  # (blocks,) predicted detections
    truth_ids: int
    pred_ids: int

    @property
    def idf1(self):
        total = self.n_truth.sum() + self.n_pred.sum()
        return float(2 * self.idtp.sum() / total) if total else 0.0

    @property
    def idp(self):
        n = self.n_pred.sum()
        return float(self.idtp.sum() / n) if n else 0.0

    @property
    def idr(self):
        n = self.n_truth.sum()
        return float(self.idtp.sum() / n) if n else 0.0


def _slices(frames, keys):
    order = np.argsort(frames, kind="stable")
    sf = frames[order]
    return order, np.searchsorted(sf, keys, "left"), np.searchsorted(sf, keys, "right")


def floor_identity(
    truth_ids,
    truth_frames,
    truth_xy,
    pred_ids,
    pred_frames,
    pred_xy,
    gate_m,
    blocks,
    block_of=lambda frames: frames,
):
    """Identity counts of predicted floor points against truth positions.
    block_of maps frames to block indices in [0, blocks)."""
    t_ids, t_frames = np.asarray(truth_ids), np.asarray(truth_frames)
    p_ids, p_frames = np.asarray(pred_ids), np.asarray(pred_frames)
    t_xy = np.asarray(truth_xy, float).reshape(-1, 2)
    p_xy = np.asarray(pred_xy, float).reshape(-1, 2)
    pairs = np.unique(np.column_stack([p_frames, p_ids]), axis=0) if len(p_ids) else p_ids
    if len(pairs) != len(p_ids):
        raise ValueError("a predicted id appears twice in one frame: merge its points first")
    t_uid, t_idx = np.unique(t_ids, return_inverse=True)
    p_uid, p_idx = np.unique(p_ids, return_inverse=True)
    n_truth = np.bincount(block_of(t_frames), minlength=blocks).astype(float)
    n_pred = np.bincount(block_of(p_frames), minlength=blocks).astype(float)
    frames = np.intersect1d(t_frames, p_frames)
    to, ta, tb = _slices(t_frames, frames)
    po, pa, pb = _slices(p_frames, frames)
    gi, pj, fr = [], [], []
    for f, a0, a1, b0, b1 in zip(frames, ta, tb, pa, pb, strict=True):
        t, p = to[a0:a1], po[b0:b1]
        d = np.linalg.norm(t_xy[t][:, None, :] - p_xy[p][None, :, :], axis=2)
        r, c = np.nonzero(d <= gate_m)  # NaN compares False: no match
        gi.append(t_idx[t[r]])
        pj.append(p_idx[p[c]])
        fr.append(np.full(len(r), f))
    gi = np.concatenate(gi) if gi else np.zeros(0, int)
    pj = np.concatenate(pj) if pj else np.zeros(0, int)
    fr = np.concatenate(fr) if fr else np.zeros(0, int)
    counts = np.zeros((len(t_uid), len(p_uid)))
    np.add.at(counts, (gi, pj), 1)
    idtp = np.zeros(blocks)
    if counts.size:
        rows, cols = linear_sum_assignment(counts, maximize=True)
        assigned = np.full(len(t_uid), -1)
        assigned[rows] = cols
        hit = assigned[gi] == pj
        idtp = np.bincount(block_of(fr[hit]), minlength=blocks).astype(float)
    return IdentityCounts(idtp, n_truth, n_pred, len(t_uid), len(p_uid))
