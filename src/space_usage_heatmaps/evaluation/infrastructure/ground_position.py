"""Ground position error of a pipeline's floor points against annotated floor
positions (WILDTRACK), spec 6 complementary metric.

Tracked boxes and annotated boxes are paired per frame in image space
(Hungarian on IoU, at least MIN_IOU), as the tracking guards match them; each
pair's error is the floor distance between the tracked box's floor point and
the annotated person's floor position. Missed people and spurious boxes are
counted (recall, precision), not scored: the error measures where the
pipeline puts the people it finds, and holds only at that recall.

A second view pairs them on the floor instead (Hungarian on floor distance,
within GROUND_GATE_M, a fixed protocol choice never tuned): a person boxed
badly but placed well counts, and a systematic offset beyond the gate shows
as lost recall rather than as large errors.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from trackers.eval import box_iou

MIN_IOU = 0.5
GROUND_GATE_M = 1.0


def _summary(e):
    return {
        "median_m": float(np.median(e)) if len(e) else None,
        "p90_m": float(np.percentile(e, 90)) if len(e) else None,
        "matched": len(e),
    }


def matched_ground_error(samples, tracks, xy, min_iou=MIN_IOU, groups=None):
    """samples: truth (boxes x, y, w, h; world floor metres); tracks: boxes
    x1, y1, x2, y2; xy: (N, 2) floor point of each tracked box. Only the
    frames the truth lists are scored. groups: name -> bool mask over the
    tracked boxes; each group's errors are summarised apart (a diagnostic)."""
    truth_xyxy = np.asarray(samples.boxes, float).copy()
    truth_xyxy[:, 2:] += truth_xyxy[:, :2]
    pred_boxes = np.asarray(tracks.boxes, float).reshape(-1, 4)
    xy = np.asarray(xy, float).reshape(-1, 2)
    errors, rows_of, tracked = [], [], 0
    for frame in np.unique(samples.frames):
        t = np.flatnonzero(samples.frames == frame)
        p = np.flatnonzero(tracks.frames == frame)
        tracked += len(p)
        if not len(p):
            continue
        iou = box_iou(truth_xyxy[t], pred_boxes[p])
        rows, cols = linear_sum_assignment(iou, maximize=True)
        ok = iou[rows, cols] >= min_iou
        d = xy[p[cols[ok]]] - samples.world[t[rows[ok]]]
        errors.extend(np.hypot(d[:, 0], d[:, 1]).tolist())
        rows_of.extend(p[cols[ok]].tolist())
    n = len(samples.frames)
    e = np.array(errors)
    rows_of = np.array(rows_of, dtype=int)
    split = {
        name: _summary(e[np.asarray(mask, bool)[rows_of]]) for name, mask in (groups or {}).items()
    }
    return {
        **_summary(e),
        "mean_m": float(e.mean()) if len(e) else None,
        "truth_boxes": n,
        "tracked_boxes": tracked,
        "recall": len(e) / n if n else None,
        "precision": len(e) / tracked if tracked else None,
        "min_iou": min_iou,
    } | ({"groups": split} if groups else {})


def ground_plane_error(samples, tracks, xy, gate_m=GROUND_GATE_M):
    """As matched_ground_error, but each frame's tracked floor points and
    annotated positions are paired by floor distance (Hungarian), at most
    gate_m apart; points that could not be placed (NaN) pair with nobody."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    errors, tracked = [], 0
    for frame in np.unique(samples.frames):
        t = np.flatnonzero(samples.frames == frame)
        p = np.flatnonzero(tracks.frames == frame)
        tracked += len(p)
        if not len(p):
            continue
        d = np.linalg.norm(samples.world[t][:, None, :] - xy[p][None, :, :], axis=2)
        cost = np.where(d <= gate_m, d, gate_m * 1e3)  # NaN compares False: out
        rows, cols = linear_sum_assignment(cost)
        ok = d[rows, cols] <= gate_m
        errors.extend(d[rows[ok], cols[ok]].tolist())
    n = len(samples.frames)
    e = np.array(errors)
    return {
        **_summary(e),
        "mean_m": float(e.mean()) if len(e) else None,
        "truth_boxes": n,
        "tracked_boxes": tracked,
        "recall": len(e) / n if n else None,
        "precision": len(e) / tracked if tracked else None,
        "gate_m": gate_m,
    }
