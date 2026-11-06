"""Tracking guards of spec 6: HOTA and IDF1 of a camera's tracked boxes against
its ground-truth boxes, in image space, with roboflow `trackers`' port of
TrackEval (Apache-2.0 over TrackEval's MIT code): HOTA over IoU thresholds
0.05-0.95, IDF1 at IoU 0.5.

Scored on the frames the camera's truth spans; tracked boxes outside them are
left out. The truth boxes are the visible part of each person (NVIDIA
ground_truth.txt), the same thing a detector boxes. Truth rows marked as
distractors (MOT17) follow TrackEval's MOT17 preprocessing: per frame, tracked
boxes are matched to every truth box (Hungarian on IoU, at least 0.5); those
matched to a distractor are dropped, and distractors are not scored.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from trackers.eval import box_iou, compute_hota_metrics, compute_identity_metrics

DISTRACTOR_IOU = 0.5


def _by_frame(frames, ids, boxes, span):
    order = np.argsort(frames, kind="stable")
    frames, ids, boxes = frames[order], ids[order], boxes[order]
    lo = np.searchsorted(frames, span, side="left")
    hi = np.searchsorted(frames, span, side="right")
    return [(ids[a:b], boxes[a:b]) for a, b in zip(lo, hi, strict=True)]


def _drop_distractor_matches(truth, distractor, pred):
    """Per frame, the tracked boxes minus those matched to a distractor; the
    truth minus its distractors."""
    kept_truth, kept_pred = [], []
    for (t_ids, t_boxes), d, (p_ids, p_boxes) in zip(truth, distractor, pred, strict=True):
        keep = np.ones(len(p_ids), dtype=bool)
        if d.any() and len(p_ids):
            iou = box_iou(t_boxes.reshape(-1, 4), p_boxes.reshape(-1, 4))
            score = np.where(iou >= DISTRACTOR_IOU - np.finfo(float).eps, iou, 0.0)
            rows, cols = linear_sum_assignment(score, maximize=True)
            hit = score[rows, cols] > 0
            keep[cols[hit & d[rows]]] = False
        kept_truth.append((t_ids[~d], t_boxes[~d]))
        kept_pred.append((p_ids[keep], p_boxes[keep]))
    return kept_truth, kept_pred


def tracking_metrics(samples, tracks):
    """{hota, deta, assa, idf1} of tracks (xyxy) against samples (x, y, w, h)."""
    span = np.arange(samples.frames.min(), samples.frames.max() + 1)
    truth_xyxy = np.asarray(samples.boxes, float).copy()
    truth_xyxy[:, 2:] += truth_xyxy[:, :2]
    truth = _by_frame(samples.frames, samples.track_ids, truth_xyxy, span)
    pred = _by_frame(tracks.frames, tracks.track_ids, np.asarray(tracks.boxes, float), span)
    if samples.distractor is not None and np.any(samples.distractor):
        flags = _by_frame(samples.frames, np.asarray(samples.distractor, bool), truth_xyxy, span)
        truth, pred = _drop_distractor_matches(truth, [d for d, _ in flags], pred)
    truth_ids = [ids for ids, _ in truth]
    pred_ids = [ids for ids, _ in pred]
    iou = [
        box_iou(t.reshape(-1, 4), p.reshape(-1, 4))
        for (_, t), (_, p) in zip(truth, pred, strict=True)
    ]
    hota = compute_hota_metrics(truth_ids, pred_ids, iou)
    identity = compute_identity_metrics(truth_ids, pred_ids, iou)
    return {
        "hota": float(hota["HOTA"]),
        "deta": float(hota["DetA"]),
        "assa": float(hota["AssA"]),
        "idf1": float(identity["IDF1"]),
    }


def pooled(per_scene):
    """Mean over scenes of each metric, as the pooled heatmap metrics."""
    return {k: float(np.mean([m[k] for m in per_scene])) for k in per_scene[0]}
