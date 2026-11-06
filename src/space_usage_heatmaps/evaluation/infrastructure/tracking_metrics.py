"""Tracking guards of spec 6: HOTA and IDF1 of a camera's tracked boxes against
its ground-truth boxes, in image space, with roboflow `trackers`' port of
TrackEval (Apache-2.0 over TrackEval's MIT code): HOTA over IoU thresholds
0.05-0.95, IDF1 at IoU 0.5.

Scored on the frames the camera's truth spans; tracked boxes outside them are
left out. The truth boxes are the visible part of each person (NVIDIA
ground_truth.txt), the same thing a detector boxes.
"""

import numpy as np
from trackers.eval import box_iou, compute_hota_metrics, compute_identity_metrics


def _by_frame(frames, ids, boxes, span):
    order = np.argsort(frames, kind="stable")
    frames, ids, boxes = frames[order], ids[order], boxes[order]
    lo = np.searchsorted(frames, span, side="left")
    hi = np.searchsorted(frames, span, side="right")
    return [(ids[a:b], boxes[a:b]) for a, b in zip(lo, hi, strict=True)]


def tracking_metrics(samples, tracks):
    """{hota, deta, assa, idf1} of tracks (xyxy) against samples (x, y, w, h)."""
    span = np.arange(samples.frames.min(), samples.frames.max() + 1)
    truth_xyxy = np.asarray(samples.boxes, float).copy()
    truth_xyxy[:, 2:] += truth_xyxy[:, :2]
    truth = _by_frame(samples.frames, samples.track_ids, truth_xyxy, span)
    pred = _by_frame(tracks.frames, tracks.track_ids, np.asarray(tracks.boxes, float), span)
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
