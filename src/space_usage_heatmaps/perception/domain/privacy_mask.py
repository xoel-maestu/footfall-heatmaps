"""R6 (spec 3): per-camera privacy masks, applied to detections before the tracker.

A box is dropped when its bottom-centre (where the feet touch the floor)
lies inside an image polygon (source pixels), or its floor point inside a
floor polygon (metres, through the camera's ground homography).
"""

import numpy as np

from space_usage_heatmaps.floorplan.domain.homography import Homography
from space_usage_heatmaps.perception.domain.frame import Detections


def _polygons(polygons):
    out = [np.asarray(p, float).reshape(-1, 2) for p in polygons]
    if any(len(p) < 3 for p in out):
        raise ValueError("a mask polygon needs at least three points")
    return out


def inside(xs, ys, polygon):
    """Even-odd rule: which points (xs, ys) lie inside the polygon."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    hit = np.zeros(xs.shape, bool)
    x0, y0 = polygon[:, 0], polygon[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for ax, ay, bx, by in zip(x0, y0, x1, y1, strict=True):
        crosses = (ay > ys) != (by > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = ax + (ys - ay) * (bx - ax) / (by - ay)
        hit ^= crosses & (xs < at)
    return hit


class PrivacyMask:
    def __init__(self, image_polygons=(), floor_polygons=(), world_to_image=None):
        self.image = _polygons(image_polygons)
        self.floor = _polygons(floor_polygons)
        if self.floor and world_to_image is None:
            raise ValueError("a floor mask needs the camera's ground homography")
        self.homography = Homography(world_to_image) if self.floor else None

    def keep(self, boxes):
        """Which x1, y1, x2, y2 boxes stay."""
        b = np.asarray(boxes, float).reshape(-1, 4)
        us, vs = (b[:, 0] + b[:, 2]) / 2, b[:, 3]
        masked = np.zeros(len(b), bool)
        for polygon in self.image:
            masked |= inside(us, vs, polygon)
        if self.floor:
            xs, ys = self.homography.to_world(us, vs)
            for polygon in self.floor:
                masked |= inside(xs, ys, polygon)
        return ~masked

    def apply(self, detections):
        keep = self.keep(detections.boxes)
        return Detections(detections.boxes.reshape(-1, 4)[keep], detections.scores[keep])
