"""The image point of a person box that goes through the ground homography."""

import numpy as np


def image_points(boxes, foot_point):
    """(us, vs) of each x1, y1, x2, y2 box: its centre ("centroid", the point
    Ultralytics' heatmap solution accumulates) or the middle of its lower edge
    ("bottom-centre", where the feet touch the floor when the box is whole)."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    us = (b[:, 0] + b[:, 2]) / 2
    if foot_point == "centroid":
        return us, (b[:, 1] + b[:, 3]) / 2
    if foot_point == "bottom-centre":
        return us, b[:, 3]
    raise ValueError(f"unknown foot point {foot_point!r}")
