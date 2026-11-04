"""The floor point of a person box.

Image points ("centroid", "bottom") go through the ground homography. "head"
back-projects the top-centre of the box onto the horizontal plane at the
person's height with the full 3x4 camera matrix P = K [R | t] (world in
metres, z up, floor at z = 0): that plane's homography is P on columns 0, 1 and
h * col 2 + col 3. "auto" picks per box: the bottom-centre unless the legs are
hidden (shelves, the lower image border), which shows as a box shorter than a
person standing on its bottom-centre floor point.
"""

import numpy as np

FOOT_POINTS = ("centroid", "bottom", "head", "auto")
PERSON_HEIGHT_M = 1.70
MIN_IMPLIED_HEIGHT_M = 1.40


def image_points(boxes, foot_point):
    """(us, vs) of each x1, y1, x2, y2 box: its centre ("centroid", the centre
    of the disc Ultralytics' heatmap solution accumulates; the disc is a render
    kernel, we project its centre) or the middle of its lower edge ("bottom",
    where the feet touch the floor when the box is whole)."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    us = (b[:, 0] + b[:, 2]) / 2
    if foot_point == "centroid":
        return us, (b[:, 1] + b[:, 3]) / 2
    if foot_point == "bottom":
        return us, b[:, 3]
    raise ValueError(f"unknown foot point {foot_point!r}")


def _plane_to_world(projection, height, us, vs):
    """Floor (x, y) of image points on the horizontal plane z = height."""
    p = np.asarray(projection, dtype=float)
    plane = np.column_stack([p[:, 0], p[:, 1], height * p[:, 2] + p[:, 3]])
    q = np.linalg.solve(plane, np.vstack([us, vs, np.ones_like(us)]))
    return q[0] / q[2], q[1] / q[2]


def implied_heights(boxes, projection):
    """Height above each box's bottom-centre floor point whose projection lies
    on the box's top edge (solved on the image row, v): about the person's
    height for a whole box, less when the bottom of the box is not the feet."""
    p = np.asarray(projection, dtype=float)
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    xs, ys = _plane_to_world(p, 0.0, *image_points(b, "bottom"))
    floor = p[:, [0, 1, 3]] @ np.vstack([xs, ys, np.ones_like(xs)])
    up = p[:, 2]
    # v_top * (floor_w + h up_w) = floor_v + h up_v, linear in h.
    v = b[:, 1]
    return (floor[1] - v * floor[2]) / (v * up[2] - up[1])


def floor_points(
    boxes,
    projection,
    image_height,
    mode,
    person_height_m=PERSON_HEIGHT_M,
    min_implied_height_m=MIN_IMPLIED_HEIGHT_M,
):
    """(xs, ys, head) of each x1, y1, x2, y2 box; head marks the boxes placed
    from their top edge. In "auto", a box uses its head when its bottom edge
    reaches the last image row (truncated) or its implied height is below
    min_implied_height_m, and keeps its bottom when its top edge is on the
    first row (the head is cut, its height says nothing)."""
    if mode not in FOOT_POINTS:
        raise ValueError(f"unknown foot point {mode!r}, expected one of {FOOT_POINTS}")
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    head = np.zeros(len(b), dtype=bool)
    if mode == "head":
        head[:] = True
    elif mode == "auto":
        truncated = b[:, 3] >= image_height - 1
        short = implied_heights(b, projection) < min_implied_height_m
        head = (truncated | short) & (b[:, 1] > 0)
    point = "centroid" if mode == "centroid" else "bottom"
    xs, ys = _plane_to_world(projection, 0.0, *image_points(b, point))
    if head.any():
        us = (b[head, 0] + b[head, 2]) / 2
        xs[head], ys[head] = _plane_to_world(projection, person_height_m, us, b[head, 1])
    return xs, ys, head
