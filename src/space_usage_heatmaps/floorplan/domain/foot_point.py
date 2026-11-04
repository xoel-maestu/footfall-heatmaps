"""The floor point of a person box.

Image points ("centroid", "bottom") go through the ground homography. "head"
back-projects the top-centre of the box onto the horizontal plane at the
person's height with the full 3x4 camera matrix P = K [R | t] (world in
metres, z up, floor at z = 0): that plane's homography is P on columns 0, 1 and
h * col 2 + col 3. "auto" picks per box: the bottom-centre unless the legs are
hidden (shelves, the lower image border), which shows as a box shorter than a
person standing on its bottom-centre floor point. "auto-track" measures that
person instead of assuming one: a track's height is a high percentile of its
boxes' implied heights (occlusion only shortens a box), a box is hidden when it
is clearly shorter than its track, and its head is placed at the track's height.
"""

import numpy as np

FOOT_POINTS = ("centroid", "bottom", "head", "auto", "auto-track")
PERSON_HEIGHT_M = 1.70
MIN_IMPLIED_HEIGHT_M = 1.40
# auto-track, chosen on the tune split's ground-truth boxes (iterations.md):
TRACK_HEIGHT_PERCENTILE = 90  # of the track's implied heights
TRACK_HEIGHT_RANGE_M = (1.0, 2.0)  # clip: a track's height is a person's
HIDDEN_RATIO = 0.88  # a box below this share of its track's height is hidden


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
    """Floor (x, y) of image points on the horizontal plane z = height (one
    height, or one per point)."""
    p = np.asarray(projection, dtype=float)
    us, vs = np.asarray(us, float), np.asarray(vs, float)
    h = np.broadcast_to(np.asarray(height, float), us.shape)
    planes = np.empty((us.size, 3, 3))
    planes[:, :, 0], planes[:, :, 1] = p[:, 0], p[:, 1]
    planes[:, :, 2] = h[:, None] * p[:, 2] + p[:, 3]
    q = np.linalg.solve(planes, np.stack([us, vs, np.ones_like(us)], axis=-1)[..., None])[..., 0]
    return q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]


def track_heights(implied, track_ids):
    """Per box, its track's height: a high percentile of the track's implied
    heights, clipped to a person's range."""
    out = np.empty(len(implied))
    ids = np.asarray(track_ids)
    for t in np.unique(ids):
        m = ids == t
        out[m] = np.percentile(implied[m], TRACK_HEIGHT_PERCENTILE)
    return np.clip(out, *TRACK_HEIGHT_RANGE_M)


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
    track_ids=None,
):
    """(xs, ys, head) of each x1, y1, x2, y2 box; head marks the boxes placed
    from their top edge. In "auto", a box uses its head when its bottom edge
    reaches the last image row (truncated) or its implied height is below
    min_implied_height_m, and keeps its bottom when its top edge is on the
    first row (the head is cut, its height says nothing). "auto-track" needs
    track_ids and replaces both person_height_m and min_implied_height_m by
    each track's own height (HIDDEN_RATIO of it for the threshold)."""
    if mode not in FOOT_POINTS:
        raise ValueError(f"unknown foot point {mode!r}, expected one of {FOOT_POINTS}")
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    head = np.zeros(len(b), dtype=bool)
    height = np.full(len(b), float(person_height_m))
    if mode == "head":
        head[:] = True
    elif mode in ("auto", "auto-track"):
        implied = implied_heights(b, projection)
        threshold = min_implied_height_m
        if mode == "auto-track":
            if track_ids is None:
                raise ValueError("foot point 'auto-track' needs the boxes' track ids")
            height = track_heights(implied, track_ids)
            threshold = HIDDEN_RATIO * height
        truncated = b[:, 3] >= image_height - 1
        head = (truncated | (implied < threshold)) & (b[:, 1] > 0)
    point = "centroid" if mode == "centroid" else "bottom"
    xs, ys = _plane_to_world(projection, 0.0, *image_points(b, point))
    if head.any():
        us = (b[head, 0] + b[head, 2]) / 2
        xs[head], ys[head] = _plane_to_world(projection, height[head], us, b[head, 1])
    return xs, ys, head
