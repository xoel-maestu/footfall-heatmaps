"""The floor point of a person box.

Image points ("centroid", "bottom") go through the ground homography. "head"
back-projects the top-centre of the box onto the horizontal plane at the
person's height with the full 3x4 camera matrix P = K [R | t] (world in
metres, z up, floor at z = 0): that plane's homography is P on columns 0, 1 and
h * col 2 + col 3. "auto" picks per box: the bottom-centre unless the legs are
hidden (shelves, the lower image border), which shows as a box shorter than a
person standing on its bottom-centre floor point. "auto-track" measures that
person instead of assuming one: a track's height is a high percentile of its
boxes' implied heights so far (occlusion only shortens a box), a box is hidden
when it is clearly shorter than its track, and its head is placed at the
track's height. It is causal (each box sees only its track's frames up to its
own), so the same rule runs live, frame by frame. Its head rule needs the
camera clearly above the heads (head_rule_applies); on a lower camera every
box keeps its bottom-centre.
"""

import heapq
import math

import numpy as np

FOOT_POINTS = ("centroid", "bottom", "head", "auto", "auto-track")
PERSON_HEIGHT_M = 1.70
MIN_IMPLIED_HEIGHT_M = 1.40
# auto-track, chosen on the tune split's ground-truth boxes (docs/experiment-log.md, "Phase 3a"):
TRACK_HEIGHT_PERCENTILE = 0.9  # quantile of the track's implied heights so far
TRACK_HEIGHT_MIN_M, TRACK_HEIGHT_MAX_M = 1.0, 2.0  # clip: a track's height is a person's
HIDDEN_RATIO = 0.88  # a box below this share of its track's height is hidden
HEAD_CUT_PX = 1.0  # a top edge at most this far from the first row: the head is cut


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


class TrackHeights:
    """Causal height of each track: at every box, the percentile (numpy's
    linear interpolation) of its track's usable implied heights up to and
    including that box's frame, clipped to [min_m, max_m]; NaN before the first
    usable one. Exact, with two heaps per track holding the lower and upper
    order statistics around the percentile: O(log k) per box, O(n log n) in
    all. The state persists across calls, so a live loop feeds it one frame at
    a time and gets what one call over the whole video gives."""

    def __init__(
        self,
        percentile=TRACK_HEIGHT_PERCENTILE,
        hidden_ratio=HIDDEN_RATIO,
        min_m=TRACK_HEIGHT_MIN_M,
        max_m=TRACK_HEIGHT_MAX_M,
    ):
        self.percentile, self.hidden_ratio = percentile, hidden_ratio
        self.min_m, self.max_m = min_m, max_m
        self._tracks = {}  # track id -> (lower max-heap, negated; upper min-heap)

    def _add(self, track, value):
        low, high = self._tracks.setdefault(track, ([], []))
        if low and value > -low[0]:
            heapq.heappush(high, value)
        else:
            heapq.heappush(low, -value)
        target = math.floor(self.percentile * (len(low) + len(high) - 1)) + 1
        while len(low) > target:
            heapq.heappush(high, -heapq.heappop(low))
        while len(low) < target:
            heapq.heappush(low, -heapq.heappop(high))

    def _value(self, track):
        low, high = self._tracks.get(track, ((), ()))
        if not low:
            return math.nan
        pos = self.percentile * (len(low) + len(high) - 1)
        below = -low[0]
        frac = pos - math.floor(pos)
        return below + frac * (high[0] - below) if frac > 0 and high else below

    def heights(self, implied, track_ids, frames, usable):
        """Per box, its track's height so far; rows are taken in frame order
        and only the usable ones add to their track."""
        ids = np.asarray(track_ids).tolist()
        values = np.asarray(implied, dtype=float).tolist()
        ok = np.asarray(usable, dtype=bool).tolist()
        out = np.empty(len(ids))
        for i in np.argsort(np.asarray(frames), kind="stable").tolist():
            if ok[i]:
                self._add(ids[i], values[i])
            out[i] = self._value(ids[i])
        return np.clip(out, self.min_m, self.max_m)


def camera_height(projection):
    """Height in metres of the camera centre above the floor (z = 0): the null
    vector C of the 3x4 matrix (P C = 0), in any scale or sign of P."""
    _, _, vt = np.linalg.svd(np.asarray(projection, dtype=float))
    c = vt[-1]
    return math.inf if c[3] == 0 else float(c[2] / c[3])


def head_rule_applies(projection, track_heights):
    """Whether auto-track may place a box from its head: the camera stands at
    least one clip span (max_m - min_m) above the tallest head plane (max_m).

    Why that margin. The head rule meets the ray through the box top with the
    plane z = h; at horizontal distance D from a camera at height Hc the ray
    drops (Hc - h) over D, so a height error dh moves the floor point by
    dh * D / (Hc - h), and at Hc <= h the plane is not met in front of the
    camera at all (the point lands behind it or at infinity). The track height
    is only known to lie in the clip [min_m, max_m], so dh can reach the span
    s = max_m - min_m. With Hc - h >= s for every h <= max_m, the worst height
    error moves the point by at most D: bounded and in front of the camera.
    Below that the factor D / (Hc - h) grows without bound. The margin comes
    from the configured clip, not from any dataset."""
    span = track_heights.max_m - track_heights.min_m
    return camera_height(projection) >= track_heights.max_m + span


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
    frames=None,
    track_heights=None,
):
    """(xs, ys, head) of each x1, y1, x2, y2 box; head marks the boxes placed
    from their top edge. In "auto", a box uses its head when its bottom edge
    reaches the last image row (truncated) or its implied height is below
    min_implied_height_m, and keeps its bottom when its top edge is within
    HEAD_CUT_PX of the first row (the head is cut, its height says nothing).
    "auto-track" needs track_ids and frames and replaces both person_height_m
    and min_implied_height_m by each track's own height so far (track_heights,
    a TrackHeights whose state carries over between calls; a fresh one with
    the defaults when None), its hidden_ratio of it for the threshold; boxes
    with a cut head do not count towards that height."""
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
        whole_head = b[:, 1] > HEAD_CUT_PX
        if mode == "auto-track":
            if track_ids is None:
                raise ValueError("foot point 'auto-track' needs the boxes' track ids")
            if frames is None:
                raise ValueError("foot point 'auto-track' needs the boxes' frames")
            estimator = TrackHeights() if track_heights is None else track_heights
            height = estimator.heights(implied, track_ids, frames, whole_head)
            threshold = estimator.hidden_ratio * height
            if not head_rule_applies(projection, estimator):
                whole_head = np.zeros(len(b), dtype=bool)  # every box keeps its bottom
        truncated = b[:, 3] >= image_height - 1
        # A NaN height (no usable box yet) compares False: the box keeps its bottom.
        head = (truncated | (implied < threshold)) & whole_head
    point = "centroid" if mode == "centroid" else "bottom"
    xs, ys = _plane_to_world(projection, 0.0, *image_points(b, point))
    if head.any():
        us = (b[head, 0] + b[head, 2]) / 2
        xs[head], ys[head] = _plane_to_world(projection, height[head], us, b[head, 1])
    return xs, ys, head


def camera_floor_points(
    cam,
    boxes,
    foot_point,
    person_height_m=PERSON_HEIGHT_M,
    min_implied_height_m=MIN_IMPLIED_HEIGHT_M,
    track_ids=None,
    frames=None,
    track_heights=None,
):
    """floor_points of x1, y1, x2, y2 boxes with one camera's calibration; the
    image points (centroid, bottom) need only its ground homography."""
    projection = cam.projection
    if projection is None:
        if foot_point not in ("centroid", "bottom"):
            raise ValueError(f"foot point {foot_point!r} needs the 3x4 camera matrix")
        projection = np.insert(cam.world_to_image, 2, 0.0, axis=1)
    return floor_points(
        boxes,
        projection,
        cam.height,
        foot_point,
        person_height_m,
        min_implied_height_m,
        track_ids,
        frames,
        track_heights,
    )
