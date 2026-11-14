import numpy as np
import pytest

from footfall_heatmaps.floorplan.domain.foot_point import (
    TrackHeights,
    camera_height,
    floor_points,
    head_rule_applies,
    image_points,
    implied_heights,
)

# x1, y1, x2, y2 in pixels
BOXES = np.array([[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 4.0, 2.0]])


def test_centroid_is_the_box_centre():
    u, v = image_points(BOXES, "centroid")

    np.testing.assert_allclose(u, [20.0, 2.0])
    np.testing.assert_allclose(v, [40.0, 1.0])


def test_bottom_is_the_middle_of_the_lower_edge():
    u, v = image_points(BOXES, "bottom")

    np.testing.assert_allclose(u, [20.0, 2.0])
    np.testing.assert_allclose(v, [60.0, 2.0])


def test_an_unknown_point_is_an_error():
    with pytest.raises(ValueError, match="foot point"):
        image_points(BOXES, "head")


# A 1920x1080 camera 4 m above the floor at (0, -10), looking along +y and
# tilted 30 degrees down; world in metres, z up.
H_IMG = 1080


def _camera(height=4.0, tilt_deg=30.0):
    k = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    tilt = np.radians(tilt_deg)
    # Camera axes in world coordinates: x right, y down in the image, z forward.
    forward = np.array([0, np.cos(tilt), -np.sin(tilt)])
    right = np.array([1.0, 0, 0])
    down = np.cross(forward, right)
    r = np.stack([right, down, forward])
    centre = np.array([0, -10.0, height])
    return k @ np.hstack([r, (-r @ centre)[:, None]])


P = _camera()


def _project(x, y, z):
    q = P @ np.array([x, y, z, 1.0])
    return q[:2] / q[2]


def _person_box(x, y, height=1.70, cut_m=0.0, half_width_px=20.0):
    """Box of a person standing at (x, y); cut_m hides the legs up to that height."""
    u_foot, v_foot = _project(x, y, cut_m)
    u_head, v_head = _project(x, y, height)
    u = (u_foot + u_head) / 2
    return np.array([[u - half_width_px, v_head, u + half_width_px, v_foot]])


def test_head_recovers_the_floor_point_of_a_standing_person():
    box = _person_box(0.0, 3.0)

    xs, ys, head = floor_points(box, P, H_IMG, "head", person_height_m=1.70)

    np.testing.assert_allclose([xs[0], ys[0]], [0.0, 3.0], atol=1e-9)
    assert head.tolist() == [True]


def test_head_off_axis_is_close_to_the_floor_point():
    box = _person_box(2.5, 1.0)

    xs, ys, _ = floor_points(box, P, H_IMG, "head", person_height_m=1.70)

    assert np.hypot(xs[0] - 2.5, ys[0] - 1.0) < 0.1


def test_implied_height_of_a_full_box_is_the_person_height():
    np.testing.assert_allclose(implied_heights(_person_box(0.0, 3.0), P), [1.70], atol=1e-9)


def test_a_box_cut_at_the_knees_implies_a_short_person_and_auto_uses_the_head():
    box = _person_box(0.0, 3.0, cut_m=0.5)

    assert implied_heights(box, P)[0] < 1.40
    xs, ys, head = floor_points(box, P, H_IMG, "auto", 1.70, 1.40)

    assert head.tolist() == [True]
    np.testing.assert_allclose([xs[0], ys[0]], [0.0, 3.0], atol=1e-9)


def test_auto_keeps_the_bottom_of_a_full_box():
    box = _person_box(1.0, 2.0, height=1.80)

    xs, ys, head = floor_points(box, P, H_IMG, "auto", 1.70, 1.40)

    assert head.tolist() == [False]
    bottom_x, bottom_y, _ = floor_points(box, P, H_IMG, "bottom")
    np.testing.assert_allclose([xs[0], ys[0]], [bottom_x[0], bottom_y[0]])


def test_auto_uses_the_head_when_the_box_touches_the_bottom_border():
    box = _person_box(0.0, 3.0)
    box[0, 3] = H_IMG  # bottom edge on the last row: the feet may be outside the image

    _, _, head = floor_points(box, P, H_IMG, "auto", 1.70, 1.40)

    assert head.tolist() == [True]


def test_auto_keeps_the_bottom_when_the_head_is_cut_by_the_top_border():
    box = _person_box(0.0, 3.0, cut_m=0.5)
    box[0, 1] = 0.0

    _, _, head = floor_points(box, P, H_IMG, "auto", 1.70, 1.40)

    assert head.tolist() == [False]


def test_bottom_and_centroid_go_through_the_ground_homography():
    box = _person_box(0.0, 2.0)

    xs, ys, head = floor_points(box, P, H_IMG, "bottom")

    np.testing.assert_allclose([xs[0], ys[0]], [0.0, 2.0], atol=1e-9)
    assert head.tolist() == [False]
    _, yc, _ = floor_points(box, P, H_IMG, "centroid")
    assert yc[0] > 2.0  # the box centre is higher in the image: farther away


def test_an_unknown_mode_is_an_error():
    with pytest.raises(ValueError, match="foot point"):
        floor_points(BOXES, P, H_IMG, "knees")


def test_auto_track_keeps_the_bottom_of_a_short_person_seen_whole():
    # A 1.20 m child: every box is whole, so its implied height is its height.
    boxes = np.vstack([_person_box(0.0, y, height=1.20) for y in (2.0, 3.0, 4.0)])
    ids = np.array([5, 5, 5])

    _, _, head_auto = floor_points(boxes, P, H_IMG, "auto")
    _, _, head_track = floor_points(
        boxes, P, H_IMG, "auto-track", track_ids=ids, frames=np.arange(3)
    )

    assert head_auto.all() and not head_track.any()


def test_auto_track_places_a_hidden_box_with_the_height_of_its_track():
    # A 1.85 m adult seen whole in four frames, cut at the knees in the fifth.
    whole = [_person_box(0.0, y, height=1.85) for y in (1.0, 2.0, 3.0, 4.0)]
    boxes = np.vstack([*whole, _person_box(0.0, 5.0, height=1.85, cut_m=0.6)])
    ids = np.full(5, 3)

    xs, ys, head = floor_points(boxes, P, H_IMG, "auto-track", track_ids=ids, frames=np.arange(5))

    assert head.tolist() == [False] * 4 + [True]
    np.testing.assert_allclose([xs[4], ys[4]], [0.0, 5.0], atol=1e-6)


def test_auto_track_needs_track_ids_and_frames():
    with pytest.raises(ValueError, match="track ids"):
        floor_points(_person_box(0.0, 3.0), P, H_IMG, "auto-track")
    with pytest.raises(ValueError, match="frames"):
        floor_points(_person_box(0.0, 3.0), P, H_IMG, "auto-track", track_ids=[1])


def test_auto_track_is_causal_a_box_is_judged_on_its_track_so_far():
    # Cut at the knees in the first frame, whole afterwards: the first box only
    # knows itself, so it keeps its bottom; the later frames do not reach back.
    cut = _person_box(0.0, 1.0, height=1.85, cut_m=0.6)
    whole = [_person_box(0.0, y, height=1.85) for y in (2.0, 3.0, 4.0)]
    boxes = np.vstack([cut, *whole, cut])
    ids = np.full(5, 3)

    _, _, head = floor_points(boxes, P, H_IMG, "auto-track", track_ids=ids, frames=np.arange(5))

    assert head.tolist() == [False, False, False, False, True]


def test_track_heights_are_the_exact_prefix_percentile_in_frame_order():
    rng = np.random.default_rng(1)
    n = 400
    implied = rng.uniform(0.5, 2.5, n)
    ids = rng.integers(0, 6, n)
    frames = rng.permutation(n)  # rows in any order; the frame decides the prefix
    usable = rng.random(n) > 0.2

    got = TrackHeights(0.9, 0.88, 1.0, 2.0).heights(implied, ids, frames, usable)

    for i in range(n):
        prefix = (ids == ids[i]) & (frames <= frames[i]) & usable
        if prefix.any():
            want = np.clip(np.percentile(implied[prefix], 90), 1.0, 2.0)
            np.testing.assert_allclose(got[i], want, atol=1e-12)
        else:
            assert np.isnan(got[i])


def test_track_heights_fed_frame_by_frame_match_one_batch():
    rng = np.random.default_rng(2)
    frames = np.repeat(np.arange(50), 3)
    ids = np.tile([1, 2, 3], 50)
    implied = rng.uniform(1.0, 2.0, frames.size)
    usable = np.ones(frames.size, dtype=bool)

    batch = TrackHeights().heights(implied, ids, frames, usable)
    live = TrackHeights()
    stream = np.concatenate(
        [
            live.heights(implied[f], ids[f], frames[f], usable[f])
            for f in np.split(np.arange(150), 50)
        ]
    )

    np.testing.assert_allclose(stream, batch)


def test_a_box_cut_by_the_top_border_stays_out_of_its_track_height():
    # A 1.85 m adult whose head is cut (top edge within 1 px of the first row)
    # in most frames: those short boxes must not pull the track height down.
    whole = _person_box(0.0, 4.0, height=1.85)
    cut = _person_box(0.0, 3.0, height=1.85, cut_m=0.6)
    cut[0, 1] = 0.8
    hidden = _person_box(0.0, 5.0, height=1.85, cut_m=0.6)
    boxes = np.vstack([*[whole] * 3, *[cut] * 20, hidden])
    ids = np.full(len(boxes), 7)

    xs, ys, head = floor_points(
        boxes, P, H_IMG, "auto-track", track_ids=ids, frames=np.arange(len(boxes))
    )

    assert not head[:-1].any() and head[-1]  # head-cut boxes keep their bottom
    np.testing.assert_allclose([xs[-1], ys[-1]], [0.0, 5.0], atol=1e-6)


def test_camera_height_is_the_camera_centre_above_the_floor():
    assert camera_height(P) == pytest.approx(4.0)
    assert camera_height(-2.5 * _camera(1.68, 9.0)) == pytest.approx(1.68)


def test_the_head_rule_needs_the_camera_a_clip_span_above_the_tallest_head():
    # Track heights clipped to [1.0, 2.0] m: the head rule needs the camera at
    # 2.0 + (2.0 - 1.0) = 3.0 m or higher.
    heights = TrackHeights(0.9, 0.88, 1.0, 2.0)
    assert head_rule_applies(_camera(3.1), heights)
    assert not head_rule_applies(_camera(2.9), heights)
    assert head_rule_applies(_camera(2.7), TrackHeights(0.9, 0.88, 1.2, 1.9))  # needs 2.6 m


def _hidden_adult(camera):
    """A 1.85 m adult seen whole four times, then cut at the knees, by camera."""
    project = _projector(camera)
    boxes = []
    for y, cut in ((1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.6)):
        u_foot, v_foot = project(0.0, y, cut)
        u_head, v_head = project(0.0, y, 1.85)
        u = (u_foot + u_head) / 2
        boxes.append([u - 20.0, v_head, u + 20.0, v_foot])
    return np.array(boxes)


def _projector(camera):
    def project(x, y, z):
        q = camera @ np.array([x, y, z, 1.0])
        return q[:2] / q[2]

    return project


def test_auto_track_keeps_the_bottom_when_the_camera_is_not_above_head_height():
    # At 1.68 m the head plane of a 1.85 m adult is above the camera: the
    # back-projection is unstable, so every box keeps its bottom-centre, also
    # one whose legs the lower image border cuts (which uses the head otherwise).
    low = _camera(1.68, 9.0)
    boxes = _hidden_adult(low)
    boxes[4, 3] = H_IMG - 1
    ids, frames = np.full(5, 3), np.arange(5)

    xs, ys, head = floor_points(boxes, low, H_IMG, "auto-track", track_ids=ids, frames=frames)
    bx, by, _ = floor_points(boxes, low, H_IMG, "bottom")

    assert not head.any()
    np.testing.assert_allclose([xs, ys], [bx, by])


def test_auto_track_still_places_a_hidden_box_from_its_head_on_a_high_camera():
    high = _camera(3.1, 30.0)
    boxes = _hidden_adult(high)

    xs, ys, head = floor_points(
        boxes, high, H_IMG, "auto-track", track_ids=np.full(5, 3), frames=np.arange(5)
    )

    assert head.tolist() == [False] * 4 + [True]
    np.testing.assert_allclose([xs[4], ys[4]], [0.0, 5.0], atol=1e-6)
