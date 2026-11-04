import numpy as np
import pytest

from space_usage_heatmaps.floorplan.domain.foot_point import (
    floor_points,
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


def _camera():
    k = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    tilt = np.radians(30)
    # Camera axes in world coordinates: x right, y down in the image, z forward.
    forward = np.array([0, np.cos(tilt), -np.sin(tilt)])
    right = np.array([1.0, 0, 0])
    down = np.cross(forward, right)
    r = np.stack([right, down, forward])
    centre = np.array([0, -10.0, 4.0])
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
