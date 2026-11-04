import numpy as np
import pytest

from space_usage_heatmaps.floorplan.domain.homography import Homography

# A camera at the origin looking along +y: u = 100 x / y + 50, v = 100 / y + 50.
# Ground points with y > 0 are in front of it.
H = np.array([[100.0, 50.0, 0.0], [0.0, 50.0, 100.0], [0.0, 1.0, 0.0]])


def test_world_to_image_applies_the_matrix_with_perspective_divide():
    u, v = Homography(H).to_image(np.array([1.0, 0.0]), np.array([2.0, 4.0]))

    np.testing.assert_allclose(u, [100.0, 50.0])
    np.testing.assert_allclose(v, [100.0, 75.0])


def test_image_to_world_inverts_world_to_image():
    h = Homography(H)
    xs, ys = np.array([0.5, -2.0, 3.0]), np.array([1.0, 5.0, 10.0])

    back_x, back_y = h.to_world(*h.to_image(xs, ys))

    np.testing.assert_allclose(back_x, xs)
    np.testing.assert_allclose(back_y, ys)


def test_sees_only_points_in_front_that_land_inside_the_image():
    h = Homography(H)
    xs = np.array([0.0, 0.0, 5.0, 0.0])
    ys = np.array([2.0, -2.0, 2.0, 0.5])
    # in view; behind the camera (lands at (50, 0) after the divide);
    # u = 300 is right of a 200 px image; v = 250 is below a 200 px image.

    seen = h.sees(xs, ys, width=200, height=200)

    np.testing.assert_array_equal(seen, [True, False, False, False])


def test_rejects_a_singular_matrix():
    with pytest.raises(ValueError):
        Homography(np.zeros((3, 3)))


def test_keeps_the_shape_of_grid_shaped_inputs():
    xs, ys = np.meshgrid([0.0, 1.0, 2.0], [2.0, 4.0])

    seen = Homography(H).sees(xs, ys, width=200, height=200)

    assert seen.shape == (2, 3)
