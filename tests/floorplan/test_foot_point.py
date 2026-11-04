import numpy as np
import pytest

from space_usage_heatmaps.floorplan.domain.foot_point import image_points

# x1, y1, x2, y2 in pixels
BOXES = np.array([[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 4.0, 2.0]])


def test_centroid_is_the_box_centre():
    u, v = image_points(BOXES, "centroid")

    np.testing.assert_allclose(u, [20.0, 2.0])
    np.testing.assert_allclose(v, [40.0, 1.0])


def test_bottom_centre_is_the_middle_of_the_lower_edge():
    u, v = image_points(BOXES, "bottom-centre")

    np.testing.assert_allclose(u, [20.0, 2.0])
    np.testing.assert_allclose(v, [60.0, 2.0])


def test_an_unknown_point_is_an_error():
    with pytest.raises(ValueError, match="foot point"):
        image_points(BOXES, "head")
