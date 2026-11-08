"""CalibrateSite: a site's fusion weights from a calibration window, frozen once.

The weights use what the cameras record and where they look (their dwell over
the window and their floor coverage), the same formula as the evaluation's
fusion (occupancy.domain.fusion.fusion_weights), and never any ground truth.
"""

import numpy as np
import pytest

from space_usage_heatmaps.occupancy.application.calibrate_site import CalibrateSite
from space_usage_heatmaps.occupancy.domain.fusion import Fusion
from space_usage_heatmaps.occupancy.domain.grid import GridSpec

T0 = 1_793_955_600
SPEC = GridSpec(origin_x=0.0, origin_y=0.0, cell_m=0.5, nx=4, ny=2)
COVERAGE = np.zeros((2, 2, 4), bool)
COVERAGE[:, 0, :2] = True  # both cameras see cells (0, 0) and (0, 1)
COVERAGE[1, 1, :] = True  # b alone sees the second row


def crowd(n, x=0.1, y=0.1):
    return np.arange(n), np.full(n, x), np.full(n, y)


def calibration(window_s=60.0, fusion=Fusion("visibility", 1.0)):
    return CalibrateSite(["a", "b"], COVERAGE, SPEC, fusion, window_s)


def test_weights_follow_each_camera_share_of_the_seen_dwell():
    c = calibration()

    c.add("a", T0, *crowd(4), seconds=1.0)
    c.add("b", T0, *crowd(2), seconds=1.0)
    w = c.weights()

    assert w[:, 0, 0] == pytest.approx([2 / 3, 1 / 3])  # ratios 1 and .5, power 1
    assert w[:, 1, 2] == pytest.approx([0.0, 1.0])  # b alone covers it
    assert w[:, 0, 3].sum() == 0  # nobody covers it


def test_samples_after_the_window_do_not_count():
    c = calibration(window_s=60.0)

    c.add("a", T0, *crowd(4), seconds=1.0)
    c.add("b", T0, *crowd(4), seconds=1.0)
    assert not c.complete
    c.add("b", T0 + 61, *crowd(40), seconds=1.0)
    c.add("a", T0 + 61, *crowd(0), seconds=1.0)

    assert c.complete
    assert c.weights()[:, 0, 0] == pytest.approx([0.5, 0.5])


def test_the_window_starts_at_the_first_sample_of_any_camera():
    c = calibration(window_s=60.0)

    c.add("a", T0 + 30, *crowd(1), seconds=1.0)
    c.add("b", T0 + 95, *crowd(1), seconds=1.0)  # past T0 + 30 + 60

    assert c.dwell_s()["b"] == 0 and c.dwell_s()["a"] == 1.0


def test_the_coverage_must_match_the_cameras_and_grid():
    with pytest.raises(ValueError, match="coverage"):
        CalibrateSite(["a"], COVERAGE, SPEC, Fusion(), 60.0)
