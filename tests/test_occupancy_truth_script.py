"""scripts/occupancy-truth.py: the H4 truth of a slot's mean occupancy, on hand-made truth."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from space_usage_heatmaps.occupancy.domain.grid import GridSpec

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "occupancy-truth.py"
T0 = 1_793_955_600  # 2026-11-06T09:00:00Z
SPEC = GridSpec(origin_x=0.0, origin_y=0.0, cell_m=0.5, nx=4, ny=2)


def script():
    spec = importlib.util.spec_from_file_location("occupancy_truth", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_truth_counts_person_seconds_inside_the_coverage_and_the_slot():
    fps = 10.0
    # two people for 20 frames: one inside the covered cell (0, 0), one outside it
    frames = np.concatenate([np.arange(20), np.arange(20)])
    world = np.array([[0.1, 0.1]] * 20 + [[1.9, 0.9]] * 20)
    coverage = np.zeros((2, 4), bool)
    coverage[0, 0] = True

    got = script().truth_person_seconds(SPEC, frames, world, fps, coverage, T0, T0)

    assert got == pytest.approx(2.0)  # 20 frames at 10 fps


def test_truth_leaves_out_frames_outside_the_slot():
    frames = np.arange(0, 2000, 100)  # 0 to 190 s at 10 fps
    world = np.full((len(frames), 2), 0.1)
    coverage = np.ones((2, 4), bool)

    # the video started 100 s before the slot: only frames at 100 s and later count
    got = script().truth_person_seconds(SPEC, frames, world, 10.0, coverage, T0 - 100, T0)

    assert got == pytest.approx(10 / 10.0)


def test_the_page_value_rounds_halves_up_as_intl_does():
    assert script().on_page(5.24958) == 5.3  # the API sends 5.25
    assert script().on_page(4.6789) == 4.7
