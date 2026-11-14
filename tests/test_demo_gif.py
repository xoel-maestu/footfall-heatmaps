"""The pure parts of scripts/demo-gif.py on a tiny synthetic input: which video
frames it shows, the k-suppressed dwell it accumulates up to each one, the
colour cap and the panel sizes. No video, no dataset."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from footfall_heatmaps.occupancy.domain.grid import GridSpec

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("demo_gif", SCRIPTS / "demo-gif.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frames_step_through_the_window_at_the_playback_speed(demo):
    frames = demo.frame_indices(start_s=10, seconds=2, speed=2, fps=10, video_fps=30)

    assert frames.tolist() == list(range(300, 420, 6))


def test_twelve_seconds_at_ten_fps_are_120_gif_frames(demo):
    frames = demo.frame_indices(start_s=480, seconds=12, speed=2, fps=10, video_fps=30)

    assert len(frames) == 120
    assert frames[-1] - frames[0] == 119 * 6


def test_a_cell_shows_only_once_k_distinct_ids_have_been_in_it(demo):
    spec = GridSpec(0.0, 0.0, 1.0, 2, 1)
    # cell 0: ids 0..5, one sample each at frames 0..5; cell 1: ids 10..13
    ids = np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13])
    frames = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    xs = np.array([0.5] * 6 + [1.5] * 4)
    ys = np.full(10, 0.5)

    grids = demo.cumulative_dwell(spec, ids, frames, xs, ys, video_fps=2, cutoffs=[3, 4, 5], k=5)

    assert [g.tolist() for g in grids] == [[[0.0, 0.0]], [[2.5, 0.0]], [[3.0, 0.0]]]


def test_samples_after_the_cutoff_never_count(demo):
    spec = GridSpec(0.0, 0.0, 1.0, 1, 1)
    ids = np.arange(5)
    frames = np.array([0, 0, 0, 0, 9])
    xs = ys = np.full(5, 0.5)

    early, late = demo.cumulative_dwell(spec, ids, frames, xs, ys, 1, [8, 9], k=5)

    assert early.tolist() == [[0.0]] and late.tolist() == [[5.0]]


def test_the_colour_cap_is_the_nearest_rank_p99_of_cells_above_zero(demo):
    grid = np.array([[0.0] * 50 + list(range(1, 201))])

    assert demo.colour_cap(grid) == 198.0
    assert demo.colour_cap(np.zeros((2, 2))) == 0.0


def test_the_ramp_runs_from_the_viewers_lightest_to_darkest_green(demo):
    assert demo.ramp_rgb(0.0) == (0xDD, 0xEF, 0xE6)
    assert demo.ramp_rgb(1.0) == (0x03, 0x31, 0x1F)
    assert demo.ramp_rgb(5.0) == demo.ramp_rgb(1.0)


def test_the_panels_fit_the_width_and_keep_the_floors_square_cells(demo):
    lay = demo.layout(width=1200, video_w=1920, video_h=1080, nx=76, ny=92)

    assert lay.width == 1200
    assert lay.video_x + lay.video_w <= lay.floor_x
    assert lay.floor_x + lay.cell_px * 76 <= 1200
    assert lay.cell_px * 92 <= lay.panel_h
    assert lay.video_h == round(lay.video_w * 1080 / 1920)
    assert lay.height > lay.panel_h  # room for the caption


def test_small_changes_keep_the_last_pixel_and_large_ones_take_the_new(demo):
    previous = np.array([[[100, 100, 100], [100, 100, 100]]], dtype=np.uint8)
    current = np.array([[[105, 95, 100], [100, 140, 100]]], dtype=np.uint8)

    out = demo.steady(previous, current, threshold=12)

    assert out.tolist() == [[[100, 100, 100], [100, 140, 100]]]
