import numpy as np
import pytest

from footfall_heatmaps.occupancy.domain.grid import Grid, GridSpec


def spec(nx=4, ny=3, cell=0.5):
    return GridSpec(origin_x=0.0, origin_y=0.0, cell_m=cell, nx=nx, ny=ny)


def test_dwell_adds_one_over_fps_per_sample():
    ids = np.array([1, 1, 1, 2])
    xs = np.array([0.1, 0.2, 0.6, 0.1])
    ys = np.array([0.1, 0.4, 1.2, 0.3])

    grid = Grid.from_samples(spec(), ids, xs, ys, fps=10)

    expected = np.zeros((3, 4))
    expected[0, 0] = 0.3  # three samples in cell (row 0, col 0)
    expected[2, 1] = 0.1  # x=0.6 -> col 1, y=1.2 -> row 2
    np.testing.assert_allclose(grid.dwell, expected)


def test_passage_counts_each_track_once_per_cell():
    ids = np.array([1, 1, 1, 2, 2])
    xs = np.array([0.1, 0.2, 0.6, 0.1, 0.3])
    ys = np.array([0.1, 0.4, 0.1, 0.3, 0.2])

    grid = Grid.from_samples(spec(), ids, xs, ys, fps=30)

    assert grid.passage[0, 0] == 2  # tracks 1 and 2, track 1 seen twice there
    assert grid.passage[0, 1] == 1
    assert grid.passage.sum() == 3


def test_samples_outside_the_grid_are_dropped_and_counted():
    ids = np.array([1, 2, 3])
    xs = np.array([-0.01, 2.0, 1.99])
    ys = np.array([0.0, 0.0, 1.49])

    grid = Grid.from_samples(spec(), ids, xs, ys, fps=1)

    assert grid.dropped == 2
    assert grid.dwell.sum() == 1
    assert grid.dwell[2, 3] == 1


def test_cell_centres_are_in_metres_from_the_origin():
    s = GridSpec(origin_x=-1.0, origin_y=2.0, cell_m=0.5, nx=2, ny=1)

    xs, ys = s.cell_centres()

    np.testing.assert_allclose(xs, [[-0.75, -0.25]])
    np.testing.assert_allclose(ys, [[2.25, 2.25]])


def test_covering_snaps_the_extent_to_two_metre_blocks():
    s = GridSpec.covering(xmin=-3.1, ymin=0.5, xmax=4.2, ymax=1.0, cell_m=0.25)

    assert (s.origin_x, s.origin_y) == (-4.0, 0.0)
    assert s.nx * s.cell_m == 10.0  # -4 .. 6
    assert s.ny * s.cell_m == 2.0
    assert s.nx % 8 == 0 and s.ny % 8 == 0


def test_cell_size_must_tile_one_metre():
    with pytest.raises(ValueError):
        GridSpec(origin_x=0, origin_y=0, cell_m=0.3, nx=1, ny=1)


def test_no_samples_give_empty_layers():
    empty = np.array([])

    grid = Grid.from_samples(spec(), empty.astype(int), empty, empty, fps=30)

    assert grid.dwell.sum() == grid.passage.sum() == grid.dropped == 0
