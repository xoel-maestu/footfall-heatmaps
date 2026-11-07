"""The viewer's colour scale (colourScale in web/static/index.html), run in Node.

A few very hot cells must not wash out the rest of the map: the scale is capped
at the 99th percentile (nearest rank) of the cells above zero, and the cells
above the cap are counted for the legend."""

from tests.web.page_js import js, needs_node, run_js

pytestmark = needs_node


def colour_scale(grid):
    return run_js(["colour-scale"], f"colourScale({js(grid)})")


def test_cap_is_the_99th_percentile_of_the_cells_above_zero():
    # 200 cells at 1..200 and empty cells: nearest rank ceil(.99 * 200) = 198
    grid = [[float(v) for v in range(1, 101)], [float(v) for v in range(101, 201)], [0.0] * 100]
    s = colour_scale(grid)
    assert s == {"max": 200.0, "cap": 198.0, "above": 2, "cells": 200, "total": 20100.0}


def test_a_few_hot_cells_do_not_set_the_scale():
    grid = [[1.0] * 99 + [300.0]] * 3  # 297 cells at 1, 3 at 300
    s = colour_scale(grid)
    assert s["max"] == 300.0
    assert s["cap"] == 1.0
    assert s["above"] == 3


def test_few_cells_keep_the_highest_cell_as_the_cap():
    s = colour_scale([[0.0, 2.0, 5.0]])
    assert (s["cap"], s["above"], s["max"], s["cells"]) == (5.0, 0, 5.0, 2)


def test_an_empty_grid_has_a_zero_scale():
    assert colour_scale([[0.0, 0.0]]) == {"max": 0, "cap": 0, "above": 0, "cells": 0, "total": 0}
