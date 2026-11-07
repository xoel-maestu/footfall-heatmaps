"""The legend and summary texts of the viewer (legendState in web/static/index.html), run in Node.

An empty map must never sit next to the numbers of the previous map: with no
data (live unavailable, no slot, fused map unavailable) every number reads as a
dash."""

from tests.web.page_js import js, needs_node, run_js

pytestmark = needs_node

NUMBERS = ["legend-min", "legend-max", "sum-total", "sum-cells", "sum-max", "sum-cell", "sum-time"]


def legend_state(data, layer="dwell", source="slot"):
    return run_js(
        ["colour-scale", "legend-state"], f"legendState({js(data)}, {js(layer)}, {js(source)})"
    )


def slot(dwell):
    ny, nx = len(dwell), len(dwell[0])
    return {
        "shape": [ny, nx],
        "cell_m": 0.5,
        "origin": [0, 0],
        "dwell": dwell,
        "passage": [[0] * nx for _ in range(ny)],
        "label": "2026-11-07 10:00 to 10:15 UTC",
    }


def test_no_data_clears_every_number():
    s = legend_state(None, source="live")
    assert {k: s[k] for k in NUMBERS} == dict.fromkeys(NUMBERS, "-")
    assert s["legend-unit"] == ""


def test_no_data_clears_the_fused_details():
    s = legend_state(None, source="fused")
    assert (s["sum-cameras"], s["sum-fusion"], s["sum-passage-ids"]) == ("-", "-", "-")


def test_no_data_keeps_the_note_of_the_source_shown():
    assert legend_state(None, source="slot")["layer-note"].startswith("Slot layer")


def test_data_fills_the_legend_and_summary():
    s = legend_state(slot([[0.0, 2.0], [4.0, 0.0]]))
    assert s["legend-min"] == "0"
    assert s["legend-max"] == "4"
    assert s["sum-total"] == "6 person-seconds"
    assert s["sum-cells"] == "2 of 4"
    assert s["sum-max"] == "4 person-seconds"
    assert s["sum-cell"] == "0.5 m"
    assert s["sum-time"] == "2026-11-07 10:00 to 10:15 UTC"
    assert "sum-cameras" not in s  # set by loadFused, not by the legend


def test_a_map_with_no_cell_above_zero_shows_zeros_not_dashes():
    s = legend_state(slot([[0.0, 0.0]]))
    assert (s["sum-cells"], s["sum-max"]) == ("0 of 2", "0 person-seconds")


def test_the_fused_passage_says_it_counts_each_cameras_ids():
    # serve --fused refuses a global-id passage: the layer is always E2a's.
    assert run_js(["passage-ids"], "PASSAGE_IDS") == "each camera's ids, weighted as in E2a"
