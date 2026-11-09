"""The honest slot reading of the viewer (improvements phase 6, H4), run in Node.

No number on the page is labelled people unless it derives from dwell: the
slot reads as occupancy (people in view on average, peak minute), passage is
in crossings, the top 2 m zones are ranked by dwell, and a stored map is
cropped to its observed extent plus 2 m."""

from tests.web.page_js import js, needs_node, run_js

pytestmark = needs_node

SLOT = {
    "start": "2026-11-06T09:00:00Z",
    "incomplete": True,
    "mean_occupancy": 4.6789,
    "peak_occupancy": 9.12,
    "peak_start": "2026-11-06T09:07:00Z",
    "observed_s": 800.0,
}


def reading(s, site=True):
    return run_js(["reading"], f"slotReading({js(s)}, {js(site)})")


def option(s, site=True):
    return run_js(["reading"], f"slotOption({js(s)}, {js(site)})")


def test_the_slot_reads_as_people_in_view_from_dwell():
    assert reading(SLOT) == {
        "occ-mean-bound": "",
        "occ-mean": "4.7",
        "occ-peak": "9.1 people, 09:07 to 09:08 UTC",
        "occ-observed": "13 min 20 s of 15 min",
    }


def test_the_slot_option_reads_occupancy_not_an_id_count():
    assert option(SLOT) == "09:00 to 09:15, on average 4.7 people in view, incomplete"
    whole = SLOT | {"incomplete": False, "observed_s": 900.0}
    assert option(whole) == "09:00 to 09:15, on average 4.7 people in view"
    assert reading(whole)["occ-observed"] == "15 min of 15 min"


def test_a_slot_without_a_reading_says_so_and_shows_dashes():
    bare = SLOT | {"mean_occupancy": None, "peak_occupancy": None, "peak_start": None,
                   "observed_s": None}  # fmt: skip
    assert option(bare) == "09:00 to 09:15, occupancy not recorded, incomplete"
    assert reading(bare) == {
        "occ-mean-bound": "", "occ-mean": "-", "occ-peak": "-",
        "occ-observed": "Not recorded for this slot",
    }  # fmt: skip
    assert reading(None)["occ-observed"] == "-"


def test_passage_is_in_crossings_and_says_an_id_is_not_a_person():
    data = {"shape": [1, 2], "cell_m": 0.5, "origin": [0, 0], "dwell": [[1, 0]],
            "passage": [[6, 5]], "label": "x"}  # fmt: skip
    s = run_js(["colour-scale", "legend-state"], f"legendState({js(data)}, 'passage', 'slot')")
    assert s["legend-title"] == "Passage (crossings)"
    assert s["sum-total"] == "11 crossings"
    assert "people" not in s["sum-total"] + s["sum-max"] + s["legend-unit"]
    assert "not a count of people" in s["layer-note"]


def grid(ny, nx, cells, value=1.0):
    g = [[0.0] * nx for _ in range(ny)]
    for i, j in cells:
        g[i][j] = value
    return g


def test_the_crop_is_the_observed_cells_plus_2_m_within_the_grid():
    d = {"shape": [40, 40], "cell_m": 0.5, "origin": [-10, -10],
         "dwell": grid(40, 40, [(10, 12)]), "passage": grid(40, 40, [(20, 30)])}  # fmt: skip
    box = run_js(["crop"], f"cropBox({js(d)})")
    assert box == {"i0": 6, "i1": 25, "j0": 8, "j1": 35}  # 4 cells = 2 m each side
    assert run_js(["crop"], f"extentText({js(d)}, {js(box)})") == "x -6 to 7.5 m, y -7 to 2.5 m"

    edge = d | {"dwell": grid(40, 40, [(0, 39)]), "passage": grid(40, 40, [])}
    assert run_js(["crop"], f"cropBox({js(edge)})") == {"i0": 0, "i1": 5, "j0": 35, "j1": 40}
    empty = d | {"dwell": grid(40, 40, []), "passage": grid(40, 40, [])}
    assert run_js(["crop"], f"cropBox({js(empty)})") == {"i0": 0, "i1": 40, "j0": 0, "j1": 40}


def test_zones_are_the_2_m_blocks_ranked_by_dwell_with_their_coordinates():
    dwell = grid(8, 12, [])
    for i in range(4):
        for j in range(4):
            dwell[i][j] = 1.0  # zone (0, 0): 16 person-s
    dwell[5][9] = 30.0  # zone (1, 2): 30 person-s
    dwell[7][5] = 2.0  # zone (1, 1): 2 person-s
    d = {"shape": [8, 12], "cell_m": 0.5, "origin": [-2, 4], "dwell": dwell}

    z = run_js(["zones"], f"topZones({js(d)})")

    assert z["zones"] == 3 and z["total"] == 48
    assert [(r["x0"], r["x1"], r["y0"], r["y1"], r["dwell"]) for r in z["rows"]] == [
        (2, 4, 6, 8, 30), (-2, 0, 4, 6, 16), (0, 2, 6, 8, 2)
    ]  # fmt: skip
    t = run_js(["zones"], f"zoneTable(topZones({js(d)}))")
    assert t["rows"][0] == ["1", "2 to 4", "6 to 8", "62.5%", "30"]
    assert t["note"].startswith("3 of 3 zones with dwell, together 100.0% of the dwell shown")


def test_the_zone_table_keeps_the_top_ten():
    dwell = grid(4, 48, [])
    for z in range(12):
        dwell[0][4 * z] = float(z + 1)
    d = {"shape": [4, 48], "cell_m": 0.5, "origin": [0, 0], "dwell": dwell}

    t = run_js(["zones"], f"zoneTable(topZones({js(d)}))")

    assert len(t["rows"]) == 10 and t["rows"][0][4] == "12" and t["rows"][-1][4] == "3"
    assert t["note"].startswith("10 of 12 zones with dwell")


def test_the_reading_note_says_sparse_minutes_count_as_zero():
    from space_usage_heatmaps.web.server import PAGE

    assert "Minutes in which fewer than 5 track ids were seen count as 0." in PAGE.read_text()


def test_a_single_camera_note_gives_the_held_out_error_range():
    # H7 (phase 7): one recall factor did not bring a single camera within 10 %,
    # so the reading is approximate, with the error measured on the sealed scenes
    # 076-080 (80 cameras, uncorrected), never on the tune scenes it was fitted on
    note = (
        "One camera: people hidden from it are not counted, so it usually reads low "
        "where the view is occluded. On held-out scenes (80 cameras), a single camera's "
        "average was off by about 14.5 % at the median, and between 43 % below and 5 % "
        "above the true value for 8 cameras in 10. A site of several cameras reads closer."
    )
    assert run_js(["reading"], "oneCameraNote(false)") == note
    assert run_js(["reading"], "oneCameraNote(true)") is None


def test_a_single_camera_slot_reads_about_n_people_never_a_bound():
    one = run_js(["reading"], f"slotReading({js(SLOT)}, false)")
    assert one["occ-mean-bound"] == "about" and one["occ-mean"] == "4.7"
    assert one["occ-peak"] == "about 9.1 people, 09:07 to 09:08 UTC"
    assert run_js(["reading"], f"slotOption({js(SLOT)}, false)") == (
        "09:00 to 09:15, on average about 4.7 people in view, incomplete"
    )


def test_no_reading_claims_a_bound():
    from space_usage_heatmaps.web.server import PAGE

    block = PAGE.read_text().split("// reading:begin")[1].split("// reading:end")[0]
    assert "at least" not in block


def test_a_site_slot_reads_without_a_bound():
    site = run_js(["reading"], f"slotReading({js(SLOT)}, true)")
    assert site["occ-mean-bound"] == "" and site["occ-mean"] == "4.7"
    assert site["occ-peak"] == "9.1 people, 09:07 to 09:08 UTC"
    assert run_js(["reading"], f"slotOption({js(SLOT)}, true)") == (
        "09:00 to 09:15, on average 4.7 people in view, incomplete"
    )


def test_a_slot_without_a_reading_has_no_bound_either():
    bare = SLOT | {"mean_occupancy": None, "peak_occupancy": None, "peak_start": None,
                   "observed_s": None}  # fmt: skip
    assert run_js(["reading"], f"slotReading({js(bare)}, false)")["occ-mean-bound"] == ""
