"""The viewer page in site mode (serve --site), its pure parts run in Node."""

from tests.web.page_js import js, needs_node, run_js

pytestmark = needs_node


def rows(cameras):
    return run_js(["camera-rows"], f"cameraRows({js(cameras)})")


def note(source, site):
    return run_js(
        ["colour-scale", "legend-state"],
        f"legendState(null, 'dwell', {js(source)}, {js(site)})['layer-note']",
    )


def test_camera_rows_read_as_text_with_a_dash_for_unknown_rates():
    got = rows([
        {"id": "635", "state": "recording", "processed_fps": 3.04, "frames_skipped": 12,
         "source_drops": 0},
        {"id": "636", "state": "not recording", "processed_fps": None, "frames_skipped": 0,
         "source_drops": 2},
    ])  # fmt: skip

    assert got == [["635", "Recording", "3.0", "12", "0"], ["636", "Not recording", "-", "0", "2"]]


def test_site_notes_say_the_map_is_every_camera_fused_with_frozen_weights():
    for source in ("live", "slot"):
        text = note(source, True)
        assert "every camera of the site" in text and "calibration" in text
        assert "fewer than 5 distinct track ids were seen" in text
        assert "does not guarantee anonymity" in text


def test_camera_notes_stay_as_they_were():
    assert note("slot", False).startswith("Slot layer")


def test_k_notes_state_the_measured_violation_rate_of_their_layer():
    """H6 (phase 7): the page states how often a shown cell had fewer than 5
    people in truth, measured on the tune scenes, per layer and per mode."""
    assert "19 %" in note("slot", False)
    assert "18 %" in note("slot", True)
    assert "58 %" in note("live", False)
    assert "48 %" in note("live", True)
    for source in ("live", "slot"):
        for site in (False, True):
            assert "fewer than 5 different people" in note(source, site)


def test_k_notes_name_the_tune_scenes_they_were_measured_on():
    """H8 (phase 7): the rates come from the tune scenes 071-075, so the notes
    say so; "test" or "held-out" scenes would claim a held-out measurement."""
    for source in ("live", "slot"):
        for site in (False, True):
            text = note(source, site)
            assert "On the tune scenes" in text
            assert "test scenes" not in text and "held-out" not in text
