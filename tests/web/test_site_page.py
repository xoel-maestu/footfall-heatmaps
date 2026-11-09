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
