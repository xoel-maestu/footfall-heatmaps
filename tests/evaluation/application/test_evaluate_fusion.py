import numpy as np
import pytest

from footfall_heatmaps.evaluation.application.evaluate_fusion import (
    FUSION_TRUTHS,
    evaluate_fusion,
)
from footfall_heatmaps.evaluation.domain.run_config import RunConfig
from footfall_heatmaps.evaluation.domain.tracks import Tracks
from footfall_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
from footfall_heatmaps.occupancy.domain.fusion import Fusion
from tests.conftest import gt_row, write_scene
from tests.evaluation.application.test_evaluate_config import (
    InMemoryTrackSource,
    tracks_from_rows,
)

CFG = RunConfig("yolov8n", "bytetrack", 640, "bottom", "ultralytics", 1)


def both(tracks_7, tracks_8):
    return InMemoryTrackSource({("001", "7"): tracks_7, ("001", "8"): tracks_8})


def joined(a, b):
    return Tracks(
        np.concatenate([a.track_ids, b.track_ids]),
        np.concatenate([a.frames, b.frames]),
        np.vstack([a.boxes, b.boxes]),
    )


@pytest.fixture
def rows(rows):
    """conftest's rows, with track 9 off a cell boundary (6.0 m rounds either way)."""
    return [*rows[:-1], gt_row(8, 9, 0, 6.03, 9.07)]


@pytest.fixture
def reader(tmp_path, rows):
    # Cameras 7 and 8 share one calibration: every covered cell is an overlap.
    return NvidiaMtmcReader(write_scene(tmp_path, rows, cameras=(7, 8)))


def test_a_person_both_cameras_see_is_counted_once(reader, rows):
    seen_by_7 = tracks_from_rows(rows)  # tracks 1 and 2
    seen_by_8 = joined(seen_by_7, tracks_from_rows(rows, camera=8))  # the same, and track 9
    source = both(seen_by_7, seen_by_8)

    r = evaluate_fusion(
        reader, [("001", "7")], (CFG, source), (CFG, source), 0.5, Fusion("visibility", 1), 20
    )

    assert list(r.results) == list(FUSION_TRUTHS)
    for truth in ("full@joint", "full@overlap", "visible@joint"):
        dwell = r.results[truth].scenes[0][0]["dwell"]
        assert dwell["mass_error"].value == pytest.approx(0, abs=1e-12)
        assert dwell["tvd"].value == pytest.approx(0, abs=1e-12)
    s = r.scenes[0]
    assert s.cameras == ("7", "8") and s.overlap_cells == s.joint_cells > 0
    # Adding the two cameras counts tracks 1 and 2 twice: 15 of 16 samples.
    assert s.sum_overlap_mass_error == pytest.approx(15 / 16)


def test_the_single_camera_baseline_misses_what_only_the_other_camera_sees(reader, rows):
    seen_by_7 = tracks_from_rows(rows)
    source = both(seen_by_7, joined(seen_by_7, tracks_from_rows(rows, camera=8)))

    r = evaluate_fusion(reader, [("001", "7")], (CFG, source), (CFG, source), 0.5, Fusion(), 20)

    full = r.results["full@anchor"]
    assert full.scenes[0][1]["dwell"]["mass_error"].value == pytest.approx(1 / 16)
    assert full.delta_pooled["dwell"]["mass_error"].value == pytest.approx(-1 / 16)
    assert r.scenes[0].anchor == "7"


def test_cameras_can_be_chosen(reader, rows):
    source = both(tracks_from_rows(rows), tracks_from_rows(rows, camera=8))

    r = evaluate_fusion(
        reader, [("001", "7")], (CFG, source), (CFG, source), 0.5, Fusion(), 5, cameras=["7"]
    )

    assert r.scenes[0].cameras == ("7",) and r.scenes[0].overlap_cells == 0
    assert ("001", "8") not in source.calls


def split_track_2(tracks):
    """Track 2, standing still, lost at frame 2 and back as track 6 on the same
    spot: one person, two ids in one cell."""
    t = tracks
    keep = ~((t.track_ids == 2) & (t.frames == 2))
    ids = np.where((t.track_ids == 2) & (t.frames > 2), 6, t.track_ids)
    return Tracks(ids[keep], t.frames[keep], t.boxes[keep])


def test_stitching_joins_a_lost_track_for_passage_and_leaves_dwell_byte_identical(reader, rows):
    from dataclasses import replace

    source = both(split_track_2(tracks_from_rows(rows)), tracks_from_rows(rows, camera=8))
    stitched = replace(CFG, stitch={"gap_s": 2.0, "radius_m": 0.5})
    plain = (CFG, source)

    r = evaluate_fusion(
        reader, [("001", "7")], (stitched, source), plain, 0.5, Fusion(), 20, compare_run=plain
    )
    base = evaluate_fusion(reader, [("001", "7")], plain, plain, 0.5, Fusion(), 20)

    for truth, res in r.results.items():
        for m, e in res.delta_pooled["dwell"].items():
            assert (e.value, e.lo, e.hi) in ((0.0, 0.0, 0.0), (None, None, None)), (truth, m)
    np.testing.assert_array_equal(r.fused["001"].dwell, base.fused["001"].dwell)
    # The fused map is compared with the plain fused map, not the single camera.
    assert r.results["full@joint"].pooled[1] == base.results["full@joint"].pooled[0]
    joined = r.results["full@joint"].delta_pooled["passage"]["mass_error"].value
    assert joined < 0
    assert (r.scenes[0].camera_tracks, base.scenes[0].camera_tracks) == (3, 4)


def test_compare_fusion_pairs_two_weightings_of_the_same_tracks(reader, rows):
    import math

    seen_by_7 = tracks_from_rows(rows)
    source = both(seen_by_7, joined(seen_by_7, tracks_from_rows(rows, camera=8)))
    run = (CFG, source)
    mean, top = Fusion("mean"), Fusion("visibility", math.inf)

    r = evaluate_fusion(
        reader, [("001", "7")], run, run, 0.5, mean, 20, compare_run=run, compare_fusion=top
    )
    assert source.calls.count(("001", "8")) == 1  # the same run's tracks are read once
    alone = {f: evaluate_fusion(reader, [("001", "7")], run, run, 0.5, f, 20) for f in (mean, top)}

    for truth, res in r.results.items():
        assert res.pooled[0] == alone[mean].results[truth].pooled[0], truth
        assert res.pooled[1] == alone[top].results[truth].pooled[0], truth
    np.testing.assert_array_equal(r.fused["001"].dwell, alone[mean].fused["001"].dwell)
    d = r.results["full@joint"].delta_pooled["dwell"]["mass_error"].value
    assert d != 0  # the mean dilutes what only camera 8 sees
