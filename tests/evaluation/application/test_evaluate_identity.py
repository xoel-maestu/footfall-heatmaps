import numpy as np
import pytest

from space_usage_heatmaps.evaluation.application.evaluate_identity import evaluate_identity
from space_usage_heatmaps.evaluation.domain.run_config import RunConfig
from space_usage_heatmaps.evaluation.domain.tracks import Tracks
from space_usage_heatmaps.evaluation.infrastructure.floor_identity import floor_identity
from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
from space_usage_heatmaps.occupancy.domain.association import NONE, Association
from space_usage_heatmaps.occupancy.domain.fusion import Fusion
from tests.conftest import gt_row, write_scene
from tests.evaluation.application.test_evaluate_config import (
    InMemoryTrackSource,
    tracks_from_rows,
)
from tests.evaluation.application.test_evaluate_fusion import joined

CFG = RunConfig("yolov8n", "bytetrack", 640, "bottom", "ultralytics", 1)
COLOCATE = Association("colocate", min_overlap_s=0.3)


@pytest.fixture
def rows(rows):
    """conftest's rows, with track 9 off a cell boundary (6.0 m rounds either way)."""
    return [*rows[:-1], gt_row(8, 9, 0, 6.03, 9.07)]


@pytest.fixture
def reader(tmp_path, rows):
    # Cameras 7 and 8 share one calibration: both see tracks 1 and 2, only 8 sees 9.
    return NvidiaMtmcReader(write_scene(tmp_path, rows, cameras=(7, 8)))


def renumbered(t, offset):
    return Tracks(t.track_ids + offset, t.frames, t.boxes)


@pytest.fixture
def source(rows):
    seen = tracks_from_rows(rows)  # tracks 1 and 2, exactly on the truth
    by_8 = joined(renumbered(seen, 100), tracks_from_rows(rows, camera=8))
    return InMemoryTrackSource({("001", "7"): seen, ("001", "8"): by_8})


def run(reader, source, associations, **kw):
    return evaluate_identity(
        reader,
        ["001"],
        (CFG, source),
        Fusion("visibility", 1.0),
        associations,
        floor_identity,
        n_resamples=20,
        **kw,
    )


def test_without_association_every_person_is_predicted_twice(reader, source):
    r = run(reader, source, {"e2b": COLOCATE, "baseline": NONE})

    # 16 truth points; tracks 1 and 2 predicted twice (15 points each), 9 once:
    # IDF1 = 2 * 16 / (16 + 31).
    assert r.idf1.scenes[0][1].value == pytest.approx(32 / 47)
    assert r.idf1.scenes[0][0].value == pytest.approx(1.0)
    assert r.idf1.delta_pooled.value == pytest.approx(15 / 47)
    s = r.scenes[0]
    assert s.cameras == ("7", "8") and s.truth_people == 3 and s.truth_points == 16
    assert s.global_ids == {"e2b": 3, "baseline": 5}
    assert s.counts["baseline"]["idp"] == pytest.approx(16 / 31)
    assert s.counts["baseline"]["idr"] == pytest.approx(1.0)


def test_the_oracle_joins_tracks_by_their_true_person(reader, source):
    r = run(reader, source, {"baseline": NONE, "none-again": NONE}, oracle=True)

    assert r.names == ("baseline", "none-again", "oracle")
    assert r.idf1.scenes[0][2].value == pytest.approx(1.0)


def test_global_ids_count_each_person_once_in_the_passage_layer(reader, source):
    r = run(reader, source, {"e2b": COLOCATE, "baseline": NONE})

    full = r.passage["full@joint"]
    # Global-id passage is exact; the fused map's dwell is the E2a one.
    assert full.scenes[0][0]["passage"]["mass_error"].value == pytest.approx(0, abs=1e-12)
    assert full.delta_pooled["dwell"]["tvd"].value == 0.0
    assert full.delta_pooled["dwell"]["mass_error"].value == 0.0
    assert full.scenes[0][0]["dwell"]["mass_error"].value == pytest.approx(0, abs=1e-12)
    assert r.fused["001"].passage.sum() > 0


def test_the_camera_tracks_are_left_as_they_were(reader, source, rows):
    before = tracks_from_rows(rows)

    run(reader, source, {"e2b": COLOCATE, "baseline": NONE})

    after = source.tracks("001", "7")
    np.testing.assert_array_equal(after.track_ids, before.track_ids)
    np.testing.assert_array_equal(after.boxes, before.boxes)
