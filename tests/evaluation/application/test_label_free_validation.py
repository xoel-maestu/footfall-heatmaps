import numpy as np
import pytest

from footfall_heatmaps.evaluation.application.label_free_validation import (
    LABEL_FREE,
    TRUE_ERRORS,
    validate_label_free,
)
from footfall_heatmaps.evaluation.domain.run_config import RunConfig
from footfall_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
from tests.conftest import write_scene
from tests.evaluation.application.test_evaluate_config import (
    InMemoryTrackSource,
    tracks_from_rows,
)

CFG = RunConfig("yolov8n", "bytetrack", 640, "bottom", "ultralytics", 1)


def test_each_scene_gets_its_label_free_metrics_next_to_its_true_error(tmp_path, rows):
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows))
    source = InMemoryTrackSource({("001", "7"): tracks_from_rows(rows)})

    v = validate_label_free(reader, [("001", "7")], CFG, source, cell=0.5)

    (scene,) = v.scenes
    assert (scene.scene, scene.camera) == ("001", "7")
    assert scene.true_error["tvd"] == pytest.approx(0, abs=1e-12)  # truth boxes as tracks
    assert scene.true_error["mass_error"] == pytest.approx(0, abs=1e-12)
    assert scene.label_free.tracks == 2
    assert set(v.correlations) == {"scene", "block"}
    for level in v.correlations.values():
        assert set(level) == {f"{m}~{e}" for m in LABEL_FREE for e in TRUE_ERRORS}
        assert all(set(c) == {"spearman", "pearson", "n"} for c in level.values())
    assert v.correlations["scene"][f"{LABEL_FREE[0]}~tvd"]["n"] == 1
    assert len(v.blocks["tvd"]) == len(v.blocks[LABEL_FREE[0]]) > 0


def test_correlations_pool_every_scene_given(tmp_path, rows):
    reader = NvidiaMtmcReader(write_scene(tmp_path, rows))
    t = tracks_from_rows(rows)
    shifted = type(t)(t.track_ids, t.frames, t.boxes + np.array([0.0, 3.0, 0.0, 3.0]))
    source = InMemoryTrackSource({("001", "7"): t})
    other = InMemoryTrackSource({("001", "7"): shifted})

    a = validate_label_free(reader, [("001", "7")], CFG, source, cell=0.5)
    b = validate_label_free(reader, [("001", "7")], CFG, other, cell=0.5)

    assert b.scenes[0].true_error["tvd"] > a.scenes[0].true_error["tvd"]
