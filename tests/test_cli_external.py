"""space-usage eval --dataset: the external evaluation-only datasets."""

import json

import numpy as np

from space_usage_heatmaps import cli
from space_usage_heatmaps.cli import main
from space_usage_heatmaps.evaluation.baseline.recipe import recipe_metadata
from tests.evaluation.infrastructure.test_caviar_reader import write_clip
from tests.evaluation.infrastructure.test_wildtrack_reader import FRAMES, write_dataset

BASE = "detector: yolov8n\ntracker: bytetrack\ninput_size: 640\nengine: ultralytics\nstride: 1\n"


def cache_truth_tracks(data, config, scene, camera, samples):
    """The truth boxes as the config's cached tracks: the detector never runs."""
    cache = data / "cache" / "tracks" / cli.cache_key(config)
    cache.mkdir(parents=True, exist_ok=True)
    xyxy = samples.boxes.copy()
    xyxy[:, 2:] += xyxy[:, :2]
    np.savez(
        cache / f"scene_{scene}_camera_{camera}.npz",
        track_ids=samples.track_ids,
        frames=samples.frames,
        boxes=xyxy,
        meta=np.array(json.dumps(recipe_metadata(config))),
    )


def setup(tmp_path, monkeypatch, foot_point="bottom"):
    data = tmp_path / "data"
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(data))
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "configs" / "best.yaml"
    path.parent.mkdir()
    path.write_text(BASE + f"foot_point: {foot_point}\n")
    return data, path, cli.load_config(path)


def test_wildtrack_reports_the_ground_error_of_the_most_oblique_camera(
    tmp_path, monkeypatch, capsys
):
    data, path, config = setup(tmp_path, monkeypatch)
    write_dataset(data, FRAMES)
    from space_usage_heatmaps.evaluation.infrastructure.wildtrack_reader import WildtrackReader

    samples = WildtrackReader(data).samples("wildtrack", "C5")
    cache_truth_tracks(data, config, "wildtrack", "C5", samples)

    code = main(["eval", "--config", str(path), "--dataset", "wildtrack"])

    assert code == 0
    out = capsys.readouterr().out
    assert "camera C5" in out and "30 cm" in out
    report = json.loads((tmp_path / "reports" / "external-wildtrack-configs_best.json").read_text())
    g = report["ground_position"]
    assert g["camera"] == "C5" and g["matched"] == 2 and g["recall"] == 1.0
    assert g["median_m"] < 0.05  # the truth boxes' bottoms sit on their positions
    assert g["target_m"] == 0.30 and g["meets_target"] is True
    assert isinstance(g["camera_height_m"], float)
    assert g["head_rule"] is None  # bottom foot point: no head rule to guard
    plane = report["ground_plane"]
    assert plane["matched"] == 2 and plane["gate_m"] == 1.0 and plane["median_m"] < 0.05
    # Both views print median, p90, recall and precision together.
    for view in ("IoU >= 0.5", "within 1.0 m"):
        line = next(x for x in out.splitlines() if view in x)
        assert all(k in line for k in ("median", "p90", "recall", "precision"))
    assert report["tracking"]["pooled"]["configs_best"]["idf1"] == 1.0


def test_caviar_reports_the_tracking_guards_per_clip_and_pooled(tmp_path, monkeypatch, capsys):
    data, path, config = setup(tmp_path, monkeypatch)
    write_clip(data, "Walk1cor")
    write_clip(data, "Walk2cor")
    from space_usage_heatmaps.evaluation.infrastructure.caviar_reader import CaviarReader

    for clip in ("Walk1cor", "Walk2cor"):
        cache_truth_tracks(data, config, clip, "cor", CaviarReader(data).samples(clip, "cor"))

    code = main(["eval", "--config", str(path), "--dataset", "caviar"])

    assert code == 0
    report = json.loads((tmp_path / "reports" / "external-caviar-configs_best.json").read_text())
    assert set(report["tracking"]["scenes"]["configs_best"]) == {"Walk1cor/cor", "Walk2cor/cor"}
    assert report["tracking"]["pooled"]["configs_best"]["hota"] == 1.0
    assert "ground_position" not in report
    assert "evaluation only" in report["use"]


def test_dataset_needs_a_config(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    code = main(["eval", "--source", "gt-world", "--dataset", "caviar"])

    assert code == 2
