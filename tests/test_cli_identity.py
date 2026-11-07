"""space-usage eval --cameras all --metric mc-idf1: multi-camera IDF1 on the
floor (E2b), with tracks from memory."""

import json

import pytest

from space_usage_heatmaps import cli
from tests.conftest import gt_row, write_scene
from tests.evaluation.application.test_evaluate_config import InMemoryTrackSource, tracks_from_rows
from tests.evaluation.application.test_evaluate_fusion import joined
from tests.evaluation.application.test_evaluate_identity import renumbered

RECIPE = (
    "detector: yolox-s\ntracker: bytetrack\ninput_size: 640\nengine: onnx-cuda\n"
    "stride: 1\nfoot_point: bottom\n"
)
SECTIONS = (
    "fusion:\n  weighting: visibility\n  power: 1\n  zone_m: 2.0\n"
    "association:\n  method: colocate\n  gate_m: 1.0\n  min_overlap_s: 0.3\n"
    "  min_close: 0.5\n"
)


@pytest.fixture
def scene(tmp_path, rows, monkeypatch):
    rows = [*rows[:-1], gt_row(8, 9, 0, 6.03, 9.07)]
    data = write_scene(tmp_path / "data", rows, cameras=(7, 8))
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(data))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(cli.SPLITS, "tiny", (("001", "7"),))
    seen = tracks_from_rows(rows)
    source = InMemoryTrackSource(
        {
            ("001", "7"): seen,
            ("001", "8"): joined(renumbered(seen, 100), tracks_from_rows(rows, 8)),
        }
    )
    monkeypatch.setattr(cli, "track_source", lambda config, reader: source)
    (tmp_path / "assoc.yaml").write_text(RECIPE + SECTIONS)
    (tmp_path / "plain.yaml").write_text(RECIPE)
    return tmp_path


def _run(*extra):
    return cli.main(
        ["eval", "--cameras", "all", "--metric", "mc-idf1", "--config", "assoc.yaml"]
        + ["--resamples", "10", *extra]
    )


def test_a_scene_prints_mc_idf1_with_intervals_against_the_baseline(scene, capsys):
    assert _run("--scene", "001") == 0

    out = capsys.readouterr().out
    assert "multi-camera IDF1 on the floor" in out and "gate 1.0 m" in out
    line = next(x for x in out.splitlines() if x.startswith("associated") and "pooled" in x)
    assert "1.000 [" in line
    assert next(x for x in out.splitlines() if x.startswith("baseline") and "pooled" in x)
    assert next(x for x in out.splitlines() if x.startswith("oracle") and "pooled" in x)
    assert "delta associated - baseline" in out and "target" in out
    report = json.loads((scene / "reports/mc-idf1-assoc-s001-0.5m.json").read_text())
    pooled = report["idf1"]["pooled"]
    assert pooled["associated"]["value"] == pytest.approx(1.0)
    assert pooled["baseline"]["value"] == pytest.approx(32 / 47)
    assert report["association"]["method"] == "colocate"
    assert report["scenes"][0]["global_ids"] == {"associated": 3, "baseline": 5, "oracle": 3}
    assert "full@joint" in report["passage"]


def test_the_fused_export_carries_the_global_id_passage(scene):
    assert _run("--scene", "001") == 0

    export = json.loads((scene / "reports/fused-s001-assoc.json").read_text())
    assert export["passage_ids"] == "global"
    assert export["k"] == 5 and export["association"]["method"] == "colocate"


def test_a_split_pools_its_scenes(scene, capsys):
    assert _run("--split", "tiny") == 0

    assert (scene / "reports/mc-idf1-assoc-tiny-0.5m.json").exists()


def test_mc_idf1_needs_cameras(scene, capsys):
    assert cli.main(["eval", "--metric", "mc-idf1", "--scene", "001"]) == 2
    assert "--metric mc-idf1 goes with --cameras" in capsys.readouterr().err


def test_the_association_section_parses(scene):
    a = cli.load_association("assoc.yaml")

    assert (a.method, a.gate_m, a.min_overlap_s, a.handover_s) == ("colocate", 1.0, 0.3, None)
    with pytest.raises(ValueError, match="no association section"):
        cli.load_association("plain.yaml")
    assert cli.load_config("assoc.yaml") == cli.load_config("plain.yaml")


def test_wildtrack_scores_every_camera_with_its_own_recipe(tmp_path, monkeypatch, capsys):
    from space_usage_heatmaps.evaluation.domain.tracks import Tracks
    from space_usage_heatmaps.evaluation.infrastructure.wildtrack_reader import WildtrackReader
    from tests.evaluation.infrastructure.test_wildtrack_reader import FRAMES, write_dataset

    data = write_dataset(tmp_path / "data", FRAMES)
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(data))
    monkeypatch.chdir(tmp_path)
    samples = WildtrackReader(data).samples("wildtrack", "C5")
    xyxy = samples.boxes.copy()
    xyxy[:, 2:] += xyxy[:, :2]
    source = InMemoryTrackSource(
        {("wildtrack", "C5"): Tracks(samples.track_ids, samples.frames, xyxy)}
    )
    recipes = []
    monkeypatch.setattr(cli, "track_source", lambda c, r: recipes.append(c) or source)
    (tmp_path / "assoc.yaml").write_text(RECIPE + SECTIONS)
    (tmp_path / "wt.yaml").write_text(RECIPE.replace("input_size: 640", "input_size: 960"))
    monkeypatch.setattr(cli, "WILDTRACK_RECIPE", "wt.yaml")

    code = cli.main(
        ["eval", "--dataset", "wildtrack", "--cameras", "all", "--metric", "mc-idf1"]
        + ["--config", "assoc.yaml", "--resamples", "10"]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "WILDTRACK  all cameras" in out and "10 s blocks" in out
    assert recipes == [cli.load_config("wt.yaml")]
    report = json.loads((tmp_path / "reports/mc-idf1-assoc-wildtrack-0.5m.json").read_text())
    assert report["recipe"] == "wt" and report["block_s"] == 10.0
    assert report["scenes"][0]["cameras"] == ["C5"]
