"""footfall eval --scene <id> --cameras all: the fused map of every camera,
against the single-camera baseline, with tracks from memory."""

import json
import math

import pytest

from footfall_heatmaps import cli
from tests.conftest import gt_row, write_scene
from tests.evaluation.application.test_evaluate_config import InMemoryTrackSource, tracks_from_rows
from tests.evaluation.application.test_evaluate_fusion import joined

RECIPE = (
    "detector: yolox-s\ntracker: bytetrack\ninput_size: 640\nengine: onnx-cuda\n"
    "stride: 1\nfoot_point: bottom\n"
)


@pytest.fixture
def fused_scene(tmp_path, rows, monkeypatch):
    # Track 1 walks across many cells and track 2 stands still: 5 distinct
    # people in a cell is never reached, so k=5 suppression empties the export.
    rows = [*rows[:-1], gt_row(8, 9, 0, 6.03, 9.07)]
    data = write_scene(tmp_path / "data", rows, cameras=(7, 8))
    monkeypatch.setenv("FOOTFALL_DATA_DIR", str(data))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(cli.SPLITS, "tiny", (("001", "7"),))
    seen_by_7 = tracks_from_rows(rows)
    source = InMemoryTrackSource(
        {("001", "7"): seen_by_7, ("001", "8"): joined(seen_by_7, tracks_from_rows(rows, 8))}
    )
    monkeypatch.setattr(cli, "track_source", lambda config, reader: source)
    (tmp_path / "fused.yaml").write_text(
        RECIPE + "fusion:\n  weighting: visibility\n  power: .inf\n  zone_m: 2.0\n"
    )
    (tmp_path / "single.yaml").write_text(RECIPE)
    return tmp_path


def _run(*extra):
    return cli.main(
        ["eval", "--cameras", "all", "--config", "fused.yaml", "--compare", "single.yaml"]
        + ["--resamples", "10", *extra]
    )


def test_a_scene_prints_the_overlap_mass_error_and_the_joint_metrics(fused_scene, capsys):
    assert _run("--scene", "001") == 0

    out = capsys.readouterr().out
    assert "cameras 7, 8 fused (2), single-camera baseline 7" in out
    assert "overlap mass error (dwell, every person, cells 2+ cameras cover):" in out
    assert "full@joint" in out and "visible@joint" in out
    assert "adding the cameras instead: 0.938" in out
    report = json.loads((fused_scene / "reports/fusion-fused-vs-single-s001-0.5m.json").read_text())
    assert report["fusion"] == {"weighting": "visibility", "power": "inf", "zone_m": 2.0}
    dwell = report["results"]["full@overlap"]["pooled"]["fused"]["dwell"]
    assert dwell["mass_error"]["value"] == pytest.approx(0, abs=1e-12)
    assert report["scenes"][0]["cameras"] == ["7", "8"]
    assert set(report["targets"]) == {"mass_error", "tvd", "wape_zone", "emd_m"}


def test_the_fused_export_is_k_suppressed(fused_scene):
    assert _run("--scene", "001") == 0

    export = json.loads((fused_scene / "reports/fused-s001-fused.json").read_text())
    assert export["cameras"] == ["7", "8"] and export["k"] == 5
    assert len(export["dwell"]) == export["shape"][0]
    assert all(v == 0 for row in export["dwell"] for v in row)  # nobody reaches 5 people
    assert all(v == 0 for row in export["passage"] for v in row)


def test_a_split_scores_its_scenes_pooled(fused_scene, capsys):
    assert _run("--split", "tiny") == 0

    out = capsys.readouterr().out
    assert "pooled" in out
    assert (fused_scene / "reports/fusion-fused-vs-single-tiny-0.5m.json").exists()


def test_cameras_needs_a_scene_or_a_split(fused_scene, capsys):
    assert cli.main(["eval", "--cameras", "all", "--config", "fused.yaml"]) == 2
    assert "--cameras needs --scene or --split" in capsys.readouterr().err


def test_the_fusion_section_parses_and_is_not_part_of_the_recipe(fused_scene):
    f = cli.load_fusion("fused.yaml")

    assert (f.weighting, math.isinf(f.power), f.zone_m) == ("visibility", True, 2.0)
    assert cli.load_config("fused.yaml") == cli.load_config("single.yaml")
    with pytest.raises(ValueError, match="no fusion section"):
        cli.load_fusion("single.yaml")


def test_covered_cells_parses_and_is_reported_only_when_on(fused_scene):
    (fused_scene / "norm.yaml").write_text(
        RECIPE + "fusion:\n  weighting: visibility\n  power: 2\n  zone_m: 2.0\n"
        "  covered_cells: true\n"
    )

    on, off = cli.load_fusion("norm.yaml"), cli.load_fusion("fused.yaml")

    assert on.covered_cells is True and off.covered_cells is False
    assert cli._fusion_dict(on) == {
        "weighting": "visibility",
        "power": 2,
        "zone_m": 2.0,
        "covered_cells": True,
    }
    assert "covered_cells" not in cli._fusion_dict(off)  # earlier reports unchanged
