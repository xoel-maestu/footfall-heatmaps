"""The data directory keeps resolving for clones set up before the rename."""

import importlib.util
from pathlib import Path

import pytest

from footfall_heatmaps import cli
from footfall_heatmaps.cli import data_dir

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FOOTFALL_DATA_DIR", raising=False)
    monkeypatch.delenv("SPACE_USAGE_DATA_DIR", raising=False)
    monkeypatch.setattr(cli, "_LEGACY_NOTICED", False)
    return tmp_path


def test_the_new_variable_wins_over_everything(home, monkeypatch, capsys):
    (home / "datasets" / "space-usage-heatmaps").mkdir(parents=True)
    monkeypatch.setenv("FOOTFALL_DATA_DIR", str(home / "new"))
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(home / "old"))
    assert data_dir() == home / "new"
    assert capsys.readouterr().err == ""


def test_the_legacy_variable_still_works_and_says_so_once(home, monkeypatch, capsys):
    monkeypatch.setenv("SPACE_USAGE_DATA_DIR", str(home / "old"))
    assert data_dir() == home / "old"
    assert data_dir() == home / "old"
    err = capsys.readouterr().err
    assert err.count("SPACE_USAGE_DATA_DIR") == 1
    assert "FOOTFALL_DATA_DIR" in err


def test_the_new_default_when_it_exists(home, capsys):
    (home / "datasets" / "footfall-heatmaps").mkdir(parents=True)
    (home / "datasets" / "space-usage-heatmaps").mkdir(parents=True)
    assert data_dir() == home / "datasets" / "footfall-heatmaps"
    assert capsys.readouterr().err == ""


def test_the_legacy_default_when_only_it_exists(home, capsys):
    (home / "datasets" / "space-usage-heatmaps").mkdir(parents=True)
    assert data_dir() == home / "datasets" / "space-usage-heatmaps"
    err = capsys.readouterr().err
    assert "space-usage-heatmaps" in err
    assert "FOOTFALL_DATA_DIR" in err


def test_the_new_default_when_nothing_exists(home, capsys):
    assert data_dir() == home / "datasets" / "footfall-heatmaps"
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("script", ["fetch-datasets.py", "export-detector.py"])
def test_scripts_resolve_the_data_dir_with_the_same_rule(script):
    spec = importlib.util.spec_from_file_location(script, SCRIPTS / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.data_dir is data_dir
