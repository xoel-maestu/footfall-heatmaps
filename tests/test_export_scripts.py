"""scripts/export-detector.py and the exporters' weights_only shim, without torch
or the network: fake checkpoints and a fake torch module only."""

import hashlib
import importlib.util
import runpy
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _export_detector():
    spec = importlib.util.spec_from_file_location("export_detector", SCRIPTS / "export-detector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_checkpoint_with_another_sha256_is_never_put_in_place(monkeypatch, tmp_path):
    ed = _export_detector()
    monkeypatch.setattr(ed, "https_get", lambda url, dest, hosts: dest.write_bytes(b"other"))
    dest = tmp_path / "model.pth"

    with pytest.raises(SystemExit, match="sha256"):
        ed.checkpoint("https://github.com/x.pth", "0" * 64, dest)

    assert not dest.exists() and not dest.with_name("model.pth.part").exists()


def test_a_checkpoint_with_the_recorded_sha256_is_put_in_place(monkeypatch, tmp_path):
    ed = _export_detector()
    monkeypatch.setattr(ed, "https_get", lambda url, dest, hosts: dest.write_bytes(b"ckpt"))
    dest = tmp_path / "model.pth"

    assert ed.checkpoint("https://github.com/x.pth", hashlib.sha256(b"ckpt").hexdigest(), dest)
    assert dest.read_bytes() == b"ckpt"


def test_the_exporter_environment_cannot_turn_weights_only_off(monkeypatch):
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere")

    env = _export_detector().child_env()

    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in env and "VIRTUAL_ENV" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_the_shim_forces_weights_only_on_every_torch_load():
    shim = runpy.run_path(str(SCRIPTS / "export" / "_weights_only.py"))
    calls = []
    torch = types.SimpleNamespace(load=lambda *a, **kw: calls.append(kw))

    shim["force_weights_only"](torch)
    torch.load("ckpt.pth", map_location="cpu")
    with pytest.raises(RuntimeError, match="weights_only=False"):
        torch.load("ckpt.pth", weights_only=False)

    assert calls == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize("script", ["yolox.py", "dfine.py", "rfdetr.py"])
def test_every_exporter_uses_the_shim(script):
    assert "force_weights_only(torch)" in (SCRIPTS / "export" / script).read_text()
