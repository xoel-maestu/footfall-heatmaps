"""scripts/fetch-datasets.py without the network: archives in memory."""

import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch-datasets.py"


def _module():
    spec = importlib.util.spec_from_file_location("fetch_datasets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return zipfile.ZipFile(buf)


def test_only_the_wanted_members_are_extracted(tmp_path):
    fd = _module()
    z = archive({"d/calib/a.xml": b"a", "d/frames/C1/0.png": b"x", "d/frames/C5/0.png": b"y"})

    assert fd.extract(z, ["d/calib/", "d/frames/C5/"], tmp_path) == 2

    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert files == ["d/calib/a.xml", "d/frames/C5/0.png"]
    assert fd.extract(z, ["d/calib/"], tmp_path) == 0  # kept at their size


def test_a_member_outside_the_output_is_refused(tmp_path):
    fd = _module()
    z = archive({"d/../../evil.txt": b"x"})

    with pytest.raises(OSError, match="outside"):
        fd.extract(z, ["d/"], tmp_path / "out")


def test_only_the_official_hosts_are_contacted():
    fd = _module()

    with pytest.raises(OSError, match="refusing"):
        fd.RemoteFile("https://example.com/MOT17.zip")
