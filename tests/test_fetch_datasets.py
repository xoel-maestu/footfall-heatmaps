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


class FakeResponse:
    def __init__(self, status, body, length=None):
        self.status, self.body = status, body
        self.headers = {"Content-Length": str(len(body) if length is None else length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def fake_server(files, served=None):
    """files: url -> (size the server announces, body it sends)."""
    calls = []

    def _open(request, timeout):
        calls.append((request.get_method(), request.full_url))
        size, body = files[request.full_url]
        if request.get_method() == "HEAD":
            return FakeResponse(200, b"", size)
        return FakeResponse(200, body, size)

    return _open, calls


URL = "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA2/x/a.xml"


def test_a_truncated_download_is_refused_and_leaves_no_file(tmp_path, monkeypatch):
    fd = _module()
    _open, _ = fake_server({URL: (10, b"short")})
    monkeypatch.setattr(fd, "_open", _open)
    dest = tmp_path / "a.xml"

    with pytest.raises(OSError, match="truncated"):
        fd.download(URL, dest)

    assert not dest.exists() and not list(tmp_path.iterdir())


def test_a_file_of_the_wrong_size_is_fetched_again(tmp_path, monkeypatch):
    fd = _module()
    _open, calls = fake_server({URL: (10, b"0123456789")})
    monkeypatch.setattr(fd, "_open", _open)
    dest = tmp_path / "a.xml"
    dest.write_bytes(b"broken")

    assert fd.download(URL, dest) is True
    assert dest.read_bytes() == b"0123456789"
    assert fd.download(URL, dest) is False  # kept at its size
    assert [m for m, _ in calls] == ["HEAD", "GET", "HEAD"]


class SlottedInfo:
    """A ZipInfo stand-in without zipfile's private overlap-check attribute."""

    __slots__ = ("filename", "header_offset")

    def __init__(self):
        self.filename, self.header_offset = "a", 5


def test_member_lookup_does_not_need_zipfiles_private_attribute():
    fd = _module()

    class Archive:
        def open(self, info):
            return io.BytesIO(b"x")

    assert fd._open_member(Archive(), SlottedInfo(), [0]).read() == b"x"


def test_a_wrapped_end_offset_does_not_block_the_member():
    # On an archive written without zip64, zipfile derives each member's end
    # from the wrapped offsets, below where the member really ends.
    fd = _module()
    z = archive({"d/a.txt": b"abc" * 100})
    info = z.getinfo("d/a.txt")
    if hasattr(info, "_end_offset"):
        info._end_offset = 1

    assert fd._open_member(z, info, [0]).read() == b"abc" * 100
