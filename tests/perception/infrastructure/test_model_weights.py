import hashlib

import pytest

from space_usage_heatmaps.perception.infrastructure import model_weights
from space_usage_heatmaps.perception.infrastructure.model_weights import (
    Weights,
    check_url,
    ensure_weights,
)

CONTENT = b"not a real model"
FAKE = Weights(
    "fake.onnx",
    "https://github.com/example/releases/download/v1/fake.onnx",
    hashlib.sha256(CONTENT).hexdigest(),
    "Apache-2.0",
)


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setitem(model_weights.WEIGHTS, "fake", FAKE)


def fetcher(content):
    calls = []

    def fetch(url, dest):
        calls.append(url)
        dest.write_bytes(content)

    return fetch, calls


def test_downloads_once_and_checks_the_sha256(tmp_path):
    fetch, calls = fetcher(CONTENT)

    path = ensure_weights(tmp_path / "models", "fake", fetch)
    again = ensure_weights(tmp_path / "models", "fake", fetch)

    assert path == again == tmp_path / "models" / "fake.onnx"
    assert path.read_bytes() == CONTENT and calls == [FAKE.url]


def test_a_download_with_another_sha256_is_rejected_and_not_kept(tmp_path):
    fetch, _ = fetcher(b"tampered")

    with pytest.raises(OSError, match="sha256"):
        ensure_weights(tmp_path, "fake", fetch)
    assert list(tmp_path.iterdir()) == []


def test_a_file_on_disk_with_another_sha256_is_an_error(tmp_path):
    (tmp_path / "fake.onnx").write_bytes(b"tampered")
    fetch, calls = fetcher(CONTENT)

    with pytest.raises(OSError, match="sha256"):
        ensure_weights(tmp_path, "fake", fetch)
    assert calls == []


def test_an_unknown_model_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="yolox-s"):
        ensure_weights(tmp_path, "nope")


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/x",
        "https://example.com/x",
        "https://github.com.evil.example/x",
        "https://user@github.com/x",
    ],
)
def test_only_https_to_the_release_hosts(url):
    with pytest.raises(OSError, match="refusing"):
        check_url(url)


def test_the_release_hosts_pass():
    check_url("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/x")
    check_url("https://release-assets.githubusercontent.com/github-production-release-asset/x")
