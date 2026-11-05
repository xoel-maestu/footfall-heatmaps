"""Detector weights: downloaded once into the models directory, checked by sha256.

Models without a release ONNX file (url None) are exported from their official
release checkpoint by scripts/export-detector.py (EXPORTS says how); a missing
one is an error that names the command.

Downloads are anonymous HTTPS GETs to GitHub's release hosts only (redirects
included); no credential or header of the user is sent. A file whose sha256
differs from the one recorded here is never used. Weights are data: they are
only ever loaded by ONNX Runtime.
"""

import hashlib
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Weights:
    file: str
    url: str | None  # None: exported locally (EXPORTS)
    sha256: str
    licence: str


@dataclass(frozen=True)
class Export:
    """How scripts/export-detector.py makes a model: the official exporter of
    repo at commit, run on the release checkpoint (checked by its sha256)."""

    repo: str
    commit: str
    repo_dir: str
    checkpoint: str
    checkpoint_url: str
    checkpoint_sha256: str
    script: str  # under scripts/export/
    args: tuple[str, ...] = ()


WEIGHTS = {
    # Official YOLOX-s COCO model, release 0.1.1rc0 of Megvii-BaseDetection/YOLOX.
    "yolox-s": Weights(
        "yolox_s.onnx",
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
        "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
        "Apache-2.0",
    ),
    # The same checkpoint exported at a larger input (H2): YOLOX's official
    # tools/export_onnx.py, release 0.3.0, test_size 960 x 960.
    "yolox-s-960": Weights(
        "yolox_s_960.onnx",
        None,
        "95316ad4705db4dff9d77c22238514d685c86c3cb6a7b9ceb46879bedc84e73b",
        "Apache-2.0",
    ),
}
_YOLOX = (
    "https://github.com/Megvii-BaseDetection/YOLOX.git",
    "419778480ab6ec0590e5d3831b3afb3b46ab2aa3",  # tag 0.3.0
    "YOLOX",
    "yolox_s.pth",
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
    "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30",
    "yolox.py",
)
EXPORTS = {
    "yolox-s-960": Export(*_YOLOX, ("960",)),
}
HOSTS = {"github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"}


def check_url(url):
    u = urlsplit(url)
    if u.scheme != "https" or u.hostname not in HOSTS or u.username or u.password:
        raise OSError(f"refusing to download from {u.scheme}://{u.hostname}")


class _AllowlistRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def https_get(url, dest):
    check_url(url)
    opener = urllib.request.build_opener(_AllowlistRedirects)
    with opener.open(urllib.request.Request(url), timeout=60) as r, dest.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def ensure_weights(models_dir, name, fetch=https_get):
    """Path of the model's file under models_dir, downloaded on first use."""
    if name not in WEIGHTS:
        raise ValueError(f"no weights for {name!r}; known: {sorted(WEIGHTS)}")
    w = WEIGHTS[name]
    path = models_dir / w.file
    if not path.exists() and w.url is None:
        raise OSError(
            f"{path} is exported, not downloaded: "
            f"uv run --no-sync python scripts/export-detector.py {name} (needs the export group)"
        )
    if not path.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        try:
            fetch(w.url, tmp)
            if _sha256(tmp) != w.sha256:
                raise OSError(f"{w.url}: sha256 differs from the recorded {w.sha256}")
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
    elif _sha256(path) != w.sha256:
        raise OSError(f"{path}: sha256 differs from the recorded {w.sha256}; delete it to refetch")
    return path
