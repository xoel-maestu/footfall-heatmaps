"""Detector weights: downloaded once into the models directory, checked by sha256.

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
    url: str
    sha256: str
    licence: str


WEIGHTS = {
    # Official YOLOX-s COCO model, release 0.1.1rc0 of Megvii-BaseDetection/YOLOX.
    "yolox-s": Weights(
        "yolox_s.onnx",
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
        "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
        "Apache-2.0",
    ),
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
