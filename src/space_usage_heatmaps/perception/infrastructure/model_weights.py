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

    repo: str | None  # None: the exporter is a package of its own environment
    commit: str | None
    repo_dir: str | None  # the clone under export/src
    checkpoint: str
    checkpoint_url: str
    checkpoint_sha256: str
    script: str  # under scripts/export/
    args: tuple[str, ...] = ()
    project: str | None = None  # its own uv project under scripts/export/, else the export group


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
    "yolox-s-1280": Weights(
        "yolox_s_1280.onnx",
        None,
        "b89fb6acc9072a74ece25ab4951b43fd21328c2eb07e4d7606220c0aa97fa33d",
        "Apache-2.0",
    ),
    # D-FINE-S trained on Objects365 then COCO (Peterande/D-FINE release
    # dfinev1.0, Apache-2.0), exported by D-FINE's tools/deployment/export_onnx.py.
    "dfine-s": Weights(
        "dfine_s_obj2coco.onnx",
        None,
        "f29f9569ef1d089dce6b392aef886fe4afd15e6193838bedae6e2ee4830b6ddd",
        "Apache-2.0",
    ),
    # RF-DETR Nano COCO (Roboflow, rfdetr 1.11.2's Apache-2.0 Nano checkpoint),
    # exported by the rfdetr package's own ONNX exporter.
    "rfdetr-n": Weights(
        "rf_detr_nano.onnx",
        None,
        "d217512443ea74ca6f025e9d9239cfd7650de318664c6fc26ee81ae705e4baac",
        "Apache-2.0",
    ),
}
_YOLOX = dict(
    repo="https://github.com/Megvii-BaseDetection/YOLOX.git",
    commit="419778480ab6ec0590e5d3831b3afb3b46ab2aa3",  # tag 0.3.0
    repo_dir="YOLOX",
    checkpoint="yolox_s.pth",
    checkpoint_url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
    checkpoint_sha256="f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30",
    script="yolox.py",
)
EXPORTS = {
    "yolox-s-960": Export(**_YOLOX, args=("960",)),
    "yolox-s-1280": Export(**_YOLOX, args=("1280",)),
    "dfine-s": Export(
        repo="https://github.com/Peterande/D-FINE.git",
        commit="956d1709314c2c6a4df6f34de232054578a7449f",  # master, 2026-08-19
        repo_dir="D-FINE",
        checkpoint="dfine_s_obj2coco.pth",
        checkpoint_url="https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj2coco.pth",
        checkpoint_sha256="75bedc333cdf665f9cddd2c6a2a9baa34e0e5e9a38db7bb8e6110edc3b4ba8eb",
        script="dfine.py",
        args=("configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml",),
    ),
    "rfdetr-n": Export(
        repo=None,
        commit=None,
        repo_dir=None,
        checkpoint="rf-detr-nano.pth",
        checkpoint_url="https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
        # md5 fb6504cce7fbdc783f7a46991f07639f, as rfdetr 1.11.2 records it
        checkpoint_sha256="d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d",
        script="rfdetr.py",
        project="rfdetr",
    ),
}
HOSTS = {"github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"}
# The export step also fetches checkpoints from the vendor's public bucket
# (RF-DETR's), never the run path.
EXPORT_HOSTS = HOSTS | {"storage.googleapis.com"}


def check_url(url, hosts=HOSTS):
    u = urlsplit(url)
    if u.scheme != "https" or u.hostname not in hosts or u.username or u.password:
        raise OSError(f"refusing to download from {u.scheme}://{u.hostname}")


def _allowlist_redirects(hosts):
    class AllowlistRedirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            check_url(newurl, hosts)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    return AllowlistRedirects


def https_get(url, dest, hosts=HOSTS):
    check_url(url, hosts)
    opener = urllib.request.build_opener(_allowlist_redirects(hosts))
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
