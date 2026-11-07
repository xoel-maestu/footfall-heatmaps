"""Fetch the external evaluation-only datasets from their official sources.

    uv run --no-sync python scripts/fetch-datasets.py wildtrack|caviar|mot17
    uv run --no-sync python scripts/fetch-datasets.py nvidia <scene> [<scene> ...]

Only what `space-usage eval --dataset` reads, into SPACE_USAGE_DATA_DIR
(default ~/datasets/space-usage-heatmaps), never into the repo:
- wildtrack: calibrations, annotations and the frames of cameras C1-C7
  (multi-camera IDs, phase 10), read by HTTP range from the official 6.8 GB
  zip (whose offsets above 4 GiB wrap: it was written without zip64);
- caviar: the 26 corridor-view clips of the second set and their ground truth;
- mot17: train sequences 02, 04 and 09 (FRCNN copy), by HTTP range from
  the official 5.9 GB MOT17.zip;
- nvidia: the video.mp4 of every calibrated camera of the named scenes of
  MTMC_Tracking_2024 (their calibration and truth already on disk), from the
  official Hugging Face dataset nvidia/PhysicalAI-SmartSpaces (multi-camera
  fusion, phase 9). huggingface.co answers with one redirect to its CDN under
  hf.co, which is the only redirect followed (check_hf_cdn).
Anonymous HTTPS to the hosts below only, no other redirect followed, no
header sent (no token, no cookie). Licences:
docs/licenses.md. Files already present at their size are kept.
"""

import io
import os
import re
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

from space_usage_heatmaps.perception.infrastructure.model_weights import check_url

HOSTS = {"documents.epfl.ch", "homepages.inf.ed.ac.uk", "motchallenge.net"}
WILDTRACK_ZIP = (
    "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip"
)
WILDTRACK = [
    "calibrations/",
    "annotations_positions/",
    *(f"Image_subsets/C{k}/" for k in range(1, 8)),
]
CAVIAR_PAGE = "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/"
CAVIAR_FILE = re.compile(r'href="\.\./(CAVIARDATA2/[A-Za-z0-9]+cor/[A-Za-z0-9]+\.(?:mpg|xml))"')
MOT17_ZIP = "https://motchallenge.net/data/MOT17.zip"
MOT17 = [
    f"MOT17/train/MOT17-{n}-FRCNN/{part}"
    for n in ("02", "04", "09")
    for part in ("img1/", "gt/", "seqinfo.ini")
]
WRAP = 1 << 32
HF_HOSTS = {"huggingface.co"}
HF_DATASET = (
    "https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/resolve/main/"
    "MTMC_Tracking_2024/test"
)
REDIRECTS = (301, 302, 303, 307, 308)


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise OSError(f"refusing redirect to {newurl}")


_open = urllib.request.build_opener(_NoRedirects).open


def _get(url, headers=None):
    check_url(url, HOSTS)
    with _open(urllib.request.Request(url, headers=headers or {}), timeout=120) as r:
        return r.status, r.headers, r.read()


class RemoteFile(io.RawIOBase):
    """A remote file read by HTTP range requests."""

    def __init__(self, url):
        check_url(url, HOSTS)
        self.url, self.pos = url, 0
        with _open(urllib.request.Request(url, method="HEAD"), timeout=120) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or not len(b):
            return 0
        end = min(self.pos + len(b), self.size) - 1
        status, _, data = _get(self.url, {"Range": f"bytes={self.pos}-{end}"})
        if status != 206:
            raise OSError(f"{self.url}: no range support (HTTP {status})")
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _open_member(archive, info, tries):
    """The member's stream; an archive written without zip64 stores its local
    header offsets modulo 4 GiB, so each candidate offset is tried, the last
    one that worked first."""
    base = info.header_offset % WRAP
    for k in sorted(range(4), key=lambda k: k != tries[0]):
        info.header_offset = base + k * WRAP
        # zipfile has no public way to skip its overlap check, which compares
        # against an end derived from the wrapped offsets (CPython >= 3.12.3;
        # older versions lack the attribute): clear it only where it exists.
        # tests/test_fetch_datasets.py covers both cases.
        if getattr(info, "_end_offset", None) is not None:
            info._end_offset = None
        try:
            stream = archive.open(info)
        except zipfile.BadZipFile:
            continue
        tries[0] = k
        return stream
    raise OSError(f"cannot locate {info.filename} in the archive")


def extract(archive, prefixes, out):
    """Members of archive under any of prefixes into out, kept when present at
    their size; names that would leave out are refused."""
    out = Path(out).resolve()
    tries, written = [0], 0
    for info in archive.infolist():
        if info.is_dir() or not info.filename.startswith(tuple(prefixes)):
            continue
        dest = (out / info.filename).resolve()
        if not dest.is_relative_to(out):
            raise OSError(f"refusing member outside {out}: {info.filename}")
        if dest.exists() and dest.stat().st_size == info.file_size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        with _open_member(archive, info, tries) as src, part.open("wb") as f:
            while chunk := src.read(1 << 22):
                f.write(chunk)
        part.replace(dest)
        written += 1
    return written


def _remote_zip(url):
    return zipfile.ZipFile(io.BufferedReader(RemoteFile(url), buffer_size=32 << 20))


def download(url, dest):
    """url into dest unless dest is already at the size the server announces;
    written to a .part file, checked against Content-Length, then renamed, so
    a truncated or interrupted download never stays in place. Whether it wrote."""
    check_url(url, HOSTS)
    with _open(urllib.request.Request(url, method="HEAD"), timeout=120) as r:
        size = int(r.headers["Content-Length"])
    if dest.exists() and dest.stat().st_size == size:
        return False
    _, headers, body = _get(url)
    if len(body) != size or len(body) != int(headers["Content-Length"]):
        raise OSError(f"{url}: truncated, {len(body)} of {size} bytes")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(body)
    part.replace(dest)
    return True


def nvidia_video_url(scene, camera):
    return f"{HF_DATASET}/scene_{int(scene):03d}/camera_{int(camera):04d}/video.mp4"


def check_hf_cdn(url):
    """The one redirect followed: HTTPS to Hugging Face's CDN (a host under
    hf.co), without credentials in the URL."""
    u = urlsplit(url)
    host = u.hostname or ""
    if u.scheme != "https" or not host.endswith(".hf.co") or u.username or u.password:
        raise OSError(f"refusing redirect to {u.scheme}://{host}")


class _Keep(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib raises the redirect as an HTTPError, read in _resolve


def _resolve(url):
    """(where the file is, its size): a HEAD to huggingface.co, whose redirect
    is read, not followed."""
    check_url(url, HF_HOSTS)
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.build_opener(_Keep).open(request, timeout=120) as r:
            return url, int(r.headers["Content-Length"])
    except urllib.error.HTTPError as e:
        if e.code not in REDIRECTS:
            raise
        return e.headers["Location"], int(e.headers["X-Linked-Size"])


def fetch_hf(url, dest):
    """A Hugging Face file into dest unless dest is already at its size;
    streamed to a .part file, checked against the announced size, then
    renamed. Whether it wrote."""
    check_url(url, HF_HOSTS)
    location, size = _resolve(url)
    if location != url:
        check_hf_cdn(location)
    if dest.exists() and dest.stat().st_size == size:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    written = 0
    try:
        with _open(urllib.request.Request(location), timeout=120) as r, part.open("wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
                written += len(chunk)
        if written != size:
            raise OSError(f"{url}: truncated, {written} of {size} bytes")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return True


def fetch_nvidia(scenes, data_dir):
    """Every calibrated camera's video of each scene, one line per file on stdout."""
    import json

    root = data_dir / "nvidia" / "MTMC_Tracking_2024" / "test"
    written = 0
    for scene in scenes:
        d = root / f"scene_{int(scene):03d}"
        sensors = json.loads((d / "calibration_2025_format.json").read_text())["sensors"]
        cameras = sorted(int(s["id"].removeprefix("Camera_")) for s in sensors)
        for camera in cameras:
            dest = d / f"camera_{camera:04d}" / "video.mp4"
            new = fetch_hf(nvidia_video_url(scene, camera), dest)
            written += new
            print(f"scene {scene} camera {camera}: {'written' if new else 'present'}", flush=True)
    return written


def fetch(name, data_dir):
    if name == "wildtrack":
        prefixes = [f"Wildtrack_dataset/{p}" for p in WILDTRACK]
        return extract(_remote_zip(WILDTRACK_ZIP), prefixes, data_dir / "wildtrack")
    if name == "mot17":
        return extract(_remote_zip(MOT17_ZIP), MOT17, data_dir / "mot17")
    _, _, page = _get(CAVIAR_PAGE)
    written = 0
    for path in sorted(set(CAVIAR_FILE.findall(page.decode("latin-1")))):
        dest = data_dir / "caviar" / path.removeprefix("CAVIARDATA2/")
        written += download(f"https://homepages.inf.ed.ac.uk/rbf/{path}", dest)
    return written


def main(argv):
    data = Path(os.environ.get("SPACE_USAGE_DATA_DIR", "~/datasets/space-usage-heatmaps"))
    if len(argv) > 1 and argv[0] == "nvidia":
        print(f"nvidia: {fetch_nvidia(argv[1:], data.expanduser())} files written")
        return
    if len(argv) != 1 or argv[0] not in ("wildtrack", "caviar", "mot17"):
        raise SystemExit(__doc__)
    print(f"{argv[0]}: {fetch(argv[0], data.expanduser())} files written")


if __name__ == "__main__":
    main(sys.argv[1:])
