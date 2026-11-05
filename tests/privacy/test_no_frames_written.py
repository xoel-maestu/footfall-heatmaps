"""R1 (spec 3): no frame or crop is ever written to disk.

The real eval path (video decoding, ByteTrack, track cache, report) runs in a
subprocess with HOME, TMPDIR, the data dir and the working directory pointed
at temp dirs, a fake detector that letterboxes each image as YOLOX does and
returns the scene's ground-truth boxes. Afterwards every file the run created
or changed is audited: no image or video by extension or by magic bytes, and
no NumPy array shaped like an image.
"""

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

from tests.conftest import write_scene
from tests.perception.synthetic_video import write_video

REPO = Path(__file__).resolve().parents[2]
EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ppm", ".pgm",
    ".pnm", ".exr", ".heic", ".avif", ".mp4", ".avi", ".mkv", ".mov", ".webm", ".h264",
    ".h265", ".ts", ".m4v", ".mjpeg", ".y4m", ".raw",
}  # fmt: skip
MAGIC = [
    (0, b"\xff\xd8\xff"),  # JPEG
    (0, b"\x89PNG"),
    (0, b"GIF8"),
    (0, b"BM"),  # BMP
    (0, b"II*\x00"),  # TIFF
    (0, b"MM\x00*"),
    (0, b"\x1aE\xdf\xa3"),  # Matroska, WebM
    (4, b"ftyp"),  # MP4, MOV, HEIC, AVIF
    (0, b"YUV4MPEG2"),
    (0, b"\x00\x00\x00\x01"),  # H.264/H.265 Annex B
    (0, b"\x76\x2f\x31\x01"),  # OpenEXR
]
RIFF = (b"WEBP", b"AVI ")


def _image_like_array(a):
    return a.dtype == np.uint8 and a.ndim in (2, 3) and min(a.shape[:2]) >= 16


def _arrays(data):
    """The arrays of a .npy or .npz file's bytes, or none if it is neither."""
    try:
        if data.startswith(b"PK"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names = [n for n in z.namelist() if n.endswith(".npy")]
                return [np.load(io.BytesIO(z.read(n)), allow_pickle=False) for n in names]
        if data.startswith(b"\x93NUMPY"):
            return [np.load(io.BytesIO(data), allow_pickle=False)]
    except (ValueError, OSError, zipfile.BadZipFile):
        return []
    return []


def offences(path):
    """Why a file looks like an image or a video, or an empty list."""
    why = []
    if {s.lower() for s in path.suffixes} & EXTENSIONS:
        why.append("extension")
    data = path.read_bytes()
    if any(data[o : o + len(m)] == m for o, m in MAGIC):
        why.append("magic bytes")
    if data[:4] == b"RIFF" and data[8:12] in RIFF:
        why.append("magic bytes")
    if any(_image_like_array(a) for a in _arrays(data)):
        why.append("image-shaped array")
    return why


def snapshot(roots):
    return {p: p.stat().st_mtime_ns for r in roots for p in r.rglob("*") if p.is_file()}


def test_the_audit_catches_frames_in_any_form(tmp_path):
    image = np.full((120, 160, 3), 90, np.uint8)
    (tmp_path / "a.png").write_bytes(cv2.imencode(".png", image)[1].tobytes())
    (tmp_path / "frame.bin").write_bytes(cv2.imencode(".jpg", image)[1].tobytes())
    np.save(tmp_path / "frame.npy", image)
    np.savez(tmp_path / "crops.npz", crop=image[10:60, 10:60])
    write_video(tmp_path / "clip.avi", n=2).rename(tmp_path / "clip.data")  # under another name
    np.savez(tmp_path / "tracks.npz", boxes=np.zeros((5, 4)), ids=np.arange(5))

    flagged = {p.name for p in tmp_path.iterdir() if offences(p)}

    assert flagged == {"a.png", "frame.bin", "frame.npy", "crops.npz", "clip.data"}


RUN = """
import json

from space_usage_heatmaps import cli
from space_usage_heatmaps.perception.domain.frame import Detections
from space_usage_heatmaps.perception.infrastructure import container
from space_usage_heatmaps.perception.infrastructure.onnx_yolox_detector import preprocess
from tests.perception.oracle_detector import GroundTruthDetector

class LetterboxingDetector(GroundTruthDetector):
    def detect(self, frame):
        x, _ = preprocess(frame.image, 640)  # the image handling a real detector does
        assert x.shape == (3, 640, 640)
        return super().detect(frame)

container.ensure_weights = lambda d, name: d / name
rows = json.load(open("rows.json"))
container.load_detector = lambda path: LetterboxingDetector(rows)
cli.SPLITS["tiny"] = (("001", "7"),)
raise SystemExit(cli.main(["eval", "--config", "h1.yaml", "--split", "tiny", "--resamples", "5"]))
"""
CONFIG = (
    "detector: yolox-s\ntracker: bytetrack\ninput_size: 640\nengine: onnx-cuda\n"
    "stride: 1\nfoot_point: auto-track\n"
)


def test_a_pipeline_run_writes_no_frame(tmp_path, rows):
    home, tmp, data, work = (tmp_path / d for d in ("home", "tmp", "data", "work"))
    for d in (home, tmp, work):
        d.mkdir()
    write_scene(data, rows)
    write_video(data / "nvidia/MTMC_Tracking_2024/test/scene_001/camera_0007/video.mp4", n=12)
    (work / "h1.yaml").write_text(CONFIG)
    (work / "rows.json").write_text(json.dumps(rows))
    roots = (home, tmp, data, work)
    before = snapshot(roots)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "LC_ALL": "C.UTF-8",
        "SPACE_USAGE_DATA_DIR": str(data),
        "PYTHONPATH": str(REPO),
    }

    out = subprocess.run(
        [sys.executable, "-c", RUN], cwd=work, env=env, capture_output=True, text=True
    )

    assert out.returncode == 0, out.stderr
    after = snapshot(roots)
    written = sorted(p for p, t in after.items() if before.get(p) != t)
    names = {p.suffix for p in written}
    assert {".npz", ".json"} <= names  # the run did write: tracks cache and report
    assert {str(p.relative_to(tmp_path)): offences(p) for p in written if offences(p)} == {}
