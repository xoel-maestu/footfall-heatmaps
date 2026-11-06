"""R1 (spec 3): no frame or crop is ever written to disk.

The real eval path (video decoding, ByteTrack, track cache, report) runs in a
subprocess with HOME, TMPDIR, the data dir and the working directory pointed
at temp dirs. An audit hook (sys.addaudithook) inside it records every file
opened for writing (any path), every os.rename and os.replace target and
every socket.connect. The hook sees Python-level I/O only; native code
(OpenCV, ONNX Runtime) is covered by the snapshot of those dirs, which catches
for example the device id and onnxruntime.db that ONNX Runtime keeps under
~/.cache/Microsoft. Afterwards every recorded path, and every file the run
created or changed under those dirs, is audited: no image or video by
extension or by magic bytes, and no NumPy array shaped like an image; no
connection at all; stdout and stderr are plain text.

The detector is a fake that letterboxes each image as YOLOX does and returns
the scene's ground-truth boxes (CI); a second run uses the real YOLOX-s on
ONNX Runtime when the runtime group and the weights are there.
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
import pytest

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


AUDIT = """
import atexit, json, os, sys

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
LOG = os.environ.pop("AUDIT_LOG")
written, connects, done = [], [], []


def _path(p):
    return os.path.abspath(os.fsdecode(p)) if isinstance(p, (str, bytes, os.PathLike)) else None


def hook(event, args):
    if done:
        return
    if event == "open":
        path, mode, flags = args
        if (isinstance(mode, str) and set(mode) & set("wax+")) or (flags or 0) & WRITE_FLAGS:
            if (p := _path(path)) is not None:
                written.append(p)
    elif event in ("os.rename", "os.replace"):
        if (p := _path(args[1])) is not None:
            written.append(p)
    elif event == "socket.connect":
        connects.append(repr(args[1]))


def dump():
    done.append(True)
    with open(LOG, "w") as f:
        json.dump({"written": written, "connects": connects}, f)


sys.addaudithook(hook)
atexit.register(dump)
"""

RUN = """
import json
import os
from pathlib import Path

from space_usage_heatmaps import cli
from space_usage_heatmaps.perception.domain.frame import Detections
from space_usage_heatmaps.perception.infrastructure import container
from space_usage_heatmaps.perception.infrastructure.onnx_yolox_detector import preprocess
from tests.perception.oracle_detector import GroundTruthDetector

class LetterboxingDetector(GroundTruthDetector):
    def detect(self, frame):
        x, _ = preprocess(frame.image, 640)  # the image handling a real detector does
        assert x.shape == (3, 640, 640)
        if leak := os.environ.get("LEAK_TO"):  # the audit's own check: a leaking detector
            import cv2
            jpeg = cv2.imencode(".jpg", frame.image)[1].tobytes()
            for path in (leak, leak + ".tmp"):
                with open(path, "wb") as f:
                    f.write(jpeg)
            os.replace(leak + ".tmp", leak + ".png")
            import socket
            try:
                socket.create_connection(("127.0.0.1", 9), timeout=0.1)
            except OSError:
                pass
        return super().detect(frame)

if real := os.environ.pop("REAL_MODELS", None):  # the real YOLOX-s, weights checked
    ensure = container.ensure_weights
    container.ensure_weights = lambda d, name: ensure(Path(real), name)
else:
    container.ensure_weights = lambda d, name: d / name
    rows = json.load(open("rows.json"))
    container.load_detector = lambda path, *_: LetterboxingDetector(rows)
cli.SPLITS["tiny"] = (("001", "7"),)
raise SystemExit(cli.main(["eval", "--config", "h1.yaml", "--split", "tiny", "--resamples", "5"]))
"""
CONFIG = (
    "detector: yolox-s\ntracker: bytetrack\ninput_size: 640\nengine: onnx-cuda\n"
    "stride: 1\nfoot_point: auto-track\n"
)


def _plain_text(data):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return all(c.isprintable() or c in "\n\r\t" for c in text)


def test_the_output_check_rejects_image_bytes():
    png = cv2.imencode(".png", np.zeros((20, 20, 3), np.uint8))[1].tobytes()

    assert _plain_text(b"scene 001 7 pooled 0.355\n")
    assert not _plain_text(b"log line " + png)


def real_models():
    """The real weights' directory, or a skip when the runtime or the weights are missing."""
    pytest.importorskip("onnxruntime", reason="runtime group not installed")
    data = Path(os.environ.get("SPACE_USAGE_DATA_DIR", "~/datasets/space-usage-heatmaps"))
    models = data.expanduser() / "models"
    if not (models / "yolox_s.onnx").exists():
        pytest.skip(f"no YOLOX-s weights in {models}")
    return models


def run_audited(tmp_path, rows, extra):
    """(subprocess result, audit log, flagged paths) of one eval run under the hook."""
    home, tmp, data, work = (tmp_path / d for d in ("home", "tmp", "data", "work"))
    for d in (home, tmp, work):
        d.mkdir()
    write_scene(data, rows)
    write_video(data / "nvidia/MTMC_Tracking_2024/test/scene_001/camera_0007/video.mp4", n=12)
    (work / "h1.yaml").write_text(CONFIG)
    (work / "rows.json").write_text(json.dumps(rows))
    roots = (home, tmp, data, work)
    before = snapshot(roots)
    log = tmp_path / "audit.json"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "LC_ALL": "C.UTF-8",
        "SPACE_USAGE_DATA_DIR": str(data),
        "PYTHONPATH": str(REPO),
        "PYTHONDONTWRITEBYTECODE": "1",
        "AUDIT_LOG": str(log),
    } | extra

    out = subprocess.run(
        [sys.executable, "-c", AUDIT + RUN], cwd=work, env=env, capture_output=True
    )

    assert out.returncode == 0, out.stderr.decode(errors="replace")
    audit = json.loads(log.read_text())
    after = snapshot(roots)
    changed = {p for p, t in after.items() if before.get(p) != t}
    recorded = {Path(p) for p in audit["written"]}
    assert {".npz", ".json"} <= {p.suffix for p in recorded}  # the run did write
    flagged = {}
    for p in sorted(recorded | changed):
        if {s.lower() for s in p.suffixes} & EXTENSIONS:
            flagged[str(p)] = ["extension"]
        elif p.is_file() and (why := offences(p)):
            flagged[str(p)] = why
    return out, audit, flagged


def test_the_hook_catches_writes_renames_and_connections_anywhere(tmp_path, rows):
    elsewhere = tmp_path / "elsewhere"  # outside every snapshotted dir
    elsewhere.mkdir()

    _, audit, flagged = run_audited(tmp_path, rows, {"LEAK_TO": str(elsewhere / "f")})

    assert set(flagged) == {str(elsewhere / "f"), str(elsewhere / "f.png")}
    assert audit["connects"] == ["('127.0.0.1', 9)"] * 12  # one per frame


@pytest.mark.parametrize("detector", ["fake", "real"])
def test_a_pipeline_run_writes_no_frame_and_connects_nowhere(tmp_path, rows, detector):
    extra = {"REAL_MODELS": str(real_models())} if detector == "real" else {}

    out, audit, flagged = run_audited(tmp_path, rows, extra)

    assert audit["connects"] == []
    assert flagged == {}
    assert _plain_text(out.stdout) and _plain_text(out.stderr)
