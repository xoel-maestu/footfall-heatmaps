"""Export a detector's official release checkpoint to the ONNX file the runtime runs.

    uv sync --group runtime --group export
    uv run --no-sync python scripts/export-detector.py yolox-s-960

Each family uses its own official exporter at a pinned commit: the repository is
cloned into SPACE_USAGE_DATA_DIR/export/src, the release checkpoint is fetched
anonymously from the vendor's release host and checked against its recorded
sha256 before anything loads it, and the exporter writes the ONNX file into
SPACE_USAGE_DATA_DIR/models. The checkpoint is loaded only by the official
loader, with torch's weights_only=True. The ONNX file's sha256 is printed:
record it in perception/infrastructure/model_weights.py and docs/licenses.md.
CPU only; no GPU is used.
"""

import os
import subprocess
import sys
from pathlib import Path

from space_usage_heatmaps.perception.infrastructure.model_weights import (
    EXPORTS,
    WEIGHTS,
    _sha256,
    https_get,
)

HERE = Path(__file__).resolve().parent / "export"


def data_dir():
    return Path(os.environ.get("SPACE_USAGE_DATA_DIR", "~/datasets/space-usage-heatmaps"))


def checkout(repo, commit, dest):
    if not dest.exists():
        subprocess.run(["git", "clone", "-q", repo, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != commit:
        raise SystemExit(f"{dest}: HEAD {head} is not the pinned {commit}")


def checkpoint(url, sha256, dest):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        https_get(url, tmp)
        tmp.replace(dest)
    if _sha256(dest) != sha256:
        raise SystemExit(f"{dest}: sha256 differs from the recorded {sha256}; delete it")
    return dest


def main(name):
    if name not in EXPORTS:
        raise SystemExit(f"no export for {name!r}; known: {sorted(EXPORTS)}")
    e, w = EXPORTS[name], WEIGHTS[name]
    root = data_dir().expanduser()
    src = root / "export" / "src" / e.repo_dir
    checkout(e.repo, e.commit, src)
    ckpt = checkpoint(
        e.checkpoint_url, e.checkpoint_sha256, root / "export" / "checkpoints" / e.checkpoint
    )
    out = root / "models" / w.file
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.onnx")
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}  # export on CPU
    cmd = [sys.executable, str(HERE / e.script), str(src), str(ckpt), str(tmp), *e.args]
    subprocess.run(cmd, check=True, cwd=src, env=env)
    tmp.replace(out)
    print(f"{out}  sha256 {_sha256(out)}  (recorded: {w.sha256})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
