"""PyAV is excluded (pyproject.toml override, docs/licenses.md): its wheels bundle
an LGPL-3.0+ FFmpeg linked with GPL-2.0+ x264 and x265, and nothing here uses it."""

import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

IMPORT_ADAPTERS = """
import sys
import space_usage_heatmaps.perception.infrastructure.container
import space_usage_heatmaps.perception.infrastructure.bytetrack_tracker
import space_usage_heatmaps.perception.infrastructure.onnx_yolox_detector
import space_usage_heatmaps.perception.infrastructure.opencv_file_source
import supervision, trackers
print(sorted(m for m in sys.modules if m == "av" or m.startswith("av.")))
"""


def test_the_perception_adapters_never_import_av():
    out = subprocess.run(
        [sys.executable, "-c", IMPORT_ADAPTERS], capture_output=True, text=True, cwd=REPO
    )

    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_the_lock_never_installs_av():
    """uv keeps an overridden package as a stub with no wheels; every edge to it
    carries the never-true marker of the override."""
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    packages = {p["name"]: p for p in lock["package"]}
    edges = [
        d
        for p in lock["package"]
        for group in [p.get("dependencies", []), *p.get("dev-dependencies", {}).values()]
        for d in group
        if d["name"] == "av"
    ]

    assert "wheels" not in packages.get("av", {})
    assert edges and all(d.get("marker") == "sys_platform == 'never'" for d in edges)
