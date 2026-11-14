"""Privacy rule R3: no appearance, facial or biometric feature in any module.

Statically, over every module of the package: no import of a re-identification,
embedding or face library, and no identifier that names one; the only tracker
classes used from `trackers` are ByteTrack and OC-SORT (motion and IoU only),
plus its evaluation metrics. At run time: each tracker adapter wraps that
class, whose parameters carry no appearance option, and never hands it the
image.
"""

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest
import trackers

from footfall_heatmaps.perception.infrastructure.bytetrack_tracker import ByteTrackTracker
from footfall_heatmaps.perception.infrastructure.ocsort_tracker import OcSortTracker

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "footfall_heatmaps"
LIBRARIES = {
    "torchreid", "fastreid", "boxmot", "deep_sort", "deep_sort_realtime", "deep_sort_pytorch",
    "insightface", "face_recognition", "facenet_pytorch", "deepface", "retinaface", "dlib",
    "mediapipe", "open_clip", "clip", "transformers", "timm", "sentence_transformers",
}  # fmt: skip
WORDS = {
    "reid", "embedding", "embeddings", "embed", "appearance", "face", "faces", "facial",
    "biometric", "gait", "osnet", "feature", "features", "descriptor", "descriptors",
}  # fmt: skip


def modules():
    return sorted(PACKAGE.rglob("*.py"))


def words(identifier):
    """snake_case and CamelCase parts, lower case: ReIDModel -> re, id, model, and reid."""
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", identifier)
    lowered = [p.lower() for p in parts]
    return set(lowered) | {a + b for a, b in zip(lowered, lowered[1:], strict=False)}


def identifiers(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield node.name
        elif isinstance(node, ast.arg) or isinstance(node, ast.keyword) and node.arg:
            yield node.arg
        elif isinstance(node, ast.alias):
            yield from node.name.split(".")
            if node.asname:
                yield node.asname


def imported(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            yield from (f"{node.module}.{a.name}" for a in node.names)


def test_the_scan_sees_appearance_names():
    assert "reid" in words("ReIDModel") and "embedding" in words("get_embedding")
    assert not WORDS & words("surface_interface")


def test_no_module_imports_an_appearance_library():
    offenders = [
        (f.name, m)
        for f in modules()
        for m in imported(ast.parse(f.read_text()))
        if m.split(".")[0] in LIBRARIES
    ]
    assert not offenders


def test_no_module_names_appearance_features():
    offenders = [
        (str(f.relative_to(PACKAGE)), name, sorted(WORDS & words(name)))
        for f in modules()
        for name in identifiers(ast.parse(f.read_text()))
        if WORDS & words(name)
    ]
    assert not offenders


def test_the_trackers_used_from_trackers_are_bytetrack_and_ocsort():
    used = set()
    for f in modules():
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "trackers"
            ):
                used.add(node.attr)
        used |= {m.split(".", 1)[1] for m in imported(tree) if m.startswith("trackers.")}
    metrics = {u for u in used if u == "eval" or u.startswith("eval.")}
    assert used - metrics == {"ByteTrackTracker", "OCSORTTracker"}


@pytest.mark.parametrize(
    ("adapter", "inner_class"),
    [(ByteTrackTracker, trackers.ByteTrackTracker), (OcSortTracker, trackers.OCSORTTracker)],
)
def test_the_tracker_has_no_appearance_and_never_sees_the_image(adapter, inner_class):
    adapter = adapter(frame_rate=30)
    inner = adapter._tracker
    assert type(inner) is inner_class
    params = set(inspect.signature(inner_class).parameters)
    assert not {w for p in params for w in words(p)} & WORDS
    calls = []
    original = inner.update

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    inner.update = spy
    for i in range(3):
        adapter.update(np.array([[10.0 + i, 10, 40, 90]]), np.array([0.9]), i)

    assert len(calls) == 3
    for args, kwargs in calls:
        assert len(args) == 1 and kwargs.get("frame") is None  # detections only
