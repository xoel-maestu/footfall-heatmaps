import numpy as np

from space_usage_heatmaps.evaluation.domain.tracks import Tracks
from space_usage_heatmaps.evaluation.infrastructure.track_cache import CachedTrackSource

TRACKS = Tracks(np.array([1, 2]), np.array([0, 5]), np.array([[1.0, 2, 3, 4], [5, 6, 7, 8]]))


class CountingSource:
    made = 0

    def __init__(self):
        CountingSource.made += 1

    def tracks(self, scene, camera):
        return TRACKS


def test_the_first_call_runs_the_source_and_later_ones_read_the_cache(tmp_path):
    CountingSource.made = 0
    cache = CachedTrackSource(CountingSource, tmp_path / "c")

    first = cache.tracks("071", "645")
    again = CachedTrackSource(CountingSource, tmp_path / "c").tracks("071", "645")

    assert CountingSource.made == 1
    for t in (first, again):
        np.testing.assert_array_equal(t.track_ids, TRACKS.track_ids)
        np.testing.assert_array_equal(t.frames, TRACKS.frames)
        np.testing.assert_array_equal(t.boxes, TRACKS.boxes)


def test_the_cache_holds_positions_and_the_recipe_metadata_only(tmp_path):
    CachedTrackSource(CountingSource, tmp_path).tracks("071", "645")

    (path,) = tmp_path.iterdir()
    assert path.name == "scene_071_camera_645.npz"
    with np.load(path, allow_pickle=False) as f:
        assert sorted(f.files) == ["boxes", "frames", "meta", "track_ids"]


META = {"ultralytics": "8.4.0", "tracker_yaml_sha256": "ab" * 32, "classes": [0], "conf": 0.1}


def test_the_cache_stores_the_recipe_metadata_and_reuses_it_when_it_matches(tmp_path, capsys):
    CountingSource.made = 0
    CachedTrackSource(CountingSource, tmp_path, META).tracks("071", "645")
    CachedTrackSource(CountingSource, tmp_path, dict(META)).tracks("071", "645")

    assert CountingSource.made == 1
    assert capsys.readouterr().err == ""


def test_a_metadata_mismatch_recomputes_and_says_why(tmp_path, capsys):
    CountingSource.made = 0
    CachedTrackSource(CountingSource, tmp_path, META).tracks("071", "645")

    newer = META | {"ultralytics": "8.5.0"}
    CachedTrackSource(CountingSource, tmp_path, newer).tracks("071", "645")
    CachedTrackSource(CountingSource, tmp_path, newer).tracks("071", "645")

    assert CountingSource.made == 2  # recomputed once, then the new entry is reused
    (line,) = capsys.readouterr().err.splitlines()
    assert "scene_071_camera_645" in line and "ultralytics" in line and "8.5.0" in line


def test_a_cache_file_without_metadata_is_recomputed(tmp_path, capsys):
    CountingSource.made = 0
    np.savez(  # written before the cache stored metadata
        tmp_path / "scene_071_camera_645.npz",
        track_ids=TRACKS.track_ids, frames=TRACKS.frames, boxes=TRACKS.boxes,
    )  # fmt: skip

    CachedTrackSource(CountingSource, tmp_path, META).tracks("071", "645")

    assert CountingSource.made == 1
    assert "no metadata" in capsys.readouterr().err


def test_unknown_expected_values_are_not_checked_and_say_so(tmp_path, capsys):
    # Without the baseline installed, the ultralytics version cannot be read:
    # the cached tracks are used, the rest of the recipe still checked.
    CountingSource.made = 0
    CachedTrackSource(CountingSource, tmp_path, META).tracks("071", "645")

    unknown = META | {"ultralytics": None, "tracker_yaml_sha256": None}
    CachedTrackSource(CountingSource, tmp_path, unknown).tracks("071", "645")

    assert CountingSource.made == 1
    (line,) = capsys.readouterr().err.splitlines()
    assert "not checked" in line and "ultralytics" in line

    CachedTrackSource(CountingSource, tmp_path, unknown | {"conf": 0.25}).tracks("071", "645")
    assert CountingSource.made == 2


def test_provenance_records_the_stored_metadata_and_what_was_not_checked(tmp_path):
    CachedTrackSource(CountingSource, tmp_path, {"ultralytics": "8.4", "conf": 0.1}).tracks(
        "071", "645"
    )
    cache = CachedTrackSource(CountingSource, tmp_path, {"ultralytics": None, "conf": 0.1})

    cache.tracks("071", "645")

    assert cache.provenance == {
        "071/645": {
            "stored": {"conf": 0.1, "ultralytics": "8.4"},
            "unverified": ["ultralytics"],
        }
    }


class RunRecordingSource(CountingSource):
    """A source that says how each computation ran (the provider of the detector)."""

    def tracks(self, scene, camera):
        self.last_run = {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
        return TRACKS


def test_how_a_run_went_is_stored_and_reported_but_not_checked(tmp_path, capsys):
    import json

    meta = {"engine": "onnx-cuda"}
    first = CachedTrackSource(RunRecordingSource, tmp_path, meta)
    first.tracks("071", "645")
    again = CachedTrackSource(CountingSource, tmp_path, meta)
    CountingSource.made = 0
    again.tracks("071", "645")

    run = {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
    with np.load(tmp_path / "scene_071_camera_645.npz") as f:
        assert json.loads(str(f["meta"])) == meta | {"run": run}
    assert CountingSource.made == 0  # "run" is not part of the recipe: still a hit
    assert first.provenance["071/645"]["stored"]["run"] == run
    assert again.provenance["071/645"]["stored"]["run"] == run


def test_a_corrupt_cache_file_is_recomputed(tmp_path, capsys):
    CountingSource.made = 0
    (tmp_path / "scene_071_camera_645.npz").write_bytes(b"not a zip file")

    CachedTrackSource(CountingSource, tmp_path, META).tracks("071", "645")

    assert CountingSource.made == 1
    assert "unreadable" in capsys.readouterr().err


def test_the_identity_of_each_input_is_checked_per_file(tmp_path, capsys):
    CountingSource.made = 0
    size = {"video_bytes": 100}
    CachedTrackSource(CountingSource, tmp_path, META, identity=lambda s, c: size).tracks(
        "071", "645"
    )
    CachedTrackSource(CountingSource, tmp_path, META, identity=lambda s, c: size).tracks(
        "071", "645"
    )
    assert CountingSource.made == 1

    other = {"video_bytes": 101}
    CachedTrackSource(CountingSource, tmp_path, META, identity=lambda s, c: other).tracks(
        "071", "645"
    )

    assert CountingSource.made == 2
    assert "video_bytes 100 != 101" in capsys.readouterr().err


class FellBackSource(CountingSource):
    last_run = {"providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider"]}


def test_tracks_after_a_tensorrt_fallback_are_not_stored(tmp_path, capsys):
    t = CachedTrackSource(FellBackSource, tmp_path, {"engine": "onnx-trt"}).tracks("071", "645")

    np.testing.assert_array_equal(t.frames, TRACKS.frames)
    assert not list(tmp_path.glob("*.npz"))
    assert "not TensorRT" in capsys.readouterr().err
