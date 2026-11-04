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
