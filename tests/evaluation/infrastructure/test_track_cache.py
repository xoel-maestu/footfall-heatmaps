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


def test_the_cache_holds_positions_only(tmp_path):
    CachedTrackSource(CountingSource, tmp_path).tracks("071", "645")

    (path,) = tmp_path.iterdir()
    assert path.name == "scene_071_camera_645.npz"
    with np.load(path, allow_pickle=False) as f:
        assert sorted(f.files) == ["boxes", "frames", "track_ids"]
