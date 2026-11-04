import pytest

pytest.importorskip("ultralytics")  # optional baseline group; skipped in CI

from space_usage_heatmaps.evaluation.baseline import ultralytics_tracks  # noqa: E402
from space_usage_heatmaps.evaluation.domain.run_config import RunConfig  # noqa: E402

CONFIG = RunConfig("yolov8n", "bytetrack", 640, "centroid", "ultralytics", 1)


def test_a_video_without_frames_is_a_clear_error_naming_it(tmp_path, monkeypatch):
    monkeypatch.setattr(ultralytics_tracks, "load_model", lambda config, weights_dir: None)
    monkeypatch.setattr(ultralytics_tracks, "video_frames", lambda path, stride: iter(()))
    video = tmp_path / "empty.mp4"
    source = ultralytics_tracks.UltralyticsTrackSource(CONFIG, lambda s, c: video, tmp_path)

    with pytest.raises(OSError, match="no frames.*empty.mp4"):
        source.tracks("071", "645")
