import pytest

from footfall_heatmaps.perception.domain.ports import SourceDropped
from footfall_heatmaps.perception.infrastructure.opencv_stream_source import (
    OpenCvStreamSource,
)
from tests.perception.synthetic_video import write_video


def test_a_stream_that_cannot_be_opened_is_dropped(tmp_path):
    with pytest.raises(SourceDropped):
        OpenCvStreamSource(str(tmp_path / "missing.avi"))


def test_a_stream_that_stops_is_dropped_after_its_frames(tmp_path):
    source = OpenCvStreamSource(str(write_video(tmp_path / "v.avi", n=5)))
    seen = []

    with pytest.raises(SourceDropped):
        for frame in source.frames():
            seen.append(frame.index)

    assert seen == [0, 1, 2, 3, 4] and source.fps == 10.0


def test_a_frame_carries_its_stream_time(tmp_path):
    source = OpenCvStreamSource(str(write_video(tmp_path / "v.avi", n=5)))
    times = []

    with pytest.raises(SourceDropped):
        for f in source.frames():
            times.append(round(f.timestamp, 3))

    assert times == [0.0, 0.1, 0.2, 0.3, 0.4]  # 10 fps: PTS, or frame count / fps
