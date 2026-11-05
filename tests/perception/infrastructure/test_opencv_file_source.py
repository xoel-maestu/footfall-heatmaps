import pytest

from space_usage_heatmaps.perception.infrastructure.opencv_file_source import OpenCvFileSource
from tests.perception.synthetic_video import write_video


def test_every_frame_with_its_index_and_timestamp(tmp_path):
    video = write_video(tmp_path / "v.avi", n=12, fps=10.0)

    frames = list(OpenCvFileSource(video).frames())

    assert [f.index for f in frames] == list(range(12))
    assert frames[3].timestamp == pytest.approx(0.3)
    assert frames[0].image.shape == (120, 160, 3)


def test_stride_skips_frames_and_keeps_their_indices(tmp_path):
    video = write_video(tmp_path / "v.avi", n=12)

    assert [f.index for f in OpenCvFileSource(video, stride=5).frames()] == [0, 5, 10]


def test_loop_restarts_the_video_and_keeps_counting(tmp_path):
    video = write_video(tmp_path / "v.avi", n=4)

    frames = OpenCvFileSource(video, loop=True).frames()

    assert [next(frames).index for _ in range(10)] == list(range(10))


def test_fps_is_the_video_rate(tmp_path):
    assert OpenCvFileSource(write_video(tmp_path / "v.avi", fps=10.0)).fps == 10.0


def test_a_missing_video_is_an_error(tmp_path):
    with pytest.raises(OSError, match="cannot open video"):
        OpenCvFileSource(tmp_path / "missing.mp4")
