import pytest

from footfall_heatmaps.perception.infrastructure.opencv_file_source import OpenCvFileSource
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


def _cut_in_half(video):
    data = video.read_bytes()
    video.write_bytes(data[: len(data) // 2])
    return video


def test_a_truncated_video_is_an_error_not_its_end(tmp_path):
    video = _cut_in_half(write_video(tmp_path / "v.avi", n=40))

    with pytest.raises(OSError, match=r"v\.avi.*decoded \d+ of 40 frames"):
        list(OpenCvFileSource(video).frames())


def test_a_truncated_video_is_an_error_with_stride_and_loop_too(tmp_path):
    video = _cut_in_half(write_video(tmp_path / "v.avi", n=40))

    with pytest.raises(OSError, match="of 40 frames"):
        list(OpenCvFileSource(video, stride=5).frames())
    with pytest.raises(OSError, match="of 40 frames"):
        list(OpenCvFileSource(video, loop=True).frames())


def test_a_video_with_no_frame_is_an_error(tmp_path, monkeypatch):
    import cv2

    video = write_video(tmp_path / "v.avi", n=4)
    monkeypatch.setattr(cv2.VideoCapture, "read", lambda self: (False, None))

    with pytest.raises(OSError, match="decoded 0 of"):
        list(OpenCvFileSource(video).frames())


def test_a_video_without_a_frame_rate_is_an_error_at_open(tmp_path, monkeypatch):
    import cv2

    video = write_video(tmp_path / "v.avi", n=4)
    real = cv2.VideoCapture.get
    monkeypatch.setattr(
        cv2.VideoCapture, "get", lambda self, p: 0.0 if p == cv2.CAP_PROP_FPS else real(self, p)
    )

    with pytest.raises(OSError, match="frame rate"):
        OpenCvFileSource(video)
