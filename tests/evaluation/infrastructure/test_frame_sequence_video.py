"""Image sequences become a lossless video the perception pipeline can read."""

import cv2
import numpy as np

from space_usage_heatmaps.evaluation.infrastructure.frame_sequence_video import sequence_video


def write_frames(d, n, size=(48, 64)):
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (*size, 3), dtype=np.uint8) for _ in range(n)]
    paths = []
    for i, im in enumerate(images):
        paths.append(d / f"{i:06d}.png")
        cv2.imwrite(str(paths[-1]), im)
    return paths, images


def test_frames_are_written_in_order_without_loss_at_the_given_rate(tmp_path):
    paths, images = write_frames(tmp_path / "img", 3)

    video = sequence_video(paths, 2.0, tmp_path / "out" / "seq.mkv")

    cap = cv2.VideoCapture(str(video))
    assert cap.get(cv2.CAP_PROP_FPS) == 2.0
    for im in images:
        ok, frame = cap.read()
        assert ok and np.array_equal(frame, im)
    assert not cap.read()[0]


def test_an_existing_video_of_the_same_frames_is_reused(tmp_path):
    paths, _ = write_frames(tmp_path / "img", 2)
    first = sequence_video(paths, 2.0, tmp_path / "seq.mkv")
    stamp = first.stat().st_mtime_ns

    again = sequence_video(paths, 2.0, tmp_path / "seq.mkv")

    assert again == first and again.stat().st_mtime_ns == stamp


def test_a_video_of_other_frames_is_rebuilt(tmp_path):
    paths, _ = write_frames(tmp_path / "img", 3)
    sequence_video(paths[:2], 2.0, tmp_path / "seq.mkv")

    video = sequence_video(paths, 2.0, tmp_path / "seq.mkv")

    assert int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT)) == 3
