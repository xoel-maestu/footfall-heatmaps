"""YOLOv8n COCO + Ultralytics' built-in ByteTrack on a video, frames in memory only.

The recipe is Ultralytics' own: model.track(persist=True) per frame with
bytetrack.yaml unchanged (this ultralytics release defaults to another
tracker, so it is named explicitly), class person, the default imgsz and the
track-mode confidence (0.1). Frames are decoded into memory, never written.
"""

import os
import sys
import time

# Before ultralytics is imported: no usage analytics and no automatic pip
# installs (both need ONLINE).
os.environ["YOLO_OFFLINE"] = "1"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from footfall_heatmaps.evaluation.baseline.recipe import (  # noqa: E402
    CLASSES,
    CONF,
    TRACKERS,
    WEIGHTS,
)
from footfall_heatmaps.evaluation.domain.tracks import Tracks  # noqa: E402


def load_model(config, weights_dir):
    """The detector, downloaded once into weights_dir by ultralytics."""
    if config.detector not in WEIGHTS or config.tracker not in TRACKERS:
        raise ValueError(f"baseline runs {sorted(WEIGHTS)} with {sorted(TRACKERS)} only")
    weights_dir.mkdir(parents=True, exist_ok=True)
    return YOLO(str(weights_dir / WEIGHTS[config.detector]))


def video_frames(path, stride, loop=False):
    """(index, image) of every stride-th frame; skipped frames are grabbed, not decoded
    to an image. With loop, the video restarts at its end and indices keep counting."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path}")
    i = 0
    try:
        while True:
            if i % stride:
                ok = cap.grab()
                image = None
            else:
                ok, image = cap.read()
            if not ok:
                if not loop or i == 0:
                    return
                cap.release()
                cap = cv2.VideoCapture(str(path))
                continue
            if image is not None:
                yield i, image
            i += 1
    finally:
        cap.release()


def track_frames(model, config, frames):
    """(index, track ids, x1 y1 x2 y2 boxes) per frame, tracked boxes only."""
    for i, image in frames:
        r = model.track(
            image,
            persist=True,
            tracker=TRACKERS[config.tracker],
            classes=CLASSES,
            conf=CONF,
            imgsz=config.input_size,
            verbose=False,
        )[0]
        b = r.boxes
        if b is None or b.id is None:
            yield i, np.empty(0, dtype=int), np.empty((0, 4))
        else:
            yield i, b.id.int().cpu().numpy(), b.xyxy.cpu().numpy().astype(float)


class UltralyticsTrackSource:
    """TrackSource over the reader's videos; a fresh model (and tracker) per video."""

    def __init__(self, config, video_of, weights_dir):
        self.config = config
        self.video_of = video_of
        self.weights_dir = weights_dir

    def tracks(self, scene, camera):
        model = load_model(self.config, self.weights_dir)
        ids, frames, boxes = [], [], []
        start = time.monotonic()
        path = self.video_of(scene, camera)
        video = video_frames(path, self.config.stride)
        for n, (i, tid, xyxy) in enumerate(track_frames(model, self.config, video), 1):
            ids.append(tid)
            frames.append(np.full(len(tid), i))
            boxes.append(xyxy)
            if n % 1000 == 0:
                fps = n / (time.monotonic() - start)
                print(f"scene {scene} camera {camera}: {n} frames, {fps:.1f} fps", file=sys.stderr)
        if not ids:
            raise OSError(f"no frames decoded from video {path}")
        return Tracks(
            np.concatenate(ids).astype(int),
            np.concatenate(frames).astype(int),
            np.concatenate(boxes).reshape(-1, 4),
        )
