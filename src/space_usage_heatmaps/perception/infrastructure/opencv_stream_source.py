"""FrameSource over a live stream (rtsp://...), decoded with OpenCV into memory only.

The connection opens in the constructor. Failing to open, or a read that
fails, raises SourceDropped: for a live stream an end is a drop, and
reconnecting_frames opens a new one.
"""

import cv2

from space_usage_heatmaps.perception.domain.frame import Frame
from space_usage_heatmaps.perception.domain.ports import SourceDropped


class OpenCvStreamSource:
    def __init__(self, url, stride=1):
        self.url, self.stride = url, stride
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            self.cap.release()
            raise SourceDropped(f"cannot open stream {url}")
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 0 else 30.0  # a stream that does not say: its usual rate

    def frames(self):
        i = 0
        try:
            while True:
                if i % self.stride:
                    ok, image = self.cap.grab(), None
                else:
                    ok, image = self.cap.read()
                if not ok:
                    raise SourceDropped(f"stream {self.url} stopped")
                if image is not None:
                    yield Frame(i, i / self.fps, image)
                i += 1
        finally:
            self.cap.release()
