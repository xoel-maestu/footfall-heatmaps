"""FrameSource over a video file, decoded with OpenCV into memory only."""

import cv2

from space_usage_heatmaps.perception.domain.frame import Frame


class OpenCvFileSource:
    def __init__(self, path, stride=1, loop=False):
        """Every stride-th frame; skipped frames are grabbed, not decoded to an
        image. With loop, the video restarts at its end and indices keep counting."""
        self.path, self.stride, self.loop = path, stride, loop
        cap = self._open()
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

    def _open(self):
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise OSError(f"cannot open video {self.path}")
        return cap

    def frames(self):
        cap = self._open()
        i = 0
        try:
            while True:
                if i % self.stride:
                    ok, image = cap.grab(), None
                else:
                    ok, image = cap.read()
                if not ok:
                    if not self.loop or i == 0:
                        return
                    cap.release()
                    cap = self._open()
                    continue
                if image is not None:
                    yield Frame(i, i / self.fps, image)
                i += 1
        finally:
            cap.release()
