"""FrameSource over a video file, decoded with OpenCV into memory only."""

import cv2

from space_usage_heatmaps.perception.domain.frame import Frame


class OpenCvFileSource:
    def __init__(self, path, stride=1, loop=False):
        """Every stride-th frame; skipped frames are grabbed, not decoded to an
        image. With loop, the video restarts at its end and indices keep counting.
        A video that ends more than one frame before its declared frame count
        (truncated, corrupt) is an error, not its end, so it is never cached
        as complete."""
        self.path, self.stride, self.loop = path, stride, loop
        cap = self._open()
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if not self.fps > 0:
            raise OSError(f"video {self.path} has no frame rate ({self.fps})")

    def _open(self):
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise OSError(f"cannot open video {self.path}")
        return cap

    def _check_complete(self, cap, read):
        """read: frames of this pass; the declared count may overstate by one."""
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if read == 0 or read < count - 1:
            raise OSError(
                f"video {self.path} ended early: decoded {read} of {count} frames"
                " (truncated or corrupt)"
            )

    def frames(self):
        cap = self._open()
        i = read = 0  # index across passes; frames of the current pass
        try:
            while True:
                if i % self.stride:
                    ok, image = cap.grab(), None
                else:
                    ok, image = cap.read()
                if not ok:
                    self._check_complete(cap, read)
                    if not self.loop:
                        return
                    cap.release()
                    cap = self._open()
                    read = 0
                    continue
                if image is not None:
                    yield Frame(i, i / self.fps, image)
                i += 1
                read += 1
        finally:
            cap.release()
