"""A tiny synthetic video written by the test itself: a bright block on grey."""

import cv2
import numpy as np


def write_video(path, n=12, fps=10.0, size=(160, 120), seed=0):
    """n frames, MJPG in AVI (no external codec needed). seed varies the pixels."""
    w, h = size
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    rng = np.random.default_rng(seed)
    for i in range(n):
        image = np.full((h, w, 3), int(rng.integers(40, 200)), np.uint8)
        image[30:90, 20 + 4 * i : 40 + 4 * i] = 255
        out.write(image)
    out.release()
    return path
