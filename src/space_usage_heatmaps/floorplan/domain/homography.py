"""Ground-plane homography between floor metres and image pixels."""

import numpy as np


class Homography:
    """3x3 matrix mapping floor (x, y, 1) in metres to image (u, v, 1) in pixels."""

    def __init__(self, world_to_image):
        self.matrix = np.asarray(world_to_image, dtype=float)
        if self.matrix.shape != (3, 3) or abs(np.linalg.det(self.matrix)) < 1e-12:
            raise ValueError("homography must be an invertible 3x3 matrix")
        self.inverse = np.linalg.inv(self.matrix)

    def _apply(self, m, a, b):
        p = m @ np.vstack([a.ravel(), b.ravel(), np.ones(a.size)])
        p = p.reshape(3, *a.shape)
        return p[0] / p[2], p[1] / p[2], p[2]

    def to_image(self, xs, ys):
        u, v, _ = self._apply(self.matrix, np.asarray(xs, float), np.asarray(ys, float))
        return u, v

    def to_world(self, us, vs):
        x, y, _ = self._apply(self.inverse, np.asarray(us, float), np.asarray(vs, float))
        return x, y

    def sees(self, xs, ys, width, height):
        """Floor points in front of the camera that project inside the image."""
        u, v, w = self._apply(self.matrix, np.asarray(xs, float), np.asarray(ys, float))
        return (w > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
