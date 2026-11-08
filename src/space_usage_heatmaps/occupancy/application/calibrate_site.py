"""Use case: a site's fusion weights from a calibration window (improvements phase 5).

The cameras run as in recording; each one's dwell is summed per cell over the
window, which starts at the first sample of any camera and lasts window_s
seconds. The weights are the evaluation's formula (fusion_weights) on those
observable inputs only: the cameras' recorded dwell and their floor coverage
(cells whose centre projects inside the image). No ground truth is read.
They are computed once and frozen in the site's weights file; recording
reads them and never updates them. Nothing per track is kept: only one dwell
grid per camera, in memory, for the window.
"""

import numpy as np

from space_usage_heatmaps.occupancy.domain.fusion import CameraGrid, fusion_weights
from space_usage_heatmaps.occupancy.domain.grid import Grid


class CalibrateSite:
    def __init__(self, cameras, coverage, spec, fusion, window_s):
        """coverage: (cameras, ny, nx) bool, in the order of cameras."""
        self.cameras = [str(c) for c in cameras]
        self.coverage = np.asarray(coverage, bool)
        if self.coverage.shape != (len(self.cameras), spec.ny, spec.nx):
            raise ValueError(
                f"camera coverage of shape {self.coverage.shape}, but the site has "
                f"{len(self.cameras)} cameras on a {spec.ny}x{spec.nx} grid"
            )
        self.spec, self.fusion, self.window_s = spec, fusion, window_s
        self._dwell = {c: np.zeros(spec.ny * spec.nx) for c in self.cameras}
        self._past = set()  # cameras with a sample past the window's end
        self.start = None

    def add(self, camera_id, t, track_ids, xs, ys, seconds):
        """One frame of one camera at t; the RecordSite signature, so the same loop drives it."""
        if camera_id not in self._dwell:
            raise KeyError(f"camera {camera_id!r} is not in the site")
        if self.start is None:
            self.start = t
        if t >= self.start + self.window_s:
            self._past.add(camera_id)
            return
        cells, _ = self.spec.cell_index(xs, ys)
        np.add.at(self._dwell[camera_id], cells, seconds)

    def mark_incomplete(self, camera_id):
        """A drop changes nothing here: the window is what each camera recorded."""

    def close(self):
        pass

    @property
    def complete(self):
        """Every camera has a sample past the window's end."""
        return len(self._past) == len(self.cameras)

    def dwell_s(self):
        """Seconds of dwell per camera in the window (a check that each one saw people)."""
        return {c: float(d.sum()) for c, d in self._dwell.items()}

    def weights(self):
        """(cameras, ny, nx) weights; on the joint coverage they sum to one."""
        shape = (self.spec.ny, self.spec.nx)
        grids = [
            CameraGrid(c, Grid(self.spec, self._dwell[c].reshape(shape), np.zeros(shape), 0), cov)
            for c, cov in zip(self.cameras, self.coverage, strict=True)
        ]
        return fusion_weights(grids, self.fusion)
