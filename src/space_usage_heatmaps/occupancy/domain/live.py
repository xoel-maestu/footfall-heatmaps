"""The live layer: dwell and passage with exponential decay, in memory only."""

import numpy as np

HALF_LIFE_S = 60.0


class LiveLayer:
    def __init__(self, shape, half_life_s=HALF_LIFE_S):
        self.half_life_s = half_life_s
        self.dwell = np.zeros(shape)
        self.passage = np.zeros(shape)
        self.t = None
        self._cells = {}  # track id -> its cell in the last frame

    def _decay(self, t):
        if self.t is not None and t > self.t:
            factor = 0.5 ** ((t - self.t) / self.half_life_s)
            self.dwell *= factor
            self.passage *= factor
        self.t = t if self.t is None else max(self.t, t)

    def add(self, t, track_ids, cells, seconds):
        """Dwell adds seconds per sample; passage adds 1 when a track enters a cell."""
        self._decay(t)
        np.add.at(self.dwell.reshape(-1), cells, seconds)
        now = dict(zip(np.asarray(track_ids).tolist(), np.asarray(cells).tolist(), strict=True))
        entered = [c for i, c in now.items() if self._cells.get(i) != c]
        np.add.at(self.passage.reshape(-1), entered, 1.0)
        self._cells = now

    def at(self, t):
        """(dwell, passage) decayed to time t."""
        self._decay(t)
        return self.dwell.copy(), self.passage.copy()
