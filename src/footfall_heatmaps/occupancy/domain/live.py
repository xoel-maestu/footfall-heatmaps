"""The live layer: dwell and passage with exponential decay, in memory only.

R5 holds here too: a cell shows only once k distinct ids were seen in it within
the last RECENT_S seconds; with fewer it reads as empty, so a lone cell never
gives away one person's recent position. The (id, cell) -> last seen
bookkeeping lives in memory and is pruned as it ages past RECENT_S.
"""

import numpy as np

from footfall_heatmaps.occupancy.domain.slots import K_MIN_PEOPLE, suppress

HALF_LIFE_S = 60.0
RECENT_S = 2 * HALF_LIFE_S  # the window of distinct people per cell: two half-lives


class LiveLayer:
    def __init__(self, shape, half_life_s=HALF_LIFE_S, k=K_MIN_PEOPLE, recent_s=RECENT_S):
        self.half_life_s, self.k, self.recent_s = half_life_s, k, recent_s
        self.dwell = np.zeros(shape)
        self.passage = np.zeros(shape)
        self.t = None
        self._cells = {}  # track id -> its cell in the last frame
        self.last_seen = {}  # (track id, cell) -> last time seen, pruned past recent_s

    def _decay(self, t):
        if self.t is not None and t > self.t:
            factor = 0.5 ** ((t - self.t) / self.half_life_s)
            self.dwell *= factor
            self.passage *= factor
        self.t = t if self.t is None else max(self.t, t)

    def _prune(self):
        cutoff = self.t - self.recent_s
        self.last_seen = {key: s for key, s in self.last_seen.items() if s >= cutoff}

    def add(self, t, track_ids, cells, seconds):
        """Dwell adds seconds per sample; passage adds 1 when a track enters a cell."""
        self._decay(t)
        np.add.at(self.dwell.reshape(-1), cells, seconds)
        now = dict(zip(np.asarray(track_ids).tolist(), np.asarray(cells).tolist(), strict=True))
        entered = [c for i, c in now.items() if self._cells.get(i) != c]
        np.add.at(self.passage.reshape(-1), entered, 1.0)
        self._cells = now
        for key in now.items():
            self.last_seen[key] = max(t, self.last_seen.get(key, t))
        self._prune()

    def people(self):
        """Distinct ids per cell seen in the last recent_s seconds."""
        counts = np.zeros(self.dwell.size)
        if self.last_seen:
            np.add.at(counts, [c for _, c in self.last_seen], 1.0)
        return counts.reshape(self.dwell.shape)

    def raw(self, t):
        """(dwell, passage, people) decayed to time t, before k: for fusion in
        memory (fuse, then suppress), never shown as is."""
        self._decay(t)
        self._prune()
        return self.dwell.copy(), self.passage.copy(), self.people()

    def at(self, t):
        """(dwell, passage) decayed to time t, empty where fewer than k people
        were seen in the last recent_s seconds."""
        self._decay(t)
        self._prune()
        people = self.people()
        dwell, _ = suppress(self.dwell, people, self.k)
        passage, _ = suppress(self.passage, people, self.k)
        return dwell, passage
