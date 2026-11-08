"""round_robin_feed: the frames of many sources, read in threads, served in turn."""

import threading
import time

import pytest

from space_usage_heatmaps.perception.domain.frame import Frame
from space_usage_heatmaps.perception.infrastructure.round_robin_feed import round_robin_feed


def frames(n, tag=0):
    def make(on_drop):
        for i in range(n):
            yield Frame(i, i / 10, tag)

    return make


def test_paced_sources_lose_no_frame_and_take_turns():
    feed = round_robin_feed({"a": frames(3), "b": frames(5)}, paced=True)

    got = [(c, f.index) for c, f in feed]

    assert got == [("a", 0), ("b", 0), ("a", 1), ("b", 1), ("a", 2), ("b", 2), ("b", 3), ("b", 4)]


def test_a_source_error_reaches_the_consumer():
    def broken(on_drop):
        yield Frame(0, 0.0, None)
        raise OSError("cannot open video x")

    with pytest.raises(OSError, match="cannot open"):
        list(round_robin_feed({"a": broken, "b": frames(3)}, paced=True))


def test_stop_ends_the_feed():
    stop = threading.Event()

    def endless(on_drop):
        i = 0
        while True:
            yield Frame(i, i / 10, None)
            i += 1

    got = []
    for _camera, f in round_robin_feed({"a": endless}, paced=True, stop=stop):
        got.append(f.index)
        if len(got) == 4:
            stop.set()
    assert got == [0, 1, 2, 3]


def test_live_sources_keep_only_the_newest_frame():
    release = threading.Event()

    def fast(on_drop):
        for i in range(50):
            yield Frame(i, i / 10, None)
        release.set()
        time.sleep(0.2)

    feed = round_robin_feed({"a": fast}, paced=False)
    release.wait(5)
    time.sleep(0.05)
    got = [f.index for _, f in feed]

    assert got[-1] == 49 and len(got) < 50  # the older ones were replaced, not queued


def test_a_live_drop_is_a_none_frame_of_that_camera():
    def dropping(on_drop):
        yield Frame(0, 0.0, None)
        time.sleep(0.05)
        on_drop()
        time.sleep(0.05)
        yield Frame(1, 0.1, None)
        time.sleep(0.05)

    feed = round_robin_feed({"a": dropping}, paced=False)
    got = [(c, None if f is None else f.index) for c, f in feed]

    assert got == [("a", 0), ("a", None), ("a", 1)]


class Flaky:
    """A live source whose connections follow a script: per call, how many
    frames it gives before it raises (an int), or None to give them and end."""

    def __init__(self, script, period=0.002):
        self.script, self.period, self.calls = list(script), period, 0

    def __call__(self, on_drop):
        n, fails = self.script[self.calls]
        self.calls += 1
        for i in range(n):
            time.sleep(self.period)
            yield Frame(i, i / 10, None)
        time.sleep(self.period)
        if fails:
            raise RuntimeError("decoder error")


def test_a_failing_live_camera_is_retried_with_backoff_and_the_others_keep_going():
    delays = []

    def sleep(s):
        delays.append(s)
        time.sleep(0.001)

    sources = {
        "ok": Flaky([(40, False)]),
        "broken": Flaky([(2, True), (3, True), (2, False)]),  # fails twice while recording
        "down": Flaky([(0, True), (0, True), (0, True), (2, False)]),  # down at start
    }
    stats = {}
    got = list(round_robin_feed(sources, paced=False, stats=stats, sleep=sleep))

    by = {c: [None if f is None else f.index for k, f in got if k == c] for c in sources}
    assert len([i for i in by["ok"] if i is not None]) >= 30  # ran on through both failures
    assert by["broken"].count(None) == 2 and by["down"].count(None) == 3
    assert sources["broken"].calls == 3 and sources["down"].calls == 4
    real = [i for i in by["broken"] if i is not None]
    assert real == sorted(real) and len(set(real)) == len(real)  # indices go on after a retry
    assert stats["broken"]["errors"] == 2 and stats["down"]["errors"] == 3
    assert stats["ok"]["errors"] == 0
    # backoff doubles while a camera stays down and starts over once a frame arrived
    assert sorted(delays) == sorted([1, 1] + [1, 2, 4])


def test_a_paced_replay_still_raises_a_source_error():
    def broken(on_drop):
        raise OSError("cannot read video x")
        yield

    with pytest.raises(OSError, match="cannot read"):
        list(round_robin_feed({"a": broken, "b": frames(3)}, paced=True, sleep=lambda s: None))
