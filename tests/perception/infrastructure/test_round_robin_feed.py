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
