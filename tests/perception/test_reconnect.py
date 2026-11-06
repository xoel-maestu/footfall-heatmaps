"""Failure mode: a dropped source reconnects with exponential backoff, 1 s up to 30 s."""

import numpy as np
import pytest

from space_usage_heatmaps.perception.application.reconnect import reconnecting_frames
from space_usage_heatmaps.perception.domain.frame import Frame
from space_usage_heatmaps.perception.domain.ports import SourceDropped

IMAGE = np.zeros((4, 4, 3), np.uint8)


class FlakySource:
    """Each connection: either fails to open, or yields some frames and drops or ends."""

    def __init__(self, script):
        self.script = list(script)  # per connection: "fail", ("drop", n) or ("end", n)
        self.opened = 0

    def open(self):
        self.opened += 1
        step = self.script.pop(0)
        if step == "fail":
            raise SourceDropped("cannot connect")
        return Connection(*step)


class Connection:
    def __init__(self, how, n):
        self.how, self.n = how, n

    def frames(self):
        for i in range(self.n):
            yield Frame(i, i / 10, IMAGE)  # every connection counts from 0
        if self.how == "drop":
            raise SourceDropped("stream stopped")


def run(script):
    source, waits, drops = FlakySource(script), [], []
    frames = list(reconnecting_frames(source.open, waits.append, lambda: drops.append(len(waits))))
    return frames, waits, drops, source


def test_a_source_that_ends_is_read_once():
    frames, waits, drops, source = run([("end", 3)])

    assert len(frames) == 3 and waits == [] and drops == [] and source.opened == 1


def test_backoff_doubles_from_one_second_up_to_thirty():
    frames, waits, _, _ = run(["fail"] * 7 + [("end", 1)])

    assert waits == [1, 2, 4, 8, 16, 30, 30]
    assert len(frames) == 1


def test_a_frame_after_reconnecting_resets_the_backoff():
    _, waits, _, _ = run([("drop", 2), "fail", ("drop", 1), ("end", 1)])

    assert waits == [1, 2, 1]


def test_each_drop_is_reported_before_waiting():
    _, waits, drops, _ = run([("drop", 2), "fail", ("end", 1)])

    assert drops == [0, 1]  # waits done when each drop was reported


def test_frame_indices_keep_counting_across_connections():
    frames, *_ = run([("drop", 2), ("end", 3)])

    assert [f.index for f in frames] == [0, 1, 2, 3, 4]


def test_other_errors_are_not_retried():
    def broken():
        raise ValueError("bad url")

    with pytest.raises(ValueError):
        list(reconnecting_frames(broken, lambda s: None, lambda: None))
