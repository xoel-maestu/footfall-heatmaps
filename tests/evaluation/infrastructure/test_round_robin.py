import itertools

import pytest

from footfall_heatmaps.evaluation.infrastructure.round_robin import round_robin


class Clock:
    """A fake monotonic clock: each step of a camera costs `cost` seconds."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def camera(clock, cost, log, name):
    for i in itertools.count():
        clock.now += cost
        log.append((name, i))
        yield


def test_round_robin_steps_the_cameras_in_turn_and_reports_processed_fps():
    clock, log = Clock(), []
    steps = [camera(clock, 0.05, log, "a"), camera(clock, 0.05, log, "b")]

    r = round_robin(steps, seconds=10, clock=clock, vram_mb=lambda: 100.0)

    assert [n for n, _ in log[:4]] == ["a", "b", "a", "b"]
    # one round = 0.1 s, so each camera processes 10 frames a second
    assert r["cameras"] == 2
    assert r["processed_fps_mean"] == pytest.approx(10.0, rel=0.02)
    assert r["processed_fps_p5"] == pytest.approx(10.0, rel=0.02)
    assert len(r["per_camera"]) == 2
    assert r["vram_peak_mb"] == 100.0
    assert 9.9 <= r["seconds"] <= 10.2


def test_the_first_round_is_warm_up_and_not_timed():
    clock = Clock()
    costs = iter([5.0] + [0.1] * 1000)  # a slow first frame: the model loads

    def slow_start():
        while True:
            clock.now += next(costs)
            yield

    r = round_robin([slow_start()], seconds=5, clock=clock, vram_mb=lambda: 0.0)

    assert r["processed_fps_mean"] == pytest.approx(10.0, rel=0.02)


def test_p5_catches_slow_seconds_and_vram_is_the_peak():
    clock = Clock()
    vram = iter(itertools.chain([50.0, 300.0], itertools.repeat(120.0)))

    def uneven():
        for i in itertools.count():
            clock.now += 0.5 if i % 50 == 49 else 0.05  # one stall every 50 frames
            yield

    r = round_robin([uneven()], seconds=60, clock=clock, vram_mb=lambda: next(vram))

    assert r["processed_fps_p5"] < r["processed_fps_mean"]
    assert r["vram_peak_mb"] == 300.0
