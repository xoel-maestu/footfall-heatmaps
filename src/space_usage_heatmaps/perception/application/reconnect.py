"""Use case: frames of a live source that may drop, reconnecting with exponential backoff.

On SourceDropped (while opening or while reading), on_drop() is called, then
sleep(delay) and a new connection; delay doubles from 1 s up to 30 s and
starts over after a frame arrives. A source that ends without dropping ends
the frames. Frame indices keep counting across connections, so the tracker
sees one sequence.
"""

from collections.abc import Callable, Iterator
from dataclasses import replace

from space_usage_heatmaps.perception.domain.frame import Frame
from space_usage_heatmaps.perception.domain.ports import FrameSource, SourceDropped

FIRST_DELAY_S, MAX_DELAY_S = 1, 30


def reconnecting_frames(
    open_source: Callable[[], FrameSource],
    sleep: Callable[[float], None],
    on_drop: Callable[[], None],
) -> Iterator[Frame]:
    delay, index = FIRST_DELAY_S, 0
    while True:
        try:
            for frame in open_source().frames():
                delay = FIRST_DELAY_S
                yield replace(frame, index=index)
                index += 1
            return
        except SourceDropped:
            on_drop()
            sleep(delay)
            delay = min(2 * delay, MAX_DELAY_S)
