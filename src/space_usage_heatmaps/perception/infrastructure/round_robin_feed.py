"""The frames of many sources, each read in its own thread, served in turn.

One process records every camera of a site with one shared detector; decoding
runs in one thread per source (OpenCV releases the GIL while it decodes), and
the consumer (record_cameras, in the caller's thread) takes the cameras in
turn, so the detector is only ever called from one thread.

- paced (video files): no frame is dropped. Each reader waits while its
  camera already has frames waiting, and the consumer waits for the camera
  whose turn it is, so a replay is deterministic and file-paced.
- live (streams): a reader never waits. Each camera keeps only its newest
  frame (the older one is replaced and counted in dropped), so a slow round
  never builds latency or blocks the stream. A camera with no new frame is
  skipped in that round. on_drop() from a source (reconnecting_frames) is
  served as (camera id, None) in that camera's turn.

Sources: camera id -> make(on_drop) -> an iterator of Frame. The feed ends
when every source ended, or when stop is set.

Errors: in a paced replay a reader's error is raised in the consumer (a replay
that skips part of a file is not the recipe). Live, one camera's error never
stops the others: it counts in that camera's errors, is served as a drop
(None, so its open slots are marked incomplete), and the camera is made again
after a backoff (1 s doubling to 30 s, back to 1 s once a frame arrived). Its
frame indices go on from the last one served, so its tracker sees one sequence.
"""

import queue
import threading
import time
from dataclasses import replace

from space_usage_heatmaps.perception.application.reconnect import FIRST_DELAY_S, MAX_DELAY_S

WAIT_S = 0.05  # live: how long the consumer waits when no camera has a new frame
JOIN_S = 5.0  # how long a closed feed waits for each reader to release its source
_END = object()


class _Error:
    def __init__(self, error):
        self.error = error


def round_robin_feed(sources, paced, stop=None, stats=None, sleep=None):
    """Starts the readers now; returns the generator of (camera id, frame or None).
    stats: a dict that gets camera id -> {"dropped": frames replaced unread,
    "drops": source drops, "errors": live source errors}, updated as it runs.
    sleep(seconds): the backoff wait before a failed live camera is made again
    (default: waits on stop)."""
    stop = stop or threading.Event()
    done = threading.Event()  # the consumer is gone: readers stop
    stats = {} if stats is None else stats
    cameras = list(sources)
    for c in cameras:
        stats[c] = {"dropped": 0, "drops": 0, "errors": 0}

    def ended():
        return stop.is_set() or done.is_set()

    def backoff(seconds):
        if sleep is not None:
            sleep(seconds)
            return
        until = time.monotonic() + seconds  # waits in slices, so a closed feed is seen too
        while not ended() and time.monotonic() < until:
            stop.wait(WAIT_S)

    if paced:
        boxes = {c: queue.Queue(maxsize=2) for c in cameras}
    else:
        boxes = {c: _Latest() for c in cameras}
        arrived = threading.Condition()

    def put(c, item):
        if paced:
            while not (stop.is_set() or done.is_set()):
                try:
                    boxes[c].put(item, timeout=WAIT_S)
                    return
                except queue.Full:
                    continue
        else:
            stats[c]["dropped"] += boxes[c].put(item)
            with arrived:
                arrived.notify()

    def read(c):
        def on_drop():
            stats[c]["drops"] += 1
            if not paced:
                boxes[c].drop()
                with arrived:
                    arrived.notify()

        if paced:
            try:
                for frame in sources[c](on_drop):
                    if ended():
                        return
                    put(c, frame)
                put(c, _END)
            except Exception as e:  # handed to the consumer, which raises it
                put(c, _Error(e))
            return
        delay, offset, last = FIRST_DELAY_S, 0, -1
        while not ended():
            try:
                for frame in sources[c](on_drop):
                    if ended():
                        return
                    delay = FIRST_DELAY_S
                    frame = replace(frame, index=frame.index + offset)
                    last = frame.index
                    put(c, frame)
                put(c, _END)
                return
            except Exception:  # this camera only: the others keep recording
                stats[c]["errors"] += 1
                on_drop()
                backoff(delay)
                delay = min(2 * delay, MAX_DELAY_S)
                offset = last + 1

    threads = [threading.Thread(target=read, args=(c,), name=f"source-{c}", daemon=True)
               for c in cameras]  # fmt: skip
    for t in threads:
        t.start()

    def take_paced(c):
        while not stop.is_set():
            try:
                return boxes[c].get(timeout=WAIT_S)
            except queue.Empty:
                continue
        return None

    def feed():
        active = list(cameras)
        try:
            while active and not stop.is_set():
                served = False
                for c in list(active):
                    if paced:
                        item = take_paced(c)
                        if item is None:
                            return
                        items = [item]
                    else:
                        items = boxes[c].take()
                    for item in items:
                        if item is _END:
                            active.remove(c)
                        elif isinstance(item, _Error):
                            raise item.error
                        else:
                            served = True
                            yield c, item
                        if stop.is_set():
                            return
                if not paced and not served and active:
                    with arrived:
                        arrived.wait(WAIT_S)
        finally:
            done.set()  # readers stop at their next frame or wait, and release their source
            for t in threads:
                t.join(timeout=JOIN_S)

    return feed()


class _Latest:
    """A camera's newest frame, a pending drop, or its end; put() says how many
    frames it replaced unread (0 or 1)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._dropped = False
        self._tail = None  # _END or an _Error, served after the last frame

    def put(self, item):
        with self._lock:
            if item is _END or isinstance(item, _Error):
                self._tail = item
                return 0
            replaced = int(self._frame is not None)
            self._frame = item
            return replaced

    def drop(self):
        with self._lock:
            self._dropped = True

    def take(self):
        """The items of this turn, in order: a drop (None), the newest frame, the end."""
        with self._lock:
            out = [None] if self._dropped else []
            if self._frame is not None:
                out.append(self._frame)
            if self._tail is not None:
                out.append(self._tail)
            self._frame, self._dropped, self._tail = None, False, None
            return out
