"""The viewer's HTTP server (driving adapter, standard library only).

GET /                  the static page (web/static/index.html)
GET /api/live?camera=  the live layer of the recorder running in this process, decayed
                       to now; memory only, never written anywhere
GET /api/slots?camera=&from=&to=
                       persisted 15-min slots starting in [from, to), read through
                       read_slots (k-suppressed, R5); times are ISO 8601 with a zone
                       or epoch seconds
"""

import html
import json
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from space_usage_heatmaps.occupancy.application.read_slots import read_slots
from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore

PAGE = Path(__file__).parent / "static" / "index.html"
DECIMALS = 3  # person-seconds and people to the thousandth keep the JSON small
END_OF_TIME = 4_102_444_800  # 2100-01-01T00:00:00Z: no slot starts later


class LiveSource:
    """The live layer of a RecordSlots that runs in another thread: attach() wraps
    the recorder so its frames and the server's reads take turns."""

    def __init__(self):
        self._lock = threading.Lock()
        self._recorder = None

    def attach(self, recorder):
        self._recorder = recorder
        return _Locked(recorder, self._lock)

    def at(self, t):
        """(dwell, passage) at t, or None before a recorder is attached."""
        if self._recorder is None:
            return None
        with self._lock:
            return self._recorder.live(t)


class _Locked:
    def __init__(self, recorder, lock):
        self._recorder, self._lock = recorder, lock

    def add(self, *args, **kwargs):
        with self._lock:
            self._recorder.add(*args, **kwargs)

    def mark_incomplete(self):
        with self._lock:
            self._recorder.mark_incomplete()

    def close(self):
        with self._lock:
            self._recorder.close()

    def __getattr__(self, name):  # written, past_retention: read after the frames end
        return getattr(self._recorder, name)


def _iso(t):
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value):
    """Epoch seconds of an ISO 8601 time with a zone, or of epoch seconds."""
    try:
        return int(float(value))
    except ValueError:
        pass
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))  # ValueError if not a time
    if t.tzinfo is None:
        raise ValueError(f"time {value!r} needs a zone")
    return int(t.timestamp())


def _grid(a):
    return a.round(DECIMALS).tolist()


def _frame(spec):
    return {"cell_m": spec.cell_m, "origin": [spec.origin_x, spec.origin_y]}


def live_payload(spec, dwell, passage, t):
    return {
        **_frame(spec),
        "shape": [spec.ny, spec.nx],
        "dwell": _grid(dwell),
        "passage": _grid(passage),
        "t": t,
    }


def slots_payload(camera_id, spec, slots):
    """One entry per slot start with both layers; slots of another grid are left out."""
    by_start = {}
    for s in slots:
        if s.cell_m != spec.cell_m or s.grid.shape != (spec.ny, spec.nx):
            continue
        entry = by_start.setdefault(
            s.start_utc,
            {"start": _iso(s.start_utc), "incomplete": s.incomplete, "n_unique": s.n_unique},
        )
        entry[s.layer] = _grid(s.grid)
        entry["incomplete"] = entry["incomplete"] or s.incomplete
    return {
        "camera": camera_id,
        **_frame(spec),
        "shape": [spec.ny, spec.nx],
        "slots": [e for _, e in sorted(by_start.items()) if "dwell" in e and "passage" in e],
    }


def make_server(host, port, camera_id, spec, db_path, live=None, clock=time.time):
    """A ThreadingHTTPServer for one camera; live is a LiveSource or None."""

    page = PAGE.read_text().replace("__CAMERA__", html.escape(camera_id)).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            if url.path in ("/", "/index.html"):
                return self._send(200, "text/html; charset=utf-8", page)
            if url.path not in ("/api/live", "/api/slots"):
                return self._json(404, {"error": "not found"})
            if query.get("camera") != camera_id:
                return self._json(404, {"error": f"unknown camera; this server has {camera_id}"})
            if url.path == "/api/live":
                return self._live()
            return self._slots(query)

        def _live(self):
            t = clock()
            layers = None if live is None else live.at(t)
            if layers is None:
                return self._json(404, {"error": "no live source in this process"})
            return self._json(200, live_payload(spec, *layers, t))

        def _slots(self, query):
            try:
                start = parse_time(query.get("from", "0"))
                end = parse_time(query.get("to", str(END_OF_TIME)))
            except ValueError as e:
                return self._json(400, {"error": f"from and to: {e}"})
            store = SqliteSlotStore(db_path)
            try:
                slots = read_slots(store, camera_id, start, end)
            finally:
                store.close()
            return self._json(200, slots_payload(camera_id, spec, slots))

        def _json(self, status, payload):
            body = json.dumps(payload, separators=(",", ":")).encode()
            self._send(status, "application/json", body)

        def _send(self, status, kind, body):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # one line per request on stderr is noise at 1 Hz
            pass

    return ThreadingHTTPServer((host, port), Handler)
