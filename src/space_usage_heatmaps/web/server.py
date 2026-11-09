"""The viewer's HTTP server (driving adapter, standard library only).

GET /                  the static page (web/static/index.html)
GET /api/live?camera=  the live layer of the recorder running in this process, decayed
                       to now and k-suppressed (R5); memory only, never written
                       anywhere; "ended" is when a file source was read to its
                       end (epoch seconds) or null; 503 once the recorder has failed
GET /api/slots?camera=&from=&to=
                       persisted 15-min slots starting in [from, to), read through
                       read_slots (k-suppressed, R5); times are ISO 8601 with a zone
                       or epoch seconds, clamped to [0, END_OF_TIME]; each slot
                       reads as occupancy from its dwell (slot_reading: mean and
                       peak 1-min people in view, observed seconds), never as a
                       count of ids
GET /api/slots/latest?camera=
                       {"start": the newest persisted slot start, or null}
GET /api/site          serve --site: the site id, its fusion and each camera's state
                       and processed fps (counts only); 404 for one camera
GET /api/fused         the fused floor map of a scene's cameras (serve --fused, an
                       export of eval --cameras), k-suppressed again here (R5); 404
                       without one
"""

import html
import json
import math
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from space_usage_heatmaps.occupancy.application.read_slots import read_minutes, read_slots
from space_usage_heatmaps.occupancy.domain.slots import K_MIN_PEOPLE, slot_reading, suppress
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
        self.error = None  # set by fail(): the recorder thread is gone
        self.ended = None  # set by end(): the file source was read to its end

    def attach(self, recorder):
        self._recorder = recorder
        return _Locked(recorder, self._lock)

    def fail(self, error):
        """The recorder stopped on error: the layer would only decay from here.
        Only the exception's type is kept: its message may hold the source URL."""
        self.error = type(error).__name__

    def end(self, t):
        """The file source ended normally at wall time t: the layer only decays from here."""
        self.ended = t

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

    def mark_incomplete(self, *args):  # a site recorder names the camera
        with self._lock:
            self._recorder.mark_incomplete(*args)

    def close(self):
        with self._lock:
            self._recorder.close()

    def __getattr__(self, name):  # written, past_retention: read after the frames end
        return getattr(self._recorder, name)


def _iso(t):
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value):
    """Epoch seconds of an ISO 8601 time with a zone, or of epoch seconds,
    clamped to [0, END_OF_TIME]; ValueError if it is neither or not finite."""
    try:
        t = float(value)
    except ValueError:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))  # ValueError if not a time
        if when.tzinfo is None:
            raise ValueError(f"time {value!r} needs a zone") from None
        t = when.timestamp()
    if not math.isfinite(t):
        raise ValueError(f"time {value!r} is not finite")
    return int(min(max(t, 0), END_OF_TIME))


def _grid(a):
    return a.round(DECIMALS).tolist()


def _frame(spec):
    return {"cell_m": spec.cell_m, "origin": [spec.origin_x, spec.origin_y]}


def live_payload(spec, dwell, passage, t, ended=None):
    return {
        **_frame(spec),
        "shape": [spec.ny, spec.nx],
        "dwell": _grid(dwell),
        "passage": _grid(passage),
        "t": t,
        "ended": ended,
    }


def _reading(dwell, minutes, incomplete):
    r = slot_reading(dwell, minutes, incomplete)
    peak = r["peak_start"]
    return {
        "mean_occupancy": None if r["mean_occupancy"] is None else round(r["mean_occupancy"], 3),
        "peak_occupancy": None if r["peak_occupancy"] is None else round(r["peak_occupancy"], 3),
        "peak_start": None if peak is None else _iso(peak),
        "observed_s": r["observed_s"],
    }


def slots_payload(camera_id, spec, slots, minutes=()):
    """One entry per slot start with both layers and its occupancy reading;
    slots of another grid are left out. minutes: the slots' SlotMinutes."""
    by_start, dwell = {}, {}
    for s in slots:
        if s.cell_m != spec.cell_m or s.grid.shape != (spec.ny, spec.nx):
            continue
        entry = by_start.setdefault(
            s.start_utc, {"start": _iso(s.start_utc), "incomplete": s.incomplete}
        )
        entry[s.layer] = _grid(s.grid)
        entry["incomplete"] = entry["incomplete"] or s.incomplete
        if s.layer == "dwell":
            dwell[s.start_utc] = s.grid
    of_start = {m.start_utc: m for m in minutes if m.cell_m == spec.cell_m}
    out = []
    for start, e in sorted(by_start.items()):
        if "dwell" in e and "passage" in e:
            out.append(e | _reading(dwell[start], of_start.get(start), e["incomplete"]))
    return {"camera": camera_id, **_frame(spec), "shape": [spec.ny, spec.nx], "slots": out}


def fused_payload(export, k=K_MIN_PEOPLE):
    """The export with both layers suppressed below k, whatever the file says."""
    dwell, passage = suppress(np.asarray(export["dwell"]), np.asarray(export["passage"]), k)
    return export | {"k": k, "dwell": _grid(dwell), "passage": _grid(passage)}


def make_server(
    host, port, camera_id, spec, db_path, live=None, clock=time.time, fused=None, site=None
):
    """A ThreadingHTTPServer for one camera; live is a LiveSource or None;
    fused, an eval --cameras export (dict) of a scene's fused map, or None.
    site: for a site (serve --site), a callable that returns /api/site; then
    camera_id is the site id, its slots and live layer are the fused ones."""
    if fused is not None and fused.get("passage_ids") == "global":
        raise ValueError(
            "the fused export's passage counts global ids (eval --metric mc-idf1), less "
            "accurate than E2a's; serve the E2a fused export (eval --cameras) instead"
        )

    page = (
        PAGE.read_text()
        .replace("__CAMERA__", html.escape(camera_id))
        .replace("__FUSED__", "0" if fused is None else "1")
        .replace("__SITE__", "0" if site is None else "1")
        .replace("__LIVE__", "0" if live is None else "1")
        .encode()
    )
    fused_body = None if fused is None else fused_payload(fused)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            if url.path in ("/", "/index.html"):
                return self._send(200, "text/html; charset=utf-8", page)
            if url.path == "/api/fused":
                if fused_body is None:
                    return self._json(404, {"error": "no fused map: start serve with --fused"})
                return self._json(200, fused_body)
            if url.path == "/api/site":
                if site is None:
                    return self._json(404, {"error": "not a site: start serve with --site"})
                return self._json(200, site())
            if url.path not in ("/api/live", "/api/slots", "/api/slots/latest"):
                return self._json(404, {"error": "not found"})
            if query.get("camera") != camera_id:
                return self._json(404, {"error": f"unknown camera; this server has {camera_id}"})
            if url.path == "/api/live":
                return self._live()
            if url.path == "/api/slots/latest":
                return self._latest()
            return self._slots(query)

        def _live(self):
            t = clock()
            if live is not None and live.error is not None:
                error = f"the recorder stopped ({live.error}); see the server log"
                return self._json(503, {"error": error})
            layers = None if live is None else live.at(t)
            if layers is None:
                return self._json(404, {"error": "no live source in this process"})
            return self._json(200, live_payload(spec, *layers, t, live.ended))

        def _latest(self):
            store = SqliteSlotStore(db_path)
            try:
                start = store.latest_start(camera_id)
            finally:
                store.close()
            return self._json(200, {"start": None if start is None else _iso(start)})

        def _slots(self, query):
            try:
                start = parse_time(query.get("from", "0"))
                end = parse_time(query.get("to", str(END_OF_TIME)))
            except ValueError as e:
                return self._json(400, {"error": f"from and to: {e}"})
            store = SqliteSlotStore(db_path)
            try:
                slots = read_slots(store, camera_id, start, end)
                minutes = read_minutes(store, camera_id, start, end)
            finally:
                store.close()
            return self._json(200, slots_payload(camera_id, spec, slots, minutes))

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
