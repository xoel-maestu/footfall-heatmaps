"""The viewer's HTTP API over a real ThreadingHTTPServer on 127.0.0.1, with a
real RecordSlots for the live layer and a real SQLite store for the slots."""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from space_usage_heatmaps.occupancy.application.record_slots import RecordSlots
from space_usage_heatmaps.occupancy.domain.grid import GridSpec
from space_usage_heatmaps.occupancy.domain.slots import SLOT_S, SlotGrid
from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore
from space_usage_heatmaps.web.server import LiveSource, make_server

T0 = 1_793_955_600  # 2026-11-06T09:00:00Z
SPEC = GridSpec(origin_x=-1.0, origin_y=2.0, cell_m=0.5, nx=4, ny=2)


def get(server, path):
    host, port = server.server_address[:2]
    try:
        with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=5) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


@pytest.fixture
def serve(tmp_path):
    started = []

    def start(live=None):
        server = make_server(
            "127.0.0.1", 0, "645", SPEC, tmp_path / "x.db", live, clock=lambda: T0 + 60
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        return server

    yield start
    for s in started:
        s.shutdown()
        s.server_close()


def recording():
    store = SqliteSlotStore(":memory:")
    recorder = RecordSlots(store, "645", SPEC)
    live = LiveSource()
    return live.attach(recorder), live


def test_live_has_the_contract_shape(serve):
    recorder, live = recording()
    recorder.add(T0, np.arange(3), np.full(3, -0.9), np.full(3, 2.1), 1.0)
    server = serve(live)

    status, kind, body = get(server, "/api/live?camera=645")

    assert status == 200 and kind.startswith("application/json")
    d = json.loads(body)
    assert set(d) == {"cell_m", "origin", "shape", "dwell", "passage", "t"}
    assert d["cell_m"] == 0.5 and d["origin"] == [-1.0, 2.0] and d["shape"] == [2, 4]
    assert np.asarray(d["dwell"]).shape == (2, 4) and np.asarray(d["passage"]).shape == (2, 4)
    assert d["dwell"][0][0] > 0 and d["passage"][0][0] > 0
    assert d["t"] == T0 + 60  # decayed to the clock: one half-life


def test_live_without_a_live_source_or_for_another_camera_is_not_found(serve):
    recorder, live = recording()
    assert get(serve(None), "/api/live?camera=645")[0] == 404
    assert get(serve(live), "/api/live?camera=999")[0] == 404


def test_slots_are_read_through_k_suppression(serve, tmp_path):
    store = SqliteSlotStore(tmp_path / "x.db")
    dwell = np.array([[9.0, 4.0, 0, 0], [0, 0, 0, 0]])
    passage = np.array([[6.0, 2.0, 0, 0], [0, 0, 0, 0]])  # the second cell is under k
    store.write(
        [
            SlotGrid("645", T0, 0.5, "dwell", dwell, 8, False),
            SlotGrid("645", T0, 0.5, "passage", passage, 8, False),
            SlotGrid("645", T0 + SLOT_S, 0.5, "dwell", dwell, 3, True),
            SlotGrid("645", T0 + SLOT_S, 0.5, "passage", passage, 3, True),
        ]
    )
    store.close()
    server = serve()

    status, kind, body = get(
        server, "/api/slots?camera=645&from=2026-11-06T09:00:00Z&to=2026-11-06T09:30:00Z"
    )

    assert status == 200 and kind.startswith("application/json")
    d = json.loads(body)
    assert set(d) == {"camera", "cell_m", "origin", "shape", "slots"}
    assert d["origin"] == [-1.0, 2.0] and d["shape"] == [2, 4]
    first, second = d["slots"]
    assert set(first) == {"start", "incomplete", "n_unique", "dwell", "passage"}
    assert first["start"] == "2026-11-06T09:00:00Z" and first["incomplete"] is False
    assert first["dwell"][0][:2] == [9.0, 0.0] and first["passage"][0][:2] == [6.0, 0.0]
    assert second["n_unique"] == 0 and second["incomplete"] is True  # 3 people: under k


def test_slots_filter_by_time_and_reject_a_bad_time(serve, tmp_path):
    store = SqliteSlotStore(tmp_path / "x.db")
    grid = np.zeros((2, 4))
    store.write([SlotGrid("645", T0, 0.5, layer, grid, 0, False) for layer in ("dwell", "passage")])
    store.close()
    server = serve()

    later = json.loads(get(server, f"/api/slots?camera=645&from={T0 + SLOT_S}")[2])
    assert later["slots"] == []
    assert len(json.loads(get(server, "/api/slots?camera=645")[2])["slots"]) == 1
    assert get(server, "/api/slots?camera=645&from=yesterday")[0] == 400
    assert get(server, "/api/slots?camera=999")[0] == 404


def test_the_page_is_served_with_the_layer_names(serve):
    status, kind, body = get(serve(), "/")

    assert status == 200 and kind.startswith("text/html")
    page = body.decode()
    assert "Dwell (person-seconds)" in page and "Passage (people)" in page
    assert "http://" not in page and "https://" not in page  # no CDN at runtime
    assert get(serve(), "/nothing")[0] == 404


def test_reading_live_writes_nothing_to_disk(serve, tmp_path):
    recorder, live = recording()
    recorder.add(T0, np.arange(3), np.full(3, -0.9), np.full(3, 2.1), 1.0)
    server = serve(live)

    get(server, "/api/live?camera=645")

    assert list(tmp_path.iterdir()) == []
