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
    recorder.add(T0, np.arange(5), np.full(5, -0.9), np.full(5, 2.1), 1.0)
    server = serve(live)

    status, kind, body = get(server, "/api/live?camera=645")

    assert status == 200 and kind.startswith("application/json")
    d = json.loads(body)
    assert set(d) == {"cell_m", "origin", "shape", "dwell", "passage", "t", "ended"}
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


@pytest.mark.parametrize("q", ["from=inf", "from=-inf", "to=nan"])
def test_a_time_that_is_not_finite_is_a_400(serve, q):
    """Regression: these dropped the connection (OverflowError in int(float(...)))."""
    assert get(serve(), f"/api/slots?camera=645&{q}")[0] == 400


@pytest.mark.parametrize(
    "q, n",
    [("from=1e30", 0), ("to=99999999999999999999", 1), ("from=-1e30", 1), ("to=-1e30", 0)],
)
def test_a_time_out_of_range_is_clamped_not_a_dropped_connection(serve, tmp_path, q, n):
    """Regression: these dropped the connection (OverflowError in the store's _iso)."""
    store = SqliteSlotStore(tmp_path / "x.db")
    store.write(
        [
            SlotGrid("645", T0, 0.5, layer, np.zeros((2, 4)), 0, False)
            for layer in ("dwell", "passage")
        ]
    )
    store.close()

    status, _, body = get(serve(), f"/api/slots?camera=645&{q}")

    assert status == 200 and len(json.loads(body)["slots"]) == n


def test_live_is_503_once_the_recorder_failed(serve):
    recorder, live = recording()
    recorder.add(T0, np.arange(5), np.full(5, -0.9), np.full(5, 2.1), 1.0)
    server = serve(live)
    assert get(server, "/api/live?camera=645")[0] == 200

    live.fail(RuntimeError("boom"))
    status, kind, body = get(server, "/api/live?camera=645")

    assert status == 503 and kind.startswith("application/json")
    assert "recorder stopped" in json.loads(body)["error"]


def test_live_is_503_when_the_recorder_failed_before_attaching(serve):
    live = LiveSource()
    live.fail(OSError("cannot open the source"))

    assert get(serve(live), "/api/live?camera=645")[0] == 503


def test_the_page_is_served_with_the_layer_names(serve):
    status, kind, body = get(serve(), "/")

    assert status == 200 and kind.startswith("text/html")
    page = body.decode()
    assert "Dwell (person-seconds)" in page and "Passage (people)" in page
    assert "http://" not in page and "https://" not in page  # no CDN at runtime
    assert get(serve(), "/nothing")[0] == 404


def test_reading_live_writes_nothing_to_disk(serve, tmp_path):
    recorder, live = recording()
    recorder.add(T0, np.arange(5), np.full(5, -0.9), np.full(5, 2.1), 1.0)
    server = serve(live)

    get(server, "/api/live?camera=645")

    assert list(tmp_path.iterdir()) == []


def test_live_says_when_the_source_ended(serve):
    recorder, live = recording()
    recorder.add(T0, np.arange(5), np.full(5, -0.9), np.full(5, 2.1), 1.0)
    server = serve(live)
    assert json.loads(get(server, "/api/live?camera=645")[2])["ended"] is None

    live.end(T0 + 30)
    status, _, body = get(server, "/api/live?camera=645")

    assert status == 200 and json.loads(body)["ended"] == T0 + 30


def test_latest_slot_is_the_newest_start_of_this_camera(serve, tmp_path):
    server = serve()
    assert json.loads(get(server, "/api/slots/latest?camera=645")[2]) == {"start": None}

    store = SqliteSlotStore(tmp_path / "x.db")
    grid = np.zeros((2, 4))
    store.write(
        [
            SlotGrid("645", T0 - 3 * 86400, 0.5, "dwell", grid, 0, False),
            SlotGrid("645", T0, 0.5, "dwell", grid, 0, False),
            SlotGrid("071", T0 + 86400, 0.5, "dwell", grid, 0, False),
        ]
    )
    store.close()

    status, _, body = get(server, "/api/slots/latest?camera=645")

    assert status == 200 and json.loads(body) == {"start": "2026-11-06T09:00:00Z"}
    assert get(server, "/api/slots/latest?camera=071")[0] == 404


def fused_export(dwell, passage):
    return {
        "scene": "071",
        "cameras": ["635", "645"],
        "config": "configs_e2a-fusion",
        "fusion": {"weighting": "visibility", "power": 1.0, "zone_m": None},
        "period": "the whole video",
        "k": 5,
        "cell_m": 0.5,
        "origin": [-1.0, 2.0],
        "shape": [2, 4],
        "dwell": dwell,
        "passage": passage,
    }


def test_fused_serves_the_export_k_suppressed_again(tmp_path):
    # An export edited by hand to show 3 people in a cell still reads as empty there.
    export = fused_export([[10, 0, 0, 0], [0, 0, 0, 8]], [[6, 0, 0, 0], [0, 0, 0, 3]])
    server = make_server("127.0.0.1", 0, "645", SPEC, tmp_path / "x.db", fused=export)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, kind, body = get(server, "/api/fused")
    finally:
        server.shutdown()
        server.server_close()

    d = json.loads(body)
    assert status == 200 and kind == "application/json"
    assert d["dwell"] == [[10, 0, 0, 0], [0, 0, 0, 0]] and d["passage"][1][3] == 0
    assert d["cameras"] == ["635", "645"] and d["scene"] == "071"


def test_fused_without_an_export_is_not_found(serve):
    status, _, body = get(serve(), "/api/fused")

    assert status == 404 and "no fused map" in json.loads(body)["error"]


def test_the_page_offers_the_fused_map_only_when_there_is_one(serve, tmp_path):
    plain = get(serve(), "/")[2].decode()
    server = make_server(
        "127.0.0.1",
        0,
        "645",
        SPEC,
        tmp_path / "x.db",
        fused=fused_export([[0] * 4] * 2, [[0] * 4] * 2),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with_fused = get(server, "/")[2].decode()
    finally:
        server.shutdown()
        server.server_close()

    assert 'data-fused="0"' in plain and 'data-fused="1"' in with_fused
    assert "Fused scene" in with_fused


def test_a_site_server_says_so_on_the_page_and_serves_its_cameras(tmp_path):
    info = {"site": "071", "cameras": [{"id": "635", "state": "recording"}]}
    server = make_server("127.0.0.1", 0, "071", SPEC, tmp_path / "x.db", site=lambda: info)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        page = get(server, "/")[2].decode()
        status, _, body = get(server, "/api/site")
    finally:
        server.shutdown()
        server.server_close()

    assert 'data-site="1"' in page and 'data-camera="071"' in page
    assert status == 200 and json.loads(body) == info


def test_a_camera_server_has_no_site(serve):
    status, _, body = get(serve(), "/api/site")

    assert status == 404 and "--site" in json.loads(body)["error"]
