"""scripts/site-parity.py: the P1 map comparison, on hand-made slots."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from space_usage_heatmaps.occupancy.domain.slots import SlotGrid
from space_usage_heatmaps.occupancy.infrastructure.sqlite_slot_store import SqliteSlotStore

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "site-parity.py"
T0 = 1_793_955_600


def parity():
    spec = importlib.util.spec_from_file_location("site_parity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tvd_compares_shapes_not_totals():
    m = parity()
    assert m.tvd([1, 1, 0], [2, 2, 0]) == 0
    assert m.tvd([1, 0], [0, 1]) == 1
    assert m.tvd([3, 1], [1, 1]) == pytest.approx(0.25)


def test_recorded_sums_the_slots_each_read_with_k(tmp_path):
    db = tmp_path / "x.db"
    store = SqliteSlotStore(db)
    for i, people in enumerate((6.0, 4.0)):  # the second slot's cell is under k on read
        start = T0 + 900 * i
        store.write([
            SlotGrid("071", start, 0.5, "dwell", np.array([[10.0, 0.0]]), 6, False),
            SlotGrid("071", start, 0.5, "passage", np.array([[people, 0.0]]), 6, False),
        ])  # fmt: skip
    store.close()

    total, info = parity().recorded(SimpleNamespace(id="071"), db)

    assert total.tolist() == [[10.0, 0.0]]
    assert info["slots"] == 2 and info["min_people_in_a_shown_cell"] == 6.0
