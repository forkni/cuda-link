"""
TDReceiverEngine.initialize_receiver() rejects a segment whose header fails validation.

The magic, version and slot-count reads go through the SHMProtocol helpers, so these
pin the reject paths (and that each one releases the mapping it opened) alongside them.
"""

from __future__ import annotations

import contextlib
import struct
import uuid
from multiprocessing.shared_memory import SharedMemory

import pytest

from cuda_link.shm_protocol import MAGIC_OFFSET, NUM_SLOTS_OFFSET, SHMLayout

from .test_tdreceiver_shm_leak import _FakeCudaFailsOnStreamCreate, _make_receiver, _write_valid_shm_frame


@pytest.fixture
def producer_shm():
    layout = SHMLayout(num_slots=1)
    shm = SharedMemory(create=True, name=f"test_hdr_{uuid.uuid4().hex[:8]}", size=layout.total_size)
    _write_valid_shm_frame(shm)
    yield shm
    shm.close()
    with contextlib.suppress(FileNotFoundError):
        shm.unlink()


def _initialize(shm: SharedMemory, monkeypatch: pytest.MonkeyPatch) -> tuple[bool, list[str], list[SharedMemory]]:
    import TDReceiver as TDReceiverMod

    opened: list[SharedMemory] = []

    class _TrackingSharedMemory(SharedMemory):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closed = False
            opened.append(self)

        def close(self):
            self.closed = True
            super().close()

    # Header rejection precedes stream creation, so the fake's failing step is never reached.
    monkeypatch.setattr(TDReceiverMod, "get_cuda_runtime", lambda device: _FakeCudaFailsOnStreamCreate())
    monkeypatch.setattr(TDReceiverMod, "SharedMemory", _TrackingSharedMemory)

    engine = _make_receiver(shm.name)
    logs: list[str] = []
    engine._log = lambda msg, *a, **k: logs.append(msg)
    return engine.initialize_receiver(), logs, opened


def test_foreign_magic_is_rejected(producer_shm: SharedMemory, monkeypatch: pytest.MonkeyPatch) -> None:
    struct.pack_into("<I", producer_shm.buf, MAGIC_OFFSET, 0xDEADBEEF)

    result, logs, opened = _initialize(producer_shm, monkeypatch)

    assert result is False
    assert any("Protocol magic mismatch" in m and "0xDEADBEEF" in m for m in logs)
    assert [h.closed for h in opened] == [True]


@pytest.mark.parametrize("num_slots", [0, 11], ids=["zero", "above-max"])
def test_out_of_range_num_slots_is_rejected(
    producer_shm: SharedMemory, monkeypatch: pytest.MonkeyPatch, num_slots: int
) -> None:
    struct.pack_into("<I", producer_shm.buf, NUM_SLOTS_OFFSET, num_slots)

    result, logs, opened = _initialize(producer_shm, monkeypatch)

    assert result is False
    assert any(f"Invalid num_slots: {num_slots}" in m for m in logs)
    assert [h.closed for h in opened] == [True]
