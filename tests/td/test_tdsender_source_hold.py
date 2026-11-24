# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""The TD Sender keeps TD's CUDAMemory object until the copy that reads it has completed.

TD gives one new CUDAMemory block per top.cudaMemory() call, and the block lives only as long
as the Python object. A dropped object's address goes to the next cudaMemory() call (probe in
TD 2025.33230, 2026-11-23). So:

- Blocking export (the default): the CUDAMemoryRef of the frame holds the TD object for the
  whole export() call. Nothing is held after the call.
- Async export (CUDALINK_EXPORT_SYNC=0): the ref is held until Exporter.slot_copy_done() says
  the queued copy has completed, with one entry per ring slot as the cap.

GPU-free harness: Exporter.open is patched to a fake that records the calls.
"""

from __future__ import annotations

import contextlib
import sys
import types
import uuid
import weakref

import pytest
import TDSender
from Exporter import FrameOutcome
from fakes import FakeTDHost, FakeTOPHandle, fake_exporter_open  # noqa: F401
from SHMProtocol import DtypeCodec
from TDConfig import TDSenderConfig
from TDSender import TDSenderEngine

_NUM_SLOTS = 3


@pytest.fixture(autouse=True)
def _drop_kept_sources():
    """The store of never-freed blocks lives in sys.modules: drop it between tests."""
    sys.modules.pop(TDSender._KEPT_SOURCES_KEY, None)
    yield
    sys.modules.pop(TDSender._KEPT_SOURCES_KEY, None)


def _kept() -> list:
    store = sys.modules.get(TDSender._KEPT_SOURCES_KEY)
    return list(store.blocks) if store is not None else []


class _FakeExporter:
    """Exporter double with a settable set of slots whose copy is still queued."""

    def __init__(self, spec) -> None:
        self.data_size = spec.width * spec.height * spec.channels * DtypeCodec.itemsize(spec.dtype)
        self.write_idx = 0
        self.frame_count = 0
        self.ipc_stream = types.SimpleNamespace(value=0)
        self.shm_handle = None
        self.pending: set[int] = set()  # slots whose copy has not completed
        self.keep_pending = True  # each export leaves its slot pending
        self.drain_calls = 0
        self.recover_code = 0  # what recover() finds latched after its stream wait
        self.recover_calls = 0
        self.closed = False
        self.owners_alive_in_export: list[bool] = []
        self._frame_owner_ref = None

    def record_source_sync(self, stream_id: int) -> None:
        pass

    def is_ready(self) -> bool:
        return True

    def export(self, frame) -> FrameOutcome:
        if self._frame_owner_ref is not None:
            self.owners_alive_in_export.append(self._frame_owner_ref() is not None)
        slot = self.write_idx % _NUM_SLOTS
        if self.keep_pending:
            self.pending.add(slot)
        self.frame_count += 1
        self.write_idx += 1
        return FrameOutcome.PUBLISHED

    def slot_copy_done(self, slot: int) -> bool:
        return slot not in self.pending

    def drain(self) -> None:
        self.drain_calls += 1
        self.pending.clear()

    def recover(self) -> int:
        self.recover_calls += 1
        if self.recover_code == 0:
            self.pending.clear()
        return self.recover_code

    def close(self) -> None:
        """Like Exporter.close(): returns None, and waits only when no error is latched."""
        self.closed = True
        if self.recover_code == 0:
            self.pending.clear()


class _Owner:
    """Stands in for TD's CUDAMemory object (weak-referenceable)."""


class _OwnerTOP(FakeTOPHandle):
    """Gives each cuda_memory() call a new owner and keeps only weak refs to them."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.owner_refs: list[weakref.ref] = []

    def cuda_memory(self, stream=None, pixel_format=None):
        cm = super().cuda_memory(stream=stream, pixel_format=pixel_format)
        cm.owner = _Owner()
        self.owner_refs.append(weakref.ref(cm.owner))
        return cm


def _setup(export_sync: bool | None) -> tuple[TDSenderEngine, _OwnerTOP]:
    top = _OwnerTOP(
        pixel_format="32-bit float (RGBA)",
        pixel_format_name="rgba32float",
        width=64,
        height=64,
        channels=4,
        gpu_ptr=0x1000,
        cuda_size=64 * 64 * 4 * 4,
    )
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = TDSenderEngine(
        host=host,
        config=TDSenderConfig(activation_barrier=False, export_sync=export_sync),
        cuda=None,
        log_fn=lambda *a, **k: None,
        num_slots=_NUM_SLOTS,
        device=0,
        shm_name=f"test_hold_{uuid.uuid4().hex[:8]}",
        verbose=False,
    )
    assert engine.export_frame() is False  # lazy-init probe opens the fake Exporter
    return engine, top


def _alive(top: _OwnerTOP) -> list[bool]:
    return [r() is not None for r in top.owner_refs]


def test_blocking_export_holds_nothing_after_the_call(fake_exporter_open: None) -> None:
    engine, top = _setup(export_sync=None)

    assert engine.export_frame() is True
    assert engine.export_frame() is True

    assert len(engine._held_sources) == 0
    assert not any(_alive(top)), "Blocking export: each owner is dropped when export_frame() returns."
    assert engine._exporter.drain_calls == 0


def test_async_export_holds_the_owner_while_the_copy_is_queued(fake_exporter_open: None) -> None:
    engine, top = _setup(export_sync=False)
    probe_owners = len(top.owner_refs)

    assert engine.export_frame() is True  # frame in slot 0, copy pending

    assert [slot for slot, _ in engine._held_sources] == [0]
    assert _alive(top)[probe_owners:] == [True], "The owner must live while its copy is queued."


def test_async_export_releases_the_owner_when_the_copy_is_done(fake_exporter_open: None) -> None:
    engine, top = _setup(export_sync=False)
    first = len(top.owner_refs)
    assert engine.export_frame() is True  # slot 0 pending

    engine._exporter.pending.discard(0)  # the copy of slot 0 completes
    assert engine.export_frame() is True  # slot 1 pending

    assert [slot for slot, _ in engine._held_sources] == [1]
    assert _alive(top)[first:] == [False, True]
    assert engine._exporter.drain_calls == 0


def test_async_release_is_in_order(fake_exporter_open: None) -> None:
    """A done slot behind a slot that is not done stays held: the copies run in order."""
    engine, _top = _setup(export_sync=False)
    assert engine.export_frame() is True  # slot 0
    assert engine.export_frame() is True  # slot 1

    engine._exporter.pending.discard(1)  # only the later one reports done
    engine._release_done_sources()

    assert [slot for slot, _ in engine._held_sources] == [0, 1]


def test_async_export_drains_at_the_cap(fake_exporter_open: None) -> None:
    """With no copy ever done, the hold never grows past one entry per ring slot."""
    engine, top = _setup(export_sync=False)
    first = len(top.owner_refs)

    for _ in range(_NUM_SLOTS):
        assert engine.export_frame() is True
    assert len(engine._held_sources) == _NUM_SLOTS
    assert engine._exporter.drain_calls == 0

    assert engine.export_frame() is True  # at the cap: wait for the stream, drop all, hold the new one

    assert engine._exporter.drain_calls == 1
    assert len(engine._held_sources) == 1
    assert _alive(top)[first:] == [False] * _NUM_SLOTS + [True]


def test_cleanup_closes_the_exporter_before_it_drops_the_owners(fake_exporter_open: None) -> None:
    engine, top = _setup(export_sync=False)
    assert engine.export_frame() is True
    exporter = engine._exporter
    order: list[str] = []
    exporter.close = lambda: order.append(f"close held={len(engine._held_sources)}")

    engine.cleanup()

    assert order == ["close held=1"], "close() waits for the queued copies; the owners go after it."
    assert len(engine._held_sources) == 0
    assert not any(_alive(top))


def test_reopen_drops_the_owners_of_the_closed_exporter(fake_exporter_open: None) -> None:
    engine, top = _setup(export_sync=False)
    assert engine.export_frame() is True
    old = engine._exporter

    top._pixel_format = "8-bit fixed (RGBA)"
    top._pixel_format_name = "rgba8fixed"
    assert engine.export_frame() is True  # reopens at uint8 and publishes

    assert old.closed
    assert engine._exporter is not old
    assert [slot for slot, _ in engine._held_sources] == [0], "Only the frame of the new Exporter is held."


def test_owner_is_alive_during_export(fake_exporter_open: None) -> None:
    """The CUDAMemoryRef of the frame holds the owner for the whole export() call."""
    engine, top = _setup(export_sync=None)
    exporter = engine._exporter
    real_export = exporter.export

    def export(frame):
        exporter._frame_owner_ref = top.owner_refs[-1]
        return real_export(frame)

    exporter.export = export
    assert engine.export_frame() is True

    assert exporter.owners_alive_in_export == [True]


def test_async_export_holds_the_owner_when_export_raises(fake_exporter_open: None) -> None:
    """export() can raise after it queued the copy (a sticky CUDA error): the block stays held."""
    engine, top = _setup(export_sync=False)
    first = len(top.owner_refs)

    def export(frame):
        raise RuntimeError("CUDA error after the copy was queued")

    engine._exporter.export = export
    with contextlib.suppress(RuntimeError):
        engine.export_frame()

    assert [slot for slot, _ in engine._held_sources] == [0]
    assert _alive(top)[first:] == [True]


def _check_raises(engine: TDSenderEngine) -> None:
    def boom(slot: int) -> bool:
        raise RuntimeError("CUDA error")

    engine._exporter.slot_copy_done = boom


def _error_status(engine: TDSenderEngine) -> list[str]:
    return [m for kind, m in engine._host.status_calls if kind == "error"]


def test_check_error_that_clears_costs_no_frame(fake_exporter_open: None) -> None:
    """The copy check raises, and recover() finds no error after its stream wait: that wait
    is the proof, so the blocks go and the same cook exports."""
    engine, top = _setup(export_sync=False)
    first = len(top.owner_refs)
    assert engine.export_frame() is True  # slot 0 held
    _check_raises(engine)

    assert engine.export_frame() is True

    assert engine._exporter.recover_calls == 1
    assert _alive(top)[first:] == [False, True]
    assert engine._hold_fault_code is None
    assert _error_status(engine) == []


def test_check_error_that_stays_stops_the_sender_with_the_owners_held(fake_exporter_open: None) -> None:
    """export_frame() must not raise into the cook callback: it logs and sets the error
    status one time, keeps the blocks held and returns False."""
    engine, top = _setup(export_sync=False)
    logs: list[str] = []
    engine._log = lambda msg, force=False: logs.append(msg)
    first = len(top.owner_refs)
    assert engine.export_frame() is True  # slot 0 held
    _check_raises(engine)
    engine._exporter.recover_code = 2  # out of memory: not permanent

    assert engine.export_frame() is False
    assert engine.export_frame() is False  # inside the retry interval: no CUDA call

    assert engine._exporter.recover_calls == 1
    assert [slot for slot, _ in engine._held_sources] == [0]
    assert _alive(top)[first:] == [True], "No new cudaMemory() call, and the held block stays."
    assert [m for m in logs if "Async copy fault" in m] == ["Async copy fault: CUDA error 2"]
    assert _error_status(engine) == ["Async copy fault: CUDA error 2 - recovery in progress"]


def test_stopped_sender_recovers_alone_when_the_error_clears(fake_exporter_open: None) -> None:
    engine, top = _setup(export_sync=False)
    first = len(top.owner_refs)
    assert engine.export_frame() is True
    _check_raises(engine)
    engine._exporter.recover_code = 2
    assert engine.export_frame() is False
    del engine._exporter.slot_copy_done  # the check works again

    engine._hold_next_try = 0.0  # the retry interval has passed
    assert engine.export_frame() is False, "The error is still latched: stay stopped."
    assert engine._exporter.recover_calls == 2

    engine._exporter.recover_code = 0  # memory is free again
    engine._hold_next_try = 0.0
    assert engine.export_frame() is True

    assert engine._hold_fault_code is None
    assert _alive(top)[first:] == [False, True]


def test_permanent_error_is_not_tried_again(fake_exporter_open: None) -> None:
    """719 is one of the codes that the CUDA header marks 'the process must be relaunched'."""
    engine, _top = _setup(export_sync=False)
    assert engine.export_frame() is True
    _check_raises(engine)
    engine._exporter.recover_code = 719

    assert engine.export_frame() is False
    assert engine._hold_next_try == float("inf")
    assert [engine.export_frame() for _ in range(5)] == [False] * 5

    assert engine._exporter.recover_calls == 1
    assert _error_status(engine) == ["Async copy fault: CUDA error 719 - restart TouchDesigner"]


def test_cleanup_without_proof_keeps_the_owners_for_the_process_life(fake_exporter_open: None) -> None:
    """Exporter.close() does not wait for the stream when a CUDA error is latched. An Active
    toggle must not free a block whose copy can still be queued."""
    engine, top = _setup(export_sync=False)
    first = len(top.owner_refs)
    assert engine.export_frame() is True  # slot 0, copy queued
    exporter = engine._exporter
    exporter.recover_code = 719

    engine.cleanup()

    assert exporter.closed and 0 in exporter.pending, "Precondition: the copy did not complete."
    assert _alive(top)[first:] == [True]
    assert len(_kept()) == 1
    assert len(engine._held_sources) == 0 and engine._hold_fault_code is None


def test_cleanup_with_proof_frees_the_owners(fake_exporter_open: None) -> None:
    """With no error after the stream wait of recover(), nothing is kept."""
    engine, top = _setup(export_sync=False)
    assert engine.export_frame() is True
    exporter = engine._exporter

    engine.cleanup()

    assert exporter.recover_calls == 1
    assert not any(_alive(top))
    assert _kept() == []


def test_reopen_without_proof_keeps_the_owners(fake_exporter_open: None) -> None:
    """The same rule on a geometry change."""
    engine, top = _setup(export_sync=False)
    first = len(top.owner_refs)
    assert engine.export_frame() is True  # slot 0, copy queued
    old = engine._exporter
    engine._release_done_sources = lambda: None  # go straight to the reopen
    old.recover_code = 719

    top._pixel_format = "8-bit fixed (RGBA)"
    top._pixel_format_name = "rgba8fixed"
    assert engine.export_frame() is True  # the new Exporter has no error

    assert old.closed and 0 in old.pending
    assert _alive(top)[first] is True
    assert len(_kept()) == 1


def test_blocking_cleanup_makes_no_recover_call(fake_exporter_open: None) -> None:
    engine, _top = _setup(export_sync=None)
    assert engine.export_frame() is True
    exporter = engine._exporter

    engine.cleanup()

    assert exporter.recover_calls == 0


def test_async_failed_export_recovers_before_the_release(fake_exporter_open: None) -> None:
    """export() can return FAILED after it queued the copy and before it recorded the slot's
    event. The slot's old event then says done, so the Sender must wait for the stream. It
    must also consume the error that export() left latched."""
    engine, _top = _setup(export_sync=False)
    real_export = engine._exporter.export
    engine._exporter.export = lambda frame: FrameOutcome.FAILED

    assert engine.export_frame() is False

    assert engine._exporter.recover_calls == 1
    assert len(engine._held_sources) == 0
    assert engine._hold_fault_code is None

    engine._exporter.export = real_export
    assert engine.export_frame() is True, "The next cook exports: no stop after a recovered FAILED."


def test_async_failed_export_keeps_the_owner_when_the_error_stays(fake_exporter_open: None) -> None:
    engine, top = _setup(export_sync=False)
    first = len(top.owner_refs)
    engine._exporter.export = lambda frame: FrameOutcome.FAILED
    engine._exporter.recover_code = 719

    assert engine.export_frame() is False
    assert engine.export_frame() is False  # stopped: no new cudaMemory() call

    assert len(top.owner_refs) == first + 1
    assert _alive(top)[first:] == [True], "The block stays held: its copy may still be queued."
    assert engine._hold_fault_code == 719


def test_blocking_failed_export_consumes_the_latched_error(fake_exporter_open: None) -> None:
    """export() only reads the latched CUDA error, so it fails on each later frame until the
    error is consumed. The Sender consumes it after a FAILED export in blocking mode too."""
    engine, _top = _setup(export_sync=None)
    real_export = engine._exporter.export
    engine._exporter.export = lambda frame: FrameOutcome.FAILED

    assert engine.export_frame() is False

    assert engine._exporter.recover_calls == 1
    assert engine._exporter.drain_calls == 0
    assert engine._hold_fault_code is None
    assert _error_status(engine) == []

    engine._exporter.export = real_export
    assert engine.export_frame() is True, "The next cook exports: no stop after a recovered FAILED."


def test_blocking_error_that_stays_stops_then_recovers_alone(fake_exporter_open: None) -> None:
    engine, top = _setup(export_sync=None)
    real_export = engine._exporter.export
    engine._exporter.export = lambda frame: FrameOutcome.FAILED
    engine._exporter.recover_code = 2

    assert engine.export_frame() is False
    calls = len(top.owner_refs)
    assert engine.export_frame() is False  # stopped, and the next attempt is one second away

    assert len(top.owner_refs) == calls, "A stopped Sender makes no cudaMemory() call."
    assert engine._exporter.recover_calls == 1
    assert _error_status(engine)[-1] == "Export fault: CUDA error 2 - recovery in progress"
    assert len(engine._held_sources) == 0, "Blocking mode holds no block."

    engine._exporter.export = real_export
    engine._exporter.recover_code = 0
    engine._hold_next_try = 0.0
    assert engine.export_frame() is True
    assert engine._hold_fault_code is None


def test_blocking_permanent_error_is_not_tried_again(fake_exporter_open: None) -> None:
    engine, _top = _setup(export_sync=None)
    engine._exporter.export = lambda frame: FrameOutcome.FAILED
    engine._exporter.recover_code = 719

    assert [engine.export_frame() for _ in range(4)] == [False] * 4

    assert engine._exporter.recover_calls == 1
    assert _error_status(engine)[-1] == "Export fault: CUDA error 719 - restart TouchDesigner"


def test_blocking_error_that_stays_restarts_the_stream(fake_exporter_open: None) -> None:
    engine, _top = _setup(export_sync=None)
    old = engine._exporter
    old.export = lambda frame: FrameOutcome.FAILED
    old.recover_code = 400

    for _ in range(TDSender._HOLD_RESTART_AFTER):
        engine._hold_next_try = 0.0
        assert engine.export_frame() is False

    assert old.closed and engine._exporter is None
    assert old.recover_calls == TDSender._HOLD_RESTART_AFTER, "No held block: the close needs no proof."
    assert _kept() == []

    assert engine.export_frame() is False  # lazy init opens the new Exporter
    assert engine._exporter is not None and engine._exporter is not old
    assert engine.export_frame() is True
    assert engine._stream_restarted is False


def _fault_until_restart(engine: TDSenderEngine, code: int = 400) -> _FakeExporter:
    """Stop the Sender on an error that stays, then run attempts until the stream restart."""
    assert engine.export_frame() is True  # slot 0 held
    old = engine._exporter
    _check_raises(engine)
    old.recover_code = code
    for _ in range(TDSender._HOLD_RESTART_AFTER):
        engine._hold_next_try = 0.0
        assert engine.export_frame() is False
    return old


def test_error_that_stays_restarts_the_stream_after_the_attempt_limit(fake_exporter_open: None) -> None:
    """400 (a handle that is not valid) never clears on the same stream. A new stream fixes it."""
    engine, top = _setup(export_sync=False)
    first = len(top.owner_refs)

    old = _fault_until_restart(engine)

    assert old.closed and engine._exporter is None
    assert old.recover_calls == TDSender._HOLD_RESTART_AFTER + 1, "Each attempt, and one before the close."
    assert _alive(top)[first] is True and len(_kept()) == 1, "No proof: the old block is kept."
    assert engine._stream_restarted is True

    assert engine.export_frame() is False  # lazy init opens the new Exporter
    assert engine._exporter is not None and engine._exporter is not old
    assert engine.export_frame() is True
    assert engine._stream_restarted is False
    assert engine._hold_fault_code is None


def test_one_stream_restart_per_fault(fake_exporter_open: None) -> None:
    """The new stream gets the fault before it publishes a frame: no second restart."""
    engine, _top = _setup(export_sync=False)
    _fault_until_restart(engine)
    assert engine.export_frame() is False  # lazy init
    new = engine._exporter
    new.export = lambda frame: FrameOutcome.FAILED
    new.recover_code = 400

    for _ in range(TDSender._HOLD_RESTART_AFTER + 3):
        engine._hold_next_try = 0.0
        assert engine.export_frame() is False

    assert engine._exporter is new and not new.closed
    assert _error_status(engine)[-1] == "Async copy fault: CUDA error 400 - toggle Active or restart TouchDesigner"


def test_stream_restart_that_cannot_open_stops_until_cleanup(
    fake_exporter_open: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _top = _setup(export_sync=False)
    _fault_until_restart(engine)
    opens: list[int] = []

    def failing_open(spec, *, policy=None, cuda=None, barrier=None):
        opens.append(1)
        raise RuntimeError("CUDA error 400")

    monkeypatch.setattr(TDSender.Exporter, "open", failing_open)
    monkeypatch.setattr(TDSender.traceback, "print_exc", lambda: None)

    assert [engine.export_frame() for _ in range(4)] == [False] * 4

    assert len(opens) == 1, "One open attempt, not one per cook."
    assert _error_status(engine)[-1] == "Stream restart failed - toggle Active or restart TouchDesigner"

    engine.cleanup()
    assert engine._restart_init_failed is False and engine._stream_restarted is False
