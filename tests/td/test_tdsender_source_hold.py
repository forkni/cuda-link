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
import types
import uuid
import weakref

from Exporter import FrameOutcome
from fakes import FakeTDHost, FakeTOPHandle, fake_exporter_open  # noqa: F401
from SHMProtocol import DtypeCodec
from TDConfig import TDSenderConfig
from TDSender import TDSenderEngine

_NUM_SLOTS = 3


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

    def close(self) -> None:
        self.closed = True
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
