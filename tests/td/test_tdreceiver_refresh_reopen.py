"""
Regression tests: TDReceiverEngine._refresh_on_version_change on a genuine sender
re-init (new IPC handles in SHM).

- New handles are opened before the old ones are closed, so a mid-loop open failure
  leaves the existing connection intact (no double-close in the caller's cleanup())
  and closes only the handles this refresh opened.
- The float16 CuPy views wrap the slot dev_ptrs, so they are rebuilt over the new
  dev_ptrs after a handle change.
- A flags-only change (RG -> Mono+Alpha) requests a Script TOP format update, and
  the frame cursor is reset so the restarted sender's first frame is not skipped.
"""

from __future__ import annotations

import contextlib
import uuid
from ctypes import c_void_p
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import MagicMock

import pytest
from td.test_tdreceiver_dtype_refresh import _make_receiver

from cuda_link.shm_protocol import (
    FLAGS_MONO_ALPHA,
    FORMAT_KIND_FLOAT,
    SHM_HEADER_SIZE,
    SLOT_SIZE,
    Metadata,
    SHMLayout,
)

NUM_SLOTS = 3


class _RefreshCuda:
    """Minimal CUDA double: tracks open IPC imports; optional failure on one slot open."""

    def __init__(self, fail_on_open_call: int | None = None) -> None:
        self.open_ptrs: set[int] = set()
        self.closed_ptrs: list[int] = []
        self.destroyed_events: list[object] = []
        self._next_ptr = 0x10000
        self._open_calls = 0
        self._fail_on_open_call = fail_on_open_call

    def adopt(self, ptr: int) -> c_void_p:
        self.open_ptrs.add(ptr)
        return c_void_p(ptr)

    def ipc_open_mem_handle(self, handle, flags: int = 1) -> c_void_p:
        call = self._open_calls
        self._open_calls += 1
        if call == self._fail_on_open_call:
            raise RuntimeError("injected ipc_open_mem_handle failure")
        self._next_ptr += 0x1000
        return self.adopt(self._next_ptr)

    def ipc_close_mem_handle(self, dev_ptr: c_void_p) -> None:
        self.open_ptrs.discard(dev_ptr.value)
        self.closed_ptrs.append(dev_ptr.value)

    def ipc_open_event_handle(self, handle) -> object:
        return object()

    def destroy_event(self, event) -> None:
        self.destroyed_events.append(event)

    def malloc_host(self, size: int) -> c_void_p:
        raise RuntimeError("no pinned memory in tests")  # exercise the pageable path

    def free_host(self, ptr) -> None:
        pass

    def host_register(self, ptr: int, size: int, flags: int = 0) -> None:
        pass

    def host_unregister(self, ptr: int) -> None:
        pass


@pytest.fixture
def shm():
    layout = SHMLayout(num_slots=NUM_SLOTS)
    handle = SharedMemory(create=True, size=layout.total_size)
    yield handle
    handle.close()
    with contextlib.suppress(FileNotFoundError):
        handle.unlink()


def _handle_bytes(tag: int) -> bytes:
    return bytes([tag]) * 64


def _write_shm(shm: SharedMemory, *, version: int, slot_tag: int, md: Metadata) -> None:
    layout = SHMLayout(num_slots=NUM_SLOTS)
    shm.buf[: layout.total_size] = layout.build_buffer(version=version, write_idx=0)
    for slot in range(NUM_SLOTS):
        base = SHM_HEADER_SIZE + slot * SLOT_SIZE
        shm.buf[base : base + 64] = _handle_bytes(slot_tag + slot)
    md.pack_into(memoryview(shm.buf), layout)


def _metadata(width: int, height: int, comps: int, bits: int, flags: int = 0) -> Metadata:
    return Metadata(
        width=width,
        height=height,
        num_comps=comps,
        format_kind=FORMAT_KIND_FLOAT,
        bits_per_comp=bits,
        flags=flags,
        data_size=width * height * comps * bits // 8,
    )


def _connect(engine, shm: SharedMemory, cuda: _RefreshCuda, md: Metadata, *, old_tag: int, last_write_idx: int):
    """Simulate a post-initialize_receiver connection whose slots hold handles old_tag+slot."""
    from CUDARuntimeTypes import cudaIpcMemHandle_t
    from TDReceiver import ReceiverConnection

    from cuda_link.importer import Format

    layout = SHMLayout(num_slots=NUM_SLOTS)
    engine.cuda = cuda
    engine._connection = ReceiverConnection(
        shm_handle=shm,
        dev_ptrs=[cuda.adopt(0x100 + slot) for slot in range(NUM_SLOTS)],
        ipc_handles=[cudaIpcMemHandle_t.from_buffer_copy(_handle_bytes(old_tag + s)) for s in range(NUM_SLOTS)],
        ipc_events=[None] * NUM_SLOTS,
        stream=None,
        layout=layout,
        num_slots=NUM_SLOTS,
        ipc_version=1,
        shutdown_offset=layout.shutdown_offset,
        last_write_idx=last_write_idx,
    )
    engine._format = Format.from_metadata(md)
    engine._initialized = True
    return [p.value for p in engine._connection.dev_ptrs]


def test_refresh_open_failure_leaves_old_handles_and_closes_new_ones(shm: SharedMemory) -> None:
    """Slot 2 of 3 fails to open: slots 0-1 new imports are closed, the old conn is untouched."""
    md = _metadata(64, 32, 4, 32)
    cuda = _RefreshCuda(fail_on_open_call=2)
    engine = _make_receiver(f"test_rro_{uuid.uuid4().hex[:8]}")
    old_ptrs = _connect(engine, shm, cuda, md, old_tag=0x10, last_write_idx=7)
    old_handles = [bytes(h) for h in engine._connection.ipc_handles]

    _write_shm(shm, version=2, slot_tag=0x20, md=md)

    assert engine._refresh_on_version_change(2) is False

    # The two new imports were released; none of the old ones were.
    assert len(cuda.closed_ptrs) == 2
    assert not set(cuda.closed_ptrs) & set(old_ptrs)
    assert cuda.open_ptrs == set(old_ptrs)
    # conn still owns exactly the old resources, so cleanup() closes each once.
    conn = engine._connection
    assert [p.value for p in conn.dev_ptrs] == old_ptrs
    assert [bytes(h) for h in conn.ipc_handles] == old_handles
    assert conn.ipc_version == 1

    conn.close(cuda, lambda *a, **k: None)
    assert sorted(cuda.closed_ptrs[2:]) == sorted(old_ptrs)
    assert cuda.open_ptrs == set()


def test_refresh_handle_change_rebuilds_f16_views_on_new_dev_ptrs(
    shm: SharedMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a handle reopen, the CuPy float16 views wrap the new dev_ptrs, not the closed ones."""
    import TDReceiver

    fake_cp = MagicMock()
    monkeypatch.setattr(TDReceiver, "cp", fake_cp)
    monkeypatch.setattr(TDReceiver, "CUPY_AVAILABLE", True)

    md = _metadata(64, 32, 4, 16)
    cuda = _RefreshCuda()
    engine = _make_receiver(f"test_rro_{uuid.uuid4().hex[:8]}")
    old_ptrs = _connect(engine, shm, cuda, md, old_tag=0x10, last_write_idx=7)
    engine._build_f16_staging(64, 32, 4, NUM_SLOTS, engine._connection.dev_ptrs, md.data_size)
    assert [c.args[0] for c in fake_cp.cuda.UnownedMemory.call_args_list] == old_ptrs
    fake_cp.cuda.UnownedMemory.reset_mock()

    _write_shm(shm, version=2, slot_tag=0x20, md=md)

    assert engine._refresh_on_version_change(2) is True

    new_ptrs = [p.value for p in engine._connection.dev_ptrs]
    assert set(new_ptrs).isdisjoint(old_ptrs)
    assert sorted(cuda.closed_ptrs) == sorted(old_ptrs)
    assert [c.args[0] for c in fake_cp.cuda.UnownedMemory.call_args_list] == new_ptrs
    assert len(engine._cupy_f16_views) == NUM_SLOTS
    assert engine._f16_cpu_buf is not None and engine._f16_cpu_buf.size == 64 * 32 * 4


def test_refresh_flags_only_change_requests_format_update_and_resets_cursor(shm: SharedMemory) -> None:
    """RG32F -> Mono+Alpha32F keeps kind/bits/comps but must still update the TOP format."""
    md_rg = _metadata(64, 32, 2, 32)
    md_ma = _metadata(64, 32, 2, 32, flags=FLAGS_MONO_ALPHA)
    cuda = _RefreshCuda()
    engine = _make_receiver(f"test_rro_{uuid.uuid4().hex[:8]}")
    _connect(engine, shm, cuda, md_rg, old_tag=0x10, last_write_idx=7)
    engine._retry.needs_format_update = False

    # Same handles (metadata-only bump), new flags.
    _write_shm(shm, version=2, slot_tag=0x10, md=md_ma)

    assert engine._refresh_on_version_change(2) is True
    assert cuda.closed_ptrs == []  # handles unchanged: nothing reopened
    assert engine._retry.needs_format_update is True
    assert engine._format.flags == FLAGS_MONO_ALPHA
    assert engine._connection.last_write_idx == 0
    assert engine._connection.ipc_version == 2


class _EventOpenFailCuda(_RefreshCuda):
    """_RefreshCuda whose ipc_open_event_handle raises on one call."""

    def __init__(self, fail_event_call: int) -> None:
        super().__init__()
        self._fail_event_call = fail_event_call
        self._event_calls = 0
        self.destroyed_streams: list[object] = []

    def ipc_open_event_handle(self, handle) -> object:
        call = self._event_calls
        self._event_calls += 1
        if call == self._fail_event_call:
            raise RuntimeError("injected ipc_open_event_handle failure")
        return object()

    def get_device(self) -> int:
        return 0

    def create_stream_with_priority(self, flags: int = 0) -> c_void_p:
        return c_void_p(0x5000)

    def destroy_stream(self, stream) -> None:
        self.destroyed_streams.append(stream)


def _write_event_handles(shm: SharedMemory, tag: int) -> None:
    for slot in range(NUM_SLOTS):
        base = SHM_HEADER_SIZE + slot * SLOT_SIZE
        shm.buf[base + 64 : base + 128] = _handle_bytes(tag + slot)


def test_refresh_event_open_failure_returns_none_and_closes_new_handles(shm: SharedMemory) -> None:
    """Slot 1's event fails to open: the refresh is abandoned, closing what it opened, old conn untouched."""
    md = _metadata(64, 32, 4, 32)
    cuda = _EventOpenFailCuda(fail_event_call=1)
    engine = _make_receiver(f"test_rro_{uuid.uuid4().hex[:8]}")
    old_ptrs = _connect(engine, shm, cuda, md, old_tag=0x10, last_write_idx=7)
    _write_shm(shm, version=2, slot_tag=0x20, md=md)
    _write_event_handles(shm, 0x40)

    assert engine._open_refresh_handles(shm, NUM_SLOTS) is None

    # Slots 0 and 1 had their mem handles opened (slot 1 before its event failed): both released.
    assert len(cuda.closed_ptrs) == 2
    assert not set(cuda.closed_ptrs) & set(old_ptrs)
    assert len(cuda.destroyed_events) == 1  # slot 0's event
    assert cuda.open_ptrs == set(old_ptrs)


def test_initialize_receiver_event_open_failure_fails_connect_and_releases_handles(
    shm: SharedMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero event handle that won't open must not silently connect without ordering."""
    import TDReceiver

    md = _metadata(64, 32, 4, 32)
    cuda = _EventOpenFailCuda(fail_event_call=1)
    monkeypatch.setattr(TDReceiver, "get_cuda_runtime", lambda device=0: cuda)
    _write_shm(shm, version=1, slot_tag=0x20, md=md)
    _write_event_handles(shm, 0x40)
    engine = _make_receiver(shm.name)

    assert engine.initialize_receiver() is False

    assert engine._initialized is False
    # Slots 0 and 1 mem handles (slot 1's included) closed; slot 0's event destroyed; stream freed.
    assert len(cuda.closed_ptrs) == 2
    assert cuda.open_ptrs == set()
    assert len(cuda.destroyed_events) == 1
    assert len(cuda.destroyed_streams) == 1
