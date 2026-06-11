"""
CPU-only producer→consumer round-trip tests over real SharedMemory.

These close the gap between the unit suites (which pin the wire protocol
against synthetic bytearrays) and the requires_cuda roundtrip tests (which
never run in CI): a real Exporter writes a real SharedMemory segment via
FakeCUDAAdapter, and the consumer side — both raw shm_protocol readers and
the real Importer — parses that same segment.

Three sections:
  A. Wire-format round-trip — exporter writes cross-checked byte-for-byte
     against independent consumer-side reads.
  B. Producer restart / version change — VERSION_CHANGED, SHUTDOWN, and the
     importer reconnect state machine.
  C. Ring-buffer slot wrap — write_idx monotonicity and memcpy destination
     cycling across the wrap boundary.

No test here requires a GPU; all run in CI.
"""

from __future__ import annotations

import dataclasses
import time
from ctypes import c_void_p
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import pytest

from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link._exporter_port import ExportPolicy, FrameSpec, GpuFrame
from cuda_link._importer_port import ImportOutcome, ImportPolicy, ImportSpec
from cuda_link.exporter import Exporter, FrameOutcome
from cuda_link.importer import Importer
from cuda_link.shm_protocol import (
    IPC_HANDLE_SIZE,
    PROTOCOL_MAGIC,
    SLOT_SIZE,
    DtypeCodec,
    Metadata,
    SHMLayout,
    SlotState,
    acquire_slot,
    read_magic,
    read_num_slots,
    read_version,
    read_write_idx,
)


@pytest.fixture(autouse=True)
def _no_gpu_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the importer's CPU-only path.

    Importer._connect eagerly builds torch buffers with device="cuda" when
    torch is importable; CI installs CPU-only torch, so that would raise.
    """
    monkeypatch.setattr("cuda_link.importer.TORCH_AVAILABLE", False)
    monkeypatch.setattr("cuda_link.importer.CUPY_AVAILABLE", False)


def _open_exporter(
    shm_name: str,
    *,
    num_slots: int = 3,
    height: int = 8,
    width: int = 8,
    dtype: str = "uint8",
    fake: FakeCUDAAdapter | None = None,
) -> tuple[Exporter, FakeCUDAAdapter]:
    fake = fake or FakeCUDAAdapter()
    spec = FrameSpec(shm_name=shm_name, height=height, width=width, dtype=dtype, num_slots=num_slots)
    exp = Exporter.open(spec, policy=ExportPolicy.for_testing(), cuda=fake)
    return exp, fake


def _open_importer(shm_name: str, *, reconnect: bool = False) -> tuple[Importer, FakeCUDAAdapter]:
    fake = FakeCUDAAdapter()
    policy = ImportPolicy.for_testing()
    if reconnect:
        policy = dataclasses.replace(policy, reconnect_enabled=True)
    imp = Importer.open(ImportSpec(shm_name=shm_name), policy=policy, cuda=fake)
    return imp, fake


def _export_one(exp: Exporter) -> None:
    # producer_stream=0 arms GPU-side ordering, suppressing the
    # unordered-export warning path.
    frame = GpuFrame(ptr=0x5000, size=exp.data_size, producer_stream=0)
    assert exp.export(frame) == FrameOutcome.PUBLISHED


class _RecordingFake(FakeCUDAAdapter):
    """FakeCUDAAdapter that records memcpy_async calls.

    FakeCUDAAdapter itself must not grow this (it is sync-pinned to
    td_exporter/CUDAAdapters.py), so the recorder lives here.
    """

    def __init__(self) -> None:
        super().__init__()
        self.memcpy_calls: list[tuple[int, int, int, int]] = []  # (dst, src, count, kind)

    def memcpy_async(self, dst: c_void_p, src: Any, count: int, kind: int, stream: Any) -> None:
        dst_int = dst.value if isinstance(dst, c_void_p) else int(dst)
        src_int = src.value if isinstance(src, c_void_p) else int(src)
        self.memcpy_calls.append((dst_int or 0, src_int or 0, count, kind))


# ---------------------------------------------------------------------------
# Section A — wire-format round-trip
# ---------------------------------------------------------------------------


def test_wire_header_after_open(temp_shm_name: str) -> None:
    """A fresh Exporter writes the exact 20-byte header consumers expect."""
    exp, _ = _open_exporter(temp_shm_name, num_slots=3)
    try:
        observer = SharedMemory(name=temp_shm_name)
        try:
            layout = SHMLayout(3)
            assert read_magic(observer.buf) == PROTOCOL_MAGIC
            assert read_version(observer.buf) == 1
            assert read_num_slots(observer.buf) == 3
            assert read_write_idx(observer.buf) == 0
            assert observer.buf[layout.shutdown_offset] == 0
            assert observer.size >= layout.total_size
        finally:
            observer.close()
    finally:
        exp.close()


def test_wire_metadata_round_trip(temp_shm_name: str) -> None:
    """Metadata written by the exporter decodes to the original FrameSpec."""
    exp, _ = _open_exporter(temp_shm_name, height=12, width=10, dtype="float32")
    try:
        observer = SharedMemory(name=temp_shm_name)
        try:
            md = Metadata.read_from(observer.buf, SHMLayout(3))
            assert (md.width, md.height, md.num_comps) == (10, 12, 4)
            assert (md.format_kind, md.bits_per_comp, md.flags) == DtypeCodec.encode("float32")
            assert md.data_size == exp.data_size
            assert md.expected_size == md.data_size
        finally:
            observer.close()
    finally:
        exp.close()


def test_acquire_slot_sees_published_frame(temp_shm_name: str) -> None:
    """A consumer-side acquire_slot observes exactly what export() published."""
    exp, _ = _open_exporter(temp_shm_name)
    try:
        observer = SharedMemory(name=temp_shm_name)
        try:
            t_before = time.perf_counter()
            _export_one(exp)
            t_after = time.perf_counter()

            result = acquire_slot(observer.buf, SHMLayout(3), last_write_idx=0, last_version=1)
            assert result.state == SlotState.NEW_FRAME
            assert result.slot == 0
            assert result.write_idx == 1
            assert t_before <= result.timestamp <= t_after
        finally:
            observer.close()
    finally:
        exp.close()


def test_ipc_handle_bytes_cross_process_offsets(temp_shm_name: str) -> None:
    """The 64-byte IPC mem handles land at the consumer's expected offsets,
    byte-for-byte, and the real Importer forwards exactly those bytes into
    ipc_open_mem_handle."""

    class _PatternedFake(FakeCUDAAdapter):
        def __init__(self) -> None:
            super().__init__()
            self._handle_count = 0

        def ipc_get_mem_handle(self, dev_ptr: Any) -> Any:
            self._handle_count += 1
            handle = super().ipc_get_mem_handle(dev_ptr)
            handle.internal = bytes([self._handle_count] * IPC_HANDLE_SIZE)
            return handle

    exp, _ = _open_exporter(temp_shm_name, num_slots=3, fake=_PatternedFake())
    try:
        layout = SHMLayout(3)
        observer = SharedMemory(name=temp_shm_name)
        try:
            for slot in range(3):
                off = layout.mem_handle_offset(slot)
                expected = bytes([slot + 1] * IPC_HANDLE_SIZE)
                assert bytes(observer.buf[off : off + IPC_HANDLE_SIZE]) == expected
        finally:
            observer.close()

        imp, imp_fake = _open_importer(temp_shm_name)
        try:
            forwarded = sorted(bytes(h.internal) for h in imp_fake.opened_mem_handles.values())
            assert forwarded == [bytes([slot + 1] * IPC_HANDLE_SIZE) for slot in range(3)]
        finally:
            imp.close()
    finally:
        exp.close()


def test_real_importer_full_path_no_gpu(temp_shm_name: str) -> None:
    """Full path: Exporter.export → SHM → Importer auto-detect → get_frame_numpy."""
    exp, _ = _open_exporter(temp_shm_name, height=8, width=8, dtype="uint8")
    try:
        imp, _ = _open_importer(temp_shm_name)
        try:
            assert imp.is_ready()
            stats = imp.get_stats()
            assert stats["shape"] == (8, 8, 4)  # auto-detected from SHM metadata
            assert stats["dtype"] == "uint8"

            _export_one(exp)
            result = imp.get_frame_numpy()
            assert result.outcome == ImportOutcome.NEW_FRAME
            assert result.frame is not None
            assert result.frame.shape == (8, 8, 4)
            assert result.frame.dtype.name == "uint8"

            # No second export → nothing new to consume.
            assert imp.get_frame_numpy().outcome == ImportOutcome.NO_FRAME
        finally:
            imp.close()
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# Section B — producer restart / version change
# ---------------------------------------------------------------------------


def test_version_changed_when_producer_reopens_live_segment(temp_shm_name: str) -> None:
    """A second producer opening the same live segment bumps the version in
    place; a consumer holding last_version=1 sees VERSION_CHANGED."""
    exp_a, _ = _open_exporter(temp_shm_name)
    exp_b = None
    try:
        observer = SharedMemory(name=temp_shm_name)
        try:
            assert read_version(observer.buf) == 1

            exp_b, _ = _open_exporter(temp_shm_name)
            result = acquire_slot(observer.buf, SHMLayout(3), last_write_idx=0, last_version=1)
            assert result.state == SlotState.VERSION_CHANGED
            assert result.new_version == 2
        finally:
            observer.close()
    finally:
        if exp_b is not None:
            exp_b.close()
        exp_a.close()


def test_importer_reconnecting_on_version_change(temp_shm_name: str) -> None:
    """On a version bump the Importer reports RECONNECTING and reopens the
    IPC handles against the new producer."""
    exp_a, _ = _open_exporter(temp_shm_name)
    exp_b = None
    imp = None
    try:
        imp, _ = _open_importer(temp_shm_name)
        _export_one(exp_a)
        assert imp.get_frame_numpy().outcome == ImportOutcome.NEW_FRAME

        exp_b, _ = _open_exporter(temp_shm_name)  # version 1 → 2 in place
        result = imp.get_frame_numpy()
        assert result.outcome == ImportOutcome.RECONNECTING
        assert imp._conn.ipc_version == 2  # _reinitialize adopted the new producer

        # Characterization of current behavior: _reinitialize retains
        # _last_write_idx (=1), so producer B's first frame (write_idx=1)
        # collides with it and is silently missed; the second frame lands.
        _export_one(exp_b)
        assert imp.get_frame_numpy().outcome == ImportOutcome.NO_FRAME
        _export_one(exp_b)
        assert imp.get_frame_numpy().outcome == ImportOutcome.NEW_FRAME
    finally:
        if imp is not None:
            imp.close()
        if exp_b is not None:
            exp_b.close()
        exp_a.close()


def test_close_sets_shutdown_zeroes_handles_and_unlinks(temp_shm_name: str) -> None:
    """close() signals shutdown, scrubs the slot handle bytes, and unlinks."""
    exp, _ = _open_exporter(temp_shm_name, num_slots=3)
    layout = SHMLayout(3)
    observer = SharedMemory(name=temp_shm_name)
    try:
        exp.close()

        assert observer.buf[layout.shutdown_offset] == 1
        slots_start = layout.slot_offset(0)
        slots_end = slots_start + 3 * SLOT_SIZE
        assert bytes(observer.buf[slots_start:slots_end]) == bytes(3 * SLOT_SIZE)

        with pytest.raises(FileNotFoundError):
            SharedMemory(name=temp_shm_name)
    finally:
        observer.close()


def test_importer_full_restart_cycle_shutdown_then_reconnect(temp_shm_name: str) -> None:
    """SHUTDOWN → RECONNECTING → reattach to the restarted producer's fresh
    segment → NEW_FRAME, via the importer's retry state machine."""
    exp_a, _ = _open_exporter(temp_shm_name)
    exp_b = None
    imp = None
    try:
        imp, _ = _open_importer(temp_shm_name, reconnect=True)
        _export_one(exp_a)
        assert imp.get_frame_numpy().outcome == ImportOutcome.NEW_FRAME

        exp_a.close()  # sets shutdown flag, then unlinks the segment
        assert imp.get_frame_numpy().outcome == ImportOutcome.SHUTDOWN

        # Producer still down: retry machinery reports RECONNECTING.
        assert imp.get_frame_numpy().outcome == ImportOutcome.RECONNECTING

        exp_b, _ = _open_exporter(temp_shm_name)  # fresh segment, version 1
        # Retry backoff means reconnection may take a few frames; bound it.
        for _ in range(6):
            imp.get_frame_numpy()
            if imp._initialized:
                break
        assert imp._initialized

        _export_one(exp_b)
        assert imp.get_frame_numpy().outcome == ImportOutcome.NEW_FRAME
    finally:
        if imp is not None:
            imp.close()
        if exp_b is not None:
            exp_b.close()


# ---------------------------------------------------------------------------
# Section C — ring-buffer slot wrap
# ---------------------------------------------------------------------------


def test_ring_wrap_write_idx_monotonic_and_slot_mapping(temp_shm_name: str) -> None:
    """write_idx stays strictly monotonic across the wrap; the consumer's
    slot mapping is (write_idx - 1) % num_slots."""
    exp, _ = _open_exporter(temp_shm_name, num_slots=3)
    try:
        observer = SharedMemory(name=temp_shm_name)
        try:
            layout = SHMLayout(3)
            for k in range(8):  # 3 slots + 5 → wraps the ring twice
                _export_one(exp)
                assert read_write_idx(observer.buf) == k + 1
                result = acquire_slot(observer.buf, layout, last_write_idx=k, last_version=1)
                assert result.state == SlotState.NEW_FRAME
                assert result.slot == k % 3
        finally:
            observer.close()
    finally:
        exp.close()


def test_ring_wrap_memcpy_destinations_cycle(temp_shm_name: str) -> None:
    """Each export D2D-copies into the ring slot's own allocation, cycling
    0,1,2,0,1,2,... across the wrap boundary."""
    fake = _RecordingFake()
    exp, _ = _open_exporter(temp_shm_name, num_slots=3, fake=fake)
    try:
        for _ in range(8):
            _export_one(exp)

        assert len(fake.memcpy_calls) == 8
        for k, (dst, src, count, kind) in enumerate(fake.memcpy_calls):
            assert dst == exp.dev_ptrs[k % 3].value
            assert fake.allocations[dst] == exp.buffer_size
            assert src == 0x5000
            assert count == exp.data_size
            assert kind == 3  # cudaMemcpyDeviceToDevice
    finally:
        exp.close()
