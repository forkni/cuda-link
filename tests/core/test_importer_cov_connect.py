"""Coverage tests for Importer._open_and_validate_shm(), _resolve_format(), and
_open_ipc_slots() -- the SHM-attach / IPC-handle-open path exercised by _connect().

Builds real (small) multiprocessing.shared_memory.SharedMemory segments with
hand-packed headers via SHMLayout -- no real GPU or CUDA driver required, since
Importer is constructed directly (bypassing .open()) with a FakeCUDAAdapter.

IMPORTANT (Windows): a Windows named shared-memory segment is destroyed the
moment its *last* open handle is closed (there is no POSIX-style persistent
unlink semantics). So the "writer" handle created by _make_valid_shm() must be
kept alive (a live Python reference held, not closed) for as long as the
Importer's _open_and_validate_shm() (which opens its own second handle to the
same name) needs the segment to exist -- close it only after the SUT call
returns/raises, in a try/finally.

Uses tests/conftest.py's shared_memory_cleanup / temp_shm_name fixtures so no
segment leaks across the (randomly-ordered) test suite.
"""

from __future__ import annotations

import os
import struct
import uuid
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import MagicMock

import pytest

from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link._importer_port import ImportPolicy, ImportSpec
from cuda_link.cuda_runtime_types import CudaIpcError, ProtocolError
from cuda_link.importer import Format, Importer
from cuda_link.shm_protocol import NUM_SLOTS_OFFSET, Metadata, SHMLayout

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_importer(shm_name: str, *, shape=None, dtype=None, wait_backend: str = "python") -> Importer:
    spec = ImportSpec(shm_name=shm_name, device=0, shape=shape, dtype=dtype, timeout_ms=1000.0)
    policy = ImportPolicy(
        wait_spin_us=0,
        allow_pageable_fallback=True,
        debug=False,
        reconnect_enabled=False,
        wait_backend=wait_backend,
    )
    return Importer(spec, policy, FakeCUDAAdapter())


def _make_valid_shm(
    name: str,
    shared_memory_cleanup: list,
    *,
    num_slots: int = 1,
    version: int = 1,
    write_idx: int = 1,
    metadata: Metadata | None = None,
    nonzero_event_slots: tuple = (),
) -> tuple[SharedMemory, SHMLayout]:
    """Create and return an OPEN writer handle -- caller must keep it referenced
    and close it (in a finally) only after the SUT call completes."""
    layout = SHMLayout(num_slots)
    shm = SharedMemory(name=name, create=True, size=layout.total_size)
    shared_memory_cleanup.append(name)
    buf = layout.build_buffer(version=version, write_idx=write_idx)
    shm.buf[: len(buf)] = buf
    if metadata is not None:
        metadata.pack_into(shm.buf, layout)
    for slot in nonzero_event_slots:
        off = layout.event_handle_offset(slot)
        shm.buf[off] = 0xAB
    return shm, layout


def _extra_name() -> str:
    return f"test_cuda_ipc_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# _open_and_validate_shm()
# ---------------------------------------------------------------------------


def test_open_and_validate_shm_success(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    writer, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=2, version=3, write_idx=1)
    try:
        imp = _make_importer(temp_shm_name)
        shm, num_slots, ipc_version = imp._open_and_validate_shm()
        try:
            assert num_slots == 2
            assert ipc_version == 3
        finally:
            shm.close()
    finally:
        writer.close()


def test_open_and_validate_shm_raises_file_not_found_when_missing() -> None:
    imp = _make_importer(_extra_name())

    with pytest.raises(FileNotFoundError):
        imp._open_and_validate_shm()


def test_open_and_validate_shm_raises_on_truncated_magic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Buffer too short for the 4-byte magic read -> struct.error, shm.close() called, re-raised.

    Windows rounds SharedMemory segment allocations up to page granularity, so a
    real segment can't be made shorter than MAGIC_SIZE bytes -- a fake stand-in
    (swapped in for the module-level SharedMemory symbol) is used instead.
    """
    import cuda_link.importer as _imp_mod

    closed: list[bool] = []

    class _FakeShm:
        def __init__(self, name: str | None = None) -> None:
            self.buf = bytearray(2)  # shorter than MAGIC_SIZE=4

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(_imp_mod, "SharedMemory", _FakeShm)
    imp = _make_importer(_extra_name())

    with pytest.raises(struct.error):
        imp._open_and_validate_shm()

    assert closed == [True]


def test_open_and_validate_shm_raises_protocol_error_on_bad_magic(
    temp_shm_name: str, shared_memory_cleanup: list
) -> None:
    writer, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=1)
    try:
        struct.pack_into("<I", writer.buf, 0, 0xBADF00D)
        imp = _make_importer(temp_shm_name)
        with pytest.raises(ProtocolError, match="Protocol magic mismatch"):
            imp._open_and_validate_shm()
    finally:
        writer.close()


def test_open_and_validate_shm_raises_protocol_error_on_zero_num_slots(
    temp_shm_name: str, shared_memory_cleanup: list
) -> None:
    writer, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=1)
    try:
        struct.pack_into("<I", writer.buf, NUM_SLOTS_OFFSET, 0)
        imp = _make_importer(temp_shm_name)
        with pytest.raises(ProtocolError, match="Invalid num_slots"):
            imp._open_and_validate_shm()
    finally:
        writer.close()


def test_open_and_validate_shm_raises_protocol_error_on_too_many_slots(
    temp_shm_name: str, shared_memory_cleanup: list
) -> None:
    writer, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=1)
    try:
        struct.pack_into("<I", writer.buf, NUM_SLOTS_OFFSET, 11)
        imp = _make_importer(temp_shm_name)
        with pytest.raises(ProtocolError, match="Invalid num_slots"):
            imp._open_and_validate_shm()
    finally:
        writer.close()


def test_open_and_validate_shm_raises_cuda_ipc_error_when_segment_too_small_for_shutdown_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """num_slots=1 header claims shutdown_offset=148, but the buffer is only 30
    bytes -- indexing shm.buf[148] raises IndexError.

    Windows rounds real SharedMemory segment allocations up to page granularity,
    so a real 30-byte segment can't be made to stay that short -- a fake
    stand-in (swapped in for the module-level SharedMemory symbol) is used
    instead.
    """
    import cuda_link.importer as _imp_mod

    header = SHMLayout(1).build_buffer(version=1, write_idx=1)
    truncated = bytearray(header[:30])

    class _FakeShm:
        def __init__(self, name: str | None = None) -> None:
            self.buf = truncated

        def close(self) -> None:
            pass

    monkeypatch.setattr(_imp_mod, "SharedMemory", _FakeShm)
    imp = _make_importer(_extra_name())

    with pytest.raises(CudaIpcError, match="Could not read shutdown flag"):
        imp._open_and_validate_shm()


def test_open_and_validate_shm_raises_protocol_error_when_shutdown_flag_set(
    temp_shm_name: str, shared_memory_cleanup: list
) -> None:
    writer, layout = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=1)
    try:
        writer.buf[layout.shutdown_offset] = 1
        imp = _make_importer(temp_shm_name)
        with pytest.raises(ProtocolError, match="shutdown flag set"):
            imp._open_and_validate_shm()
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# _resolve_format()
# ---------------------------------------------------------------------------


def _valid_metadata(shape=(8, 8, 4), dtype="uint8") -> Metadata:
    from cuda_link.shm_protocol import DtypeCodec

    height, width, num_comps = shape
    kind, bits, flags = DtypeCodec.encode(dtype)
    itemsize = DtypeCodec.itemsize(dtype)
    return Metadata(
        width=width,
        height=height,
        num_comps=num_comps,
        format_kind=kind,
        bits_per_comp=bits,
        flags=flags,
        data_size=width * height * num_comps * itemsize,
    )


def test_resolve_format_auto_detects_shape_and_dtype_from_shm(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    md = _valid_metadata(shape=(8, 8, 4), dtype="uint8")
    shm, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=1, metadata=md)
    try:
        imp = _make_importer(temp_shm_name, shape=None, dtype=None)
        fmt, provisional = imp._resolve_format(shm, 1)
        assert provisional is False
        assert fmt.shape == (8, 8, 4)
        assert fmt.dtype_str == "uint8"
    finally:
        shm.close()


def test_resolve_format_overrides_shape_only_diverging_from_shm(
    temp_shm_name: str, shared_memory_cleanup: list
) -> None:
    """spec.shape given (differs from SHM), spec.dtype=None -- exercises the
    Format.from_overrides() branch (fmt != fmt_from_shm) plus the dtype-auto-detect log."""
    md = _valid_metadata(shape=(8, 8, 4), dtype="uint8")
    shm, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=1, metadata=md)
    try:
        imp = _make_importer(temp_shm_name, shape=(16, 16, 4), dtype=None)
        fmt, provisional = imp._resolve_format(shm, 1)
        assert provisional is False
        assert fmt.shape == (16, 16, 4)
        assert fmt.dtype_str == "uint8"
    finally:
        shm.close()


def test_resolve_format_overrides_dtype_only_diverging_from_shm(
    temp_shm_name: str, shared_memory_cleanup: list
) -> None:
    md = _valid_metadata(shape=(8, 8, 4), dtype="uint8")
    shm, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=1, metadata=md)
    try:
        imp = _make_importer(temp_shm_name, shape=None, dtype="int16")
        fmt, provisional = imp._resolve_format(shm, 1)
        assert provisional is False
        assert fmt.shape == (8, 8, 4)
        assert fmt.dtype_str == "int16"
    finally:
        shm.close()


def test_resolve_format_falls_back_when_no_metadata_present(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """SHM metadata block is all-zero -- Format.from_shm() returns None -> hard-coded fallback."""
    shm, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=1)
    try:
        imp = _make_importer(temp_shm_name, shape=None, dtype=None)
        fmt, provisional = imp._resolve_format(shm, 1)
        assert provisional is True
        assert fmt.shape == (512, 512, 4)
        assert fmt.dtype_str == "float32"
    finally:
        shm.close()


def test_resolve_format_explicit_overrides_skip_shm_entirely(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """Both spec.shape and spec.dtype given -- SHM is never read for metadata."""
    shm, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=1)
    try:
        imp = _make_importer(temp_shm_name, shape=(4, 4, 4), dtype="int8")
        fmt, provisional = imp._resolve_format(shm, 1)
        assert provisional is False
        assert fmt.shape == (4, 4, 4)
        assert fmt.dtype_str == "int8"
    finally:
        shm.close()


# ---------------------------------------------------------------------------
# _open_ipc_slots()
# ---------------------------------------------------------------------------


def test_open_ipc_slots_success_no_events(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    shm, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=2)
    imp = _make_importer(temp_shm_name, shape=(4, 4, 4), dtype="uint8", wait_backend="python")
    fmt = Format.from_overrides((4, 4, 4), "uint8")

    conn = imp._open_ipc_slots(shm, 2, 1, fmt)

    assert conn.num_slots == 2
    assert all(p is not None for p in conn.dev_ptrs)
    assert conn.ipc_events == [None, None]
    assert conn.doorbell_handle is None
    conn.close()


def test_open_ipc_slots_opens_event_handle_when_nonzero_bytes_present(
    temp_shm_name: str, shared_memory_cleanup: list
) -> None:
    shm, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=1, nonzero_event_slots=(0,))
    imp = _make_importer(temp_shm_name, shape=(4, 4, 4), dtype="uint8", wait_backend="python")
    fmt = Format.from_overrides((4, 4, 4), "uint8")

    conn = imp._open_ipc_slots(shm, 1, 1, fmt)

    assert conn.ipc_events[0] is not None
    conn.close()


def test_open_ipc_slots_swallows_event_open_failure(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    shm, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=1, nonzero_event_slots=(0,))
    imp = _make_importer(temp_shm_name, shape=(4, 4, 4), dtype="uint8", wait_backend="python")
    imp._cuda.ipc_open_event_handle = MagicMock(side_effect=RuntimeError("boom"))
    fmt = Format.from_overrides((4, 4, 4), "uint8")

    conn = imp._open_ipc_slots(shm, 1, 1, fmt)  # must not raise

    assert conn.ipc_events[0] is None
    conn.close()


def test_open_ipc_slots_doorbell_policy_true_not_found_logs_warning(
    temp_shm_name: str, shared_memory_cleanup: list
) -> None:
    """policy.doorbell=True but no producer-created event exists for this random
    shm name -- open_doorbell() returns None (real Win32 OpenEventW call, safe:
    ERROR_FILE_NOT_FOUND) -- exercises the "requested but not found" warning branch."""
    shm, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=1)
    spec = ImportSpec(shm_name=temp_shm_name, device=0, shape=(4, 4, 4), dtype="uint8", timeout_ms=1000.0)
    policy = ImportPolicy(
        wait_spin_us=0,
        allow_pageable_fallback=True,
        reconnect_enabled=False,
        wait_backend="python",  # want_native False -- isolates doorbell-only logic
        doorbell=True,
    )
    imp = Importer(spec, policy, FakeCUDAAdapter())
    fmt = Format.from_overrides((4, 4, 4), "uint8")

    conn = imp._open_ipc_slots(shm, 1, 1, fmt)

    assert conn.doorbell_handle is None
    conn.close()


def test_open_ipc_slots_native_resolution_failure_closes_speculative_doorbell(
    temp_shm_name: str, shared_memory_cleanup: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """want_native=True (wait_backend='auto'), policy.doorbell=False: doorbell is
    opened speculatively; when native resolution fails, the speculative handle
    must be closed again rather than leaked."""
    shm, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=1)
    monkeypatch.setattr(os, "name", "nt")
    import cuda_link.importer as _imp_mod

    monkeypatch.setattr(_imp_mod._doorbell, "open_doorbell", lambda name: 0xFEED)  # noqa: ARG005
    closed: list = []
    monkeypatch.setattr(_imp_mod._doorbell, "close", lambda h: closed.append(h))

    spec = ImportSpec(shm_name=temp_shm_name, device=0, shape=(4, 4, 4), dtype="uint8", timeout_ms=1000.0)
    policy = ImportPolicy(
        wait_spin_us=0,
        allow_pageable_fallback=True,
        reconnect_enabled=False,
        wait_backend="auto",
        doorbell=False,
    )
    imp = Importer(spec, policy, FakeCUDAAdapter())
    monkeypatch.setattr(imp, "_resolve_wait_backend", lambda: None)
    fmt = Format.from_overrides((4, 4, 4), "uint8")

    conn = imp._open_ipc_slots(shm, 1, 1, fmt)

    assert conn.doorbell_handle is None
    assert closed == [0xFEED]
    conn.close()


def test_open_ipc_slots_native_resolution_success_keeps_doorbell(
    temp_shm_name: str, shared_memory_cleanup: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    shm, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=1)
    monkeypatch.setattr(os, "name", "nt")
    import cuda_link.importer as _imp_mod

    monkeypatch.setattr(_imp_mod._doorbell, "open_doorbell", lambda name: 0xFEED)  # noqa: ARG005

    spec = ImportSpec(shm_name=temp_shm_name, device=0, shape=(4, 4, 4), dtype="uint8", timeout_ms=1000.0)
    policy = ImportPolicy(
        wait_spin_us=0,
        allow_pageable_fallback=True,
        reconnect_enabled=False,
        wait_backend="auto",
        doorbell=False,
    )
    imp = Importer(spec, policy, FakeCUDAAdapter())
    sentinel_backend = object()
    monkeypatch.setattr(imp, "_resolve_wait_backend", lambda: sentinel_backend)
    fmt = Format.from_overrides((4, 4, 4), "uint8")

    conn = imp._open_ipc_slots(shm, 1, 1, fmt)

    assert conn.doorbell_handle == 0xFEED
    assert imp._wait_backend is sentinel_backend
    conn.close()


def test_open_ipc_slots_rolls_back_partial_slots_on_exception(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """Slot 0 opens fine; slot 1's ipc_open_mem_handle raises -- the except
    BaseException block must close_ipc_handles() on the partial state and re-raise."""
    shm, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=2)
    imp = _make_importer(temp_shm_name, shape=(4, 4, 4), dtype="uint8", wait_backend="python")
    fmt = Format.from_overrides((4, 4, 4), "uint8")

    real_open = imp._cuda.ipc_open_mem_handle
    call_count = {"n": 0}

    def _flaky_open(handle, flags=1):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated open failure on slot 1")
        return real_open(handle, flags)

    imp._cuda.ipc_open_mem_handle = _flaky_open

    with pytest.raises(RuntimeError, match="simulated open failure"):
        imp._open_ipc_slots(shm, 2, 1, fmt)

    # Slot 0's mem handle must have been rolled back (no leaked GPU mapping).
    assert imp._cuda.opened_mem_handles == {}
    shm.close()


# ---------------------------------------------------------------------------
# _connect()
# ---------------------------------------------------------------------------


def test_connect_closes_shm_and_reraises_when_open_ipc_slots_fails(
    temp_shm_name: str, shared_memory_cleanup: list
) -> None:
    """_connect() itself (not _open_ipc_slots) must close the SharedMemory
    handle it opened via _open_and_validate_shm() and re-raise when the
    downstream _resolve_format()/_open_ipc_slots() step fails -- _open_ipc_slots
    only rolls back the per-slot resources it opened, not the shm handle
    _connect() is holding."""
    writer, _ = _make_valid_shm(temp_shm_name, shared_memory_cleanup, num_slots=2)
    try:
        imp = _make_importer(temp_shm_name, shape=(4, 4, 4), dtype="uint8", wait_backend="python")
        real_open = imp._cuda.ipc_open_mem_handle
        call_count = {"n": 0}

        def _flaky_open(handle, flags=1):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated open failure on slot 1")
            return real_open(handle, flags)

        imp._cuda.ipc_open_mem_handle = _flaky_open

        with pytest.raises(RuntimeError, match="simulated open failure"):
            imp._connect()

        assert imp._initialized is False
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# Importer.open()
# ---------------------------------------------------------------------------


def test_open_uses_policy_from_env_when_policy_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """policy=None -- open() must fall back to ImportPolicy.from_env(). The env
    var is explicitly deleted so the default (reconnect_enabled=True) is used
    deterministically, regardless of what's set in the ambient shell."""
    monkeypatch.delenv("CUDALINK_IMPORT_RECONNECT", raising=False)
    spec = ImportSpec(shm_name=_extra_name(), device=0, timeout_ms=1000.0)

    imp = Importer.open(spec, cuda=FakeCUDAAdapter())

    # Producer doesn't exist -- FileNotFoundError is swallowed because
    # reconnect_enabled defaulted to True; open() returns an unconnected Importer.
    assert imp._initialized is False
    assert imp._policy.reconnect_enabled is True
