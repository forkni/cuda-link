"""
Tests for CUDAIPCExporter (Python-side GPU exporter).

Tests are split into two groups:
- Pure unit tests (no CUDA required): constructor validation, dtype mapping, etc.
- CUDA integration tests (@pytest.mark.requires_cuda): SHM protocol, ring buffer, etc.
"""

from __future__ import annotations

import struct
import time
from multiprocessing.shared_memory import SharedMemory

import pytest

# ---------------------------------------------------------------------------
# Pure unit tests (no CUDA required)
# ---------------------------------------------------------------------------


def test_constructor_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default parameters produce expected attribute values (clean env)."""
    for var in (
        "CUDALINK_EXPORT_SYNC",
        "CUDALINK_EXPORT_PROFILE",
        "CUDALINK_EXPORT_FLUSH_PROBE",
        "CUDALINK_USE_GRAPHS",
        "CUDALINK_STICKY_ERROR_CHECK",
        "CUDALINK_ACTIVATION_BARRIER",
    ):
        monkeypatch.delenv(var, raising=False)

    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    exp = CUDAIPCExporter(shm_name="test_ipc", height=512, width=512)

    assert exp.shm_name == "test_ipc"
    assert exp.height == 512
    assert exp.width == 512
    assert exp.channels == 4
    assert exp.dtype == "uint8"
    assert exp.num_slots == 2
    assert exp.debug is False
    assert not exp.is_ready()
    assert exp._export_sync is True  # Phase 4: default flipped to ON
    assert exp._barrier_enabled is True  # Phase 4: default flipped to ON


def test_constructor_custom_params() -> None:
    """Custom parameters are correctly stored."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    exp = CUDAIPCExporter(
        shm_name="my_ipc",
        height=1080,
        width=1920,
        channels=4,
        dtype="float16",
        num_slots=3,
        debug=True,
    )

    assert exp.height == 1080
    assert exp.width == 1920
    assert exp.dtype == "float16"
    assert exp.num_slots == 3
    assert exp.debug is True


def test_kind_bits_mapping() -> None:
    """dtype strings map to correct (format_kind, bits, flags) wire encoding."""
    from cuda_link.cuda_ipc_exporter import (
        _DTYPE_TO_KIND_BITS,
        FLAGS_BFLOAT16,
        FORMAT_KIND_FLOAT,
        FORMAT_KIND_UNSIGNED,
    )

    assert _DTYPE_TO_KIND_BITS["float32"] == (FORMAT_KIND_FLOAT, 32, 0)
    assert _DTYPE_TO_KIND_BITS["float16"] == (FORMAT_KIND_FLOAT, 16, 0)
    assert _DTYPE_TO_KIND_BITS["uint8"] == (FORMAT_KIND_UNSIGNED, 8, 0)
    assert _DTYPE_TO_KIND_BITS["uint16"] == (FORMAT_KIND_UNSIGNED, 16, 0)
    # bfloat16 flag must not collide with any standard mapping
    assert FLAGS_BFLOAT16 == 0x0001


def test_dtype_itemsize_mapping() -> None:
    """dtype strings map to correct byte sizes."""
    from cuda_link.cuda_ipc_exporter import _DTYPE_ITEMSIZE_MAP

    assert _DTYPE_ITEMSIZE_MAP["float32"] == 4
    assert _DTYPE_ITEMSIZE_MAP["float16"] == 2
    assert _DTYPE_ITEMSIZE_MAP["uint8"] == 1


def test_data_size_calculation_uint8() -> None:
    """data_size is correctly computed for uint8."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    # 512 x 512 x 4 channels x 1 byte = 1,048,576 bytes
    exp = CUDAIPCExporter(shm_name="x", height=512, width=512, channels=4, dtype="uint8")
    assert exp.data_size == 512 * 512 * 4 * 1


def test_data_size_calculation_float32() -> None:
    """data_size is correctly computed for float32."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    # 256 x 256 x 4 x 4 bytes = 1,048,576 bytes
    exp = CUDAIPCExporter(shm_name="x", height=256, width=256, channels=4, dtype="float32")
    assert exp.data_size == 256 * 256 * 4 * 4


def test_invalid_dtype_raises() -> None:
    """Unsupported dtype raises ValueError."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    with pytest.raises(ValueError, match="Unsupported dtype"):
        CUDAIPCExporter(shm_name="x", height=64, width=64, dtype="int32")


def test_num_slots_zero_raises() -> None:
    """num_slots=0 raises ValueError."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    with pytest.raises(ValueError, match="num_slots"):
        CUDAIPCExporter(shm_name="x", height=64, width=64, num_slots=0)


def test_num_slots_over_max_raises() -> None:
    """num_slots=11 raises ValueError."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    with pytest.raises(ValueError, match="num_slots"):
        CUDAIPCExporter(shm_name="x", height=64, width=64, num_slots=11)


def test_num_slots_max_valid() -> None:
    """num_slots=10 is accepted."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    exp = CUDAIPCExporter(shm_name="x", height=64, width=64, num_slots=10)
    assert exp.num_slots == 10


def test_is_ready_before_init() -> None:
    """is_ready() returns False before initialize()."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    exp = CUDAIPCExporter(shm_name="x", height=64, width=64)
    assert not exp.is_ready()


def test_get_stats_before_init() -> None:
    """get_stats() returns dict with expected keys before initialize()."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    exp = CUDAIPCExporter(shm_name="x", height=64, width=64)
    stats = exp.get_stats()

    assert isinstance(stats, dict)
    assert "initialized" in stats
    assert "frame_count" in stats
    assert stats["initialized"] is False
    assert stats["frame_count"] == 0


def test_double_cleanup_no_crash() -> None:
    """Calling cleanup() twice does not raise."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    exp = CUDAIPCExporter(shm_name="x", height=64, width=64)
    exp.cleanup()  # Should be safe even without initialize()
    exp.cleanup()  # Double-cleanup must not raise


# ---------------------------------------------------------------------------
# CUDA integration tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_cuda
def test_initialize_success(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """initialize() returns True and is_ready() becomes True."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    shared_memory_cleanup.append(temp_shm_name)
    exp = CUDAIPCExporter(shm_name=temp_shm_name, height=64, width=64)
    try:
        result = exp.initialize()
        assert result is True
        assert exp.is_ready()
    finally:
        exp.cleanup()


@pytest.mark.requires_cuda
def test_initialize_idempotent(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """Calling initialize() twice returns True both times without error."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    shared_memory_cleanup.append(temp_shm_name)
    exp = CUDAIPCExporter(shm_name=temp_shm_name, height=64, width=64)
    try:
        assert exp.initialize() is True
        assert exp.initialize() is True  # Second call must be idempotent
    finally:
        exp.cleanup()


@pytest.mark.requires_cuda
def test_shm_protocol_magic(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """SharedMemory header contains correct protocol magic 0x43495044 ('CIPD')."""
    from cuda_link.cuda_ipc_exporter import PROTOCOL_MAGIC, CUDAIPCExporter

    shared_memory_cleanup.append(temp_shm_name)
    exp = CUDAIPCExporter(shm_name=temp_shm_name, height=64, width=64)
    try:
        exp.initialize()
        shm = SharedMemory(name=temp_shm_name)
        try:
            magic = struct.unpack_from("<I", shm.buf, 0)[0]
            assert magic == PROTOCOL_MAGIC, f"Expected 0x{PROTOCOL_MAGIC:08x}, got 0x{magic:08x}"
        finally:
            shm.close()
    finally:
        exp.cleanup()


@pytest.mark.requires_cuda
def test_shm_protocol_num_slots(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """SharedMemory header encodes the correct num_slots value."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    NUM_SLOTS_OFFSET = 12

    shared_memory_cleanup.append(temp_shm_name)
    exp = CUDAIPCExporter(shm_name=temp_shm_name, height=64, width=64, num_slots=3)
    try:
        exp.initialize()
        shm = SharedMemory(name=temp_shm_name)
        try:
            num_slots = struct.unpack_from("<I", shm.buf, NUM_SLOTS_OFFSET)[0]
            assert num_slots == 3
        finally:
            shm.close()
    finally:
        exp.cleanup()


@pytest.mark.requires_cuda
def test_ring_buffer_write_idx_increments(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """write_idx in SharedMemory increments after each export_frame() call."""

    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter
    from cuda_link.cuda_ipc_wrapper import get_cuda_runtime

    WRITE_IDX_OFFSET = 16

    shared_memory_cleanup.append(temp_shm_name)
    exp = CUDAIPCExporter(shm_name=temp_shm_name, height=64, width=64, dtype="uint8")
    cuda = get_cuda_runtime()

    try:
        exp.initialize()
        shm = SharedMemory(name=temp_shm_name)

        # Allocate a small GPU buffer for the test
        test_size = 64 * 64 * 4
        gpu_buf = cuda.malloc(test_size)

        try:
            for expected_idx in range(1, 4):
                exp.export_frame(gpu_ptr=int(gpu_buf.value), size=test_size)
                write_idx = struct.unpack_from("<I", shm.buf, WRITE_IDX_OFFSET)[0]
                assert write_idx == expected_idx, f"Expected write_idx={expected_idx}, got {write_idx}"
        finally:
            cuda.free(gpu_buf)
            shm.close()
    finally:
        exp.cleanup()


@pytest.mark.requires_cuda
def test_shutdown_flag_set_on_cleanup(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """cleanup() writes shutdown flag=1 to SharedMemory before unlinking."""

    from cuda_link.cuda_ipc_exporter import SHM_HEADER_SIZE, SLOT_SIZE, CUDAIPCExporter

    shared_memory_cleanup.append(temp_shm_name)
    num_slots = 2
    exp = CUDAIPCExporter(shm_name=temp_shm_name, height=64, width=64, num_slots=num_slots)
    exp.initialize()

    # Open a second handle BEFORE cleanup to observe the flag
    shm_observer = SharedMemory(name=temp_shm_name)
    shutdown_offset = SHM_HEADER_SIZE + (num_slots * SLOT_SIZE)

    try:
        # Flag should be 0 before cleanup
        assert shm_observer.buf[shutdown_offset] == 0

        exp.cleanup()

        # Flag should be 1 after cleanup (written before SHM close)
        assert shm_observer.buf[shutdown_offset] == 1
    finally:
        shm_observer.close()
        # Do NOT unlink — cleanup() already did it; shared_memory_cleanup will suppress FileNotFoundError


@pytest.mark.requires_cuda
def test_context_manager(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """Context manager initializes and cleans up correctly."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    shared_memory_cleanup.append(temp_shm_name)
    with CUDAIPCExporter(shm_name=temp_shm_name, height=64, width=64) as exp:
        exp.initialize()
        assert exp.is_ready()

    # After context exit, should no longer be ready
    assert not exp.is_ready()


@pytest.mark.requires_cuda
def test_timestamp_uses_perf_counter(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """Producer timestamp in SharedMemory is from time.perf_counter() (monotonic clock).

    We verify this by checking the timestamp increases between frames AND that the
    delta is consistent with perf_counter resolution (not epoch seconds).
    """
    from cuda_link.cuda_ipc_exporter import (
        METADATA_SIZE,
        SHM_HEADER_SIZE,
        SHUTDOWN_FLAG_SIZE,
        SLOT_SIZE,
        CUDAIPCExporter,
    )
    from cuda_link.cuda_ipc_wrapper import get_cuda_runtime

    shared_memory_cleanup.append(temp_shm_name)
    num_slots = 2
    exp = CUDAIPCExporter(shm_name=temp_shm_name, height=64, width=64, num_slots=num_slots)
    cuda = get_cuda_runtime()

    try:
        exp.initialize()
        shm = SharedMemory(name=temp_shm_name)
        timestamp_offset = SHM_HEADER_SIZE + (num_slots * SLOT_SIZE) + SHUTDOWN_FLAG_SIZE + METADATA_SIZE

        test_size = 64 * 64 * 4
        gpu_buf = cuda.malloc(test_size)

        try:
            t_before = time.perf_counter()
            exp.export_frame(gpu_ptr=int(gpu_buf.value), size=test_size)
            t_after = time.perf_counter()

            ts_bytes = bytes(shm.buf[timestamp_offset : timestamp_offset + 8])
            ts = struct.unpack("<d", ts_bytes)[0]

            # Timestamp must be within [t_before, t_after + small_epsilon]
            assert t_before <= ts <= t_after + 0.001, (
                f"Timestamp {ts:.6f} not in expected perf_counter range [{t_before:.6f}, {t_after + 0.001:.6f}]"
            )
        finally:
            cuda.free(gpu_buf)
            shm.close()
    finally:
        exp.cleanup()


# ---------------------------------------------------------------------------
# Improvement 2: SharedMemory write ordering (atomicity)
# ---------------------------------------------------------------------------


def _make_exporter_with_mock_state(num_slots: int = 2, dtype: str = "uint8") -> object:
    """Build a CUDAIPCExporter with manually-injected state (no real CUDA required).

    Tests that verify SharedMemory write ordering inject all state via MagicMock and a
    bytearray buffer, bypassing real CUDA IPC initialization entirely.
    """
    from unittest.mock import MagicMock

    from cuda_link.cuda_ipc_exporter import (
        METADATA_SIZE,
        SHM_HEADER_SIZE,
        SHUTDOWN_FLAG_SIZE,
        SLOT_SIZE,
        TIMESTAMP_SIZE,
        CUDAIPCExporter,
    )

    shm_size = SHM_HEADER_SIZE + num_slots * SLOT_SIZE + SHUTDOWN_FLAG_SIZE + METADATA_SIZE + TIMESTAMP_SIZE

    exp = object.__new__(CUDAIPCExporter)
    exp.shm_name = "mock_shm"
    exp.height = 8
    exp.width = 8
    exp.channels = 4
    exp.dtype = dtype
    exp.num_slots = num_slots
    exp.debug = False
    exp.device = 0
    exp.data_size = 8 * 8 * 4  # H * W * C * itemsize (uint8 = 1 byte)
    exp.write_idx = 0
    exp.frame_count = 0
    exp.source_sync_event = None
    exp.ipc_stream = MagicMock()
    exp.cuda = MagicMock()
    exp.dev_ptrs = [MagicMock() for _ in range(num_slots)]
    exp.ipc_events = [None] * num_slots
    exp._shutdown_offset = SHM_HEADER_SIZE + num_slots * SLOT_SIZE
    exp._ts_offset = exp._shutdown_offset + SHUTDOWN_FLAG_SIZE + METADATA_SIZE
    exp._initialized = True
    exp._export_sync = False

    # C2 fields (Batch 2A)
    exp._strict_device = False
    exp._source_sync_device_warned = False
    exp._ptr_device_cache = set()

    # Phase 2 CUDA Graphs state (disabled in mock — no real CUDA context)
    exp._use_graphs = False
    exp._graphs_disabled = False
    exp._graph_execs = [None] * num_slots
    exp._graph_memcpy_nodes = [None] * num_slots

    # Phase 3 diagnostic knobs (off in mock)
    exp._export_profile = False
    exp._export_flush_probe = False
    exp.total_sync_us = 0.0
    exp.total_sticky_check_us = 0.0
    exp.total_flush_probe_us = 0.0

    # F9 activation barrier (disabled in mock)
    exp._barrier_enabled = False
    exp._barrier_stale_ns = 5_000_000_000
    exp._barrier_shm = None
    exp._barrier_skip_log_last_ns = 0
    exp._barrier_stale_log_last_ns = 0

    # Mock pointer_get_attributes to return device=0, type=2 (device memory) — valid
    mock_attrs = MagicMock()
    mock_attrs.type = 2  # cudaMemoryTypeDevice
    mock_attrs.device = 0
    exp.cuda.pointer_get_attributes.return_value = mock_attrs

    exp.shm_handle = MagicMock()
    exp.shm_handle.buf = bytearray(shm_size)

    return exp


def test_shm_write_correctness_after_export_frame() -> None:
    """After export_frame(), shutdown_flag=0 and write_idx is incremented by 1."""
    exp = _make_exporter_with_mock_state()
    buf = exp.shm_handle.buf

    # Simulate stale shutdown_flag=1 left by a prior producer session
    buf[exp._shutdown_offset] = 1
    initial_write_idx = exp.write_idx

    result = exp.export_frame(gpu_ptr=0, size=exp.data_size)

    assert result is True
    assert buf[exp._shutdown_offset] == 0, "shutdown_flag must be 0 after export_frame()"
    actual_write_idx = struct.unpack_from("<I", buf, 16)[0]
    assert actual_write_idx == initial_write_idx + 1, (
        f"write_idx must increment from {initial_write_idx} to {initial_write_idx + 1}, got {actual_write_idx}"
    )


def test_shm_write_ordering_shutdown_before_write_idx() -> None:
    """shutdown_flag is cleared BEFORE write_idx is published.

    Atomicity invariant: the consumer reads shutdown_flag BEFORE write_idx.
    Publishing write_idx last ensures the consumer always sees shutdown_flag=0
    when it detects a new frame, even with a stale flag=1 from a prior session.
    """
    import struct as real_struct
    from unittest.mock import MagicMock, patch

    write_log: list[tuple] = []

    class _SpyBuf(bytearray):
        """bytearray subclass that records __setitem__ writes in write_log."""

        def __setitem__(self, key, val) -> None:
            if isinstance(key, int):
                write_log.append(("setitem", key, val))
            super().__setitem__(key, val)

    # Spy on _ST_U32.pack_into (write_idx publish uses this pre-compiled Struct)
    real_st_u32 = real_struct.Struct("<I")
    real_st_f64 = real_struct.Struct("<d")

    def spy_u32_pack_into(buf, offset: int, *args) -> None:
        write_log.append(("pack_into", offset))
        real_st_u32.pack_into(buf, offset, *args)

    def spy_f64_pack_into(buf, offset: int, *args) -> None:
        real_st_f64.pack_into(buf, offset, *args)

    mock_st_u32 = MagicMock()
    mock_st_u32.pack_into.side_effect = spy_u32_pack_into
    mock_st_f64 = MagicMock()
    mock_st_f64.pack_into.side_effect = spy_f64_pack_into

    exp = _make_exporter_with_mock_state()

    # Replace the plain bytearray with our spy; set stale shutdown_flag=1
    spy_buf = _SpyBuf(len(exp.shm_handle.buf))
    spy_buf[exp._shutdown_offset] = 1
    write_log.clear()  # discard the setup write above
    exp.shm_handle.buf = spy_buf

    with (
        patch("cuda_link.cuda_ipc_exporter._ST_U32", mock_st_u32),
        patch("cuda_link.cuda_ipc_exporter._ST_F64", mock_st_f64),
    ):
        result = exp.export_frame(gpu_ptr=0, size=exp.data_size)

    assert result is True

    WRITE_IDX_OFFSET = 16
    shutdown_pos = next(
        (i for i, e in enumerate(write_log) if e[0] == "setitem" and e[1] == exp._shutdown_offset),
        None,
    )
    write_idx_pos = next(
        (i for i, e in enumerate(write_log) if e[0] == "pack_into" and e[1] == WRITE_IDX_OFFSET),
        None,
    )

    assert shutdown_pos is not None, "shutdown_flag must be written during export_frame()"
    assert write_idx_pos is not None, "write_idx must be published during export_frame()"
    assert shutdown_pos < write_idx_pos, (
        f"shutdown_flag write (log[{shutdown_pos}]) must precede "
        f"write_idx publish (log[{write_idx_pos}]); full log: {write_log}"
    )


def test_release_fence_called_between_flag_and_write_idx() -> None:
    """C3: _release_fence() is called after shutdown_flag clear and before write_idx publish.

    The fence must sit between the two writes to form the mechanical release-barrier
    guarantee. This test verifies the position using the _SpyBuf + _release_fence spy approach.
    """
    import struct as real_struct
    from unittest.mock import MagicMock, patch

    fence_calls: list[int] = []  # index in write_log at time of fence call
    write_log: list[tuple] = []

    class _SpyBuf(bytearray):
        def __setitem__(self, key, val) -> None:
            if isinstance(key, int):
                write_log.append(("setitem", key, val))
            super().__setitem__(key, val)

    real_st_u32 = real_struct.Struct("<I")
    real_st_f64 = real_struct.Struct("<d")

    def spy_u32_pack_into(buf, offset: int, *args) -> None:
        write_log.append(("pack_into", offset))
        real_st_u32.pack_into(buf, offset, *args)

    def spy_f64_pack_into(buf, offset: int, *args) -> None:
        real_st_f64.pack_into(buf, offset, *args)

    def spy_fence() -> None:
        fence_calls.append(len(write_log))  # record position in log

    mock_st_u32 = MagicMock()
    mock_st_u32.pack_into.side_effect = spy_u32_pack_into
    mock_st_f64 = MagicMock()
    mock_st_f64.pack_into.side_effect = spy_f64_pack_into

    exp = _make_exporter_with_mock_state()
    spy_buf = _SpyBuf(len(exp.shm_handle.buf))
    spy_buf[exp._shutdown_offset] = 1
    write_log.clear()
    exp.shm_handle.buf = spy_buf

    with (
        patch("cuda_link.cuda_ipc_exporter._ST_U32", mock_st_u32),
        patch("cuda_link.cuda_ipc_exporter._ST_F64", mock_st_f64),
        patch("cuda_link.cuda_ipc_exporter._release_fence", spy_fence),
    ):
        result = exp.export_frame(gpu_ptr=0, size=exp.data_size)

    assert result is True
    assert len(fence_calls) == 1, "exactly one fence call per export_frame"

    WRITE_IDX_OFFSET = 16
    shutdown_pos = next(
        (i for i, e in enumerate(write_log) if e[0] == "setitem" and e[1] == exp._shutdown_offset),
        None,
    )
    write_idx_pos = next(
        (i for i, e in enumerate(write_log) if e[0] == "pack_into" and e[1] == WRITE_IDX_OFFSET),
        None,
    )
    fence_pos = fence_calls[0]

    assert shutdown_pos is not None
    assert write_idx_pos is not None
    assert shutdown_pos < fence_pos, "fence must come AFTER shutdown_flag write"
    assert fence_pos <= write_idx_pos, "fence must come BEFORE write_idx publish"
