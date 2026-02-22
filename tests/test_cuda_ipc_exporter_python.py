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


def test_constructor_defaults() -> None:
    """Default parameters produce expected attribute values."""
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


def test_dtype_code_mapping() -> None:
    """dtype strings map to correct internal protocol codes."""
    from cuda_link.cuda_ipc_exporter import (
        _DTYPE_CODE_MAP,
        DTYPE_FLOAT16,
        DTYPE_FLOAT32,
        DTYPE_UINT8,
    )

    assert _DTYPE_CODE_MAP["float32"] == DTYPE_FLOAT32
    assert _DTYPE_CODE_MAP["float16"] == DTYPE_FLOAT16
    assert _DTYPE_CODE_MAP["uint8"] == DTYPE_UINT8


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
    """SharedMemory header contains correct protocol magic 0x43495043."""
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
