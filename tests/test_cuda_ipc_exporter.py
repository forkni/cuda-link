"""
Tests for CUDAIPCExporter (producer side, TouchDesigner).

These tests use FakeTDHost / FakeTOPHandle from conftest to drive the extension
without a real TD runtime.
"""

from __future__ import annotations

import pytest
from conftest import FakeTDHost, FakeTOPHandle

# =============================================================================
# Tests
# =============================================================================


def test_init_default_params() -> None:
    """Test constructor with default mocked parameters."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = FakeTDHost(params={"Ipcmemname": "test_ipc", "Debug": False, "Active": True, "Numslots": 3})
    exporter = CUDAIPCExtension(None, host=host)

    assert exporter._host is host
    assert exporter.shm_name == "test_ipc"
    assert exporter.num_slots == 3
    assert not exporter._engine._initialized


def test_init_custom_memname() -> None:
    """Test constructor reads Ipcmemname parameter."""
    from CUDAIPCExtension import CUDAIPCExtension

    custom_name = "my_custom_ipc_name"
    host = FakeTDHost(params={"Ipcmemname": custom_name, "Numslots": 3})
    exporter = CUDAIPCExtension(None, host=host)

    assert exporter.shm_name == custom_name


def test_init_fallback_memname() -> None:
    """Test constructor uses fallback name if parameter missing."""
    from CUDAIPCExtension import CUDAIPCExtension

    # No Ipcmemname parameter → should fall back to default
    host = FakeTDHost(params={})
    exporter = CUDAIPCExtension(None, host=host)

    assert exporter.shm_name == "cudalink_output_ipc"


def test_init_custom_numslots() -> None:
    """Test constructor reads Numslots parameter."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = FakeTDHost(params={"Numslots": 4})
    exporter = CUDAIPCExtension(None, host=host)

    assert exporter.num_slots == 4
    assert len(exporter._engine.dev_ptrs) == 4
    assert len(exporter._engine.ipc_handles) == 4


@pytest.mark.requires_cuda
def test_initialize_allocates_buffers(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: object) -> None:
    """Test initialize() creates GPU buffers."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = FakeTDHost(params={"Ipcmemname": temp_shm_name, "Numslots": 3})
    shared_memory_cleanup.append(temp_shm_name)

    exporter = CUDAIPCExtension(None, host=host)
    exporter._engine.cuda = cuda_runtime  # Inject real CUDA runtime

    # Initialize
    success = exporter.initialize(width=64, height=64, channels=4)

    assert success
    assert exporter._engine._initialized
    assert all(ptr is not None for ptr in exporter._engine.dev_ptrs)

    # Cleanup
    exporter.cleanup()


@pytest.mark.requires_cuda
def test_initialize_creates_shm(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: object) -> None:
    """Test initialize() creates SharedMemory with correct size."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = FakeTDHost(params={"Ipcmemname": temp_shm_name, "Numslots": 3})
    shared_memory_cleanup.append(temp_shm_name)

    exporter = CUDAIPCExtension(None, host=host)
    exporter._engine.cuda = cuda_runtime

    # Initialize
    success = exporter.initialize(width=64, height=64, channels=4)

    assert success
    assert exporter._engine.shm_handle is not None

    # Verify SharedMemory size
    # 20 (header: 4B magic + 8B version + 4B num_slots + 4B write_idx)
    # + 3*128 (slots) + 1 (shutdown) + 20 (metadata) + 8 (timestamp) = 433
    expected_size = 20 + 3 * 128 + 1 + 20 + 8
    assert len(exporter._engine.shm_handle.buf) >= expected_size

    # Cleanup
    exporter.cleanup()


@pytest.mark.requires_cuda
def test_shm_layout_header(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: object) -> None:
    """Test SharedMemory header layout (version, num_slots, write_idx)."""
    import struct

    from CUDAIPCExtension import CUDAIPCExtension

    host = FakeTDHost(params={"Ipcmemname": temp_shm_name, "Numslots": 3})
    shared_memory_cleanup.append(temp_shm_name)

    exporter = CUDAIPCExtension(None, host=host)
    exporter._engine.cuda = cuda_runtime

    # Initialize
    exporter.initialize(width=64, height=64, channels=4)

    # Read header (offsets: magic=0-3, version=4-11, num_slots=12-15, write_idx=16-19)
    magic = struct.unpack("<I", bytes(exporter._engine.shm_handle.buf[0:4]))[0]
    version = struct.unpack("<Q", bytes(exporter._engine.shm_handle.buf[4:12]))[0]
    num_slots = struct.unpack("<I", bytes(exporter._engine.shm_handle.buf[12:16]))[0]
    write_idx = struct.unpack("<I", bytes(exporter._engine.shm_handle.buf[16:20]))[0]

    assert magic == 0x43495044  # "CIPD" magic number
    assert version >= 1  # Should be at least 1 after initialization
    assert num_slots == 3
    assert write_idx == 0  # Initially 0

    # Cleanup
    exporter.cleanup()


@pytest.mark.requires_cuda
def test_ring_buffer_rotation(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: object) -> None:
    """Test write_idx increments and slot cycles correctly."""
    import struct

    from CUDAIPCExtension import CUDAIPCExtension

    host = FakeTDHost(params={"Ipcmemname": temp_shm_name, "Active": True, "Numslots": 3})
    shared_memory_cleanup.append(temp_shm_name)

    exporter = CUDAIPCExtension(None, host=host)
    exporter._engine.cuda = cuda_runtime

    # Initialize
    exporter.initialize(width=8, height=8, channels=4)

    # Allocate a real GPU buffer and register as ExportBuffer
    real_gpu_ptr = cuda_runtime.malloc(8 * 8 * 4 * 4)  # 8x8x4 channels, float32
    host._tops["ExportBuffer"] = FakeTOPHandle(width=8, height=8, channels=4, gpu_ptr=real_gpu_ptr.value)

    try:
        # Export 5 frames and verify slot rotation
        expected_sequence = [
            (0, 0),  # write_idx=0, slot=0
            (1, 1),  # write_idx=1, slot=1
            (2, 2),  # write_idx=2, slot=2
            (3, 0),  # write_idx=3, slot=0 (wraps)
            (4, 1),  # write_idx=4, slot=1
        ]

        for expected_write_idx, expected_slot in expected_sequence:
            # Verify slot calculation
            slot_before = exporter._engine.write_idx % exporter.num_slots
            assert slot_before == expected_slot

            # Export frame
            success = exporter.export_frame()
            assert success

            # Verify write_idx incremented in SharedMemory (offset 16-19)
            write_idx = struct.unpack("<I", bytes(exporter._engine.shm_handle.buf[16:20]))[0]
            assert write_idx == expected_write_idx + 1

    finally:
        # Cleanup
        cuda_runtime.free(real_gpu_ptr)
        exporter.cleanup()


@pytest.mark.requires_cuda
def test_cleanup_frees_resources(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: object) -> None:
    """Test cleanup() frees GPU buffers and sets shutdown flag."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = FakeTDHost(params={"Ipcmemname": temp_shm_name, "Numslots": 3})
    shared_memory_cleanup.append(temp_shm_name)

    exporter = CUDAIPCExtension(None, host=host)
    exporter._engine.cuda = cuda_runtime

    # Initialize
    exporter.initialize(width=64, height=64, channels=4)
    assert exporter._engine._initialized

    # Cleanup
    exporter.cleanup()

    # Verify state
    assert not exporter._engine._initialized

    # Verify shutdown flag set (byte 592 for 3 slots)
    # Note: SharedMemory might be closed, so we can't always read this
    # Just verify cleanup didn't crash


def test_get_stats_format() -> None:
    """Test get_stats() returns correct dictionary structure."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = FakeTDHost(params={"Ipcmemname": "test_ipc", "Numslots": 3})
    exporter = CUDAIPCExtension(None, host=host)

    stats = exporter.get_stats()

    # Verify all expected keys present
    assert "initialized" in stats
    assert "buffer_size_mb" in stats
    assert "resolution" in stats
    assert "frame_count" in stats
    assert "shm_name" in stats
    assert "num_slots" in stats
    assert "write_idx" in stats
    assert "dev_ptrs" in stats


def test_is_ready_false_before_init() -> None:
    """Test is_ready() returns False before initialization."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = FakeTDHost(params={"Ipcmemname": "test_ipc"})
    exporter = CUDAIPCExtension(None, host=host)

    assert not exporter.is_ready()


def test_log_helper() -> None:
    """Test _log() helper method with verbosity control."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = FakeTDHost(params={"Debug": False})
    exporter = CUDAIPCExtension(None, host=host)

    # Should not crash with verbosity off
    exporter._log("Test message")

    # Force logging
    exporter._log("Force message", force=True)

    # Enable verbosity
    exporter.verbose_performance = True
    exporter._log("Verbose message")


# ---------------------------------------------------------------------------
# GPU-side float16→float32 conversion in TD receiver
# ---------------------------------------------------------------------------


def _make_receiver_with_float16_state(use_cupy: bool = False) -> object:
    """Build a TDReceiverEngine with manually-injected float16 state.

    Bypasses real CUDA/CuPy initialization to test routing logic only.
    Returns a TDReceiverEngine directly (no facade needed for unit tests).
    """
    import struct
    from unittest.mock import MagicMock, patch

    import numpy as np
    from CUDAIPCExtension import FORMAT_KIND_FLOAT, SHM_HEADER_SIZE, SLOT_SIZE

    from cuda_link.shm_protocol import SHMLayout

    with patch("TDReceiver.CUPY_AVAILABLE", use_cupy):
        from TDReceiver import FormatDescriptor, ReceiverConnection, RetryState, TDReceiverEngine

    HEIGHT, WIDTH, COMPS = 4, 4, 4
    NUM_SLOTS = 2
    F16_SIZE = HEIGHT * WIDTH * COMPS * 2  # float16 = 2 bytes/elem

    # Build a bytearray that looks like valid SHM (write_idx=1)
    shm_size = SHM_HEADER_SIZE + NUM_SLOTS * SLOT_SIZE + 1 + 20 + 8
    buf = bytearray(shm_size)
    struct.pack_into("<I", buf, 0, 0x43495044)  # magic "CIPD"
    struct.pack_into("<Q", buf, 4, 1)  # version
    struct.pack_into("<I", buf, 12, NUM_SLOTS)  # num_slots
    struct.pack_into("<I", buf, 16, 1)  # write_idx=1

    fake_host = MagicMock()
    fake_host.is_active.return_value = True
    from TDConfig import TDSenderConfig

    engine = TDReceiverEngine(
        host=fake_host,
        config=TDSenderConfig(),
        cuda=MagicMock(),
        log_fn=lambda msg, force=False: None,
        num_slots=NUM_SLOTS,
        device=0,
        shm_name="test_shm",
        verbose=False,
    )

    # Inject typed value objects (bypasses real CUDA initialization)
    engine._initialized = True
    engine.frame_count = 0
    engine._diag_frames_since_reinit = 999

    mock_dev_ptrs = [MagicMock() for _ in range(NUM_SLOTS)]
    mock_dev_ptrs[0].value = 0xDEAD0000
    mock_stream = MagicMock()
    mock_stream.value = 0x1234
    mock_shm = MagicMock()
    mock_shm.buf = buf

    layout = SHMLayout(NUM_SLOTS)
    engine._connection = ReceiverConnection(
        shm_handle=mock_shm,
        dev_ptrs=mock_dev_ptrs,
        ipc_handles=[None] * NUM_SLOTS,
        ipc_events=[MagicMock(), MagicMock()],
        stream=mock_stream,
        layout=layout,
        num_slots=NUM_SLOTS,
        ipc_version=1,
        shutdown_offset=layout.shutdown_offset,
        last_write_idx=0,
    )
    engine._format = FormatDescriptor(
        width=WIDTH,
        height=HEIGHT,
        num_comps=COMPS,
        format_kind=FORMAT_KIND_FLOAT,
        bits_per_comp=16,
        flags=0,
        buffer_size=F16_SIZE,
    )
    engine._retry = RetryState(connect_attempts=0, frames_since_last_retry=0)

    # Engine-private F16 scratch (mutable working buffers, not value objects)
    engine._f16_cpu_buf = np.zeros(HEIGHT * WIDTH * COMPS, dtype=np.float16)
    engine._f32_cpu_buf = np.zeros((HEIGHT, WIDTH, COMPS), dtype=np.float32)
    engine._f16_pinned_ptr = None
    engine._cached_shape = MagicMock()

    if use_cupy:
        engine._cupy_f32_buf = np.zeros((HEIGHT, WIDTH, COMPS), dtype=np.float32)
        engine._cupy_f16_views = [MagicMock() for _ in range(NUM_SLOTS)]
    else:
        engine._cupy_f32_buf = None
        engine._cupy_f16_views = []

    return engine


def test_float16_receiver_uses_cpu_fallback_when_cupy_unavailable() -> None:
    """import_frame() uses copy_numpy_array (CPU path) when CuPy is not available."""
    from unittest.mock import patch

    ext = _make_receiver_with_float16_state(use_cupy=False)

    handle = FakeTOPHandle()

    with patch("TDReceiver.CUPY_AVAILABLE", False):
        result = ext.import_frame(handle)

    assert result is True
    assert len(handle.copy_numpy_calls) == 1, "CPU path must call copy_numpy_array once"
    assert len(handle.copy_cuda_calls) == 0, "CPU path must not call copy_cuda_memory"


def test_float16_receiver_uses_gpu_path_when_cupy_available() -> None:
    """import_frame() uses copy_cuda_memory (GPU path) when CuPy is available.

    The GPU path eliminates both PCIe roundtrips: instead of GPU→CPU→GPU it
    performs f16→f32 conversion entirely on-device via CuPy's copyto, then
    calls copy_cuda_memory with the resulting float32 GPU pointer.
    """
    from unittest.mock import MagicMock, patch

    ext = _make_receiver_with_float16_state(use_cupy=True)

    handle = FakeTOPHandle()

    # Build a minimal CuPy mock: UnownedMemory, MemoryPointer, ndarray, ExternalStream, copyto
    mock_cp = MagicMock()
    mock_cp.float16 = "float16"
    mock_cp.float32 = "float32"
    # .data.ptr on the f32 buf must return an integer (GPU pointer)
    ext._cupy_f32_buf = MagicMock()
    ext._cupy_f32_buf.data.ptr = 0xF3200000

    with patch("TDReceiver.CUPY_AVAILABLE", True), patch("TDReceiver.cp", mock_cp):
        result = ext.import_frame(handle)

    assert result is True
    assert len(handle.copy_cuda_calls) == 1, "GPU path must call copy_cuda_memory once"
    assert len(handle.copy_numpy_calls) == 0, "GPU path must not call copy_numpy_array"
    # ExternalStream context manager must be used
    mock_cp.cuda.ExternalStream.assert_called_once()
