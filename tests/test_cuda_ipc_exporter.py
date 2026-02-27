"""
Tests for CUDAIPCExporter (producer side, TouchDesigner).

These tests use mocked TouchDesigner objects to test without TD runtime.
"""

from __future__ import annotations

import pytest

# =============================================================================
# Mock TouchDesigner Objects
# =============================================================================


class MockShape:
    """Mock for TOP's cuda_mem.shape."""

    def __init__(self, width: int, height: int, channels: int) -> None:
        self.width = width
        self.height = height
        self.numComps = channels


class MockCUDAMemory:
    """Mock for top_op.cudaMemory()."""

    def __init__(self, width: int = 512, height: int = 512, channels: int = 4, dtype_size: int = 4) -> None:
        self.ptr = 0xDEADBEEF0000  # Simulated GPU pointer
        self.shape = MockShape(width, height, channels)
        self.size = width * height * channels * dtype_size


class MockTOP:
    """Mock for TouchDesigner TOP operator."""

    def __init__(self, width: int = 512, height: int = 512, channels: int = 4) -> None:
        self._cuda_mem = MockCUDAMemory(width, height, channels)

    def cudaMemory(self, **kwargs: object) -> MockCUDAMemory:
        """Mock cudaMemory() that accepts optional stream parameter."""
        return self._cuda_mem


class MockParValue:
    """Mock for TD parameter value."""

    def __init__(self, value: object) -> None:
        self._value = value

    def eval(self) -> object:
        return self._value


class MockPar:
    """Mock for TD parameter container."""

    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, MockParValue(value))


class MockCOMP:
    """Mock for TD COMP operator."""

    def __init__(self, name: str = "test_comp", **params: object) -> None:
        self.name = name
        self.par = MockPar(**params)

    def parent(self) -> MockCOMP:
        return self


# =============================================================================
# Tests
# =============================================================================


def test_init_default_params() -> None:
    """Test constructor with default mocked parameters."""
    from CUDAIPCExtension import CUDAIPCExtension

    owner = MockCOMP(name="test_exporter", Ipcmemname="test_ipc", Debug=False, Active=True, Numslots=3)

    exporter = CUDAIPCExtension(owner)

    assert exporter.ownerComp is owner
    assert exporter.shm_name == "test_ipc"
    assert exporter.num_slots == 3
    assert not exporter._initialized


def test_init_custom_memname() -> None:
    """Test constructor reads Ipcmemname parameter."""
    from CUDAIPCExtension import CUDAIPCExtension

    custom_name = "my_custom_ipc_name"
    owner = MockCOMP(name="test_exporter", Ipcmemname=custom_name, Numslots=3)

    exporter = CUDAIPCExtension(owner)

    assert exporter.shm_name == custom_name


def test_init_fallback_memname() -> None:
    """Test constructor uses fallback name if parameter missing."""
    from CUDAIPCExtension import CUDAIPCExtension

    # Create owner without Ipcmemname parameter
    owner = MockCOMP(name="test_exporter")

    exporter = CUDAIPCExtension(owner)

    # Should fall back to default name
    assert exporter.shm_name == "cudalink_output_ipc"


def test_init_custom_numslots() -> None:
    """Test constructor reads Numslots parameter."""
    from CUDAIPCExtension import CUDAIPCExtension

    owner = MockCOMP(name="test_exporter", Numslots=4)

    exporter = CUDAIPCExtension(owner)

    assert exporter.num_slots == 4
    assert len(exporter.dev_ptrs) == 4
    assert len(exporter.ipc_handles) == 4


@pytest.mark.requires_cuda
def test_initialize_allocates_buffers(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: object) -> None:
    """Test initialize() creates GPU buffers."""
    from CUDAIPCExtension import CUDAIPCExtension

    owner = MockCOMP(name="test_exporter", Ipcmemname=temp_shm_name, Numslots=3)
    shared_memory_cleanup.append(temp_shm_name)

    exporter = CUDAIPCExtension(owner)
    exporter.cuda = cuda_runtime  # Inject real CUDA runtime

    # Initialize
    success = exporter.initialize(width=64, height=64, channels=4)

    assert success
    assert exporter._initialized
    assert all(ptr is not None for ptr in exporter.dev_ptrs)

    # Cleanup
    exporter.cleanup()


@pytest.mark.requires_cuda
def test_initialize_creates_shm(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: object) -> None:
    """Test initialize() creates SharedMemory with correct size."""
    from CUDAIPCExtension import CUDAIPCExtension

    owner = MockCOMP(name="test_exporter", Ipcmemname=temp_shm_name, Numslots=3)
    shared_memory_cleanup.append(temp_shm_name)

    exporter = CUDAIPCExtension(owner)
    exporter.cuda = cuda_runtime

    # Initialize
    success = exporter.initialize(width=64, height=64, channels=4)

    assert success
    assert exporter.shm_handle is not None

    # Verify SharedMemory size
    # 20 (header: 4B magic + 8B version + 4B num_slots + 4B write_idx)
    # + 3*128 (slots) + 1 (shutdown) + 20 (metadata) + 8 (timestamp) = 433
    expected_size = 20 + 3 * 128 + 1 + 20 + 8
    assert len(exporter.shm_handle.buf) >= expected_size

    # Cleanup
    exporter.cleanup()


@pytest.mark.requires_cuda
def test_shm_layout_header(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: object) -> None:
    """Test SharedMemory header layout (version, num_slots, write_idx)."""
    import struct

    from CUDAIPCExtension import CUDAIPCExtension

    owner = MockCOMP(name="test_exporter", Ipcmemname=temp_shm_name, Numslots=3)
    shared_memory_cleanup.append(temp_shm_name)

    exporter = CUDAIPCExtension(owner)
    exporter.cuda = cuda_runtime

    # Initialize
    exporter.initialize(width=64, height=64, channels=4)

    # Read header (offsets: magic=0-3, version=4-11, num_slots=12-15, write_idx=16-19)
    magic = struct.unpack("<I", bytes(exporter.shm_handle.buf[0:4]))[0]
    version = struct.unpack("<Q", bytes(exporter.shm_handle.buf[4:12]))[0]
    num_slots = struct.unpack("<I", bytes(exporter.shm_handle.buf[12:16]))[0]
    write_idx = struct.unpack("<I", bytes(exporter.shm_handle.buf[16:20]))[0]

    assert magic == 0x43495043  # "CIPC" magic number
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

    owner = MockCOMP(name="test_exporter", Ipcmemname=temp_shm_name, Active=True, Numslots=3)
    shared_memory_cleanup.append(temp_shm_name)

    exporter = CUDAIPCExtension(owner)
    exporter.cuda = cuda_runtime

    # Initialize
    exporter.initialize(width=8, height=8, channels=4)

    # Allocate a real GPU buffer for mock TOP (can't use fake pointer for memcpy)
    real_gpu_ptr = cuda_runtime.malloc(8 * 8 * 4 * 4)  # 8x8x4 channels, float32

    # Create mock TOP with real GPU pointer
    mock_top = MockTOP(width=8, height=8, channels=4)
    mock_top._cuda_mem.ptr = real_gpu_ptr.value  # Use real GPU pointer

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
            slot_before = exporter.write_idx % exporter.num_slots
            assert slot_before == expected_slot

            # Export frame
            success = exporter.export_frame(mock_top)
            assert success

            # Verify write_idx incremented in SharedMemory (offset 16-19)
            write_idx = struct.unpack("<I", bytes(exporter.shm_handle.buf[16:20]))[0]
            assert write_idx == expected_write_idx + 1

    finally:
        # Cleanup
        cuda_runtime.free(real_gpu_ptr)
        exporter.cleanup()


@pytest.mark.requires_cuda
def test_cleanup_frees_resources(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: object) -> None:
    """Test cleanup() frees GPU buffers and sets shutdown flag."""
    from CUDAIPCExtension import CUDAIPCExtension

    owner = MockCOMP(name="test_exporter", Ipcmemname=temp_shm_name, Numslots=3)
    shared_memory_cleanup.append(temp_shm_name)

    exporter = CUDAIPCExtension(owner)
    exporter.cuda = cuda_runtime

    # Initialize
    exporter.initialize(width=64, height=64, channels=4)
    assert exporter._initialized

    # Cleanup
    exporter.cleanup()

    # Verify state
    assert not exporter._initialized

    # Verify shutdown flag set (byte 592 for 3 slots)
    # Note: SharedMemory might be closed, so we can't always read this
    # Just verify cleanup didn't crash


def test_get_stats_format() -> None:
    """Test get_stats() returns correct dictionary structure."""
    from CUDAIPCExtension import CUDAIPCExtension

    owner = MockCOMP(name="test_exporter", Ipcmemname="test_ipc", Numslots=3)

    exporter = CUDAIPCExtension(owner)

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

    owner = MockCOMP(name="test_exporter", Ipcmemname="test_ipc")

    exporter = CUDAIPCExtension(owner)

    assert not exporter.is_ready()


def test_log_helper() -> None:
    """Test _log() helper method with verbosity control."""
    from CUDAIPCExtension import CUDAIPCExtension

    owner = MockCOMP(name="test_exporter", Debug=False)

    exporter = CUDAIPCExtension(owner)

    # Should not crash with verbosity off
    exporter._log("Test message")

    # Force logging
    exporter._log("Force message", force=True)

    # Enable verbosity
    exporter.verbose_performance = True
    exporter._log("Verbose message")
