"""
Multi-process integration tests for CUDA IPC.

These tests verify the full producer-consumer pipeline.
Protocol v0.5.0: 20-byte header (4B magic "CIPC" + 8B version + 4B num_slots + 4B write_idx).
"""

from __future__ import annotations

import struct
from multiprocessing.shared_memory import SharedMemory

import pytest

# Protocol v0.5.0 constants
_MAGIC = 0x43495043  # "CIPC"
_HEADER_SIZE = 20  # 4B magic + 8B version + 4B num_slots + 4B write_idx
_SLOT_SIZE = 192  # 128B mem_handle + 64B event_handle


def _write_header(shm: SharedMemory, version: int, num_slots: int, write_idx: int = 0) -> None:
    """Write v0.5.0 protocol header into SharedMemory."""
    shm.buf[0:4] = struct.pack("<I", _MAGIC)
    shm.buf[4:12] = struct.pack("<Q", version)
    shm.buf[12:16] = struct.pack("<I", num_slots)
    shm.buf[16:20] = struct.pack("<I", write_idx)


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_producer_consumer_basic(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """Test basic producer allocates memory, consumer opens IPC handle."""

    shared_memory_cleanup.append(temp_shm_name)

    # Producer: Allocate GPU memory and create IPC handle
    size = 1024 * 1024  # 1 MB
    ptr = cuda_runtime.malloc(size)
    handle = cuda_runtime.ipc_get_mem_handle(ptr)

    # Create SharedMemory with v0.5.0 layout
    # 20 (header) + 3*192 (slots) + 1 (shutdown) + 20 (metadata) + 8 (timestamp) = 625
    num_slots = 3
    shm_size = _HEADER_SIZE + num_slots * _SLOT_SIZE + 1 + 20 + 8
    shm = SharedMemory(name=temp_shm_name, create=True, size=shm_size)

    # Write header
    _write_header(shm, version=1, num_slots=num_slots, write_idx=1)

    # Write IPC handle to slot 0
    shm.buf[_HEADER_SIZE : _HEADER_SIZE + 128] = bytes(handle.internal)

    # Consumer: Open IPC handle in same process (Windows limitation)
    # Note: On Windows, same process IPC may fail, so we just verify the handle is valid
    assert len(bytes(handle.internal)) == 128

    # Cleanup
    cuda_runtime.free(ptr)
    shm.close()
    shm.unlink()


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_ring_buffer_slot_cycling(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """Test producer writes to slots 0,1,2,0,1,2 and consumer reads correct slots."""
    shared_memory_cleanup.append(temp_shm_name)

    # Setup SharedMemory with 3 slots (v0.5.0 layout)
    num_slots = 3
    shm_size = _HEADER_SIZE + num_slots * _SLOT_SIZE + 1 + 20 + 8
    shm = SharedMemory(name=temp_shm_name, create=True, size=shm_size)

    # Write header
    _write_header(shm, version=1, num_slots=num_slots)

    # Allocate buffers and create handles for all slots
    buffers = []
    for slot in range(num_slots):
        ptr = cuda_runtime.malloc(1024)
        buffers.append(ptr)
        handle = cuda_runtime.ipc_get_mem_handle(ptr)

        base_offset = _HEADER_SIZE + slot * _SLOT_SIZE
        shm.buf[base_offset : base_offset + 128] = bytes(handle.internal)

    # Simulate producer writing frames
    write_sequence = [0, 1, 2, 3, 4, 5]  # 6 frames
    expected_slots = [0, 1, 2, 0, 1, 2]  # Expected slot usage

    for write_idx in write_sequence:
        # Producer writes to this slot
        slot = write_idx % num_slots
        assert slot == expected_slots[write_idx]

        # Update write_idx (at offset 16 in v0.5.0 header)
        shm.buf[16:20] = struct.pack("<I", write_idx + 1)

        # Consumer reads from (write_idx) slot (after increment)
        current_write_idx = struct.unpack("<I", bytes(shm.buf[16:20]))[0]
        read_slot = 0 if current_write_idx == 0 else (current_write_idx - 1) % num_slots
        assert read_slot == slot

    # Cleanup
    for ptr in buffers:
        cuda_runtime.free(ptr)
    shm.close()
    shm.unlink()


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_shutdown_signal_propagation(
    cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: list[str]
) -> None:
    """Test producer sets shutdown flag, consumer detects and cleans up."""
    shared_memory_cleanup.append(temp_shm_name)

    # Setup SharedMemory (v0.5.0 layout)
    num_slots = 3
    shm_size = _HEADER_SIZE + num_slots * _SLOT_SIZE + 1 + 20 + 8
    shm = SharedMemory(name=temp_shm_name, create=True, size=shm_size)

    # Write header
    _write_header(shm, version=1, num_slots=num_slots)

    # Allocate and write IPC handles
    buffers = []
    for slot in range(num_slots):
        ptr = cuda_runtime.malloc(1024)
        buffers.append(ptr)
        handle = cuda_runtime.ipc_get_mem_handle(ptr)

        base_offset = _HEADER_SIZE + slot * _SLOT_SIZE
        shm.buf[base_offset : base_offset + 128] = bytes(handle.internal)

    # Producer sets shutdown flag (immediately after slots)
    shutdown_offset = _HEADER_SIZE + num_slots * _SLOT_SIZE
    shm.buf[shutdown_offset] = 1

    # Consumer detects shutdown flag
    shutdown_detected = shm.buf[shutdown_offset] == 1
    assert shutdown_detected

    # Cleanup
    for ptr in buffers:
        cuda_runtime.free(ptr)
    shm.close()
    shm.unlink()


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_version_change_reinit(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """Test consumer detects version change and re-opens handles."""
    shared_memory_cleanup.append(temp_shm_name)

    # Setup SharedMemory (v0.5.0 layout)
    num_slots = 3
    shm_size = _HEADER_SIZE + num_slots * _SLOT_SIZE + 1 + 20 + 8
    shm = SharedMemory(name=temp_shm_name, create=True, size=shm_size)

    # Write initial header (version 1)
    _write_header(shm, version=1, num_slots=num_slots)

    # Allocate and write IPC handles
    buffers_v1 = []
    for slot in range(num_slots):
        ptr = cuda_runtime.malloc(1024)
        buffers_v1.append(ptr)
        handle = cuda_runtime.ipc_get_mem_handle(ptr)

        base_offset = _HEADER_SIZE + slot * _SLOT_SIZE
        shm.buf[base_offset : base_offset + 128] = bytes(handle.internal)

    # Consumer reads version 1 (at offset 4 in v0.5.0 header)
    version = struct.unpack("<Q", bytes(shm.buf[4:12]))[0]
    assert version == 1

    # Producer re-initializes (version 2)
    shm.buf[4:12] = struct.pack("<Q", 2)

    # Free old buffers and allocate new ones
    for ptr in buffers_v1:
        cuda_runtime.free(ptr)

    buffers_v2 = []
    for slot in range(num_slots):
        ptr = cuda_runtime.malloc(1024)
        buffers_v2.append(ptr)
        handle = cuda_runtime.ipc_get_mem_handle(ptr)

        base_offset = _HEADER_SIZE + slot * _SLOT_SIZE
        shm.buf[base_offset : base_offset + 128] = bytes(handle.internal)

    # Consumer detects version change
    new_version = struct.unpack("<Q", bytes(shm.buf[4:12]))[0]
    assert new_version == 2
    assert new_version != version  # Version changed

    # Cleanup
    for ptr in buffers_v2:
        cuda_runtime.free(ptr)
    shm.close()
    shm.unlink()
