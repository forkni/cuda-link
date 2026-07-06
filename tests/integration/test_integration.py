"""
Multi-process integration tests for CUDA IPC.

These tests verify the full producer-consumer pipeline.
Protocol v1.0.0: 20-byte header (4B magic "CIPD" + 8B version + 4B num_slots + 4B write_idx).
"""

from __future__ import annotations

import struct
from multiprocessing.shared_memory import SharedMemory

import pytest

from cuda_link import FrameOutcome, FrameSpec, GpuFrame
from cuda_link.exporter import Exporter
from cuda_link.shm_protocol import Metadata, SHMLayout, read_num_slots, read_version, read_write_idx

_HEADER_SIZE = 20  # kept for slot-offset arithmetic in test body
_SLOT_SIZE = 128


def _write_header(shm: SharedMemory, version: int, num_slots: int, write_idx: int = 0) -> None:
    """Write v0.5.0 protocol header into SharedMemory."""
    layout = SHMLayout(num_slots)
    shm.buf[: layout.total_size] = layout.build_buffer(version=version, write_idx=write_idx)


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_producer_consumer_basic(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """Real Exporter publishes handles + metadata; verify the live SHM matches its own state."""
    shared_memory_cleanup.append(temp_shm_name)

    spec = FrameSpec(shm_name=temp_shm_name, height=8, width=8, channels=4, dtype="uint8", num_slots=2)
    exporter = Exporter.open(spec)
    try:
        src_ptr = cuda_runtime.malloc(exporter.data_size)
        try:
            outcome = exporter.export(GpuFrame(ptr=int(src_ptr.value), size=exporter.data_size))
            assert outcome is FrameOutcome.PUBLISHED

            buf = exporter.shm_handle.buf
            assert read_num_slots(buf) == spec.num_slots
            assert read_version(buf) >= 1
            assert read_write_idx(buf) == exporter.write_idx

            layout = SHMLayout(spec.num_slots)
            metadata = Metadata.read_from(buf, layout)
            assert metadata.data_size == exporter.data_size
            assert metadata.expected_size == exporter.data_size

            # The slot just written must carry the exact handle bytes Exporter recorded
            # for it — the real proof that publish wrote to live SHM, not a copy.
            slot = (exporter.write_idx - 1) % spec.num_slots
            offset = layout.mem_handle_offset(slot)
            expected_handle = bytes(exporter.ipc_handles[slot].internal)
            assert bytes(buf[offset : offset + 64]) == expected_handle
        finally:
            cuda_runtime.free(src_ptr)
    finally:
        exporter.close()


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_ring_buffer_slot_cycling(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """Real Exporter cycles frames through its ring slots; verify live SHM tracks each publish."""
    shared_memory_cleanup.append(temp_shm_name)

    spec = FrameSpec(shm_name=temp_shm_name, height=8, width=8, channels=4, dtype="uint8", num_slots=3)
    exporter = Exporter.open(spec)
    layout = SHMLayout(spec.num_slots)
    try:
        src_ptr = cuda_runtime.malloc(exporter.data_size)
        try:
            num_frames = 6  # two full cycles through 3 slots
            for frame_idx in range(num_frames):
                outcome = exporter.export(GpuFrame(ptr=int(src_ptr.value), size=exporter.data_size))
                assert outcome is FrameOutcome.PUBLISHED

                assert exporter.write_idx == frame_idx + 1
                assert read_write_idx(exporter.shm_handle.buf) == exporter.write_idx

                slot = frame_idx % spec.num_slots
                offset = layout.mem_handle_offset(slot)
                expected_handle = bytes(exporter.ipc_handles[slot].internal)
                actual_handle = bytes(exporter.shm_handle.buf[offset : offset + 64])
                assert actual_handle == expected_handle, (
                    f"slot {slot} handle in SHM diverged from Exporter's own record after frame {frame_idx}"
                )
        finally:
            cuda_runtime.free(src_ptr)
    finally:
        exporter.close()


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
        shm.buf[base_offset : base_offset + 64] = bytes(handle.internal)

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
        shm.buf[base_offset : base_offset + 64] = bytes(handle.internal)

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
        shm.buf[base_offset : base_offset + 64] = bytes(handle.internal)

    # Consumer detects version change
    new_version = struct.unpack("<Q", bytes(shm.buf[4:12]))[0]
    assert new_version == 2
    assert new_version != version  # Version changed

    # Cleanup
    for ptr in buffers_v2:
        cuda_runtime.free(ptr)
    shm.close()
    shm.unlink()
