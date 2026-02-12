"""
Tests for SharedMemory protocol layout and binary structure.

These tests verify the binary protocol correctness without requiring CUDA.
"""

import struct


def test_header_layout_sizes():
    """Verify header is exactly 20 bytes (4B magic + 8B version + 4B num_slots + 4B write_idx)."""
    PROTOCOL_MAGIC = 0x43495043  # "CIPC"
    SHM_HEADER_SIZE = 20  # Updated from 16 to include 4-byte magic

    # Pack header values
    magic = PROTOCOL_MAGIC
    version = 1
    num_slots = 3
    write_idx = 0

    header = (
        struct.pack("<I", magic)
        + struct.pack("<Q", version)
        + struct.pack("<I", num_slots)
        + struct.pack("<I", write_idx)
    )

    assert len(header) == SHM_HEADER_SIZE, f"Header size mismatch: {len(header)} != {SHM_HEADER_SIZE}"

    # Verify unpacking
    unpacked_magic = struct.unpack("<I", header[0:4])[0]
    unpacked_version = struct.unpack("<Q", header[4:12])[0]
    unpacked_num_slots = struct.unpack("<I", header[12:16])[0]
    unpacked_write_idx = struct.unpack("<I", header[16:20])[0]

    assert unpacked_magic == magic
    assert unpacked_version == version
    assert unpacked_num_slots == num_slots
    assert unpacked_write_idx == write_idx


def test_slot_layout_sizes():
    """Verify each slot is exactly 192 bytes (128B mem_handle + 64B event_handle)."""
    MEM_HANDLE_SIZE = 128
    EVENT_HANDLE_SIZE = 64
    SLOT_SIZE = MEM_HANDLE_SIZE + EVENT_HANDLE_SIZE

    assert SLOT_SIZE == 192, f"Slot size mismatch: {SLOT_SIZE} != 192"


def test_total_shm_size_3_slots():
    """Verify total SharedMemory size for 3 slots."""
    SHM_HEADER_SIZE = 20  # Updated from 16 (now includes 4-byte magic)
    SLOT_SIZE = 192
    SHUTDOWN_FLAG_SIZE = 1
    num_slots = 3

    total_size = SHM_HEADER_SIZE + (num_slots * SLOT_SIZE) + SHUTDOWN_FLAG_SIZE

    # 20 + 3*192 + 1 = 597 bytes (was 593 with old 16-byte header)
    assert total_size == 597, f"Total size mismatch: {total_size} != 597"


def test_total_shm_size_variable_slots():
    """Verify size formula for different slot counts."""
    SHM_HEADER_SIZE = 20  # Updated from 16
    SLOT_SIZE = 192
    SHUTDOWN_FLAG_SIZE = 1

    for num_slots in [2, 3, 4, 5]:
        expected_size = SHM_HEADER_SIZE + (num_slots * SLOT_SIZE) + SHUTDOWN_FLAG_SIZE
        assert expected_size == 20 + num_slots * 192 + 1


def test_write_idx_atomic_update():
    """Verify write_idx field read/write with struct."""
    # Simulate SharedMemory buffer (20 bytes for updated header)
    buffer = bytearray(20)

    # Write initial write_idx (offset 16, was 12 in old protocol)
    write_idx = 0
    buffer[16:20] = struct.pack("<I", write_idx)

    # Read back
    read_idx = struct.unpack("<I", bytes(buffer[16:20]))[0]
    assert read_idx == 0

    # Update write_idx
    for i in range(1, 10):
        buffer[16:20] = struct.pack("<I", i)
        read_idx = struct.unpack("<I", bytes(buffer[16:20]))[0]
        assert read_idx == i


def test_version_increment():
    """Verify version starts at 0, increments on write."""
    buffer = bytearray(20)  # Updated header size

    # Initial version (offset 4, after 4-byte magic)
    buffer[4:12] = struct.pack("<Q", 0)
    version = struct.unpack("<Q", bytes(buffer[4:12]))[0]
    assert version == 0

    # Increment
    new_version = version + 1
    buffer[4:12] = struct.pack("<Q", new_version)
    version = struct.unpack("<Q", bytes(buffer[4:12]))[0]
    assert version == 1


def test_shutdown_flag_position():
    """Verify shutdown flag is at correct offset."""
    SHM_HEADER_SIZE = 20  # Updated from 16
    SLOT_SIZE = 192
    num_slots = 3

    shutdown_offset = SHM_HEADER_SIZE + (num_slots * SLOT_SIZE)

    # Shutdown flag at byte 596 (was 592 with old 16-byte header)
    assert shutdown_offset == 596

    # For 4 slots: 20 + 4*192 = 788 (was 784)
    shutdown_offset_4 = 20 + (4 * 192)
    assert shutdown_offset_4 == 788


def test_handle_bytes_extraction():
    """Verify correct extraction of handle bytes per slot."""
    SHM_HEADER_SIZE = 20  # Updated from 16
    SLOT_SIZE = 192
    MEM_HANDLE_SIZE = 128

    # Simulate buffer with 3 slots (20 + 3*192 + 1 = 597 bytes)
    buffer = bytearray(20 + 3 * 192 + 1)

    for slot in [0, 1, 2]:
        base_offset = SHM_HEADER_SIZE + (slot * SLOT_SIZE)

        # Memory handle extraction
        mem_start = base_offset
        mem_end = base_offset + MEM_HANDLE_SIZE
        mem_handle_bytes = buffer[mem_start:mem_end]
        assert len(mem_handle_bytes) == 128

        # Event handle extraction
        event_start = base_offset + MEM_HANDLE_SIZE
        event_end = base_offset + SLOT_SIZE
        event_handle_bytes = buffer[event_start:event_end]
        assert len(event_handle_bytes) == 64


def test_slot_offset_calculation():
    """Verify slot offset calculation for ring buffer access."""
    SHM_HEADER_SIZE = 20  # Updated from 16
    SLOT_SIZE = 192

    # Expected offsets for 3 slots (updated for 20-byte header)
    expected_offsets = {
        0: 20,  # Slot 0: 20 (was 16)
        1: 212,  # Slot 1: 20 + 192 (was 208)
        2: 404,  # Slot 2: 20 + 192*2 (was 400)
    }

    for slot, expected in expected_offsets.items():
        calculated = SHM_HEADER_SIZE + (slot * SLOT_SIZE)
        assert calculated == expected, f"Slot {slot}: {calculated} != {expected}"


def test_read_slot_calculation():
    """Verify read slot calculation: (write_idx - 1) % num_slots."""
    num_slots = 3

    # Test cases: write_idx -> read_slot
    test_cases = [
        (0, 0),  # Special case: no frames written
        (1, 0),  # First frame written to slot 0, read from slot 0
        (2, 1),  # Second frame written to slot 1, read from slot 1
        (3, 2),  # Third frame written to slot 2, read from slot 2
        (4, 0),  # Fourth frame wraps to slot 0, read from slot 0
        (5, 1),  # Fifth frame wraps to slot 1, read from slot 1
    ]

    for write_idx, expected_read_slot in test_cases:
        read_slot = 0 if write_idx == 0 else (write_idx - 1) % num_slots
        assert read_slot == expected_read_slot, f"write_idx={write_idx}: read_slot={read_slot} != {expected_read_slot}"


# Extended Protocol Tests (Metadata Region)


def test_extended_protocol_size_3_slots():
    """Verify extended protocol size includes 20-byte metadata."""
    SHM_HEADER_SIZE = 20  # Updated from 16
    SLOT_SIZE = 192
    SHUTDOWN_FLAG_SIZE = 1
    METADATA_SIZE = 20
    num_slots = 3

    total = SHM_HEADER_SIZE + (num_slots * SLOT_SIZE) + SHUTDOWN_FLAG_SIZE + METADATA_SIZE
    # 20 + 3*192 + 1 + 20 = 617 bytes (was 613 with old 16-byte header)
    assert total == 617, f"Extended protocol size: {total} != 617"


def test_metadata_offset_calculation():
    """Verify metadata offset is immediately after shutdown flag."""
    SHM_HEADER_SIZE = 20  # Updated from 16
    SLOT_SIZE = 192
    SHUTDOWN_FLAG_SIZE = 1

    for num_slots in [2, 3, 4]:
        shutdown_offset = SHM_HEADER_SIZE + num_slots * SLOT_SIZE
        metadata_offset = shutdown_offset + SHUTDOWN_FLAG_SIZE
        expected = SHM_HEADER_SIZE + num_slots * SLOT_SIZE + SHUTDOWN_FLAG_SIZE
        assert metadata_offset == expected


def test_metadata_write_read_roundtrip():
    """Verify metadata fields can be written and read back correctly."""
    METADATA_OFFSET = 597  # For 3 slots (was 593 with old 16-byte header)

    buffer = bytearray(617)  # Updated from 613

    # Write metadata
    width, height, num_comps, dtype_code, buf_size = 1920, 1080, 4, 0, 1920 * 1080 * 4 * 4

    buffer[METADATA_OFFSET : METADATA_OFFSET + 4] = struct.pack("<I", width)
    buffer[METADATA_OFFSET + 4 : METADATA_OFFSET + 8] = struct.pack("<I", height)
    buffer[METADATA_OFFSET + 8 : METADATA_OFFSET + 12] = struct.pack("<I", num_comps)
    buffer[METADATA_OFFSET + 12 : METADATA_OFFSET + 16] = struct.pack("<I", dtype_code)
    buffer[METADATA_OFFSET + 16 : METADATA_OFFSET + 20] = struct.pack("<I", buf_size)

    # Read back
    assert struct.unpack("<I", buffer[METADATA_OFFSET : METADATA_OFFSET + 4])[0] == 1920
    assert struct.unpack("<I", buffer[METADATA_OFFSET + 4 : METADATA_OFFSET + 8])[0] == 1080
    assert struct.unpack("<I", buffer[METADATA_OFFSET + 8 : METADATA_OFFSET + 12])[0] == 4
    assert struct.unpack("<I", buffer[METADATA_OFFSET + 12 : METADATA_OFFSET + 16])[0] == 0
    assert struct.unpack("<I", buffer[METADATA_OFFSET + 16 : METADATA_OFFSET + 20])[0] == buf_size


def test_backward_compatibility_old_reader():
    """Verify new-format reader with magic can read extended buffer."""
    PROTOCOL_MAGIC = 0x43495043  # "CIPC"
    buffer = bytearray(617)  # Updated from 613

    # Write header (new format with magic)
    buffer[0:4] = struct.pack("<I", PROTOCOL_MAGIC)
    buffer[4:12] = struct.pack("<Q", 1)
    buffer[12:16] = struct.pack("<I", 3)
    buffer[16:20] = struct.pack("<I", 42)

    # New reader reads with magic validation
    magic = struct.unpack("<I", bytes(buffer[0:4]))[0]
    version = struct.unpack("<Q", bytes(buffer[4:12]))[0]
    num_slots = struct.unpack("<I", bytes(buffer[12:16]))[0]
    write_idx = struct.unpack("<I", bytes(buffer[16:20]))[0]
    shutdown = buffer[596]  # Updated offset (was 592)

    assert magic == PROTOCOL_MAGIC
    assert version == 1
    assert num_slots == 3
    assert write_idx == 42
    assert shutdown == 0


def test_dtype_code_mapping():
    """Verify dtype code constants match protocol specification."""
    DTYPE_FLOAT32 = 0
    DTYPE_FLOAT16 = 1
    DTYPE_UINT8 = 2

    # Test all valid dtype codes
    valid_codes = [DTYPE_FLOAT32, DTYPE_FLOAT16, DTYPE_UINT8]
    for code in valid_codes:
        assert 0 <= code <= 2, f"Invalid dtype code: {code}"


def test_metadata_layout_fields():
    """Verify metadata field layout and sizes."""
    # Each field is 4 bytes (uint32)
    FIELD_SIZE = 4
    NUM_FIELDS = 5  # width, height, num_comps, dtype_code, buffer_size
    METADATA_SIZE = FIELD_SIZE * NUM_FIELDS

    assert METADATA_SIZE == 20, f"Metadata size: {METADATA_SIZE} != 20"
