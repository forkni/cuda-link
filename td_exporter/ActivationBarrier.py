"""Cross-process SHM activation barrier for cuda-link.

Coordinates Python producer <-> TD-side Sender activation windows.
When a Sender is initializing, it increments active_count; the producer
skips export_frame while non-zero (best-effort, no OS atomics needed —
the 5 s stale-timeout recovers from any stuck state).

Segment layout (64 bytes, little-endian):
  Offset  Size  Field            Description
  ------  ----  -----            -----------
  0       4     magic            0xCDA1BAAA — guards against alien segments
  4       4     version          1 — bumped if layout changes
  8       4     active_count     Number of Senders inside an activation window
  12      4     _pad             Align last_change_ns to 8 bytes
  16      8     last_change_ns   time.monotonic_ns() of most recent write
  24      4     barrier_skips    Producer-incremented skip-frame counter
  28      4     last_writer_pid  Diagnostic: PID of last active_count writer
  32      32    reserved         Zero-filled; reserved for future fields
"""

from __future__ import annotations

import struct
import time
from multiprocessing.shared_memory import SharedMemory

SHM_NAME = "cudalink_activation_barrier"
SHM_SIZE = 64
MAGIC = 0xCDA1BAAA
VERSION = 1

# Struct: magic(u32) version(u32) active_count(u32) pad(u32) last_change_ns(u64)
#         barrier_skips(u32) last_writer_pid(u32) reserved(32s)
_STRUCT = struct.Struct("<IIIIQII32s")

assert _STRUCT.size == SHM_SIZE, f"Layout error: struct size {_STRUCT.size} != {SHM_SIZE}"


def open_or_create(*, create: bool) -> SharedMemory:
    """Open the existing segment or create and initialise it.

    Args:
        create: When True, create the segment on FileNotFoundError and write
                the magic/version header. When False, raise FileNotFoundError
                if the segment does not yet exist.

    Returns:
        Open SharedMemory handle (caller must close when done).
    """
    try:
        return SharedMemory(name=SHM_NAME)
    except FileNotFoundError:
        if not create:
            raise
        shm = SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
        _STRUCT.pack_into(shm.buf, 0, MAGIC, VERSION, 0, 0, 0, 0, 0, b"\x00" * 32)
        return shm


def read_state(shm: SharedMemory) -> tuple[int, int, int]:
    """Return (active_count, last_change_ns, barrier_skips).

    Snapshot-reads the full 64-byte segment to avoid tearing.
    """
    fields = _STRUCT.unpack(bytes(shm.buf[:SHM_SIZE]))
    # (magic, version, active_count, pad, last_change_ns, barrier_skips, pid, reserved)
    return fields[2], fields[4], fields[5]


def increment(shm: SharedMemory, pid: int) -> int:
    """Increment active_count, refresh last_change_ns and last_writer_pid.

    Best-effort: no OS-level atomic. Race window is microseconds; the
    producer-side stale-timeout absorbs any stuck state.

    Returns:
        New active_count value.
    """
    fields = list(_STRUCT.unpack(bytes(shm.buf[:SHM_SIZE])))
    fields[2] += 1  # active_count
    fields[4] = time.monotonic_ns()  # last_change_ns
    fields[6] = pid  # last_writer_pid
    _STRUCT.pack_into(shm.buf, 0, *fields)
    return fields[2]


def decrement(shm: SharedMemory, pid: int) -> int:
    """Decrement active_count (clamps at zero), refresh timestamps.

    Returns:
        New active_count value.
    """
    fields = list(_STRUCT.unpack(bytes(shm.buf[:SHM_SIZE])))
    fields[2] = max(0, fields[2] - 1)  # active_count, no underflow
    fields[4] = time.monotonic_ns()  # last_change_ns
    fields[6] = pid  # last_writer_pid
    _STRUCT.pack_into(shm.buf, 0, *fields)
    return fields[2]


def bump_skip(shm: SharedMemory) -> None:
    """Increment barrier_skips counter (producer-only diagnostic)."""
    fields = list(_STRUCT.unpack(bytes(shm.buf[:SHM_SIZE])))
    fields[5] += 1  # barrier_skips
    _STRUCT.pack_into(shm.buf, 0, *fields)
