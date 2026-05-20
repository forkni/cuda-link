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

import contextlib
import logging
import struct
import time
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from typing import Callable

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Role classes built on the SHM primitives above
# ---------------------------------------------------------------------------


@dataclass
class CheckerBarrier:
    """Producer/exporter-side checker for the activation-barrier SHM protocol.

    Reads active_count from the shared segment and returns True from
    should_skip_publish() while a Sender is inside its activation window.
    Lazily opens the segment on first call; applies a stale-timeout so a
    crashed Sender cannot block the producer indefinitely.
    """

    enabled: bool
    stale_ns: int
    shm: SharedMemory | None = None
    _skip_log_last_ns: int = field(init=False, default=0, repr=False)
    _stale_log_last_ns: int = field(init=False, default=0, repr=False)

    def should_skip_publish(self) -> bool:
        """Hot path: True => caller skips this frame."""
        if not self.enabled:
            return False
        if self.shm is None:
            try:
                self.shm = open_or_create(create=False)
            except FileNotFoundError:
                return False
        try:
            active_count, last_change_ns, _ = read_state(self.shm)
        except (OSError, RuntimeError, struct.error):
            return False
        if active_count <= 0:
            return False
        now_ns = time.monotonic_ns()
        if now_ns - last_change_ns > self.stale_ns:
            if now_ns - self._stale_log_last_ns > 1_000_000_000:
                logger.warning(
                    "[ACTIVATION_BARRIER] stale barrier (count=%d, age=%.1fs) — ignoring",
                    active_count,
                    (now_ns - last_change_ns) / 1e9,
                )
                self._stale_log_last_ns = now_ns
            return False
        with contextlib.suppress(OSError, RuntimeError, struct.error):
            bump_skip(self.shm)
        if now_ns - self._skip_log_last_ns > 1_000_000_000:
            logger.info("[ACTIVATION_BARRIER] skipping publish (active_count=%d)", active_count)
            self._skip_log_last_ns = now_ns
        return True

    def close(self) -> None:
        """Idempotent: close SHM handle if held."""
        if self.shm is not None:
            with contextlib.suppress(OSError, RuntimeError):
                self.shm.close()
            self.shm = None


@dataclass
class HolderBarrier:
    """TD Sender-side holder for the activation-barrier SHM protocol.

    Increments active_count during Sender init, decrements after a settle
    countdown completes, and force-decrements on cleanup.
    """

    enabled: bool
    settle_frames: int
    held: bool = False
    settle_remaining: int = 0
    shm: object = None

    def acquire(self, pid: int, *, log_fn: Callable) -> None:
        """Open-or-create the segment, increment, set held=True. Log+swallow failures."""
        if not self.enabled:
            return
        try:
            if self.shm is None:
                self.shm = open_or_create(create=True)
            count = increment(self.shm, pid)
            self.held = True
            log_fn(f"[ACTIVATION_BARRIER] held +1 (count={count}) for Sender init", force=True)
        except (OSError, RuntimeError, struct.error) as _exc:
            log_fn(f"[ACTIVATION_BARRIER] init increment failed (ignored): {_exc}", force=True)

    def arm_settle_countdown(self) -> None:
        """Called from initialize() tail when init succeeds — settle_remaining = settle_frames."""
        if self.held:
            self.settle_remaining = self.settle_frames

    def tick_and_maybe_release(self, pid: int, *, log_fn: Callable) -> bool:
        """Per-frame: decrement settle_remaining. When it hits 0 and held, release barrier.

        Returns True iff the release fired this frame.
        """
        if self.settle_remaining <= 0:
            return False
        self.settle_remaining -= 1
        if self.settle_remaining == 0 and self.held and self.shm is not None:
            try:
                count = decrement(self.shm, pid)
                log_fn(
                    f"[ACTIVATION_BARRIER] released after {self.settle_frames}-frame settle (count now {count})",
                    force=True,
                )
                return True
            except (OSError, RuntimeError, struct.error) as _exc:
                log_fn(f"[ACTIVATION_BARRIER] settle decrement failed (ignored): {_exc}", force=True)
            finally:
                self.held = False
        return False

    def force_release(self, pid: int, *, log_fn: Callable) -> None:
        """Cleanup-time: if still held, decrement and clear. Idempotent."""
        if not (self.held and self.shm is not None):
            return
        try:
            count = decrement(self.shm, pid)
            log_fn(
                f"[ACTIVATION_BARRIER] released on cleanup (mid-settle, count now {count})",
                force=True,
            )
        except (OSError, RuntimeError, struct.error) as _exc:
            log_fn(f"[ACTIVATION_BARRIER] cleanup decrement failed (ignored): {_exc}", force=True)
        finally:
            self.held = False

    def close(self) -> None:
        """Idempotent: close SHM handle if held."""
        if self.shm is not None:
            with contextlib.suppress(OSError, RuntimeError):
                self.shm.close()
            self.shm = None
