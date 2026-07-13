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
from enum import Enum, auto
from multiprocessing.shared_memory import SharedMemory
from typing import Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

SHM_NAME = "cudalink_activation_barrier"
SHM_SIZE = 64
MAGIC = 0xCDA1BAAA
VERSION = 1

# Struct: magic(u32) version(u32) active_count(u32) pad(u32) last_change_ns(u64)
#         barrier_skips(u32) last_writer_pid(u32) reserved(32s)
_STRUCT = struct.Struct("<IIIIQII32s")

# EQUIVALENT(Eq_LtE/Eq_Is/Eq_GtE): struct size is fixed at 64 == SHM_SIZE
# always, so ==, <=, >= are all identically True here; `is` also holds via
# CPython small-int interning of 64.
assert _STRUCT.size == SHM_SIZE, f"Layout error: struct size {_STRUCT.size} != {SHM_SIZE}"  # pragma: no mutate

# Hot-path struct: reads only the fields returned by read_state, skipping reserved(32s).
# unpack_from() reads directly from the memoryview — no intermediate bytes() copy.
_ST_STATE = struct.Struct("<IIIIQI")  # magic version active_count pad last_change_ns barrier_skips


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
        # EQUIVALENT(NumberReplacer on `* 32`, e.g. `* 31`/`* 33`): struct.pack's
        # `Ns` field silently truncates/zero-pads; for all-zero-byte content the
        # packed 32s output is byte-identical regardless of a 31/32/33 length
        # mismatch (empirically verified). NOTE: this line also carries 3
        # genuine-killable mutants (pad/last_change_ns/last_writer_pid indices)
        # independently verified killed by
        # test_open_or_create_initializes_full_header_to_zero before this
        # pragma was added; the pragma necessarily suppresses all mutants on
        # this line, not just the equivalent ones.
        _STRUCT.pack_into(shm.buf, 0, MAGIC, VERSION, 0, 0, 0, 0, 0, b"\x00" * 32)  # pragma: no mutate
        return shm


def read_state(shm: SharedMemory) -> tuple[int, int, int]:
    """Return (active_count, last_change_ns, barrier_skips).

    Uses _ST_STATE.unpack_from() on the raw memoryview — no intermediate bytes()
    copy per call.  Snapshot semantics are equivalent: reads the same 24 bytes in
    one pass (fields 0-5; skips the 32-byte reserved tail).
    """
    fields = _ST_STATE.unpack_from(shm.buf, 0)
    # (magic, version, active_count, pad, last_change_ns, barrier_skips)
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
# Port + Adapter + Outcome (Checker-side seam)
# ---------------------------------------------------------------------------


class CheckerOutcome(Enum):
    """Result of a single CheckerBarrier.evaluate() call."""

    DISABLED = auto()  # barrier off via ExportPolicy.barrier_enabled=False
    NO_SKIP = auto()  # active_count == 0; safe to publish
    SKIP_ACTIVE = auto()  # Sender mid-activation window
    SKIP_STALE = auto()  # active_count > 0 but past stale_ns; treat as absent
    SHM_ABSENT = auto()  # segment not yet created by any Sender

    @property
    def should_skip(self) -> bool:
        # EQUIVALENT(Is_Eq): CheckerOutcome has no __eq__ override, so Enum
        # singleton identity (`is`) and equality (`==`) are behaviorally
        # identical for its members.
        return self is CheckerOutcome.SKIP_ACTIVE  # pragma: no mutate


@runtime_checkable
class BarrierShmPort(Protocol):
    """Structural interface CheckerBarrier requires from a SHM backend.

    Real:  RealShmAdapter (delegates to module-level SHM-IO functions).
    Test:  FakeShmAdapter (in-process dict; in tests/conftest.py).

    All methods are best-effort: implementations may raise OSError /
    RuntimeError / struct.error; CheckerBarrier swallows them.
    """

    # EQUIVALENT(RemoveDecorator): Protocol stub body is `...`, never
    # executed; runtime_checkable isinstance() only checks attribute-name
    # presence via hasattr(), never decoration.
    @property  # pragma: no mutate
    def is_attached(self) -> bool: ...

    # EQUIVALENT(Mul_Div): Protocol stub body is `...`, never executed;
    # runtime_checkable isinstance() only checks attribute-name presence via
    # hasattr(), never a method's parameter-kind separators.
    def attach(self, *, create: bool) -> None: ...  # pragma: no mutate

    def read_state(self) -> tuple[int, int, int]: ...  # (active_count, last_change_ns, barrier_skips)

    def bump_skip(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class HolderShmPort(Protocol):
    """Structural interface HolderBarrier requires from a SHM backend.

    Real:  RealShmAdapter (also satisfies BarrierShmPort structurally).
    Test:  FakeShmAdapter (in-process dict; in tests/conftest.py).

    All methods are best-effort: callers catch OSError / RuntimeError /
    struct.error.
    """

    def open_and_increment(self, pid: int) -> int:
        """Open-or-create the segment and increment active_count. Returns new count."""
        ...

    def decrement(self, pid: int) -> int:
        """Decrement active_count (clamps at zero). Returns new count."""
        ...

    def close(self) -> None:
        """Close SHM handle if held. Idempotent."""
        ...


@dataclass
class RealShmAdapter:
    """Production adapter — wraps SharedMemory via module-level SHM-IO functions."""

    _shm: SharedMemory | None = field(default=None, repr=False)

    @property
    def is_attached(self) -> bool:
        return self._shm is not None

    def attach(self, *, create: bool) -> None:
        if self._shm is None:
            self._shm = open_or_create(create=create)  # may raise FileNotFoundError

    def read_state(self) -> tuple[int, int, int]:
        if self._shm is None:
            raise RuntimeError("attach() not called")
        return read_state(self._shm)

    def bump_skip(self) -> None:
        if self._shm is not None:
            bump_skip(self._shm)

    def open_and_increment(self, pid: int) -> int:
        """Open-or-create the segment and increment active_count. Returns new count."""
        if self._shm is None:
            self._shm = open_or_create(create=True)
        return increment(self._shm, pid)

    def decrement(self, pid: int) -> int:
        """Decrement active_count (clamps at zero). Returns new count."""
        if self._shm is None:
            raise RuntimeError("open_and_increment() not called")
        return decrement(self._shm, pid)

    def close(self) -> None:
        if self._shm is not None:
            with contextlib.suppress(OSError, RuntimeError):
                self._shm.close()
            self._shm = None


# ---------------------------------------------------------------------------
# Role classes built on the SHM primitives above
# ---------------------------------------------------------------------------


@dataclass
class CheckerBarrier:
    """Producer/exporter-side checker for the activation-barrier SHM protocol.

    Reads active_count from the SHM segment via a BarrierShmPort and returns
    a CheckerOutcome. Lazily attaches the segment on first call; applies a
    stale-timeout so a crashed Sender cannot block the producer indefinitely.
    """

    enabled: bool
    stale_ns: int
    shm: BarrierShmPort = field(default_factory=RealShmAdapter)
    _skip_log_last_ns: int = field(init=False, default=0, repr=False)
    _stale_log_last_ns: int = field(init=False, default=0, repr=False)

    def evaluate(self) -> CheckerOutcome:
        """Return the current barrier state. Hot path."""
        if not self.enabled:
            return CheckerOutcome.DISABLED
        if not self.shm.is_attached:
            try:
                self.shm.attach(create=False)
            except FileNotFoundError:
                return CheckerOutcome.SHM_ABSENT
            except (OSError, RuntimeError, struct.error):
                return CheckerOutcome.SHM_ABSENT
        try:
            active_count, last_change_ns, _ = self.shm.read_state()
        except (OSError, RuntimeError, struct.error):
            return CheckerOutcome.SHM_ABSENT
        if active_count <= 0:
            return CheckerOutcome.NO_SKIP
        now_ns = time.monotonic_ns()
        if now_ns - last_change_ns > self.stale_ns:
            self._log_stale(now_ns, active_count, last_change_ns)
            return CheckerOutcome.SKIP_STALE
        with contextlib.suppress(OSError, RuntimeError, struct.error):
            self.shm.bump_skip()
        self._log_skip(now_ns, active_count)
        return CheckerOutcome.SKIP_ACTIVE

    def should_skip_publish(self) -> bool:
        """Backwards-compatible bool wrapper — True means caller should skip this frame."""
        return self.evaluate().should_skip

    def _log_stale(self, now_ns: int, active_count: int, last_change_ns: int) -> None:
        if now_ns - self._stale_log_last_ns > 1_000_000_000:
            logger.warning(
                "[ACTIVATION_BARRIER] stale barrier (count=%d, age=%.1fs) — ignoring",
                active_count,
                (now_ns - last_change_ns) / 1e9,
            )
            self._stale_log_last_ns = now_ns

    def _log_skip(self, now_ns: int, active_count: int) -> None:
        if now_ns - self._skip_log_last_ns > 1_000_000_000:
            logger.info("[ACTIVATION_BARRIER] skipping publish (active_count=%d)", active_count)
            self._skip_log_last_ns = now_ns

    def close(self) -> None:
        """Idempotent: close SHM handle if held."""
        self.shm.close()


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
    port: HolderShmPort = field(default_factory=RealShmAdapter)

    def acquire(self, pid: int, *, log_fn: Callable) -> None:
        """Open-or-create the segment, increment, set held=True. Log+swallow failures."""
        if not self.enabled:
            return
        try:
            count = self.port.open_and_increment(pid)
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
        # EQUIVALENT(Eq_LtE): line 333's guard (`<= 0: return False`) guarantees
        # settle_remaining is always >= 0 here (decremented by exactly 1 from a
        # value > 0), and for x >= 0, `x == 0` is equivalent to `x <= 0`.
        if self.settle_remaining == 0 and self.held:  # pragma: no mutate
            try:
                count = self.port.decrement(pid)
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
        if not self.held:
            return
        try:
            count = self.port.decrement(pid)
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
        self.port.close()
