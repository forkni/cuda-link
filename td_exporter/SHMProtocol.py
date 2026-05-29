"""
SHM protocol v0.5.0 — canonical source of truth for the CUDA IPC shared-memory layout.

All binary constants, struct codecs, dtype mappings, and publish/acquire ordering
live here. Every module that reads or writes the SHM region must import from here;
never define SHM_HEADER_SIZE or _ST_U32 locally.

Binary layout (total = SHMLayout(num_slots).total_size):
  [0-3]    magic        uint32 LE = PROTOCOL_MAGIC
  [4-11]   version      uint64 LE (monotonic; incremented each sender init)
  [12-15]  num_slots    uint32 LE
  [16-19]  write_idx    uint32 LE (monotonic; 0 = no frames written yet)
  [20 + slot*128 ...]   IPC handles (64B mem + 64B event per slot, N slots)
  [20 + N*128]          shutdown_flag uint8
  [21 + N*128 ...]      metadata 20B (width/height/num_comps/kind/bits/flags/data_size)
  [41 + N*128 ...]      timestamp float64 LE (producer wall-clock time)
"""

from __future__ import annotations

import struct
import threading
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

PROTOCOL_MAGIC: int = 0x43495044  # "CIPD" — protocol validation magic number (v1.0.0)

MAGIC_OFFSET: int = 0
MAGIC_SIZE: int = 4
VERSION_OFFSET: int = 4
VERSION_SIZE: int = 8
NUM_SLOTS_OFFSET: int = 12
NUM_SLOTS_SIZE: int = 4
WRITE_IDX_OFFSET: int = 16
WRITE_IDX_SIZE: int = 4
SHM_HEADER_SIZE: int = 20  # 4B magic + 8B version + 4B num_slots + 4B write_idx

SLOT_SIZE: int = 128  # 64B cudaIpcMemHandle_t + 64B cudaIpcEventHandle_t

SHUTDOWN_FLAG_SIZE: int = 1
METADATA_SIZE: int = 20  # 4B width + 4B height + 4B num_comps + 1B kind + 1B bits + 2B flags + 4B data_size
TIMESTAMP_SIZE: int = 8  # float64 LE producer wall-clock time

# ---------------------------------------------------------------------------
# DtypeCodec constants (cudaChannelFormatKind)
# ---------------------------------------------------------------------------

FORMAT_KIND_SIGNED: int = 0  # cudaChannelFormatKindSigned
FORMAT_KIND_UNSIGNED: int = 1  # cudaChannelFormatKindUnsigned
FORMAT_KIND_FLOAT: int = 2  # cudaChannelFormatKindFloat
FLAGS_BFLOAT16: int = 0x0001  # bit0: bfloat16 (kind=Float, bits=16)


class _DtypeEntry(NamedTuple):
    """Single row of the dtype registry — wire encoding + backend representations."""

    kind: int
    bits: int
    flags: int
    itemsize: int
    typestr: str  # __cuda_array_interface__ typestr, e.g. "<f4"
    numpy_name: str | None  # np.dtype(name) works; None = needs ml_dtypes (bfloat16)
    cupy_name: str | None  # cp.dtype(name) works; None = unsupported (bfloat16)


# dtype string → all backend representations.
# Single authoritative registry.  Adding a dtype is one row here; no other file changes.
_DTYPE_TABLE: dict[str, _DtypeEntry] = {
    "float32": _DtypeEntry(FORMAT_KIND_FLOAT, 32, 0, 4, "<f4", "float32", "float32"),
    "float16": _DtypeEntry(FORMAT_KIND_FLOAT, 16, 0, 2, "<f2", "float16", "float16"),
    "bfloat16": _DtypeEntry(FORMAT_KIND_FLOAT, 16, FLAGS_BFLOAT16, 2, "<u2", None, None),
    "uint8": _DtypeEntry(FORMAT_KIND_UNSIGNED, 8, 0, 1, "|u1", "uint8", "uint8"),
    "uint16": _DtypeEntry(FORMAT_KIND_UNSIGNED, 16, 0, 2, "<u2", "uint16", "uint16"),
    "int8": _DtypeEntry(FORMAT_KIND_SIGNED, 8, 0, 1, "|i1", "int8", "int8"),
    "int16": _DtypeEntry(FORMAT_KIND_SIGNED, 16, 0, 2, "<i2", "int16", "int16"),
}
_DECODE_TABLE: dict[tuple[int, int, int], str] = {(e.kind, e.bits, e.flags): name for name, e in _DTYPE_TABLE.items()}

# ---------------------------------------------------------------------------
# Pre-compiled struct codecs (hot-path, saves ~50-100ns per call)
# ---------------------------------------------------------------------------

_ST_U32 = struct.Struct("<I")  # uint32 LE (write_idx, num_slots, metadata fields)
_ST_U64 = struct.Struct("<Q")  # uint64 LE (version)
_ST_F64 = struct.Struct("<d")  # float64 LE (timestamp)
_ST_BBH = struct.Struct("<BBH")  # uint8 + uint8 + uint16 LE (format_kind, bits_per_comp, flags)

# ---------------------------------------------------------------------------
# Release fence — C3 ordering guarantee
# ---------------------------------------------------------------------------
# On x86/x64 plain stores are TSO-ordered by hardware, but CPython provides no
# compiler-level guarantee between two separate bytearray writes. threading.Lock
# acquire/release issues OS-level memory barriers on all supported platforms,
# providing the needed release semantics.
#
# CPython internals: PyThread_release_lock() calls pthread_mutex_unlock() on
# POSIX and WakeAllConditionVariable() / ReleaseMutex() on Windows — both
# emit a full store-fence before the unlock. The Python language spec does not
# guarantee this, but CPython 3.x documents it via the GIL memory-model notes
# (https://docs.python.org/3/c-api/init.html#thread-state-and-the-global-interpreter-lock).
#
# Cost: ~80 ns — below the noise floor of a single cudaMemcpyAsync call
# (~500 ns on the same machine), so the fence adds no perceptible latency.
#
# PEP 703 (free-threaded Python, 3.13+): without the GIL, the same OS barrier
# semantics still hold for threading.Lock, but the per-thread store buffer
# assumptions change. Re-evaluate if this codebase targets nogil builds.

_fence_lock = threading.Lock()


def _release_fence() -> None:
    with _fence_lock:
        pass


# ---------------------------------------------------------------------------
# SHMLayout — pre-computes all byte offsets for a given num_slots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SHMLayout:
    """Pre-computed byte offsets for a SHM region with num_slots IPC slots."""

    num_slots: int

    def slot_offset(self, i: int) -> int:
        return SHM_HEADER_SIZE + i * SLOT_SIZE

    @property
    def shutdown_offset(self) -> int:
        return SHM_HEADER_SIZE + self.num_slots * SLOT_SIZE

    @property
    def metadata_offset(self) -> int:
        return self.shutdown_offset + SHUTDOWN_FLAG_SIZE

    @property
    def timestamp_offset(self) -> int:
        return self.metadata_offset + METADATA_SIZE

    @property
    def total_size(self) -> int:
        return self.timestamp_offset + TIMESTAMP_SIZE

    def build_buffer(self, *, version: int = 1, write_idx: int = 0) -> bytearray:
        """Allocate a SHM-sized bytearray with the 20-byte header packed in.

        Test factories and out-of-process probes use this; production callers use
        publish_frame / bump_version against an existing mmap buffer.
        """
        buf = bytearray(self.total_size)
        _ST_U32.pack_into(buf, MAGIC_OFFSET, PROTOCOL_MAGIC)
        _ST_U64.pack_into(buf, VERSION_OFFSET, version)
        _ST_U32.pack_into(buf, NUM_SLOTS_OFFSET, self.num_slots)
        _ST_U32.pack_into(buf, WRITE_IDX_OFFSET, write_idx)
        return buf


# ---------------------------------------------------------------------------
# Metadata — typed representation of the 20-byte metadata region
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metadata:
    """Typed representation of the 20-byte metadata region."""

    width: int
    height: int
    num_comps: int
    format_kind: int  # cudaChannelFormatKind
    bits_per_comp: int
    flags: int
    data_size: int

    def pack_into(self, buf: memoryview, layout: SHMLayout) -> None:
        offset = layout.metadata_offset
        _ST_U32.pack_into(buf, offset, self.width)
        _ST_U32.pack_into(buf, offset + 4, self.height)
        _ST_U32.pack_into(buf, offset + 8, self.num_comps)
        _ST_BBH.pack_into(buf, offset + 12, self.format_kind, self.bits_per_comp, self.flags)
        _ST_U32.pack_into(buf, offset + 16, self.data_size)

    @classmethod
    def read_from(cls, buf: memoryview, layout: SHMLayout) -> Metadata:
        offset = layout.metadata_offset
        width = _ST_U32.unpack_from(buf, offset)[0]
        height = _ST_U32.unpack_from(buf, offset + 4)[0]
        num_comps = _ST_U32.unpack_from(buf, offset + 8)[0]
        kind, bits, flags = _ST_BBH.unpack_from(buf, offset + 12)
        data_size = _ST_U32.unpack_from(buf, offset + 16)[0]
        return cls(
            width=width,
            height=height,
            num_comps=num_comps,
            format_kind=kind,
            bits_per_comp=bits,
            flags=flags,
            data_size=data_size,
        )


# ---------------------------------------------------------------------------
# DtypeCodec — encode/decode dtype strings
# ---------------------------------------------------------------------------


class DtypeCodec:
    """Encode/decode dtype strings across all backend representations.

    All dtype knowledge lives in _DTYPE_TABLE above.  Adding a dtype is a
    single-row edit there; no other file needs to change.

    Wire leg:  encode() / decode() / itemsize()
    Backend:   typestr() / numpy_name() / cupy_name()
    """

    @staticmethod
    def encode(dtype: str) -> tuple[int, int, int]:
        """dtype string → (format_kind, bits_per_comp, flags).

        Raises:
            KeyError: if dtype is not in the supported set.
        """
        e = _DTYPE_TABLE[dtype]
        return (e.kind, e.bits, e.flags)

    @staticmethod
    def decode(kind: int, bits: int, flags: int) -> str:
        """(format_kind, bits_per_comp, flags) → dtype string.

        Returns "float32" for unknown triples (forward-compat fallback).
        """
        return _DECODE_TABLE.get((kind, bits, flags), "float32")

    @staticmethod
    def itemsize(dtype: str) -> int:
        """Return the byte width of one element of the given dtype.

        Raises:
            KeyError: if dtype is not in the supported set.
        """
        return _DTYPE_TABLE[dtype].itemsize

    @staticmethod
    def supported() -> tuple[str, ...]:
        """Tuple of all supported dtype strings, in registration order."""
        return tuple(_DTYPE_TABLE)

    @staticmethod
    def typestr(dtype: str) -> str:
        """Return the __cuda_array_interface__ typestr for dtype (e.g. "<f4").

        For bfloat16 this is "<u2" — a uint16 backing view; callers that build
        torch tensors must follow up with tensor.view(torch.bfloat16).

        Raises:
            KeyError: if dtype is not in the supported set.
        """
        return _DTYPE_TABLE[dtype].typestr

    @staticmethod
    def numpy_name(dtype: str) -> str | None:
        """Return the numpy dtype name string for dtype (e.g. "float32").

        Returns None for bfloat16 — the caller must use ml_dtypes.bfloat16
        instead of np.dtype(name).

        Raises:
            KeyError: if dtype is not in the supported set.
        """
        return _DTYPE_TABLE[dtype].numpy_name

    @staticmethod
    def cupy_name(dtype: str) -> str | None:
        """Return the cupy dtype name string for dtype (e.g. "float32").

        Returns None for bfloat16 — CuPy has no bfloat16 dtype; callers should
        raise a clear ValueError rather than proceeding.

        Raises:
            KeyError: if dtype is not in the supported set.
        """
        return _DTYPE_TABLE[dtype].cupy_name


# ---------------------------------------------------------------------------
# Header helpers — read/write the 20-byte header region
# ---------------------------------------------------------------------------


def read_magic(buf: memoryview) -> int:
    return _ST_U32.unpack_from(buf, MAGIC_OFFSET)[0]


def read_version(buf: memoryview) -> int:
    return _ST_U64.unpack_from(buf, VERSION_OFFSET)[0]


def read_num_slots(buf: memoryview) -> int:
    return _ST_U32.unpack_from(buf, NUM_SLOTS_OFFSET)[0]


def read_write_idx(buf: memoryview) -> int:
    return _ST_U32.unpack_from(buf, WRITE_IDX_OFFSET)[0]


def bump_version(buf: memoryview) -> int:
    """Increment the version counter in-place; return the new version."""
    try:
        current = read_version(buf)
    except (struct.error, ValueError, IndexError):
        current = 0
    new_version = current + 1
    _ST_U64.pack_into(buf, VERSION_OFFSET, new_version)
    return new_version


# ---------------------------------------------------------------------------
# publish_frame — the only place that encodes the C3 ordering guarantee
# ---------------------------------------------------------------------------


def publish_frame(buf: memoryview, layout: SHMLayout, write_idx: int, timestamp: float) -> None:
    """Write timestamp, clear shutdown_flag, fence, then publish write_idx LAST.

    Ordering is critical: the consumer reads shutdown_flag BEFORE write_idx.
    Clearing shutdown_flag before incrementing write_idx ensures the consumer
    always sees shutdown_flag=0 when it first observes a new frame.

    Callers must not replicate this sequence outside this function.
    """
    _ST_F64.pack_into(buf, layout.timestamp_offset, timestamp)
    buf[layout.shutdown_offset] = 0
    _release_fence()  # C3 release barrier: shutdown_flag visible before write_idx
    _ST_U32.pack_into(buf, WRITE_IDX_OFFSET, write_idx)


def set_shutdown(buf: memoryview, layout: SHMLayout) -> None:
    """Signal producer exit by writing shutdown_flag = 1.

    Emits a release fence so the flag is visible to consumers before any
    subsequent write_idx update.  Call once during producer teardown.
    """
    buf[layout.shutdown_offset] = 1
    _release_fence()


def clear_shutdown(buf: memoryview, layout: SHMLayout) -> None:
    """Clear shutdown_flag to 0 at init or barrier-skip time.

    No fence required: callers either immediately follow with publish_frame
    (which owns the fence), or make no write_idx advance (barrier-skip path).
    """
    buf[layout.shutdown_offset] = 0


# ---------------------------------------------------------------------------
# acquire_slot — consumer-side frame acquisition
# ---------------------------------------------------------------------------


class SlotState(Enum):
    NO_FRAME = "no_frame"
    NEW_FRAME = "new_frame"
    SHUTDOWN = "shutdown"
    VERSION_CHANGED = "version_changed"


@dataclass
class AcquireResult:
    """Result of acquire_slot()."""

    state: SlotState
    slot: int = -1
    timestamp: float = 0.0
    new_version: int = 0
    write_idx: int = 0


def acquire_slot(
    buf: memoryview,
    layout: SHMLayout,
    last_write_idx: int,
    last_version: int,
) -> AcquireResult:
    """Read SHM state and classify the result for the consumer.

    Returns an AcquireResult with one of four states:
    - NO_FRAME: write_idx unchanged; nothing to consume.
    - NEW_FRAME: new frame at .slot; read and process it, update last_write_idx to .write_idx.
    - SHUTDOWN: shutdown_flag=1; producer has exited, consumer should clean up.
    - VERSION_CHANGED: SHM was re-initialised; consumer must reopen IPC handles.

    Folds _get_read_slot() (importer) and the three identical preambles in
    get_frame / get_frame_numpy / get_frame_cupy into one location.
    """
    if buf[layout.shutdown_offset] != 0:
        return AcquireResult(state=SlotState.SHUTDOWN)

    version = read_version(buf)
    if version != last_version and last_version != 0:
        return AcquireResult(state=SlotState.VERSION_CHANGED, new_version=version)

    write_idx = read_write_idx(buf)
    if write_idx == 0 or write_idx == last_write_idx:
        return AcquireResult(state=SlotState.NO_FRAME)

    slot = (write_idx - 1) % layout.num_slots
    try:
        timestamp = _ST_F64.unpack_from(buf, layout.timestamp_offset)[0]
    except struct.error:
        timestamp = 0.0

    return AcquireResult(state=SlotState.NEW_FRAME, slot=slot, timestamp=timestamp, write_idx=write_idx)
