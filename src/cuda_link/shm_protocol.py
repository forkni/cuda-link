# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""
SHM protocol v0.6.0 — canonical source of truth for the CUDA IPC shared-memory layout.

All binary constants, struct codecs, dtype mappings, and publish/acquire ordering
live here. Every module that reads or writes the SHM region must import from here;
never define SHM_HEADER_SIZE or _ST_U32 locally.

Binary layout (total = SHMLayout(num_slots).total_size = 48 + N*160), all little-endian:
  [0-3]    magic        uint32 = PROTOCOL_MAGIC ("CIP6")
  [4-7]    num_slots    uint32
  [8-15]   version      uint64 (monotonic; incremented each sender init; 8-aligned)
  [16-19]  write_idx    uint32 (monotonic; 0 = no frames written yet)
  [20-23]  sender_pid   uint32 (sender process id; 0 = unknown)
  [24 + i*160 ...]      slot i: 64B mem handle, 64B event handle, 32B slot meta
  [24 + N*160]          shutdown_flag uint8
  [25 + N*160 ...]      metadata 20B (width/height/num_comps/kind/bits/flags/data_size)
  [45 + N*160 ...]      3B pad (total is a multiple of 8)

Slot meta, relative to the slot base (+128 .. +159):
  +128 host_timestamp      float64 (sender perf_counter / QPC seconds at publish)
  +136 producer_frame      uint64  (sender frame number; the cross-stream match key)
  +144 source_timestamp_us uint64  (optional producer time, not interpreted)
  +152 stream_id           uint32  (user-set stream name)
  +156 meta_flags          uint32  (reserved, always 0)

v0.5 (cuda-link 1.x) used magic 0x43495044 ("CIPD") and a different header; it is kept
as PROTOCOL_MAGIC_V05 only so receivers can name the version mismatch.
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

PROTOCOL_MAGIC: int = 0x43495036  # "CIP6" — protocol v0.6 validation magic number
PROTOCOL_MAGIC_V05: int = 0x43495044  # "CIPD" — v0.5 (cuda-link 1.x); only to name the mismatch

MAGIC_OFFSET: int = 0
MAGIC_SIZE: int = 4
NUM_SLOTS_OFFSET: int = 4
NUM_SLOTS_SIZE: int = 4
VERSION_OFFSET: int = 8
VERSION_SIZE: int = 8
WRITE_IDX_OFFSET: int = 16
WRITE_IDX_SIZE: int = 4
SENDER_PID_OFFSET: int = 20
SENDER_PID_SIZE: int = 4
SHM_HEADER_SIZE: int = 24  # 4B magic + 4B num_slots + 8B version + 4B write_idx + 4B sender_pid

IPC_HANDLE_SIZE: int = 64  # cudaIpc{Mem,Event}Handle_t are each 64 bytes
SLOT_META_OFFSET_IN_SLOT: int = 128  # after the two IPC handles
SLOT_META_SIZE: int = 32
SLOT_SIZE: int = 160  # 64B mem handle + 64B event handle + 32B slot meta
MAX_SLOTS: int = 10  # ring size limit: a header num_slots outside 1..MAX_SLOTS is a protocol error

# Slot-meta field offsets, relative to the start of the slot meta.
SLOT_META_HOST_TIMESTAMP_OFFSET: int = 0  # float64
SLOT_META_PRODUCER_FRAME_OFFSET: int = 8  # uint64
SLOT_META_SOURCE_TIMESTAMP_US_OFFSET: int = 16  # uint64
SLOT_META_STREAM_ID_OFFSET: int = 24  # uint32
SLOT_META_FLAGS_OFFSET: int = 28  # uint32, reserved (0)

SHUTDOWN_FLAG_SIZE: int = 1
METADATA_SIZE: int = 20  # 4B width + 4B height + 4B num_comps + 1B kind + 1B bits + 2B flags + 4B data_size
TRAILER_PAD_SIZE: int = 3  # pads the total to a multiple of 8

# ---------------------------------------------------------------------------
# DtypeCodec constants (cudaChannelFormatKind)
# ---------------------------------------------------------------------------

FORMAT_KIND_SIGNED: int = 0  # cudaChannelFormatKindSigned
FORMAT_KIND_UNSIGNED: int = 1  # cudaChannelFormatKindUnsigned
FORMAT_KIND_FLOAT: int = 2  # cudaChannelFormatKindFloat
FLAGS_BFLOAT16: int = 0x0001  # bit0: bfloat16 (kind=Float, bits=16)
FLAGS_MONO_ALPHA: int = 0x0002  # bit1: 2-channel source is mono+alpha, not RG
FLAGS_BGRA: int = 0x0004  # bit2: 4-channel uint8 source is BGRA-ordered, not RGBA


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
_ST_SLOT_META = struct.Struct("<dQQII")  # slot meta: host_ts, producer_frame, source_ts_us, stream_id, flags
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

    def mem_handle_offset(self, slot: int) -> int:
        """Byte offset of the 64-byte cudaIpcMemHandle_t within slot *slot*."""
        return self.slot_offset(slot)

    def event_handle_offset(self, slot: int) -> int:
        """Byte offset of the 64-byte cudaIpcEventHandle_t within slot *slot*."""
        return self.slot_offset(slot) + IPC_HANDLE_SIZE

    def slot_meta_offset(self, slot: int) -> int:
        """Byte offset of the 32-byte slot meta within slot *slot*."""
        return self.slot_offset(slot) + SLOT_META_OFFSET_IN_SLOT

    @property
    def shutdown_offset(self) -> int:
        return SHM_HEADER_SIZE + self.num_slots * SLOT_SIZE

    @property
    def metadata_offset(self) -> int:
        # shutdown_offset (24 + num_slots*160) is a multiple of 8 and
        # SHUTDOWN_FLAG_SIZE (1) is bit 0: + == ^ == | here for any num_slots >= 0.
        return self.shutdown_offset + SHUTDOWN_FLAG_SIZE  # pragma: no mutate

    @property
    def total_size(self) -> int:
        return self.metadata_offset + METADATA_SIZE + TRAILER_PAD_SIZE

    def build_buffer(self, *, version: int = 1, write_idx: int = 0, sender_pid: int = 0) -> bytearray:
        """Allocate a SHM-sized bytearray with the 24-byte header packed in.

        Test factories and out-of-process probes use this; production callers use
        publish_frame / bump_version against an existing mmap buffer.
        """
        buf = bytearray(self.total_size)
        _ST_U32.pack_into(buf, MAGIC_OFFSET, PROTOCOL_MAGIC)
        _ST_U32.pack_into(buf, NUM_SLOTS_OFFSET, self.num_slots)
        _ST_U64.pack_into(buf, VERSION_OFFSET, version)
        _ST_U32.pack_into(buf, WRITE_IDX_OFFSET, write_idx)
        _ST_U32.pack_into(buf, SENDER_PID_OFFSET, sender_pid)
        return buf


# ---------------------------------------------------------------------------
# SlotMeta — the 32-byte per-slot metadata block
# ---------------------------------------------------------------------------


class SlotMeta(NamedTuple):
    """Per-slot metadata the sender writes with each frame (v0.6).

    host_timestamp: sender perf_counter() / QPC seconds at publish.
    producer_frame: sender frame number (TD absolute frame); 0 = not set.
    source_timestamp_us: optional producer-supplied time in µs; not interpreted.
    stream_id: user-set integer naming the stream.
    meta_flags: reserved, always 0.
    """

    host_timestamp: float = 0.0
    producer_frame: int = 0
    source_timestamp_us: int = 0
    stream_id: int = 0
    meta_flags: int = 0


EMPTY_SLOT_META = SlotMeta()


def read_slot_meta(buf: memoryview, layout: SHMLayout, slot: int) -> SlotMeta:
    """Read the slot meta of *slot* (plain read; call after the acquire of write_idx)."""
    return SlotMeta._make(_ST_SLOT_META.unpack_from(buf, layout.slot_meta_offset(slot)))


# TD CHOP samples and Info CHOP values are float32, exact for integers only below 2^24.
# A producer_frame shown in a CHOP channel is producer_frame % INFO_CHOP_FRAME_MOD; compare
# two such values with (a - b) % INFO_CHOP_FRAME_MOD. Twin of the C++ kInfoChopFrameMod.
INFO_CHOP_FRAME_MOD = 1 << 24

# Slot meta channels a receiver shows, in order: the In TOP Info CHOP channels and the
# .tox `slot_meta` Script CHOP use these names, so a B network reads either the same way.
SLOT_META_CHANNELS = ("producer_frame", "stream_id", "latency_us", "frame_gap_count", "producer_skip_count")


class FrameStep(Enum):
    """Step between two frames a receiver showed (twin of cpp_top core/frame_step.h)."""

    FIRST = 0  # no previous frame on this connection (prev_write_idx == 0)
    NEXT = 1  # write_idx step 1 and producer_frame step <= 1 (or not set)
    FRAME_GAP = 2  # write_idx step > 1: the ring overwrote frames before the read
    PRODUCER_SKIP = 3  # write_idx step 1, producer_frame step > 1: the sender skipped frames


def classify_frame_step(
    prev_write_idx: int, prev_producer_frame: int, write_idx: int, producer_frame: int
) -> FrameStep:
    """Classify the step from the previous shown frame to this one.

    producer_frame 0 means "not set" and never counts as a skip; write_idx wraps at 2^32.
    """
    if prev_write_idx == 0:
        return FrameStep.FIRST
    write_step = (write_idx - prev_write_idx) & 0xFFFFFFFF
    if write_step > 1:
        return FrameStep.FRAME_GAP
    if write_step == 1 and prev_producer_frame != 0 and producer_frame > prev_producer_frame + 1:
        return FrameStep.PRODUCER_SKIP
    return FrameStep.NEXT


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

    @property
    def expected_size(self) -> int:
        """Expected frame byte count: width * height * num_comps * (bits_per_comp // 8).

        Used to validate that data_size from the wire matches the geometry.
        bits_per_comp is always a multiple of 8 for all registered dtypes.
        """
        return self.width * self.height * self.num_comps * (self.bits_per_comp // 8)

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
        # _ST_U32 unpacks exactly 1 field; [0] and [-1] are always equal here.
        width = _ST_U32.unpack_from(buf, offset)[0]  # pragma: no mutate
        # _ST_U32 unpacks exactly 1 field; [0] and [-1] are always equal here.
        height = _ST_U32.unpack_from(buf, offset + 4)[0]  # pragma: no mutate
        # _ST_U32 unpacks exactly 1 field; [0] and [-1] are always equal here.
        num_comps = _ST_U32.unpack_from(buf, offset + 8)[0]  # pragma: no mutate
        kind, bits, flags = _ST_BBH.unpack_from(buf, offset + 12)
        # _ST_U32 unpacks exactly 1 field; [0] and [-1] are always equal here.
        data_size = _ST_U32.unpack_from(buf, offset + 16)[0]  # pragma: no mutate
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

    # never called on an instance (DtypeCodec is a pure namespace class); @staticmethod is unobservable here.
    @staticmethod  # pragma: no mutate
    def encode(dtype: str) -> tuple[int, int, int]:
        """dtype string → (format_kind, bits_per_comp, flags).

        Raises:
            KeyError: if dtype is not in the supported set.
        """
        e = _DTYPE_TABLE[dtype]
        return (e.kind, e.bits, e.flags)

    # never called on an instance (DtypeCodec is a pure namespace class); @staticmethod is unobservable here.
    @staticmethod  # pragma: no mutate
    def decode(kind: int, bits: int, flags: int) -> str:
        """(format_kind, bits_per_comp, flags) → dtype string.

        Returns "float32" for unknown triples (forward-compat fallback).

        Only FLAGS_BFLOAT16 is dtype-relevant; FLAGS_MONO_ALPHA (bit1) and
        FLAGS_BGRA (bit2) describe channel semantics (R+A vs R+G; BGRA vs RGBA
        component order) and do NOT change the element dtype or itemsize.
        Masking ensures e.g. (UNSIGNED, 8, FLAGS_MONO_ALPHA | FLAGS_BGRA) →
        "uint8" rather than falling back to "float32" with a 4× oversized itemsize.
        """
        return _DECODE_TABLE.get((kind, bits, flags & FLAGS_BFLOAT16), "float32")

    # never called on an instance (DtypeCodec is a pure namespace class); @staticmethod is unobservable here.
    @staticmethod  # pragma: no mutate
    def itemsize(dtype: str) -> int:
        """Return the byte width of one element of the given dtype.

        Raises:
            KeyError: if dtype is not in the supported set.
        """
        return _DTYPE_TABLE[dtype].itemsize

    # never called on an instance (DtypeCodec is a pure namespace class); @staticmethod is unobservable here.
    @staticmethod  # pragma: no mutate
    def supported() -> tuple[str, ...]:
        """Tuple of all supported dtype strings, in registration order."""
        return tuple(_DTYPE_TABLE)

    # never called on an instance (DtypeCodec is a pure namespace class); @staticmethod is unobservable here.
    @staticmethod  # pragma: no mutate
    def typestr(dtype: str) -> str:
        """Return the __cuda_array_interface__ typestr for dtype (e.g. "<f4").

        For bfloat16 this is "<u2" — a uint16 backing view; callers that build
        torch tensors must follow up with tensor.view(torch.bfloat16).

        Raises:
            KeyError: if dtype is not in the supported set.
        """
        return _DTYPE_TABLE[dtype].typestr

    # never called on an instance (DtypeCodec is a pure namespace class); @staticmethod is unobservable here.
    @staticmethod  # pragma: no mutate
    # lazy-evaluated annotation (from __future__ import annotations); the | is never executed at runtime.
    def numpy_name(dtype: str) -> str | None:  # pragma: no mutate
        """Return the numpy dtype name string for dtype (e.g. "float32").

        Returns None for bfloat16 — the caller must use ml_dtypes.bfloat16
        instead of np.dtype(name).

        Raises:
            KeyError: if dtype is not in the supported set.
        """
        return _DTYPE_TABLE[dtype].numpy_name

    # never called on an instance (DtypeCodec is a pure namespace class); @staticmethod is unobservable here.
    @staticmethod  # pragma: no mutate
    # lazy-evaluated annotation (from __future__ import annotations); the | is never executed at runtime.
    def cupy_name(dtype: str) -> str | None:  # pragma: no mutate
        """Return the cupy dtype name string for dtype (e.g. "float32").

        Returns None for bfloat16 — CuPy has no bfloat16 dtype; callers should
        raise a clear ValueError rather than proceeding.

        Raises:
            KeyError: if dtype is not in the supported set.
        """
        return _DTYPE_TABLE[dtype].cupy_name


# ---------------------------------------------------------------------------
# Header helpers — read/write the 24-byte header region
# ---------------------------------------------------------------------------


def read_magic(buf: memoryview) -> int:
    # _ST_U32 unpacks exactly 1 field; [0] and [-1] are always equal here.
    return _ST_U32.unpack_from(buf, MAGIC_OFFSET)[0]  # pragma: no mutate


def read_version(buf: memoryview) -> int:
    # _ST_U64 unpacks exactly 1 field; [0] and [-1] are always equal here.
    return _ST_U64.unpack_from(buf, VERSION_OFFSET)[0]  # pragma: no mutate


def read_num_slots(buf: memoryview) -> int:
    # _ST_U32 unpacks exactly 1 field; [0] and [-1] are always equal here.
    return _ST_U32.unpack_from(buf, NUM_SLOTS_OFFSET)[0]  # pragma: no mutate


def read_write_idx(buf: memoryview) -> int:
    # _ST_U32 unpacks exactly 1 field; [0] and [-1] are always equal here.
    return _ST_U32.unpack_from(buf, WRITE_IDX_OFFSET)[0]  # pragma: no mutate


def read_sender_pid(buf: memoryview) -> int:
    # _ST_U32 unpacks exactly 1 field; [0] and [-1] are always equal here.
    return _ST_U32.unpack_from(buf, SENDER_PID_OFFSET)[0]  # pragma: no mutate


def write_sender_pid(buf: memoryview, pid: int) -> None:
    """Write the sender process id. Call before bump_version, so the version fence covers it."""
    _ST_U32.pack_into(buf, SENDER_PID_OFFSET, pid & 0xFFFFFFFF)


def is_valid_num_slots(num_slots: int) -> bool:
    """True when ``num_slots`` is a ring size the protocol allows (1..MAX_SLOTS)."""
    return 0 < num_slots <= MAX_SLOTS


def write_header(buf: memoryview, num_slots: int, sender_pid: int) -> None:
    """Write magic, num_slots, write_idx = 0 and the sender process id.

    The version is not written here. The sender calls bump_version last, after the slot
    handles, so a receiver that sees the new version reads a complete header.
    """
    _ST_U32.pack_into(buf, MAGIC_OFFSET, PROTOCOL_MAGIC)
    _ST_U32.pack_into(buf, NUM_SLOTS_OFFSET, num_slots)
    _ST_U32.pack_into(buf, WRITE_IDX_OFFSET, 0)
    write_sender_pid(buf, sender_pid)


_ZERO_SLOT: bytes = bytes(SLOT_SIZE)


def clear_slot(buf: memoryview, layout: SHMLayout, slot: int) -> None:
    """Zero one slot (both IPC handles and the slot meta), so a late receiver cannot open them."""
    base = layout.slot_offset(slot)
    buf[base : base + SLOT_SIZE] = _ZERO_SLOT


SAME_PROCESS_MESSAGE: str = (
    "sender and receiver appear to be in the same process -- CUDA IPC cannot loop back within one "
    "process; run the sender in a separate TD instance or use the Python peer"
)


def is_same_process_sender(sender_pid: int, own_pid: int) -> bool:
    """True when the segment names this process as its sender (0 = unknown: never same)."""
    return sender_pid != 0 and sender_pid == (own_pid & 0xFFFFFFFF)


def protocol_mismatch_message(magic: int) -> str:
    """Return the receiver error text for a segment whose magic is not PROTOCOL_MAGIC."""
    if magic == PROTOCOL_MAGIC_V05:
        return "sender uses protocol v0.5 (cuda-link 1.x); this receiver needs v0.6"
    return (
        f"Protocol magic mismatch: expected 0x{PROTOCOL_MAGIC:08X}, got 0x{magic:08X}. "
        "The SHM name is in use by an unrelated process."
    )


def bump_version(buf: memoryview) -> int:
    """Increment the version counter in-place; return the new version."""
    try:
        current = read_version(buf)
    except (struct.error, ValueError, IndexError):
        current = 0
    new_version = current + 1
    _ST_U64.pack_into(buf, VERSION_OFFSET, new_version)
    return new_version


def set_version(buf: memoryview, version: int) -> None:
    """Write an explicit version value, overriding the counter.

    Used by the sender engine to maintain a monotonic version counter across
    close()+open() cycles: Exporter.close() unlinks the SHM segment, so the
    next open() creates a fresh region where bump_version would reset to 1.
    Calling set_version() immediately after open() injects the engine-held
    monotonic value, guaranteeing the receiver's `version != last_version`
    check fires even when a dtype-change reopen produces a fresh segment.
    """
    _ST_U64.pack_into(buf, VERSION_OFFSET, version)


# ---------------------------------------------------------------------------
# publish_frame — the only place that encodes the C3 ordering guarantee
# ---------------------------------------------------------------------------


def publish_frame(
    buf: memoryview,
    layout: SHMLayout,
    write_idx: int,
    *,
    host_timestamp: float,
    producer_frame: int = 0,
    source_timestamp_us: int = 0,
    stream_id: int = 0,
) -> None:
    """Write the slot meta, clear shutdown_flag, fence, then publish write_idx LAST.

    The frame goes to slot (write_idx - 1) % num_slots, and its slot meta is written
    there before the fence, so a consumer that acquires write_idx sees it.

    Ordering is critical: the consumer reads shutdown_flag BEFORE write_idx.
    Clearing shutdown_flag before incrementing write_idx ensures the consumer
    always sees shutdown_flag=0 when it first observes a new frame.

    Callers must not replicate this sequence outside this function.
    """
    slot = (write_idx - 1) % layout.num_slots
    _ST_SLOT_META.pack_into(
        buf,
        layout.slot_meta_offset(slot),
        host_timestamp,
        producer_frame & 0xFFFFFFFFFFFFFFFF,
        source_timestamp_us & 0xFFFFFFFFFFFFFFFF,
        stream_id & 0xFFFFFFFF,
        0,
    )
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


class AcquireResult(NamedTuple):
    """Result of acquire_slot().

    NamedTuple for lighter per-call construction (no __dict__, no dataclass
    overhead) on the import hot path.  Fields are never mutated after creation.
    """

    state: SlotState
    slot: int = -1
    slot_meta: SlotMeta = EMPTY_SLOT_META
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
    # buf[...] is an unsigned byte in [0,255]; != 0 and > 0 agree here.
    if buf[layout.shutdown_offset] != 0:  # pragma: no mutate
        return AcquireResult(state=SlotState.SHUTDOWN)

    version = read_version(buf)
    if version != last_version and last_version != 0:
        return AcquireResult(state=SlotState.VERSION_CHANGED, new_version=version)

    write_idx = read_write_idx(buf)
    if (
        write_idx == 0  # u32 from read_write_idx is never negative; == 0 and <= 0 agree  # pragma: no mutate
        or write_idx == last_write_idx
    ):
        return AcquireResult(state=SlotState.NO_FRAME)

    slot = (write_idx - 1) % layout.num_slots
    # The slot meta lies before the shutdown byte read above, so it is always in the buffer.
    slot_meta = read_slot_meta(buf, layout, slot)
    return AcquireResult(state=SlotState.NEW_FRAME, slot=slot, slot_meta=slot_meta, write_idx=write_idx)
