# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""
TDReceiver - Receiver engine for CUDAIPCExtension.

Owns all Receiver-mode CUDA IPC resources: SHM attachment, IPC handle opening,
per-frame GPU event sync, and Script TOP copyCUDAMemory calls.

textDAT name: TDReceiver  (must match the importable module name inside the COMP namespace)
"""

from __future__ import annotations

import contextlib
import os
import struct
import time
import traceback
from ctypes import c_void_p
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable

try:
    import numpy
except ImportError:
    numpy = None  # Will be imported at runtime in TD

import Doorbell as _doorbell  # noqa: E402
from CUDAAdapters import CTypesCUDAAdapter  # noqa: E402
from CUDAIPCWrapper import get_cuda_runtime  # noqa: E402
from NVTXShim import pop_range as _nvtx_pop  # noqa: E402
from NVTXShim import push_range as _nvtx_push
from NVTXShim import verbose_range as _nvtx_verbose
from FrameProfile import ReportWindow  # noqa: E402
from Importer import (  # noqa: E402
    BackoffNotice,
    Format,
    HeaderFault,
    SlotEventOpenError,
    _RetryState,
    check_header,
    open_slot,
    read_slot_handle_bytes,
)
from SHMProtocol import (  # noqa: E402
    FLAGS_MONO_ALPHA,
    FORMAT_KIND_FLOAT,
    METADATA_SIZE,
    SAME_PROCESS_MESSAGE,
    SHUTDOWN_FLAG_SIZE,
    EMPTY_SLOT_META,
    INFO_CHOP_FRAME_MOD,
    SLOT_META_CHANNELS,
    FrameStep,
    Metadata,
    SHMLayout,
    SlotMeta,
    SlotState,
    acquire_slot,
    classify_frame_step,
    protocol_mismatch_message,
    read_version,
    read_write_idx,
)
from TDConfig import TDReceiverConfig  # noqa: E402
from TDHost import TDHost  # noqa: E402

# The process id the same-process check compares with the segment's sender_pid; tests that
# run a sender and a receiver in one process on fake CUDA replace it.
_current_pid = os.getpid

# Framewaitms: on a frame start with no new frame, wait this long for the sender's doorbell
# before the frame is a repeat. Same rules as the C++ In TOP (cpp_top/src/core/frame_wait.h,
# CudaLinkInTOP::waitForFreshFrame): default 3 ms, range 0-8, 0 turns the wait off.
FRAME_WAIT_DEFAULT_MS = 3
FRAME_WAIT_MAX_MS = 8
# The doorbell is an auto-reset event that wakes ONE waiter, so a second consumer of the
# same name can take the signal. The wait runs in slices and polls write_idx after each:
# a lost signal costs one slice, not the budget.
_FRAME_WAIT_SLICE_MS = 1


def next_wait_slice_ms(remaining_us: float, slice_ms: int = _FRAME_WAIT_SLICE_MS) -> int:
    """Timeout (ms) of the next doorbell wait; 0 when the budget is spent.

    Rounds up, so a remainder under 1 ms still waits (port of next_wait_slice_ms in
    cpp_top/src/core/frame_wait.h).
    """
    if remaining_us <= 0 or slice_ms <= 0:
        return 0
    return min(slice_ms, -(-int(remaining_us) // 1000) or 1)


# Framewaitms guard: the sender's publish cadence, from the slot meta host_timestamp of each
# shown frame (perf_counter seconds; QueryPerformanceCounter, shared by all processes on a
# Windows machine). Skips a wait that cannot get a frame: a sender slower than this receiver
# (a 30 fps camera into a 60 fps project) or a paused sender. Port of PublishCadence /
# cadence_add / cadence_wait_useful in cpp_top/src/core/frame_wait.h: same rules, same numbers.
CADENCE_MIN_SAMPLES = 4  # samples before the guard trusts the period
CADENCE_MAX_OUTLIERS = 4  # consecutive samples over 3 periods: the rate changed, start over
CADENCE_MAX_PERIOD_S = 1.0  # a gap of 1 s or more is a pause, never a sample
CADENCE_SLACK_S = 0.001  # one wait slice of slack on "next publish within the budget"


@dataclass
class PublishCadence:
    """Smoothed seconds per published frame, and the timestamp of the newest frame shown."""

    last_ts: float = 0.0
    period_s: float = 0.0
    samples: int = 0
    outliers: int = 0


def cadence_add(c: PublishCadence, ts: float, idx_step: int) -> None:
    """Add the shown frame with host_timestamp ts, idx_step ring writes after the one before.

    ts <= 0 (a sender that writes no timestamp) leaves the cadence unknown.
    """
    if ts <= 0.0:
        return
    if c.last_ts > 0.0 and ts > c.last_ts and idx_step > 0:
        dt = (ts - c.last_ts) / idx_step
        if dt < CADENCE_MAX_PERIOD_S:
            if c.samples >= CADENCE_MIN_SAMPLES and dt > 3.0 * c.period_s:
                c.outliers += 1
                if c.outliers >= CADENCE_MAX_OUTLIERS:
                    c.samples = 0
                    c.outliers = 0
            else:
                c.period_s = dt if c.samples == 0 else c.period_s + (dt - c.period_s) / 8.0
                c.samples = min(c.samples + 1, CADENCE_MIN_SAMPLES)
                c.outliers = 0
    c.last_ts = ts


def cadence_wait_useful(c: PublishCadence, now_s: float, wait_ms: int) -> bool:
    """True when a wait of wait_ms can get a frame; an unknown cadence returns True.

    False when the next publish is not expected within the budget plus one slice, or the
    newest frame is older than two periods (a paused or stalled sender).
    """
    if c.samples < CADENCE_MIN_SAMPLES or c.period_s <= 0.0 or c.last_ts <= 0.0:
        return True
    since = now_s - c.last_ts
    if since > 2.0 * c.period_s:
        return False
    ahead = c.period_s - since  # time to the next expected publish
    return ahead <= wait_ms / 1000.0 + CADENCE_SLACK_S


# Pre-built NVTX range name strings — eliminates per-frame f-string allocation when NVTX is enabled.
_NVTX_RECEIVER_SLOT_NAMES: tuple[str, ...] = tuple(f"cudalink.receiver.import_frame.slot{i}" for i in range(10))


def td_format_string(fmt: Format) -> str:
    """Map a Format value object → TouchDesigner Script TOP par.format string.

    Produces strings matching TD's internal par.format menu values, e.g.:
      (FLOAT, 32, 4, flags=0)               → "rgba32float"
      (UNSIGNED, 16, 4, flags=0)            → "rgba16fixed"
      (UNSIGNED, 8, 1, flags=0)             → "r8fixed"
      (FLOAT, 32, 2, flags=FLAGS_MONO_ALPHA) → "monoalpha32float"
      (FLOAT, 32, 2, flags=0)               → "rg32float"

    TD uses these to allocate the Script TOP's output texture. Setting
    par.format before copyCUDAMemory ensures the texture is the right size
    for the incoming dtype, preventing partial writes into an oversized buffer.

    FLAGS_MONO_ALPHA is preserved in fmt.flags so that mono+alpha sources are
    correctly labelled in the par.format string and in Info/Status messages —
    the user's only signal that the channel layout is R+A (not R+G).
    """
    format_kind, bits_per_comp, num_comps, flags = fmt.kind, fmt.bits, fmt.num_comps, fmt.flags
    if num_comps == 2 and flags & FLAGS_MONO_ALPHA:
        ch = "monoalpha"
    else:
        ch = {1: "r", 2: "rg", 3: "rgb", 4: "rgba"}.get(num_comps, "rgba")
    suffix = "float" if format_kind == FORMAT_KIND_FLOAT else "fixed"
    return f"{ch}{bits_per_comp}{suffix}"


# CuPy import deferred (heavy; only needed for float16 receiver path)
CUPY_AVAILABLE: bool = False
cp = None


# ---------------------------------------------------------------------------
# Value objects — extract the _rx_* bag into typed containers
# ---------------------------------------------------------------------------


@dataclass
class ReceiverConnection:
    """Holds all CUDA IPC + SHM handles for one active receiver session.

    Created by initialize_receiver(); torn down by close(). close() is idempotent.
    """

    shm_handle: object = None  # SharedMemory | None
    dev_ptrs: list = field(default_factory=list)
    ipc_handles: list = field(default_factory=list)
    ipc_events: list = field(default_factory=list)
    stream: object = None
    layout: object = None  # SHMLayout | None
    num_slots: int = 0
    ipc_version: int = 0
    shutdown_offset: int = 0
    last_write_idx: int = 0  # per-frame protocol cursor; mutates inside import_frame

    def is_open(self) -> bool:
        return self.shm_handle is not None and bool(self.dev_ptrs)

    def close(self, cuda: object, log_fn: Callable) -> None:
        """Idempotent teardown — safe to call multiple times.

        Consolidates cleanup() L745-789: mem handles → events → stream → SHM, in order.
        """
        _t0 = time.perf_counter()
        _close_ms = _events_ms = _stream_ms = 0.0

        if cuda and self.dev_ptrs:
            _ct0 = time.perf_counter()
            for slot, dev_ptr in enumerate(self.dev_ptrs):
                if dev_ptr:
                    _st0 = time.perf_counter()
                    try:
                        cuda.ipc_close_mem_handle(dev_ptr)
                        log_fn(f"Closed IPC handle for slot {slot} ({(time.perf_counter() - _st0) * 1000:.1f} ms)")
                    except (RuntimeError, OSError) as e:
                        log_fn(f"Error closing IPC handle for slot {slot}: {e}", force=True)
            _close_ms = (time.perf_counter() - _ct0) * 1000.0

        if cuda and self.ipc_events:
            _et0 = time.perf_counter()
            for slot, event in enumerate(self.ipc_events):
                if event:
                    _st0 = time.perf_counter()
                    try:
                        cuda.destroy_event(event)
                        log_fn(f"Destroyed IPC event for slot {slot} ({(time.perf_counter() - _st0) * 1000:.1f} ms)")
                    except (RuntimeError, OSError) as e:
                        log_fn(f"Error destroying event for slot {slot}: {e}", force=True)
            _events_ms = (time.perf_counter() - _et0) * 1000.0

        if cuda and self.stream:
            _st0 = time.perf_counter()
            try:
                cuda.destroy_stream(self.stream)
                _stream_ms = (time.perf_counter() - _st0) * 1000.0
                log_fn(f"Destroyed receiver stream ({_stream_ms:.1f} ms)", force=True)
            except (RuntimeError, OSError) as e:
                log_fn(f"Error destroying receiver stream: {e}", force=True)

        if self.shm_handle is not None:
            try:
                self.shm_handle.close()
            except (OSError, BufferError) as e:
                log_fn(f"Error closing SharedMemory: {e}", force=True)

        self.dev_ptrs = []
        self.ipc_handles = []
        self.ipc_events = []
        self.stream = None
        self.shm_handle = None
        self.num_slots = 0

        _total_ms = (time.perf_counter() - _t0) * 1000.0
        log_fn(
            f"Receiver cleanup complete (total {_total_ms:.1f} ms, "
            f"bypass 0.0 ms, ipc_close {_close_ms:.1f} ms, "
            f"events {_events_ms:.1f} ms, stream {_stream_ms:.1f} ms)",
            force=True,
        )


# The connect retry and backoff state is the Importer's: one set of rules for both receivers.
RetryState = _RetryState


@dataclass
class PendingUpdates:
    """Script TOP updates the engine owes after a connect or a format change.

    They are applied outside the cook that found them (update_receiver_resolution /
    update_receiver_format), or taken by script_top_callbacks through the consume methods.
    """

    resolution: bool = False
    format: bool = False  # set when bits_per_comp/format_kind/num_comps changes

    def consume_resolution(self) -> bool:
        """Return True and clear the flag if a resolution update is pending."""
        if self.resolution:
            self.resolution = False
            return True
        return False

    def consume_format(self) -> bool:
        """Return True and clear the flag if a pixel-format update is pending."""
        if self.format:
            self.format = False
            return True
        return False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TDReceiverEngine:
    """Receiver-mode engine: owns all GPU/SHM resources for the Receiver path.

    Constructed by the CUDAIPCExtension facade and replaced (not mutated) on
    mode switches - guaranteeing zero state leak between Sender and Receiver.
    """

    def __init__(
        self,
        host: TDHost,
        config: TDReceiverConfig,
        cuda: Any,
        log_fn: Callable,
        num_slots: int,
        device: int,
        shm_name: str,
        verbose: bool,
    ) -> None:
        self._host = host
        self._config = config
        self.cuda = cuda
        self._log = log_fn
        self.num_slots = num_slots
        self.device = device
        self.shm_name = shm_name
        self.verbose_performance = verbose

        self._initialized = False

        self._connection = ReceiverConnection()
        self._format = Format.from_overrides((0, 0, 0), "float32")  # zero-value until connected
        self._last_fmt_status: str = ""  # cached set_info_status string; rebuilt on format change
        self._retry = RetryState()
        self._pending = PendingUpdates()

        # Engine-private F16 conversion scratch (mutable per-format caches; not value objects)
        self._f16_cpu_buf = None
        self._f32_cpu_buf = None
        self._f16_pinned_ptr = None
        self._f16_host_registered = False  # True only when pageable fallback was page-locked via cudaHostRegister
        self._cupy_f32_buf = None
        self._cupy_f16_views: list = []
        self._cached_shape = None

        self._diag_frames_since_reinit: int = 0

        # frame_count mirrored from sender SHM - exposed for get_stats()
        self.frame_count = 0

        # Slot meta of the frame on the output (set after a successful copy), plus the
        # session-lifetime gap counters the In TOP shows as frame_gap_count /
        # producer_skip_count.
        self.last_slot_meta: SlotMeta = EMPTY_SLOT_META
        # write_idx of the last frame that reached the output; 0 = none on this connection.
        # Not _connection.last_write_idx: that one also advances for a frame the copy drops.
        self._shown_write_idx: int = 0
        self.last_latency_us: float = 0.0
        self.frame_gap_count: int = 0
        self.producer_skip_count: int = 0

        # Framewaitms doorbell wait: handle opened on the first waiting frame start (None
        # until the sender has made the event), and the frames the wait got.
        self._doorbell_handle: object = None
        self.frame_wait_rescued_count: int = 0
        # Guard: publish cadence from _record_slot_meta(), and the no-frame frame starts
        # that skipped the wait because it could not get a frame.
        self.publish_cadence: PublishCadence = PublishCadence()
        self.frame_wait_skipped_count: int = 0

        # Per-frame performance tracking for Debug summary line — shares
        # ReportWindow with TDSenderEngine.
        self._rx_report_every: int = config.report_every
        self._rx_window = ReportWindow()

    # --- Facade-compat property wrappers (keep CUDAIPCExtension getattr calls working) ---

    @property
    def shm_handle(self) -> object:
        return self._connection.shm_handle

    @shm_handle.setter
    def shm_handle(self, value: object) -> None:
        self._connection.shm_handle = value

    @property
    def dev_ptrs(self) -> list:
        return self._connection.dev_ptrs

    @property
    def ipc_handles(self) -> list:
        return self._connection.ipc_handles

    @property
    def write_idx(self) -> int:
        return self._connection.last_write_idx

    # --- Engine verbs (replace facade-poke patterns) ---

    def request_immediate_reconnect(self) -> None:
        """Force next import_frame to attempt reconnection.

        Called from parexecute_callbacks after IPC name or slot-count changes.
        """
        self._retry.request_immediate_reconnect()

    def consume_pending_resolution(self) -> tuple | None:
        """Return (width, height) if resolution sync is pending, else None (and clear flag).

        Called from script_top_callbacks.onCook to drive ImportBuffer Script TOP par updates.
        """
        if self._pending.consume_resolution():
            return (self._format.width, self._format.height)
        return None

    def consume_pending_format(self) -> str | None:
        """Return the TD par.format string if a pixel-format update is pending, else None.

        Called from script_top_callbacks.onCook to drive ImportBuffer par.format updates in
        the fallback path (modoutsidecook disabled). Mirrors consume_pending_resolution.

        Returns a string like 'rgba16fixed', 'r32float', 'rgba8fixed', etc., or None.
        """
        if self._pending.consume_format():
            return td_format_string(self._format)
        return None

    # --- Core API ---

    def is_ready(self) -> bool:
        """True when initialized and all GPU buffer slots are open."""
        return (
            self._initialized
            and bool(self._connection.dev_ptrs)
            and all(ptr is not None for ptr in self._connection.dev_ptrs)
        )

    def has_new_frame(self) -> bool:
        """Return True if the sender has new data or an engine action is needed.

        Called from onFrameStart as a cheap SHM read before import_buffer.cook()
        to skip idle cooks when the sender has not written a new frame.

        Returns True in all cases where import_frame must run: init/retry,
        version change, shutdown signal, or a new write_idx.

        Returns False ONLY when all of these hold:
          - receiver is fully initialized
          - SHM is accessible
          - write_idx equals our cursor (no new frame since last import)
          - version field is unchanged (no sender restart)
          - shutdown flag is 0
        """
        if not self._initialized:
            return True  # retry/init path must run
        shm = self._connection.shm_handle
        if shm is None:
            return True  # can't check; let import_frame decide
        try:
            buf = shm.buf
            # Header reads via SHMProtocol's precompiled-Struct helpers.
            if read_write_idx(buf) != self._connection.last_write_idx:
                return True  # new frame available
            if read_version(buf) != self._connection.ipc_version:
                return True  # sender restarted
            # Check for sender shutdown signal; return False iff truly idle
            return buf[self._connection.shutdown_offset] != 0
        except (AttributeError, struct.error, ValueError, IndexError, OSError):
            return True  # SHM unreadable; let import_frame handle it

    def frame_wait_ms(self) -> int:
        """Framewaitms par as a wait budget in ms, clamped to 0..FRAME_WAIT_MAX_MS.

        A COMP without the par (a .tox built before it existed) gets the default.
        """
        value = self._host.param_value("Framewaitms")
        if value is None:
            return FRAME_WAIT_DEFAULT_MS
        try:
            wait_ms = int(value)
        except (TypeError, ValueError):
            return FRAME_WAIT_DEFAULT_MS
        return min(max(wait_ms, 0), FRAME_WAIT_MAX_MS)

    def wait_for_new_frame(self, wait_ms: int | None = None) -> bool:
        """Wait up to wait_ms for the sender's doorbell; True when has_new_frame() turned True.

        Called from onFrameStart after has_new_frame() returned False. Sender and receiver
        cook at the same rate with a slowly drifting phase; when the publish lands just
        after this poll, jitter makes repeat/skip pairs. The wait turns those repeats into
        new frames that come a little later. It blocks TD's main thread for at most the
        budget plus one slice. Returns False at once when the wait is off, the receiver is
        not connected, or the sender made no doorbell (doorbell=False in its policy, a 1.x sender,
        non-Windows): the classic poll-once behaviour.
        """
        if wait_ms is None:
            wait_ms = self.frame_wait_ms()
        if wait_ms <= 0 or not self._initialized or self._connection.shm_handle is None:
            return False
        if not cadence_wait_useful(self.publish_cadence, time.perf_counter(), wait_ms):
            # Next publish not due within the budget (slower sender) or sender paused.
            self.frame_wait_skipped_count += 1
            return False
        if self._doorbell_handle is None:
            # Retried on each waiting frame start until the sender has made the event.
            self._doorbell_handle = _doorbell.open_doorbell(_doorbell.doorbell_event_name(self.shm_name))
            if self._doorbell_handle is None:
                return False

        deadline = time.perf_counter() + wait_ms / 1000.0
        # Iteration cap: _doorbell.wait() returns False for a timeout and for a failed wait,
        # and a failed wait returns at once. Without the cap a dead handle would spin here.
        for _ in range(2 * wait_ms + 2):
            slice_ms = next_wait_slice_ms((deadline - time.perf_counter()) * 1e6)
            if slice_ms == 0:
                break
            # A wake can be a stale signal from a publish already read, and a timeout can
            # hide a signal another consumer took: poll after both.
            _doorbell.wait(self._doorbell_handle, slice_ms)
            if self.has_new_frame():
                self.frame_wait_rescued_count += 1
                return True
        return False

    def _close_doorbell(self) -> None:
        """Close the Framewaitms doorbell handle (idempotent)."""
        if self._doorbell_handle is not None:
            _doorbell.close(self._doorbell_handle)
            self._doorbell_handle = None

    def _record_slot_meta(self, prev_write_idx: int, write_idx: int, meta: SlotMeta) -> None:
        """Store the shown frame's slot meta and count gaps (same rules as the C++ In TOP)."""
        step = classify_frame_step(prev_write_idx, self.last_slot_meta.producer_frame, write_idx, meta.producer_frame)
        if step is FrameStep.FRAME_GAP:
            self.frame_gap_count += 1
        elif step is FrameStep.PRODUCER_SKIP:
            self.producer_skip_count += 1
        self.last_slot_meta = meta
        ts = meta.host_timestamp
        self.last_latency_us = (time.perf_counter() - ts) * 1e6 if ts > 0 else 0.0
        # A new connection or producer session (prev_write_idx 0) starts the cadence over.
        if prev_write_idx == 0:
            self.publish_cadence = PublishCadence()
        cadence_add(self.publish_cadence, ts, (write_idx - prev_write_idx) & 0xFFFFFFFF)

    def slot_meta_channels(self) -> dict[str, float]:
        """Slot meta of the frame on the output, as the `slot_meta` Script CHOP channels.

        Same names and meaning as the C++ In TOP Info CHOP: producer_frame is shown mod
        2^24 (exact in float32); ``last_slot_meta.producer_frame`` has the full value.
        """
        meta = self.last_slot_meta
        # No zip(strict=True): it needs Python 3.10 and this file runs on 3.9. The test
        # suite checks that the result has exactly SLOT_META_CHANNELS, in order.
        values = (
            float(meta.producer_frame % INFO_CHOP_FRAME_MOD),
            float(meta.stream_id),
            float(self.last_latency_us),
            float(self.frame_gap_count),
            float(self.producer_skip_count),
        )
        return dict(zip(SLOT_META_CHANNELS, values))

    def get_stats(self) -> dict:
        """Receiver statistics dict."""
        return {
            "mode": "Receiver",
            "initialized": self._initialized,
            "frame_count": self.frame_count,
            "shm_name": self.shm_name,
            "num_slots": self.num_slots,
            "rx_resolution": (
                f"{self._format.width}x{self._format.height}x{self._format.num_comps}"
                if self._format.width > 0
                else "N/A"
            ),
            "rx_buffer_size_mb": self._format.frame_nbytes / 1024 / 1024 if self._format.frame_nbytes > 0 else 0,
            "rx_last_write_idx": self._connection.last_write_idx,
            "rx_frame_wait_rescued": self.frame_wait_rescued_count,
            "rx_frame_wait_skipped": self.frame_wait_skipped_count,
            "rx_dev_ptrs": [f"0x{ptr.value:016x}" if ptr else "NULL" for ptr in self._connection.dev_ptrs],
        }

    def import_frame(self, handle: object) -> bool:
        """Import frame from CUDA IPC into ImportBuffer (Script TOP).

        Can be called from:
        - Inside ImportBuffer's onCook callback (TD 2023+ compatibility)
        - Execute DAT onFrameStart with modoutsidecook enabled (TD 2025+)

        Args:
            handle: TOPHandle wrapping the ImportBuffer Script TOP (wrapped by facade)

        Returns:
            True if import successful, False otherwise.
        """
        # Check Active parameter (hot path via TDHost.is_active())
        if not self._host.is_active():
            return False

        # Lazy initialization with exponential backoff retry logic
        if not self._initialized:
            if not self._retry.tick():
                return False  # Wait before retrying

            if not self.initialize_receiver():
                notice = self._retry.back_off()
                if notice is BackoffNotice.WAITING:
                    self._log(
                        f"Waiting for sender... (attempt {self._retry.connect_attempts}, "
                        f"next retry in {self._retry.retry_interval_frames} frames)"
                    )
                elif notice is BackoffNotice.NOW_SILENT:
                    self._log("Sender not found. Will keep retrying silently.", force=True)
                return False

        _nvtx_push(
            _NVTX_RECEIVER_SLOT_NAMES[self._connection.last_write_idx % max(self._connection.num_slots, 1)],
            "blue",
        )
        try:
            result = acquire_slot(
                self._connection.shm_handle.buf,
                self._connection.layout,
                self._connection.last_write_idx,
                self._connection.ipc_version,
            )
            if result.state is SlotState.SHUTDOWN:
                self._log("Sender shutdown detected. Cleaning up.", force=True)
                self.cleanup()
                return False
            if result.state is SlotState.VERSION_CHANGED:
                self._log(
                    f"Sender updated (v{self._connection.ipc_version} -> v{result.new_version}). Refreshing in-place...",
                    force=True,
                )
                if not self._refresh_on_version_change(result.new_version):
                    self._log("In-place refresh failed — falling back to full reinit.", force=True)
                    self.cleanup()
                return False  # No frame to consume this tick regardless of refresh outcome
            if result.state is SlotState.NO_FRAME:
                return False

            if self._connection.last_write_idx == 0:
                # New connection or sender re-init: no previous shown frame to step from.
                self._shown_write_idx = 0
            self._connection.last_write_idx = result.write_idx
            write_idx = result.write_idx
            read_slot = result.slot
            _ts = result.slot_meta.host_timestamp  # producer perf_counter() at publish, for latency
            if not self._rx_window.started:
                # Session baseline for windowed FPS (frame_count is never reset).
                self._rx_window.start(time.perf_counter(), self.frame_count)

            _diag = self.verbose_performance and self._diag_frames_since_reinit < 5
            _t_event = _t_copy = 0.0  # pre-init for static analyzers; only read when _diag is True
            if _diag:
                self._diag_frames_since_reinit += 1
                _t_event = time.perf_counter()

            # Wait on IPC event for this slot (stream-ordered, non-blocking to CPU)
            with _nvtx_verbose("cudalink.receiver.event_wait", "blue"):
                if self._connection.ipc_events[read_slot]:
                    self.cuda.stream_wait_event(
                        self._connection.stream,
                        self._connection.ipc_events[read_slot],
                        0,
                    )
                else:
                    # Fallback when no IPC event: drain the stream now.
                    # Note: float16 path will call stream_synchronize again below, but
                    # synchronizing an already-idle stream is a no-op in CUDA.
                    self.cuda.stream_synchronize(self._connection.stream)

            if _diag:
                _event_ms = (time.perf_counter() - _t_event) * 1000.0
                _t_copy = time.perf_counter()

            _t_copy_wall = time.perf_counter()  # for running copy-time average (all frames)
            # Copy CUDA memory into ImportBuffer texture using cached shape
            address = self._connection.dev_ptrs[read_slot].value

            if self._format.dtype_str == "float16":
                if CUPY_AVAILABLE and self._cupy_f32_buf is not None:
                    # GPU-side float16→float32 conversion (Ch5: minimize PCIe traffic).
                    # stream_wait_event (enqueued above on _connection.stream) guarantees GPU data is ready.
                    # We create a zero-copy CuPy view of the IPC pointer, run an elementwise
                    # f16→f32 cast entirely on GPU via ExternalStream, then call copyCUDAMemory —
                    # eliminating two PCIe roundtrips and the CPU numpy.copyto call.
                    rx_stream_int = int(self._connection.stream.value)
                    f16_size = self._format.frame_nbytes  # original float16 byte count
                    f32_size = f16_size * 2  # float32 = 2× bytes

                    cupy_f16 = self._cupy_f16_views[read_slot]
                    try:
                        # Run conversion on _connection.stream so copyCUDAMemory (also on
                        # _connection.stream) automatically serializes after the elementwise cast kernel.
                        with cp.cuda.ExternalStream(rx_stream_int):
                            cp.copyto(self._cupy_f32_buf, cupy_f16, casting="same_kind")

                        handle.copy_cuda_memory(
                            self._cupy_f32_buf.data.ptr,
                            f32_size,
                            self._cached_shape,  # dataType=float32 set during initialize_receiver()
                            stream=rx_stream_int,
                        )
                    except Exception as _e:  # noqa: BLE001  # CuPy OutOfMemoryError is not MemoryError/RuntimeError (see :944) — keep broad so a GPU OOM here degrades this frame instead of escaping into onCook
                        self._log(f"float16 GPU conversion failed ({_e}); dropping frame", force=True)
                        return False
                else:
                    # CPU fallback: D2H + numpy convert + copyNumpyArray.
                    # Used when CuPy is not installed or GPU buffer allocation failed.
                    if self._f16_cpu_buf is None or self._f32_cpu_buf is None:
                        self._log("[CUDAIPCLink] float16 CPU buffers not allocated — skipping frame", force=True)
                        return False

                    # D2H on _connection.stream: stream_wait_event (enqueued earlier) guarantees data is ready.
                    cpu_ptr = self._f16_cpu_buf.ctypes.data_as(c_void_p)
                    self.cuda.memcpy_async(
                        cpu_ptr, c_void_p(address), self._format.frame_nbytes, 2, self._connection.stream
                    )
                    self.cuda.stream_synchronize(self._connection.stream)
                    numpy.copyto(
                        self._f32_cpu_buf,
                        self._f16_cpu_buf.reshape(self._format.height, self._format.width, self._format.num_comps),
                        casting="same_kind",
                    )
                    handle.copy_numpy_array(self._f32_cpu_buf)
            else:
                handle.copy_cuda_memory(
                    address,
                    self._format.frame_nbytes,
                    self._cached_shape,
                    stream=int(self._connection.stream.value),
                )

            self._rx_window.add_sample(time.perf_counter() - _t_copy_wall)

            if _diag:
                _copy_ms = (time.perf_counter() - _t_copy) * 1000.0
                self._log(
                    f"[DIAG] import_frame #{self._diag_frames_since_reinit}: "
                    f"slot={read_slot} write_idx={write_idx} addr=0x{address:x} "
                    f"stream_wait={_event_ms:.2f}ms copyCUDAMemory={_copy_ms:.2f}ms",
                )

            self.frame_count += 1
            self._connection.last_write_idx = write_idx
            self._record_slot_meta(self._shown_write_idx, write_idx, result.slot_meta)
            self._shown_write_idx = write_idx

            if not self._last_fmt_status:
                # Rebuild only when format changed (initialize_receiver and update_receiver_format
                # reset this to "" so the first frame after a format change always rebuilds).
                _fmt_name = td_format_string(self._format)
                self._last_fmt_status = f"{self._format.width}x{self._format.height} {_fmt_name}"
            self._host.set_info_status(self._last_fmt_status)

            # Debug summary line — matches standalone Python receiver format
            if self.verbose_performance and self.frame_count % self._rx_report_every == 0:
                _now = time.perf_counter()
                _fps = self._rx_window.fps(_now, self.frame_count)
                _latency_ms = (_now - _ts) * 1000.0 if _ts > 0 else 0.0
                _avg_copy_us = self._rx_window.avg_and_reset() * 1e6
                self._log(
                    f"Frame {self.frame_count:5d} | {_fps:5.1f} FPS | "
                    f"shape=({self._format.height}, {self._format.width}, {self._format.num_comps}) "
                    f"dtype={self._format.dtype_str} | "
                    f"latency={_latency_ms:.2f} ms | copy={_avg_copy_us:.1f} µs avg "
                    f"(slot={read_slot}, write_idx={write_idx})"
                )

            return True

        except (RuntimeError, OSError) as e:
            self._log(f"Import failed: {e}", force=True)

            traceback.print_exc()
            return False
        finally:
            _nvtx_pop()

    def update_receiver_resolution(self, handle: object) -> bool:
        """Update ImportBuffer resolution from outside the cook cycle.

        Safe to call from Execute DAT when modoutsidecook is enabled on the Script TOP (TD 2025+).
        When modoutsidecook is NOT available, this is a no-op (resolution handled in onCook).

        Args:
            handle: TOPHandle wrapping the ImportBuffer Script TOP (wrapped by facade)

        Returns:
            True if resolution was updated, False if no update needed or not applicable
        """
        if not self._pending.resolution:
            return False

        try:
            handle.set_resolution(self._format.width, self._format.height)
            self._pending.resolution = False
            self._log(
                f"Set ImportBuffer resolution to {self._format.width}x{self._format.height} (from Execute DAT)",
                force=True,
            )
            return True
        except (AttributeError, RuntimeError) as e:
            self._log(f"Could not set ImportBuffer resolution: {e}", force=True)
            return False

    def update_receiver_format(self, handle: object) -> bool:
        """Update ImportBuffer pixel format from outside the cook cycle.

        Safe to call from Execute DAT (alongside update_receiver_resolution).
        Sets par.format on the Script TOP so copyCUDAMemory allocates an output
        texture of the correct pixel depth — preventing partial writes when dtype
        changes (e.g. float32→uint16 would only fill half a float32-sized texture).

        Args:
            handle: TOPHandle wrapping the ImportBuffer Script TOP.

        Returns:
            True if the format was updated, False if no update needed or not applicable.
        """
        if not self._pending.format:
            return False
        try:
            fmt = td_format_string(self._format)
            handle.set_format(fmt)
            self._pending.format = False
            self._log(f"Set ImportBuffer pixel format to {fmt!r}", force=True)
            return True
        except (AttributeError, RuntimeError) as e:
            self._log(f"Could not set ImportBuffer pixel format: {e}", force=True)
            return False

    def initialize_receiver(self) -> bool:
        """Initialize receiver: open SharedMemory, read handles, open IPC handles.

        Returns:
            True if initialization successful, False otherwise.
        """
        if self._initialized:
            return True

        # Numslots is always disabled in Receiver mode (sender controls slot count)
        self._host.set_param_enabled("Numslots", False)

        _t0 = time.perf_counter()
        # Tracks whether self._connection was overwritten with this attempt's handles below —
        # the outer except uses this to decide whether a targeted teardown is needed (a raise
        # after the commit must not leak the just-opened IPC/SHM handles into the next retry).
        connection_committed = False
        # Local shm_handle opened below (line ~663). Tracked here so the outer except can close
        # it on a raise that lands after the validation guards but before slot-loop coverage
        # (e.g. create_stream_with_priority) — that window is outside both the validation
        # guards' shm_handle.close() calls and _cleanup_partial's coverage, so without this it
        # leaks the SHM mapping on every failed attempt / backoff retry.
        shm_handle = None
        try:
            # The `cuda` constructor argument wins (tests, and a caller that already holds a
            # port). Without one, wrap the process-wide runtime in the production adapter.
            if self.cuda is None:
                self.cuda = CTypesCUDAAdapter(get_cuda_runtime(device=self.device))
            if not getattr(self, "_runtime_load_logged", False):
                self._log(f"Loaded CUDA runtime on device {self.cuda.get_device()}", force=True)
                self._runtime_load_logged = True

            # Open SharedMemory (sender must have created it)
            try:
                shm_handle = SharedMemory(name=self.shm_name)
            except FileNotFoundError:
                self._log(f"SharedMemory '{self.shm_name}' not found. Sender not ready?")
                return False

            # Validate the header: protocol magic, same-process rule, slot count.
            header = check_header(shm_handle.buf, _current_pid())
            if header.fault is not None:
                self._log(
                    {
                        HeaderFault.BAD_MAGIC: protocol_mismatch_message(header.magic),
                        HeaderFault.SAME_PROCESS: SAME_PROCESS_MESSAGE,
                        HeaderFault.UNREADABLE: "Cannot read protocol magic. Sender may be using old protocol version.",
                        HeaderFault.BAD_NUM_SLOTS: f"Invalid num_slots: {header.num_slots}. Protocol error.",
                    }[header.fault],
                    force=True,
                )
                shm_handle.close()
                return False
            ipc_version, num_slots = header.ipc_version, header.num_slots

            # Sync UI parameter to show sender's slot count (informational only).
            # Do NOT set self.num_slots — that's the sender-specific working value.
            # Receiver always uses connection.num_slots for its own arrays.
            self._host.set_param_value("Numslots", num_slots)

            # Cache receiver layout once — avoids per-frame arithmetic in import_frame()
            layout = SHMLayout(num_slots)
            shutdown_offset = layout.shutdown_offset
            metadata_offset = shutdown_offset + SHUTDOWN_FLAG_SIZE

            # Read and validate metadata via the canonical Metadata value object.
            if len(shm_handle.buf) >= metadata_offset + METADATA_SIZE:
                try:
                    _shm_mv = memoryview(bytes(shm_handle.buf))  # bytes copy: no mmap exported pointer
                    _md = Metadata.read_from(_shm_mv, layout)
                except (struct.error, ValueError, IndexError) as _e:
                    self._log(f"Failed to read SHM metadata: {_e}", force=True)
                    shm_handle.close()
                    return False
                width, height, num_comps = _md.width, _md.height, _md.num_comps
                format_kind, bits_per_comp, flags = _md.format_kind, _md.bits_per_comp, _md.flags
                buffer_size = _md.data_size
                self._log(
                    f"Read metadata: {width}x{height}x{num_comps}, "
                    f"kind={format_kind} bits={bits_per_comp} flags=0x{flags:04x}, "
                    f"buf_size={buffer_size}",
                    force=True,
                )
                # Strict size invariant: data_size must exactly equal W*H*C*(bits/8).
                if bits_per_comp == 0 or _md.data_size != _md.expected_size:
                    self._log(
                        f"Metadata size invariant failed: W*H*C*(bits/8)={_md.expected_size} "
                        f"but buf_size={_md.data_size}. Sender/receiver protocol mismatch.",
                        force=True,
                    )
                    shm_handle.close()
                    return False
            else:
                self._log("No extended metadata in SharedMemory (legacy sender)", force=True)
                shm_handle.close()
                return False  # Cannot proceed without knowing dimensions

            # Validate metadata
            if width == 0 or height == 0 or buffer_size == 0:
                self._log(
                    "Metadata contains zeros - sender may not have written frame yet",
                    force=True,
                )
                shm_handle.close()
                return False

            # Check for shutdown signal BEFORE opening IPC handles.
            try:
                if shm_handle.buf[shutdown_offset] == 1:
                    self._log(
                        "Shutdown flag is set — producer has exited. "
                        "SharedMemory contains stale IPC handles. Will retry.",
                        force=True,
                    )
                    shm_handle.close()
                    return False
            except (OSError, BufferError, IndexError) as e:
                self._log(f"Could not read shutdown flag: {e}", force=True)
                shm_handle.close()
                return False

            # Log write_idx for diagnostics (0 = no frames sent yet, handles still valid)
            try:
                write_idx_diag = read_write_idx(shm_handle.buf)
                self._log(f"Producer write_idx={write_idx_diag} (0 = no frames sent yet)", force=True)
            except (struct.error, ValueError):
                pass

            # Initialize arrays for this session
            dev_ptrs = [None] * num_slots
            ipc_handles = [None] * num_slots
            ipc_events = [None] * num_slots

            # Create dedicated non-blocking stream for receiver IPC operations
            # MUST happen before ipc_open_mem_handle to establish CUDA context
            # Reuse existing stream on re-init to avoid leaks on reconnection cycles
            stream_is_new = self._connection.stream is None
            if stream_is_new:
                stream = self.cuda.create_stream_with_priority(flags=0x01)
                self._log(
                    f"Created receiver stream: 0x{int(stream.value):016x}",
                    force=True,
                )
            else:
                stream = self._connection.stream
                self._log(
                    f"Reusing receiver stream: 0x{int(stream.value):016x}",
                    force=True,
                )

            # Open all IPC handles (per slot)
            for slot in range(num_slots):
                raw = read_slot_handle_bytes(shm_handle.buf, layout, slot)

                if not raw.has_mem:
                    self._log(
                        f"Slot {slot}: IPC mem handle is all zeros - "
                        "sender hasn't written handles yet. Will retry with backoff.",
                        force=True,
                    )
                    # Partial cleanup — slots 0..slot-1 already opened
                    self._cleanup_partial(slot, dev_ptrs, ipc_events, stream, stream_is_new, shm_handle)
                    return False

                self._log(f"[IPC-HEX] slot{slot} read handle prefix: {raw.mem[:16].hex()}...")

                try:
                    ipc_handles[slot], dev_ptrs[slot], ipc_events[slot] = open_slot(self.cuda, raw)
                except SlotEventOpenError as e:
                    # Without the event the wait has nothing to order against the
                    # sender, so every frame would be read unsynced: fail the init
                    # (slot + 1 also releases this slot's just-opened mem handle).
                    dev_ptrs[slot] = e.dev_ptr
                    self._log(
                        f"Slot {slot}: failed to open IPC event: {e.cause}. Will retry with backoff.",
                        force=True,
                    )
                    self._cleanup_partial(slot + 1, dev_ptrs, ipc_events, stream, stream_is_new, shm_handle)
                    return False
                except RuntimeError as e:
                    self._log(
                        f"Slot {slot}: cudaIpcOpenMemHandle failed: {e}. "
                        "Possible causes: sender process exited, GPU memory freed, "
                        "or CUDA device mismatch. Will retry with backoff.",
                        force=True,
                    )
                    self._cleanup_partial(slot, dev_ptrs, ipc_events, stream, stream_is_new, shm_handle)
                    return False

                self._log(
                    f"Opened slot {slot}: GPU at 0x{dev_ptrs[slot].value:016x}, "
                    f"event={'YES' if ipc_events[slot] else 'NO'}",
                    force=True,
                )

            # All slots opened — commit to connection and format
            self._connection = ReceiverConnection(
                shm_handle=shm_handle,
                dev_ptrs=dev_ptrs,
                ipc_handles=ipc_handles,
                ipc_events=ipc_events,
                stream=stream,
                layout=layout,
                num_slots=num_slots,
                ipc_version=ipc_version,
                shutdown_offset=shutdown_offset,
                last_write_idx=0,
            )
            connection_committed = True
            self._format = Format.from_metadata(_md)
            self._last_fmt_status = ""  # force status string rebuild on first imported frame

            # Flag that Script TOP resolution needs to be updated (applied outside cook cycle).
            self._pending.resolution = True

            # Flag pixel-format update when the resolved format is anything other than the
            # Script TOP default (rgba32float).  copyCUDAMemory adapts channel count from
            # _cached_shape, but par.format controls the output texture allocation (bit depth
            # AND channel layout).  Only rgba32float matches the saved/default Script TOP
            # format and therefore needs no explicit set_format call.  All other formats —
            # uint8, uint16, AND float32 variants with non-RGBA channel layout (monoalpha32float,
            # rg32float, r32float, etc.) — must be set explicitly so copyCUDAMemory writes into
            # a correctly-sized texture.  The previous guard (bits_per_comp != 32) was too broad:
            # it skipped the update for ALL float32 sources, including 2-channel monoalpha32float.
            if td_format_string(self._format) != "rgba32float":
                self._pending.format = True

            # Cache CUDAMemoryShape to avoid per-frame object creation
            if numpy is None:
                import numpy as np_module
            else:
                np_module = numpy

            # CUDAMemoryShape dtype: float16 must be presented as float32 because
            # copyCUDAMemory doesn't accept float16 — the actual conversion happens via D2H + copyNumpyArray.
            # For all other dtypes, use the numpy dtype decoded from wire metadata.
            if self._format.dtype_str == "float16":
                np_dtype = np_module.float32  # shape dtype for copyCUDAMemory; real f16 handled via D2H
            elif self._format.numpy_dtype is not None:
                np_dtype = self._format.numpy_dtype.type  # scalar type (e.g. numpy.uint8), not dtype instance
            else:
                np_dtype = np_module.float32  # safe fallback (e.g. bfloat16, no numpy dtype)

            # float16: allocate CPU buffers for D2H conversion (copyCUDAMemory doesn't support float16)
            if self._format.dtype_str == "float16":
                self._build_f16_staging(width, height, num_comps, num_slots, dev_ptrs, buffer_size)

            self._cached_shape = self._host.make_cuda_shape(width, height, num_comps, np_dtype)

            self._initialized = True
            self._diag_frames_since_reinit = 0  # Reset so import_frame logs the next 5 calls
            _init_ms = (time.perf_counter() - _t0) * 1000.0
            self._log(
                f"Receiver initialized: {num_slots} slots, {width}x{height}x{num_comps} (init took {_init_ms:.1f} ms)",
                force=True,
            )
            return True

        except (OSError, RuntimeError, ValueError) as e:
            self._log(f"Receiver initialization failed: {e}", force=True)

            if connection_committed:
                # Post-commit work (format/shape resolution, float16 buffer allocation) raised
                # after self._connection above was already overwritten with this attempt's
                # IPC/SHM handles. Without this, those handles are dropped unclosed and leak
                # on every subsequent backoff retry. Release resources only — do NOT call the
                # full cleanup() here, since it also resets self._retry's backoff counters,
                # which would make the receiver retry every frame instead of backing off.
                self._release_resources()
            elif shm_handle is not None:
                # Raised after shm_handle was opened but before any validation guard's
                # shm_handle.close() or the slot loop's _cleanup_partial coverage began
                # (e.g. create_stream_with_priority) — close it directly so it doesn't leak.
                with contextlib.suppress(OSError, BufferError):
                    shm_handle.close()

            traceback.print_exc()
            return False

    def _cleanup_partial(
        self,
        failed_slot: int,
        dev_ptrs: list,
        ipc_events: list,
        stream: object,
        stream_is_new: bool,
        shm_handle: object,
    ) -> None:
        """Cleanup partially-opened resources when initialization fails mid-slot.

        Called when initialize_receiver() fails partway through slot iteration.
        Closes IPC handles already opened for slots 0..failed_slot-1 to prevent
        GPU resource leaks across backoff retries.

        Args:
            failed_slot: The slot index that failed (0-based). Cleans up slots 0..failed_slot-1.
            dev_ptrs: In-progress dev_ptrs list from this init attempt.
            ipc_events: In-progress ipc_events list from this init attempt.
            stream: Stream created for this init attempt (only closed if freshly created).
            stream_is_new: True when `stream` was created by this attempt (not reused from
                self._connection.stream) — only then is it safe to destroy it here.
            shm_handle: SHM handle to close and clear.
        """
        for i in range(failed_slot):
            if dev_ptrs[i] is not None:
                try:
                    self.cuda.ipc_close_mem_handle(dev_ptrs[i])
                    self._log(f"Cleaned up partial slot {i} mem handle")
                except (RuntimeError, OSError):
                    pass
                dev_ptrs[i] = None
            if ipc_events[i] is not None:
                with contextlib.suppress(RuntimeError, OSError):
                    self.cuda.destroy_event(ipc_events[i])
                ipc_events[i] = None

        # Destroy the stream only if this attempt created it fresh — a stream reused from
        # self._connection.stream must survive so the next retry can keep reusing it (a
        # freshly-created stream left un-destroyed here was a latent leak on every failed
        # per-slot open before the first successful connect).
        if stream_is_new and stream is not None:
            with contextlib.suppress(RuntimeError, OSError):
                self.cuda.destroy_stream(stream)

        # Close SharedMemory so next retry re-opens fresh (avoids reading stale content)
        if shm_handle is not None:
            with contextlib.suppress(OSError, BufferError):
                shm_handle.close()

    def _release_resources(self) -> None:
        """Close the active connection and free float16 D2H buffers (idempotent).

        Resource-teardown only — does NOT touch self._retry backoff counters or
        self._initialized. Shared by cleanup() (full teardown on deactivate) and
        initialize_receiver()'s post-commit failure path, where resetting backoff state
        would make the receiver retry every frame instead of backing off (see the
        outer except in initialize_receiver()).
        """
        self._connection.close(self.cuda, self._log)
        self._close_doorbell()
        self.publish_cadence = PublishCadence()
        self._release_f16_staging()
        self._cached_shape = None

    def _build_f16_staging(
        self,
        width: int,
        height: int,
        num_comps: int,
        num_slots: int,
        dev_ptrs: list,
        buffer_size: int,
    ) -> None:
        """Allocate the float16 conversion scratch for the given geometry and slots.

        copyCUDAMemory has no float16 input, so a float16 sender is converted either
        on the GPU (CuPy views over each slot's dev_ptr + a float32 staging buffer) or
        via a D2H copy into a pinned (or page-locked pageable) host buffer. The CuPy
        views wrap ``dev_ptrs`` directly, so they must be rebuilt whenever the slot
        handles are reopened — see _refresh_on_version_change().
        """
        if numpy is None:
            import numpy as np_module
        else:
            np_module = numpy

        n_elems = width * height * num_comps
        f16_bytes = n_elems * 2
        self._f16_pinned_ptr = None
        try:
            import ctypes as _ctypes

            self._f16_pinned_ptr = self.cuda.malloc_host_alloc(f16_bytes)
            _buf = (_ctypes.c_ubyte * f16_bytes).from_address(self._f16_pinned_ptr.value)
            self._f16_cpu_buf = np_module.frombuffer(_buf, dtype=np_module.float16)
            self._log("float16 receiver: allocated pinned CPU buffer for D2H (async path)", force=True)
        except (RuntimeError, OSError) as _e:
            self._f16_pinned_ptr = None
            self._f16_cpu_buf = np_module.empty(n_elems, dtype=np_module.float16)
            self._log(f"float16 receiver: pinned alloc failed ({_e}), using pageable buffer", force=True)
            # R3: page-lock the pageable fallback buffer so D2H runs at pinned bandwidth
            try:
                self.cuda.host_register(self._f16_cpu_buf.ctypes.data, self._f16_cpu_buf.nbytes)
                self._f16_host_registered = True
                self._log("float16 receiver: pageable buffer page-locked via cudaHostRegister", force=True)
            except (RuntimeError, OSError) as _re:
                self._f16_host_registered = False
                self._log(f"float16 receiver: host_register skipped ({_re})", force=True)
        self._f32_cpu_buf = np_module.empty((height, width, num_comps), dtype=np_module.float32)

        # GPU-side float32 staging buffer for CuPy conversion path
        global CUPY_AVAILABLE, cp
        if cp is None:
            try:
                import cupy as cp  # noqa: PLC0415

                CUPY_AVAILABLE = True
            except ImportError:
                CUPY_AVAILABLE = False

        if CUPY_AVAILABLE:
            try:
                self._cupy_f32_buf = cp.empty((height, width, num_comps), dtype=cp.float32)
                self._log(
                    "float16 receiver: CuPy GPU float32 buffer allocated (GPU-side conversion path)",
                    force=True,
                )
                self._cupy_f16_views = []
                for _i in range(num_slots):
                    _ptr = dev_ptrs[_i].value
                    _mem = cp.cuda.UnownedMemory(_ptr, buffer_size, owner=self)
                    _memptr = cp.cuda.MemoryPointer(_mem, 0)
                    self._cupy_f16_views.append(
                        cp.ndarray(
                            (height, width, num_comps),
                            dtype=cp.float16,
                            memptr=_memptr,
                        )
                    )
            except Exception as _e:  # noqa: BLE001  # CuPy OutOfMemoryError is not MemoryError; keep broad so alloc failures always degrade to CPU fallback
                self._cupy_f32_buf = None
                self._cupy_f16_views = []
                self._log(
                    f"float16 receiver: CuPy GPU buffer alloc failed ({_e}), CPU fallback active",
                    force=True,
                )

    def _release_f16_staging(self) -> None:
        """Free the float16 conversion scratch built by _build_f16_staging() (idempotent)."""
        # Free pinned float16 D2H buffer if allocated
        if self._f16_pinned_ptr is not None:
            try:
                self.cuda.free_host(self._f16_pinned_ptr)
            except (RuntimeError, OSError) as e:
                self._log(f"free_host skipped (context gone): {e}")
            self._f16_pinned_ptr = None
        # R3: unregister the pageable fallback buffer if it was page-locked
        if self._f16_host_registered and self._f16_cpu_buf is not None:
            try:
                self.cuda.host_unregister(self._f16_cpu_buf.ctypes.data)
            except (RuntimeError, OSError) as e:
                self._log(f"host_unregister skipped (context gone): {e}")
            self._f16_host_registered = False
        self._f16_cpu_buf = None
        self._f32_cpu_buf = None
        self._cupy_f32_buf = None  # CuPy memory pool handles GPU free on GC
        self._cupy_f16_views = []

    def cleanup(self) -> None:
        """Cleanup Receiver CUDA IPC resources."""
        # Guard against double-cleanup (matches cleanup_sender() pattern)
        if not self._initialized and self._connection.shm_handle is None:
            return

        self._release_resources()

        self._initialized = False
        # Drop the backoff from the last connect loop so the first retry after a drop is immediate
        self._retry.reset()
        # Reset per-session perf tracking so reconnects get a fresh window baseline
        self._rx_window.reset()

    def _refresh_on_version_change(self, new_version: int) -> bool:
        """Refresh format and IPC handles in-place after a sender version bump.

        Keeps SHM, stream, and unchanged IPC handles open. Only re-reads the
        20-byte metadata block and rebuilds self._format and self._cached_shape.
        For genuine sender re-inits (new IPC handles), also closes old handles
        and opens the new ones — preserving the SHM connection throughout.

        Mirrors src/cuda_link/cuda_ipc_importer.py:_reinitialize.

        Returns:
            True if refresh succeeded (caller skips cleanup and continues).
            False on any error (caller falls back to cleanup + full reinit).
        """
        conn = self._connection
        shm = conn.shm_handle
        if shm is None or conn.layout is None:
            return False

        layout = conn.layout
        if len(shm.buf) < layout.metadata_offset + METADATA_SIZE:
            return False
        try:
            _md = Metadata.read_from(memoryview(shm.buf), layout)
        except (struct.error, ValueError, IndexError):
            return False

        # Validate metadata invariant before accepting new format
        if _md.bits_per_comp == 0 or _md.width == 0 or _md.height == 0 or _md.num_comps == 0:
            return False
        if _md.data_size != _md.expected_size:
            self._log(
                f"Metadata invariant failed during refresh: "
                f"W*H*C*(bits/8)={_md.expected_size} but buf_size={_md.data_size}. "
                "Falling back to full reinit.",
                force=True,
            )
            return False

        width, height, num_comps = _md.width, _md.height, _md.num_comps
        format_kind, bits_per_comp = _md.format_kind, _md.bits_per_comp

        # Detect whether IPC handles changed (metadata-only bump vs genuine sender re-init)
        handles_changed = False
        if self.cuda is not None and conn.ipc_handles:
            for slot in range(conn.num_slots):
                new_mem_bytes = read_slot_handle_bytes(shm.buf, layout, slot).mem
                old_handle = conn.ipc_handles[slot] if slot < len(conn.ipc_handles) else None
                if old_handle is None or bytes(old_handle) != new_mem_bytes:
                    handles_changed = True
                    break

        if handles_changed and self.cuda is not None:
            # Genuine sender re-init: open the new IPC imports first (keep SHM+stream).
            # The old ones stay open and owned by conn until every new slot opened, so a
            # failure here leaves conn intact for the caller's cleanup() + full reinit.
            opened = self._open_refresh_handles(shm, conn.num_slots)
            if opened is None:
                return False
            new_dev_ptrs, new_ipc_handles, new_ipc_events = opened

            for dev_ptr in conn.dev_ptrs:
                if dev_ptr:
                    with contextlib.suppress(RuntimeError, OSError):
                        self.cuda.ipc_close_mem_handle(dev_ptr)
            for event in conn.ipc_events:
                if event:
                    with contextlib.suppress(RuntimeError, OSError):
                        self.cuda.destroy_event(event)

            conn.dev_ptrs = new_dev_ptrs
            conn.ipc_handles = new_ipc_handles
            conn.ipc_events = new_ipc_events

        # Rebuild format from the validated Metadata
        prev_format = self._format
        self._format = Format.from_metadata(_md)
        self._last_fmt_status = ""  # force status string rebuild after version-change refresh

        # The float16 CuPy views wrap the slot dev_ptrs and the staging buffers are
        # sized for the old geometry: rebuild both when either changed (also covers a
        # float32 <-> float16 switch, which allocates or frees the scratch).
        if self.cuda is not None and (handles_changed or self._format.layout_differs_from(prev_format)):
            self._release_f16_staging()
            if self._format.dtype_str == "float16":
                self._build_f16_staging(width, height, num_comps, conn.num_slots, conn.dev_ptrs, _md.data_size)

        # Rebuild _cached_shape via the TDHost seam (keeps CUDAMemoryShape behind the seam)
        if self._cached_shape is not None:
            if numpy is None:
                import numpy as np_module
            else:
                np_module = numpy

            if self._format.dtype_str == "float16":
                np_dtype = np_module.float32
            elif self._format.numpy_dtype is not None:
                np_dtype = self._format.numpy_dtype.type  # scalar type (e.g. numpy.uint8), not dtype instance
            else:
                np_dtype = np_module.float32

            self._cached_shape = self._host.make_cuda_shape(width, height, num_comps, np_dtype)

        # Advance version counter; signal resolution and/or pixel-format update if needed.
        conn.ipc_version = new_version
        # The re-initialized sender counts write_idx from scratch; a stale index from the
        # previous session would make its first frame read as already consumed.
        conn.last_write_idx = 0
        if width != prev_format.width or height != prev_format.height:
            self._pending.resolution = True
        if (
            bits_per_comp != prev_format.bits
            or format_kind != prev_format.kind
            or num_comps != prev_format.num_comps
            or _md.flags != prev_format.flags
        ):
            # flags carries the channel layout (e.g. RG vs Mono+Alpha at the same comps).
            # Script TOP's par.format must be updated so the next copyCUDAMemory writes
            # into a correctly-sized output texture (e.g. uint16 → float32 half-fills).
            self._pending.format = True

        self._log(
            f"Format refreshed in-place: {width}x{height}x{num_comps}, "
            f"kind={format_kind} bits={bits_per_comp} v{new_version}"
            + (" (handles reopened)" if handles_changed else ""),
            force=True,
        )
        return True

    def _open_refresh_handles(self, shm: object, num_slots: int) -> tuple[list, list, list] | None:
        """Open every slot's new IPC mem/event handles for an in-place refresh.

        Returns (dev_ptrs, ipc_handles, ipc_events), or None when a slot handle is
        zero or fails to open — in which case only the handles opened by this call
        are closed; the caller's existing connection is left untouched.
        """
        new_dev_ptrs: list = [None] * num_slots
        new_ipc_handles: list = [None] * num_slots
        new_ipc_events: list = [None] * num_slots
        layout = SHMLayout(num_slots)
        for slot in range(num_slots):
            raw = read_slot_handle_bytes(shm.buf, layout, slot)
            if not raw.has_mem:
                self._log(f"Refresh: slot {slot} handle is zero — falling back to full reinit", force=True)
                self._close_refresh_handles(new_dev_ptrs, new_ipc_events)
                return None
            try:
                new_ipc_handles[slot], new_dev_ptrs[slot], new_ipc_events[slot] = open_slot(self.cuda, raw)
            except SlotEventOpenError as e:
                new_dev_ptrs[slot] = e.dev_ptr
                self._log(f"Refresh: slot {slot} ipc_open_event_handle failed: {e.cause} — falling back", force=True)
                self._close_refresh_handles(new_dev_ptrs, new_ipc_events)
                return None
            except (RuntimeError, OSError) as e:
                self._log(f"Refresh: slot {slot} ipc_open_mem_handle failed: {e} — falling back", force=True)
                self._close_refresh_handles(new_dev_ptrs, new_ipc_events)
                return None
        return new_dev_ptrs, new_ipc_handles, new_ipc_events

    def _close_refresh_handles(self, dev_ptrs: list, ipc_events: list) -> None:
        """Close the partially-opened handles of a failed _open_refresh_handles() call."""
        for dev_ptr in dev_ptrs:
            if dev_ptr:
                with contextlib.suppress(RuntimeError, OSError):
                    self.cuda.ipc_close_mem_handle(dev_ptr)
        for event in ipc_events:
            if event:
                with contextlib.suppress(RuntimeError, OSError):
                    self.cuda.destroy_event(event)
