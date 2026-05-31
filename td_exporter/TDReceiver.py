"""
TDReceiver - Receiver engine for CUDAIPCExtension.

Owns all Receiver-mode CUDA IPC resources: SHM attachment, IPC handle opening,
per-frame GPU event sync, and Script TOP copyCUDAMemory calls.

textDAT name: TDReceiver  (must match the importable module name inside the COMP namespace)
"""

from __future__ import annotations

import contextlib
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

from CUDAIPCWrapper import get_cuda_runtime  # noqa: E402
from CUDARuntimeTypes import cudaIpcEventHandle_t, cudaIpcMemHandle_t  # noqa: E402
from NVTXShim import pop_range as _nvtx_pop  # noqa: E402
from NVTXShim import push_range as _nvtx_push
from NVTXShim import verbose_range as _nvtx_verbose
from Importer import Format  # noqa: E402
from SHMProtocol import (  # noqa: E402
    FLAGS_MONO_ALPHA,
    FORMAT_KIND_FLOAT,
    MAGIC_OFFSET,
    MAGIC_SIZE,
    METADATA_SIZE,
    NUM_SLOTS_OFFSET,
    NUM_SLOTS_SIZE,
    PROTOCOL_MAGIC,
    SHM_HEADER_SIZE,
    SHUTDOWN_FLAG_SIZE,
    SLOT_SIZE,
    VERSION_OFFSET,
    VERSION_SIZE,
    WRITE_IDX_OFFSET,
    Metadata,
    SHMLayout,
    SlotState,
    acquire_slot,
)
from TDConfig import TDSenderConfig  # noqa: E402
from TDHost import TDHost  # noqa: E402

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


@dataclass
class RetryState:
    """Retry policy and transient counters for the connection-attempt loop."""

    connect_attempts: int = 0
    max_connect_attempts: int = 20
    backoff_intervals: tuple = (1, 2, 4, 8, 16, 32, 64, 120)
    retry_interval_frames: int = 1
    frames_since_last_retry: int = 0
    needs_resolution_update: bool = False
    needs_format_update: bool = False  # set when bits_per_comp/format_kind/num_comps changes

    def request_immediate_reconnect(self) -> None:
        """Force the next import_frame call to attempt reconnection."""
        self.frames_since_last_retry = self.retry_interval_frames

    def consume_resolution_update(self) -> bool:
        """Return True and clear the flag if a resolution update is pending."""
        if self.needs_resolution_update:
            self.needs_resolution_update = False
            return True
        return False

    def consume_format_update(self) -> bool:
        """Return True and clear the flag if a pixel-format update is pending."""
        if self.needs_format_update:
            self.needs_format_update = False
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
        config: TDSenderConfig,
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
        self._retry = RetryState()

        # Engine-private F16 conversion scratch (mutable per-format caches; not value objects)
        self._f16_cpu_buf = None
        self._f32_cpu_buf = None
        self._f16_pinned_ptr = None
        self._cupy_f32_buf = None
        self._cupy_f16_views: list = []
        self._cached_shape = None

        self._diag_frames_since_reinit: int = 0

        # frame_count mirrored from sender SHM - exposed for get_stats()
        self.frame_count = 0

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
        if self._retry.consume_resolution_update():
            return (self._format.width, self._format.height)
        return None

    def consume_pending_format(self) -> str | None:
        """Return the TD par.format string if a pixel-format update is pending, else None.

        Called from script_top_callbacks.onCook to drive ImportBuffer par.format updates in
        the fallback path (modoutsidecook disabled). Mirrors consume_pending_resolution.

        Returns a string like 'rgba16fixed', 'r32float', 'rgba8fixed', etc., or None.
        """
        if self._retry.consume_format_update():
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
            self._retry.frames_since_last_retry += 1
            if self._retry.frames_since_last_retry < self._retry.retry_interval_frames:
                return False  # Wait before retrying

            self._retry.frames_since_last_retry = 0
            self._retry.connect_attempts += 1

            if not self.initialize_receiver():
                backoff_idx = min(self._retry.connect_attempts, len(self._retry.backoff_intervals) - 1)
                self._retry.retry_interval_frames = self._retry.backoff_intervals[backoff_idx]
                if self._retry.connect_attempts <= self._retry.max_connect_attempts:
                    self._log(
                        f"Waiting for sender... (attempt {self._retry.connect_attempts}, "
                        f"next retry in {self._retry.retry_interval_frames} frames)"
                    )
                elif self._retry.connect_attempts == self._retry.max_connect_attempts + 1:
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

            self._connection.last_write_idx = result.write_idx
            write_idx = result.write_idx
            read_slot = result.slot

            _diag = self._diag_frames_since_reinit < 5
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
                    # Run conversion on _connection.stream so copyCUDAMemory (also on _connection.stream)
                    # automatically serializes after the elementwise cast kernel.
                    with cp.cuda.ExternalStream(rx_stream_int):
                        cp.copyto(self._cupy_f32_buf, cupy_f16, casting="same_kind")

                    handle.copy_cuda_memory(
                        self._cupy_f32_buf.data.ptr,
                        f32_size,
                        self._cached_shape,  # dataType=float32 set during initialize_receiver()
                        stream=rx_stream_int,
                    )
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

            if _diag:
                _copy_ms = (time.perf_counter() - _t_copy) * 1000.0
                self._log(
                    f"[DIAG] import_frame #{self._diag_frames_since_reinit}: "
                    f"slot={read_slot} write_idx={write_idx} addr=0x{address:x} "
                    f"stream_wait={_event_ms:.2f}ms copyCUDAMemory={_copy_ms:.2f}ms",
                    force=True,
                )

            self.frame_count += 1
            self._connection.last_write_idx = write_idx

            _fmt_name = td_format_string(self._format)
            self._host.set_info_status(f"{self._format.width}x{self._format.height} {_fmt_name}")

            # Debug logging (97 = prime, avoids aliasing with slot counts 2,4,5)
            if self.verbose_performance and self.frame_count % 97 == 0:
                self._log(f"Frame {self.frame_count}: read_slot={read_slot}, write_idx={write_idx}")

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
        if not self._retry.needs_resolution_update:
            return False

        try:
            handle.set_resolution(self._format.width, self._format.height)
            self._retry.needs_resolution_update = False
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
        if not self._retry.needs_format_update:
            return False
        try:
            fmt = td_format_string(self._format)
            handle.set_format(fmt)
            self._retry.needs_format_update = False
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
        try:
            self.cuda = get_cuda_runtime(device=self.device)
            if not getattr(self, "_runtime_load_logged", False):
                self._log(f"Loaded CUDA runtime on device {self.cuda.get_device()}", force=True)
                self._runtime_load_logged = True

            # Open SharedMemory (sender must have created it)
            try:
                shm_handle = SharedMemory(name=self.shm_name)
            except FileNotFoundError:
                self._log(f"SharedMemory '{self.shm_name}' not found. Sender not ready?")
                return False

            # Validate protocol magic number (new in this version)
            try:
                magic = struct.unpack(
                    "<I",
                    bytes(shm_handle.buf[MAGIC_OFFSET : MAGIC_OFFSET + MAGIC_SIZE]),
                )[0]
                if magic != PROTOCOL_MAGIC:
                    self._log(
                        f"Protocol magic mismatch: expected 0x{PROTOCOL_MAGIC:08X}, got 0x{magic:08X}. "
                        "Sender using incompatible protocol version.",
                        force=True,
                    )
                    shm_handle.close()
                    return False
            except (struct.error, ValueError, IndexError):
                self._log(
                    "Cannot read protocol magic. Sender may be using old protocol version.",
                    force=True,
                )
                shm_handle.close()
                return False

            # Read header
            ipc_version = struct.unpack(
                "<Q",
                bytes(shm_handle.buf[VERSION_OFFSET : VERSION_OFFSET + VERSION_SIZE]),
            )[0]
            num_slots = struct.unpack(
                "<I",
                bytes(shm_handle.buf[NUM_SLOTS_OFFSET : NUM_SLOTS_OFFSET + NUM_SLOTS_SIZE]),
            )[0]

            if num_slots == 0 or num_slots > 10:
                self._log(
                    f"Invalid num_slots: {num_slots}. Protocol error.",
                    force=True,
                )
                shm_handle.close()
                return False

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
                except Exception as _e:  # noqa: BLE001
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
                write_idx_diag = struct.unpack_from("<I", shm_handle.buf, WRITE_IDX_OFFSET)[0]
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
            if self._connection.stream is None:
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
                base_offset = SHM_HEADER_SIZE + (slot * SLOT_SIZE)

                # Read + open memory handle
                mem_handle_bytes = bytes(shm_handle.buf[base_offset : base_offset + 64])

                if not any(mem_handle_bytes):
                    self._log(
                        f"Slot {slot}: IPC mem handle is all zeros - "
                        "sender hasn't written handles yet. Will retry with backoff.",
                        force=True,
                    )
                    # Partial cleanup — slots 0..slot-1 already opened
                    self._cleanup_partial(slot, dev_ptrs, ipc_events, stream, shm_handle)
                    return False

                self._log(f"[IPC-HEX] slot{slot} read handle prefix: {mem_handle_bytes[:16].hex()}...")
                ipc_handles[slot] = cudaIpcMemHandle_t.from_buffer_copy(mem_handle_bytes)

                try:
                    dev_ptrs[slot] = self.cuda.ipc_open_mem_handle(ipc_handles[slot], flags=1)
                except RuntimeError as e:
                    self._log(
                        f"Slot {slot}: cudaIpcOpenMemHandle failed: {e}. "
                        "Possible causes: sender process exited, GPU memory freed, "
                        "or CUDA device mismatch. Will retry with backoff.",
                        force=True,
                    )
                    self._cleanup_partial(slot, dev_ptrs, ipc_events, stream, shm_handle)
                    return False

                # Read + open event handle
                event_handle_bytes = bytes(shm_handle.buf[base_offset + 64 : base_offset + 128])
                if any(event_handle_bytes):
                    try:
                        ipc_event_handle = cudaIpcEventHandle_t.from_buffer_copy(event_handle_bytes)
                        ipc_events[slot] = self.cuda.ipc_open_event_handle(ipc_event_handle)
                    except (RuntimeError, OSError) as e:
                        self._log(f"Failed to open IPC event for slot {slot}: {e}")
                        ipc_events[slot] = None

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
            self._format = Format.from_metadata(_md)

            # Flag that Script TOP resolution needs to be updated (applied outside cook cycle).
            self._retry.needs_resolution_update = True

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
                self._retry.needs_format_update = True

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
                n_elems = width * height * num_comps
                f16_bytes = n_elems * 2
                self._f16_pinned_ptr = None
                try:
                    import ctypes as _ctypes

                    self._f16_pinned_ptr = self.cuda.malloc_host(f16_bytes)
                    _buf = (_ctypes.c_ubyte * f16_bytes).from_address(self._f16_pinned_ptr.value)
                    self._f16_cpu_buf = np_module.frombuffer(_buf, dtype=np_module.float16)
                    self._log("float16 receiver: allocated pinned CPU buffer for D2H (async path)", force=True)
                except (RuntimeError, OSError) as _e:
                    self._f16_pinned_ptr = None
                    self._f16_cpu_buf = np_module.empty(n_elems, dtype=np_module.float16)
                    self._log(f"float16 receiver: pinned alloc failed ({_e}), using pageable buffer", force=True)
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
                    except Exception as _e:
                        self._cupy_f32_buf = None
                        self._cupy_f16_views = []
                        self._log(
                            f"float16 receiver: CuPy GPU buffer alloc failed ({_e}), CPU fallback active",
                            force=True,
                        )

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

            traceback.print_exc()
            return False

    def _cleanup_partial(
        self,
        failed_slot: int,
        dev_ptrs: list,
        ipc_events: list,
        stream: object,
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

        # Close SharedMemory so next retry re-opens fresh (avoids reading stale content)
        if shm_handle is not None:
            with contextlib.suppress(OSError, BufferError):
                shm_handle.close()

    def cleanup(self) -> None:
        """Cleanup Receiver CUDA IPC resources."""
        # Guard against double-cleanup (matches cleanup_sender() pattern)
        if not self._initialized and self._connection.shm_handle is None:
            return

        self._connection.close(self.cuda, self._log)

        # Free pinned float16 D2H buffer if allocated
        if self._f16_pinned_ptr is not None:
            try:
                self.cuda.free_host(self._f16_pinned_ptr)
            except (RuntimeError, OSError) as e:
                self._log(f"free_host skipped (context gone): {e}")
            self._f16_pinned_ptr = None
        self._f16_cpu_buf = None
        self._f32_cpu_buf = None
        self._cupy_f32_buf = None  # CuPy memory pool handles GPU free on GC
        self._cupy_f16_views = []
        self._cached_shape = None

        self._initialized = False
        self._retry.connect_attempts = 0
        self._retry.frames_since_last_retry = 0

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
                base_offset = SHM_HEADER_SIZE + slot * SLOT_SIZE
                new_mem_bytes = bytes(shm.buf[base_offset : base_offset + 64])
                old_handle = conn.ipc_handles[slot] if slot < len(conn.ipc_handles) else None
                if old_handle is None or bytes(old_handle) != new_mem_bytes:
                    handles_changed = True
                    break

        if handles_changed and self.cuda is not None:
            # Genuine sender re-init: close old IPC imports, open new ones (keep SHM+stream).
            for dev_ptr in conn.dev_ptrs:
                if dev_ptr:
                    with contextlib.suppress(RuntimeError, OSError):
                        self.cuda.ipc_close_mem_handle(dev_ptr)
            for event in conn.ipc_events:
                if event:
                    with contextlib.suppress(RuntimeError, OSError):
                        self.cuda.destroy_event(event)

            new_dev_ptrs = [None] * conn.num_slots
            new_ipc_handles = [None] * conn.num_slots
            new_ipc_events = [None] * conn.num_slots
            for slot in range(conn.num_slots):
                base_offset = SHM_HEADER_SIZE + slot * SLOT_SIZE
                mem_handle_bytes = bytes(shm.buf[base_offset : base_offset + 64])
                if not any(mem_handle_bytes):
                    self._log(f"Refresh: slot {slot} handle is zero — falling back to full reinit", force=True)
                    return False
                new_ipc_handles[slot] = cudaIpcMemHandle_t.from_buffer_copy(mem_handle_bytes)
                try:
                    new_dev_ptrs[slot] = self.cuda.ipc_open_mem_handle(new_ipc_handles[slot], flags=1)
                except RuntimeError as e:
                    self._log(f"Refresh: slot {slot} ipc_open_mem_handle failed: {e} — falling back", force=True)
                    return False
                event_handle_bytes = bytes(shm.buf[base_offset + 64 : base_offset + 128])
                if any(event_handle_bytes):
                    with contextlib.suppress(RuntimeError, OSError):
                        ipc_evt_handle = cudaIpcEventHandle_t.from_buffer_copy(event_handle_bytes)
                        new_ipc_events[slot] = self.cuda.ipc_open_event_handle(ipc_evt_handle)

            conn.dev_ptrs = new_dev_ptrs
            conn.ipc_handles = new_ipc_handles
            conn.ipc_events = new_ipc_events

        # Rebuild format from the validated Metadata
        prev_format = self._format
        self._format = Format.from_metadata(_md)

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
        if width != prev_format.width or height != prev_format.height:
            self._retry.needs_resolution_update = True
        if bits_per_comp != prev_format.bits or format_kind != prev_format.kind or num_comps != prev_format.num_comps:
            # Script TOP's par.format must be updated so the next copyCUDAMemory writes
            # into a correctly-sized output texture (e.g. uint16 → float32 half-fills).
            self._retry.needs_format_update = True

        self._log(
            f"Format refreshed in-place: {width}x{height}x{num_comps}, "
            f"kind={format_kind} bits={bits_per_comp} v{new_version}"
            + (" (handles reopened)" if handles_changed else ""),
            force=True,
        )
        return True
