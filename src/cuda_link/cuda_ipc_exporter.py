"""
CUDA IPC Exporter for Python Process
Exports GPU memory FROM Python TO TouchDesigner via CUDA IPC handles.

Enables the reverse direction: Python AI pipeline → TouchDesigner display.
The TD side receives frames using CUDAIPCExtension in "Receiver" mode.

Usage:
    from cuda_link import CUDAIPCExporter

    # Export AI-generated frames to TouchDesigner
    with CUDAIPCExporter(
        shm_name="ai_output_ipc",
        height=512, width=512, channels=4, dtype="uint8",
    ) as exporter:
        exporter.initialize()
        while running:
            # output_tensor: (H, W, 4) uint8 BGRA on GPU
            exporter.export_frame(
                gpu_ptr=output_tensor.data_ptr(),
                size=output_tensor.nelement() * output_tensor.element_size(),
            )

Architecture:
    Python GPU tensor --> cudaMemcpy D2D --> Persistent IPC Ring Buffer
                                                     |
                                      IPC Handle in SharedMemory (v0.5.0 protocol)
                                                     |
                        TouchDesigner (CUDAIPCExtension Receiver) opens handle once
                        --> import_frame(script_top) --> copyCUDAMemory() per frame

Performance:
    - Initialization: ~1ms (buffer alloc + handle export)
    - Per-frame: ~177µs at 512x512 @ 60 FPS (includes producer-side stream_synchronize)
    - Per-frame: ~219µs at 1080p @ 60 FPS (async D2D + stream_synchronize + protocol writes)

Compatibility:
    - Protocol: v0.5.0 (byte-identical to CUDAIPCExtension/CUDAIPCImporter)
    - TD side: CUDAIPCExtension in "Receiver" mode reads the SharedMemory
    - Platform: Windows only (CUDA IPC limitation)
"""

from __future__ import annotations

import contextlib
import logging
import os
import struct
import threading
import time
import traceback
from ctypes import c_void_p
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING

from .activation_barrier import bump_skip as _ab_bump
from .activation_barrier import open_or_create as _ab_open
from .activation_barrier import read_state as _ab_read
from .cuda_ipc_wrapper import (  # noqa: F401
    CUDART_GRAPHS_MIN_VERSION,
    CUDAGraph_t,
    CUDAGraphExec_t,
    CUDAGraphNode_t,
    CUDARuntimeAPI,
    CUDAStream_t,
    cudaIpcMemHandle_t,
    get_cuda_runtime,
)

if TYPE_CHECKING:
    from .nvml_observer import NVMLObserver

logger = logging.getLogger(__name__)

# Protocol layout constants (must match CUDAIPCExtension and CUDAIPCImporter)
PROTOCOL_MAGIC = 0x43495044  # "CIPD" - protocol validation magic number (v1.0.0)
SHM_HEADER_SIZE = 20  # 4B magic + 8B version + 4B num_slots + 4B write_idx
SLOT_SIZE = 128  # 64B mem_handle + 64B event_handle
SHUTDOWN_FLAG_SIZE = 1
METADATA_SIZE = 20  # 4B width + 4B height + 4B num_comps + 1B kind + 1B bits + 2B flags + 4B data_size
TIMESTAMP_SIZE = 8  # 8B float64 producer timestamp

# Pre-compiled struct objects for hot-path SHM reads/writes (~50-100ns saved per call vs format-string lookup)
_ST_U32 = struct.Struct("<I")  # uint32 LE (write_idx, num_slots, metadata fields)
_ST_U64 = struct.Struct("<Q")  # uint64 LE (version)
_ST_F64 = struct.Struct("<d")  # float64 LE (timestamp)
_ST_BBH = struct.Struct("<BBH")  # uint8 + uint8 + uint16 LE (format_kind, bits_per_comp, flags)

# CUDA-aligned dtype encoding (cudaChannelFormatKind values):
FORMAT_KIND_SIGNED = 0  # cudaChannelFormatKindSigned
FORMAT_KIND_UNSIGNED = 1  # cudaChannelFormatKindUnsigned
FORMAT_KIND_FLOAT = 2  # cudaChannelFormatKindFloat
FLAGS_BFLOAT16 = 0x0001  # flag bit: bfloat16 (kind=Float, bits=16)

# Map dtype string → (format_kind, bits_per_component, flags)
_DTYPE_TO_KIND_BITS: dict[str, tuple[int, int, int]] = {
    "float32": (FORMAT_KIND_FLOAT, 32, 0),
    "float16": (FORMAT_KIND_FLOAT, 16, 0),
    "uint8": (FORMAT_KIND_UNSIGNED, 8, 0),
    "uint16": (FORMAT_KIND_UNSIGNED, 16, 0),
}

_DTYPE_ITEMSIZE_MAP = {
    "float32": 4,
    "float16": 2,
    "uint8": 1,
    "uint16": 2,
}

# C3: CPU release-fence between shutdown_flag write and write_idx publish.
# On x86/x64 the hardware guarantees TSO (total-store-order) for plain stores,
# but CPython makes no compiler-level ordering guarantee between two separate
# bytearray writes. threading.Lock acquire/release issues OS-level memory barriers
# on all supported platforms, providing the needed release semantics. Cost: ~80ns.
_fence_lock = threading.Lock()


def _release_fence() -> None:
    with _fence_lock:
        pass


class CUDAIPCExporter:
    """Python-side exporter for CUDA IPC GPU memory.

    Sends GPU frames FROM Python TO TouchDesigner via CUDA IPC.
    Pairs with CUDAIPCExtension in "Receiver" mode on the TD side.

    Responsibilities:
    - Allocate persistent GPU ring buffer (cudaMalloc, 2 MiB aligned)
    - Export IPC handles + metadata via SharedMemory (v0.5.0 protocol, once at startup)
    - Per-frame: accept raw GPU pointer, async D2D memcpy to ring slot, record IPC event
    - 7-step cleanup: shutdown signal → events → stream → SHM close → grace → free → unlink

    Performance:
    - Initialization: ~1ms (buffer alloc + handle export)
    - Per-frame overhead: ~177µs at 512x512, ~219µs at 1080p @ 60 FPS (async D2D + stream_synchronize)
    - Zero CPU memory copies (GPU-direct)
    """

    def __init__(
        self,
        shm_name: str,
        height: int,
        width: int,
        channels: int = 4,
        dtype: str = "uint8",
        num_slots: int = 2,
        debug: bool = False,
        device: int = 0,
    ) -> None:
        """Initialize CUDA IPC exporter.

        Args:
            shm_name: SharedMemory name. Must match the TD Receiver's Ipcmemname parameter.
            height: Frame height in pixels.
            width: Frame width in pixels.
            channels: Number of channels (default: 4 for BGRA/RGBA).
            dtype: Data type string: "float32", "float16", or "uint8" (default: "uint8").
            num_slots: Ring buffer slot count (default: 2 for double-buffering). Range: 1-10.
            debug: Enable verbose per-frame performance logging.
            device: CUDA device index to use (default: 0). Sender and receiver must
                    use the same device; IPC handles are device-scoped.

        Raises:
            ValueError: If dtype is unsupported or num_slots is out of range.
        """
        if dtype not in _DTYPE_TO_KIND_BITS:
            raise ValueError(f"Unsupported dtype: {dtype!r}. Must be one of {list(_DTYPE_TO_KIND_BITS)}")
        if not (0 < num_slots <= 10):
            raise ValueError(f"num_slots must be 1-10, got {num_slots}")

        self.shm_name = shm_name
        self.height = height
        self.width = width
        self.channels = channels
        self.dtype = dtype
        self.num_slots = num_slots
        self.debug = debug
        self.device = device

        # CUDALINK_EXPORT_SYNC=1 restores the old behaviour of blocking the CPU on
        # ipc_stream after each record_event(). Default off: the ~13-100µs sync is
        # redundant for consumers using the stream-ordered get_frame(stream=...) path.
        self._export_sync: bool = os.getenv("CUDALINK_EXPORT_SYNC", "0") == "1"

        # CUDALINK_USE_GRAPHS=1: capture the memcpy_async into a 1-node CUDA Graph and
        # replay via graph_launch.  IPC events (cudaEventInterprocess) and external
        # stream_wait_event deps cannot be captured, so the graph contains only the D2D
        # memcpy.  Per-frame cost when source_sync not used:
        #   graph_launch (1 WDDM) + record_event (1 WDDM) = 2 submissions vs 3 legacy
        # When record_source_sync() has been called at least once, stream_wait_event is
        # issued before graph_launch (3 WDDM — same as legacy).
        # Requires CUDA 12.x runtime (Python side). On by default; set CUDALINK_USE_GRAPHS=0
        # to revert to the legacy 3-submission stream path.
        self._use_graphs: bool = os.getenv("CUDALINK_USE_GRAPHS", "1") == "1"
        self._graphs_disabled: bool = False  # set True if build/launch fails at runtime
        self._source_sync_recorded: bool = False  # set True on first record_source_sync()
        # One CUDAGraphExec_t + template CUDAGraph_t per ring slot.
        # Template is kept alive so node handles remain valid for SetParams calls.
        self._graph_execs: list[CUDAGraphExec_t | None] = [None] * num_slots
        self._graph_templates: list[CUDAGraph_t | None] = [None] * num_slots
        self._graph_memcpy_nodes: list[CUDAGraphNode_t | None] = [None] * num_slots
        # CUDALINK_EXPORT_PROFILE=1: enables fine-grained per-region sub-timers in export_frame.
        # Mirrors td_exporter/CUDAIPCExtension.py's same knob. Forces debug=True.
        self._export_profile: bool = os.getenv("CUDALINK_EXPORT_PROFILE", "0") == "1"
        # CUDALINK_EXPORT_FLUSH_PROBE: calls cudaStreamQuery after check_sticky_error when
        # _export_sync=False. Forces WDDM-deferred commands to submit without CPU blocking.
        # Default ON per Phase 3 decision (2026-05-04): ~12 µs/frame cost, collapses
        # Windows Task Manager 3D-engine reading from ~65% to ~7% on rigs where WDDM
        # defers submissions. NVML true compute load is unchanged. Set to "0" to disable.
        self._export_flush_probe: bool = os.getenv("CUDALINK_EXPORT_FLUSH_PROBE", "1") == "1"
        # F9 — CUDALINK_ACTIVATION_BARRIER=1: read cudalink_activation_barrier SHM on each
        # export_frame and skip publishing while a TD-side Sender is in its activation window.
        # Cross-process backpressure mechanism — no CUDA stream coupling.
        self._barrier_enabled: bool = os.getenv("CUDALINK_ACTIVATION_BARRIER", "0") == "1"
        self._barrier_stale_ns: int = int(os.getenv("CUDALINK_BARRIER_STALE_NS", str(5 * 1_000_000_000)))
        self._barrier_shm: SharedMemory | None = None
        self._barrier_skip_log_last_ns: int = 0
        self._barrier_stale_log_last_ns: int = 0
        if self._export_profile:
            self.debug = True  # profile mode requires timing path (mirrors TD L248-249)

        # Derived sizes
        itemsize = _DTYPE_ITEMSIZE_MAP[dtype]
        self.data_size = height * width * channels * itemsize  # Actual data bytes

        # CUDA state
        self.cuda: CUDARuntimeAPI | None = None
        self._initialized = False
        self.ipc_stream = None  # Dedicated non-blocking CUDA stream
        self.source_sync_event = None  # Cross-stream sync event (GPU-side, non-blocking CPU)

        # Ring buffer state (arrays sized by num_slots)
        self.dev_ptrs: list = [None] * num_slots  # GPU buffer pointers
        self.ipc_handles: list = [None] * num_slots  # IPC memory handles
        self.ipc_events: list = [None] * num_slots  # IPC events for GPU sync
        self.ipc_event_handles: list = [None] * num_slots  # Exportable event handles
        self.write_idx: int = 0  # Monotonic frame counter

        # SharedMemory
        self.shm_handle: SharedMemory | None = None
        self.buffer_size: int = self.data_size  # Will be 2MiB-aligned in initialize()

        # Performance tracking
        self.frame_count: int = 0
        self.total_memcpy_us: float = 0.0
        self.total_export_us: float = 0.0
        self.total_stream_wait_us: float = 0.0
        self.total_record_event_us: float = 0.0
        self.total_shm_write_us: float = 0.0
        self.total_sync_us: float = 0.0
        self.total_sticky_check_us: float = 0.0
        self.total_flush_probe_us: float = 0.0

        # Cached SharedMemory offsets (computed once in initialize(), constant thereafter)
        self._ts_offset: int = 0
        self._shutdown_offset: int = 0

        # C2: device-affinity validation
        # CUDALINK_STRICT_DEVICE=1 raises ValueError on mismatch; default warns+continues.
        self._strict_device: bool = os.getenv("CUDALINK_STRICT_DEVICE", "0") == "1"
        self._source_sync_device_warned: bool = False  # emit at most one log per instance
        # Cache of ptr values already validated (capped at 8 — covers typical buffer-rotation)
        self._ptr_device_cache: set[int] = set()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Allocate GPU ring buffer, create IPC handles, write to SharedMemory.

        Must be called before export_frame(). Safe to call multiple times
        (idempotent — returns True if already initialized).

        Returns:
            True if initialization succeeded, False on error.
        """
        if self._initialized:
            logger.debug("Already initialized")
            return True

        try:
            # Load CUDA runtime bound to the requested device
            self.cuda = get_cuda_runtime(device=self.device)
            actual_device = self.cuda.get_device()
            if actual_device != self.device:
                raise RuntimeError(
                    f"Device mismatch: requested device {self.device} but CUDA context "
                    f"is bound to device {actual_device}. Ensure no other code calls "
                    "cudaSetDevice() with a different index before initialize()."
                )
            logger.info("Loaded CUDA runtime on device %d", actual_device)

            # Create or reuse dedicated non-blocking IPC stream.
            # cudaStreamNonBlocking (0x01) prevents the default stream from
            # implicitly synchronising with this stream. Default is high-priority
            # so the D2D memcpy preempts lower-priority compute work in the TD context.
            # F7 — CUDALINK_LIB_STREAM_PRIO=normal: drop to default-priority stream.
            # Use when a TD-side Sender-B coexists with this Python producer in the
            # same machine and the high-priority stream contends with TD init.
            if self.ipc_stream is None:
                lib_stream_high_prio = os.environ.get("CUDALINK_LIB_STREAM_PRIO", "high") != "normal"
                if lib_stream_high_prio:
                    self.ipc_stream = self.cuda.create_stream_with_priority(flags=0x01)
                    logger.info("Created IPC stream (high-priority): 0x%016x", int(self.ipc_stream.value))
                else:
                    self.ipc_stream = self.cuda.create_stream(flags=0x01)
                    logger.info("Created IPC stream (normal-priority): 0x%016x", int(self.ipc_stream.value))
            else:
                logger.debug("Reusing IPC stream: 0x%016x", int(self.ipc_stream.value))

            # Create cross-stream sync event for GPU-side producer ordering.
            # ipc_stream is non-blocking and does NOT implicitly sync with any other stream,
            # so callers MUST either call record_source_sync() or torch.cuda.synchronize()
            # before export_frame(). The event enables the GPU-side-only path.
            if self.source_sync_event is None:
                self.source_sync_event = self.cuda.create_sync_event()
                logger.info("Created cross-stream source sync event")

            # Apply 2 MiB alignment (NVIDIA requirement: prevents information disclosure)
            alignment = 2 * 1024 * 1024
            self.buffer_size = ((self.data_size + alignment - 1) // alignment) * alignment
            logger.info(
                "Buffer: %.1f KB data, %.1f KB aligned, %d slots",
                self.data_size / 1024,
                self.buffer_size / 1024,
                self.num_slots,
            )

            # PHASE 1: Allocate GPU ring buffer + create IPC handles
            for slot in range(self.num_slots):
                self.dev_ptrs[slot] = self.cuda.malloc(self.buffer_size)
                logger.info(
                    "Slot %d: allocated %.1f KB at 0x%016x",
                    slot,
                    self.buffer_size / 1024,
                    self.dev_ptrs[slot].value,
                )

                # Memory handle (once at startup — reused every frame)
                self.ipc_handles[slot] = self.cuda.ipc_get_mem_handle(self.dev_ptrs[slot])
                logger.debug("Slot %d: created IPC mem handle (64 bytes)", slot)

                # Event handle for GPU-side synchronization
                self.ipc_events[slot] = self.cuda.create_ipc_event()
                self.ipc_event_handles[slot] = self.cuda.ipc_get_event_handle(self.ipc_events[slot])
                logger.debug("Slot %d: created IPC event (64 bytes)", slot)

            logger.info("Created %d IPC buffer slots with GPU-side sync", self.num_slots)

            # PHASE 2: Create SharedMemory
            shm_size = (
                SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE) + SHUTDOWN_FLAG_SIZE + METADATA_SIZE + TIMESTAMP_SIZE
            )
            try:
                self.shm_handle = SharedMemory(name=self.shm_name)
                logger.info("Opened existing SharedMemory: %s", self.shm_name)
            except FileNotFoundError:
                self.shm_handle = SharedMemory(name=self.shm_name, create=True, size=shm_size)
                logger.info("Created SharedMemory: %s (%d bytes)", self.shm_name, shm_size)

            # PHASE 3: Write protocol header + IPC handles + metadata
            self._write_handles_to_shm()
            self._write_metadata_to_shm()

            # Cache constant SharedMemory offsets so export_frame() avoids per-frame arithmetic
            self._ts_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE) + SHUTDOWN_FLAG_SIZE + METADATA_SIZE

            self._initialized = True

            # CUDA Graphs build (after IPC stream / events / ring buffer are ready).
            # Gated on cudart >= 11.4 (cudaGraphInstantiateWithFlags + the
            # EventRecordNodeSetEvent / EventWaitNodeSetEvent APIs all require 11.4+).
            # cudaGraphInstantiateWithFlags was introduced specifically to avoid the
            # 5-arg (10.0-11.8) vs 3-arg (12.0+) ABI split in cudaGraphInstantiate.
            if self._use_graphs:
                try:
                    rt_version = self.cuda.get_runtime_version()
                except (RuntimeError, OSError) as exc:
                    rt_version = 0
                    logger.warning("cudaRuntimeGetVersion failed (%s) — disabling graphs", exc)
                if rt_version >= CUDART_GRAPHS_MIN_VERSION:
                    self._build_export_graphs()
                else:
                    logger.warning(
                        "CUDALINK_USE_GRAPHS=1 ignored: cudart %d < %d "
                        "(cudaGraphInstantiateWithFlags requires 11.4+). "
                        "Falling back to legacy stream path.",
                        rt_version,
                        CUDART_GRAPHS_MIN_VERSION,
                    )
                    self._graphs_disabled = True

            logger.info("Initialization complete — ready for zero-copy GPU transfer")
            return True

        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Initialization failed: %s", e)
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Protocol write helpers
    # ------------------------------------------------------------------

    def _write_handles_to_shm(self) -> None:
        """Write v0.5.0 protocol header + IPC handles to SharedMemory.

        Layout:
            [0-3]    magic (uint32 LE) = 0x43495044
            [4-11]   version (uint64 LE) — incremented on each init
            [12-15]  num_slots (uint32 LE)
            [16-19]  write_idx (uint32 LE) — initialized to 0

        Per slot (128 bytes):
            [base..+63]   cudaIpcMemHandle_t (64 bytes)
            [base+64..+64] cudaIpcEventHandle_t (64 bytes)

        Footer:
            shutdown_flag (1 byte) = 0
        """
        if self.shm_handle is None or not all(self.ipc_handles):
            return

        # Increment version (detect re-initialization from consumer side)
        try:
            current_version = struct.unpack_from("<Q", self.shm_handle.buf, 4)[0]
        except (struct.error, ValueError, IndexError):
            current_version = 0
        new_version = current_version + 1

        # Write header
        struct.pack_into("<I", self.shm_handle.buf, 0, PROTOCOL_MAGIC)
        struct.pack_into("<Q", self.shm_handle.buf, 4, new_version)
        struct.pack_into("<I", self.shm_handle.buf, 12, self.num_slots)
        struct.pack_into("<I", self.shm_handle.buf, 16, 0)  # write_idx = 0 initially

        # Write per-slot handles
        for slot in range(self.num_slots):
            base_offset = SHM_HEADER_SIZE + (slot * SLOT_SIZE)

            mem_handle_bytes = bytes(self.ipc_handles[slot].internal)
            self.shm_handle.buf[base_offset : base_offset + 64] = mem_handle_bytes

            if self.ipc_event_handles[slot]:
                event_handle_bytes = bytes(self.ipc_event_handles[slot].reserved)
                self.shm_handle.buf[base_offset + 64 : base_offset + 128] = event_handle_bytes

        # Initialize shutdown flag to 0 and cache its offset for export_frame() reassertion
        self._shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
        struct.pack_into("<B", self.shm_handle.buf, self._shutdown_offset, 0)

        logger.info("Wrote IPC handles v%d to SharedMemory", new_version)

    def _write_metadata_to_shm(self) -> None:
        """Write texture metadata to the extended protocol region.

        Layout (20 bytes after shutdown flag):
            +0   width         (uint32)
            +4   height        (uint32)
            +8   num_comps     (uint32)
            +12  format_kind   (uint8)   — cudaChannelFormatKind: 0=Signed,1=Unsigned,2=Float
            +13  bits_per_comp (uint8)   — 8/16/32/64
            +14  flags         (uint16)  — bit0=bfloat16; rest reserved=0
            +16  data_size     (uint32)  — actual bytes (before 2MiB alignment)
        """
        if self.shm_handle is None or self.data_size == 0:
            return

        shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
        metadata_offset = shutdown_offset + SHUTDOWN_FLAG_SIZE

        kind, bits, flags = _DTYPE_TO_KIND_BITS[self.dtype]

        struct.pack_into("<I", self.shm_handle.buf, metadata_offset, self.width)
        struct.pack_into("<I", self.shm_handle.buf, metadata_offset + 4, self.height)
        struct.pack_into("<I", self.shm_handle.buf, metadata_offset + 8, self.channels)
        _ST_BBH.pack_into(self.shm_handle.buf, metadata_offset + 12, kind, bits, flags)
        struct.pack_into("<I", self.shm_handle.buf, metadata_offset + 16, self.data_size)

        logger.debug(
            "Wrote metadata: %dx%dx%d, dtype=%s (kind=%d bits=%d flags=0x%04x), data_size=%dB",
            self.width,
            self.height,
            self.channels,
            self.dtype,
            kind,
            bits,
            flags,
            self.data_size,
        )

    # ------------------------------------------------------------------
    # CUDA Graph helpers (Phase 2, CUDALINK_USE_GRAPHS=1)
    # ------------------------------------------------------------------

    def _build_export_graphs(self) -> None:
        """Capture the D2D memcpy into a 1-node CUDA Graph exec per ring slot.

        Graph topology per slot (1 node):
            MemcpyNode(D2D, ring_slot[slot] ← placeholder_src)

        stream_wait_event is NOT captured: external events (recorded outside this
        capture) do not produce EventWait nodes in global-mode capture.
        record_event on IPC events is NOT captured: cudaEventInterprocess events
        raise cudaErrorStreamCaptureUnsupported (error 900) during capture.

        Per-frame cost:
          - source_sync not used (common): graph_launch + record_event = 2 WDDM
          - source_sync used: stream_wait_event + graph_launch + record_event = 3 WDDM

        On failure the stream is restored to normal mode before returning so that
        the legacy fallback path can use it without error.
        """
        assert self.cuda is not None
        assert self.ipc_stream is not None

        placeholder_src = self.dev_ptrs[0]

        for slot in range(self.num_slots):
            capture_started = False
            try:
                self.cuda.stream_begin_capture(self.ipc_stream, mode=0)
                capture_started = True
                self.cuda.memcpy_async(
                    dst=self.dev_ptrs[slot],
                    src=placeholder_src,
                    count=self.data_size,
                    kind=3,  # D2D
                    stream=self.ipc_stream,
                )
                template_graph = self.cuda.stream_end_capture(self.ipc_stream)
                capture_started = False

                nodes = self.cuda.graph_get_nodes(template_graph)
                if len(nodes) != 1:
                    self.cuda.graph_destroy(template_graph)
                    raise RuntimeError(f"Unexpected graph node count {len(nodes)} (expected 1: MemcpyNode).")
                memcpy_node = nodes[0]

                graph_exec = self.cuda.graph_instantiate(template_graph)
                # Keep template alive: node handles from the template must remain
                # valid for cudaGraphExecMemcpyNodeSetParams1D per-frame updates.
                # Template is destroyed in _destroy_export_graphs().

                self._graph_execs[slot] = graph_exec
                self._graph_templates[slot] = template_graph
                self._graph_memcpy_nodes[slot] = memcpy_node
                logger.debug("Built export graph for slot %d (1-node: Memcpy)", slot)

            except (RuntimeError, OSError) as exc:
                if capture_started:
                    try:
                        abandoned_graph = self.cuda.stream_end_capture(self.ipc_stream)
                        self.cuda.graph_destroy(abandoned_graph)
                    except (RuntimeError, OSError):
                        pass
                logger.warning(
                    "CUDA Graph build failed for slot %d (%s) — "
                    "disabling graphs for this exporter instance and falling back to "
                    "legacy stream path. Set CUDALINK_USE_GRAPHS=0 to suppress.",
                    slot,
                    exc,
                )
                self._graphs_disabled = True
                self._destroy_export_graphs()
                return

        logger.info("CUDA export graphs built for %d slots (CUDALINK_USE_GRAPHS=1)", self.num_slots)

    def _destroy_export_graphs(self) -> None:
        """Destroy all CUDA Graph exec objects and their templates (called from cleanup())."""
        if self.cuda is None:
            return
        for slot, graph_exec in enumerate(self._graph_execs):
            if graph_exec is not None:
                try:
                    self.cuda.graph_exec_destroy(graph_exec)
                    logger.debug("Destroyed export graph exec slot %d", slot)
                except (RuntimeError, OSError) as e:
                    logger.error("Error destroying graph exec slot %d: %s", slot, e)
                self._graph_execs[slot] = None
        for slot, template in enumerate(getattr(self, "_graph_templates", [])):
            if template is not None:
                with contextlib.suppress(RuntimeError, OSError):
                    self.cuda.graph_destroy(template)
                self._graph_templates[slot] = None
        self._graph_memcpy_nodes = [None] * self.num_slots

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def record_source_sync(self, producer_stream_handle: int) -> None:
        """Record sync event on the producer's CUDA stream (GPU-side, non-blocking CPU).

        Call this AFTER your GPU kernel writes to the source buffer, BEFORE export_frame().
        export_frame() will make ipc_stream wait for this event GPU-side before the D2D
        memcpy, ensuring source data is ready without blocking the CPU.

        This replaces ``torch.cuda.synchronize()`` and saves ~0.2-0.5ms per frame.

        ``ipc_stream`` is created with cudaStreamNonBlocking and does NOT implicitly
        synchronize with any other stream (including the legacy default stream). Without
        this call, export_frame() has no ordering guarantee with the caller's GPU work.

        Args:
            producer_stream_handle: Raw CUDA stream integer. Examples:
                - PyTorch:  ``torch.cuda.current_stream().cuda_stream``
                - CuPy:     ``cupy.cuda.get_current_stream().ptr``
                - Raw CUDA: the ``cudaStream_t`` cast to int

        If this method is never called, ``source_sync_event`` stays in its initial
        unrecorded state and stream_wait_event() in export_frame() is a benign no-op
        (backward compatible). The caller is then responsible for their own sync.
        """
        if self.source_sync_event is not None and self.cuda is not None:
            # C2: opportunistic stream-device check. CUDA Runtime < 12.8 has no
            # cudaStreamGetDevice; we validate via the current context device instead.
            # Emitted at most once per exporter instance to avoid log spam.
            if not self._source_sync_device_warned:
                current_device = self.cuda.get_device()
                if current_device != self.device:
                    msg = (
                        f"record_source_sync: current CUDA device ({current_device}) "
                        f"does not match exporter device ({self.device}). "
                        "Call cudaSetDevice(device) before creating your producer stream. "
                        "Set CUDALINK_STRICT_DEVICE=1 to raise instead of warn."
                    )
                    if self._strict_device:
                        raise ValueError(msg)
                    logger.error(msg)
                    self._source_sync_device_warned = True
            self.cuda.record_event(self.source_sync_event, CUDAStream_t(producer_stream_handle))
            self._source_sync_recorded = True

    def export_frame(self, gpu_ptr: int, size: int) -> bool:
        """Export one frame from GPU memory via IPC ring buffer.

        Args:
            gpu_ptr: Source GPU pointer (from tensor.data_ptr()).
            size: Buffer size in bytes (from tensor.nelement() * tensor.element_size()).

        Returns:
            True if export succeeded, False on error.
        """
        if not self._initialized:
            logger.warning("Not initialized — call initialize() first")
            return False
        if size != self.data_size:
            logger.error("Size mismatch: expected %d, got %d", self.data_size, size)
            return False
        # F9 — skip publish if a TD-side Sender is in its activation window.
        if self._barrier_enabled and self._check_activation_barrier():
            # F10 — reassert the per-frame heartbeat even on the skip path.
            # The consumer reads shutdown_flag == 1 as "producer gone"; bypassing the
            # success-path heartbeat write on skip frames would leave any stale 1-byte
            # uncleared and trip a false "Sender shutdown detected" on the TD receiver.
            if self.shm_handle is not None and self._shutdown_offset:
                with contextlib.suppress(OSError, BufferError):
                    self.shm_handle.buf[self._shutdown_offset] = 0
            return False
        debug = self.debug
        if debug:
            frame_start = time.perf_counter()
        try:
            slot = self.write_idx % self.num_slots

            # C2: validate source pointer's device and memory type on first appearance.
            # Cache keyed by pointer integer (cap at 8 to cover typical buffer-rotation).
            gpu_ptr_int = gpu_ptr if isinstance(gpu_ptr, int) else int(gpu_ptr)
            if gpu_ptr_int not in self._ptr_device_cache:
                attrs = self.cuda.pointer_get_attributes(gpu_ptr_int)
                if attrs.type not in (2, 3):  # 2=device, 3=managed (both valid for D2D)
                    msg = (
                        f"export_frame: gpu_ptr 0x{gpu_ptr_int:016x} is not device/managed "
                        f"memory (type={attrs.type}). Pass a GPU-resident pointer. "
                        "Set CUDALINK_STRICT_DEVICE=1 to raise instead of warn."
                    )
                    if self._strict_device:
                        raise ValueError(msg)
                    logger.error(msg)
                elif attrs.device != self.device:
                    msg = (
                        f"export_frame: gpu_ptr 0x{gpu_ptr_int:016x} belongs to device "
                        f"{attrs.device}, but exporter is bound to device {self.device}. "
                        "Set CUDALINK_STRICT_DEVICE=1 to raise instead of warn."
                    )
                    if self._strict_device:
                        raise ValueError(msg)
                    logger.error(msg)
                if len(self._ptr_device_cache) < 8:
                    self._ptr_device_cache.add(gpu_ptr_int)

            # --- GPU copy + sync: graph path or legacy path ---
            #
            # Graph path (CUDALINK_USE_GRAPHS=1): replays a 1-node graph (MemcpyNode).
            # stream_wait_event is issued before graph_launch only when record_source_sync()
            # has been called (tracked by _source_sync_recorded); otherwise it is skipped
            # (the event is in its initial "complete" state — no ordering needed).
            # record_event is issued after graph_launch (IPC events not capturable).
            #   - source_sync not used: graph_launch + record_event = 2 WDDM (vs 3 legacy)
            #   - source_sync used:     stream_wait + graph_launch + record_event = 3 WDDM
            if self._use_graphs and not self._graphs_disabled:
                if debug:
                    _t = time.perf_counter()
                try:
                    self.cuda.graph_exec_memcpy_node_set_params_1d(
                        self._graph_execs[slot],
                        self._graph_memcpy_nodes[slot],
                        dst=self.dev_ptrs[slot],
                        src=c_void_p(gpu_ptr),
                        count=self.data_size,
                        kind=3,
                    )
                    if self._source_sync_recorded and self.source_sync_event is not None:
                        self.cuda.stream_wait_event(self.ipc_stream, self.source_sync_event, 0)
                    self.cuda.graph_launch(self._graph_execs[slot], self.ipc_stream)
                    if self.ipc_events[slot]:
                        self.cuda.record_event(self.ipc_events[slot], stream=self.ipc_stream)
                except (RuntimeError, OSError) as _graph_err:
                    logger.warning(
                        "Graph launch failed (%s) — disabling graphs, retrying via legacy path",
                        _graph_err,
                    )
                    self._graphs_disabled = True
                    goto_legacy = True
                else:
                    goto_legacy = False
                if debug:
                    self.total_memcpy_us += (time.perf_counter() - _t) * 1_000_000
            else:
                goto_legacy = True

            if goto_legacy:
                # Legacy path: 3 separate WDDM submissions per frame.
                # GPU-side wait on source_sync_event is a no-op if record_source_sync()
                # was never called (event stays in its initial "complete" state).
                if debug:
                    _t = time.perf_counter()
                if self.source_sync_event is not None:
                    self.cuda.stream_wait_event(self.ipc_stream, self.source_sync_event, 0)
                if debug:
                    self.total_stream_wait_us += (time.perf_counter() - _t) * 1_000_000

                if debug:
                    memcpy_start = time.perf_counter()
                self.cuda.memcpy_async(
                    dst=self.dev_ptrs[slot],
                    src=c_void_p(gpu_ptr),
                    count=self.data_size,
                    kind=3,  # cudaMemcpyDeviceToDevice
                    stream=self.ipc_stream,
                )
                if debug:
                    self.total_memcpy_us += (time.perf_counter() - memcpy_start) * 1_000_000

                if debug:
                    _t = time.perf_counter()
                if self.ipc_events[slot]:
                    self.cuda.record_event(self.ipc_events[slot], stream=self.ipc_stream)
                if debug:
                    self.total_record_event_us += (time.perf_counter() - _t) * 1_000_000

            # Optional CPU-blocking sync after record_event. Disabled by default.
            #
            # When enabled (CUDALINK_EXPORT_SYNC=1): blocks the CPU until the GPU has
            # executed the memcpy + record_event, guaranteeing query_event() returns True
            # on the first poll. Cost: ~13µs @ 512², ~100µs @ 1080p per frame.
            #
            # When disabled (default): the GPU event is recorded asynchronously.
            # Consumers using the stream-ordered path (get_frame(stream=...)) issue
            # cudaStreamWaitEvent, which correctly waits for the event-record to execute
            # regardless of whether the CPU has synced. The _wait_for_slot() timeout
            # guards the polling path. Skipping this sync saves ~13-100µs/frame.
            if self._export_sync:
                if debug and self._export_profile:
                    _t_sync = time.perf_counter()
                self.cuda.stream_synchronize(self.ipc_stream)
                if debug and self._export_profile:
                    self.total_sync_us += (time.perf_counter() - _t_sync) * 1_000_000

            if debug and self._export_profile:
                _t_sticky = time.perf_counter()
            self.cuda.check_sticky_error("export_frame")
            if debug and self._export_profile:
                self.total_sticky_check_us += (time.perf_counter() - _t_sticky) * 1_000_000

            # WDDM deferred-submission probe: forces pending GPU work to submit without
            # blocking. Per CUDA Handbook p3/pg56, WDDM buffers commands until a flush;
            # cudaStreamQuery triggers that flush. Only active when EXPORT_FLUSH_PROBE=1
            # and EXPORT_SYNC=0 (if sync is on, the stream is already flushed above).
            if self._export_flush_probe and not self._export_sync:
                if debug and self._export_profile:
                    _t_fp = time.perf_counter()
                self.cuda.stream_query(self.ipc_stream)
                if debug and self._export_profile:
                    self.total_flush_probe_us += (time.perf_counter() - _t_fp) * 1_000_000

            # Write timestamp, clear shutdown_flag, then publish write_idx LAST.
            # Ordering matters: the consumer reads shutdown_flag BEFORE write_idx, so
            # clearing it before incrementing write_idx ensures the consumer always sees
            # shutdown_flag=0 when it detects a new frame (atomicity improvement).
            if debug:
                _t = time.perf_counter()
            self.write_idx += 1
            _ST_F64.pack_into(self.shm_handle.buf, self._ts_offset, time.perf_counter())
            self.shm_handle.buf[self._shutdown_offset] = 0
            _release_fence()  # C3: release barrier — shutdown_flag visible before write_idx
            _ST_U32.pack_into(self.shm_handle.buf, 16, self.write_idx)  # publish last
            if debug:
                self.total_shm_write_us += (time.perf_counter() - _t) * 1_000_000

            self.frame_count += 1

            if debug:
                frame_time = (time.perf_counter() - frame_start) * 1_000_000
                self.total_export_us += frame_time

                if self.frame_count % 97 == 0:
                    n = self.frame_count
                    logger.debug(
                        "Frame %d: slot=%d | stream_wait=%.1fus memcpy=%.1fus "
                        "record_event=%.1fus shm_write=%.1fus | total=%.1fus",
                        n,
                        slot,
                        self.total_stream_wait_us / n,
                        self.total_memcpy_us / n,
                        self.total_record_event_us / n,
                        self.total_shm_write_us / n,
                        self.total_export_us / n,
                    )
                    if self._export_profile:
                        avg_wait = self.total_stream_wait_us / n
                        avg_memcpy = self.total_memcpy_us / n
                        avg_record = self.total_record_event_us / n
                        avg_sync = self.total_sync_us / n
                        avg_sticky = self.total_sticky_check_us / n
                        avg_fp = self.total_flush_probe_us / n
                        avg_shm = self.total_shm_write_us / n
                        avg_total = self.total_export_us / n
                        avg_unacc = avg_total - (
                            avg_wait + avg_memcpy + avg_record + avg_sync + avg_sticky + avg_fp + avg_shm
                        )
                        logger.debug(
                            "Frame %d [PROFILE] pre=0.0us interop=0.0us post=0.0us"
                            " memcpy=%.1fus record=%.1fus sync=%.1fus"
                            " sticky=%.1fus flush_probe=%.1fus shm=%.1fus unacc=%.1fus",
                            n,
                            avg_memcpy,
                            avg_record,
                            avg_sync,
                            avg_sticky,
                            avg_fp,
                            avg_shm,
                            avg_unacc,
                        )

            return True

        except (OSError, RuntimeError) as e:
            logger.error("Export failed: %s", e)
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _is_cuda_context_valid(self) -> bool:
        """Check if CUDA context is still valid.

        The CUDA context may be destroyed before cleanup() is called
        (e.g., when the Python process is terminating). Checking this
        prevents spurious CUDA errors during cleanup.
        """
        if self.cuda is None:
            return False
        try:
            self.cuda.cudart.cudaGetLastError()
            return True
        except (OSError, RuntimeError):
            return False

    def _check_activation_barrier(self) -> bool:
        """Return True if the activation barrier is held and this frame should be skipped.

        Lazily opens the segment on first call. Applies a stale-timeout so a Sender
        that crashes mid-init cannot block the producer indefinitely.
        """
        if self._barrier_shm is None:
            try:
                self._barrier_shm = _ab_open(create=False)
            except FileNotFoundError:
                return False  # no Sender has ever activated — fast path
        try:
            active_count, last_change_ns, _ = _ab_read(self._barrier_shm)
        except (OSError, RuntimeError, struct.error):
            return False
        if active_count <= 0:
            return False
        now_ns = time.monotonic_ns()
        if now_ns - last_change_ns > self._barrier_stale_ns:
            # Sender crashed mid-init and never decremented — stale, ignore.
            if now_ns - self._barrier_stale_log_last_ns > 1_000_000_000:
                logger.warning(
                    "[ACTIVATION_BARRIER] stale barrier (count=%d, age=%.1fs) — ignoring",
                    active_count,
                    (now_ns - last_change_ns) / 1e9,
                )
                self._barrier_stale_log_last_ns = now_ns
            return False
        with contextlib.suppress(OSError, RuntimeError, struct.error):
            _ab_bump(self._barrier_shm)
        if now_ns - self._barrier_skip_log_last_ns > 1_000_000_000:
            logger.info("[ACTIVATION_BARRIER] skipping publish (active_count=%d)", active_count)
            self._barrier_skip_log_last_ns = now_ns
        return True

    def cleanup(self) -> None:
        """Cleanup all CUDA IPC resources.

        7-step shutdown sequence (order is critical):
        1. Signal shutdown to consumer via SharedMemory flag
        2. Destroy IPC events (sender-side resources, safe to destroy)
        3. Destroy IPC stream
        4. Close SharedMemory (don't unlink yet — consumer may still read flag)
        5. Grace period (100ms) for consumer to detect shutdown and close handles
        6. Free GPU buffers (cudaFree blocks until consumer closes IPC handles)
        7. Unlink SharedMemory (producer owns it and is responsible for cleanup)
        """
        # Double-cleanup guard: skip if already cleaned up
        if not self._initialized and self.shm_handle is None:
            return

        cuda_valid = self._is_cuda_context_valid()
        if not cuda_valid:
            logger.warning("CUDA context already destroyed — skipping GPU cleanup")

        # STEP 1: Signal shutdown to consumer
        if self.shm_handle:
            try:
                shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
                struct.pack_into("<B", self.shm_handle.buf, shutdown_offset, 1)
                logger.info("Shutdown signal sent to consumer")
            except (OSError, BufferError) as e:
                logger.warning("Could not write shutdown signal: %s", e)

        # STEP 1b: Zero out IPC handle bytes so any reader sees invalid handles.
        # On Windows, unlink() is a no-op (SharedMemory uses CreateFileMapping kernel
        # objects), so the SharedMemory may persist with stale non-zero handles that
        # pass the all-zero validation check and trigger error 201 on open.
        if self.shm_handle:
            try:
                for slot in range(self.num_slots):
                    base_offset = SHM_HEADER_SIZE + (slot * SLOT_SIZE)
                    self.shm_handle.buf[base_offset : base_offset + SLOT_SIZE] = b"\x00" * SLOT_SIZE
                logger.debug("Zeroed IPC handle bytes in SharedMemory")
            except (OSError, BufferError) as e:
                logger.warning("Could not zero IPC handles: %s", e)

        # STEP 1c: Destroy CUDA Graph execs (before events/stream they reference)
        if cuda_valid and getattr(self, "_use_graphs", False):
            self._destroy_export_graphs()

        # STEP 2: Destroy IPC events + cross-stream sync event
        if cuda_valid and self.cuda and self.ipc_events:
            for slot, event in enumerate(self.ipc_events):
                if event:
                    try:
                        self.cuda.destroy_event(event)
                        logger.debug("Destroyed IPC event slot %d", slot)
                    except (RuntimeError, OSError) as e:
                        logger.error("Error destroying event slot %d: %s", slot, e)

        if cuda_valid and self.cuda and self.source_sync_event:
            try:
                self.cuda.destroy_event(self.source_sync_event)
                logger.debug("Destroyed cross-stream sync event")
            except (RuntimeError, OSError) as e:
                logger.error("Error destroying sync event: %s", e)
            self.source_sync_event = None

        # STEP 3: Destroy IPC stream
        if cuda_valid and self.cuda and self.ipc_stream:
            try:
                self.cuda.destroy_stream(self.ipc_stream)
                logger.info("Destroyed IPC stream")
                self.ipc_stream = None
            except (RuntimeError, OSError) as e:
                logger.error("Error destroying IPC stream: %s", e)

        # STEP 4: Close SharedMemory (don't unlink yet)
        if self.shm_handle:
            try:
                self.shm_handle.close()
                logger.debug("Closed SharedMemory")
            except (OSError, BufferError) as e:
                logger.error("Error closing SharedMemory: %s", e)
            self.shm_handle = None

        # STEP 5: Grace period for consumer to close IPC handles
        if cuda_valid:
            time.sleep(0.1)  # 100ms for consumer to detect shutdown and close handles

        # STEP 6: Free GPU buffers (safe now that consumer has closed handles).
        # cudaFree on an IPC-exported pointer blocks until all receivers call
        # cudaIpcCloseMemHandle. On Windows WDDM, a crashed receiver that never
        # closed its handle will cause cudaFree to hang indefinitely. We run each
        # free in a daemon thread with a 0.5 s watchdog; on timeout we log and
        # continue — the OS reclaims VRAM when the process exits regardless.
        if cuda_valid and self.cuda and self.dev_ptrs:
            for slot, dev_ptr in enumerate(self.dev_ptrs):
                if dev_ptr:

                    def _free(ptr: c_void_p, s: int = slot) -> None:
                        try:
                            self.cuda.free(ptr)
                            logger.debug("Freed GPU buffer slot %d", s)
                        except (RuntimeError, OSError) as e:
                            logger.error("Error freeing GPU buffer slot %d: %s", s, e)

                    t = threading.Thread(target=_free, args=(dev_ptr,), daemon=True)
                    t.start()
                    t.join(timeout=0.5)
                    if t.is_alive():
                        logger.warning(
                            "cudaFree slot %d timed out (0x%016x) — receiver may not have closed "
                            "the IPC handle. Leaking GPU memory; OS will reclaim on process exit.",
                            slot,
                            dev_ptr.value,
                        )

        # STEP 7: Unlink SharedMemory (producer is owner and responsible for cleanup)
        try:
            shm_temp = SharedMemory(name=self.shm_name)
            shm_temp.close()
            shm_temp.unlink()
            logger.info("Unlinked SharedMemory")
        except FileNotFoundError:
            pass  # Already unlinked
        except (OSError, RuntimeError) as e:
            logger.warning("Could not unlink SharedMemory: %s", e)

        # F9 — close activation barrier SHM handle (producer never decrements, just closes).
        _bshm = getattr(self, "_barrier_shm", None)
        if _bshm is not None:
            with contextlib.suppress(OSError, RuntimeError):
                _bshm.close()
            self._barrier_shm = None

        # Reset all state to prevent double-free on re-entry
        self.dev_ptrs = [None] * self.num_slots
        self.ipc_events = [None] * self.num_slots
        self.ipc_handles = [None] * self.num_slots
        self.ipc_event_handles = [None] * self.num_slots
        self._initialized = False

        logger.info("Cleanup complete")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> CUDAIPCExporter:
        """Enter context manager — returns self for use in 'with' statement."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager — cleanup resources regardless of exception."""
        self.cleanup()
        return None  # Don't suppress exceptions

    def __del__(self) -> None:
        """Destructor — cleanup on garbage collection if not already done."""
        if getattr(self, "_initialized", False):
            self.cleanup()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Check if exporter is ready to export frames.

        Returns:
            True if initialized with all GPU buffers allocated.
        """
        return self._initialized and all(ptr is not None for ptr in self.dev_ptrs)

    def attach_nvml_observer(self, observer: NVMLObserver) -> None:
        """Attach an NVMLObserver for GPU telemetry in get_stats().

        Args:
            observer: NVMLObserver instance (must already be started).
        """
        self._nvml_observer = observer

    def get_stats(self) -> dict:
        """Get exporter statistics for monitoring.

        Returns:
            Dictionary with current exporter state and performance metrics.
            Includes an 'nvml' sub-dict when an NVMLObserver is attached.
        """
        avg_memcpy = self.total_memcpy_us / self.frame_count if self.frame_count > 0 else 0.0
        avg_total = self.total_export_us / self.frame_count if self.frame_count > 0 else 0.0
        stats: dict = {
            "initialized": self._initialized,
            "shm_name": self.shm_name,
            "resolution": f"{self.width}x{self.height}x{self.channels}",
            "dtype": self.dtype,
            "num_slots": self.num_slots,
            "data_size_kb": self.data_size / 1024,
            "buffer_size_mb": self.buffer_size / (1024 * 1024),
            "frame_count": self.frame_count,
            "write_idx": self.write_idx,
            "avg_memcpy_us": avg_memcpy,
            "avg_total_us": avg_total,
            "dev_ptrs": [f"0x{ptr.value:016x}" if ptr else "NULL" for ptr in self.dev_ptrs],
        }
        observer = getattr(self, "_nvml_observer", None)
        if observer is not None:
            stats["nvml"] = observer.snapshot()
        return stats
