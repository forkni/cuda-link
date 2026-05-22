"""
TDSender - Sender engine for CUDAIPCExtension.

Owns all Sender-mode CUDA IPC resources: GPU ring-buffer allocation, IPC handle
export, SHM write-back, CUDA graph capture, and activation-barrier signalling.

textDAT name: TDSender  (must match the importable module name inside the COMP namespace)
"""

from __future__ import annotations

import contextlib
import os
import time
import traceback
from ctypes import c_void_p
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable

from ActivationBarrier import HolderBarrier  # noqa: E402, I001
from CUDAIPCWrapper import get_cuda_runtime  # noqa: E402
from CUDARuntimeTypes import CUDART_GRAPHS_MIN_VERSION  # noqa: E402
from FrameProfile import FrameProfile  # noqa: E402
from NVMLObserver import NVML_AVAILABLE, NVMLObserver  # noqa: E402
from NVTXShim import pop_range as _nvtx_pop  # noqa: E402
from NVTXShim import push_range as _nvtx_push
from NVTXShim import verbose_range as _nvtx_verbose
from SHMProtocol import (  # noqa: E402
    _ST_U32,
    FORMAT_KIND_FLOAT,
    FORMAT_KIND_UNSIGNED,
    MAGIC_OFFSET,
    METADATA_SIZE,
    NUM_SLOTS_OFFSET,
    PROTOCOL_MAGIC,
    SHM_HEADER_SIZE,
    SHUTDOWN_FLAG_SIZE,
    SLOT_SIZE,
    TIMESTAMP_SIZE,
    WRITE_IDX_OFFSET,
    Metadata,
    SHMLayout,
    bump_version,
    publish_frame,
)
from TDConfig import TDSenderConfig  # noqa: E402
from TDHost import TDHost  # noqa: E402

_CUDA_UNSUPPORTED_PIXEL_FORMATS = {
    # Rejected outright by cudaMemory(): "Texture is invalid pixel format to be shared with CUDA."
    "16-bit float",
    "16float",
    # Bug A: rejected outright with "Source TOP has unsupported pixel format." (no fallback)
    "10-bit",
    "10bit",
    # Bug B: cudaMemory() "succeeds" but returns dataType=uint8/numComps=4 — raw byte layout,
    # NOT the 11:11:10 packed float semantic. Silent receiver corruption without conversion.
    "11-bit",
    "11bit",
}
_EXPORT_BUFFER_NAME = "ExportBuffer"

# Pre-built NVTX range name strings per slot — eliminates f-string allocation on every export_frame call.
_NVTX_SENDER_SLOT_NAMES: tuple[str, ...] = tuple(f"cudalink.sender.export_frame.slot{i}" for i in range(10))

# FrameProfile region names for TDSender — defines column order for report() / avg().
_SENDER_PROFILE_REGIONS: tuple[str, ...] = (
    "pre_interop",
    "cuda_memory",
    "post_interop",
    "memcpy",
    "record_event",
    "sync",
    "sticky_check",
    "flush_probe",
    "shm_publish",
    "export",
    "unaccounted",
)


class TDSenderEngine:
    """Sender-mode engine: owns all GPU/SHM resources for the Sender path.

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

        self.dev_ptrs = [None] * self.num_slots
        self.buffer_size = 0
        self.data_size = 0
        self.width = 0
        self.height = 0
        self.channels = 4

        self.ipc_handles = [None] * self.num_slots
        self.ipc_events = [None] * self.num_slots
        self.ipc_event_handles = [None] * self.num_slots

        self._pending_free_ptrs: list = []
        self._pending_free_events: list = []
        self._deferred_free_at_frame = 0

        self.write_idx = 0
        self.shm_handle = None
        self._layout: SHMLayout | None = None
        self._shutdown_offset = 0
        self._ts_offset = 0
        self.frame_count = 0
        self.cuda_mem_ref = None
        self.sync_interval = 10

        self._export_sync: bool = self._config.export_sync
        self._export_profile: bool = self._config.export_profile
        self._export_flush_probe: bool = self._config.export_flush_probe
        self._use_graphs: bool = self._config.use_graphs
        self._log(
            f"[GRAPHS_INIT] _use_graphs={self._use_graphs} "
            f"(config.use_graphs={self._config.use_graphs}, "
            f"CUDALINK_TD_USE_GRAPHS env={os.environ.get('CUDALINK_TD_USE_GRAPHS', '(unset)')})",
            force=True,
        )
        self._graphs_disabled: bool = False
        self._graph_execs: list = [None] * self.num_slots
        self._graph_templates: list = [None] * self.num_slots
        self._graph_memcpy_nodes: list = [None] * self.num_slots
        self._stream_high_prio: bool = self._config.stream_high_prio
        self._init_pace: bool = self._config.init_pace
        self._graphs_pending: bool = False
        self._graphs_deferred: bool = self._config.graphs_deferred
        self._persist_stream: bool = self._config.persist_stream
        self._barrier = HolderBarrier(
            enabled=self._config.activation_barrier,
            settle_frames=self._config.barrier_settle_frames,
        )

        self._nvml_observer: NVMLObserver | None = None

        self._profile: FrameProfile = FrameProfile(_SENDER_PROFILE_REGIONS)

        self._warned_format = False
        self._export_buffer: object = None
        self._last_pixel_fmt: str = ""
        self._last_fmt_needs_conv: bool = False

        self.ipc_stream = None
        self._last_cuda_mem_err = ""
        self._detected_numpy_dtype: object = None
        self._last_numpy_dtype: object = None
        self._last_status_tuple: tuple | None = None

        # Profiling events (created lazily in initialize when _export_profile=True)
        self._timing_start = None
        self._timing_end = None

    def is_ready(self) -> bool:
        """True when initialized and all GPU buffer slots are allocated."""
        return self._initialized and all(ptr is not None for ptr in self.dev_ptrs)

    def get_stats(self) -> dict:
        """Sender statistics dict."""
        return {
            "mode": "Sender",
            "initialized": self._initialized,
            "frame_count": self.frame_count,
            "shm_name": self.shm_name,
            "num_slots": self.num_slots,
            "buffer_size_mb": self.buffer_size / 1024 / 1024 if self.buffer_size > 0 else 0,
            "resolution": f"{self.width}x{self.height}x{self.channels}" if self.width > 0 else "N/A",
            "write_idx": self.write_idx,
            "dev_ptrs": [f"0x{ptr.value:016x}" if ptr else "NULL" for ptr in self.dev_ptrs],
        }

    def _is_unsupported_format(self, top_op: object) -> bool:
        """Return True if the TOP's pixel format is unsupported by cudaMemory() in TD 2025.

        Empirical probe (verification/results/cuda_memory_probe_20260510_090919.json,
        TD 2025.32820): cudaMemory() rejects all 4 float16 variants and 10:10:10:2 fixed
        outright; 11:11:10 float "succeeds" but returns dataType=uint8/numComps=4 (raw
        byte layout, NOT semantic) — silent corruption. On True: sender skips the frame
        and emits a component warning (addScriptError); on False: warning is cleared.

        top_op may be a RealTOPHandle, FakeTOPHandle, or raw TD TOP (backward compat).
        """
        if hasattr(top_op, "pixel_format"):
            pixel_fmt = str(top_op.pixel_format)
        else:
            pixel_fmt = str(getattr(top_op, "pixelFormat", ""))
        if pixel_fmt == self._last_pixel_fmt:
            return self._last_fmt_needs_conv
        self._last_pixel_fmt = pixel_fmt
        pixel_lower = pixel_fmt.lower()
        self._last_fmt_needs_conv = any(u in pixel_lower for u in _CUDA_UNSUPPORTED_PIXEL_FORMATS)
        return self._last_fmt_needs_conv

    def initialize(self, width: int, height: int, channels: int = 4, buffer_size: int | None = None) -> bool:
        """Initialize CUDA IPC resources.

        Args:
            width: Texture width in pixels
            height: Texture height in pixels
            channels: Number of channels (default: 4 for RGBA)
            buffer_size: Actual buffer size in bytes (optional, auto-calculated if None)

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            self._log("Already initialized")
            return True

        # Lock Numslots while active — changing slot count at runtime causes array size mismatch
        self._host.set_param_enabled("Numslots", False)

        try:
            # Activation-barrier hold: signal the Python producer to pause pushes
            # during this Sender's WDDM-saturating init burst.
            self._barrier.acquire(os.getpid(), log_fn=self._log)

            # Load CUDA runtime bound to the configured device
            self.cuda = get_cuda_runtime(device=self.device)
            if not getattr(self, "_runtime_load_logged", False):
                self._log(f"Loaded CUDA runtime on device {self.cuda.get_device()}", force=True)
                self._runtime_load_logged = True

            # Create dedicated non-blocking stream for IPC operations.
            # Reuse existing stream on re-init to avoid leaks.
            if self.ipc_stream is None:
                # Default normal-priority (Phase 4.1). Set CUDALINK_TD_STREAM_PRIO=high
                # for explicit single-pair lowest-latency.
                if self._stream_high_prio:
                    self.ipc_stream = self.cuda.create_stream_with_priority(flags=0x01)
                    self._log(
                        f"Created IPC stream (high-priority): 0x{int(self.ipc_stream.value):016x}",
                        force=True,
                    )
                else:
                    self.ipc_stream = self.cuda.create_stream(flags=0x01)
                    self._log(
                        f"Created IPC stream (normal-priority): 0x{int(self.ipc_stream.value):016x}",
                        force=True,
                    )
            else:
                self._log(
                    f"Reusing IPC stream: 0x{int(self.ipc_stream.value):016x}",
                    force=True,
                )

            # Store dimensions
            self.width = width
            self.height = height
            self.channels = channels
            # Use provided buffer_size (from cuda_mem.size) or calculate
            raw_size = buffer_size if buffer_size is not None else width * height * channels * 4
            # Round up to 2MiB alignment (NVIDIA requirement: prevents unintended information disclosure)
            alignment = 2 * 1024 * 1024  # 2 MiB
            self.buffer_size = ((raw_size + alignment - 1) // alignment) * alignment
            self.data_size = raw_size  # Store actual data size for memcpy and comparisons

            # Defensive array resize: num_slots may have changed between cleanup and init
            # (e.g. handle_numslots_change() sets num_slots after cleanup resets arrays)
            if len(self.dev_ptrs) != self.num_slots:
                self.dev_ptrs = [None] * self.num_slots
                self.ipc_handles = [None] * self.num_slots
                self.ipc_events = [None] * self.num_slots
                self.ipc_event_handles = [None] * self.num_slots

            # Allocate ring buffer slots
            for slot in range(self.num_slots):
                # Allocate persistent GPU buffer for this slot
                self.dev_ptrs[slot] = self.cuda.malloc(self.buffer_size)
                self._log(
                    f"Allocated GPU buffer slot {slot}: "
                    f"{self.buffer_size / 1024 / 1024:.1f} MB at 0x{self.dev_ptrs[slot].value:016x}",
                    force=True,
                )

                # Create IPC handle for this buffer (ONCE - reuse for all frames)
                self.ipc_handles[slot] = self.cuda.ipc_get_mem_handle(self.dev_ptrs[slot])
                self._log(f"Created IPC handle for slot {slot} (64 bytes)")

                # Create IPC event for GPU-side synchronization (per-slot)
                self.ipc_events[slot] = self.cuda.create_ipc_event()
                self.ipc_event_handles[slot] = self.cuda.ipc_get_event_handle(self.ipc_events[slot])
                self._log(f"Created IPC event for slot {slot} (64 bytes)")

            self._log(f"Created {self.num_slots} IPC buffer slots with events", force=True)

            # INIT_PACE checkpoint 1/3 — CUDALINK_TD_INIT_PACE=1: flush WDDM queue after per-slot
            # alloc burst (cudaMalloc + IpcGetMemHandle + EventCreate + IpcGetEventHandle × N).
            if self._init_pace:
                self.cuda.stream_synchronize(self.ipc_stream)
                time.sleep(0.02)
                self._log("[INIT_PACE] checkpoint 1/3 (post-slot-alloc)", force=True)

            # Create SharedMemory for IPC handle transfer
            # Size: header + slots + shutdown flag + metadata + timestamp (for extended protocol)
            shm_size = (
                SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE) + SHUTDOWN_FLAG_SIZE + METADATA_SIZE + TIMESTAMP_SIZE
            )

            try:
                # Try to open existing SharedMemory first
                self.shm_handle = SharedMemory(name=self.shm_name)
                self._log(f"Opened existing SharedMemory: {self.shm_name}", force=True)
            except FileNotFoundError:
                # Create new SharedMemory if doesn't exist
                self.shm_handle = SharedMemory(name=self.shm_name, create=True, size=shm_size)
                self._log(
                    f"Created new SharedMemory: {self.shm_name} ({shm_size} bytes)",
                    force=True,
                )

            # Write IPC handle to SharedMemory (ONCE - Python process reads at startup)
            self._write_handle_to_shm()

            # Write texture metadata to extended protocol region
            self._write_metadata_to_shm()

            # INIT_PACE checkpoint 2/3 — after SHM segment creation + handle/metadata writes.
            if self._init_pace:
                self.cuda.stream_synchronize(self.ipc_stream)
                time.sleep(0.02)
                self._log("[INIT_PACE] checkpoint 2/3 (post-SHM-write)", force=True)

            # Cache SHM offsets: avoid recomputing these on every export_frame() call
            self._layout = SHMLayout(self.num_slots)
            self._shutdown_offset = self._layout.shutdown_offset
            self._ts_offset = self._layout.timestamp_offset

            # Cache ExportBuffer as TOPHandle — eliminates per-frame ownerComp.op() lookup
            self._export_buffer = self._host.find_top(_EXPORT_BUFFER_NAME)

            # Create GPU timing events (only when Debug is ON for benchmarking)
            if self.verbose_performance:
                self._timing_start = self.cuda.create_timing_event()
                self._timing_end = self.cuda.create_timing_event()
                self._log("Created GPU timing events for benchmarking", force=False)
            else:
                self._timing_start = None
                self._timing_end = None

            self._initialized = True
            self._barrier.arm_settle_countdown()
            self._log("Initialization complete - ready for zero-copy GPU transfer", force=True)

            # CUDA Graphs build (after IPC stream / events / ring buffer are ready).
            # Gated on CUDALINK_TD_USE_GRAPHS=1 AND cudart >= 11.4
            # (cudaGraphInstantiateWithFlags + EventRecordNodeSetEvent require 11.4+).
            if self._use_graphs:
                try:
                    rt_version = self.cuda.get_runtime_version()
                except (RuntimeError, OSError) as exc:
                    rt_version = 0
                    self._log(f"cudaRuntimeGetVersion failed ({exc}) — disabling graphs", force=True)
                if rt_version >= CUDART_GRAPHS_MIN_VERSION:
                    if self._graphs_deferred:
                        # CUDALINK_TD_GRAPHS_DEFERRED=1: defer graph capture to first
                        # warm frame (frame_count >= 30) so the init burst doesn't overlap
                        # with Receiver-A's 60 Hz stream. First 30 frames use legacy memcpy_async.
                        self._graphs_pending = True
                        self._log(
                            "CUDA export graph build deferred to first warm frame (CUDALINK_TD_GRAPHS_DEFERRED=1)",
                            force=True,
                        )
                    else:
                        self._build_export_graphs()
                        # INIT_PACE checkpoint 3/3 — after graph capture + instantiation.
                        if self._init_pace:
                            self.cuda.stream_synchronize(self.ipc_stream)
                            time.sleep(0.02)
                            self._log("[INIT_PACE] checkpoint 3/3 (post-graph-build)", force=True)
                else:
                    self._log(
                        f"CUDALINK_TD_USE_GRAPHS=1 ignored: cudart {rt_version} < {CUDART_GRAPHS_MIN_VERSION} "
                        "(cudaGraphInstantiateWithFlags requires 11.4+).",
                        force=True,
                    )
                    self._graphs_disabled = True

            if NVML_AVAILABLE and self._config.nvml:
                obs = NVMLObserver(device=self.device, enabled=True)
                if obs.start():
                    self._nvml_observer = obs
                    self._log(f"NVMLObserver attached on device {self.device}", force=True)

            return True

        except (OSError, RuntimeError, ValueError) as e:
            self._log(f"Initialization failed: {e}", force=True)
            self._host.set_error_status(f"Initialization failed: {e}")
            traceback.print_exc()
            return False

    def _build_export_graphs(self) -> None:
        """Capture the D2D memcpy into a 1-node CUDA Graph exec per ring slot.

        Mirrors CUDAIPCExporter._build_export_graphs() on the Python side.
        Captures only the memcpy_async (IPC events / external waits cannot be
        captured in global mode).  On failure the stream is restored to normal
        mode and self._graphs_disabled is set so the legacy stream path is used.
        """
        if self.cuda is None or self.ipc_stream is None:
            return

        placeholder_src = self.dev_ptrs[0]

        for slot in range(self.num_slots):
            capture_started = False
            try:
                # Relaxed: required to coexist with TensorRT's per-engine
                # cudaStreamBeginCapture (PyTorch, CuPy, TRT may all be in the same process).
                self.cuda.stream_begin_capture(self.ipc_stream, mode=2)
                capture_started = True
                self.cuda.memcpy_async(
                    dst=self.dev_ptrs[slot],
                    src=placeholder_src,
                    count=self.data_size,
                    kind=3,  # cudaMemcpyDeviceToDevice
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
                # Keep template alive so the captured node handle stays valid for
                # the per-frame cudaGraphExecMemcpyNodeSetParams1D updates.
                self._graph_execs[slot] = graph_exec
                self._graph_templates[slot] = template_graph
                self._graph_memcpy_nodes[slot] = memcpy_node
                self._log(f"Built export graph for slot {slot} (1-node: Memcpy)")

            except (RuntimeError, OSError) as exc:
                if capture_started:
                    try:
                        abandoned_graph = self.cuda.stream_end_capture(self.ipc_stream)
                        self.cuda.graph_destroy(abandoned_graph)
                    except (RuntimeError, OSError):
                        pass
                self._log(
                    f"CUDA Graph build failed for slot {slot} ({exc}) — disabling graphs "
                    "for this session and falling back to legacy stream path. "
                    "Set CUDALINK_TD_USE_GRAPHS=0 to suppress.",
                    force=True,
                )
                self._graphs_disabled = True
                self._destroy_export_graphs()
                return

        self._log(
            f"CUDA export graphs built for {self.num_slots} slots (CUDALINK_TD_USE_GRAPHS=1)",
            force=True,
        )

    def _destroy_export_graphs(self) -> None:
        """Destroy all CUDA Graph exec objects and their templates."""
        if self.cuda is None:
            return
        for slot, graph_exec in enumerate(getattr(self, "_graph_execs", [])):
            if graph_exec is not None:
                try:
                    self.cuda.graph_exec_destroy(graph_exec)
                except (RuntimeError, OSError) as e:
                    self._log(f"Error destroying graph exec slot {slot}: {e}", force=True)
                self._graph_execs[slot] = None
        for slot, template in enumerate(getattr(self, "_graph_templates", [])):
            if template is not None:
                with contextlib.suppress(RuntimeError, OSError):
                    self.cuda.graph_destroy(template)
                self._graph_templates[slot] = None
        if hasattr(self, "_graph_memcpy_nodes"):
            self._graph_memcpy_nodes = [None] * self.num_slots

    def _write_handle_to_shm(self) -> None:
        """Write magic + version + num_slots + write_idx + all IPC handles to SharedMemory.

        Layout (20 + NUM_SLOTS*192 + 1 bytes):
        [0-3]     magic (4B) - protocol validation "CIPD"
        [4-11]    version (8B)
        [12-15]   num_slots (4B)
        [16-19]   write_idx (4B)

        For each slot (128 bytes per slot):
        [20+slot*128 : 84+slot*128]   mem_handle (64B)
        [84+slot*128 : 148+slot*128]  event_handle (64B)

        [20+NUM_SLOTS*128]  shutdown flag (1B)
        """
        if self.shm_handle is None or not all(self.ipc_handles):
            return

        self._layout = SHMLayout(self.num_slots)

        # Write protocol header: magic, bump version, num_slots, reset write_idx
        _ST_U32.pack_into(self.shm_handle.buf, MAGIC_OFFSET, PROTOCOL_MAGIC)
        new_version = bump_version(self.shm_handle.buf)
        _ST_U32.pack_into(self.shm_handle.buf, NUM_SLOTS_OFFSET, self.num_slots)
        _ST_U32.pack_into(self.shm_handle.buf, WRITE_IDX_OFFSET, 0)  # write_idx=0 initially

        # Write handles for each slot
        for slot in range(self.num_slots):
            base_offset = self._layout.slot_offset(slot)

            # Write memory handle (64 bytes)
            mem_handle_bytes = bytes(self.ipc_handles[slot].internal)
            self.shm_handle.buf[base_offset : base_offset + 64] = mem_handle_bytes
            print(f"[SENDER-HEX] slot{slot} mem handle: {mem_handle_bytes[:16].hex()}...", flush=True)

            # Write event handle (64 bytes) if available
            if self.ipc_event_handles[slot]:
                event_handle_bytes = bytes(self.ipc_event_handles[slot].reserved)
                self.shm_handle.buf[base_offset + 64 : base_offset + 128] = event_handle_bytes
                self._log(f"Wrote slot {slot} handles: mem={len(mem_handle_bytes)}B, event={len(event_handle_bytes)}B")
            else:
                self._log(f"Wrote slot {slot} mem handle: {len(mem_handle_bytes)}B")

        # Clear shutdown flag — matches CUDAIPCExporter._write_handles_to_shm() on the Python side.
        # Without this, a stale shutdown_flag=1 from a previous session (or a race where another
        # sender initialised after this one) would block the receiver indefinitely.
        self.shm_handle.buf[self._layout.shutdown_offset] = 0

        self._log(
            f"Wrote all IPC handles v{new_version} to SharedMemory ({SHM_HEADER_SIZE + self.num_slots * SLOT_SIZE + SHUTDOWN_FLAG_SIZE + METADATA_SIZE + TIMESTAMP_SIZE} bytes total)",
            force=True,
        )

    def _write_metadata_to_shm(self) -> None:
        """Write texture metadata to the extended protocol region after shutdown flag.

        Extended protocol layout (20 bytes):
        [+0  : 4B]  width         (uint32 LE)
        [+4  : 4B]  height        (uint32 LE)
        [+8  : 4B]  num_comps     (uint32 LE)
        [+12 : 1B]  format_kind   (uint8)  — cudaChannelFormatKind: 0=Signed,1=Unsigned,2=Float
        [+13 : 1B]  bits_per_comp (uint8)  — 8/16/32/64
        [+14 : 2B]  flags         (uint16 LE) — bit0=bfloat16; rest reserved=0
        [+16 : 4B]  data_size     (uint32 LE) — actual bytes (before 2MiB alignment)
        """
        if self.shm_handle is None or self.data_size == 0:
            return

        # Encode format as (kind, bits, flags) — self-describing, receiver-compatible.
        # Primary source: _detected_numpy_dtype from cuda_mem.data_type (authoritative).
        # The GPU allocation size (self.data_size) may be padded or reflect the previous
        # format when dtype changes with a constant allocation, so it must not drive
        # kind/bits alone. Ratio-based fallback is used only when dtype is unavailable.
        pixel_count = self.width * self.height * self.channels if (self.width and self.height and self.channels) else 0

        flags = 0
        # Fallback: derive bits/kind from GPU allocation ratio.
        _ratio_bits = self.data_size // pixel_count * 8 if pixel_count > 0 and self.data_size % pixel_count == 0 else 32
        bits = _ratio_bits
        kind = FORMAT_KIND_UNSIGNED if bits == 8 else FORMAT_KIND_FLOAT

        # Override with authoritative dtype hint when cuda_mem.data_type was reported.
        _hint = self._detected_numpy_dtype
        if _hint is not None:
            try:
                import numpy as _np

                _hint = _np.dtype(_hint)
                if _hint == _np.dtype("uint8"):
                    bits, kind = 8, FORMAT_KIND_UNSIGNED
                elif _hint == _np.dtype("uint16"):
                    bits, kind = 16, FORMAT_KIND_UNSIGNED
                elif _hint == _np.dtype("float16"):
                    bits, kind = 16, FORMAT_KIND_FLOAT
                elif _hint == _np.dtype("float64"):
                    bits, kind = 64, FORMAT_KIND_FLOAT
                else:  # float32 and any future dtype
                    bits, kind = 32, FORMAT_KIND_FLOAT
            except Exception:  # noqa: BLE001
                pass  # keep ratio-derived fallback

        # Use active-region size (W*H*C*(bits/8)) as the metadata data_size so
        # the receiver invariant W*H*C*(bits/8)==data_size is always satisfied.
        # self.data_size is the GPU allocation (may be padded/stale when dims change
        # with a constant allocation), so it must not flow directly into the metadata
        # field that the receiver validates. Python-side exporter does the same.
        meta_data_size = pixel_count * (bits // 8)
        Metadata(
            width=self.width,
            height=self.height,
            num_comps=self.channels,
            format_kind=kind,
            bits_per_comp=bits,
            flags=flags,
            data_size=meta_data_size,
        ).pack_into(self.shm_handle.buf, self._layout)

        # Track last written dtype for change detection
        self._last_numpy_dtype = self._detected_numpy_dtype

        self._log(
            f"Wrote metadata: {self.width}x{self.height}x{self.channels}, "
            f"kind={kind} bits={bits} flags=0x{flags:04x}, size={meta_data_size}B"
        )

    def _has_dtype_changed(self) -> bool:
        """Check if detected numpy dtype differs from last written metadata.

        Both attributes are pre-initialized to None in __init__ and set as numpy.dtype
        objects (from cuda_mem.shape.dataType / _write_metadata_to_shm), so direct
        comparison is safe — no per-frame np.dtype() construction needed.
        """
        if self._detected_numpy_dtype is None or self._last_numpy_dtype is None:
            return False  # Not yet detected or not yet written
        return self._detected_numpy_dtype != self._last_numpy_dtype

    def _bump_version(self) -> None:
        """Increment SharedMemory version counter to signal consumers to re-read metadata."""
        if self.shm_handle is None:
            return
        new_version = bump_version(self.shm_handle.buf)
        self._log(f"Version bumped to {new_version} (metadata-only change)")

    def export_frame(self, top_op: TOP | None = None) -> bool:
        """Export the ExportBuffer TOP texture via CUDA IPC.

        Resolves ExportBuffer internally from ownerComp so the correct frame
        is always exported regardless of what the caller previously passed.

        Args:
            top_op: Deprecated. Accepted for backwards compatibility but ignored.
                ExportBuffer is always resolved from ownerComp internally.

        Returns:
            True if export successful, False otherwise
        """
        top_op = self._export_buffer
        if top_op is None or not top_op.is_valid():
            self._export_buffer = None  # invalidate stale cache
            # Lazy lookup: op may have been added after initialize() (e.g. dynamic network edits)
            top_op = self._host.find_top(_EXPORT_BUFFER_NAME)
            if top_op is None:
                self._log(f"'{_EXPORT_BUFFER_NAME}' not found in component", force=True)
                return False
            self._export_buffer = top_op  # cache for subsequent frames

        # Check if Active parameter is enabled (hot path via TDHost.is_active())
        if not self._host.is_active():
            return False

        # Start frame timer (only if verbose)
        if self.verbose_performance:
            frame_start = time.perf_counter()
            if self._export_profile:
                _t_pre = frame_start
                # initialize per-frame profile locals so unaccounted calc is always defined
                _this_pre = _this_post = _this_sync = _this_sticky = _this_fp = _this_shm = 0.0
                # record_event_time is only set in the ipc_events path; init here for the fallback
                record_event_time = 0.0

        _nvtx_push(_NVTX_SENDER_SLOT_NAMES[self.write_idx % self.num_slots], "green")
        try:
            # Ensure CUDA runtime and stream exist BEFORE first cudaMemory() call.
            # cuda and ipc_stream are always set together; one outer check covers both.
            if self.ipc_stream is None:
                if self.cuda is None:
                    self.cuda = get_cuda_runtime(device=self.device)
                # Honour CUDALINK_TD_STREAM_PRIO in the pre-init lazy path too (mirror of init).
                if self._stream_high_prio:
                    self.ipc_stream = self.cuda.create_stream_with_priority(flags=0x01)
                    self._log(
                        f"Created IPC stream (pre-init, high-priority): 0x{int(self.ipc_stream.value):016x}",
                        force=True,
                    )
                else:
                    self.ipc_stream = self.cuda.create_stream(flags=0x01)
                    self._log(
                        f"Created IPC stream (pre-init, normal-priority): 0x{int(self.ipc_stream.value):016x}",
                        force=True,
                    )
            _cuda = self.cuda

            # Block transfer when the source pixel format is unsupported by cudaMemory().
            # Probe (verification/results/cuda_memory_probe_20260510_090919.json) confirmed
            # 6 formats fail: all 4 float16 variants (hard exception), 10:10:10:2 (hard
            # exception), 11:11:10 (succeeds but returns raw uint8/4ch — silent corruption).
            # Tint the COMP yellow every bad frame (idempotent; keeps tint alive); log once.
            if self._is_unsupported_format(top_op):
                src_fmt = (
                    top_op.pixel_format if hasattr(top_op, "pixel_format") else getattr(top_op, "pixelFormat", "?")
                )
                warn_msg = f"unsupported pixel format {src_fmt!r}"
                self._host.set_warning_status(warn_msg)
                if not self._warned_format:
                    self._log(
                        f"Pixel format {src_fmt!r} unsupported by cudaMemory() — "
                        "transfer suspended; component tinted yellow",
                        force=True,
                    )
                    self._warned_format = True
                return False
            if self._warned_format:
                self._host.clear_status()
                self._log("Source pixel format now supported — transfer resumed", force=True)
                self._warned_format = False

            # Time cudaMemory() call (OpenGL→CUDA interop)
            if self.verbose_performance:
                if self._export_profile:
                    _this_pre = (time.perf_counter() - _t_pre) * 1_000_000
                    self._profile.record("pre_interop", _this_pre)
                cuda_mem_start = time.perf_counter()

            # Get TOP's CUDA memory — always pass a valid stream (never None)
            try:
                cuda_mem = top_op.cuda_memory(stream=int(self.ipc_stream.value))
            except Exception as cuda_err:
                pixel_fmt = (
                    top_op.pixel_format
                    if hasattr(top_op, "pixel_format")
                    else getattr(top_op, "pixelFormat", "unknown")
                )
                err_msg = f"cudaMemory() failed (pixelFormat={pixel_fmt}): {cuda_err}"
                if err_msg != self._last_cuda_mem_err:
                    self._log(err_msg, force=True)
                    self._last_cuda_mem_err = err_msg
                return False

            if self.verbose_performance:
                cuda_mem_time = (time.perf_counter() - cuda_mem_start) * 1_000_000  # microseconds
                self._profile.record("cuda_memory", cuda_mem_time)
                if self._export_profile:
                    _t_post = time.perf_counter()

            # Reset error suppression on success
            if self._last_cuda_mem_err:
                self._log("cudaMemory() recovered.", force=True)
                self._last_cuda_mem_err = ""

            if cuda_mem is None:
                self._log(f"Failed to get CUDA memory from {top_op}", force=True)
                return False

            # CRITICAL: Keep reference to prevent garbage collection
            self.cuda_mem_ref = cuda_mem

            # CUDAMemoryRef fields are plain Python ints — direct access, no shape indirection
            actual_width = cuda_mem.width
            actual_height = cuda_mem.height
            actual_channels = cuda_mem.channels
            actual_size = cuda_mem.size
            self._detected_numpy_dtype = cuda_mem.data_type  # numpy.dtype or None
            _dt = self._detected_numpy_dtype
            _dtype_str = (
                getattr(_dt, "name", None) or getattr(_dt, "__name__", str(_dt)) if _dt is not None else "unknown"
            )
            _status_key = (actual_width, actual_height, actual_channels, _dtype_str)
            if _status_key != self._last_status_tuple:
                self._host.set_info_status(f"{actual_width}x{actual_height} {_dtype_str} {actual_channels}ch")
                self._last_status_tuple = _status_key

            # Check if we need to (re)initialize
            if not self._initialized or actual_size != self.data_size:
                if self._initialized:
                    self._log(
                        f"Resolution changed: {self.width}x{self.height}x{self.channels} -> {actual_width}x{actual_height}x{actual_channels}",
                        force=True,
                    )
                    # Queue old resources for deferred free (cudaFree blocks on IPC memory)
                    self._pending_free_ptrs.extend([p for p in self.dev_ptrs if p])
                    self._pending_free_events.extend([e for e in self.ipc_events if e])
                    self.dev_ptrs = [None] * self.num_slots
                    self.ipc_events = [None] * self.num_slots
                    self.ipc_handles = [None] * self.num_slots
                    self.ipc_event_handles = [None] * self.num_slots
                    self._initialized = False
                    # Schedule deferred free after 30 frames (receiver needs time to close handles)
                    self._deferred_free_at_frame = self.frame_count + 30

                if not self.initialize(actual_width, actual_height, actual_channels, actual_size):
                    return False
                # Metadata already written by initialize()

            elif (
                actual_width != self.width
                or actual_height != self.height
                or actual_channels != self.channels
                or self._has_dtype_changed()
            ):
                # Metadata-only update: buffer size unchanged so GPU handles stay valid.
                # Rewrite the 20-byte metadata region and bump version to signal consumers.
                self.width = actual_width
                self.height = actual_height
                self.channels = actual_channels
                self._write_metadata_to_shm()
                self._bump_version()
                self._log(
                    "Metadata changed (dtype/dimensions) without size change — updated in-place",
                    force=True,
                )

            # Calculate current slot for ring buffer rotation
            slot = self.write_idx % self.num_slots

            # Time cudaMemcpyAsync D2D (non-blocking) - only if verbose
            if self.verbose_performance:
                if self._export_profile:
                    _this_post = (time.perf_counter() - _t_post) * 1_000_000
                    self._profile.record("post_interop", _this_post)
                memcpy_start = time.perf_counter()
                # Record GPU timing start event (actual GPU time measurement)
                if self._timing_start:
                    _cuda.record_event(self._timing_start, stream=self.ipc_stream)

            # Deferred graph build (CUDALINK_TD_GRAPHS_DEFERRED=1): fires once after 30
            # steady-state frames so the capture burst doesn't overlap Sender-B's cold activation.
            if self._graphs_pending and self.frame_count >= 30:
                self._build_export_graphs()
                self._graphs_pending = False

            # Copy TOP texture to this slot's persistent buffer (async on IPC stream).
            # When CUDALINK_TD_USE_GRAPHS=1 and the per-slot graph exec is built, replay
            # a 1-node CUDA Graph (MemcpyNode) instead of the imperative memcpy_async —
            # this collapses the kernel-mode submission into a single cudaGraphLaunch.
            # Falls back automatically (and permanently for this instance) if launch fails.
            if self._use_graphs and not self._graphs_disabled and self._graph_execs[slot] is not None:
                try:
                    _cuda.graph_exec_memcpy_node_set_params_1d(
                        self._graph_execs[slot],
                        self._graph_memcpy_nodes[slot],
                        dst=self.dev_ptrs[slot],
                        src=c_void_p(cuda_mem.ptr),
                        count=self.data_size,
                        kind=3,  # cudaMemcpyDeviceToDevice
                    )
                    _cuda.graph_launch(self._graph_execs[slot], self.ipc_stream)
                except (RuntimeError, OSError) as _graph_err:
                    self._log(
                        f"Graph launch failed ({_graph_err}) — disabling graphs, "
                        "falling back to legacy memcpy_async this frame",
                        force=True,
                    )
                    self._graphs_disabled = True
                    _cuda.memcpy_async(
                        dst=self.dev_ptrs[slot],
                        src=c_void_p(cuda_mem.ptr),
                        count=self.data_size,
                        kind=3,
                        stream=self.ipc_stream,
                    )
            else:
                with _nvtx_verbose("cudalink.sender.memcpy", "green"):
                    _cuda.memcpy_async(
                        dst=self.dev_ptrs[slot],
                        src=c_void_p(cuda_mem.ptr),
                        count=self.data_size,
                        kind=3,  # cudaMemcpyDeviceToDevice
                        stream=self.ipc_stream,
                    )

            if self.verbose_performance:
                # Record GPU timing end event (actual GPU time measurement)
                if self._timing_end:
                    _cuda.record_event(self._timing_end, stream=self.ipc_stream)
                memcpy_time = (
                    time.perf_counter() - memcpy_start
                ) * 1_000_000  # microseconds (enqueue time only, copy is async)
                self._profile.record("memcpy", memcpy_time)

            # GPU-side synchronization with CUDA IPC Events
            if self.ipc_events[slot]:
                if self.verbose_performance:
                    record_start = time.perf_counter()

                # Record event for this slot after async memcpy (stream-ordered)
                with _nvtx_verbose("cudalink.sender.record_event", "green"):
                    _cuda.record_event(self.ipc_events[slot], stream=self.ipc_stream)

                if self.verbose_performance:
                    record_event_time = (time.perf_counter() - record_start) * 1_000_000
                    self._profile.record("record_event", record_event_time)

                # CUDALINK_EXPORT_SYNC=1: CPU-blocks on ipc_stream after record_event.
                # Default is now "0" (receiver cudaStreamWaitEvent guarantees correctness).
                # Enable for regression testing or if downstream consumers rely on CPU-timing.
                if self._export_sync:
                    if self.verbose_performance and self._export_profile:
                        _t_sync = time.perf_counter()
                    _cuda.stream_synchronize(self.ipc_stream)
                    if self.verbose_performance and self._export_profile:
                        _this_sync = (time.perf_counter() - _t_sync) * 1_000_000
                        self._profile.record("sync", _this_sync)

                if self.verbose_performance and self._export_profile:
                    _t_sticky = time.perf_counter()
                _cuda.check_sticky_error("export_frame")
                if self.verbose_performance and self._export_profile:
                    _this_sticky = (time.perf_counter() - _t_sticky) * 1_000_000
                    self._profile.record("sticky_check", _this_sticky)

                # WDDM deferred-submission probe: forces pending GPU work to submit without
                # blocking. Per CUDA Handbook p3/pg56, WDDM buffers commands until a flush;
                # cudaStreamQuery triggers that flush. Only active when EXPORT_FLUSH_PROBE=1
                # and EXPORT_SYNC=0 (if sync is on, the stream is already flushed above).
                if self._export_flush_probe and not self._export_sync:
                    if self.verbose_performance and self._export_profile:
                        _t_fp = time.perf_counter()
                    _cuda.stream_query(self.ipc_stream)
                    if self.verbose_performance and self._export_profile:
                        _this_fp = (time.perf_counter() - _t_fp) * 1_000_000
                        self._profile.record("flush_probe", _this_fp)

            else:
                # FALLBACK: Conditional CPU synchronization
                if self.frame_count % self.sync_interval == 0:
                    _cuda.synchronize()

            # Publish: timestamp + clear shutdown_flag + fence + write_idx — in that order.
            # publish_frame() encodes the C3 ordering guarantee; do not replicate inline.
            if self.verbose_performance and self._export_profile:
                _t_shm = time.perf_counter()
            self.write_idx += 1
            publish_frame(self.shm_handle.buf, self._layout, self.write_idx, time.perf_counter())
            if self.verbose_performance and self._export_profile:
                _this_shm = (time.perf_counter() - _t_shm) * 1_000_000
                self._profile.record("shm_publish", _this_shm)

            # Frame tracking
            self.frame_count += 1

            # Barrier settle countdown: release the cross-process activation barrier
            # after settle_frames successful exports have elapsed post-init.
            self._barrier.tick_and_maybe_release(os.getpid(), log_fn=self._log)

            # Calculate total frame time (only if verbose)
            if self.verbose_performance:
                frame_time = (time.perf_counter() - frame_start) * 1_000_000
                self._profile.record("export", frame_time)
                if self._export_profile:
                    _this_accounted = (
                        _this_pre
                        + cuda_mem_time
                        + _this_post
                        + memcpy_time
                        + record_event_time
                        + _this_sync
                        + _this_sticky
                        + _this_fp
                        + _this_shm
                    )
                    self._profile.record("unaccounted", frame_time - _this_accounted)

            # Detailed first-frame diagnostic (one-time, not affected by 100-frame interval)
            if self.verbose_performance and self.frame_count == 1:
                self._log(
                    f"FIRST FRAME: cudaMemory={cuda_mem_time:.1f}us, "
                    f"memcpy={memcpy_time:.1f}us, total={frame_time:.1f}us, "
                    f"res={actual_width}x{actual_height}, size={actual_size / (1024 * 1024):.1f}MB",
                    force=True,
                )

            # Log performance metrics every 97 frames (prime — avoids aliasing with slot counts 2,4,5)
            if self.verbose_performance and self.frame_count % 97 == 0:
                n = self.frame_count
                avg_memcpy = self._profile.avg("memcpy", n)
                avg_record = self._profile.avg("record_event", n) if all(self.ipc_events) else 0
                avg_total = self._profile.avg("export", n)
                avg_cuda_mem = self._profile.avg("cuda_memory", n)
                sync_mode = (
                    f"GPU-Events[{self.num_slots}]" if all(self.ipc_events) else f"CPU-Sync(1/{self.sync_interval})"
                )

                graphs_label = "ON" if self._use_graphs and not self._graphs_disabled else "OFF"
                log_msg = (
                    f"Frame {self.frame_count}: slot {slot}, "
                    f"avg cudaMemory={avg_cuda_mem:.1f}us, "
                    f"avg memcpy={avg_memcpy:.1f}us, record={avg_record:.1f}us, "
                    f"total={avg_total:.1f}us, mode={sync_mode}, graphs={graphs_label}"
                )

                # Add GPU elapsed time if timing events available
                if self._timing_start and self._timing_end:
                    try:
                        # Wait for timing events to complete before reading (prevents error 600)
                        self.cuda.wait_event(self._timing_end)
                        gpu_memcpy_ms = self.cuda.event_elapsed_time(self._timing_start, self._timing_end)
                        log_msg += f", GPU memcpy={gpu_memcpy_ms * 1000:.1f}us (actual GPU time)"
                    except RuntimeError as e:
                        # Rare: event wait/query failed
                        log_msg += f", GPU timing: {e}"

                if self._nvml_observer is not None:
                    snap = self._nvml_observer.snapshot()
                    if snap.get("nvml_available"):
                        log_msg += (
                            f" | [NVML] gpu={snap.get('gpu_util_pct', '?')}%"
                            f" mem={snap.get('mem_bw_util_pct', '?')}%"
                            f" sm={snap.get('sm_clock_mhz', '?')}MHz"
                            f" pcie_tx={snap.get('pcie_tx_kbps', '?')}kbps"
                            f" pcie_rx={snap.get('pcie_rx_kbps', '?')}kbps"
                            f" temp={snap.get('temp_c', '?')}C"
                            f" power={snap.get('power_w', '?')}W"
                        )
                        reasons = snap.get("throttle_reasons") or []
                        if reasons:
                            log_msg += f" throttle={','.join(reasons)}"

                if self._export_profile:
                    avg_pre = self._profile.avg("pre_interop", n)
                    avg_post = self._profile.avg("post_interop", n)
                    avg_sync = self._profile.avg("sync", n)
                    avg_sticky = self._profile.avg("sticky_check", n)
                    avg_fp = self._profile.avg("flush_probe", n)
                    avg_shm = self._profile.avg("shm_publish", n)
                    avg_unacc = self._profile.avg("unaccounted", n)
                    log_msg += (
                        f" | [PROFILE] pre={avg_pre:.1f}us"
                        f" interop={avg_cuda_mem:.1f}us"
                        f" post={avg_post:.1f}us"
                        f" memcpy={avg_memcpy:.1f}us"
                        f" record={avg_record:.1f}us"
                        f" sync={avg_sync:.1f}us"
                        f" sticky={avg_sticky:.1f}us"
                        f" flush_probe={avg_fp:.1f}us"
                        f" shm={avg_shm:.1f}us"
                        f" unacc={avg_unacc:.1f}us"
                    )

                self._log(log_msg, force=False)

            return True

        except (OSError, RuntimeError, AttributeError) as e:
            self._log(f"Export failed: {e}", force=True)

            traceback.print_exc()
            return False
        finally:
            _nvtx_pop()

    def _check_deferred_cleanup(self) -> None:
        """Execute deferred GPU cleanup if scheduled and enough frames have passed.

        Lightweight check meant to be called from onFrameStart for minimal overhead.
        """
        if self._pending_free_ptrs and self.frame_count >= self._deferred_free_at_frame:
            self._deferred_free()

    def _deferred_free(self) -> None:
        """Free GPU resources queued from export_frame() when deferred frame threshold is reached.

        Called via _check_deferred_cleanup() after receiver has had time to close IPC handles.
        """
        if self.cuda is None:
            return

        freed_count = 0
        for ptr in self._pending_free_ptrs:
            try:
                self.cuda.free(ptr)
                freed_count += 1
            except (RuntimeError, OSError) as e:
                self._log(f"Deferred free failed: {e}")
        self._pending_free_ptrs.clear()

        for event in self._pending_free_events:
            try:
                self.cuda.destroy_event(event)
            except (RuntimeError, OSError) as e:
                self._log(f"Deferred event destroy failed: {e}")
        self._pending_free_events.clear()

        if freed_count > 0:
            self._log(
                f"Deferred cleanup complete: freed {freed_count} GPU buffers",
                force=True,
            )

    def _is_cuda_context_valid(self) -> bool:
        """Check if CUDA context is still valid (TD may destroy it before __delTD__)."""
        if self.cuda is None:
            return False
        try:
            self.cuda.cudart.cudaGetLastError()
            return True
        except (OSError, RuntimeError):
            return False

    def cleanup(self) -> None:
        """Cleanup Sender CUDA IPC resources (all ring buffer slots).

        CRITICAL ORDER: Signal shutdown FIRST, then free GPU resources.
        cudaFree() blocks until all processes close IPC handles.
        """
        # Release activation barrier if still held (mid-settle cleanup path).
        self._barrier.force_release(os.getpid(), log_fn=self._log)
        self._barrier.close()

        # Skip if already cleaned up (prevents double-cleanup from Active toggle + __delTD__)
        if not self._initialized and self.shm_handle is None:
            return

        if self._nvml_observer is not None:
            self._nvml_observer.stop()
            self._nvml_observer = None

        cuda_valid = self._is_cuda_context_valid()
        if not cuda_valid:
            self._log("CUDA context already destroyed — skipping GPU cleanup", force=True)

        # Signal shutdown to consumer (before closing SharedMemory)
        if self.shm_handle and self.shm_handle.buf is not None:
            try:
                shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
                self.shm_handle.buf[shutdown_offset] = 1
                self._log("Shutdown signal sent to consumer", force=True)
            except (OSError, BufferError) as e:
                self._log(f"Warning: Could not write shutdown signal: {e}", force=True)

        # Zero out IPC handle bytes so any reader sees invalid handles.
        # On Windows, unlink() is a no-op (SharedMemory uses CreateFileMapping kernel
        # objects), so the SharedMemory may persist with stale non-zero handles that
        # pass the all-zero validation check. Zeroing them prevents error 201 when a
        # new Receiver reads before the SHM is destroyed or overwritten by a new producer.
        if self.shm_handle and self.shm_handle.buf is not None:
            try:
                for slot in range(self.num_slots):
                    base_offset = SHM_HEADER_SIZE + (slot * SLOT_SIZE)
                    self.shm_handle.buf[base_offset : base_offset + SLOT_SIZE] = b"\x00" * SLOT_SIZE
                self._log("Zeroed IPC handle bytes in SharedMemory", force=True)
            except (OSError, BufferError) as e:
                self._log(f"Warning: Could not zero IPC handles: {e}", force=True)

        # Destroy CUDA Graph execs first — they hold references into the IPC stream
        # and (transitively) the ring-buffer pointers, so they must be torn down before
        # the events/stream/buffers below.
        if cuda_valid and getattr(self, "_use_graphs", False):
            self._destroy_export_graphs()

        # Destroy IPC events (sender-side resources, safe to destroy)
        if cuda_valid and hasattr(self, "ipc_events") and self.cuda:
            for slot, event in enumerate(self.ipc_events):
                if event:
                    try:
                        self.cuda.destroy_event(event)
                        self._log(f"Destroyed IPC event slot {slot}", force=True)
                    except (RuntimeError, OSError) as e:
                        self._log(f"Error destroying event slot {slot}: {e}", force=True)

        # Destroy GPU timing events (benchmarking resources)
        if cuda_valid and self.cuda:
            if hasattr(self, "_timing_start") and self._timing_start:
                try:
                    self.cuda.destroy_event(self._timing_start)
                    self._log("Destroyed GPU timing start event", force=False)
                except (RuntimeError, OSError) as e:
                    self._log(f"Error destroying timing start event: {e}", force=True)
                finally:
                    self._timing_start = None
            if hasattr(self, "_timing_end") and self._timing_end:
                try:
                    self.cuda.destroy_event(self._timing_end)
                    self._log("Destroyed GPU timing end event", force=False)
                except (RuntimeError, OSError) as e:
                    self._log(f"Error destroying timing end event: {e}", force=True)
                finally:
                    self._timing_end = None

        # Destroy dedicated IPC stream (set to None to prevent double-free).
        # CUDALINK_TD_PERSIST_STREAM=1: skip destroy so the stream survives
        # deactivate/reactivate cycles; initialize() reuses it via the existing
        # `if self.ipc_stream is None` guard.
        if cuda_valid and hasattr(self, "ipc_stream") and self.ipc_stream and self.cuda:
            if self._persist_stream:
                self._log(
                    f"[PERSIST_STREAM] keeping ipc_stream=0x{int(self.ipc_stream.value):016x} across cleanup",
                    force=True,
                )
            else:
                try:
                    self.cuda.destroy_stream(self.ipc_stream)
                    self._log("Destroyed IPC stream", force=True)
                    self.ipc_stream = None
                except (RuntimeError, OSError) as e:
                    self._log(f"Error destroying IPC stream: {e}", force=True)

        # Close SharedMemory (but don't unlink yet)
        if self.shm_handle:
            try:
                self.shm_handle.close()
                self._log("Closed SharedMemory", force=True)
            except (OSError, BufferError) as e:
                self._log(f"Error closing SharedMemory: {e}", force=True)

        # Grace period for receiver to close IPC handles
        if cuda_valid:
            time.sleep(0.1)  # 100ms for receiver to detect shutdown and close handles

        # Free GPU buffers (now safe, receiver has closed IPC handles)
        if cuda_valid and hasattr(self, "dev_ptrs") and self.cuda:
            for slot, dev_ptr in enumerate(self.dev_ptrs):
                if dev_ptr:
                    try:
                        self.cuda.free(dev_ptr)
                        self._log(f"Freed GPU buffer slot {slot}", force=True)
                    except (RuntimeError, OSError) as e:
                        self._log(f"Error freeing GPU buffer slot {slot}: {e}", force=True)

        # Free any pending deferred resources
        if cuda_valid and hasattr(self, "_pending_free_ptrs"):
            self._deferred_free()

        if self._warned_format:
            self._host.clear_status()
            self._warned_format = False

        # Unlink SharedMemory (sender is owner and should clean up)
        if hasattr(self, "shm_name"):
            try:
                try:
                    shm_temp = SharedMemory(name=self.shm_name)
                    shm_temp.close()
                    shm_temp.unlink()
                    self._log("Unlinked SharedMemory", force=True)
                except FileNotFoundError:
                    pass  # Already unlinked
            except (OSError, RuntimeError, AttributeError) as e:
                self._log(f"Warning: Could not unlink SharedMemory: {e}", force=True)

        # Reset state to prevent double-free on re-entry.
        # Use empty lists — initialize() will resize to current self.num_slots.
        self.dev_ptrs = []
        self.ipc_events = []
        self.ipc_handles = []
        self.ipc_event_handles = []
        if not self._persist_stream:
            self.ipc_stream = None
        self.shm_handle = None
        self._warned_format = False
        self._export_buffer = None

        # Reset per-session counters so averages are accurate after reinit
        # and slot selection starts from 0 (matching SharedMemory write_idx=0 written on init).
        self.write_idx = 0
        self.frame_count = 0
        self._profile = FrameProfile(_SENDER_PROFILE_REGIONS)
        self._last_status_tuple = None

        self._initialized = False
        self._log("Sender cleanup complete", force=True)
