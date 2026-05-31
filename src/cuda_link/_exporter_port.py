"""
Exporter port — Protocol, value objects, and outcome type.

Contains everything a caller needs to express "what the Exporter needs from CUDA"
as a structural type, plus the four value objects that form the public interface:

  FrameSpec     — immutable frame geometry + SHM routing
  ExportPolicy  — immutable behavioural knobs (env-readable, preset constructors)
  GpuFrame      — a single frame to export
  FrameOutcome  — result of Exporter.export()
  CudaPort      — Protocol satisfied by CTypesCUDAAdapter (prod) and FakeCUDAAdapter (test)
"""

from __future__ import annotations

from ctypes import c_void_p
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

from ._env import env_bool, env_int, env_str
from .cuda_runtime_types import (
    CUDAEvent_t,
    CUDAGraph_t,
    CUDAGraphExec_t,
    CUDAGraphNode_t,
    CUDAStream_t,
    cudaIpcEventHandle_t,
    cudaIpcMemHandle_t,
)

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameSpec:
    """Immutable description of one export channel — geometry + SHM routing.

    Both the exporter and any downstream receiver must agree on all fields.
    The SHM name is the only routing key; the remaining fields describe the GPU
    buffer layout that the protocol header encodes.
    """

    shm_name: str
    height: int
    width: int
    channels: int = 4
    dtype: str = "uint8"
    num_slots: int = 3
    device: int = 0
    extra_flags: int = 0  # extra metadata flags to OR into the SHM Metadata.flags field


@dataclass(frozen=True)
class ExportPolicy:
    """Immutable set of behavioural knobs for the Exporter.

    All CUDALINK_* env-var reads are concentrated in from_env(). Pass a frozen
    ExportPolicy into Exporter.open() so the exporter never touches os.environ
    on the per-frame hot path.
    """

    export_sync: bool = False
    use_graphs: bool = True
    flush_probe: bool = True
    strict_device: bool = False
    barrier_enabled: bool = True
    barrier_stale_ns: int = 5_000_000_000
    high_priority_stream: bool = True
    export_profile: bool = False

    @classmethod
    def from_env(cls) -> ExportPolicy:
        """Read all CUDALINK_* env vars and return a frozen policy."""
        return cls(
            export_sync=env_bool("CUDALINK_EXPORT_SYNC", default=False),
            use_graphs=env_bool("CUDALINK_USE_GRAPHS", default=True),
            flush_probe=env_bool("CUDALINK_EXPORT_FLUSH_PROBE", default=True),
            strict_device=env_bool("CUDALINK_STRICT_DEVICE", default=False),
            barrier_enabled=env_bool("CUDALINK_ACTIVATION_BARRIER", default=True),
            barrier_stale_ns=env_int("CUDALINK_BARRIER_STALE_NS", default=5_000_000_000),
            high_priority_stream=env_str("CUDALINK_LIB_STREAM_PRIO", default="high") != "normal",
            export_profile=env_bool("CUDALINK_EXPORT_PROFILE", default=False),
        )

    @classmethod
    def low_latency(cls) -> ExportPolicy:
        """Preset for minimum per-frame overhead.

        Disables CPU-blocking stream sync and flush probe. Suitable for
        single-producer setups without a co-resident TD Sender on the same machine.
        """
        return cls(
            export_sync=False,
            flush_probe=False,
            use_graphs=True,
            strict_device=False,
            barrier_enabled=True,
            high_priority_stream=True,
            export_profile=False,
        )

    @classmethod
    def for_testing(cls) -> ExportPolicy:
        """Preset safe for unit tests without a real GPU.

        Disables CUDA Graphs (require cudart 11.4+), export sync, flush probe,
        activation barrier, profiling, and device-affinity checking so tests can
        run with a FakeCUDAAdapter without touching any GPU resource.
        """
        return cls(
            export_sync=False,
            use_graphs=False,
            flush_probe=False,
            strict_device=False,
            barrier_enabled=False,
            barrier_stale_ns=0,
            high_priority_stream=False,
            export_profile=False,
        )


@dataclass(frozen=True)
class GpuFrame:
    """A single GPU frame ready for export.

    ptr and size describe the source device buffer. producer_stream, if given,
    is the raw CUDA stream handle (as int) on which the producer has already
    enqueued work that writes to ptr. The Exporter will issue a GPU-side
    stream_wait_event before copying, maintaining ordering without blocking the CPU.
    """

    ptr: int
    size: int
    producer_stream: int | None = None


class FrameOutcome(Enum):
    """Result of a single Exporter.export() call."""

    PUBLISHED = auto()
    SKIPPED_BARRIER = auto()  # activation-barrier signalled: skip this frame
    SKIPPED_NOT_READY = auto()  # ring slot still held by consumer (not yet used)
    FAILED = auto()  # unrecoverable error; caller should close()


# ---------------------------------------------------------------------------
# CudaPort Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CudaPort(Protocol):
    """Structural interface that Exporter requires from the CUDA runtime.

    Production adapter: CTypesCUDAAdapter  (wraps CUDARuntimeAPI; in _cuda_adapters.py)
    Test adapter:       FakeCUDAAdapter    (in-memory, no GPU needed; in _cuda_adapters.py)

    All methods raise RuntimeError on CUDA failure (mirrors CUDARuntimeAPI.check_error).
    """

    # --- Device ------------------------------------------------------------

    def get_device(self) -> int:
        """Return the CUDA device index currently bound to this context."""
        ...

    def peek_last_error(self) -> int:
        """Non-destructively read the thread-local sticky CUDA error code.

        Returns 0 (SUCCESS) when no error is latched. Does NOT clear the error.
        Used by the Exporter to probe whether the CUDA context is still valid
        (e.g., before attempting cleanup after a process-level failure).
        """
        ...

    # --- Memory ------------------------------------------------------------

    def malloc(self, size: int) -> c_void_p:
        """Allocate size bytes of device memory. Returns an opaque device pointer."""
        ...

    def free(self, dev_ptr: c_void_p) -> None:
        """Free device memory previously allocated with malloc()."""
        ...

    def memcpy_async(
        self,
        dst: c_void_p,
        src: c_void_p,
        count: int,
        kind: int,
        stream: CUDAStream_t,
    ) -> None:
        """Enqueue an async memory copy on stream. kind=3 for device-to-device."""
        ...

    # --- Streams -----------------------------------------------------------

    def create_stream(self, flags: int = 0x01) -> CUDAStream_t:
        """Create a CUDA stream. flags=0x01 = cudaStreamNonBlocking."""
        ...

    def create_stream_with_priority(self, flags: int = 0x01, priority: int | None = None) -> CUDAStream_t:
        """Create a high-priority stream. None → highest priority on this device."""
        ...

    def destroy_stream(self, stream: CUDAStream_t) -> None:
        """Destroy a CUDA stream."""
        ...

    def stream_wait_event(self, stream: CUDAStream_t, event: CUDAEvent_t, flags: int = 0) -> None:
        """GPU-side: make stream wait until event has been recorded (non-blocking CPU)."""
        ...

    def stream_synchronize(self, stream: CUDAStream_t) -> None:
        """CPU-blocking wait until all operations on stream have completed."""
        ...

    def stream_query(self, stream: CUDAStream_t) -> bool:
        """Non-blocking: True if all stream operations have completed."""
        ...

    # --- Events ------------------------------------------------------------

    def create_ipc_event(self) -> CUDAEvent_t:
        """Create a CUDA event with cudaEventInterprocess | cudaEventDisableTiming."""
        ...

    def create_sync_event(self) -> CUDAEvent_t:
        """Create a timing-disabled event for stream-ordering use."""
        ...

    def ipc_get_event_handle(self, event: CUDAEvent_t) -> cudaIpcEventHandle_t:
        """Export an IPC event handle for sharing with another process."""
        ...

    def record_event(self, event: CUDAEvent_t, stream: CUDAStream_t | None = None) -> None:
        """Record event on stream (None → default stream)."""
        ...

    def destroy_event(self, event: CUDAEvent_t) -> None:
        """Destroy a CUDA event."""
        ...

    # --- IPC memory --------------------------------------------------------

    def ipc_get_mem_handle(self, dev_ptr: c_void_p) -> cudaIpcMemHandle_t:
        """Export a device pointer as an IPC memory handle."""
        ...

    # --- Pointer attributes ------------------------------------------------

    def pointer_get_attributes(self, ptr: int) -> Any:
        """Return cudaPointerAttributes-like object with .type and .device fields."""
        ...

    # --- Error checking ----------------------------------------------------

    def check_sticky_error(self, context: str) -> None:
        """Warn and raise if a sticky CUDA error is latched. No-op when all is well."""
        ...

    # --- CUDA Graphs (optional path; gated by ExportPolicy.use_graphs) ----

    def get_runtime_version(self) -> int:
        """Return the cudart version integer (e.g., 12080 = CUDA 12.8)."""
        ...

    def stream_begin_capture(self, stream: CUDAStream_t, mode: int = 0) -> None:
        """Begin capturing stream into a graph. mode=2 = relaxed."""
        ...

    def stream_end_capture(self, stream: CUDAStream_t) -> CUDAGraph_t:
        """End capture; return the template graph."""
        ...

    def graph_instantiate(self, graph: CUDAGraph_t, flags: int = 0) -> CUDAGraphExec_t:
        """Instantiate a graph template into an executable graph."""
        ...

    def graph_launch(self, graph_exec: CUDAGraphExec_t, stream: CUDAStream_t) -> None:
        """Launch an executable graph on stream."""
        ...

    def graph_destroy(self, graph: CUDAGraph_t) -> None:
        """Destroy a graph template."""
        ...

    def graph_exec_destroy(self, graph_exec: CUDAGraphExec_t) -> None:
        """Destroy an executable graph."""
        ...

    def graph_get_nodes(self, graph: CUDAGraph_t) -> list[CUDAGraphNode_t]:
        """Return all nodes of a graph in capture order."""
        ...

    def graph_exec_memcpy_node_set_params_1d(
        self,
        graph_exec: CUDAGraphExec_t,
        node: CUDAGraphNode_t,
        dst: c_void_p,
        src: c_void_p,
        count: int,
        kind: int,
    ) -> None:
        """Update a 1D memcpy node's src/dst per ring slot. CUDA 11.3+."""
        ...

    # --- IPC memory (consumer / Importer side) ----------------------------

    def ipc_open_mem_handle(self, handle: cudaIpcMemHandle_t, flags: int = 1) -> c_void_p:
        """Open an IPC memory handle exported by the producer process.

        flags=1 = cudaIpcMemLazyEnablePeerAccess.
        """
        ...

    def ipc_close_mem_handle(self, dev_ptr: c_void_p) -> None:
        """Close an IPC memory handle opened with ipc_open_mem_handle()."""
        ...

    def ipc_open_event_handle(self, handle: cudaIpcEventHandle_t) -> CUDAEvent_t:
        """Open an IPC event handle exported by the producer process."""
        ...

    # --- Events (consumer) ------------------------------------------------

    def query_event(self, event: CUDAEvent_t) -> bool:
        """Non-blocking: True if the event has been recorded and completed."""
        ...

    # --- Device sync -------------------------------------------------------

    def synchronize(self) -> None:
        """CPU-blocking wait until all operations on the current device have completed."""
        ...

    # --- Pinned host memory (Importer side) --------------------------------

    def malloc_host_alloc(self, size: int, flags: int = 0x01) -> c_void_p:
        """Allocate portable pinned host memory via cudaHostAlloc.

        flags=0x01 = cudaHostAllocPortable (accessible from any CUDA context).
        """
        ...

    def free_host(self, ptr: c_void_p) -> None:
        """Free pinned host memory allocated with malloc_host_alloc()."""
        ...

    def host_register(self, ptr: int, size: int, flags: int = 0) -> None:
        """Page-lock an existing host allocation via cudaHostRegister."""
        ...

    def host_unregister(self, ptr: int) -> None:
        """Unregister a page-locked host allocation."""
        ...
