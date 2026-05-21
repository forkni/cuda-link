"""
Importer port — Protocol, value objects, and outcome type.

Contains everything a caller needs to express "what the Importer needs from CUDA"
as a structural type, plus the four value objects that form the public interface:

  ImportSpec        — immutable frame geometry + SHM routing + timeout
  ImportPolicy      — immutable behavioural knobs (env-readable, preset constructors)
  ImportResult      — result of Importer.get_frame*() (generic over frame type)
  ImportOutcome     — NEW_FRAME / NO_FRAME / SHUTDOWN / RECONNECTING / TIMEOUT
  ImporterCudaPort  — Protocol satisfied by CTypesCudaAdapter and FakeCudaAdapter
"""

from __future__ import annotations

from ctypes import c_void_p
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, Protocol, TypeVar, runtime_checkable

from ._env import env_bool, env_int, env_str
from .cuda_runtime_types import (
    CUDAEvent_t,
    CUDAStream_t,
    cudaIpcEventHandle_t,
    cudaIpcMemHandle_t,
)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportSpec:
    """Immutable description of one import channel — SHM routing + geometry hints.

    shm_name is the only required field. shape and dtype may be None to enable
    auto-detection from the SHM metadata block written by the producer; if
    absent from SHM the Importer falls back to (512, 512, 4) / 'float32'.

    Both the Importer and any downstream consumer must agree on all fields.
    The SHM name is the only routing key.
    """

    shm_name: str
    device: int = 0
    shape: tuple[int, int, int] | None = None
    dtype: str | None = None
    timeout_ms: float = 5000.0


@dataclass(frozen=True)
class ImportPolicy:
    """Immutable set of behavioural knobs for the Importer.

    All CUDALINK_* env-var reads are concentrated in from_env(). Pass a frozen
    ImportPolicy into Importer.open() so the importer never touches os.environ
    on the per-frame hot path.
    """

    wait_spin_us: int = 200
    d2h_num_streams: int = 1
    d2h_stream_high_priority: bool = False
    allow_pageable_fallback: bool = False
    debug: bool = False
    reconnect_enabled: bool = True
    reconnect_max_attempts: int = 20
    reconnect_backoff_frames: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 120)

    @classmethod
    def from_env(cls) -> ImportPolicy:
        """Read all CUDALINK_* env vars and return a frozen policy."""
        return cls(
            wait_spin_us=env_int("CUDALINK_WAIT_SPIN_US", default=200),
            d2h_num_streams=max(1, env_int("CUDALINK_D2H_STREAMS", default=1)),
            d2h_stream_high_priority=env_str("CUDALINK_D2H_STREAM_PRIO", default="normal") == "high",
            allow_pageable_fallback=env_bool("CUDALINK_ALLOW_PAGEABLE_FALLBACK", default=False),
            debug=False,
            reconnect_enabled=env_bool("CUDALINK_IMPORT_RECONNECT", default=True),
            reconnect_max_attempts=env_int("CUDALINK_IMPORT_RECONNECT_MAX_ATTEMPTS", default=20),
        )

    @classmethod
    def for_testing(cls) -> ImportPolicy:
        """Preset safe for unit tests without a real GPU.

        Disables spin-wait, multi-stream D2H, stream priority, and enables
        pageable fallback so tests can run with FakeCudaAdapter without any GPU.
        reconnect_enabled=False so unit tests don't incur reconnect-wait delays.
        """
        return cls(
            wait_spin_us=0,
            d2h_num_streams=1,
            d2h_stream_high_priority=False,
            allow_pageable_fallback=True,
            debug=False,
            reconnect_enabled=False,
        )

    @classmethod
    def low_latency(cls) -> ImportPolicy:
        """Preset for minimum per-frame overhead.

        Maximises spin-wait time and enables two-stream D2H for large buffers.
        Suitable for tight consumer loops where CPU sleeps increase jitter.
        """
        return cls(
            wait_spin_us=2000,
            d2h_num_streams=2,
            d2h_stream_high_priority=True,
            allow_pageable_fallback=False,
            debug=False,
        )


class ImportOutcome(Enum):
    """Result classification for a single Importer.get_frame*() call."""

    NEW_FRAME = auto()  # new frame available; ImportResult.frame is populated
    NO_FRAME = auto()  # ring slot not yet updated by producer
    SHUTDOWN = auto()  # producer set shutdown flag; Importer is now closed
    RECONNECTING = auto()  # SHM version changed; IPC handles reopened (retry next tick)
    TIMEOUT = auto()  # producer event wait exceeded ImportSpec.timeout_ms


@dataclass(frozen=True)
class ImportResult(Generic[T]):
    """Result of a single Importer.get_frame*() call.

    outcome is always set. frame is the tensor/array only when
    outcome is ImportOutcome.NEW_FRAME, else None.

    Usage::

        result = importer.get_frame()
        if result.outcome is ImportOutcome.NEW_FRAME:
            process(result.frame)
        elif result.outcome is ImportOutcome.SHUTDOWN:
            break
    """

    outcome: ImportOutcome
    frame: T | None = None


# ---------------------------------------------------------------------------
# ImporterCudaPort Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ImporterCudaPort(Protocol):
    """Structural interface that Importer requires from the CUDA runtime.

    Production adapter: CTypesCudaAdapter  (wraps CUDARuntimeAPI; in _cuda_adapters.py)
    Test adapter:       FakeCudaAdapter    (in-memory, no GPU needed; in _cuda_adapters.py)

    CTypesCudaAdapter and FakeCudaAdapter from _cuda_adapters.py satisfy both
    CudaPort (exporter) and ImporterCudaPort (importer) structurally — the same
    adapter classes serve both sides without code duplication.

    All methods raise RuntimeError on CUDA failure.
    """

    # --- Device ------------------------------------------------------------

    def get_device(self) -> int:
        """Return the CUDA device index currently bound to this context."""
        ...

    def peek_last_error(self) -> int:
        """Non-destructively read the thread-local sticky CUDA error code.

        Returns 0 (SUCCESS) when no error is latched. Does NOT clear the error.
        """
        ...

    # --- IPC memory --------------------------------------------------------

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

    # --- Events ------------------------------------------------------------

    def query_event(self, event: CUDAEvent_t) -> bool:
        """Non-blocking: True if the event has been recorded and completed."""
        ...

    def create_sync_event(self) -> CUDAEvent_t:
        """Create a timing-disabled event for stream-ordering use."""
        ...

    def destroy_event(self, event: CUDAEvent_t) -> None:
        """Destroy a CUDA event."""
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

    def synchronize(self) -> None:
        """CPU-blocking wait until all operations on the current device have completed."""
        ...

    # --- Memory (D2H) ------------------------------------------------------

    def memcpy_async(
        self,
        dst: c_void_p,
        src: c_void_p,
        count: int,
        kind: int,
        stream: CUDAStream_t,
    ) -> None:
        """Enqueue an async memory copy on stream. kind=2 for device-to-host."""
        ...

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

    # --- Error checking ----------------------------------------------------

    def check_sticky_error(self, context: str) -> None:
        """Warn and raise if a sticky CUDA error is latched. No-op when all is well."""
        ...
