"""
Importer port — Protocol, value objects, and outcome type.

Contains everything a caller needs to express "what the Importer needs from CUDA"
as a structural type, plus the four value objects that form the public interface:

  ImportSpec        — immutable frame geometry + SHM routing + timeout
  ImportPolicy      — immutable behavioural knobs (env-readable, preset constructors)
  ImportResult      — result of Importer.get_frame*() (generic over frame type)
  ImportOutcome     — NEW_FRAME / NO_FRAME / SHUTDOWN / RECONNECTING / TIMEOUT
  ImporterCudaPort  — Protocol satisfied by CTypesCUDAAdapter and FakeCUDAAdapter
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, TypeVar

from Env import env_bool, env_int, env_str
from ExporterPort import CudaPort as ImporterCudaPort  # noqa: F401 — re-exported for callers

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
    # opt-in pipelined D2H — enqueue current slot's copy, return previous frame.
    # First call returns NO_FRAME; steady-state adds +1 frame latency in exchange for
    # overlapping D2H with consumer CPU work.  Only beneficial when consumer takes
    # longer than D2H copy time (~0.5–2 ms for 4K RGBA).
    d2h_pipelined: bool = False
    # R1: GPU-side wait for get_frame() (torch backend only).
    # When True and no explicit stream= is passed, issues cudaStreamWaitEvent on
    # torch.cuda.current_stream() instead of CPU-polling the IPC event.  The CPU
    # returns immediately; ordering is enforced by the GPU scheduler.  Consequence:
    # ImportOutcome.TIMEOUT is unreachable on this path (a hung producer stalls the
    # stream rather than raising TimeoutError).  Opt-in; default=False preserves the
    # existing CPU-spin/sleep behaviour and TIMEOUT detection.
    # Enable via CUDALINK_TORCH_GPU_WAIT=1.
    torch_gpu_wait: bool = False
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
            d2h_pipelined=env_bool("CUDALINK_D2H_PIPELINED", default=False),
            torch_gpu_wait=env_bool("CUDALINK_TORCH_GPU_WAIT", default=False),
            debug=False,
            reconnect_enabled=env_bool("CUDALINK_IMPORT_RECONNECT", default=True),
            reconnect_max_attempts=env_int("CUDALINK_IMPORT_RECONNECT_MAX_ATTEMPTS", default=20),
        )

    @classmethod
    def for_testing(cls) -> ImportPolicy:
        """Preset safe for unit tests without a real GPU.

        Disables spin-wait, multi-stream D2H, stream priority, and enables
        pageable fallback so tests can run with FakeCUDAAdapter without any GPU.
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
# ImporterCudaPort — alias for CudaPort
# ---------------------------------------------------------------------------
# CudaPort (in _exporter_port.py) is the unified CUDA Protocol covering both
# exporter and importer operations.  ImporterCudaPort is kept as an explicit
# alias so existing callers (importer.py, __init__.py, tests) need no changes.
#
# The alias was imported above:
#   from ._exporter_port import CudaPort as ImporterCudaPort
#
# Both CTypesCUDAAdapter and FakeCUDAAdapter satisfy it structurally.
