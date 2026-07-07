"""
Shared helpers for the runnable examples — non-pedagogical boilerplate only.

Everything a numbered example needs to *run reliably* but that is not part of
the lesson lives here: hardware probing, unique SHM names, the torch/cupy
eager-buffer preflight, and a reusable demo producer process (so every consumer
example runs standalone without a live TouchDesigner network).

Deliberately NOT here — this repetition in the numbered scripts IS the lesson:
  * Exporter.open() / Importer.open() calls and their FrameSpec/ImportSpec/policy
    arguments,
  * export() / get_frame*() calls,
  * FrameOutcome / ImportOutcome branching.

Every helper is safe to call from a multiprocessing "spawn" child: this module
is imported by name (the examples put their own directory on sys.path before
``import _common``), and ``demo_producer_worker`` is a module-level function so
the spawn context can pickle a reference to it.
"""

from __future__ import annotations

import ctypes
import dataclasses
import multiprocessing
import os
import sys
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess

    from cuda_link import CudaPort


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

# The examples print typographic characters (arrows, en/em dashes).  When
# stdout is piped or redirected on Windows, Python defaults to the legacy
# cp1252 codec, which cannot encode them and raises UnicodeEncodeError.
# Substituting '?' beats crashing; interactive consoles are unaffected.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")


def banner(title: str) -> None:
    """Print a section header so multi-phase example output stays readable."""
    line = "=" * max(len(title) + 4, 64)
    print(f"\n{line}\n  {title}\n{line}", flush=True)


# ---------------------------------------------------------------------------
# SHM naming
# ---------------------------------------------------------------------------


def unique_shm_name(prefix: str) -> str:
    """Return a SharedMemory name that cannot collide with another run.

    PID + random suffix means two examples (or two invocations of the same
    example) never fight over one ring buffer.  Pass --shm-name explicitly
    when you want to attach to a live TouchDesigner sender instead.
    """
    return f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Hardware / optional-dependency probes
# ---------------------------------------------------------------------------


def probe_cuda() -> str | None:
    """Return None when the real CUDA runtime is usable, else the reason it is not.

    Instantiating CUDARuntimeAPI loads cudart via ctypes and initializes the
    device — the same probe tests/conftest.py uses to skip GPU tests.
    """
    try:
        from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

        CUDARuntimeAPI()
    except (RuntimeError, OSError) as e:
        return str(e)
    return None


def probe_opencv() -> bool:
    """True when opencv-python is importable (it is not a cuda-link extra)."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def torch_cuda_ready() -> bool:
    """True only when torch is installed AND built with working CUDA support."""
    try:
        import torch
    except (ImportError, OSError):
        return False
    try:
        return bool(torch.cuda.is_available())
    except RuntimeError:
        return False


def cupy_cuda_ready() -> bool:
    """True only when cupy is installed AND can reach at least one CUDA device."""
    try:
        import cupy
    except (ImportError, OSError):
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except (RuntimeError, OSError):
        return False


# ---------------------------------------------------------------------------
# The eager-buffer preflight ("the landmine")
# ---------------------------------------------------------------------------


def make_importer_open_safe(*, using_fake: bool) -> list[str]:
    """Ensure Importer.open() cannot crash in THIS process; return notes to print.

    Importer.open() eagerly builds per-slot zero-copy torch/cupy views whenever
    those packages are merely *importable* — it calls
    ``torch.as_tensor(wrapper, device="cuda")`` regardless of which CudaPort
    adapter you passed.  Two situations make that eager build blow up:

      * FakeCUDAAdapter: the IPC "pointers" are fabricated integers; wrapping
        them in a real torch/cupy GPU array is undefined behaviour.
      * A CPU-only torch (or broken cupy) install: ``device="cuda"`` raises
        even though your intent was the numpy path.

    The fix mirrors tests/integration/test_cpu_roundtrip.py: flip the importer
    module's TORCH_AVAILABLE / CUPY_AVAILABLE flags to False *before* open().
    Process-local and one-way — call this again in every spawn child that opens
    an Importer.  Full write-up: examples/README.md § Troubleshooting.
    """
    import cuda_link.importer as importer_module

    notes: list[str] = []
    if using_fake:
        if importer_module.TORCH_AVAILABLE or importer_module.CUPY_AVAILABLE:
            importer_module.TORCH_AVAILABLE = False
            importer_module.CUPY_AVAILABLE = False
            notes.append(
                "FakeCUDAAdapter mode: disabled torch/cupy zero-copy buffers "
                "(fake IPC pointers must never reach real torch/cupy)."
            )
        return notes
    if importer_module.TORCH_AVAILABLE and not torch_cuda_ready():
        importer_module.TORCH_AVAILABLE = False
        notes.append(
            "torch is installed but has no working CUDA — disabled torch zero-copy "
            "buffers so Importer.open() survives (see examples/README.md)."
        )
    if importer_module.CUPY_AVAILABLE and not cupy_cuda_ready():
        importer_module.CUPY_AVAILABLE = False
        notes.append(
            "cupy is installed but cannot reach a CUDA device — disabled cupy "
            "zero-copy buffers so Importer.open() survives (see examples/README.md)."
        )
    return notes


# ---------------------------------------------------------------------------
# Adapter selection
# ---------------------------------------------------------------------------


def pick_cuda_adapter(device: int = 0, *, force_fake: bool = False) -> tuple[CudaPort, bool]:
    """Return ``(adapter, is_real)`` — production adapter when CUDA works, fake otherwise.

    The fallback prints why, so a machine without an NVIDIA GPU still runs the
    examples end-to-end (SHM protocol, ring buffer, outcomes) with no real
    pixels moving.  Both sides of a producer/consumer pair must agree; passing
    the same ``force_fake`` to both keeps them consistent.
    """
    if not force_fake:
        problem = probe_cuda()
        if problem is None:
            from cuda_link import CTypesCUDAAdapter

            return CTypesCUDAAdapter.for_device(device=device), True
        print(f"[examples] Real CUDA unavailable ({problem}) — using FakeCUDAAdapter.", flush=True)
    from cuda_link import FakeCUDAAdapter

    return FakeCUDAAdapter(device=device), False


# ---------------------------------------------------------------------------
# Producer discovery
# ---------------------------------------------------------------------------


def wait_for_shm(shm_name: str, timeout_s: float = 20.0) -> bool:
    """Poll until the named SharedMemory exists. True if it appeared in time.

    A freshly spawned producer needs ~1-2 s to load CUDA and allocate the ring
    buffer; polling SharedMemory existence is the cheap way to wait for it
    (no CUDA calls in this process until the producer is actually there).
    """
    from multiprocessing.shared_memory import SharedMemory

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            shm = SharedMemory(name=shm_name)
            shm.close()
            return True
        except FileNotFoundError:
            time.sleep(0.1)
    return False


# ---------------------------------------------------------------------------
# Demo producer — stands in for TouchDesigner so every example runs standalone
# ---------------------------------------------------------------------------


def demo_producer_worker(
    shm_name: str,
    height: int,
    width: int,
    channels: int,
    dtype: str,
    num_frames: int,
    fps: float,
    num_slots: int,
    device: int,
    force_fake: bool,
    doorbell: bool,
) -> None:
    """Export ``num_frames`` synthetic frames, then close (sets the shutdown flag).

    Runs in a separate OS process (spawn) — CUDA IPC requires that: on Windows,
    cudaIpcOpenMemHandle fails with error 201 when the opening process is the
    one that created the handle.  Module-level so spawn can pickle it.

    Each frame is filled with a per-frame constant (frame index, wrapped) so
    consumers can watch pixel values change and verify real data flow.
    """
    import numpy as np

    from cuda_link import Exporter, ExportPolicy, FrameSpec, GpuFrame

    adapter, is_real = pick_cuda_adapter(device, force_fake=force_fake)
    # for_testing() turns off CUDA Graphs and the activation barrier — both
    # need a real driver; production defaults stay on for the real adapter.
    policy = (
        ExportPolicy(doorbell=doorbell)
        if is_real
        else dataclasses.replace(ExportPolicy.for_testing(), doorbell=doorbell)
    )

    exporter = Exporter.open(
        FrameSpec(
            shm_name=shm_name,
            height=height,
            width=width,
            channels=channels,
            dtype=dtype,
            num_slots=num_slots,
            device=device,
        ),
        policy=policy,
        cuda=adapter,
    )
    data_size = exporter.data_size
    src_ptr = adapter.malloc(data_size)
    staging = np.empty((height, width, channels), dtype=dtype)
    interval = 1.0 / fps if fps > 0 else 0.0
    try:
        for i in range(num_frames):
            if is_real:
                # Animate: integer dtypes count 0..255, float dtypes ramp 0.0..1.0.
                staging[...] = (i % 256) if staging.dtype.kind in "iu" else (i % 256) / 255.0
                adapter.memcpy(
                    dst=src_ptr,
                    src=ctypes.c_void_p(staging.ctypes.data),
                    count=data_size,
                    kind=1,  # cudaMemcpyHostToDevice
                )
            # The H2D copy above is synchronous, so the source buffer is
            # complete; recording the sync point on the default stream (0)
            # arms the exporter's cross-stream ordering.  Armed on the fake
            # too (its event ops are no-ops) so --force-fake runs don't show
            # the un-ordered-export warning.  Example 07 tells the full
            # ordering-rule story for asynchronous producers.
            exporter.record_source_sync(0)
            exporter.export(GpuFrame(ptr=int(src_ptr.value or 0), size=data_size))
            if interval:
                time.sleep(interval)
        # Grace period: let the consumer drain the last frame before close()
        # publishes the shutdown flag.
        time.sleep(1.0)
    finally:
        adapter.free(src_ptr)
        exporter.close()


def spawn_demo_producer(
    shm_name: str,
    *,
    height: int = 512,
    width: int = 512,
    channels: int = 4,
    dtype: str = "uint8",
    num_frames: int = 300,
    fps: float = 60.0,
    num_slots: int = 3,
    device: int = 0,
    force_fake: bool = False,
    doorbell: bool = False,
) -> BaseProcess:
    """Start demo_producer_worker in a spawn process and return the handle.

    daemon=True so an interrupted example never leaves an orphan producer;
    examples still join() it in their finally block for a clean shutdown.
    """
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=demo_producer_worker,
        args=(
            shm_name,
            height,
            width,
            channels,
            dtype,
            num_frames,
            fps,
            num_slots,
            device,
            force_fake,
            doorbell,
        ),
        daemon=True,
        name=f"demo-producer-{shm_name}",
    )
    proc.start()
    return proc
