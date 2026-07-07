"""
Example 07 — Python → TouchDesigner: the Exporter side, plus a latency benchmark.

WHAT
    Reverses the usual direction: YOUR Python program is the producer.  It
    renders frames on the GPU (with torch if available, raw CUDA otherwise),
    exports them through cuda-link, and measures what export() actually costs
    per frame (mean / median / p95 / p99).  A demo consumer process drains the
    ring like TD's receiver would.  Combines docs Examples 7 and 6.

HARDWARE
    NVIDIA GPU.  torch optional (used to render when CUDA-enabled).
    numpy required (for the ctypes render path and the demo consumer).
    --force-fake runs the mechanics without a GPU (timings meaningless).

HOW TO RUN
    python examples/07_python_to_td_exporter_and_benchmark.py
    python examples/07_python_to_td_exporter_and_benchmark.py --frames 600 --width 1920 --height 1080
    # pure benchmark, nothing draining the ring:
    python examples/07_python_to_td_exporter_and_benchmark.py --no-demo-consumer

WHAT IT TEACHES
    * FrameSpec/ExportPolicy/Exporter.open on the producer side.
    * GpuFrame is just (device pointer, byte size) — cuda-link does not care
      whether the pointer came from torch, CuPy, or cudaMalloc.
    * THE ORDERING RULE (correctness, not tuning): export() enqueues a
      device-to-device copy on cuda-link's own CUDA stream.  If your kernels
      write the source buffer on a DIFFERENT stream, you must hand cuda-link
      a sync point — record_source_sync(stream) after your writes (or pass
      producer_stream in GpuFrame, same thing).  Skip it and the copy can
      race your kernel: the consumer receives a torn, half-written frame.
      CPU-side torch.cuda.synchronize() also works but stalls your pipeline.
    * What export() costs: it is non-blocking on the steady path — the D2D
      copy is enqueued, not awaited — so p50 is typically well under a
      millisecond even at 1080p.
    * ExportPolicy(export_profile=True): the exporter's built-in per-region
      profiling — get_stats() then reports avg_memcpy_us / avg_total_us, a
      free cross-check of your own wall-clock numbers.
"""

from __future__ import annotations

import argparse
import dataclasses
import multiprocessing
import queue
import statistics
import sys
import time
from pathlib import Path

# Make _common importable when this file runs as a script from any cwd.
# Unconditional at module top: the spawn child re-imports this module too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common

# ---------------------------------------------------------------------------
# Demo consumer — drains the ring like TD's receiver (module level for spawn)
# ---------------------------------------------------------------------------


def _drain_consumer_worker(shm_name: str, force_fake: bool, rq: object) -> None:
    """Consume frames until the producer shuts down; report how many arrived.

    TD's actual receiver maps the ring buffer zero-copy on the GPU; this
    stand-in drains via the numpy path only because it is the fewest moving
    parts — the export() side being benchmarked is identical either way.
    """
    try:
        from cuda_link import Importer, ImportOutcome, ImportPolicy, ImportSpec

        if not _common.wait_for_shm(shm_name, timeout_s=20.0):
            rq.put(("ERROR", "producer SHM never appeared"))  # type: ignore[attr-defined]
            return
        time.sleep(0.3)

        adapter, is_real = _common.pick_cuda_adapter(0, force_fake=force_fake)
        _common.make_importer_open_safe(using_fake=not is_real)

        importer = Importer.open(
            ImportSpec(shm_name=shm_name, shape=None, dtype=None),
            policy=None if is_real else ImportPolicy.for_testing(),
            cuda=adapter,
        )
        # Handshake: tell the producer we are attached.  It must not start
        # (or worse, finish) the benchmark before this point — see main().
        rq.put(("CONNECTED", None))  # type: ignore[attr-defined]
        received = 0
        deadline = time.perf_counter() + 120.0
        while time.perf_counter() < deadline:
            result = importer.get_frame_numpy()
            if result.outcome is ImportOutcome.NEW_FRAME:
                received += 1
            elif result.outcome is ImportOutcome.NO_FRAME:
                time.sleep(0.001)
            elif result.outcome in (ImportOutcome.SHUTDOWN, ImportOutcome.TIMEOUT):
                break
        importer.close()
        rq.put(("OK", received))  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001 — report ANY child failure to the parent
        import traceback

        rq.put(("ERROR", f"{e}\n{traceback.format_exc()}"))  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Producer + benchmark (this process)
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--frames", type=int, default=300, help="timed export() calls (default 300)")
    parser.add_argument("--warmup", type=int, default=30, help="untimed warmup frames (default 30)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-demo-consumer", action="store_true", help="benchmark with nothing draining the ring")
    parser.add_argument("--force-fake", action="store_true", help="run without a GPU (timings meaningless)")
    args = parser.parse_args()

    import ctypes

    import numpy as np

    from cuda_link import Exporter, ExportPolicy, FrameSpec, GpuFrame

    if not args.force_fake:
        problem = _common.probe_cuda()
        if problem is not None:
            print(f"This example needs a real NVIDIA GPU (CUDA runtime unavailable: {problem}).")
            print("Re-run with --force-fake to walk the mechanics without one.")
            return 2

    shm_name = _common.unique_shm_name("cudalink_ex07")
    use_torch = not args.force_fake and _common.torch_cuda_ready()

    # ------------------------------------------------------------------
    # Optional demo consumer (plays TD's role)
    # ------------------------------------------------------------------
    consumer = None
    rq = None
    if not args.no_demo_consumer:
        _common.banner("Spawn demo consumer (stands in for TD's receiver)")
        ctx = multiprocessing.get_context("spawn")
        rq = ctx.Queue()
        consumer = ctx.Process(target=_drain_consumer_worker, args=(shm_name, args.force_fake, rq), daemon=True)
        consumer.start()
        print(f"consumer pid={consumer.pid}")

    # ------------------------------------------------------------------
    # Open the Exporter — this is the code that goes in YOUR program
    # ------------------------------------------------------------------
    _common.banner("Open Exporter")
    adapter, is_real = _common.pick_cuda_adapter(0, force_fake=args.force_fake)
    exporter = Exporter.open(
        FrameSpec(shm_name=shm_name, height=args.height, width=args.width, channels=4, dtype="uint8"),
        # Production defaults (CUDA Graphs on) for real hardware; for_testing()
        # strips everything driver-dependent for the fake.  export_profile=True
        # turns on the exporter's built-in per-region timing — without it,
        # get_stats()'s avg_memcpy_us / avg_total_us stay 0.0.
        policy=(
            ExportPolicy(export_profile=True)
            if is_real
            else dataclasses.replace(ExportPolicy.for_testing(), export_profile=True)
        ),
        cuda=adapter,
    )
    data_size = exporter.data_size
    print(f"shm='{shm_name}'  {args.width}x{args.height} RGBA uint8  data_size={data_size / 1e6:.1f} MB")

    # ------------------------------------------------------------------
    # Source buffer: torch when available, raw cudaMalloc otherwise.
    # GpuFrame only ever sees (pointer, size) — the origin is irrelevant.
    # ------------------------------------------------------------------
    torch = None
    frame_tensor = None
    src_ptr = None
    if use_torch:
        import torch

        frame_tensor = torch.empty((args.height, args.width, 4), dtype=torch.uint8, device="cuda")
        src_addr = frame_tensor.data_ptr()
        print("render path: torch tensor on GPU (with per-frame record_source_sync)")
    else:
        src_ptr = adapter.malloc(data_size)
        src_addr = int(src_ptr.value or 0)
        if is_real:
            staging = np.full((args.height, args.width, 4), 128, dtype=np.uint8)
            adapter.memcpy(
                dst=src_ptr,
                src=ctypes.c_void_p(staging.ctypes.data),
                count=data_size,
                kind=1,  # cudaMemcpyHostToDevice
            )
        # The buffer above is written ONCE, synchronously, before the loop —
        # a single sync point on the default stream (0) orders every export
        # after it (the flag is sticky).  The torch path below re-arms per
        # frame instead, because there the buffer changes every frame.
        # No-op on the fake, but keeps --force-fake runs warning-free.
        exporter.record_source_sync(0)
        print("render path: raw CUDA allocation (install CUDA-enabled torch for the torch path)")

    def render_and_export(i: int) -> object:
        """One frame: write pixels on the GPU, sync-point, export."""
        if use_torch and frame_tensor is not None and torch is not None:
            # "Render": a GPU kernel writes the frame on torch's current stream.
            frame_tensor.fill_(i % 256)
            # THE ORDERING RULE in action — tell cuda-link's IPC stream to
            # GPU-wait for the write above before its D2D copy.  Equivalent:
            # GpuFrame(..., producer_stream=torch.cuda.current_stream().cuda_stream)
            exporter.record_source_sync(torch.cuda.current_stream().cuda_stream)
        return exporter.export(GpuFrame(ptr=src_addr, size=data_size))

    # ------------------------------------------------------------------
    # Handshake: wait until the demo consumer is actually attached.  The
    # spawn child needs seconds to boot Python + CUDA, while the timed loop
    # below finishes in MILLISECONDS — without this wait the exporter could
    # open, publish every frame, and close before the consumer's
    # Importer.open() ever finds the ring buffer.
    # ------------------------------------------------------------------
    if consumer is not None and rq is not None:
        try:
            status, payload = rq.get(timeout=60.0)
        except queue.Empty:
            status, payload = "ERROR", "consumer never reported readiness within 60 s"
        if status != "CONNECTED":
            print("demo consumer failed to start:", payload)
            exporter.close()
            consumer.terminate()
            return 1
        print("consumer connected — benchmark runs against a live drain")

    # ------------------------------------------------------------------
    # Warmup, then timed loop
    # ------------------------------------------------------------------
    _common.banner(f"Benchmark: {args.warmup} warmup + {args.frames} timed frames")
    # Warmup absorbs one-time costs (CUDA Graph capture, lazy allocations,
    # driver JIT) so the timed numbers reflect the steady state.
    for i in range(args.warmup):
        render_and_export(i)

    timings_ms: list[float] = []
    outcomes: dict[str, int] = {}
    for i in range(args.frames):
        t0 = time.perf_counter()
        outcome = render_and_export(args.warmup + i)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
        outcomes[outcome.name] = outcomes.get(outcome.name, 0) + 1  # type: ignore[attr-defined]

    time.sleep(1.0)  # let the consumer drain the last frames
    if src_ptr is not None:
        adapter.free(src_ptr)
    exporter.close()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    _common.banner("Results")
    q = statistics.quantiles(timings_ms, n=100)  # q[k] = (k+1)-th percentile
    mean = statistics.fmean(timings_ms)
    print(f"export() wall time over {args.frames} frames ({args.width}x{args.height} RGBA):")
    print(f"  mean   {mean:7.3f} ms")
    print(f"  median {q[49]:7.3f} ms")
    print(f"  p95    {q[94]:7.3f} ms")
    print(f"  p99    {q[98]:7.3f} ms")
    print(f"  max    {max(timings_ms):7.3f} ms")
    print(f"  → sustainable rate ≈ {1000.0 / mean:,.0f} fps (producer side alone)")
    print(f"outcomes: {outcomes}")
    if not is_real:
        print("(--force-fake: numbers exercise the code path only, not real GPU copies)")

    rc = 0
    if consumer is not None and rq is not None:
        consumer.join(timeout=30.0)
        if consumer.is_alive():
            consumer.terminate()
        while not rq.empty():
            status, payload = rq.get_nowait()
            if status == "OK":
                print(f"demo consumer received {payload} frames")
            else:
                print("demo consumer ERROR:", payload)
                rc = 1
    stats = exporter.get_stats()
    print(f"exporter stats: frame_count={stats['frame_count']}  avg_total_us={stats['avg_total_us']:.1f}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
