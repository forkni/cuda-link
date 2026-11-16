"""
bench_r1_wait.py — R1 wait-path before/after benchmark.

Measures per-frame get_frame() CPU time for the torch backend (zero-copy path):

  cpu-spin   Existing default: CPU busy-polls cudaEventQuery then sleeps.
             Always available (no R1 required).

  gpu-wait   R1 opt-in: cudaStreamWaitEvent — CPU returns immediately, GPU
             serialises in stream order.  Auto-detected: runs only when
             ImportPolicy has a 'torch_gpu_wait' field (i.e. R1 is implemented).

  native     R5 opt-in: native (C++) notification-wait backend. Auto-detected:
             runs only when ImportPolicy has a 'wait_backend' field (i.e. R5 is
             implemented). NOTE: the printed p50/p95 for this arm are total
             get_frame() wall-clock time (spin + tensor materialization +
             Python overhead), NOT the publish->detect notification latency
             that PLAN-002's accept gate (p50<10us, p95<50us) is defined on.
             This script has no cross-process publish timestamp to compare
             against, so it cannot itself evaluate the gate -- see
             scripts/profiling/bench_doorbell.py's native arm for that.

  cupy       Reference: CuPy already defaults to GPU-side wait.
             Runs if cupy is importable.

The script spawns a producer process at 30/60 fps (the project's real operating
points) and runs each consumer arm sequentially.  Windows CUDA IPC requires separate processes (error 201 in-process).

Usage
-----
Baseline (development branch, before R1):
    python scripts/profiling/bench_r1_wait.py

After R1 (feature branch):
    python scripts/profiling/bench_r1_wait.py
    (gpu-wait arm activates automatically)

Full options:
    python scripts/profiling/bench_r1_wait.py --frames 400 --spin-us 200
    python scripts/profiling/bench_r1_wait.py --outfile .profiling/r1_before.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import sys
import time
import uuid
from pathlib import Path
from statistics import median, quantiles

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMUP_FRAMES = 40
# 512x512 RGBA float32 — 4 MB; small enough to export quickly, large enough
# that the GPU event fires with non-trivial timing variance.
HEIGHT, WIDTH, CHANNELS, DTYPE = 512, 512, 4, "float32"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: int) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    return quantiles(values, n=100)[p - 1]


def _wait_for_shm(shm_name: str, timeout_s: float = 20.0) -> bool:
    from multiprocessing.shared_memory import SharedMemory

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            shm = SharedMemory(name=shm_name)
            shm.close()
            return True
        except FileNotFoundError:
            time.sleep(0.05)
    return False


def _policy_has_gpu_wait() -> bool:
    """Return True if ImportPolicy has a 'torch_gpu_wait' field (R1 implemented)."""
    try:
        from cuda_link._importer_port import ImportPolicy

        return any(f.name == "torch_gpu_wait" for f in dataclasses.fields(ImportPolicy))
    except Exception:  # noqa: BLE001
        return False


def _policy_has_wait_backend() -> bool:
    """Return True if ImportPolicy has a 'wait_backend' field (R5 implemented)."""
    try:
        from cuda_link._importer_port import ImportPolicy

        return any(f.name == "wait_backend" for f in dataclasses.fields(ImportPolicy))
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Worker functions (module-level so they can be pickled by spawn)
# ---------------------------------------------------------------------------


def _worker_producer(
    shm_name: str,
    num_frames: int,
    producer_fps: float,
    result_q: object,
) -> None:
    """Export synthetic GPU frames at a fixed cadence.

    doorbell=True unconditionally: harmless for arms that don't use it
    (SetEvent is ~sub-microsecond per publish, CHANGELOG 1.11.0), and required
    for the torch/native (R5) arm to reach its real target -- the native wait
    backend's block phase cannot beat the Python fallback without a doorbell
    to actually block on (see ImportPolicy.wait_backend docstring / D4).
    """
    import ctypes

    try:
        from cuda_link import FrameSpec, GpuFrame
        from cuda_link._exporter_port import ExportPolicy
        from cuda_link.cuda_ipc_wrapper import get_cuda_runtime
        from cuda_link.exporter import Exporter

        cuda = get_cuda_runtime()
        exporter = Exporter.open(
            FrameSpec(
                shm_name=shm_name,
                height=HEIGHT,
                width=WIDTH,
                channels=CHANNELS,
                dtype=DTYPE,
                num_slots=2,
            ),
            policy=ExportPolicy(doorbell=True),
        )

        nbytes = exporter.data_size
        src_ptr = cuda.malloc(nbytes)
        if src_ptr is None or src_ptr.value == 0:
            result_q.put(("ERROR", "cudaMalloc failed in producer"))
            return

        host_buf = (ctypes.c_uint8 * nbytes)(*([0] * nbytes))
        cuda.memcpy(
            dst=src_ptr,
            src=ctypes.c_void_p(ctypes.addressof(host_buf)),
            count=nbytes,
            kind=1,
        )
        exporter.record_source_sync(0)

        producer_sleep_s = 1.0 / producer_fps
        # Each arm opens a FRESH Importer connection and independently needs
        # its own (WARMUP_FRAMES + num_frames) budget from this shared
        # producer stream -- arms do not share/reuse already-consumed frames.
        # N_ARMS = cpu-spin + native(R5) + gpu-wait(R1) + adaptive(R1-auto) +
        # cupy = 5. Diagnosed 2026-07-04: the previous formula added
        # WARMUP_FRAMES only ONCE for all arms combined (already marginal at
        # 4 arms; a 5th arm made it fail outright -- the native (R5) arm
        # measured 0 samples over its full 60s deadline because the shared
        # producer had exhausted its under-sized budget and gone silent).
        # OVERHEAD_FRAMES accounts for the roughly-constant per-arm cost of
        # process spawn + CUDA context init + SHM discovery BEFORE an arm's
        # measurement loop even starts -- this is wall-clock overhead, not
        # frame-count-proportional, so a frame-count-only budget silently runs
        # out on real hardware even when the math "should" have enough frames
        # left. Diagnosed again 2026-07-04: 100 frames/arm (~1.67s at 60fps)
        # was still insufficient -- 4/5 arms succeeded but the 5th arm (cupy,
        # last in sequence) still hit "Too few samples: 0" because the
        # producer's total runtime (~1000 frames = 16.7s + 2s teardown sleep)
        # was exhausted before cupy's turn; measured real per-arm overhead
        # (subprocess spawn + torch/cupy import + CUDA context init + SHM
        # discovery + the 0.4s IPC-handle settle sleep) runs closer to 4-5s,
        # not the originally-assumed ~1-2s. Bumped to 400 frames/arm (~6.7s)
        # for real headroom; re-verify against wall-clock arm duration if a
        # 6th arm is ever added.
        n_arms = 5
        overhead_frames = 400
        total = (WARMUP_FRAMES + num_frames + overhead_frames) * n_arms + 50
        for _ in range(total):
            exporter.export(GpuFrame(ptr=int(src_ptr.value), size=nbytes))
            time.sleep(producer_sleep_s)

        time.sleep(2.0)
        cuda.free(src_ptr)
        exporter.close()
        result_q.put(("OK", None))
    except Exception as e:  # noqa: BLE001
        import traceback

        result_q.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


def _worker_consumer_torch(
    shm_name: str,
    num_frames: int,
    spin_us: int,
    gpu_wait: bool,
    producer_fps: float,
    wait_backend: str,
    result_q: object,
) -> None:
    """Torch get_frame() consumer — CPU spin, GPU-side wait, or R5 native wait.

    wait_backend="native" targets the same torch-CPU-wait scenario as the
    cpu-spin baseline (gpu_wait=False) — R5 is explicitly scoped to
    numpy/cupy/torch-CPU, not the torch GPU-wait path, which already
    short-circuits before ever reaching _wait_for_slot.
    """
    try:
        from cuda_link._importer_port import ImportOutcome, ImportPolicy, ImportSpec
        from cuda_link.importer import Importer

        if not _wait_for_shm(shm_name, timeout_s=30.0):
            result_q.put(("ERROR", f"SharedMemory '{shm_name}' never appeared"))
            return

        time.sleep(0.4)  # let producer write IPC handles

        policy_kwargs: dict = {
            "wait_spin_us": spin_us,
            "debug": True,
        }
        if gpu_wait and _policy_has_gpu_wait():
            policy_kwargs["torch_gpu_wait"] = True
        if _policy_has_wait_backend():
            # Always set explicitly (never omitted): ImportPolicy()'s own
            # default is wait_backend="auto", not "python" -- on a machine
            # with the native extension built, omitting this would let every
            # non-R5 arm silently pick up the native backend via "auto",
            # contaminating the baseline/R1/cupy comparisons.
            policy_kwargs["wait_backend"] = wait_backend

        policy = ImportPolicy(**policy_kwargs)
        spec = ImportSpec(
            shm_name=shm_name,
            shape=(HEIGHT, WIDTH, CHANNELS),
            dtype=DTYPE,
        )
        imp = Importer.open(spec, policy=policy)

        gf_samples: list[float] = []
        frames = 0
        deadline = time.perf_counter() + 60.0

        while frames < num_frames + WARMUP_FRAMES and time.perf_counter() < deadline:
            t0 = time.perf_counter()
            result = imp.get_frame()
            gf_us = (time.perf_counter() - t0) * 1e6

            if result.outcome is ImportOutcome.NEW_FRAME:
                if frames >= WARMUP_FRAMES:
                    gf_samples.append(gf_us)
                frames += 1
            elif result.outcome is ImportOutcome.NO_FRAME:
                time.sleep(0.001)
            elif result.outcome in (ImportOutcome.SHUTDOWN, ImportOutcome.TIMEOUT):
                break
            else:
                time.sleep(0.001)

        stats = imp.get_stats()
        fc = imp.frame_count
        # Direct attribute access for fields not exposed by get_stats()
        avg_wait_us = imp.total_wait_event_time / fc if fc > 0 else 0.0
        avg_gf_us = imp.total_get_frame_time / fc if fc > 0 else 0.0

        imp.close()

        if len(gf_samples) < 10:
            result_q.put(("ERROR", f"Too few samples: {len(gf_samples)}"))
            return

        result_q.put(
            (
                "OK",
                {
                    "n": len(gf_samples),
                    "p50_us": median(gf_samples),
                    "p95_us": _percentile(gf_samples, 95),
                    "p99_us": _percentile(gf_samples, 99),
                    "min_us": min(gf_samples),
                    "max_us": max(gf_samples),
                    # Importer telemetry (debug=True)
                    "avg_wait_us": avg_wait_us,
                    "avg_gf_us": avg_gf_us,
                    "wait_spin_hits": stats["wait_spin_hits"],
                    "wait_sleep_hits": stats["wait_sleep_hits"],
                    "avg_spin_us": stats["avg_spin_us"],
                    "avg_sleep_us": stats["avg_sleep_us"],
                },
            )
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        result_q.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


def _worker_consumer_torch_adaptive(
    shm_name: str,
    num_frames: int,
    spin_us: int,
    producer_fps: float,
    result_q: object,
) -> None:
    """Torch consumer with adaptive gpu-wait enabled — auto-latches once blocking is detected."""
    try:
        from cuda_link._importer_port import ImportOutcome, ImportPolicy, ImportSpec
        from cuda_link.importer import Importer

        if not _policy_has_gpu_wait():
            result_q.put(("SKIP", "torch_gpu_wait not available (pre-R1 branch)"))
            return

        if not _wait_for_shm(shm_name, timeout_s=30.0):
            result_q.put(("ERROR", f"SharedMemory '{shm_name}' never appeared"))
            return

        time.sleep(0.4)

        policy_kwargs: dict = {
            "wait_spin_us": spin_us,
            "torch_gpu_wait_adaptive": True,
            "debug": True,
        }
        if _policy_has_wait_backend():
            # Pin explicitly -- ImportPolicy()'s default is "auto", which would
            # silently pick up the native extension if built (see the note in
            # _worker_consumer_torch above).
            policy_kwargs["wait_backend"] = "python"
        policy = ImportPolicy(**policy_kwargs)
        spec = ImportSpec(shm_name=shm_name, shape=(HEIGHT, WIDTH, CHANNELS), dtype=DTYPE)
        imp = Importer.open(spec, policy=policy)

        gf_samples: list[float] = []
        frames = 0
        deadline = time.perf_counter() + 60.0

        while frames < num_frames + WARMUP_FRAMES and time.perf_counter() < deadline:
            t0 = time.perf_counter()
            result = imp.get_frame()
            gf_us = (time.perf_counter() - t0) * 1e6

            if result.outcome is ImportOutcome.NEW_FRAME:
                if frames >= WARMUP_FRAMES:
                    gf_samples.append(gf_us)
                frames += 1
            elif result.outcome is ImportOutcome.NO_FRAME:
                time.sleep(0.001)
            elif result.outcome in (ImportOutcome.SHUTDOWN, ImportOutcome.TIMEOUT):
                break
            else:
                time.sleep(0.001)

        stats = imp.get_stats()
        fc = imp.frame_count
        avg_wait_us = imp.total_wait_event_time / fc if fc > 0 else 0.0
        avg_gf_us = imp.total_get_frame_time / fc if fc > 0 else 0.0
        latched_frame = stats.get("gpu_wait_latched_frame")

        imp.close()

        if len(gf_samples) < 10:
            result_q.put(("ERROR", f"Too few samples: {len(gf_samples)}"))
            return

        result_q.put(
            (
                "OK",
                {
                    "n": len(gf_samples),
                    "p50_us": median(gf_samples),
                    "p95_us": _percentile(gf_samples, 95),
                    "p99_us": _percentile(gf_samples, 99),
                    "min_us": min(gf_samples),
                    "max_us": max(gf_samples),
                    "avg_wait_us": avg_wait_us,
                    "avg_gf_us": avg_gf_us,
                    "wait_spin_hits": stats["wait_spin_hits"],
                    "wait_sleep_hits": stats["wait_sleep_hits"],
                    "avg_spin_us": stats["avg_spin_us"],
                    "avg_sleep_us": stats["avg_sleep_us"],
                    "gpu_wait_active": stats.get("gpu_wait_active", False),
                    "gpu_wait_latched_frame": latched_frame,
                },
            )
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        result_q.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


def _worker_consumer_cupy(
    shm_name: str,
    num_frames: int,
    producer_fps: float,
    result_q: object,
) -> None:
    """CuPy get_frame_cupy() consumer — GPU-side wait, existing default."""
    try:
        from cuda_link._importer_port import ImportOutcome, ImportPolicy, ImportSpec
        from cuda_link.importer import Importer

        try:
            import cupy  # noqa: F401
        except ImportError:
            result_q.put(("SKIP", "cupy not installed"))
            return

        if not _wait_for_shm(shm_name, timeout_s=30.0):
            result_q.put(("ERROR", f"SharedMemory '{shm_name}' never appeared"))
            return

        time.sleep(0.4)

        policy_kwargs: dict = {"debug": True}
        if _policy_has_wait_backend():
            # Pin explicitly -- CuPy's backend also routes through
            # _wait_for_slot (R5's scope includes cupy), so "auto" would
            # silently engage the native backend here too if installed.
            policy_kwargs["wait_backend"] = "python"
        policy = ImportPolicy(**policy_kwargs)
        spec = ImportSpec(
            shm_name=shm_name,
            shape=(HEIGHT, WIDTH, CHANNELS),
            dtype=DTYPE,
        )
        imp = Importer.open(spec, policy=policy)

        gf_samples: list[float] = []
        frames = 0
        deadline = time.perf_counter() + 60.0

        while frames < num_frames + WARMUP_FRAMES and time.perf_counter() < deadline:
            t0 = time.perf_counter()
            result = imp.get_frame_cupy()
            gf_us = (time.perf_counter() - t0) * 1e6

            if result.outcome is ImportOutcome.NEW_FRAME:
                if frames >= WARMUP_FRAMES:
                    gf_samples.append(gf_us)
                frames += 1
            elif result.outcome is ImportOutcome.NO_FRAME:
                time.sleep(0.001)
            elif result.outcome in (ImportOutcome.SHUTDOWN, ImportOutcome.TIMEOUT):
                break
            else:
                time.sleep(0.001)

        fc = imp.frame_count
        avg_wait_us = imp.total_wait_event_time / fc if fc > 0 else 0.0
        avg_gf_us = imp.total_get_frame_time / fc if fc > 0 else 0.0
        stats = imp.get_stats()
        imp.close()

        if len(gf_samples) < 10:
            result_q.put(("ERROR", f"Too few samples: {len(gf_samples)}"))
            return

        result_q.put(
            (
                "OK",
                {
                    "n": len(gf_samples),
                    "p50_us": median(gf_samples),
                    "p95_us": _percentile(gf_samples, 95),
                    "p99_us": _percentile(gf_samples, 99),
                    "min_us": min(gf_samples),
                    "max_us": max(gf_samples),
                    "avg_wait_us": avg_wait_us,
                    "avg_gf_us": avg_gf_us,
                    "wait_spin_hits": stats["wait_spin_hits"],
                    "wait_sleep_hits": stats["wait_sleep_hits"],
                    "avg_spin_us": stats["avg_spin_us"],
                    "avg_sleep_us": stats["avg_sleep_us"],
                },
            )
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        result_q.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


# ---------------------------------------------------------------------------
# Run one consumer arm against a running producer
# ---------------------------------------------------------------------------


def _run_arm(
    label: str,
    target: object,
    args: tuple,
    ctx: object,
    producer_fps: float = 120.0,
) -> dict | None:
    q = ctx.Queue()
    proc = ctx.Process(target=target, args=(*args, q), daemon=True)
    proc.start()
    try:
        status, payload = q.get(timeout=90)
    except Exception as e:  # noqa: BLE001
        proc.terminate()
        print(f"  [{label}] TIMEOUT or queue error: {e}")
        return None
    proc.join(timeout=5)

    if status == "SKIP":
        print(f"  [{label}] SKIPPED: {payload}")
        return None
    if status == "ERROR":
        print(f"  [{label}] ERROR: {payload}")
        return None

    r = payload
    print(
        f"  [{label}]"
        f"  get_frame p50={r['p50_us']:6.1f} µs"
        f"  p95={r['p95_us']:6.1f} µs"
        f"  p99={r['p99_us']:6.1f} µs"
        f"  avg_wait={r['avg_wait_us']:6.1f} µs"
        f"  spin_hits={r['wait_spin_hits']}"
        f"  sleep_hits={r['wait_sleep_hits']}"
        f"  n={r['n']}"
    )
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_scenario(
    label: str,
    producer_fps: float,
    frames: int,
    spin_us: int,
    has_gpu_wait: bool,
    has_wait_backend: bool,
    ctx: object,
) -> dict[str, dict]:
    """Start a producer at producer_fps and run all consumer arms against it."""
    shm_name = f"__bench_r1_{uuid.uuid4().hex[:8]}"

    prod_q = ctx.Queue()
    prod = ctx.Process(
        target=_worker_producer,
        args=(shm_name, frames, producer_fps, prod_q),
        daemon=True,
    )
    prod.start()

    if not _wait_for_shm(shm_name, timeout_s=20.0):
        print(f"  ERROR: producer SHM never appeared for scenario '{label}'")
        prod.terminate()
        return {}

    print(f"\nScenario: {label} (producer={producer_fps:.0f} fps)\n")

    results: dict[str, dict] = {}

    r = _run_arm(
        label="torch/cpu-spin",
        target=_worker_consumer_torch,
        args=(shm_name, frames, spin_us, False, producer_fps, "python"),
        ctx=ctx,
        producer_fps=producer_fps,
    )
    if r:
        results["torch_cpu_spin"] = r

    if has_wait_backend:
        r = _run_arm(
            label="torch/native (R5)",
            target=_worker_consumer_torch,
            args=(shm_name, frames, spin_us, False, producer_fps, "native"),
            ctx=ctx,
            producer_fps=producer_fps,
        )
        if r:
            results["torch_native"] = r
    else:
        print("  [torch/native (R5)] SKIPPED -- ImportPolicy.wait_backend not present yet")

    if has_gpu_wait:
        r = _run_arm(
            label="torch/gpu-wait (R1)",
            target=_worker_consumer_torch,
            args=(shm_name, frames, spin_us, True, producer_fps, "python"),
            ctx=ctx,
            producer_fps=producer_fps,
        )
        if r:
            results["torch_gpu_wait"] = r

        r = _run_arm(
            label="torch/adaptive (R1-auto)",
            target=_worker_consumer_torch_adaptive,
            args=(shm_name, frames, spin_us, producer_fps),
            ctx=ctx,
            producer_fps=producer_fps,
        )
        if r:
            results["torch_adaptive"] = r
            if r.get("gpu_wait_active"):
                print(f"    (latch engaged at frame {r['gpu_wait_latched_frame']})")
            else:
                print("    (latch never engaged -- producer too fast)")
    else:
        print("  [torch/gpu-wait (R1)] SKIPPED -- ImportPolicy.torch_gpu_wait not present yet")
        print("  [torch/adaptive (R1-auto)] SKIPPED -- R1 not present yet")

    r = _run_arm(
        label="cupy/gpu-wait",
        target=_worker_consumer_cupy,
        args=(shm_name, frames, producer_fps),
        ctx=ctx,
        producer_fps=producer_fps,
    )
    if r:
        results["cupy_gpu_wait"] = r

    prod.join(timeout=15)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="R1 wait-path benchmark -- torch GPU-wait vs CPU-spin")
    parser.add_argument(
        "--frames", type=int, default=300, help="Measurement frames per arm (after warmup, default 300)"
    )
    parser.add_argument("--spin-us", type=int, default=200, dest="spin_us", help="CPU spin budget us (default 200)")
    parser.add_argument(
        "--producer-fps",
        type=float,
        default=0,
        dest="producer_fps",
        help="Producer FPS (default: run both 120 fps and 30 fps scenarios)",
    )
    parser.add_argument(
        "--outfile",
        default=".profiling/r1_wait.json",
        help="Output JSON path (default: .profiling/r1_wait.json)",
    )
    args = parser.parse_args()

    has_gpu_wait = _policy_has_gpu_wait()
    has_wait_backend = _policy_has_wait_backend()

    print("=" * 72)
    print("R1 wait-path benchmark")
    print(f"  Resolution : {HEIGHT}x{WIDTH}x{CHANNELS} {DTYPE}")
    print(f"  Frames/arm : {args.frames}  Warmup: {WARMUP_FRAMES}")
    print(f"  Spin budget: {args.spin_us} us")
    print(f"  R1 (torch_gpu_wait): {'PRESENT -- gpu-wait arm enabled' if has_gpu_wait else 'ABSENT -- baseline only'}")
    print(
        f"  R5 (wait_backend)  : {'PRESENT -- native arm enabled' if has_wait_backend else 'ABSENT -- baseline only'}"
    )
    print("=" * 72)

    ctx = multiprocessing.get_context("spawn")

    scenarios: list[tuple[str, float]] = (
        [("fast-producer", args.producer_fps)]
        if args.producer_fps > 0
        else [
            ("60fps (vsync; consumer waits at event boundary)", 60.0),
            ("30fps (consumer waits longer; R1 most impactful)", 30.0),
        ]
    )

    all_results: dict[str, dict] = {}
    for scenario_label, fps in scenarios:
        all_results[scenario_label] = _run_scenario(
            label=scenario_label,
            producer_fps=fps,
            frames=args.frames,
            spin_us=args.spin_us,
            has_gpu_wait=has_gpu_wait,
            has_wait_backend=has_wait_backend,
            ctx=ctx,
        )

    # --- Summary ---
    labels = {
        "torch_cpu_spin": "torch / cpu-spin (baseline)",
        "torch_native": "torch / native (R5 opt-in)",
        "torch_gpu_wait": "torch / gpu-wait (R1 opt-in)",
        "cupy_gpu_wait": "cupy  / gpu-wait (reference)",
    }
    for scenario_label, results in all_results.items():
        print()
        print("=" * 72)
        print(f"SUMMARY: {scenario_label}")
        print("=" * 72)
        header = f"  {'Arm':<30} {'p50 us':>8} {'p95 us':>8} {'p99 us':>8} {'avg_wait us':>12} {'spin_hits':>10} {'sleep_hits':>11}"
        print(header)
        print("  " + "-" * 90)
        for key, arm_label in labels.items():
            if key in results:
                r = results[key]
                print(
                    f"  {arm_label:<30}"
                    f" {r['p50_us']:>8.1f}"
                    f" {r['p95_us']:>8.1f}"
                    f" {r['p99_us']:>8.1f}"
                    f" {r['avg_wait_us']:>12.1f}"
                    f" {r['wait_spin_hits']:>10}"
                    f" {r['wait_sleep_hits']:>11}"
                )
        print()
        if "torch_cpu_spin" in results and "torch_gpu_wait" in results:
            baseline_wait = results["torch_cpu_spin"]["avg_wait_us"]
            improved_wait = results["torch_gpu_wait"]["avg_wait_us"]
            delta_wait = baseline_wait - improved_wait
            baseline_p50 = results["torch_cpu_spin"]["p50_us"]
            improved_p50 = results["torch_gpu_wait"]["p50_us"]
            delta_p50 = baseline_p50 - improved_p50
            print(
                f"  R1 wait component : {baseline_wait:.1f} us -> {improved_wait:.1f} us  (delta {delta_wait:+.1f} us)"
            )
            print(f"  R1 get_frame p50  : {baseline_p50:.1f} us -> {improved_p50:.1f} us  (delta {delta_p50:+.1f} us)")
            spin_h = results["torch_cpu_spin"]["wait_spin_hits"]
            sleep_h = results["torch_cpu_spin"]["wait_sleep_hits"]
            print(
                f"  CPU-spin profile  : spin_hits={spin_h}  sleep_hits={sleep_h}  (sleep hits > 0 = real blocking saved by R1)"
            )
        elif "torch_gpu_wait" not in results:
            print("  [R1 comparison not available -- run again after implementing R1]")

        if "torch_cpu_spin" in results and "torch_native" in results:
            baseline_wait = results["torch_cpu_spin"]["avg_wait_us"]
            native_wait = results["torch_native"]["avg_wait_us"]
            delta_wait = baseline_wait - native_wait
            baseline_p50 = results["torch_cpu_spin"]["p50_us"]
            native_p50 = results["torch_native"]["p50_us"]
            delta_p50 = baseline_p50 - native_p50
            print(f"  R5 wait component : {baseline_wait:.1f} us -> {native_wait:.1f} us  (delta {delta_wait:+.1f} us)")
            print(f"  R5 get_frame p50  : {baseline_p50:.1f} us -> {native_p50:.1f} us  (delta {delta_p50:+.1f} us)")
            native_spin_hits = results["torch_native"]["wait_spin_hits"]
            native_sleep_hits = results["torch_native"]["wait_sleep_hits"]
            native_avg_spin = results["torch_native"]["avg_spin_us"]
            native_avg_sleep_ms = results["torch_native"]["avg_sleep_us"] / 1000.0
            print(
                f"  R5 detection      : avg_spin={native_avg_spin:.3f}us over {native_spin_hits} frame-pending"
                f" wait(s); {native_sleep_hits} pacing block(s) avg {native_avg_sleep_ms:.1f}ms (consumer"
                " arrived before next frame -- excluded from gate)"
            )
            print(
                "  R5 accept gate    : p50<10us p95<50us is publish->detect latency -- run"
                " scripts/profiling/bench_doorbell.py (native arm) for the authoritative measurement"
            )
        elif "torch_native" not in results and has_wait_backend:
            print("  [R5 native arm ran but produced no comparable result this scenario]")
        elif not has_wait_backend:
            print("  [R5 comparison not available -- run again on a host with the native extension built]")

    print()
    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "2",
        "config": {
            "height": HEIGHT,
            "width": WIDTH,
            "channels": CHANNELS,
            "dtype": DTYPE,
            "frames": args.frames,
            "warmup_frames": WARMUP_FRAMES,
            "spin_us": args.spin_us,
            "r1_present": has_gpu_wait,
            "r5_present": has_wait_backend,
        },
        "results": all_results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
