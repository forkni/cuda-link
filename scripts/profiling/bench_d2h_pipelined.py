"""
bench_d2h_pipelined.py — P5 pipelined D2H double-buffer benchmark.

Measures the wall-clock gain from CUDALINK_D2H_PIPELINED=1 vs 0 by running a
standalone producer+consumer pair where the consumer does a synthetic CPU workload
(--work-ms busy-spin) after each frame.

Hypothesis:
  non-pipelined per-frame ≈ copy + work
  pipelined    per-frame ≈ max(copy, work)

At 1080p (~1.3 ms D2H copy) with --work-ms 5, pipelined wall-time should drop
toward 5 ms/frame from ~6.3 ms.

Windows: CUDA IPC requires separate processes (cudaIpcOpenMemHandle returns error 201
in the same process that created the handle). Workers are module-level functions so
they can be pickled by the "spawn" multiprocessing context.

Usage:
    # Non-pipelined (baseline):
    python scripts/profiling/bench_d2h_pipelined.py --pipelined 0 --work-ms 5

    # Pipelined:
    python scripts/profiling/bench_d2h_pipelined.py --pipelined 1 --work-ms 5

    # Sweep resolutions:
    python scripts/profiling/bench_d2h_pipelined.py --resolution all --work-ms 5

    # nsys capture (run this script, not via run_nsys.ps1):
    SET CUDALINK_NVTX=1
    SET CUDALINK_NVTX_VERBOSE=1
    nsys profile --trace=cuda,nvtx,wddm --output .profiling/d2h_pipelined -- python ...
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
import uuid
from pathlib import Path
from statistics import median, quantiles

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

WARMUP_FRAMES = 30
# Producer cadence: export at ~120 FPS so the consumer is never starved.
PRODUCER_SLEEP_S = 1.0 / 120.0

RESOLUTIONS = {
    "512": (512, 512, 4),
    "1080p": (1080, 1920, 4),
    "4k": (2160, 3840, 4),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _busy_spin_ms(ms: float) -> None:
    """Burn CPU for ~ms milliseconds (simulates consumer workload)."""
    deadline = time.perf_counter() + ms * 1e-3
    while time.perf_counter() < deadline:
        pass


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


# ---------------------------------------------------------------------------
# Worker functions (module-level — required for pickle / spawn)
# ---------------------------------------------------------------------------


def _worker_producer(
    shm_name: str,
    height: int,
    width: int,
    channels: int,
    dtype: str,
    num_frames: int,
    result_q: object,
) -> None:
    """Export synthetic GPU frames at a fixed cadence."""
    try:
        from cuda_link import FrameSpec, GpuFrame
        from cuda_link.cuda_ipc_wrapper import get_cuda_runtime
        from cuda_link.exporter import Exporter

        cuda = get_cuda_runtime()
        exporter = Exporter.open(
            FrameSpec(
                shm_name=shm_name,
                height=height,
                width=width,
                channels=channels,
                dtype=dtype,
                num_slots=2,
            )
        )

        nbytes = exporter.data_size
        src_ptr = cuda.malloc(nbytes)
        if src_ptr is None or src_ptr.value == 0:
            result_q.put(("ERROR", "cudaMalloc failed in producer"))
            return

        # Fill with a known pattern
        host_buf = (ctypes.c_uint8 * nbytes)(*([42] * nbytes))
        cuda.memcpy(
            dst=src_ptr,
            src=ctypes.c_void_p(ctypes.addressof(host_buf)),
            count=nbytes,
            kind=1,  # H2D
        )

        exporter.record_source_sync(0)

        for _ in range(num_frames + WARMUP_FRAMES + 4):  # +4 for P5 priming headroom
            exporter.export(GpuFrame(ptr=int(src_ptr.value), size=nbytes))
            time.sleep(PRODUCER_SLEEP_S)

        time.sleep(2.0)  # grace period for consumer to drain
        cuda.free(src_ptr)
        exporter.close()
        result_q.put(("OK", num_frames))
    except Exception as e:  # noqa: BLE001
        import traceback

        result_q.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


def _worker_consumer(
    shm_name: str,
    height: int,
    width: int,
    channels: int,
    dtype: str,
    num_frames: int,
    work_ms: float,
    pipelined: bool,
    result_q: object,
) -> None:
    """Consume frames via the modern Importer.open() API and record per-frame wall-time."""
    try:
        from cuda_link._importer_port import ImportOutcome, ImportPolicy, ImportSpec
        from cuda_link.importer import Importer

        if not _wait_for_shm(shm_name, timeout_s=20.0):
            result_q.put(("ERROR", f"SharedMemory '{shm_name}' never appeared"))
            return

        time.sleep(0.3)  # let producer write IPC handles

        policy = ImportPolicy(
            d2h_pipelined=pipelined,
            debug=True,  # enable internal timing; harmless
        )
        spec = ImportSpec(
            shm_name=shm_name,
            shape=(height, width, channels),
            dtype=dtype,
        )
        importer = Importer.open(spec, policy=policy)

        d2h_samples: list[float] = []  # get_frame_numpy() call alone
        cycle_samples: list[float] = []  # get_frame_numpy() + workload (full cycle)
        priming_observed = False
        frames_received = 0
        deadline = time.perf_counter() + 30.0

        while frames_received < num_frames + WARMUP_FRAMES and time.perf_counter() < deadline:
            cycle_t0 = time.perf_counter()
            result = importer.get_frame_numpy()
            d2h_us = (time.perf_counter() - cycle_t0) * 1e6

            if result.outcome is ImportOutcome.NEW_FRAME:
                if frames_received >= WARMUP_FRAMES:
                    d2h_samples.append(d2h_us)
                # Simulate consumer workload AFTER receiving the frame
                _busy_spin_ms(work_ms)
                cycle_us = (time.perf_counter() - cycle_t0) * 1e6
                if frames_received >= WARMUP_FRAMES:
                    cycle_samples.append(cycle_us)
                frames_received += 1
            elif result.outcome is ImportOutcome.NO_FRAME:
                if pipelined and not priming_observed:
                    priming_observed = True  # first call is the priming NO_FRAME
                time.sleep(0.005)
            elif result.outcome in (ImportOutcome.SHUTDOWN, ImportOutcome.TIMEOUT):
                break
            else:
                time.sleep(0.005)

        importer.close()

        if len(cycle_samples) < 10:
            result_q.put(("ERROR", f"Too few samples: {len(cycle_samples)} (got {frames_received} frames total)"))
            return

        result_q.put(
            (
                "OK",
                {
                    "n": len(cycle_samples),
                    "priming_observed": priming_observed,
                    # D2H call time (get_frame_numpy only)
                    "d2h_median_us": median(d2h_samples),
                    "d2h_p95_us": _percentile(d2h_samples, 95),
                    "d2h_p99_us": _percentile(d2h_samples, 99),
                    # Full cycle = D2H + workload (the P5 gain is visible here)
                    "median_us": median(cycle_samples),
                    "p95_us": _percentile(cycle_samples, 95),
                    "p99_us": _percentile(cycle_samples, 99),
                    "min_us": min(cycle_samples),
                    "max_us": max(cycle_samples),
                },
            )
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        result_q.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


# ---------------------------------------------------------------------------
# Single benchmark cell
# ---------------------------------------------------------------------------


def _run_cell(
    height: int,
    width: int,
    channels: int,
    dtype: str,
    num_frames: int,
    work_ms: float,
    pipelined: bool,
    label: str,
) -> dict:
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    shm_name = f"__bench_d2h_{uuid.uuid4().hex[:8]}"

    prod_q: multiprocessing.Queue = ctx.Queue()
    cons_q: multiprocessing.Queue = ctx.Queue()

    prod = ctx.Process(
        target=_worker_producer,
        args=(shm_name, height, width, channels, dtype, num_frames, prod_q),
        daemon=True,
    )
    cons = ctx.Process(
        target=_worker_consumer,
        args=(shm_name, height, width, channels, dtype, num_frames, work_ms, pipelined, cons_q),
        daemon=True,
    )

    print(f"  [{label}] starting …", flush=True)
    t_start = time.perf_counter()
    prod.start()
    cons.start()

    # Collect consumer result (30s timeout per cell)
    cons_result = cons_q.get(timeout=60)
    cons.join(timeout=5)
    prod.join(timeout=10)
    elapsed_s = time.perf_counter() - t_start

    if cons_result[0] == "ERROR":
        raise RuntimeError(f"Consumer failed: {cons_result[1]}")
    if cons_result[0] == "SKIP":
        return {"skipped": True, "reason": cons_result[1]}

    stats = cons_result[1]
    print(
        f"  [{label}] d2h={stats['d2h_median_us']:.0f}µs  "
        f"cycle={stats['median_us']:.0f}µs (p99={stats['p99_us']:.0f}µs)  "
        f"n={stats['n']}  pipelined={pipelined}  "
        f"({elapsed_s:.1f}s total)",
        flush=True,
    )
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="P5 pipelined D2H benchmark — producer+consumer spawn.")
    parser.add_argument(
        "--resolution",
        choices=list(RESOLUTIONS) + ["all"],
        default="1080p",
        help="Frame resolution to benchmark (default: 1080p)",
    )
    parser.add_argument(
        "--pipelined",
        type=int,
        choices=[0, 1],
        default=None,
        help="0=non-pipelined, 1=pipelined, absent=both",
    )
    parser.add_argument(
        "--work-ms",
        type=float,
        default=5.0,
        dest="work_ms",
        help="Synthetic CPU workload per frame in ms (default: 5.0)",
    )
    parser.add_argument("--frames", type=int, default=200, help="Measurement frames (after warmup)")
    parser.add_argument("--dtype", default="uint8", choices=["uint8", "float32", "float16"])
    parser.add_argument(
        "--outfile",
        default=".profiling/d2h_pipelined.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    resolutions = (
        list(RESOLUTIONS.items()) if args.resolution == "all" else [(args.resolution, RESOLUTIONS[args.resolution])]
    )
    arms = [False, True] if args.pipelined is None else [bool(args.pipelined)]

    print(f"P5 D2H pipelined benchmark — work_ms={args.work_ms}  frames={args.frames}  dtype={args.dtype}")
    print(f"  resolutions: {[r for r, _ in resolutions]}")
    print(f"  arms: {['pipelined' if a else 'non-pipelined' for a in arms]}")
    print()

    results = {}
    for res_name, (height, width, channels) in resolutions:
        nbytes = height * width * channels * (1 if args.dtype == "uint8" else 4)
        print(f"Resolution {res_name} ({height}x{width}x{channels} {args.dtype}, {nbytes / 1e6:.1f} MB/frame):")
        results[res_name] = {}
        for pipelined in arms:
            arm_label = f"{'pipelined' if pipelined else 'non-pipelined'}"
            label = f"{res_name}/{arm_label}"
            try:
                stats = _run_cell(
                    height=height,
                    width=width,
                    channels=channels,
                    dtype=args.dtype,
                    num_frames=args.frames,
                    work_ms=args.work_ms,
                    pipelined=pipelined,
                    label=label,
                )
                results[res_name][arm_label] = stats
            except Exception as e:
                print(f"  [{label}] ERROR: {e}", flush=True)
                results[res_name][arm_label] = {"error": str(e)}

        # Print gain summary if both arms ran
        if "pipelined" in results[res_name] and "non-pipelined" in results[res_name]:
            p = results[res_name]["pipelined"]
            np_ = results[res_name]["non-pipelined"]
            if "median_us" in p and "median_us" in np_:
                cycle_gain = np_["median_us"] - p["median_us"]
                pct = 100 * cycle_gain / np_["median_us"] if np_["median_us"] > 0 else 0
                d2h_np = np_.get("d2h_median_us", 0)
                d2h_p = p.get("d2h_median_us", 0)
                priming = p.get("priming_observed", False)
                print(
                    f"  [{res_name}] cycle gain: {cycle_gain:.0f}µs p50 ({pct:.0f}%)  "
                    f"d2h: {d2h_np:.0f}us->{d2h_p:.0f}us  "
                    f"priming_NO_FRAME={'YES' if priming else 'NO (check P5 flag)'}",
                    flush=True,
                )
        print()

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "1",
        "config": {
            "work_ms": args.work_ms,
            "frames": args.frames,
            "warmup_frames": WARMUP_FRAMES,
            "dtype": args.dtype,
        },
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
