"""
bench_sweep.py — Full IPC sweep benchmark for cuda-link.

Measures export_frame() latency (producer/CUDA-Graphs side), get_frame_numpy()
D2H latency (consumer side), and end-to-end roundtrip latency across a matrix
of resolutions × dtypes × CUDA-Graphs on/off.

CUDA IPC requires separate OS processes on Windows (cudaIpcOpenMemHandle returns
error 201 if called in the same process that created the handle). Each cell runs
producer + consumer as separate spawn processes.

Usage (run from repo root):
    python benchmarks\\bench_sweep.py              # full 16-cell sweep
    python benchmarks\\bench_sweep.py --quick      # single cell smoke test
    python benchmarks\\bench_sweep.py --resolutions 512 1080
    python benchmarks\\bench_sweep.py --dtypes f32
    python benchmarks\\bench_sweep.py --graphs 1   # graphs-on only

Output:
    benchmarks\\results\\sweep_YYYY-MM-DD_HHMMSS.{csv,json}
    benchmarks\\results\\sweep_latest.{csv,json}     (overwritten each run)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ─── Constants ────────────────────────────────────────────────────────────────

_RESOLUTION_MAP: dict[str, tuple[int, int]] = {
    "512": (512, 512),
    "720": (1280, 720),
    "1080": (1920, 1080),
    "2160": (3840, 2160),
}
_DTYPE_MAP: dict[str, str] = {"f32": "float32", "u8": "uint8"}
_DTYPE_ITEMSIZE: dict[str, int] = {"float32": 4, "uint8": 1}
_CHANNELS = 4  # RGBA

# ─── System info ─────────────────────────────────────────────────────────────


def _get_system_info() -> dict:
    info: dict = {"gpu": "unknown", "driver": "unknown", "os": platform.version()}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            if len(parts) >= 1:
                info["gpu"] = parts[0]
            if len(parts) >= 2:
                info["driver"] = parts[1]
            if len(parts) >= 3:
                info["compute_cap"] = parts[2]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    try:
        r2 = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r2.returncode == 0:
            parts2 = [p.strip() for p in r2.stdout.strip().split(",")]
            if len(parts2) >= 3:
                info["vram"] = parts2[0]
                info["pcie_gen"] = parts2[1]
                info["pcie_width"] = parts2[2]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return info


# ─── Statistics ───────────────────────────────────────────────────────────────


def _stats(data: list[float]) -> dict:
    if not data:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    s = sorted(data)
    n = len(s)

    def _pct(p: float) -> float:
        idx = (n - 1) * p / 100.0
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    return {
        "mean": sum(s) / n,
        "p50": _pct(50),
        "p95": _pct(95),
        "p99": _pct(99),
        "min": s[0],
        "max": s[-1],
        "n": n,
    }


# ─── SharedMemory wait (module-level so spawn can pickle it) ──────────────────


def _wait_for_shm(shm_name: str, timeout_s: float = 30.0) -> bool:
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


# ─── Worker functions (module-level for spawn pickling) ───────────────────────


def _worker_producer(
    shm_name: str,
    width: int,
    height: int,
    dtype: str,
    n_warmup: int,
    n_frames: int,
    use_graphs: bool,
    fps: float,
    rq: object,
) -> None:
    """Export GPU frames, time each export_frame() call, send results via queue."""
    try:
        import logging

        logging.disable(logging.WARNING)  # suppress per-frame info/debug noise

        # Set before any cuda_link import so the module reads the right value.
        os.environ["CUDALINK_USE_GRAPHS"] = "1" if use_graphs else "0"
        # EXPORT_SYNC=1 means export_frame() CPU-blocks until GPU work is done,
        # making wall-clock timing of export_frame() meaningful.
        os.environ["CUDALINK_EXPORT_SYNC"] = "1"

        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from cuda_link.cuda_ipc_exporter import CUDAIPCExporter
        from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

        data_size = height * width * _CHANNELS * _DTYPE_ITEMSIZE[dtype]
        cuda = CUDARuntimeAPI()
        gpu_buf = cuda.malloc(data_size)

        exp = CUDAIPCExporter(
            shm_name=shm_name,
            height=height,
            width=width,
            channels=_CHANNELS,
            dtype=dtype,
            num_slots=2,
        )
        if not exp.initialize():
            rq.put(("producer_error", "initialize() returned False"))
            cuda.free(gpu_buf)
            return

        frame_interval = 1.0 / fps
        t_next = time.perf_counter() + frame_interval

        for _ in range(n_warmup):
            exp.export_frame(gpu_ptr=int(gpu_buf.value), size=data_size)
            sleep_s = t_next - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            t_next += frame_interval

        export_us: list[float] = []
        for _ in range(n_frames):
            t0 = time.perf_counter()
            exp.export_frame(gpu_ptr=int(gpu_buf.value), size=data_size)
            export_us.append((time.perf_counter() - t0) * 1e6)
            sleep_s = t_next - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            t_next += frame_interval

        time.sleep(3.0)  # Grace period so consumer can drain
        cuda.free(gpu_buf)
        exp.cleanup()
        # Send only the stats dict (tiny), NOT the raw list — large lists overflow the
        # pipe buffer (~65 KB) causing rq.put() to block until the main process reads,
        # but the main process only reads after joining → deadlock.
        rq.put(("producer_ok", _stats(export_us)))
    except Exception as exc:  # noqa: BLE001
        import traceback

        rq.put(("producer_error", f"{exc}\n{traceback.format_exc()}"))


def _worker_consumer(
    shm_name: str,
    width: int,
    height: int,
    dtype: str,
    n_warmup: int,
    n_frames: int,
    fps: float,
    rq: object,
) -> None:
    """Read frames via get_frame_numpy(), time D2H per frame, record E2E from producer timestamp."""
    try:
        import logging

        logging.disable(logging.WARNING)  # suppress per-frame info/debug noise

        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from cuda_link.cuda_ipc_importer import NUMPY_AVAILABLE, CUDAIPCImporter

        if not NUMPY_AVAILABLE:
            rq.put(("consumer_error", "numpy not available"))
            return

        if not _wait_for_shm(shm_name, timeout_s=30.0):
            rq.put(("consumer_error", f"SharedMemory '{shm_name}' never appeared within 30s"))
            return

        time.sleep(0.3)  # Let producer write IPC handles into SHM header

        imp = CUDAIPCImporter(
            shm_name=shm_name,
            shape=(height, width, _CHANNELS),
            dtype=dtype,
        )
        if not imp.is_ready():
            rq.put(("consumer_error", "CUDAIPCImporter.is_ready() returned False"))
            return

        get_us: list[float] = []
        e2e_us: list[float] = []
        frames_seen = 0
        total_frames = n_warmup + n_frames
        deadline = time.perf_counter() + total_frames / fps + 30.0

        while frames_seen < total_frames and time.perf_counter() < deadline:
            if not imp.is_ready():  # producer shutdown detected; importer cleaned itself up
                break
            t0 = time.perf_counter()
            frame = imp.get_frame_numpy()
            dt = (time.perf_counter() - t0) * 1e6  # µs

            if frame is not None:
                if frames_seen >= n_warmup:
                    get_us.append(dt)
                    if imp.last_latency > 0:
                        # last_latency is in ms; convert to µs
                        e2e_us.append(imp.last_latency * 1000.0)
                frames_seen += 1
            else:
                if not imp.is_ready():  # shutdown detected inside get_frame_numpy
                    break
                time.sleep(0.0001)  # 100 µs poll when no new frame ready

        imp.cleanup()
        rq.put(("consumer_ok", {"get_numpy_us": _stats(get_us), "e2e_us": _stats(e2e_us)}))
    except Exception as exc:  # noqa: BLE001
        import traceback

        rq.put(("consumer_error", f"{exc}\n{traceback.format_exc()}"))


# ─── Per-cell runner ──────────────────────────────────────────────────────────


def _run_cell(
    width: int,
    height: int,
    dtype: str,
    use_graphs: bool,
    n_warmup: int,
    n_frames: int,
    fps: float,
    cell_idx: int,
    total_cells: int,
) -> dict | None:
    """Run one sweep cell in separate producer+consumer processes. Returns result dict or None."""
    shm_name = f"_bench_sweep_{os.getpid()}_{width}x{height}_{dtype}_g{int(use_graphs)}"
    graphs_label = "on" if use_graphs else "off"
    print(
        f"  [{cell_idx:>2}/{total_cells}] {width}x{height:>4}  {dtype:<8}  graphs={graphs_label:<3}",
        end="  ",
        flush=True,
    )

    ctx = multiprocessing.get_context("spawn")
    rq = ctx.Queue()

    producer = ctx.Process(
        target=_worker_producer,
        args=(shm_name, width, height, dtype, n_warmup, n_frames, use_graphs, fps, rq),
    )
    consumer = ctx.Process(
        target=_worker_consumer,
        args=(shm_name, width, height, dtype, n_warmup, n_frames, fps, rq),
    )

    cell_timeout = (n_warmup + n_frames) / fps + 60.0
    producer.start()
    consumer.start()
    producer.join(timeout=cell_timeout)
    consumer.join(timeout=cell_timeout)

    if producer.is_alive():
        producer.terminate()
        print("TIMEOUT (producer)")
        return None
    if consumer.is_alive():
        consumer.terminate()
        print("TIMEOUT (consumer)")
        return None

    collected: dict = {}
    while not rq.empty():
        tag, val = rq.get_nowait()
        collected[tag] = val

    if "producer_error" in collected:
        print(f"ERROR producer: {str(collected['producer_error'])[:100]}")
        return None
    if "consumer_error" in collected:
        print(f"ERROR consumer: {str(collected['consumer_error'])[:100]}")
        return None
    if "producer_ok" not in collected or "consumer_ok" not in collected:
        print("ERROR: missing results from workers")
        return None

    export_st = collected["producer_ok"]
    get_st = collected["consumer_ok"]["get_numpy_us"]
    e2e_st = collected["consumer_ok"]["e2e_us"]

    bytes_per_frame = width * height * _CHANNELS * _DTYPE_ITEMSIZE[dtype]
    throughput_gbs = (
        bytes_per_frame / (e2e_st["mean"] / 1e6) / 1e9 if e2e_st["mean"] > 0 and e2e_st["n"] > 0 else float("nan")
    )

    print(
        f"export={export_st['mean']:>6.1f}us  "
        f"d2h={get_st['mean']:>7.1f}us  "
        f"e2e={e2e_st['mean']:>7.1f}us  "
        f"tput={throughput_gbs:.3f} GB/s"
    )

    return {
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "dtype": dtype,
        "graphs": graphs_label,
        "bytes_per_frame": bytes_per_frame,
        "export_us": export_st,
        "get_numpy_us": get_st,
        "e2e_us": e2e_st,
        "throughput_gbs": throughput_gbs,
    }


# ─── Output writers ───────────────────────────────────────────────────────────


def _write_csv(path: Path, cells: list[dict], system: dict, params: dict) -> None:
    fieldnames = [
        "date",
        "gpu",
        "driver",
        "resolution",
        "dtype",
        "graphs",
        "metric",
        "mean",
        "p50",
        "p95",
        "p99",
        "min",
        "max",
        "n",
        "unit",
    ]
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for cell in cells:
        for metric_key, unit in [
            ("export_us", "µs"),
            ("get_numpy_us", "µs"),
            ("e2e_us", "µs"),
        ]:
            st = cell[metric_key]
            rows.append(
                {
                    "date": date_str,
                    "gpu": system.get("gpu", ""),
                    "driver": system.get("driver", ""),
                    "resolution": cell["resolution"],
                    "dtype": cell["dtype"],
                    "graphs": cell["graphs"],
                    "metric": metric_key,
                    "mean": f"{st['mean']:.2f}",
                    "p50": f"{st['p50']:.2f}",
                    "p95": f"{st['p95']:.2f}",
                    "p99": f"{st['p99']:.2f}",
                    "min": f"{st['min']:.2f}",
                    "max": f"{st['max']:.2f}",
                    "n": st["n"],
                    "unit": unit,
                }
            )
        rows.append(
            {
                "date": date_str,
                "gpu": system.get("gpu", ""),
                "driver": system.get("driver", ""),
                "resolution": cell["resolution"],
                "dtype": cell["dtype"],
                "graphs": cell["graphs"],
                "metric": "throughput_gbs",
                "mean": f"{cell['throughput_gbs']:.4f}",
                "p50": "",
                "p95": "",
                "p99": "",
                "min": "",
                "max": "",
                "n": cell["e2e_us"]["n"],
                "unit": "GB/s",
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, cells: list[dict], system: dict, params: dict) -> None:
    out = {
        "system": system,
        "params": params,
        "generated": datetime.datetime.now().isoformat(),
        "cells": {},
    }
    for cell in cells:
        key = f"{cell['resolution']}_{cell['dtype']}_graphs{cell['graphs']}"
        out["cells"][key] = {
            "resolution": cell["resolution"],
            "dtype": cell["dtype"],
            "graphs": cell["graphs"],
            "bytes_per_frame": cell["bytes_per_frame"],
            "export_us": cell["export_us"],
            "get_numpy_us": cell["get_numpy_us"],
            "e2e_us": cell["e2e_us"],
            "throughput_gbs": cell["throughput_gbs"],
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="cuda-link full IPC sweep benchmark")
    parser.add_argument("--frames", type=int, default=2000, help="Measurement frames per cell (default: 2000)")
    parser.add_argument("--warmup", type=int, default=500, help="Warmup frames to discard per cell (default: 500)")
    parser.add_argument("--fps", type=float, default=60.0, help="Producer target FPS (default: 60)")
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=list(_RESOLUTION_MAP.keys()),
        choices=list(_RESOLUTION_MAP.keys()),
        metavar="{512|720|1080|2160}",
        help="Resolutions: 512=512x512, 720=1280x720, 1080=1920x1080, 2160=3840x2160",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=list(_DTYPE_MAP.keys()),
        choices=list(_DTYPE_MAP.keys()),
        metavar="{f32|u8}",
        help="Dtypes: f32=float32, u8=uint8",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        type=int,
        default=[0, 1],
        choices=[0, 1],
        metavar="{0|1}",
        help="CUDA Graphs: 0=off, 1=on",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results"),
        help="Output directory for CSV/JSON results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 512x512 / f32 / graphs=on only, 100 warmup + 200 frames (~1 min)",
    )
    args = parser.parse_args(argv)

    if args.quick:
        resolutions = ["512"]
        dtypes = ["f32"]
        graphs_flags = [1]
        n_warmup, n_frames = 100, 200
    else:
        resolutions = args.resolutions
        dtypes = args.dtypes
        graphs_flags = args.graphs
        n_warmup, n_frames = args.warmup, args.frames

    cells_spec = [
        (_RESOLUTION_MAP[r], _DTYPE_MAP[d], bool(g)) for r in resolutions for d in dtypes for g in graphs_flags
    ]
    total = len(cells_spec)
    est_min = total * (n_warmup + n_frames) / args.fps / 60

    system = _get_system_info()
    params = {
        "n_warmup": n_warmup,
        "n_frames": n_frames,
        "fps": args.fps,
        "quick": args.quick,
    }

    print(
        f"\nbench_sweep  --  {total} cells  {n_warmup}+{n_frames} frames @ {args.fps:.0f} FPS  (~{est_min:.0f} min est.)"
    )
    print(f"GPU   : {system.get('gpu', 'unknown')}")
    print(
        f"Driver: {system.get('driver', 'unknown')}  "
        f"PCIe gen{system.get('pcie_gen', '?')} x{system.get('pcie_width', '?')}  "
        f"VRAM: {system.get('vram', '?')}"
    )
    print()
    print(
        f"  {'#':>4}  {'resolution':>10}  {'dtype':<8}  {'graphs':9}  "
        f"{'export mean':>12}  {'d2h mean':>10}  {'e2e mean':>10}  {'tput':>11}"
    )
    print("  " + "-" * 80)

    results: list[dict] = []
    for idx, ((w, h), dtype, use_graphs) in enumerate(cells_spec, start=1):
        cell = _run_cell(w, h, dtype, use_graphs, n_warmup, n_frames, args.fps, idx, total)
        if cell is not None:
            results.append(cell)

    if not results:
        print("\nNo cells completed successfully — check CUDA setup and try --quick first.")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    csv_ts = args.output / f"sweep_{ts}.csv"
    json_ts = args.output / f"sweep_{ts}.json"
    csv_latest = args.output / "sweep_latest.csv"
    json_latest = args.output / "sweep_latest.json"

    _write_csv(csv_ts, results, system, params)
    _write_json(json_ts, results, system, params)
    shutil.copy2(csv_ts, csv_latest)
    shutil.copy2(json_ts, json_latest)

    print("\nResults saved:")
    print(f"  {csv_ts}")
    print(f"  {json_ts}")
    print(f"  {csv_latest}  (latest)")
    print(f"  {json_latest}  (latest)")

    print("\n-- Summary (p50) " + "-" * 55)
    print(
        f"  {'resolution':>10}  {'dtype':<8}  {'graphs':7}  "
        f"{'export us':>10}  {'d2h us':>8}  {'e2e us':>8}  {'GB/s':>7}"
    )
    print("  " + "-" * 62)
    for cell in results:
        print(
            f"  {cell['resolution']:>10}  {cell['dtype']:<8}  {cell['graphs']:7}  "
            f"{cell['export_us']['p50']:>10.1f}  "
            f"{cell['get_numpy_us']['p50']:>8.1f}  "
            f"{cell['e2e_us']['p50']:>8.1f}  "
            f"{cell['throughput_gbs']:>7.3f}"
        )
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
