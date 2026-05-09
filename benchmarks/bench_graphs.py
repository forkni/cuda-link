"""
bench_graphs.py — A/B benchmark for CUDA Graphs on the Python producer hot path.

Measures export_frame() latency with CUDALINK_USE_GRAPHS=0 vs =1 over N_FRAMES
iterations at each of several frame sizes.  Prints mean / p50 / p95 / p99 per
configuration.

Usage (run from the cuda-link repo root):
    python -m benchmarks.bench_graphs
    python -m benchmarks.bench_graphs --frames 5000
    python -m benchmarks.bench_graphs --sizes 512 1920

Requirements: CUDA-capable GPU; run from the repo root (src/ is auto-inserted).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Put this repo's src/ ahead of any installed package before cuda_link imports.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    idx = (len(s) - 1) * pct / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _run_export_bench(width: int, height: int, n_frames: int, use_graphs: bool) -> list[float]:
    """Return per-frame export_frame() wall-clock times in µs."""
    import time

    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter
    from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

    os.environ["CUDALINK_USE_GRAPHS"] = "1" if use_graphs else "0"
    # Force CPU sync so wall-clock timing reflects actual GPU work completion.
    os.environ["CUDALINK_EXPORT_SYNC"] = "1"

    channels, itemsize = 4, 4  # RGBA float32
    data_size = height * width * channels * itemsize
    shm_name = f"_bench_graphs_{os.getpid()}_{width}x{height}"

    cuda = CUDARuntimeAPI()
    gpu_buf = cuda.malloc(data_size)

    exp = CUDAIPCExporter(
        shm_name=shm_name,
        height=height,
        width=width,
        channels=channels,
        dtype="float32",
        num_slots=2,
    )
    try:
        exp.initialize()

        for _ in range(50):  # warm-up
            exp.export_frame(gpu_ptr=gpu_buf.value, size=data_size)

        latencies: list[float] = []
        for _ in range(n_frames):
            t0 = time.perf_counter()
            exp.export_frame(gpu_ptr=gpu_buf.value, size=data_size)
            latencies.append((time.perf_counter() - t0) * 1e6)
    finally:
        exp.cleanup()
        cuda.free(gpu_buf)

    return latencies


def _fmt_row(label: str, tag: str, data: list[float]) -> str:
    mean = sum(data) / len(data)
    return (
        f"{label:<14}  {tag:<7}  {mean:>8.1f}  "
        f"{_percentile(data, 50):>7.1f}  "
        f"{_percentile(data, 95):>7.1f}  "
        f"{_percentile(data, 99):>7.1f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A/B benchmark: CUDA Graphs for export_frame()")
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 1920])
    args = parser.parse_args(argv)

    sizes = [(w, w * 9 // 16 if w >= 1280 else w) for w in args.sizes]

    print(f"\nbench_graphs — {args.frames} frames per config\n")
    hdr = f"{'size':<14}  {'graphs':<7}  {'mean µs':>8}  {'p50 µs':>7}  {'p95 µs':>7}  {'p99 µs':>7}"
    print(hdr)
    print("-" * len(hdr))

    for w, h in sizes:
        for use_graphs in (False, True):
            try:
                data = _run_export_bench(w, h, args.frames, use_graphs)
                print(_fmt_row(f"{w}x{h}", "on" if use_graphs else "off", data))
            except Exception as exc:  # noqa: BLE001
                print(f"{f'{w}x{h}':<14}  {'on' if use_graphs else 'off':<7}  ERROR: {exc}")

    print()


if __name__ == "__main__":
    main()
