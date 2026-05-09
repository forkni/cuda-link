"""
bench_d2h_streams.py — Benchmark multi-stream D2H for get_frame_numpy().

Directly times the D2H copy portion of get_frame_numpy() by allocating device
and pinned-host buffers via CUDARuntimeAPI and issuing the chunked memcpy pattern
with CUDALINK_D2H_STREAMS ∈ {1, 2, 4} over N_FRAMES iterations at each frame size.

Usage (run from the cuda-link repo root):
    python -m benchmarks.bench_d2h_streams
    python -m benchmarks.bench_d2h_streams --frames 1000 --streams 1 2 4 8
    python -m benchmarks.bench_d2h_streams --sizes 512 1920

Requirements: CUDA-capable GPU; run from the repo root (src/ is auto-inserted).
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    idx = (len(s) - 1) * pct / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _run_d2h_bench(width: int, height: int, n_frames: int, n_streams: int) -> list[float]:
    """Return per-frame D2H copy wall-clock times in ms."""
    import contextlib

    from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

    channels, itemsize = 4, 4  # RGBA float32
    nbytes = height * width * channels * itemsize

    cuda = CUDARuntimeAPI()
    gpu_buf = cuda.malloc(nbytes)
    pinned_ptr = cuda.malloc_host_alloc(nbytes, flags=0x01)

    streams = [cuda.create_stream(flags=0x01) for _ in range(n_streams)]
    events = [cuda.create_sync_event() for _ in range(n_streams)]

    def d2h_copy() -> None:
        if n_streams <= 1:
            cuda.memcpy_async(
                dst=ctypes.c_void_p(pinned_ptr.value),
                src=gpu_buf,
                count=nbytes,
                kind=2,  # cudaMemcpyDeviceToHost
                stream=streams[0],
            )
            cuda.stream_synchronize(streams[0])
        else:
            chunk = ((nbytes + n_streams - 1) // n_streams + 15) & ~15
            issued = 0
            for i in range(n_streams):
                offset = i * chunk
                size = min(chunk, nbytes - offset)
                if size <= 0:
                    break
                cuda.memcpy_async(
                    dst=ctypes.c_void_p(pinned_ptr.value + offset),
                    src=ctypes.c_void_p(gpu_buf.value + offset),
                    count=size,
                    kind=2,
                    stream=streams[i],
                )
                cuda.record_event(events[i], stream=streams[i])
                issued = i + 1
            for i in range(issued):
                cuda.wait_event(events[i])

    try:
        for _ in range(30):  # warm-up
            d2h_copy()

        latencies: list[float] = []
        for _ in range(n_frames):
            t0 = time.perf_counter()
            d2h_copy()
            latencies.append((time.perf_counter() - t0) * 1e3)
    finally:
        for evt in events:
            with contextlib.suppress(RuntimeError, OSError):
                cuda.destroy_event(evt)
        for s in streams:
            with contextlib.suppress(RuntimeError, OSError):
                cuda.destroy_stream(s)
        with contextlib.suppress(RuntimeError, OSError):
            cuda.free_host(pinned_ptr)
        cuda.free(gpu_buf)

    return latencies


def _fmt_row(label: str, n: int, data: list[float], nbytes: int) -> str:
    mean = sum(data) / len(data)
    gbs = (nbytes / 1e9) / (mean / 1e3) if mean > 0 else 0.0
    return (
        f"{label:<14}  {n:>7}  {mean:>8.2f}  "
        f"{_percentile(data, 50):>7.2f}  "
        f"{_percentile(data, 95):>7.2f}  "
        f"{_percentile(data, 99):>7.2f}  "
        f"{gbs:>6.2f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark multi-stream D2H for get_frame_numpy()")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--streams", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 1920])
    args = parser.parse_args(argv)

    sizes = [(w, w * 9 // 16 if w >= 1280 else w) for w in args.sizes]

    print(f"\nbench_d2h_streams — {args.frames} frames per config\n")
    hdr = f"{'size':<14}  {'streams':>7}  {'mean ms':>8}  {'p50 ms':>7}  {'p95 ms':>7}  {'p99 ms':>7}  {'GB/s':>6}"
    print(hdr)
    print("-" * len(hdr))

    for w, h in sizes:
        channels, itemsize = 4, 4
        nbytes = h * w * channels * itemsize
        for n in args.streams:
            try:
                data = _run_d2h_bench(w, h, args.frames, n)
                print(_fmt_row(f"{w}x{h}", n, data, nbytes))
            except Exception as exc:  # noqa: BLE001
                print(f"{f'{w}x{h}':<14}  {n:>7}  ERROR: {exc}")

    print()


if __name__ == "__main__":
    main()
