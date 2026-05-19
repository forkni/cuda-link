"""
profile_export.py — Export-path benchmarking harness for cuda-link.

Runs CUDAIPCExporter.initialize() + N × export_frame() against a fixed synthetic
GPU buffer, then writes per-region timing stats to a JSON file.

This script is the measurement target for scalene.  Run it two ways:

    # 1. Collect timing-only baseline (no scalene, any machine):
    python scripts/profiling/profile_export.py [--frames 1000] [--outfile .profiling/baseline.json]

    # 2. Full scalene profile (requires scalene install, pip install scalene):
    python -m scalene --cli --json --outfile .profiling/scalene_baseline.json -- \\
        scripts/profiling/profile_export.py --frames 1000

Compare two baselines:
    python scripts/profiling/compare.py .profiling/baseline.json .profiling/current.json

Requires: a CUDA GPU + cuda-link installed (or src/ on sys.path).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, quantiles
from typing import Any

# Ensure repo src/ is on the path when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

WARMUP_FRAMES = 50


def _percentile(values: list[float], p: int) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    return quantiles(values, n=100)[p - 1]


def _region_stats(samples_us: list[float]) -> dict[str, float]:
    return {
        "n": len(samples_us),
        "min_us": min(samples_us),
        "median_us": median(samples_us),
        "p95_us": _percentile(samples_us, 95),
        "p99_us": _percentile(samples_us, 99),
        "max_us": max(samples_us),
    }


def run(frames: int, width: int, height: int, channels: int, dtype: str, slot_count: int) -> dict[str, Any]:
    from cuda_link import CUDAIPCExporter
    from cuda_link.cuda_ipc_wrapper import get_cuda_runtime

    cuda = get_cuda_runtime()
    nbytes = width * height * channels * (1 if dtype == "uint8" else 4)
    gpu_buf = cuda.malloc(nbytes)
    if gpu_buf is None or gpu_buf.value == 0:
        raise RuntimeError("cudaMalloc failed — is a CUDA GPU present?")
    gpu_ptr: int = gpu_buf.value

    shm_name = f"__profiler_{uuid.uuid4().hex[:8]}"
    total_samples: list[float] = []

    try:
        with CUDAIPCExporter(
            shm_name=shm_name,
            height=height,
            width=width,
            channels=channels,
            dtype=dtype,
            num_slots=slot_count,
        ) as exporter:
            exporter.initialize()

            for i in range(frames + WARMUP_FRAMES):
                t0 = time.perf_counter()
                exporter.export_frame(gpu_ptr=gpu_ptr, size=nbytes)
                elapsed_us = (time.perf_counter() - t0) * 1e6
                if i >= WARMUP_FRAMES:
                    total_samples.append(elapsed_us)
    finally:
        cuda.free(gpu_buf)

    return {
        "version": "1",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "config": {
            "frames": frames,
            "warmup_frames": WARMUP_FRAMES,
            "width": width,
            "height": height,
            "channels": channels,
            "dtype": dtype,
            "slot_count": slot_count,
            "nbytes": nbytes,
        },
        "regions": {
            "export_frame_total": _region_stats(total_samples),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="cuda-link export-path benchmark harness.")
    parser.add_argument("--frames", type=int, default=1000, help="Measurement frames (after warmup)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--dtype", default="uint8", choices=["uint8", "float32", "float16"])
    parser.add_argument("--slot-count", type=int, default=2, dest="slot_count")
    parser.add_argument(
        "--outfile",
        default=".profiling/baseline.json",
        help="Output JSON path (default: .profiling/baseline.json)",
    )
    args = parser.parse_args()

    print(
        f"cuda-link export harness: {args.width}x{args.height} {args.dtype} "
        f"{args.channels}ch, {args.slot_count} slots, {args.frames} frames"
    )

    try:
        result = run(
            frames=args.frames,
            width=args.width,
            height=args.height,
            channels=args.channels,
            dtype=args.dtype,
            slot_count=args.slot_count,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    stats = result["regions"]["export_frame_total"]
    print(f"export_frame: median={stats['median_us']:.1f}µs  p95={stats['p95_us']:.1f}µs  p99={stats['p99_us']:.1f}µs")
    print(f"Baseline written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
