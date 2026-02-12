"""
Benchmark script for TouchDesigner Shared Mem Out TOP.

This script measures end-to-end latency, frame delay, and throughput for TD's
built-in Shared Mem Out TOP (CPU SharedMemory with mutex synchronization).

Usage:
    python benchmark_sharedmem.py --frames 600 --warmup 60 --shm-name benchmark_shm
    python benchmark_sharedmem.py --frames 1000 --csv results.csv

Requirements:
    - TouchDesigner running with Shared Mem Out TOP configured
    - Optional: Shared timestamp channel for end-to-end latency measurement
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np

# td_sharedmem_reader is now in the same directory as this script
from td_sharedmem_reader import TDSharedMemReader


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute statistical summary of latency values.

    Args:
        values: List of latency measurements in milliseconds

    Returns:
        Dictionary with avg, p50, p95, p99, min, max values
    """
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}

    arr = np.array(values)
    return {
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run_benchmark(
    shm_name: str,
    num_frames: int,
    warmup_frames: int = 60,
    output_csv: str | None = None,
    timestamp_shm: str | None = None,
    max_retries: int = 30,
    retry_interval: float = 1.0,
) -> dict[str, object] | None:
    """Run Shared Mem Out TOP benchmark and collect metrics.

    Args:
        shm_name: SharedMemory name for Shared Mem Out TOP
        num_frames: Number of frames to measure
        warmup_frames: Number of warmup frames before measurement
        output_csv: Optional CSV file path for raw data export
        timestamp_shm: Optional timestamp SharedMemory name for end-to-end latency
        max_retries: Maximum connection attempts to wait for TD sender
        retry_interval: Seconds between retry attempts

    Returns:
        Dictionary with benchmark results and statistics, or None on failure
    """
    print(f"\n{'=' * 60}")
    print("Shared Mem Out TOP Benchmark")
    print(f"{'=' * 60}")
    print(f"Shared Mem Out name: {shm_name}")
    print(f"Warmup frames: {warmup_frames}")
    print(f"Measurement frames: {num_frames}")
    if timestamp_shm:
        print(f"Timestamp channel: {timestamp_shm}")
    print(f"{'=' * 60}\n")

    # Initialize reader with retry logic for TD startup
    print("Initializing TDSharedMemReader...")
    reader = None
    last_error = None

    for attempt in range(max_retries):
        try:
            candidate = TDSharedMemReader(shm_name, debug=False)
            if candidate.connect():
                reader = candidate
                break
            else:
                last_error = "connect() returned False"
                candidate.cleanup()
        except (OSError, RuntimeError) as e:
            last_error = str(e)
            if attempt == 0:
                print(f"  Connection attempt failed: {e}")

        print(f"  Waiting for TD sender... ({attempt + 1}/{max_retries})")
        time.sleep(retry_interval)

    # Final check after retry loop
    if reader is None:
        print(f"X Failed to connect after {max_retries} attempts")
        if last_error:
            print(f"   Last error: {last_error}")
        print("\nTroubleshooting:")
        print("  1. Ensure TouchDesigner is running")
        print("  2. Verify Shared Mem Out TOP exists with matching 'name' parameter")
        print("  3. Check that the TOP is cooking (viewer shows image)")
        print(f"  4. Expected data mapping: TouchSHM{shm_name}")
        return None

    print(
        f"OK Connected to sender (resolution: {reader.width}x{reader.height}, "
        f"format: {reader.pixel_format_name}, dtype: {reader.dtype}, channels: {reader.num_channels})"
    )
    print(f"   Data size: {reader.data_size:,} bytes\n")

    # Auto-detect or use provided timestamp channel
    ts_shm_handle = None
    if not timestamp_shm:
        # Try auto-detecting standard timestamp channel
        timestamp_shm = "cuda_ipc_benchmark_ts"

    try:
        ts_shm_handle = SharedMemory(name=timestamp_shm)
        print(f"OK Timestamp channel connected ({timestamp_shm})\n")
    except FileNotFoundError:
        print("! Timestamp channel not found (end-to-end latency disabled)\n")
        print("  To enable: Add Execute DAT in TD with td_exporter/benchmark_timestamp.py\n")
        timestamp_shm = None

    # Warmup phase
    print(f"Warming up ({warmup_frames} frames)...")
    warmed = 0
    warmup_timeout = 10.0
    warmup_start = time.perf_counter()

    while warmed < warmup_frames:
        if time.perf_counter() - warmup_start > warmup_timeout:
            print(f"!  Warmup timeout — got {warmed}/{warmup_frames} frames")
            break
        try:
            # Use timestamp channel for new-frame detection if available
            frame = reader.read_frame(timestamp_shm_name=timestamp_shm)
            if frame is None:
                time.sleep(0.0005)
                continue

            warmed += 1
        except (RuntimeError, OSError) as e:
            print(f"X Warmup failed: {e}")
            reader.cleanup()
            if ts_shm_handle:
                ts_shm_handle.close()
            return None

    print("OK Warmup complete\n")

    # Measurement phase
    print(f"Collecting measurements ({num_frames} frames)...")
    latencies = []
    frame_times = []
    successful_frames = 0

    csv_rows = []

    start_time = time.perf_counter()
    timeout = 60.0  # 60 second safety timeout

    while successful_frames < num_frames:
        # Safety timeout
        if time.perf_counter() - start_time > timeout:
            print(f"!  Timeout after {timeout}s — got {successful_frames}/{num_frames} frames")
            break

        frame_start = time.perf_counter()

        # Use timestamp channel for new-frame detection (returns None for stale frames)
        frame = reader.read_frame(timestamp_shm_name=timestamp_shm)

        if frame is None:
            time.sleep(0.0005)  # 0.5ms sleep to avoid busy-wait
            continue

        successful_frames += 1

        # Record end-to-end latency if timestamp channel available
        end_to_end_latency = 0.0
        if ts_shm_handle:
            try:
                frame_counter, producer_ts = struct.unpack_from("<Id", ts_shm_handle.buf, 0)
                consumer_ts = time.perf_counter()
                end_to_end_latency = (consumer_ts - producer_ts) * 1000  # ms
                latencies.append(end_to_end_latency)
            except (OSError, struct.error):
                pass  # Skip latency if timestamp read fails

        # Record frame processing time
        frame_time = (time.perf_counter() - frame_start) * 1000  # ms
        frame_times.append(frame_time)

        # Store for CSV export
        if output_csv:
            csv_rows.append(
                {
                    "frame": successful_frames,
                    "end_to_end_latency_ms": end_to_end_latency,
                    "frame_processing_time_ms": frame_time,
                }
            )

        # Progress indicator
        if successful_frames % 100 == 0:
            print(f"  Progress: {successful_frames}/{num_frames} frames ({successful_frames / num_frames * 100:.0f}%)")

    total_time = time.perf_counter() - start_time

    # Compute statistics
    latency_stats = compute_statistics(latencies) if latencies else None
    frame_time_stats = compute_statistics(frame_times)

    # Calculate throughput
    average_fps = successful_frames / total_time if total_time > 0 else 0

    # Build results
    results = {
        "config": {
            "shm_name": shm_name,
            "resolution": f"{reader.width}x{reader.height}",
            "format": f"{reader.pixel_format_name}",
            "dtype": str(reader.dtype),
            "channels": reader.num_channels,
            "data_size_bytes": reader.data_size,
        },
        "measurement": {
            "total_frames_requested": num_frames,
            "successful_frames": successful_frames,
            "total_time_sec": total_time,
            "average_fps": average_fps,
        },
        "end_to_end_latency": latency_stats,
        "frame_processing_time": frame_time_stats,
    }

    # Export CSV if requested
    if output_csv and csv_rows:
        try:
            with open(output_csv, "w", newline="") as f:
                fieldnames = ["frame", "end_to_end_latency_ms", "frame_processing_time_ms"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\nOK Raw data exported to: {output_csv}")
        except (OSError, PermissionError) as e:
            print(f"\n!  Failed to export CSV: {e}")

    # Cleanup
    reader.cleanup()
    if ts_shm_handle:
        ts_shm_handle.close()

    return results


def print_results(results: dict[str, object]) -> None:
    """Print formatted benchmark results.

    Args:
        results: Dictionary from run_benchmark()
    """
    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}\n")

    # Configuration
    print("Configuration:")
    for key, value in results["config"].items():
        print(f"  {key}: {value}")

    # Measurement summary
    print("\nMeasurement Summary:")
    m = results["measurement"]
    print(f"  Total frames requested: {m['total_frames_requested']}")
    print(f"  Successful frames: {m['successful_frames']}")
    print(f"  Total time: {m['total_time_sec']:.2f} seconds")
    print(f"  Average FPS: {m['average_fps']:.1f}")

    # End-to-end latency statistics (if available)
    if results["end_to_end_latency"]:
        print("\nEnd-to-End Latency (Producer Timestamp -> Consumer):")
        lat = results["end_to_end_latency"]
        print(f"  Average:     {lat['avg']:.2f} ms")
        print(f"  Median (p50): {lat['p50']:.2f} ms")
        print(f"  p95:         {lat['p95']:.2f} ms")
        print(f"  p99:         {lat['p99']:.2f} ms")
        print(f"  Min:         {lat['min']:.2f} ms")
        print(f"  Max:         {lat['max']:.2f} ms")
    else:
        print("\nEnd-to-End Latency: Not measured (no timestamp channel)")

    # Frame processing time statistics
    print("\nFrame Processing Time (read_frame() execution):")
    ft = results["frame_processing_time"]
    print(f"  Average:     {ft['avg']:.2f} ms ({ft['avg'] * 1000:.0f} us)")
    print(f"  Median (p50): {ft['p50']:.2f} ms ({ft['p50'] * 1000:.0f} us)")
    print(f"  p95:         {ft['p95']:.2f} ms ({ft['p95'] * 1000:.0f} us)")
    print(f"  p99:         {ft['p99']:.2f} ms ({ft['p99'] * 1000:.0f} us)")
    print(f"  Min:         {ft['min']:.2f} ms ({ft['min'] * 1000:.0f} us)")
    print(f"  Max:         {ft['max']:.2f} ms ({ft['max'] * 1000:.0f} us)")

    print(f"\n{'=' * 60}\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark TouchDesigner Shared Mem Out TOP performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark with default settings
  python benchmark_sharedmem.py

  # Benchmark with custom frame count and CSV export
  python benchmark_sharedmem.py --frames 1000 --csv results.csv

  # Benchmark with custom name and timestamp channel
  python benchmark_sharedmem.py --shm-name my_shm --timestamp-shm my_ts --frames 600

Note:
  - Ensure TouchDesigner is running with Shared Mem Out TOP configured
  - End-to-end latency requires a timestamp channel (optional)
  - Frame processing time is always measured
        """,
    )

    parser.add_argument("--frames", type=int, default=600, help="Number of frames to measure (default: 600)")
    parser.add_argument(
        "--warmup", type=int, default=60, help="Number of warmup frames before measurement (default: 60)"
    )
    parser.add_argument(
        "--shm-name",
        type=str,
        default="benchmark_shm",
        help="SharedMemory name for Shared Mem Out TOP 'name' parameter (default: benchmark_shm)",
    )
    parser.add_argument(
        "--timestamp-shm",
        type=str,
        default=None,
        help="Timestamp SharedMemory channel name for end-to-end latency (optional)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=30, help="Maximum connection attempts to wait for TD sender (default: 30)"
    )
    parser.add_argument(
        "--retry-interval", type=float, default=1.0, help="Seconds between retry attempts (default: 1.0)"
    )
    parser.add_argument("--csv", type=str, default=None, help="Export raw per-frame data to CSV file (optional)")

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(
        shm_name=args.shm_name,
        num_frames=args.frames,
        warmup_frames=args.warmup,
        output_csv=args.csv,
        timestamp_shm=args.timestamp_shm,
        max_retries=args.max_retries,
        retry_interval=args.retry_interval,
    )

    if results is None:
        print("\nX Benchmark failed. See error messages above.")
        sys.exit(1)

    # Print results
    print_results(results)

    print("OK Benchmark complete!")


if __name__ == "__main__":
    main()
