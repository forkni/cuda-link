"""
Benchmark comparison script for CUDA IPC vs TouchDesigner-native solutions.

This script measures end-to-end latency, frame delay, and throughput for CUDA IPC
texture transfer from TouchDesigner to Python consumer.

Usage:
    python benchmark_comparison.py --frames 600 --warmup 60 --shm-name cuda_ipc_default
    python benchmark_comparison.py --frames 1000 --csv results.csv

Requirements:
    - TouchDesigner sender running (timestamps are always written)
    - CUDAIPCImporter configured with matching SharedMemory name
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np

# Import from cuda_link package (must be installed)
from cuda_link import CUDAIPCImporter

# SharedMemory protocol constants
WRITE_IDX_OFFSET = 16  # Offset for write_idx in SharedMemory buffer
SHM_HEADER_SIZE = 20  # magic(4) + version(8) + num_slots(4) + write_idx(4)
SLOT_SIZE = 192  # mem_handle(128) + event_handle(64)
NUM_SLOTS_OFFSET = 12  # Offset for num_slots in SharedMemory header


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
    max_retries: int = 30,
    retry_interval: float = 1.0,
) -> dict[str, object] | None:
    """Run CUDA IPC benchmark and collect metrics.

    Args:
        shm_name: SharedMemory name for CUDA IPC
        num_frames: Number of frames to measure
        warmup_frames: Number of warmup frames before measurement
        output_csv: Optional CSV file path for raw data export
        max_retries: Maximum connection attempts to wait for TD sender
        retry_interval: Seconds between retry attempts

    Returns:
        Dictionary with benchmark results and statistics
    """
    print(f"\n{'=' * 60}")
    print("CUDA IPC Benchmark Comparison")
    print(f"{'=' * 60}")
    print(f"SharedMemory name: {shm_name}")
    print(f"Warmup frames: {warmup_frames}")
    print(f"Measurement frames: {num_frames}")
    print(f"{'=' * 60}\n")

    # Initialize importer with retry logic for TD startup
    print("Initializing CUDAIPCImporter...")
    importer = None
    last_error = None

    for attempt in range(max_retries):
        # Phase 1: Lightweight pre-check — SharedMemory exists and sender is active?
        try:
            shm = SharedMemory(name=shm_name)
            # Read num_slots to compute shutdown_flag offset
            num_slots = struct.unpack_from("<I", shm.buf, NUM_SLOTS_OFFSET)[0]
            shutdown_offset = SHM_HEADER_SIZE + (num_slots * SLOT_SIZE)
            shutdown_flag = shm.buf[shutdown_offset]
            shm.close()

            if shutdown_flag == 1:
                if attempt == 0:
                    print("  SharedMemory found but sender has shut down (stale)")
                print(f"  Waiting for TD sender... ({attempt + 1}/{max_retries})")
                time.sleep(retry_interval)
                continue
        except FileNotFoundError:
            if attempt == 0:
                print("  SharedMemory not found — waiting for TD sender to start")
            print(f"  Waiting for TD sender... ({attempt + 1}/{max_retries})")
            time.sleep(retry_interval)
            continue
        except (OSError, RuntimeError) as e:
            if attempt == 0:
                print(f"  Pre-check failed: {e}")
            print(f"  Waiting for TD sender... ({attempt + 1}/{max_retries})")
            time.sleep(retry_interval)
            continue

        # Phase 2: Full initialization — SharedMemory valid, sender active
        try:
            candidate = CUDAIPCImporter(shm_name=shm_name, debug=True)
            # Check if initialization actually succeeded (_initialize() returns False silently)
            if candidate._initialized and candidate.shape is not None:
                importer = candidate
                break
            else:
                # Init failed silently — cleanup and retry
                if candidate.shm_handle is not None:
                    candidate.cleanup()
                last_error = "initialization returned False"
        except (OSError, RuntimeError) as e:
            last_error = str(e)
            if attempt == 0:
                print(f"  Full init failed: {e}")

        print(f"  Waiting for TD sender... ({attempt + 1}/{max_retries})")
        time.sleep(retry_interval)

    # Final check after retry loop
    if importer is None or not importer._initialized or importer.shape is None:
        print(f"❌ Failed to initialize importer after {max_retries} attempts")
        if last_error:
            print(f"   Last error: {last_error}")
        print("\nTroubleshooting:")
        print("  1. Ensure TouchDesigner sender is running")
        print("  2. Verify SharedMemory name matches TD parameter")
        print("  3. Check that Active=ON in TD sender component")
        print("  4. Check TD textport for sender initialization errors")
        return None

    height, width, channels = importer.shape
    print(
        f"✅ Connected to sender (resolution: {width}x{height}, format: {channels} channels, dtype: {importer.dtype})"
    )
    print(f"   Ring buffer: {importer.num_slots} slots\n")

    # Warmup phase
    print(f"Warming up ({warmup_frames} frames)...")
    warmed = 0
    warmup_timeout = 10.0
    warmup_start = time.perf_counter()
    while warmed < warmup_frames:
        if time.perf_counter() - warmup_start > warmup_timeout:
            print(f"⚠️  Warmup timeout — got {warmed}/{warmup_frames} frames")
            break
        try:
            frame = importer.get_frame()
            if frame is None:
                time.sleep(0.0005)
                continue
            warmed += 1
        except (RuntimeError, OSError) as e:
            print(f"❌ Warmup failed: {e}")
            importer.cleanup()
            return None
    print("✅ Warmup complete\n")

    # Measurement phase
    print(f"Collecting measurements ({num_frames} frames)...")
    latencies = []
    frame_times = []
    frame_skips = 0
    last_write_idx = None

    csv_rows = []

    start_time = time.perf_counter()
    successful_frames = 0
    poll_count = 0
    timeout = max(60.0, num_frames * 0.1)  # Scale with frame count, minimum 60s

    while successful_frames < num_frames:
        # Safety timeout
        if time.perf_counter() - start_time > timeout:
            print(f"⚠️  Timeout after {timeout}s — got {successful_frames}/{num_frames} frames")
            break

        frame_start = time.perf_counter()

        frame = importer.get_frame()

        if frame is None:
            poll_count += 1
            time.sleep(0.0005)  # 0.5ms sleep to avoid busy-wait
            continue

        successful_frames += 1

        # Record end-to-end latency (from producer timestamp)
        if importer.last_latency > 0:
            latencies.append(importer.last_latency)

        # Record frame processing time
        frame_time = (time.perf_counter() - frame_start) * 1000  # ms
        frame_times.append(frame_time)

        # Detect frame skips by reading write_idx directly from SharedMemory
        current_write_idx = struct.unpack_from("<I", importer.shm_handle.buf, WRITE_IDX_OFFSET)[0]
        if last_write_idx is not None:
            # Frame skips occur when write_idx jumps by more than 1
            idx_diff = current_write_idx - last_write_idx
            if idx_diff > 1:
                frame_skips += idx_diff - 1
        last_write_idx = current_write_idx

        # Store for CSV export
        if output_csv:
            csv_rows.append(
                {
                    "frame": successful_frames,
                    "end_to_end_latency_ms": importer.last_latency,
                    "frame_processing_time_ms": frame_time,
                    "write_idx": current_write_idx or 0,
                }
            )

        # Progress indicator
        if successful_frames % 100 == 0:
            print(f"  Progress: {successful_frames}/{num_frames} frames ({successful_frames / num_frames * 100:.0f}%)")

    total_time = time.perf_counter() - start_time

    # Compute statistics
    latency_stats = compute_statistics(latencies)
    frame_time_stats = compute_statistics(frame_times)

    # Calculate throughput (successful_frames already computed in loop)
    average_fps = successful_frames / total_time if total_time > 0 else 0

    # Build results
    results = {
        "config": {
            "shm_name": shm_name,
            "resolution": f"{width}x{height}",
            "format": f"{channels} channels",
            "dtype": str(importer.dtype),
            "num_slots": importer.num_slots,
        },
        "measurement": {
            "total_frames_requested": num_frames,
            "successful_frames": successful_frames,
            "frame_skips": frame_skips,
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
                fieldnames = ["frame", "end_to_end_latency_ms", "frame_processing_time_ms", "write_idx"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\n✅ Raw data exported to: {output_csv}")
        except (OSError, PermissionError) as e:
            print(f"\n⚠️  Failed to export CSV: {e}")

    # Cleanup
    importer.cleanup()

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
    print(f"  Frame skips detected: {m['frame_skips']}")
    print(f"  Total time: {m['total_time_sec']:.2f} seconds")
    print(f"  Average FPS: {m['average_fps']:.1f}")

    # End-to-end latency statistics
    print("\nEnd-to-End Latency (Producer Timestamp → Consumer):")
    lat = results["end_to_end_latency"]
    print(f"  Average:     {lat['avg']:.2f} ms")
    print(f"  Median (p50): {lat['p50']:.2f} ms")
    print(f"  p95:         {lat['p95']:.2f} ms")
    print(f"  p99:         {lat['p99']:.2f} ms")
    print(f"  Min:         {lat['min']:.2f} ms")
    print(f"  Max:         {lat['max']:.2f} ms")

    # Frame processing time statistics
    print("\nFrame Processing Time (get_frame() execution):")
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
        description="Benchmark CUDA IPC texture transfer performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark with default settings
  python benchmark_comparison.py

  # Benchmark with custom frame count and CSV export
  python benchmark_comparison.py --frames 1000 --csv results.csv

  # Benchmark with custom SharedMemory name
  python benchmark_comparison.py --shm-name my_cuda_ipc --frames 600

Note:
  - Ensure TouchDesigner sender is running
  - End-to-end latency uses unconditional producer timestamps
        """,
    )

    parser.add_argument("--frames", type=int, default=600, help="Number of frames to measure (default: 600)")
    parser.add_argument(
        "--warmup", type=int, default=60, help="Number of warmup frames before measurement (default: 60)"
    )
    parser.add_argument(
        "--shm-name",
        type=str,
        default="cuda_ipc_handle",
        help="SharedMemory name for CUDA IPC (default: cuda_ipc_handle)",
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
        max_retries=args.max_retries,
        retry_interval=args.retry_interval,
    )

    if results is None:
        print("\n❌ Benchmark failed. See error messages above.")
        sys.exit(1)

    # Print results
    print_results(results)

    print("✅ Benchmark complete!")


if __name__ == "__main__":
    main()
