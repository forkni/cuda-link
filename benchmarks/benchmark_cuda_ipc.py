"""
CUDA IPC Benchmark Script
Measures baseline performance for CUDA IPC texture sharing.

Usage:
    python benchmarks/benchmark_cuda_ipc.py --frames 1000
    python benchmarks/benchmark_cuda_ipc.py --fps 60 --duration 10
    python benchmarks/benchmark_cuda_ipc.py --resolution 1920x1080 --events

Metrics Tracked:
    - memcpy_us: Time for D2D GPU copy
    - record_event_us: Time to record synchronization event (if enabled)
    - wait_event_us: Time to wait on event (consumer side)
    - total_frame_us: Total per-frame latency
    - throughput_fps: Achieved frames per second
"""

from __future__ import annotations

import argparse
import statistics
import struct
import time
import traceback
from multiprocessing import Process, shared_memory

# Import from cuda_link package (must be installed)
from cuda_link.cuda_ipc_wrapper import cudaIpcMemHandle_t, get_cuda_runtime


class BenchmarkProducer:
    """Producer process - simulates TouchDesigner GPU frame export."""

    def __init__(
        self,
        shm_name: str,
        buffer_size: int,
        num_frames: int,
        target_fps: float | None = None,
        enable_events: bool = False,
    ):
        self.shm_name = shm_name
        self.buffer_size = buffer_size
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.enable_events = enable_events

        # Frame timing for FPS control
        self.frame_interval = 1.0 / target_fps if target_fps else 0

        # CUDA runtime
        self.cuda = None
        self.dev_ptr = None
        self.ipc_handle = None
        self.ipc_event = None
        self.shm_handle = None

        # Timing metrics
        self.metrics = {
            "memcpy_times": [],
            "record_event_times": [],
            "total_frame_times": [],
        }

    def initialize(self) -> None:
        """Initialize CUDA IPC producer."""
        print("[Producer] Initializing...")

        # Load CUDA runtime
        self.cuda = get_cuda_runtime()

        # Allocate GPU buffer
        self.dev_ptr = self.cuda.malloc(self.buffer_size)
        print(
            f"[Producer] Allocated GPU buffer: {self.buffer_size / 1024 / 1024:.1f} MB at 0x{self.dev_ptr.value:016x}"
        )

        # Create IPC handle
        self.ipc_handle = self.cuda.ipc_get_mem_handle(self.dev_ptr)
        print("[Producer] Created IPC handle")

        # Create IPC event (if enabled)
        if self.enable_events:
            self.ipc_event = self.cuda.create_ipc_event()
            event_handle = self.cuda.ipc_get_event_handle(self.ipc_event)
            print("[Producer] Created IPC event")
        else:
            event_handle = None

        # Create SharedMemory for handle transfer
        shm_size = 208 if self.enable_events else 136
        self.shm_handle = shared_memory.SharedMemory(name=self.shm_name, create=True, size=shm_size)

        # Write IPC handle to SharedMemory
        self.shm_handle.buf[0:8] = struct.pack("<Q", 1)  # version
        self.shm_handle.buf[8:136] = bytes(self.ipc_handle.internal)  # 128-byte handle

        if self.enable_events and event_handle:
            self.shm_handle.buf[136:200] = bytes(event_handle.reserved)  # 64-byte event handle

        print(f"[Producer] Wrote IPC handle to SharedMemory: {self.shm_name}")

    def produce_frames(self) -> None:
        """Produce frames with D2D copy and optional event recording."""
        print(f"[Producer] Starting frame production: {self.num_frames} frames")

        # Allocate temp buffer for memcpy source
        temp_ptr = self.cuda.malloc(self.buffer_size)

        frame_start_time = time.perf_counter()

        for frame_num in range(self.num_frames):
            iter_start = time.perf_counter()

            # D2D memcpy (simulates TD texture copy)
            memcpy_start = time.perf_counter()
            self.cuda.memcpy(
                dst=self.dev_ptr,
                src=temp_ptr,
                count=self.buffer_size,
                kind=3,  # D2D
            )
            memcpy_time = (time.perf_counter() - memcpy_start) * 1_000_000
            self.metrics["memcpy_times"].append(memcpy_time)

            # Record event (if enabled)
            record_event_time = 0.0
            if self.enable_events and self.ipc_event:
                record_start = time.perf_counter()
                self.cuda.record_event(self.ipc_event)
                record_event_time = (time.perf_counter() - record_start) * 1_000_000
                self.metrics["record_event_times"].append(record_event_time)
            else:
                # CPU sync every 10 frames if no events
                if frame_num % 10 == 0:
                    self.cuda.synchronize()

            # Total frame time
            frame_time = (time.perf_counter() - iter_start) * 1_000_000
            self.metrics["total_frame_times"].append(frame_time)

            # FPS rate limiting
            if self.target_fps:
                elapsed = time.perf_counter() - iter_start
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # Final sync
        self.cuda.synchronize()

        # Calculate stats
        total_time = time.perf_counter() - frame_start_time
        achieved_fps = self.num_frames / total_time

        print(f"[Producer] Completed {self.num_frames} frames in {total_time:.2f}s ({achieved_fps:.1f} FPS)")

        # Free temp buffer
        self.cuda.free(temp_ptr)

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.dev_ptr:
            self.cuda.free(self.dev_ptr)
        if self.shm_handle:
            self.shm_handle.close()
            self.shm_handle.unlink()
        print("[Producer] Cleanup complete")

    def get_summary(self) -> dict[str, float]:
        """Get performance summary statistics."""

        def percentile(data: list[float], p: float) -> float:
            return sorted(data)[int(len(data) * p / 100)] if data else 0

        return {
            "memcpy_avg": statistics.mean(self.metrics["memcpy_times"]) if self.metrics["memcpy_times"] else 0,
            "memcpy_p50": percentile(self.metrics["memcpy_times"], 50),
            "memcpy_p95": percentile(self.metrics["memcpy_times"], 95),
            "memcpy_p99": percentile(self.metrics["memcpy_times"], 99),
            "record_event_avg": statistics.mean(self.metrics["record_event_times"])
            if self.metrics["record_event_times"]
            else 0,
            "total_frame_avg": statistics.mean(self.metrics["total_frame_times"])
            if self.metrics["total_frame_times"]
            else 0,
            "total_frame_p50": percentile(self.metrics["total_frame_times"], 50),
            "total_frame_p95": percentile(self.metrics["total_frame_times"], 95),
            "total_frame_p99": percentile(self.metrics["total_frame_times"], 99),
        }


class BenchmarkConsumer:
    """Consumer process - simulates Python AI process reading frames."""

    def __init__(
        self,
        shm_name: str,
        num_frames: int,
        enable_events: bool = False,
    ):
        self.shm_name = shm_name
        self.num_frames = num_frames
        self.enable_events = enable_events

        # CUDA runtime
        self.cuda = None
        self.dev_ptr = None
        self.ipc_event = None
        self.shm_handle = None

        # Timing metrics
        self.metrics = {
            "wait_event_times": [],
            "total_frame_times": [],
        }

    def initialize(self):
        """Initialize CUDA IPC consumer."""
        print("[Consumer] Initializing...")

        # Load CUDA runtime
        self.cuda = get_cuda_runtime()

        # Open SharedMemory
        retry_count = 0
        while retry_count < 10:
            try:
                self.shm_handle = shared_memory.SharedMemory(name=self.shm_name)
                break
            except FileNotFoundError:
                retry_count += 1
                time.sleep(0.1)

        if not self.shm_handle:
            raise RuntimeError("Failed to open SharedMemory")

        # Read IPC handle

        mem_handle_bytes = bytes(self.shm_handle.buf[8:136])
        ipc_handle = cudaIpcMemHandle_t.from_buffer_copy(mem_handle_bytes)

        # Open IPC memory
        self.dev_ptr = self.cuda.ipc_open_mem_handle(ipc_handle, flags=1)
        print(f"[Consumer] Opened IPC handle: GPU at 0x{self.dev_ptr.value:016x}")

        # Open IPC event (if enabled)
        if self.enable_events:
            # Import event handle type
            from cuda_ipc_wrapper import cudaIpcEventHandle_t

            event_handle_bytes = bytes(self.shm_handle.buf[136:200])
            if any(event_handle_bytes):
                ipc_event_handle = cudaIpcEventHandle_t.from_buffer_copy(event_handle_bytes)
                self.ipc_event = self.cuda.ipc_open_event_handle(ipc_event_handle)
                print("[Consumer] Opened IPC event")

    def consume_frames(self) -> None:
        """Consume frames with optional event wait."""
        print(f"[Consumer] Starting frame consumption: {self.num_frames} frames")

        # Try importing torch for optional GPU sync
        try:
            import torch

            torch_available = True
        except ImportError:
            torch_available = False

        frame_start_time = time.perf_counter()

        for _frame_num in range(self.num_frames):
            iter_start = time.perf_counter()

            # Wait for event or sync
            wait_start = time.perf_counter()
            if self.ipc_event:
                self.cuda.wait_event(self.ipc_event)
            elif torch_available:
                torch.cuda.synchronize()
            else:
                self.cuda.synchronize()
            wait_time = (time.perf_counter() - wait_start) * 1_000_000
            self.metrics["wait_event_times"].append(wait_time)

            # Verify pointer is valid (just check it's not NULL)
            assert self.dev_ptr.value != 0

            # Total frame time
            frame_time = (time.perf_counter() - iter_start) * 1_000_000
            self.metrics["total_frame_times"].append(frame_time)

        # Final sync
        self.cuda.synchronize()

        total_time = time.perf_counter() - frame_start_time
        achieved_fps = self.num_frames / total_time

        print(f"[Consumer] Completed {self.num_frames} frames in {total_time:.2f}s ({achieved_fps:.1f} FPS)")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.dev_ptr:
            self.cuda.ipc_close_mem_handle(self.dev_ptr)
        if self.shm_handle:
            self.shm_handle.close()
        print("[Consumer] Cleanup complete")

    def get_summary(self) -> dict[str, float]:
        """Get performance summary statistics."""

        def percentile(data: list[float], p: int) -> float:
            return sorted(data)[int(len(data) * p / 100)] if data else 0

        return {
            "wait_event_avg": statistics.mean(self.metrics["wait_event_times"])
            if self.metrics["wait_event_times"]
            else 0,
            "wait_event_p50": percentile(self.metrics["wait_event_times"], 50),
            "wait_event_p95": percentile(self.metrics["wait_event_times"], 95),
            "wait_event_p99": percentile(self.metrics["wait_event_times"], 99),
            "total_frame_avg": statistics.mean(self.metrics["total_frame_times"])
            if self.metrics["total_frame_times"]
            else 0,
            "total_frame_p50": percentile(self.metrics["total_frame_times"], 50),
            "total_frame_p95": percentile(self.metrics["total_frame_times"], 95),
            "total_frame_p99": percentile(self.metrics["total_frame_times"], 99),
        }


def run_producer(
    shm_name: str, buffer_size: int, num_frames: int, target_fps: float | None, enable_events: bool
) -> None:
    """Producer process entry point."""
    try:
        producer = BenchmarkProducer(shm_name, buffer_size, num_frames, target_fps, enable_events)
        producer.initialize()
        producer.produce_frames()

        summary = producer.get_summary()
        print("\n[Producer] Performance Summary:")
        print(
            f"  memcpy:       avg={summary['memcpy_avg']:.1f}µs, p50={summary['memcpy_p50']:.1f}µs, p95={summary['memcpy_p95']:.1f}µs, p99={summary['memcpy_p99']:.1f}µs"
        )
        if enable_events:
            print(f"  record_event: avg={summary['record_event_avg']:.1f}µs")
        print(
            f"  total_frame:  avg={summary['total_frame_avg']:.1f}µs, p50={summary['total_frame_p50']:.1f}µs, p95={summary['total_frame_p95']:.1f}µs"
        )

        producer.cleanup()
    except (RuntimeError, OSError) as e:
        print(f"[Producer] Error: {e}")
        traceback.print_exc()


def run_consumer(shm_name: str, num_frames: int, enable_events: bool) -> None:
    """Consumer process entry point."""
    try:
        consumer = BenchmarkConsumer(shm_name, num_frames, enable_events)
        consumer.initialize()
        consumer.consume_frames()

        summary = consumer.get_summary()
        print("\n[Consumer] Performance Summary:")
        print(
            f"  wait_event:  avg={summary['wait_event_avg']:.1f}µs, p50={summary['wait_event_p50']:.1f}µs, p95={summary['wait_event_p95']:.1f}µs, p99={summary['wait_event_p99']:.1f}µs"
        )
        print(
            f"  total_frame: avg={summary['total_frame_avg']:.1f}µs, p50={summary['total_frame_p50']:.1f}µs, p95={summary['total_frame_p95']:.1f}µs"
        )

        consumer.cleanup()
    except (RuntimeError, OSError) as e:
        print(f"[Consumer] Error: {e}")
        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA IPC Benchmark")
    parser.add_argument("--frames", type=int, default=1000, help="Number of frames to benchmark")
    parser.add_argument("--fps", type=float, help="Target FPS (default: unlimited)")
    parser.add_argument("--duration", type=int, help="Duration in seconds (overrides --frames)")
    parser.add_argument("--resolution", type=str, default="512x512", help="Resolution (e.g., 1920x1080)")
    parser.add_argument("--events", action="store_true", help="Enable IPC events for GPU-side sync")
    parser.add_argument("--stress", action="store_true", help="Stress test mode (long duration)")

    args = parser.parse_args()

    # Parse resolution
    width, height = map(int, args.resolution.split("x"))
    buffer_size = width * height * 4 * 4  # RGBA float32

    # Calculate frames
    if args.duration and args.fps:
        num_frames = int(args.duration * args.fps)
    elif args.stress:
        num_frames = 36000  # 10 minutes at 60 FPS
    else:
        num_frames = args.frames

    shm_name = "cuda_ipc_benchmark"

    print("=" * 80)
    print("CUDA IPC Benchmark")
    print("=" * 80)
    print(f"Resolution:    {width}x{height}")
    print(f"Buffer size:   {buffer_size / 1024 / 1024:.1f} MB")
    print(f"Frames:        {num_frames}")
    print(f"Target FPS:    {args.fps if args.fps else 'Unlimited'}")
    print(f"IPC Events:    {'Enabled' if args.events else 'Disabled (CPU sync fallback)'}")
    print("=" * 80)
    print()

    # Launch producer and consumer processes
    producer_proc = Process(target=run_producer, args=(shm_name, buffer_size, num_frames, args.fps, args.events))
    consumer_proc = Process(target=run_consumer, args=(shm_name, num_frames, args.events))

    producer_proc.start()
    time.sleep(0.5)  # Let producer initialize first
    consumer_proc.start()

    producer_proc.join()
    consumer_proc.join()

    print("\n" + "=" * 80)
    print("Benchmark Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
