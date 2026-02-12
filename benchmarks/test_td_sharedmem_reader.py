"""Simple test script for TDSharedMemReader.

This script connects to a TD Shared Mem Out TOP and reads 10 frames,
printing metadata and basic stats. Use this to verify the reader works
before running the full benchmark.

Setup in TouchDesigner:
1. Create a Noise TOP (or any source)
2. Add Shared Mem Out TOP
   - Connect: noise1 → sharedmemout1
   - Parameters: name = "test_shm", downloadtype = Delayed(Fast)
3. Run this script

Usage:
    python benchmarks/test_td_sharedmem_reader.py
    python benchmarks/test_td_sharedmem_reader.py --name my_custom_name
"""

import argparse
import sys
import time

# td_sharedmem_reader is now in the same directory as this script
from td_sharedmem_reader import TDSharedMemReader


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test TDSharedMemReader connection and read frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        type=str,
        default="test_shm",
        help="SharedMemory name matching TD Shared Mem Out TOP 'name' parameter (default: test_shm)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of frames to read (default: 10)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    print(f"{'=' * 60}")
    print("TDSharedMemReader Test")
    print(f"{'=' * 60}")
    print(f"SharedMemory name: {args.name}")
    print(f"Frames to read: {args.frames}")
    print(f"{'=' * 60}\n")

    # Create reader (always use debug=True since this is a diagnostic tool)
    reader = TDSharedMemReader(args.name, debug=True)

    # Connect
    print("Connecting to TD Shared Mem Out TOP...")
    if not reader.connect():
        print("X Failed to connect")
        print("\nTroubleshooting:")
        print("  1. Ensure TouchDesigner is running")
        print("  2. Check that Shared Mem Out TOP exists with matching name")
        print("  3. Verify the TOP is cooking (check its viewer)")
        print(f"  4. Expected mutex name: TouchSHM{args.name}Mutex")
        print(f"  5. Expected data mapping: TouchSHM{args.name}")
        return 1

    print("OK Connected!\n")
    print("Metadata:")
    print(f"  Resolution: {reader.width}x{reader.height}")
    print(f"  Pixel format: {reader.pixel_format_name} (enum value: {reader.pixel_format})")
    print(f"  NumPy dtype: {reader.dtype}")
    print(f"  Channels: {reader.num_channels}")
    print(f"  Shape: {reader.shape}")
    print(f"  Data size: {reader.data_size:,} bytes\n")

    # Read frames
    print(f"Reading {args.frames} frames...\n")

    frame_times = []
    successful_reads = 0

    for i in range(args.frames):
        start = time.perf_counter()

        frame = reader.read_frame()

        elapsed_ms = (time.perf_counter() - start) * 1000

        if frame is not None:
            successful_reads += 1
            frame_times.append(elapsed_ms)

            # Print progress every frame
            print(
                f"Frame {i + 1}/{args.frames}: shape={frame.shape}, dtype={frame.dtype}, read_time={elapsed_ms:.2f}ms"
            )

            # Print pixel value stats for first frame
            if i == 0:
                print(
                    f"  First frame pixel stats: "
                    f"min={frame.min():.3f}, max={frame.max():.3f}, "
                    f"mean={frame.mean():.3f}, std={frame.std():.3f}"
                )
        else:
            print(f"Frame {i + 1}/{args.frames}: !  Read returned None")

        # Small delay to avoid hammering the mutex
        time.sleep(0.016)  # ~60 FPS pacing

    # Cleanup
    reader.cleanup()

    # Print summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")
    print(f"Successful reads: {successful_reads}/{args.frames}")

    if frame_times:
        import numpy as np

        times = np.array(frame_times)
        print("\nFrame Read Times:")
        print(f"  Average: {times.mean():.2f} ms")
        print(f"  Median:  {np.median(times):.2f} ms")
        print(f"  Min:     {times.min():.2f} ms")
        print(f"  Max:     {times.max():.2f} ms")
        print(f"  Std:     {times.std():.2f} ms")

    print("\nOK Test complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
