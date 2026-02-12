"""
TouchDesigner Execute DAT script for shared timestamp channel.

This script creates a Python SharedMemory segment that both CUDA IPC and
Shared Mem Out TOP benchmarks can read to measure end-to-end latency.

Setup in TouchDesigner:
1. Create Execute DAT
2. Paste this script into the DAT
3. Set to: DAT Execute → Active = ON
4. Callbacks: onFrameEnd = ON (all others OFF)

The timestamp channel writes:
- frame_counter (uint32) - increments each frame
- timestamp (float64) - time.perf_counter() when frame ends

Both benchmark scripts read this to compute end-to-end latency:
  consumer_time - producer_time = latency
"""

import contextlib
import struct
import time
from multiprocessing.shared_memory import SharedMemory

# Global state (persists across frame callbacks)
shm = None
frame_counter = 0

# SharedMemory name must match benchmark script --timestamp-shm argument
TIMESTAMP_SHM_NAME = "cuda_ipc_benchmark_ts"


def onFrameEnd(frame: int) -> None:
    """Called after all operators finish cooking each frame.

    This is the ideal timing point for producer timestamps since all
    TOPs (including Shared Mem Out TOP and CUDA IPC sender) have already
    written their data.
    """
    global shm, frame_counter

    # Create SharedMemory on first frame
    if shm is None:
        try:
            # Try opening existing segment first (consumer may have created it)
            shm = SharedMemory(name=TIMESTAMP_SHM_NAME)
        except FileNotFoundError:
            # Create new segment if it doesn't exist
            # Size: 4 bytes (uint32) + 8 bytes (float64) = 12 bytes (use 16 for alignment)
            shm = SharedMemory(name=TIMESTAMP_SHM_NAME, create=True, size=16)

    # Increment frame counter
    frame_counter += 1

    # Write timestamp: frame_counter (uint32) + timestamp (float64)
    # Use perf_counter() for high-resolution timing
    timestamp = time.perf_counter()
    struct.pack_into("<Id", shm.buf, 0, frame_counter, timestamp)


def onExit() -> None:
    """Called when TD closes or DAT is deleted.

    Cleanup SharedMemory to avoid stale segments.
    """
    global shm
    if shm is not None:
        # Note: We don't unlink() because consumer may still be reading
        # The OS will clean up when both processes close the segment
        with contextlib.suppress(OSError, BufferError):
            shm.close()
        shm = None
