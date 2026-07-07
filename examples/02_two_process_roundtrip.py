"""
Example 02 — real two-process roundtrip with pixel verification.

WHAT
    A producer process exports real GPU frames; THIS process imports them and
    verifies the pixel values crossed the process boundary intact.  This is
    the smallest complete demonstration of what cuda-link actually does:
    zero-copy GPU memory shared between two OS processes via CUDA IPC.

HARDWARE
    NVIDIA GPU + driver (any CUDA-capable card).  numpy required.
    No TouchDesigner needed — the producer here stands in for TD.
    Pass --force-fake to run the same choreography without a GPU
    (pixel verification is skipped; the fake adapter moves no real data).

HOW TO RUN
    python examples/02_two_process_roundtrip.py
    python examples/02_two_process_roundtrip.py --frames 30 --width 1280 --height 720
    python examples/02_two_process_roundtrip.py --force-fake

WHAT IT TEACHES
    * WHY two processes are mandatory: cudaIpcOpenMemHandle fails (error 201
      on Windows) in the process that created the handle.  Every real
      cuda-link pipeline is producer-process + consumer-process.
    * THE spawn pattern you must copy into your own integration:
        - multiprocessing.get_context("spawn")  (fork would inherit a CUDA
          context and break IPC; spawn is also the only Windows option)
        - worker functions at MODULE level so spawn can pickle them
        - a Queue for child → parent status reporting
        - _common.wait_for_shm(): poll cheaply until the producer's ring
          buffer exists before touching CUDA in the consumer
    * The consumer loop with the full ImportOutcome switch — the template
      every later example builds on.
"""

from __future__ import annotations

import argparse
import ctypes
import multiprocessing
import sys
import time
from pathlib import Path

# Make _common importable when this file runs as a script from any cwd.  The
# spawn child re-imports this module by path, so this must run unconditionally
# at module top — not inside `if __name__ == "__main__"`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common

# ---------------------------------------------------------------------------
# Producer worker — MODULE level, because spawn pickles a reference to it.
# A nested function or lambda here would raise "Can't pickle local object".
# ---------------------------------------------------------------------------


def _producer_worker(
    shm_name: str,
    height: int,
    width: int,
    channels: int,
    num_frames: int,
    fps: float,
    force_fake: bool,
    rq: object,
) -> None:
    """Export num_frames, each filled with a distinct pixel value (frame index).

    Runs in its own OS process.  In your integration this role is played by
    TouchDesigner (or your own renderer) — the code is identical either way:
    open an Exporter, hand it a device pointer per frame, close.
    """
    try:
        import numpy as np

        from cuda_link import Exporter, ExportPolicy, FrameSpec, GpuFrame

        adapter, is_real = _common.pick_cuda_adapter(0, force_fake=force_fake)
        # Real GPU → production defaults (CUDA Graphs on, activation barrier
        # on).  Fake → for_testing() turns off everything driver-dependent.
        policy = None if is_real else ExportPolicy.for_testing()

        exporter = Exporter.open(
            FrameSpec(shm_name=shm_name, height=height, width=width, channels=channels, dtype="uint8"),
            policy=policy,
            cuda=adapter,
        )

        # One device buffer, reused for every frame — cuda-link copies it
        # device-to-device into the ring slot, so the producer may overwrite
        # it as soon as export() returns.
        data_size = exporter.data_size
        src_ptr = adapter.malloc(data_size)
        staging = np.empty((height, width, channels), dtype=np.uint8)

        sent = 0
        interval = 1.0 / fps if fps > 0 else 0.0
        for i in range(num_frames):
            if is_real:
                # Fill the whole frame with the frame index (mod 256) so the
                # consumer can verify exactly which frame it received.
                staging[...] = i % 256
                adapter.memcpy(
                    dst=src_ptr,
                    src=ctypes.c_void_p(staging.ctypes.data),
                    count=data_size,
                    kind=1,  # cudaMemcpyHostToDevice
                )
            # ORDERING: the H2D copy above is synchronous, so the source
            # buffer is complete — record the sync point on the default
            # stream (0) so export()'s D2D copy is ordered after it.
            # Producers writing on their own stream pass THAT stream's
            # handle instead — example 07 tells the full story.  Armed on
            # the fake too (its event ops are no-ops), which keeps
            # --force-fake runs free of the un-ordered-export warning.
            exporter.record_source_sync(0)
            outcome = exporter.export(GpuFrame(ptr=int(src_ptr.value or 0), size=data_size))
            if outcome.name == "PUBLISHED":
                sent += 1
            if interval:
                time.sleep(interval)

        time.sleep(1.0)  # grace: let the consumer drain the last frame
        adapter.free(src_ptr)
        exporter.close()  # publishes the shutdown flag the consumer will see
        rq.put(("OK", sent))  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001 — report ANY child failure to the parent
        import traceback

        rq.put(("ERROR", f"{e}\n{traceback.format_exc()}"))  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Consumer — runs right here in the main process, like your real application
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--frames", type=int, default=60, help="frames the consumer should receive (default 60)")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=60.0, help="producer frame rate (default 60)")
    parser.add_argument("--force-fake", action="store_true", help="run without a GPU (FakeCUDAAdapter)")
    args = parser.parse_args()

    import numpy as np

    from cuda_link import Importer, ImportOutcome, ImportPolicy, ImportSpec

    shape = (args.height, args.width, 4)
    shm_name = _common.unique_shm_name("cudalink_ex02")

    # ------------------------------------------------------------------
    # Launch the producer as a separate OS process (the spawn pattern)
    # ------------------------------------------------------------------
    _common.banner("Spawn producer process")
    ctx = multiprocessing.get_context("spawn")
    rq = ctx.Queue()  # child → parent status channel
    producer = ctx.Process(
        target=_producer_worker,
        # Producer sends a ~2s surplus beyond the consumer's target so the
        # consumer never starves waiting for its last frame.
        args=(shm_name, args.height, args.width, 4, args.frames + 120, args.fps, args.force_fake, rq),
        daemon=True,  # never leave an orphan producer if we crash
    )
    producer.start()
    print(f"producer pid={producer.pid}  shm='{shm_name}'")

    # The producer needs ~1-2 s to load CUDA and allocate the ring buffer.
    # Poll for its SharedMemory instead of sleeping blind — and do it BEFORE
    # any CUDA call in this process.
    if not _common.wait_for_shm(shm_name, timeout_s=20.0):
        print("ERROR: producer SharedMemory never appeared — see producer error below.")
        producer.join(timeout=5.0)
        while not rq.empty():
            print(rq.get_nowait())
        return 1
    time.sleep(0.3)  # settle: let the producer finish writing IPC handles

    # ------------------------------------------------------------------
    # Open the Importer (consumer side — this is YOUR program's code)
    # ------------------------------------------------------------------
    _common.banner("Open Importer and consume frames")
    adapter, is_real = _common.pick_cuda_adapter(0, force_fake=args.force_fake)
    # Preflight the torch/cupy eager-buffer landmine before open() — a
    # CPU-only torch install would otherwise crash Importer.open() here.
    for note in _common.make_importer_open_safe(using_fake=not is_real):
        print("NOTE:", note)

    importer = Importer.open(
        ImportSpec(shm_name=shm_name, shape=shape, dtype="uint8"),
        policy=None if is_real else ImportPolicy.for_testing(),
        cuda=adapter,
    )

    # The consumer loop: poll get_frame_numpy() and branch on EVERY outcome.
    # This switch is the core integration template — copy it.  numpy here
    # because the PROOF needs pixels on the CPU; a production GPU pipeline
    # uses zero-copy get_frame() instead (example 03) — same loop, same
    # outcomes, no per-frame device-to-host copy.
    received: list[int] = []
    mismatches = 0
    errors: list[str] = []
    deadline = time.perf_counter() + 30.0
    while len(received) < args.frames and time.perf_counter() < deadline:
        result = importer.get_frame_numpy()

        if result.outcome is ImportOutcome.NEW_FRAME:
            frame = result.frame
            assert isinstance(frame, np.ndarray)
            # Verify: the producer filled the whole frame with one value, so
            # min == max proves the D2D copy + IPC + D2H copy were lossless.
            lo, hi = int(frame.min()), int(frame.max())
            if lo != hi:
                mismatches += 1
            received.append(lo)
            if len(received) <= 5 or len(received) % 20 == 0:
                print(f"frame {len(received):>3}: shape={frame.shape} fill={lo}" + (" MIXED!" if lo != hi else ""))

        elif result.outcome is ImportOutcome.NO_FRAME:
            # Nothing new in the ring yet — yield briefly and poll again.
            # (Example 08 replaces this sleep with a doorbell block.)
            time.sleep(0.001)

        elif result.outcome is ImportOutcome.RECONNECTING:
            # Producer restarted; the importer reopened IPC handles itself.
            continue

        elif result.outcome is ImportOutcome.SHUTDOWN:
            print("producer shut down cleanly (shutdown flag) — stopping.")
            break

        elif result.outcome is ImportOutcome.TIMEOUT:
            errors.append("TIMEOUT (no producer event within ImportSpec.timeout_ms) — producer hung or died uncleanly")
            break

    importer.close()

    # ------------------------------------------------------------------
    # Join the producer and report
    # ------------------------------------------------------------------
    _common.banner("Results")
    producer.join(timeout=15.0)
    if producer.is_alive():
        producer.terminate()
        errors.append("producer did not exit in time — terminated")
    while not rq.empty():
        status, payload = rq.get_nowait()
        if status == "ERROR":
            errors.append(f"producer: {payload}")
        else:
            print(f"producer reports {payload} frames published")

    distinct = sorted(set(received))
    print(f"consumer received {len(received)} frames, {len(distinct)} distinct fill values")
    if received:
        print(f"fill values seen: {distinct[:10]}{' ...' if len(distinct) > 10 else ''}")

    for e in errors:
        print("ERROR:", e)

    if errors or mismatches:
        return 1
    if not received:
        print("FAILED: no frames received")
        return 1
    if is_real and len(distinct) < 2:
        # Real pixels should ANIMATE (fill value = frame index); a single
        # value would mean we kept re-reading one stale slot.
        print("FAILED: pixel values never changed — data did not flow")
        return 1
    if not is_real:
        print("fake mode: protocol verified (pixel values are not real in fake mode)")
    else:
        print("SUCCESS: real GPU pixels crossed the process boundary losslessly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
