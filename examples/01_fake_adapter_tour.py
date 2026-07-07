"""
Example 01 — API tour with FakeCUDAAdapter (no GPU, no TouchDesigner, one process).

WHAT
    Walks the complete cuda-link public API — Exporter, Importer, FrameSpec,
    ImportSpec, policies, GpuFrame, FrameOutcome, ImportOutcome — using the
    in-memory FakeCUDAAdapter, so it runs on ANY machine: no NVIDIA GPU, no
    driver, no TouchDesigner.

WHY START HERE
    Real CUDA IPC forbids opening a memory handle in the process that created
    it (cudaIpcOpenMemHandle → error 201 on Windows), so every *real* example
    needs two OS processes.  The fake adapter has no such restriction, which
    lets this tour show producer and consumer side by side in one script with
    zero concurrency noise.  The SHM ring-buffer protocol underneath is the
    REAL production code path — only the CUDA calls are simulated, so pixel
    values come back as zeros while outcomes, slot cycling, metadata, and
    shutdown behave exactly like production.

HOW TO RUN
    python examples/01_fake_adapter_tour.py
    python examples/01_fake_adapter_tour.py --frames 8

WHAT IT TEACHES
    * The hexagonal seam: Exporter/Importer talk to CUDA only through the
      CudaPort protocol — swap CTypesCUDAAdapter for FakeCUDAAdapter and the
      whole pipeline runs GPU-free (this is also how you unit-test YOUR code).
    * FrameSpec / ImportSpec: geometry + the SHM name as the only routing key.
    * The ring buffer: num_slots slots written round-robin by the producer.
    * Every FrameOutcome / ImportOutcome branch your code must handle.
    * Clean shutdown: exporter.close() → consumer sees ImportOutcome.SHUTDOWN.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make _common importable when this file runs as a script from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--frames", type=int, default=6, help="frames to publish/consume (default 6)")
    args = parser.parse_args()

    # Imports live inside main() so `--help` works even on a machine where an
    # optional dependency is broken (nothing heavy loads at module import).
    import numpy as np

    from cuda_link import (
        Exporter,
        ExportPolicy,
        FakeCUDAAdapter,
        FrameSpec,
        GpuFrame,
        Importer,
        ImportOutcome,
        ImportPolicy,
        ImportSpec,
    )

    # ------------------------------------------------------------------
    # Step 0 — adapters (the hexagonal seam)
    # ------------------------------------------------------------------
    # Exporter and Importer never call CUDA directly; they call a CudaPort.
    # FakeCUDAAdapter satisfies that protocol with in-memory bookkeeping:
    # malloc() hands out fake pointer integers, memcpy is a no-op, IPC events
    # are always "ready".  One adapter instance per side, like production.
    _common.banner("Step 0 — construct FakeCUDAAdapter for both sides")
    producer_cuda = FakeCUDAAdapter(device=0)
    consumer_cuda = FakeCUDAAdapter(device=0)
    print("Producer adapter:", producer_cuda)
    print("Consumer adapter:", consumer_cuda)

    # The one landmine: Importer.open() eagerly wraps ring-buffer pointers in
    # torch/cupy GPU arrays whenever those packages are importable — with fake
    # pointers that must not happen.  This flips the importer's module flags
    # off, exactly like the repo's own GPU-free tests do.
    for note in _common.make_importer_open_safe(using_fake=True):
        print("NOTE:", note)

    # ------------------------------------------------------------------
    # Step 1 — producer side: Exporter.open(FrameSpec)
    # ------------------------------------------------------------------
    # FrameSpec is the frame geometry plus SHM routing.  The shm_name is the
    # ONLY key linking producer and consumer — both sides must use the same
    # string.  num_slots sizes the ring buffer: the producer writes slots
    # round-robin so the consumer can read slot N-1 while slot N is written.
    _common.banner("Step 1 — Exporter.open(FrameSpec)")
    shm_name = _common.unique_shm_name("cudalink_ex01")
    spec = FrameSpec(
        shm_name=shm_name,
        height=64,
        width=64,
        channels=4,  # RGBA
        dtype="uint8",
        num_slots=3,
    )
    # ExportPolicy.for_testing() disables CUDA Graphs and the activation
    # barrier — both need a real driver.  With a real GPU you would pass no
    # policy at all (defaults) or ExportPolicy.from_env().
    exporter = Exporter.open(spec, policy=ExportPolicy.for_testing(), cuda=producer_cuda)
    print(f"Exporter open on shm '{shm_name}'")
    print(f"data_size = {exporter.data_size} bytes  (64 * 64 * 4 channels * 1 byte)")

    # ------------------------------------------------------------------
    # Step 2 — consumer side: Importer.open(ImportSpec)
    # ------------------------------------------------------------------
    # ImportSpec mirrors FrameSpec from the consumer's point of view.  Shape
    # and dtype are given explicitly here; example 05 shows shape=None
    # auto-detection from the producer's SHM metadata.
    _common.banner("Step 2 — Importer.open(ImportSpec)")
    importer = Importer.open(
        ImportSpec(shm_name=shm_name, shape=(64, 64, 4), dtype="uint8"),
        policy=ImportPolicy.for_testing(),  # no spin-wait, no reconnect delays
        cuda=consumer_cuda,
    )
    print(f"Importer connected: is_ready = {importer.is_ready()}")

    # ------------------------------------------------------------------
    # Step 3 — the frame loop: export() → get_frame_numpy()
    # ------------------------------------------------------------------
    # The producer hands export() a GpuFrame: a raw device pointer + byte
    # size.  cuda-link copies device-to-device into the current ring slot and
    # publishes it.  Here the "device buffer" is a fake allocation.
    _common.banner(f"Step 3 — publish and consume {args.frames} frames")
    src_ptr = producer_cuda.malloc(exporter.data_size)
    print(f"Fake device buffer at {int(src_ptr.value or 0):#x}\n")

    for i in range(args.frames):
        # -- producer -----------------------------------------------------
        # FrameOutcome tells you what happened; a healthy loop sees PUBLISHED.
        #   SKIPPED_BARRIER   — activation barrier held the frame back
        #   SKIPPED_NOT_READY — exporter not initialized / already closed
        #   FAILED            — CUDA error during the copy/publish
        outcome = exporter.export(GpuFrame(ptr=int(src_ptr.value or 0), size=exporter.data_size))
        slot = i % spec.num_slots  # ring cycles: 0, 1, 2, 0, 1, 2, ...
        print(f"frame {i}: export → {outcome.name:<10} (ring slot {slot})")

        # -- consumer -----------------------------------------------------
        # get_frame_numpy() is the CPU output mode (device-to-host copy in
        # production; a no-op here, so pixels read as zeros — the PROTOCOL is
        # what this tour demonstrates).  The zero-copy GPU mode, get_frame(),
        # needs real CUDA memory — example 03.  ImportResult is a frozen
        # dataclass with .outcome and .frame fields (not a tuple).
        result = importer.get_frame_numpy()
        if result.outcome is ImportOutcome.NEW_FRAME:
            frame = result.frame
            assert isinstance(frame, np.ndarray)
            print(f"         import → NEW_FRAME   shape={frame.shape} dtype={frame.dtype}")
        else:
            print(f"         import → {result.outcome.name}")

    # Polling again without a new export returns NO_FRAME — the consumer has
    # already seen the newest slot.  Your loop should treat NO_FRAME as "wait
    # briefly and poll again" (or block on the doorbell — example 08).
    result = importer.get_frame_numpy()
    print(f"\nextra poll without export → {result.outcome.name} (nothing new in the ring)")

    # ------------------------------------------------------------------
    # Step 4 — stats
    # ------------------------------------------------------------------
    _common.banner("Step 4 — get_stats() on both sides")
    for key, value in exporter.get_stats().items():
        print(f"exporter  {key:>24} = {value}")
    print()
    for key, value in importer.get_stats().items():
        print(f"importer  {key:>24} = {value}")

    # ------------------------------------------------------------------
    # Step 5 — clean shutdown
    # ------------------------------------------------------------------
    # exporter.close() writes a shutdown flag into SHM before releasing
    # resources.  The consumer's NEXT get_frame*() call reports SHUTDOWN and
    # closes the importer — this is how a TD consumer knows TD quit cleanly
    # (versus TIMEOUT, which means the producer vanished without notice).
    _common.banner("Step 5 — shutdown handshake")
    producer_cuda.free(src_ptr)
    exporter.close()
    print("exporter.close() called (shutdown flag published)")

    result = importer.get_frame_numpy()
    print(f"consumer poll after close → {result.outcome.name}")
    importer.close()

    # FakeCUDAAdapter tracks allocations — a leak check for free, exactly how
    # your own unit tests can assert cuda-link (and your code) cleaned up.
    leaked = len(producer_cuda.allocations) + len(consumer_cuda.allocations)
    print(f"leak check: {leaked} fake device allocations still open")

    _common.banner("Tour complete")
    print("Next: examples/02_two_process_roundtrip.py — the same flow with a REAL")
    print("GPU and the two-process spawn pattern that real CUDA IPC requires.")
    return 0 if leaked == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
