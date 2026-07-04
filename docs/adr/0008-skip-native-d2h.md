# ADR-0008: No native work on the D2H readback path

**Status**: Accepted
**Date**: 2026-07-04
**Applies to**: the D2H readback path (`importer.py` numpy backend, pinned-host
machinery, `CUDALINK_D2H_*` knobs).

---

## Context

A 2026-07 evaluation of extending cuda-link with C++ (see `docs/plans/`) split the
per-frame cost into four surfaces: the TD-side sender (PLAN-001), the consumer wake
path (PLAN-002), FFI call overhead (PLAN-003), and D2H readback. The first three have
CPU-side headroom; D2H does not:

- Measured D2H throughput (`docs/BENCHMARKS.md`, RTX 4090 / PCIe 4.0 x16):
  0.18–5.7 ms at **~21–24 GB/s**, within a few percent of the ~26 GB/s practical
  PCIe 4.0 x16 ceiling.
- Per-call ctypes overhead is 0.3–1.5 µs. A dozen calls per frame is < 0.1% of a 1080p
  transfer (1.32 ms). No binding technology (nanobind, Cython, `cuda.bindings`, raw C)
  changes bus bandwidth.
- ADR-0006 already established that cuda-link's performance is hardware-bound and the
  Python share is small; on D2H that argument holds *a fortiori* — the bottleneck is
  the bus, not the interpreter.

## Decision

**No native extension, and no `cuda.bindings`/Rust/C++ work, targets the D2H path.**
The optimization budget for D2H is configuration only: pinned-memory defaults, stream
count, and copy/compute overlap (executed as
[PLAN-004](../plans/PLAN-004-d2h-tuning.md)).

This decision also scopes PLAN-001/002/003: their native or binding work must not grow
D2H-facing surface.

## Rejected alternatives

- **Native pipelined copier** — the copy is bus-bound; a C++ driver of the same
  `cudaMemcpyAsync` calls moves the same bytes at the same speed.
- **GPUDirect / compression (nvcomp-style) tricks** — out of scope for a texture-sharing
  library; consumers who need less data should transfer smaller/packed formats.
- **Moving D2H into the C++ Custom TOP** — consumers are Python processes by design;
  the TD plugin never performs D2H.

## Consequences

- D2H stays PCIe-bound and predictable; tuning is env-var-level
  (`CUDALINK_D2H_STREAMS`, `CUDALINK_D2H_PIPELINED`, pinned-memory policy).
- Small frames (< ~256 KB), where fixed per-call overhead *does* dominate the transfer,
  are accepted as-is — users needing lower small-frame latency should stay on the
  zero-copy GPU paths (`get_frame()` / `get_frame_cupy()`, < 5 µs).
- PLAN-004's benchmark matrix becomes the canonical D2H tuning reference in
  `docs/BENCHMARKS.md`.

## Reopen condition

Revisit if any of:

- PCIe 5.0-class hardware shows a > 10% gap between measured D2H throughput and the bus
  ceiling (would indicate a software bottleneck appeared);
- a supported workload profile becomes dominated by sub-64 KB frames;
- profiling attributes > 5% of D2H wall time to Python/ctypes dispatch.
