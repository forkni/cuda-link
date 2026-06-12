# ADR-0006: Stay pure-Python — do not rewrite in Rust

**Status**: Accepted
**Date**: 2026-06-07
**Applies to**: the whole library — `src/cuda_link/`, `td_exporter/`,
packaging (`pyproject.toml`), and the deployment model documented in
[ADR-0002](0002-byte-identical-td-mirror.md) and
[ADR-0003](0003-library-install-sys-path-bootstrap.md).

---

## Context

It was asked whether rewriting `cuda-link` — in whole or in part — in Rust
(via `cuda-oxide`, `cudarc`, or raw FFI) would improve **performance** and
**reliability**. Like the VMM-vs-legacy-IPC question in
[ADR-0004](0004-legacy-cuda-ipc-over-vmm.md), this is the kind of large
directional decision worth recording so it is not re-litigated from scratch.

The investigation found:

1. **Performance is hardware-bound, not Python-bound.** Per
   [`docs/BENCHMARKS.md`](../BENCHMARKS.md) (RTX 4090 / PCIe 4.0), end-to-end
   latency is dominated by the PCIe D2H copy (0.18–5.7 ms, 512²→4K) and the GPU
   D2D memcpy (22–367 µs); cross-process IPC notification is ~136–286 µs.
   Python-specific overhead (struct pack/unpack in `shm_protocol.py`, the
   event-poll spin in `importer.py`, SHM publish) is on the order of ~200 µs —
   roughly **4 %** of a typical frame. Rust cannot speed up the PCIe bus or GPU
   memory bandwidth, which is where the time actually goes.

2. **The producer side is locked to TouchDesigner's embedded CPython.**
   `td_exporter/` runs inside TD's bundled interpreter; it cannot be replaced by
   a Rust binary — at most a Rust `.pyd` that TD's Python imports. Only the
   standalone consumer (`src/cuda_link/`) is even a candidate for a native
   extension.

3. **A native extension forfeits the deployment story.** The library is a
   pure-Python, zero-required-dependency, ~30 KB platform-independent wheel with
   no build step, and a classic Text-DAT `.tox` mode. That model is load-bearing
   (ADR-0002, ADR-0003). Rust means a build toolchain, per-platform/per-CUDA
   compiled wheels, and breaks the byte-identical TD mirror.

4. **`cuda-oxide` specifically is the wrong tool.** It is an experimental
   Rust→CUDA *kernel compiler* (driver API), Linux-only (Ubuntu 24.04), alpha
   (v0.2.0), with no documented CUDA IPC support. cuda-link writes **no
   kernels**, is **Windows-only** (CUDA IPC mem/event handles are Windows-specific
   in this design), and is built **entirely around IPC** — three head-on
   mismatches. If Rust were ever pursued, `cudarc` (safe runtime bindings + IPC)
   or raw FFI would be the only viable choice.

5. **The one genuine Rust benefit is cheaper in Python.** Rust's compile-time
   safety would help the hand-written ctypes seam (struct layout, pointer
   widths, handle lifetimes), which is exactly where static typing was weakest.
   That gap was closed in Python instead — see
   [ADR-0005](0005-static-typing-hardening.md) (scoped type-checking + CI gate +
   Protocol drift guard) — at a fraction of the cost and with zero deployment
   impact.

## Decision

**Keep cuda-link pure Python. Do not rewrite the library or its consumer side
in Rust.** Address the reliability motivation through Python-side hardening
(ADR-0005), not a language change.

The only Rust option left on the table is a **narrow, optional PyO3 extension**
(over `cudarc` or raw FFI — never `cuda-oxide`) for a consumer-side hot path,
and only if profiling later proves Python overhead material. It is a
revisit-if, not a current direction.

## Rejected alternatives

- **Full rewrite in Rust.** Negligible performance upside (≈4 % of a frame is
  Python); forfeits the pure-Python wheel; impossible for the TD-embedded
  producer side.
- **Consumer-only (`src/cuda_link/`) rewrite.** Same deployment cost; splits the
  codebase and breaks the byte-identical TD mirror for shared modules.
- **`cuda-oxide` adoption.** Kernel compiler, Linux-only, alpha, no IPC — solves
  a problem this project does not have and lacks the one feature it is built on.
- **Optional PyO3 hot-path extension now.** Premature: no profiling evidence
  that the Python slice is a bottleneck. Retained only as a revisit-if.

## Consequences

- The pure-Python, zero-dependency, drop-into-TD distribution model is
  preserved.
- Reliability is improved where it was genuinely weak (the ctypes seam) via
  static-typing hardening rather than a rewrite.
- Future "should we use Rust?" explorations can start from this record and the
  benchmarks rather than re-deriving the conclusion.

## Reopen condition

Revisit if any of the following change the premises above:

- GPU↔GPU transfer stops being the bottleneck (e.g. NVLink / Grace-Hopper /
  unified memory makes the PCIe/D2D copy negligible), exposing Python overhead.
- Profiling shows the Python-bound share of a frame exceeds ~20 %.
- The project moves off Windows and/or off TouchDesigner's embedded CPython,
  removing the producer-side interpreter constraint.
