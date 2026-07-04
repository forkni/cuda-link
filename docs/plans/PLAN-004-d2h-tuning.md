# PLAN-004: D2H — skip native work, tune configuration instead

**Status**: Proposed
**Date**: 2026-07-04
**Size**: S (days, 1 phase) — **do this plan first**
**Depends on**: —
**Related ADRs**: [ADR-0008](../adr/0008-skip-native-d2h.md) (this plan's decision
record, written first), [ADR-0006](../adr/0006-stay-pure-python-no-rust.md)

---

## Goal & non-goals

**Goal**: (a) record, as ADR-0008, the decision that **no native/C++/binding work
targets the D2H readback path** — it is PCIe-bandwidth-bound and language cannot help;
(b) extract the remaining non-native headroom via configuration: pinned-memory
defaults, stream count, and copy/compute overlap.

**Non-goals**: any native extension surface; changing `get_frame()`/`get_frame_cupy()`
zero-copy GPU paths (already < 5 µs).

## Evidence (why native cannot help)

From `docs/BENCHMARKS.md` (RTX 4090, PCIe 4.0 x16):

| Operation | p50 | Effective bandwidth |
|---|---|---|
| `get_frame_numpy()` D2H 512×512 f32 | 0.18 ms | ~22 GB/s |
| `get_frame_numpy()` D2H 1080p f32 | 1.32 ms | ~24 GB/s |
| `get_frame_numpy()` D2H 4K f32 | 5.7 ms | ~21 GB/s |

Practical PCIe 4.0 x16 ceiling is ~26 GB/s — the path already runs within a few percent
of the bus. Per-call ctypes overhead is 0.3–1.5 µs; even a dozen calls is < 0.1% of a
1080p transfer. This extends ADR-0006's cost model *a fortiori*: where the bus, not the
CPU, is the bottleneck, rewriting the caller changes nothing.

## Existing knobs (already implemented — tune, don't rebuild)

- `CUDALINK_D2H_STREAMS` / `CUDALINK_D2H_STREAM_PRIO` — multi-stream chunked D2H
- `CUDALINK_D2H_PIPELINED` — double-buffered pipelining
- Pinned host memory (`cudaMallocHost` / `cudaHostRegister` in
  `cuda_ipc_wrapper.py`) with `CUDALINK_ALLOW_PAGEABLE_FALLBACK`

## Experiments

Each experiment: hypothesis → benchmark command → accept/record. Run on the existing
D2H harness; results land in `docs/BENCHMARKS.md`; default changes land in
`src/cuda_link/_env.py` / `_importer_port.py`.

1. **Pinned-by-default audit** — the default is already fail-loud:
   `allow_pageable_fallback` defaults to `False` and the pinned-alloc failure path
   *raises* (`importer.py` ~L487) rather than silently degrading; when opted in
   (`=True`) the `cudaHostRegister`→pageable path already logs at WARNING
   (`importer.py` ~L493/L507). So this item does **not** flip a default or ship a
   breaking change. Instead: (a) enrich that opt-in WARNING with a *measured*
   throughput delta (it currently says "~2x" qualitatively); (b) confirm no other
   path (e.g. `malloc_host` in `cuda_ipc_wrapper.py`) engages pageable memory
   without a WARNING; (c) record the fail-loud default in BENCHMARKS.md so it isn't
   re-proposed.
2. **Stream-count sweep** — `CUDALINK_D2H_STREAMS ∈ {1,2,3,4}` ×
   {720p, 1080p, 4K} × {uint8, float16, float32}; expect diminishing returns past 2 on
   one PCIe link; set the measured best as default.
3. **Pipelined chunking sensitivity** — with `CUDALINK_D2H_PIPELINED=1`, measure
   chunk-size sensitivity (or document the implicit split policy); goal: ≥ 80% of the
   copy overlapped behind consumer compute at 4K.
4. **`cudaHostAllocWriteCombined` one-shot** — default vs write-combined pinned memory;
   expect WC to *hurt* D2H (WC is H2D-oriented; the consumer reads the buffer).
   Measure once, document so nobody re-asks.
5. **Overlap check** — confirm the consumer's post-D2H CPU work overlaps the next
   frame's copy (NVTX ranges via `_nvtx.py` + Nsight Systems capture; addendum to
   `docs/PROFILING.md`).
6. **Defaults + docs PR** — one PR adjusting `_env.py` defaults per findings; a "D2H
   tuning" recommendation matrix in `docs/BENCHMARKS.md`; cross-link from ADR-0008.

## Verification

Re-run the full D2H benchmark suite before/after the defaults change. **Accept**: ≥
baseline throughput at every point in the matrix (no regressions); wins expected only
where pageable fallback or stream count was suboptimal.

## Risks

Essentially none — docs + env defaults. The only guarded change is fail-loud pageable
fallback (breaking-change note if flipped).
