# PLAN-003: Optional `cuda.bindings` adapter beside `CTypesCUDAAdapter`

**Status**: Proposed
**Date**: 2026-07-04
**Size**: M (1–2 weeks, 4 phases) — **benchmark-gated with an explicit kill criterion**
**Depends on**: — (independent; note D2 interplay with PLAN-002)
**Related ADRs**: [ADR-0001](../adr/0001-port-adapter-deepening.md) (the adapter seam
this slots into), [ADR-0005](../adr/0005-static-typing-hardening.md) (the Protocol
artifact is a typing win), [ADR-0006](../adr/0006-stay-pure-python-no-rust.md) (outcome
recorded in its consequences either way)

---

## Goal & non-goals

**Goal**: cut per-call FFI overhead on the export/import hot path (~8–12 cudart calls
per frame) by offering NVIDIA's Cython-based
[`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/) (part of
cuda-python) as an **optional alternative adapter** beside the ctypes one.

**Expected win is modest and bounded**: ctypes costs ~0.3–1.5 µs per call vs ~100–200 ns
for Cython bindings; CUDA Graphs already collapsed the export path to ~1 submission per
frame with a ~2.5 µs native launch floor. Realistic saving: **~5–10 µs/frame at 1080p**
(meaningful at 512×512 where export is ~14 µs; <10% at 4K). Because the win may not
justify the maintenance, **"rejected, with the numbers recorded" is a valid outcome** of
this plan.

**Non-goals**: making cuda-python a required dependency (core stays zero-dep); changing
the TD side (`td_exporter/` is excluded — TD's embedded Python cannot be assumed to have
the wheel; ADR-0002 mirror stays pure-ctypes); touching D2H (ADR-0008).

## Architecture decisions

### D1 — Full-surface adapter, one backend per process, never mixed

New module `src/cuda_link/_cuda_bindings_adapter.py`, class `CudaBindingsAdapter`,
implementing the **entire** method-name surface that exporter/importer/graphs code calls
on the adapter. Because `CTypesCUDAAdapter.__getattr__` delegates dynamically
(`src/cuda_link/_cuda_adapters.py`), the true contract is "names used by callers" — so
the plan's **first artifact** is that explicit inventory, promoted to a
`CUDAAdapterProtocol` in `_cuda_adapters.py` that both adapters and the test fakes are
type-checked against (pyrefly/CI, per ADR-0005).

Backend chosen once at init via `CUDALINK_CUDA_BACKEND=ctypes|cuda-bindings` (default
`ctypes`), never call-by-call.

*Rejected: hot-call-overlay hybrid (override only `graph_launch`/`memcpy_async` on top
of ctypes) — mixes two cudart instances mid-frame and doubles the parity-test surface
for marginal scoping benefit.*

### D2 — Double-cudart posture: all-or-nothing

`cuda.bindings.runtime` loads its own cudart (via `nvidia-cuda-runtime` wheels),
distinct from the `cudart64_1x.dll` ctypes loads. Handles are driver-level and both
instances share the device primary context, so cross-instance handles *generally* work —
but this plan treats mixing as **unsupported**: with `cuda-bindings` selected,
`CUDARuntimeAPI` (ctypes) is never instantiated in that process (init branches before
DLL load).

One deliberate exception to verify in Phase 1: PLAN-002's native waiter resolves
`cudaEventQuery` from whatever cudart module is loaded — identify which DLL
`cuda.bindings` loads and teach the waiter's resolver its name, or fall back to
python-wait under this backend.

### D3 — Error-semantics parity is the highest-risk area

`cuda.bindings` returns `(err, value)` tuples; the adapter converts to the exact
exception types/messages the ctypes `errcheck` raises (code + name), preserving the
NOT_READY sentinel behavior of `query_event`/`stream_query` and the
`check_sticky_error`/`cudaPeekAtLastError` semantics. Dedicated test module with fault
injection (bad device, freed pointer).

### D4 — Packaging

pyproject extra `cuda-bindings = ["cuda-python>=12.6"]` in
`[project.optional-dependencies]`; import guarded; loader test proves a plain
`pip install cuda-link` never triggers the import. `scripts/sync_td_wrapper.py` never
mirrors the new module.

### D5 — Migration order inside the adapter (review sequencing, not runtime mixing)

1. init / device / context + error surface
2. stream/event lifecycle (`create_stream*`, `record_event[_with_flags]`,
   `query_event`, `stream_wait_event`, `stream_query`, `stream_synchronize`)
3. memory (`malloc/free`, `memcpy_async`, pinned-host alloc/register,
   `pointer_get_attributes`)
4. IPC (`ipc_get_mem_handle` / `open` / `close`, event handles) — the
   `cudaIpcMemHandle_t` **byte layout must round-trip identically** into the 128-byte
   SHM slots: a golden test packs a handle via both backends and compares bytes
5. CUDA Graphs (relaxed capture, `cudaEventRecordWithFlags` EXTERNAL,
   `graph_exec_memcpy_node_set_params_1d`, instantiate-with-flags ABI) — trickiest
   mapping, done last

## Phases

1. **Inventory (S)** — adapter-surface inventory → `CUDAAdapterProtocol`; type-check
   both existing adapters + fakes against it; identify cuda-bindings' cudart DLL (D2).
2. **Core groups 1–3 (M)** — `CudaBindingsAdapter` + error-parity tests. GPU-free
   proof: the importer/exporter suites already run against fakes, proving callers are
   backend-agnostic. GPU proof: `verification/verify_backend_parity.py` runs one full
   export → import round trip per backend and diffs frame bytes, stats, and
   fault-injection behavior.
3. **IPC + Graphs (M)** — groups 4–5, incl. the handle-byte golden test; wire backend
   selection into exporter/importer init.
4. **Benchmark gate (S)** — existing benchmarks at 1080p/4K, export path + importer
   wait, 3 runs per backend.
   **Accept** if hot-path median improves ≥ 5 µs at 1080p with no p95 regression.
   **Kill** otherwise: don't land (or land behind an `experimental` flag), record the
   numbers in `docs/BENCHMARKS.md` and a consequence note in ADR-0006.

## Verification

- Ubuntu CI unchanged (fakes); new marked GPU parity script in `verification/`.
- Loader test: without the extra installed, `_cuda_bindings_adapter` import is never
  triggered.
- The benchmark table is committed **either way** — the number is the deliverable.

## Risk register

| Risk | Sev | Mitigation |
|---|---|---|
| Two cudart instances in one process | High | All-or-nothing backend (D1/D2); ctypes DLL never loaded under cuda-bindings |
| Error-semantics drift (tuples vs errcheck raise, sticky errors) | High | Dedicated parity module (D3), GPU fault injection |
| Graph API mapping mismatch (node param structs, instantiate ABI) | Med | Done last (D5); kill criterion caps sunk cost |
| Win too small to justify maintenance | Med (likely) | Explicit ≥ 5 µs gate; documented rejection is a valid outcome |
| PLAN-002 waiter can't resolve cudart under this backend | Low | Teach resolver the wheel's DLL name, else python-wait fallback |
