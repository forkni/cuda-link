# PLAN-002: Native notification waiter for the Python consumer

**Status**: Proposed
**Date**: 2026-07-04
**Size**: M (1–2 weeks, 4 phases)
**Depends on**: — (independent; do after PLAN-004)
**Related ADRs**: [ADR-0006](../adr/0006-stay-pure-python-no-rust.md) — this plan
exercises its escape-hatch clause ("narrow optional extension for a consumer-side hot
path"); [ADR-0007](../adr/0007-spout-as-launcher-not-transport.md) — not violated: the
extension loads in the **consumer's own Python process**, never inside TD.

---

## Goal & non-goals

**Goal**: consumer frame-arrival wake latency **< 10 µs typical** (baseline 136–286 µs
p50 cross-process), by moving the spin-then-block wait into one native call that engages
the existing Win32 doorbell.

**Non-goals**: touching the producer hot path, the wire protocol, or the D2H path
(ADR-0008). The core `cuda-link` wheel stays pure-Python.

## Baseline & why native is required here

Today's wait (`src/cuda_link/importer.py::_wait_for_slot`, ~line 1369) is two-phase:
spin on ctypes `cudaEventQuery` for `CUDALINK_WAIT_SPIN_US` (default 200 µs), then a
`time.sleep(0.0001)` loop whose real floor is ~1 ms (`timeBeginPeriod`). Each Python
poll iteration costs ~µs (ctypes call ~0.3–1.5 µs + interpreter dispatch), so the spin
phase detects with multi-µs granularity and the sleep phase with ~ms granularity.

The doorbell (`src/cuda_link/_doorbell.py`, named auto-reset event
`Local\cudalink_db_<shm_name>`, `CUDALINK_DOORBELL=1`, default off) already tightens
p95 ~10× (CHANGELOG 1.11.0) but is a primitive for receiver loops — it is not wired
into `get_frame()`'s wait, and the Python loop cost remains.

Windows kernel wake primitives cost single-digit µs. The gap to 136–286 µs is
Python-side loop granularity — the one place ctypes-vs-C++ genuinely matters, because a
native spin iteration is tens of ns. (`WaitOnAddress` is intra-process only, so the
cross-process primitive remains the named event we already have.)

## Architecture decisions

### D1 — Separate package `native/` (`cuda-link-native`), cloning the `spout/` pattern

scikit-build-core + pybind11 ≥ 2.12 + CMake ≥ 3.24, C++17; native module
`_native_waiter`; pure-Python wheel fallback when not `WIN32`; `_native.py` lazy loader
with `os.add_dll_directory`; `_backend.py` `Protocol` seam + `FakeWaitBackend` for
GPU-free CI; separate `native-tests` CI job with coverage gate. Consumers opt in via
`pip install cuda-link[native]` (extra pointing at `cuda-link-native`).

*Rejected: optional extension inside the core wheel — turns every core release into a
cibuildwheel matrix and forfeits ADR-0006's headline "pure-Python zero-dep wheel"
property. The escape hatch says narrow and optional; a sidecar wheel expresses that.*

### D2 — One native call per frame: hybrid spin-then-block, GIL released

```text
wait_slot(event_ptr, doorbell_handle, write_idx_addr, last_write_idx,
          spin_us, timeout_ms) -> WaitResult{status, waited_us, method}
```

1. `pybind11::gil_scoped_release` for the whole call (all args passed by value first).
2. **Spin phase**: tight loop on `cudaEventQuery(event)` + `QueryPerformanceCounter`
   until `spin_us` — native iterations are tens of ns, so the event is caught within
   ~1 µs of the GPU signal.
3. **Block phase**: loop { volatile read of `write_idx` (lost-wakeup guard, same
   pre-check as `wait_for_doorbell`) → `cudaEventQuery` →
   `WaitForSingleObject(doorbell, slice_ms)` } until ready or deadline.
4. Status enum `READY_SPIN | READY_DOORBELL | READY_LATE | TIMEOUT`, mapped onto the
   existing `wait_spin_hits` / `wait_sleep_hits` counters and
   `_adaptive_observe_wait` bookkeeping.

### D3 — cudart access via `GetModuleHandleW` + `GetProcAddress` (no second runtime)

The module never links or loads cudart. At init it resolves `cudaEventQuery` from the
**already-loaded** module — probing `cudart64_13.dll`, then `cudart64_12.dll`, then the
`cudart64_11.dll`/`cudart64_110.dll` fallbacks, the same order `cuda_ipc_wrapper.py` uses
(it deliberately probes CUDA 13 before 12 so a torch built against CUDA 13 shares its
already-resident runtime); if none is loaded it refuses to activate and the Python path
is used. Same DLL instance ⇒ same runtime state ⇒ zero double-context risk,
and no CUDA toolkit is needed at build time (only `windows.h` + a local typedef).

### D4 — Seam: `ImportPolicy.wait_backend`

`ImportPolicy.wait_backend: "auto" | "python" | "native" = "auto"` (env
`CUDALINK_WAIT_BACKEND`), resolved once at connect in `_open_ipc_slots`: `auto` →
native iff `os.name == "nt"` and `cuda_link_native` imports and cudart resolves and a
doorbell handle opened. A single branch at the top of `_wait_for_slot`: if a backend
object is present, call `backend.wait_slot(...)`, translate the result into existing
counters, return; otherwise fall through to the current code **unchanged**. Every
failure degrades silently to `python` with one INFO line. The
`torch_gpu_wait` / `cupy` GPU-wait paths remain upstream and unaffected.

With the native backend active, `CUDALINK_DOORBELL` defaults to **on** (producer-side
cost is one `SetEvent` ≈ sub-µs per publish; the exporter already rings it when enabled,
`exporter.py` publish path).

### D5 — Testing without GPU/Windows

`WaitBackend` Protocol + `FakeWaitBackend` (scripted: ready-after-N-calls, timeout,
late) run the existing importer suite on Ubuntu; a loader test verifies the
graceful-degradation matrix (no module / no cudart / no doorbell / non-Windows). A
marked native smoke test (Windows + GPU, manual) mirrors spout's
`test_native_smoke.py`. The C++ spin/block state machine takes injected function
pointers so a fake "event" function unit-tests it without CUDA.

## Phases

1. **Scaffold (S)** — `native/` package copied from `spout/` (pyproject, CMake, loader,
   backend Protocol, fakes, CI job). *Exit*: pure wheel builds everywhere; native wheel
   builds on a Windows runner.
2. **Waiter (M)** — `_cpp/native_waiter.cpp` per D2/D3; state machine unit-tested via
   injected function pointers. *Exit*: C++ tests green; smoke test on a GPU box.
3. **Seam (S)** — `ImportPolicy` field + backend resolution + `_wait_for_slot` branch +
   doorbell default flip when native active. *Exit*: full importer suite green with
   `FakeWaitBackend` on Ubuntu; `CUDALINK_WAIT_BACKEND=python` bit-identical to today.
4. **Bench + docs (S)** — extend the notification-latency harness
   (`scripts/profiling/bench_doorbell.py` / `bench_r1_wait.py`) with backend comparison;
   BENCHMARKS.md table; CHANGELOG; note in ADR-0006's consequences that the escape hatch
   was exercised.

## Verification & acceptance

- Notification latency at 60 fps producer, 1080p: **p50 < 10 µs, p95 < 50 µs**
  (baselines: poll-sleep 136–286 µs; doorbell-only per CHANGELOG 1.11.0 — native must
  beat or match doorbell-only while removing the Python loop cost).
- CPU: consumer core utilization not worse than the current `wait_spin_us=200` config.
- Fallback: uninstalling `cuda-link-native` or `CUDALINK_WAIT_BACKEND=python`
  reproduces current behavior exactly (existing tests are the proof).

## Risk register

| Risk | Sev | Mitigation |
|---|---|---|
| cudart not yet loaded when the backend initializes | Med | Resolve at connect, after `CUDARuntimeAPI` load; refuse + fallback otherwise |
| Auto-reset event is single-consumer (second consumer starves) | Med | Already documented in `_doorbell.py`; native path pre-checks `write_idx` + event before blocking, so a missed signal costs one `slice_ms`, not a hang |
| GIL released while touching Python-owned memory | Low | All args passed by value (ints) before release |
| Windows-only wheel matrix maintenance | Low | Exact `spout/` precedent already maintained |
| Interaction with PLAN-003's cuda-bindings backend (different cudart DLL name) | Low | Resolver taught the wheel's DLL name, else python-wait fallback (see PLAN-003 D2) |
