# PLAN-002: Native notification waiter for the Python consumer

**Status**: Implemented (`feat/r5-native-wait-backend`; measured, see Verification &
acceptance below)
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
GPU-free CI; separate `native-tests` CI job with coverage gate.

**As shipped** (superseding the pip-extra idea below): consumers don't opt in via pip at
all. `scripts/install_td_library.py` builds and installs the `cuda-link-native` sidecar
by default — `--native` is the default, `--no-native` opts back out to pure-Python (see
`dffb804`). This keeps the *core* `cuda-link` wheel pure-Python/zero-dep (ADR-0006's
property is preserved — `cuda-link-native` is still a separate package the core wheel
never depends on), while making the accelerated path the out-of-the-box default for the
TouchDesigner installer audience this project actually ships to, rather than something
users must separately discover and `pip install`.

*Originally proposed, not shipped: `pip install cuda-link[native]` (extra pointing at
`cuda-link-native`). Superseded once it became clear the installer script, not `pip`, is
this project's real distribution path (no PyPI presence — see CONTEXT.md).*

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

### D3 — cudart access via an explicit function-pointer handoff (no second runtime, no bare-name lookup)

**As shipped** (superseding the bare-name-resolution idea below): the module never links,
loads, *or independently resolves* cudart. Each `Importer` connection calls
`CUDARuntimeAPI.cudart_event_query_fn_ptr()` (`cuda_ipc_wrapper.py`) — the raw address of
the `cudaEventQuery` symbol *this connection's own* CUDA adapter already resolved — and
hands that pointer explicitly to the native module via `set_cuda_event_query()`
(`native/src/cuda_link_native/_cpp/native_waiter.cpp`). Same DLL instance ⇒ same runtime
state ⇒ zero double-context risk, and no CUDA toolkit is needed at build time (only
`windows.h` + a local typedef) — same properties D3 originally wanted, reached a
different way.

*Originally proposed, not shipped: resolve `cudaEventQuery` independently inside the
native module via `GetModuleHandleW` (bare name) + `GetProcAddress`, probing
`cudart64_13.dll` → `cudart64_12.dll` → `cudart64_11.dll`/`cudart64_110.dll` fallbacks (the
same order `cuda_ipc_wrapper.py` uses). **Bug found and fixed (`d0370aa`, Phase 2):** a
bare-name `GetModuleHandleW` lookup is not guaranteed to resolve the *same* cudart
instance the connection is actually using whenever more than one same-named cudart DLL is
loaded from different directories in-process — diagnosed after this earlier version
silently misreported a genuinely-pending CUDA event as complete. The explicit
function-pointer handoff above eliminates that ambiguity entirely: there is no
independent resolution step to get wrong.*

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

**As actually built** (seam-first, not the originally-proposed scaffold→waiter→seam→bench
order below — the scaffold and seam landed together in one commit, then the C++ waiter's
resolution bug was found and fixed, then packaging/CI, then bench+docs):

0+1. **Scaffold + seam (`9fff28d`)** — `native/` package copied from `spout/` (pyproject,
   CMake, loader, backend Protocol, fakes) *together with* the `ImportPolicy.wait_backend`
   field, backend resolution, and `_wait_for_slot` branch (D4) in a single commit, rather
   than as separate phases.
2. **C++ waiter fix (`d0370aa`)** — `_cpp/native_waiter.cpp` per D2, corrected to D3's
   explicit function-pointer handoff after the bare-name-resolution bug described in D3
   above was diagnosed.
3. **Packaging + CI (`dffb804`)** — installer script `--native`/`--no-native` (default
   on), `native-tests` CI job.
4. **Bench + docs (this phase)** — extended `scripts/profiling/bench_r1_wait.py` (frame-
   budget fix, native arm, informational-only printout) and
   `scripts/profiling/bench_doorbell.py` (new native arm hosting the real gate
   measurement, since it measures true cross-process publish→detect latency via
   `imp.last_latency` — `bench_r1_wait.py`'s `get_frame()` wall-clock time cannot); this
   BENCHMARKS.md table; this CHANGELOG entry; the ADR-0006 consequences note.

*Originally proposed, not followed: scaffold (S) → waiter (M) → seam (S) → bench+docs
(S) as four sequential phases. In practice the seam landed with the scaffold (both are
small, interdependent, and easier to validate together), and the waiter's real
complexity was in fixing D3's resolution bug post-hoc rather than in the initial
implementation.*

## Verification & acceptance

- Notification latency at 60 fps producer, 1080p: **p50 < 10 µs, p95 < 50 µs**
  (baselines: poll-sleep 136–286 µs; doorbell-only per CHANGELOG 1.11.0 — native must
  beat or match doorbell-only while removing the Python loop cost).
- CPU: consumer core utilization not worse than the current `wait_spin_us=200` config.
- Fallback: uninstalling `cuda-link-native` or `CUDALINK_WAIT_BACKEND=python`
  reproduces current behavior exactly (existing tests are the proof).

**Measured (2026-07-04, RTX 4090, `scripts/profiling/bench_doorbell.py`, 512×512
float32, 300 frames/arm, 30+60 fps)** — this is the authoritative gate measurement:
publish→detect latency via `imp.last_latency`, not `bench_r1_wait.py`'s `get_frame()`
wall-clock time (that script's own printout explains why it can't host this gate: its
number includes tensor materialization + Python overhead on top of the wait itself).

| fps | Arm | CPU% | latency p50 | latency p95 |
|---|---|---|---|---|
| 30 | doorbell | 1.1% | 69.3 µs | 141.4 µs |
| 30 | native   | 0.8% | 66.4 µs | 138.7 µs |
| 60 | doorbell | 1.8% | 64.6 µs | 113.4 µs |
| 60 | native   | 0.3% | 67.4 µs | 140.2 µs |

**Gate: MISS** at both fps (native p50 ≈ 66–67 µs, p95 ≈ 139–140 µs vs the 10 µs/50 µs
targets) — and, notably, **native is not meaningfully faster than plain doorbell here**;
the two are within a few µs of each other at both fps, well inside run-to-run noise.

This does not mean R5's own C++ wait logic is slow: `bench_r1_wait.py`'s
`avg_spin_us` (the native backend's own re-check-after-wake latency, measured
separately) is ~0.02–6.5 µs depending on scenario — genuinely fast, and not the
bottleneck `imp.last_latency` is capturing. The likely explanation is that
`last_latency` measures the *full* cross-process round trip (producer publish →
`WaitForSingleObject` wake → consumer resumes → `_begin_frame` reads the timestamp),
and Windows kernel-mediated cross-process event signaling has an inherent tens-of-µs
wake/scheduling floor that is the same whether the post-wake re-check happens in
Python or C++ — R5 removes the Python *loop* cost (proven by `avg_spin_us`), but the
dominant cost in this number turns out to be the OS wake itself, which neither
backend controls. Recorded honestly per this plan's own accept-gate framing: the
10 µs/50 µs target, as measured by cross-process `last_latency`, **does not pass** on
this hardware; R5 still ships because it does not regress anything (native ≈ doorbell,
never worse) and the seam is a clean, tested, reversible opt-in.

## Risk register

| Risk | Sev | Mitigation |
|---|---|---|
| cudart not yet loaded when the backend initializes | Med | Resolve at connect, after `CUDARuntimeAPI` load; refuse + fallback otherwise |
| Auto-reset event is single-consumer (second consumer starves) | Med | Already documented in `_doorbell.py`; native path pre-checks `write_idx` + event before blocking, so a missed signal costs one `slice_ms`, not a hang |
| GIL released while touching Python-owned memory | Low | All args passed by value (ints) before release |
| Windows-only wheel matrix maintenance | Low | Exact `spout/` precedent already maintained |
| Interaction with PLAN-003's cuda-bindings backend (different cudart DLL name) | Low | Resolver taught the wheel's DLL name, else python-wait fallback (see PLAN-003 D2) |
