# ADR-0011: Port CUDA Graphs to the In TOP via a keyed graph-exec cache

**Status**: Accepted (implementation merged, default OFF; live-validated 2026-07-06 — see
"Live-test results")
**Date**: 2026-07-06
**Applies to**: the env-gated CUDA-Graphs submission collapse in the In TOP
(`CudaLinkInTOP::tryGraphedCopy`, PLAN-005 §2.1 receiver half, `CUDALINK_CPP_USE_GRAPHS`).

---

## Context

ADR-0010 parked the Out TOP graphs path because its copy **source** is a TD-owned
`cudaArray_t` that changes cook-to-cook, and exec-level memcpy node updates reject changed
allocations. It named the In TOP as the pivot (PLAN-005 task #13): the receiver's copy
**source** is linear `mySlotDevPtrs[readSlot]` — project-owned, stable for the session, the
same shape the Python exporter updates successfully.

Implementation exploration surfaced the wrinkle ADR-0010 glossed over: the In TOP's copy
**destination** is *also* a TD-owned `cudaArray_t`, re-obtained every cook via
`TOP_Output::createCUDAArray()` with no documented cross-cook stability guarantee
(TOP_CPlusPlusBase.h). Exec-level memcpy updates require **1D linear operands on both
sides** — the Python wrapper's own docstring says so
(`src/cuda_link/cuda_graphs.py:249-273`, `graph_exec_memcpy_node_set_params_1d`) — so a node
whose destination is a CUDA array can never be exec-updated, regardless of how stable the
source is. The Python-style per-frame `SetParams1D` hot path is structurally unusable here,
exactly as it was for the Out TOP, just on the opposite operand.

## Decision

**Never update graph nodes at all. Cache one instantiated exec per observed
`(readSlot, dstArrayPointer)` pair, with both memcpy operands baked in at capture time.**

- **Key**: `(readSlot, dstArray)`. Slot pointers are finite (num_slots ≤ 4) and part of the
  key; the destination array pointer is the other half. A cache hit means launching an exec
  whose baked operands are pointer-identical to what this cook would have passed to the
  legacy pair — launching is equivalent to re-issuing `cudaStreamWaitEvent` +
  `cudaMemcpy2DToArrayAsync` with those exact arguments.
- **Capture** (`tryGraphedCopy`, mirroring the Out TOP's live-test-hardened scaffolding):
  `cudaStreamBeginCapture(myStream, cudaStreamCaptureModeRelaxed)` →
  `cudaStreamWaitEvent(..., cudaEventWaitExternal)` on the imported interprocess slot event →
  `cudaMemcpy2DToArrayAsync` → end capture → node-count == 2 validation →
  `cudaGraphInstantiate` → destroy the template graph → cache → `cudaGraphLaunch` **the same
  cook** (capture only records; the launch performs this cook's copy — no dropped frame on
  build cooks). Relaxed mode is mandatory: `myStream` is shared with TD's Vulkan interop, and
  Thread-Local/Global capture would abort on any concurrent stream touch.
- **Cap 4 entries per slot, hard-disable on overflow — never LRU-evict.** A cap breach means
  the dst-pointer-stability assumption failed; eviction would silently mask exactly the
  instability signal this port exists to measure. Any capture/instantiate/launch failure also
  latches `myGraphsDisabled` (sticky, session-long) with a `graphs: ...` debug line, falling
  through to the legacy pair the same cook.
- **The cache doubles as the live diagnostic** for TD's output-array pointer behavior — the
  research question ADR-0010's reopen condition poses. `graph_builds` plateauing at ≤
  num_slots × (small array set) proves stability; a climbing count followed by the cap latch
  proves churn. Either outcome is a written answer.
- **Lifecycle**: execs are held in `CudaGraphExecGuard` RAII wrappers
  (`cpp_top/src/common/raii_handles.h`); `destroyGraphs()` runs at the top of
  `closeHandles()`, the single funnel for both VERSION_CHANGED and teardown, because every
  cached exec bakes a slot device pointer and slot event that funnel invalidates.
  Resolution/format changes need no explicit invalidation: a new array ⇒ new pointer ⇒ cache
  miss ⇒ natural key rollover; stale entries idle harmlessly until the next `closeHandles()`
  (`cudaGraphExecDestroy` frees exec bookkeeping only — it never dereferences baked operands,
  so destroying an entry whose array TD already freed is safe; only *launching* it would not
  be, and the pointer-keyed lookup prevents that).
- **Observability**: Info CHOP channels `graph_hits` / `graph_builds` (Debug-gated, matching
  the `noframe_count`/`rescued_count` convention). On graph cooks `gpu_copy_us` brackets the
  whole `cudaGraphLaunch` (wait + copy fused) and `event_wait_us` reads 0.0 — not separately
  observable inside a graph. A/B is therefore graph `gpu_copy_us` vs legacy
  `gpu_copy_us + event_wait_us`, plus `bench:` `avg_cook_us`.
- **Default OFF**, opt-in via `CUDALINK_CPP_USE_GRAPHS` — deliberately the same variable the
  Out TOP reads: its parked path self-disables harmlessly at frame 3 (ADR-0010), and one
  variable enables "the graphs feature" project-wide, matching the Python side's single
  `CUDALINK_USE_GRAPHS` gate. Default stays OFF regardless of the A/B outcome; the
  deliverable is a validated, safe, opt-in path plus a written verdict.

## The one unproven CUDA step

Capturing a **wait** on an *imported* interprocess event (`cudaIpcOpenEventHandle`, flags
`cudaEventDisableTiming | cudaEventInterprocess`) with `cudaEventWaitExternal` inside stream
capture. The Python graphs path only ever proved *recording* an IPC event inside capture
(exporter side). The first build attempt IS the probe: the pre-recorded-state precondition is
naturally satisfied (the producer records the slot event before `acquire_slot()` ever
classifies that slot NewFrame), and if the runtime rejects the capture, the latch produces
one `graphs: ... disabling` line and a clean legacy session — the probe result costs nothing.

## Rejected alternatives

- **Per-frame exec update (`cudaGraphExecMemcpyNodeSetParams`)** — structurally impossible:
  exec-level memcpy updates require 1D linear operands on both sides; the destination is a
  CUDA array.
- **LRU eviction on cap breach** — masks the instability signal; a churning-array session
  would silently pay capture+instantiate per cook, strictly worse than legacy.
- **Per-cook re-capture** — same objection as ADR-0010: capture+instantiate costs far more
  than the few µs a launch saves.
- **Staging the dst through an owned linear buffer** — adds a third full-frame copy to remove
  one submission; same rejection as ADR-0010's staging alternative.

## Accepted residual risk: ABA on the destination pointer

TD could free array A and allocate a *different* array at the same address with identical
dims/format between cooks. No vendor-exposed generation counter exists to detect this. The
blast radius is a corrupted frame or an async CUDA error surfaced on a later call — not a
crash — because the baked copy geometry (widthInBytes/height) still matches the format.
Accepted because: the destination is re-requested from TD every cook and handed straight to
the lookup (narrow window), the feature is opt-in Debug-era tooling, and any live-test visual
artifact is grounds to re-park. This is the standing reopen/re-park trigger.

## Consequences

- `CUDALINK_CPP_USE_GRAPHS=1` now enables a *working* path on the receiver (pending live
  probe) while the sender half still self-disables at frame 3 — one env var, two documented
  behaviors (ADR-0010 + this ADR).
- Hot-path cooks with the env unset are byte-identical to before: both gate booleans are
  false and the branch is never taken; `graph_hits`/`graph_builds` read 0.
- A successful session collapses the receiver's two per-cook submissions (waitEvent +
  memcpy2DToArray) into one `cudaGraphLaunch` — a WDDM-submission-overhead play, not a
  bandwidth one. Honest expectation from Python P3: single-digit µs per cook.

## Live-test results (2026-07-06, `CUDALINK_CPP_USE_GRAPHS=1`, Debug ON, ninja DLLs)

7,932 frames over ~143 s (~55.5 FPS), 3-slot ring. Zero `graphs:` failures, zero disable
events, no visual artifacts reported.

- **IPC-event-wait capture probe: PASSED.** Every capture of the
  `cudaStreamWaitEvent(imported IPC event, cudaEventWaitExternal)` + memcpy pair built and
  instantiated first try — 21/21 builds across the session, no `disabling` line ever.
- **dst-array stability: CONFIRMED (per SHM session segment).** Within each segment TD handed
  back the *same* `cudaArray_t` pointer every cook: each slot's cache held exactly one entry
  (`1/4`), so `graph_builds` plateaued at 3 (= num_slots × 1 array) within 3 frames of each
  (re)connect. The cap was never approached. The pointer changed only across
  VERSION_CHANGED boundaries — and only 3 distinct values appeared across 7 segments, i.e.
  TD even reuses addresses (which is also why the ABA caveat stays documented).
- **Lifecycle: exercised 7×.** Six mid-session VERSION_CHANGED events (producer restarts)
  plus final teardown each logged `destroyGraphs: dropping 3 cached graph exec(s)` via the
  `closeHandles()` funnel and rebuilt within 1–3 frames. No crash, no leak, no stall.
- **A/B vs the immediately preceding legacy session** (env unset, 42k frames, same 3-slot
  setup): CPU `avg_cook_us` p50 **384 µs (graphs) vs 451 µs (legacy)** (~15 % lower); means
  413 vs 436 µs; p25 230 vs 282 µs. Direction is positive, but the two sessions were not
  load-controlled, so treat the magnitude as indicative, not measured-in-isolation.
- **GPU-side channels are not a clean instrument here**: graph-mode `gpu_copy_us` fuses the
  producer wait into the copy (avg 723 µs, spikes to ~2.5 ms), and the legacy comparison
  (`gpu_copy_us + event_wait_us` avg ≈ 1356 µs) is equally dominated by producer timing
  (legacy `event_wait_us` swung 3 µs → 2.5 ms across windows). Both numbers measure "how
  long the GPU sat waiting for the producer," not submission overhead. `avg_cook_us` is the
  usable A/B metric.
- The Out TOP exhibited exactly its documented ADR-0010 behavior under the shared env var:
  built slots 0–2, `graphs: node update failed: invalid argument` at frame 3, latched off,
  clean legacy session thereafter.

### Controlled A/B (scripted Python producer → TD, 2026-07-06)

The TD→TD comparison above was not load-controlled, so the A/B was re-run against the
scripted Python producer (512×512 RGBA uint8, 60 FPS target, 3-slot ring, sender graphs ON
via `CUDALINK_USE_GRAPHS` default in both legs) with the receiver's
`CUDALINK_CPP_USE_GRAPHS` as the **only** variable. Both legs ran Debug ON on the same .toe,
and each happened to include one benign TD-side SHM reconnect (leg A frame 1089, leg B frame
1345 — recovered within ~10 frames both times).

| In TOP metric | Leg A — legacy (env unset) | Leg B — graphs ON | Delta |
|---|---|---|---|
| Receiver cooks / bench windows | 1,455 / 15 | 1,746 / 17 | — |
| `avg_cook_us` (mean of windows) | **482.3 µs** | **412.6 µs** | **−14.5 %** |
| NoFrame ratio | 2.27 % | 2.18 % | comparable load |
| Rescued cooks | 17.0 % | 16.3 % | comparable load |
| Legacy `event_wait + gpu_copy` vs fused graph `gpu_copy` | 43.0 + 25.3 = 68.3 µs | 203.2 µs (fused) | not comparable |

- **CPU cook time dropped 14.5 % (482.3 → 412.6 µs) under identical producer load** —
  excluding each leg's reconnect-straddling windows the graphs mean is 389.0 µs (−19 %).
  This upgrades the TD→TD ~15 % figure from indicative to measured: the win is real and
  lives where the design predicted, in per-cook CPU submission overhead.
- The GPU channels failed as an instrument again, this time in the opposite direction from
  the TD→TD run: the fused graph bracket (203 µs) and the legacy split-sum (68 µs) measure
  different spans, both dominated by where the cook lands in the producer's 16.7 ms cadence
  rather than by submission cost. `avg_cook_us` remains the only A/B-valid channel.
- Graphs behavior was identical to the TD→TD session: builds plateaued at 3 (`1/4` per
  slot), the same dst-array address appeared in both segments (address reuse again), zero
  capture/launch failures, and `destroyGraphs: dropping 3 cached graph exec(s)` fired
  cleanly at both the mid-session reconnect and final teardown.

**Verdict: the port works and is safe. Default stays OFF** per the decision above — the win
is real (−14.5 % CPU cook, confirmed under a controlled producer) but modest, and the path
remains opt-in diagnostics/perf tooling.

## Reopen / re-park conditions

- Any live-test visual artifact attributable to the ABA case → re-park immediately.
- The cap latch trips in normal sessions (TD churns output arrays) → the keyed-cache premise
  is dead; record the churn verdict and re-park.
- CUDA adds exec-level memcpy updates accepting array operands or a TD API guarantees stable
  output arrays → revisit both this design (the cache becomes over-engineering) and
  ADR-0010's parked sender path.
