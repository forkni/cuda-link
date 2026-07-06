# ADR-0011: Port CUDA Graphs to the In TOP via a keyed graph-exec cache

**Status**: Accepted (implementation merged, default OFF; live A/B verdict pending — see
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

## Live-test results

*Pending — to be filled in by a follow-up docs edit after the first
`CUDALINK_CPP_USE_GRAPHS=1` TD session:*

- IPC-event-wait capture probe: **TBD** (built vs `graphs: ... disabling` on first capture)
- `graph_builds` plateau value and dst-array stability verdict: **TBD**
- A/B (`avg_cook_us`, graph `gpu_copy_us` vs legacy `gpu_copy_us + event_wait_us`): **TBD**

## Reopen / re-park conditions

- Any live-test visual artifact attributable to the ABA case → re-park immediately.
- The cap latch trips in normal sessions (TD churns output arrays) → the keyed-cache premise
  is dead; record the churn verdict and re-park.
- CUDA adds exec-level memcpy updates accepting array operands or a TD API guarantees stable
  output arrays → revisit both this design (the cache becomes over-engineering) and
  ADR-0010's parked sender path.
