# ADR-0010: Park the Out TOP CUDA-Graphs path (array-source node update rejected)

**Status**: Accepted (design replaced 2026-08-14 by a keyed exec cache; re-measured
2026-11-17: no gain, not recommended. See the two dated sections at the end.)
**Date**: 2026-07-06
**Applies to**: the env-gated CUDA-Graphs submission collapse in the Out TOP
(`CudaLinkOutTOP::tryGraphCopy`, PLAN-005 §2.1, `CUDALINK_CPP_USE_GRAPHS`).

---

## Context

PLAN-005 §2.1 ported the Python exporter's CUDA-Graphs submission collapse to the Out TOP:
capture the per-slot IPC memcpy + interprocess event-record once, then relaunch the
instantiated exec each cook after refreshing the copy node's source via
`cudaGraphExecMemcpyNodeSetParams`. Live testing (2026-07-06, ninja DLLs) kills the path on
its first update cook: the node update fails with `invalid argument` at frame 3 and
`myGraphsDisabled` latches, falling back to the legacy per-op path for the rest of the
session.

The root cause is structural, not a bug in the wiring:

- The Out TOP's copy **source is a `cudaArray_t` handed to it by TD each cook**
  (`TOP_Output::createCUDAArray`), and the handle can change cook-to-cook. Exec-level memcpy
  node updates are documented to reject parameter changes that alter the operand's memory
  *allocation* (as opposed to offsets within the same allocation) — a different
  `cudaArray_t` each cook is exactly that rejected case.
- The Python reference never hits this because it only ever updates a **1D linear-to-linear**
  copy: `graph_exec_memcpy_node_set_params_1d` (`src/cuda_link/cuda_graphs.py:249`, hot path
  `exporter.py:632`) swaps device pointers within pre-allocated linear buffers it owns. The
  Out TOP's TD-owned array source has no equivalent — there is no stable placeholder
  allocation to capture against.

## Decision

**Park the Out TOP graphs path as-is: env-gated OFF by default, no revert, no further work on
the array-source update.** The code stays in-tree (capture, validation, fallback, and the
`graphs: ...` debug lines are all live-test-hardened and cost nothing when the gate is off);
`myGraphsDisabled` continues to guarantee a correct legacy-path session on any failure.

The future graphs candidate is the **In TOP**, whose copy *source* is a linear
`mySlotDevPtrs` device pointer this project allocates and owns — the same shape the Python
reference updates successfully (tracked as PLAN-005 follow-up work, task #13).

## Rejected alternatives

- **Re-capture the graph whenever the source array changes** — TD may hand back a different
  array every cook; per-cook capture+instantiate costs far more than the ~4 µs/frame the
  graph saves, and defeats the point of the collapse.
- **Copy TD's array into an owned staging linear buffer, then graph the staging→IPC copy** —
  adds a third full-frame copy to remove one driver submission; strictly worse on the 16F/32F
  sizes where submission overhead matters least.
- **Instantiate with `cudaGraphInstantiateFlagDeviceLaunch`-era update tricks or per-cook
  exec rebuild** — same per-cook instantiate cost problem as re-capture.
- **Revert the graphs code entirely** — the gate is default-off with zero shipping impact,
  and the capture/fallback scaffolding is exactly what the In TOP pivot will reuse.

## Consequences

- (Until 2026-08-14.) `CUDALINK_CPP_USE_GRAPHS=1` on the Out TOP ran the graph path for the
  first cook per slot, then latched disabled at the first node update. The section
  "Design change 2026-08-14" below describes the current code.
- The Out TOP keeps its legacy two-submission cook (memcpy + event record); PLAN-005 §2.1's
  measured win remains unrealized on the producer side.
- The In TOP pivot inherits working, live-tested capture/instantiate/fallback code.

## Reopen condition

Revisit if any of:

- CUDA adds exec-level memcpy node updates that accept a changed array allocation (release
  notes / `cudaGraphExecMemcpyNodeSetParams` docs);
- the TD Custom TOP API grows a way to obtain a stable per-slot output array across cooks;
- the In TOP graphs port (task #13) lands and producer-side submission overhead still shows
  up in the `gpu_ipc_us` timing channel.

The In TOP port landed as a keyed graph-exec cache — see
[ADR-0011](0011-in-top-graphs-keyed-cache.md) for the design and its live dst-array-stability
verdict.

## Design change 2026-08-14: keyed exec cache (commit d4b81d1)

The Out TOP no longer updates a graph node. It uses the same design as the In TOP
(ADR-0011), with the key on the other operand:

- `CudaLinkOutTOP::tryGraphCopy` keeps a cache of instantiated execs for each slot, with
  the key `(slot, srcArray)`. Both memcpy operands are fixed at capture time.
- On a cache hit it launches the exec. On a miss it captures the slot copy and the
  interprocess event record (2 nodes), instantiates the graph, stores the exec and
  launches it in the same cook.
- The cap is 4 entries per slot (`kGraphCacheCapPerSlot`). An overflow, or a failure in
  capture, instantiate or launch, sets `myGraphsDisabled` for the session, and the cook
  continues on the legacy path.
- `destroyGraphs()` frees all execs before the slot buffers and events are freed.

So the path works on the Out TOP: it does not latch off at frame 3. The text above, from
"Context" to "Reopen condition", records the first design and why it failed.

## Re-measure 2026-11-17 (Debug off)

Dev box (RTX 4090, driver 621.24, TD 2025.33230), six Out TOPs in the sync-kit sender,
with `Debug` off on every TOP. Values are medians of Info CHOP samples:

| Out TOP metric | Graphs | Legacy |
|---|---|---|
| `copy_us` | 67 µs | 59 µs |
| `cook_us` | 161 µs | 175 µs |

An Nsight Systems trace (CUDA and WDDM, no CPU sampling) gives a `cudaGraphLaunch` API
median of 30 µs for each cook.

Result: the graph path gives no gain on the Out TOP. `copy_us`, which holds the
submission, is higher with graphs. The `cook_us` difference is smaller than its spread
from frame to frame.

**Decision**: keep the code, default off. Do not set `CUDALINK_CPP_USE_GRAPHS` on the dev
box or on the show machine. With `Debug` on, the GPU timers make Out TOP `copy_us` about
3.5 times larger (about 230 µs), so do not compare graph and legacy runs with `Debug` on.
