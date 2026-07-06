# ADR-0010: Park the Out TOP CUDA-Graphs path (array-source node update rejected)

**Status**: Accepted
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

- `CUDALINK_CPP_USE_GRAPHS=1` on the Out TOP runs the graph path for the first cook per slot,
  then latches disabled at the first node update — harmless but pointless; the variable
  should stay unset.
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
