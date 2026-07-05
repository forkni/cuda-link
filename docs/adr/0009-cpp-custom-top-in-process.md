# ADR-0009: Accept in-process native code inside TouchDesigner as a C++ Custom TOP

**Status**: Proposed (flips to Accepted when PLAN-001 Phase 0 spike validates)
**Date**: 2026-07-04
**Applies to**: the planned `cpp_top/` component (`CudaLinkOutTOP` / `CudaLinkInTOP`
Custom Operator DLLs; see [PLAN-001](../plans/PLAN-001-cpp-custom-top.md)).

---

## Context

[ADR-0007](0007-spout-as-launcher-not-transport.md) rejected loading a native
`.pyd` inside TD's embedded Python for the Spout bridge on **GPU-state blast-radius**
grounds: a native module owning a D3D11 device + CUDA contexts inside TD has no
recovery path short of restarting TD, while a sidecar contains failure to one console
window.

PLAN-001 proposes running native code inside TD again — as C++ Custom TOPs — to
eliminate the `top.cudaMemory()` staging alloc+copy (80–105 µs/frame) and per-frame
Python scaffolding (~40 µs), targeting ~2× lower and much more deterministic sender
cook cost. Any C++ Custom TOP is, by definition, in-process native code, so this plan
must engage ADR-0007's prior explicitly.

## Decision

**Accept in-process native code for this case, scoped to Custom TOPs, because the risk
calculus differs from the Spout `.pyd` case:**

1. **Sanctioned extension mechanism.** Custom Operators are TD's own documented,
   first-class plugin surface ([Custom Operators](https://docs.derivative.ca/Custom_Operators),
   [Write a CPlusPlus TOP](https://docs.derivative.ca/Write_a_CPlusPlus_TOP)) with a
   versioned ABI (`setAPIVersion`), a defined CUDA interop contract
   (`beginCUDAOperations`/`endCUDAOperations`), and TD-managed texture lifecycle — not
   a foreign extension smuggled into TD's Python interpreter.
2. **Minimal foreign state.** The plugin uses only cudart — a runtime TD already loads —
   plus TD-provided cudaArrays. No D3D11 device, no Spout SDK, no second GPU API
   (the specific state ADR-0007 feared).
3. **Mitigations (mandatory, enforced in review):**
   - No C++ exceptions cross the ABI boundary — every entry point is wrapped; errors
     surface via `getErrorString`/`getWarningString` badges.
   - Every CUDA call checked; on error the TOP degrades to an error badge + no-op cook,
     never a crash path; IPC open failures are never fatal.
   - Release CRT (`/MD`) enforced by the build.
   - The plugin owns its `cudaStream_t` and never calls `cudaSetDevice` away from
     `TOP_Context::getCUDADeviceIndex()`.
   - CUDA-major runtime check vs TD's bundled runtime → refuse to cook on mismatch.
   - 1-hour soak at 60 fps (leak- and crash-free) gates each release (PLAN-001 Phase 4).

## Rejected alternatives

- **Stay Python-only in TD** — leaves the 80–105 µs per-frame `cudaMemory()`
  alloc+copy and GIL-bound cook jitter on the table; the Python `.tox` remains as the
  reference path regardless.
- **Sidecar process for the TD side** (ADR-0007's pattern) — cannot work here: the
  texture must be read inside TD's cook, on TD's device, within TD's Vulkan↔CUDA
  interop bracket. A sidecar would still need `cudaMemory()` or Spout to get the
  pixels out, re-adding the copy this plan removes.
- **Wait for a first-class TD IPC feature** — TouchEngine exists but is a heavier
  host-process model, not a per-TOP texture export.

## Consequences

- A bug in the plugin can crash the TouchDesigner process. This is the accepted trade
  for ~2× sender latency and deterministic cook cost; the mitigations above plus the
  soak gate bound the risk.
- Two build artifacts per release (CUDA 11.8 / TD 2023.1x and CUDA 12.8 / TD 2025.3x)
  enter the maintenance surface — unlike the version-agnostic Python `.tox`.
- The Python `.tox` is **not** deprecated; it is the fallback on any TD build the
  plugin doesn't cover and the behavioral reference for the wire protocol.
- ADR-0007 remains in force for its own scope (Spout transport in TD's Python).
  Scope note: that decision governs `.pyd` loading in TD's *embedded Python*; this one
  governs Custom Operator DLLs in TD's *plugin surface*.

## Reopen condition

Revisit (toward retreating to Python-only or a sidecar) if:

- soak or field use shows recurring TD crashes attributable to the plugin;
- Derivative changes the Custom OP CUDA contract in a way that breaks the
  one-copy design (e.g., removes linear-memory interop);
- maintenance of the dual-CUDA build matrix demonstrably outweighs the measured
  latency win.
