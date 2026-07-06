# ADR-0007: Expose Spout as a sidecar-launcher COMP, not an embedded transport

**Status**: Accepted
**Date**: 2026-06-27
**Applies to**: `spout/` (`cuda_link_spout` package) and the `CUDALinkSpoutBridge.tox`
component.

---

## Context

cuda-link-spout is an opt-in native add-on (pybind11 + D3D11 + CUDA, Windows-only) that
bridges CUDA-IPC GPU frames to **Spout** (D3D11 texture sharing protocol).  After Phase 1
(installer flag `--spout`), its canonical runtime entry point was:

```
python -m cuda_link_spout.bridge --dir out --ipc my_ipc --spout my_sender
```

The unique value of cuda-link-spout is **headless Python↔Spout on-GPU**: a torch/cupy
tensor becomes a Spout sender with one GPU de-swizzle copy, never touching CPU.  This is
impossible via SpoutGL's GL-based Python binding, which requires an OpenGL context and a CPU
round-trip.  TouchDesigner already ships native **Spout In/Out TOPs** that handle the in-TD
direction directly.

After Phase 1, the only way for a TD artist to *run* the bridge was to open a terminal — a
step they will never take.  This repo already has this failure mode: `td_exporter/TDConfig.py`
exposes ten sender knobs exclusively as `CULALINK_*` env vars with no visible TD param.  The
same trap is what we are explicitly avoiding.

Three options were considered:

| # | Approach | Summary |
|---|---|---|
| 1 | Embed Spout as a transport inside the existing CUDA-Link COMP | `Mode = Spout` toggle; imports `_spout_bridge.pyd` in-process inside TD |
| 2 | Dedicated droppable "Spout Bridge" launcher COMP | Spawns and supervises the existing `bridge.py` sidecar |
| 3 | Documentation only | Document the CLI; leave no TD runtime entry point |

---

## Decision

**Option 2: dedicated, droppable `CUDALinkSpoutBridge` COMP.**

The COMP does no GPU work itself.  It spawns and supervises the existing `bridge.py`
sidecar via `subprocess.Popen`.  Every bridge input is a visible TD custom parameter (no
env-var reads).  The sidecar's visible console window is the runtime log with no extra
infrastructure.

---

## Rationale

### Against option 1 (embedded in-process transport)

- **Redundant with native Spout TOPs for the in-TD case.**  When TD is both the source and
  the sink, native Spout In/Out TOPs handle it directly and are already wired into TD's cook
  graph.  Adding `Mode = Spout` to the CUDA-Link COMP would duplicate them with no added
  value.

- **Loading a native extension inside TD's embedded Python carries GPU-state risk.**
  `_spout_bridge.pyd` owns a D3D11 device and CUDA contexts.  A GPU device error leaves
  these in a bad state with no recovery path short of restarting TD.  The sidecar model
  contains the blast radius to one console window; TD stays running.

- **Coupling creep.**  A transport-mode toggle on the existing COMP mixes two concerns
  (CUDA-IPC channel management and Spout bridging), adding params, dispatch handlers, and
  test surface to an already complex COMP.

### Against option 3 (docs only)

**Invisible feature.**  A CLI command a TD artist never sees is effectively not shipped.
Hidden features include env-var-only knobs (already a known failure mode in this repo) and
terminal-only entry points.

### For option 2

- **Thin**: three new files in `td_exporter/` + the `.tox` binary.  No new GPU code, no
  new transport — the sidecar was already shipping.
- **Visible at drop time**: a TD artist drops the COMP, sets `Active = ON`, done.  The
  console window IS the bridge log.
- **Blast-contained**: sidecar crash ↔ closed console; `Status` flips to `Exited (code n)`.
  TD keeps running.
- **Symmetric with existing launchers**: `example_sender_launcher.py` and
  `example_receiver_launcher.py` already solve the full subprocess lifecycle
  (`CTRL_BREAK_EVENT`→terminate→kill, crash detection, `CREATE_NEW_PROCESS_GROUP`).  This
  COMP adapts those proven patterns with per-instance state (not a module-level global) so
  multiple dropped COMPs never share a process handle.
- **Zero env-var regression**: the `spout/` package has no `os.environ` reads.  The COMP
  maintains that invariant — every knob is a visible TD custom parameter.

---

## Rejected alternatives

- **Option 1 (embedded in-process Spout)**: redundant with native Spout TOPs for the
  TD-as-endpoint case; native extension inside TD's interpreter carries GPU-state risk;
  couples Spout to the core COMP.
- **Option 3 (docs only)**: leaves the headless Python↔Spout scenario invisible — the
  exact hidden-feature failure mode the design was built to avoid.

---

## Consequences

- `CUDALinkSpoutBridge` is a **standalone launcher** — it must be wired to a cuda-link
  channel by matching `Ipcname` to the producing process's `shm_name`.  There is intentionally
  no auto-coupling to a sibling CUDA-Link COMP (deferred enhancement, not in scope).
- **BGRA8 output is not auto-derivable**: both RGBA8 and BGRA8 share the `uint8` dtype, and
  channel order is not carried in CUDA-IPC metadata.  The `--dir out` path assumes RGBA
  channel order.  A future `--fmt` override can restore BGRA8 for consumers that require it.
- Loading `cuda_link_spout` (or `_spout_bridge.pyd`) directly inside the TD-embedded Python
  process anywhere — e.g., a helper DAT or a Mode toggle — would violate this ADR's
  blast-containment rationale and must reopen this decision.

---

## Reopen condition

Revisit if either:

- The performance profile of loading `_spout_bridge.pyd` inside TD's interpreter is measured
  and the GPU-state risk is quantified as acceptable for the target use case.
- A future TD API exposes first-class subprocess management, making the external sidecar
  pattern unnecessary.
