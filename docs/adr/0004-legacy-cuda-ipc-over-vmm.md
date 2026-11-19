# ADR-0004: Legacy CUDA Runtime IPC over VMM driver API

**Status**: Accepted
**Date**: 2026-05-31
**Applies to**: `src/cuda_link/cuda_ipc_wrapper.py`, `src/cuda_link/exporter.py`,
`src/cuda_link/importer.py`, `src/cuda_link/shm_protocol.py`,
`cpp_top/src/common/cuda_device_session.h`,
`docs/ARCHITECTURE.md` §"Why Legacy IPC Over VMM API".

---

## Context

This decision has been re-litigated by external research briefs — most recently
*"ctypes Correctness and CUDA IPC / Zero-Copy GPU Sharing on Windows, RTX 5090 / Blackwell,
CUDA 12.8+"* — arguing for migration to the VMM driver API
(`cuMemCreate` + `CU_MEM_HANDLE_TYPE_WIN32` + `DuplicateHandle`) and claiming that legacy
`cudaIpc*` memory IPC is Linux-only and fails on Windows WDDM with error 801.

NVIDIA's own documentation supports the "Linux-only" half of that claim. CUDA Programming Guide §4.15:
*"The CUDA IPC API is only currently supported on Linux platforms."* §4.15.1 repeats it: *"The IPC API is
only supported on Linux."* The `simpleIPC` sample gates its Windows path explicitly:
`// CUDA IPC on Windows is only supported on TCC`, followed by
`if (!prop.tccDriver) { printf("Device %d is not in TCC mode\n", i); continue; }`. This project runs on
the default WDDM driver model, not TCC — so it is running outside NVIDIA's documented support envelope,
not disproving it.

What this codebase's testing does establish is that, *in practice*, it works there anyway: production IPC
has been validated on Windows WDDM with CUDA 12.x across multiple format variants and teardown/reconnect
cycles with no error 801. The "err 801" the migration brief cites arises on the `torch.multiprocessing`
IPC path, not on raw `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`. This is an engineering bet on
observed behaviour outside the documented envelope, not a refutation of NVIDIA's stated support matrix —
treat it as such: re-validate after driver/CUDA upgrades, and see the IPC capability probe
(`CUDARuntimeAPI.check_ipc_capability()`) added to surface a diagnostic if this ever stops holding.

The C++ Custom TOP plugin (`cpp_top/`) makes the identical bet on the identical driver model, since it
runs in the same TouchDesigner process against the same WDDM device: `cudalink::common::CudaDeviceSession`
(`cpp_top/src/common/cuda_device_session.h`) carries the C++-side analogue of the same probe, gated the
same way (`cudaDevAttrIpcEventSupport` is a CUDA 12.0+ attribute, so both the compile-time reference —
`#if CUDART_VERSION >= 12000` — and the runtime query are gated on that floor). On success it records a
`note` field with the same WDDM/TCC caveat text as the Python probe, surfaced via the debug log and a
dedicated `init_note` Info DAT row in both `CudaLinkInTOP` and `CudaLinkOutTOP` (PLAN-001 §D3's own
`cudaRuntimeGetVersion()` major-version guard sits right beside it in the same ctor path). The reopen
condition below applies equally to both implementations.

Recording this as an ADR stops future explorers from spending time re-investigating a decision that has
already been made with empirical evidence.

## Decision

Keep the **CUDA Runtime API IPC** approach:

- **Memory handles** — `cudaIpcGetMemHandle` (producer) / `cudaIpcOpenMemHandle` (consumer) /
  `cudaIpcCloseMemHandle` (consumer teardown). 64-byte opaque blob.
- **Event handles** — `cudaIpcGetEventHandle` / `cudaIpcOpenEventHandle`. 64-byte opaque blob.
- **Transport** — `multiprocessing.shared_memory` with `SLOT_SIZE = 128` (64 B mem + 64 B event,
  defined in `shm_protocol.py`).

Rationale:

1. **Linear memory only.** TouchDesigner's `top_op.cudaMemory()` returns linearized pixel data from a
   `cudaMalloc` buffer. The VMM API's advantages — texture layout preservation, virtual address
   manipulation, fine-grained access control with `SECURITY_ATTRIBUTES` — solve problems this project does
   not have. The memory is linear, contiguous, and does not require sparse/partial mappings.

2. **Validated on Windows WDDM.** IPC roundtrip sweeps confirm the legacy path works on WDDM with CUDA
   12.x. CuPy (Python GPU arrays) and dora-rs (robotics middleware) use the same approach on Windows.

3. **Lower complexity.** Runtime IPC is ~600 lines end-to-end; the VMM path requires 4 allocation steps
   (create + reserve + map + set-access), manual CUDA driver context management, Win32
   `DuplicateHandle`/`OpenProcess` plumbing, and security descriptor handling — roughly 1 500+ lines for
   equivalent functionality.

| Factor | Legacy IPC (chosen) | VMM API (rejected) |
| --- | --- | --- |
| Code volume | ~600 lines | ~1 500+ lines |
| Allocation | 1 step (`cudaMalloc`) | 4 steps (create + reserve + map + access) |
| API level | Runtime (automatic context) | Driver (manual context) |
| IPC overhead | ~3–8 µs | Same for linear D2D |
| TD compatibility | Proven (WDDM, CUDA 12.x) | Unvalidated |
| Win32 surface area | None | `DuplicateHandle`, `OpenProcess`, `SECURITY_ATTRIBUTES` |

## Rejected alternative

**VMM driver API** — `cuMemCreate` + `cuMemExportToShareableHandle(CU_MEM_HANDLE_TYPE_WIN32)` +
`cuMemImportFromShareableHandle` + `cuMemMap` + `cuMemSetAccess`, with `DuplicateHandle`/`OpenProcess` to
transfer the Win32 NT handle cross-process.

Rejected because: (a) adds substantial complexity and Win32 surface area for zero benefit on linear memory
that already shares cleanly; (b) requires driver-level context management, which is harder to reason about
than the runtime API's automatic context; (c) unvalidated on this WDDM/TouchDesigner stack. The only thing
VMM buys over legacy IPC on linear memory is virtual address aliasing and SECURITY_ATTRIBUTES — neither is
needed here.

A proof-of-concept probe (`scripts/probe/driver_api_ipc_probe.py`) was written and confirmed the VMM path
is available on this hardware, but it was never integrated into production.

## Consequences

**Positive:**

- Production path stays simple and auditable (~600 lines, all in `cuda_ipc_wrapper.py`).
- No Win32 handle management, no security descriptor plumbing, no driver-context lifecycle.
- Fully validated; existing test suite and TD smoke tests cover the IPC path.

**Negative / trade-offs:**

- Cannot share `cudaArray`/texture objects directly (opaque, swizzled layout). If TD ever stops linearizing
  GPU memory before handing it off, the transport design would need to change.
- Legacy `cudaIpc*` memory IPC is documented as "a similar concept" to VMM in NVIDIA's programming guide —
  long-term driver support on WDDM is assumed but not guaranteed.

## Reopen condition

Revisit this decision if **either** of the following occurs:

1. The codebase needs to share a `cudaArray` / mipmapped texture / sparse resource without linearization.
2. A future CUDA driver drops support for `cudaIpcGetMemHandle` on Windows WDDM for linear `cudaMalloc`
   allocations.

In either case, the VMM probe script (`scripts/probe/driver_api_ipc_probe.py`) is the starting point for
the migration path.
