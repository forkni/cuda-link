# cuda-link ↔ Spout Bridge — Design & Integration Specification

**Status**: Design proposal (pre-implementation)
**Date**: 2026-06-25
**Owner**: cuda-link
**Companion**: strategic rationale in
[`plugin-expansion-analysis.md`](plugin-expansion-analysis.md) (§3.1 — why Spout,
where cuda-link beats it, where it's a wash). This document is the *engineering*
spec for the bridge itself.

> **Premise.** The pure-Python / zero-dependency rule
> ([ADR-0006](../adr/0006-stay-pure-python-no-rust.md)) is **relaxed for this
> component only**: the Spout bridge is an **optional, separately-distributed,
> native (C++/pybind11) add-on**. The core `cuda_link` wheel stays pure Python,
> zero required deps, ~30 KB; a user who never touches Spout never installs the
> bridge and never pays its dependency cost.

---

## 1. Goal

Let cuda-link exchange GPU frames with the entire Windows VJ/engine ecosystem —
**Resolume, TouchDesigner, Unreal, OBS, Notch, Unity, MadMapper, vvvv** — through
**one** adapter, in **both** directions, staying on the GPU the whole way:

- **Egress** (`cuda-link → Spout`): publish a Python AI tensor (or a TD frame
  already in cuda-link) as a Spout sender any VJ app can drag in as a source.
- **Ingress** (`Spout → cuda-link`): receive any Spout sender's output as a
  zero-copy `torch`/`cupy` tensor in a Python ML process.

Spout is the right and only target at this layer: it is Windows-only,
same-machine, GPU-texture-sharing — the architectural twin of cuda-link's
same-machine GPU-IPC — and every relevant app speaks it natively or via a
maintained plugin.

---

## 2. How Spout works (the parts that constrain the design)

- **Transport**: a shared **DirectX 11 texture**. The sender creates an
  `ID3D11Texture2D` with a share handle; the handle + metadata go into a named
  shared-memory map; receivers `OpenSharedResource` the same texture. Zero-copy on
  the **same adapter**. OpenGL apps bridge in via `WGL_NV_DX_interop` — but the
  wire object is always a D3D11 texture.
- **Share-handle type — the load-bearing detail**: Spout's default stores the
  share handle as a **`uint32_t`**, i.e. the *legacy global (KMT)* handle from
  `IDXGIResource::GetSharedHandle`. This maps **directly** onto CUDA's
  `cudaExternalMemoryHandleTypeD3D11ResourceKmt`. NT-handle mode (`bNThandle`,
  DX11.1) is opt-in; the bridge must **detect which** and pick the matching CUDA
  import type.
- **Synchronization**: a **named mutex** guards texture access (chosen over keyed
  mutex for DX9 compatibility) plus a **named frame-count event**
  (`SetFrameSync` / `WaitFrameSync` / `IsFrameNew`). Keyed mutex is an opt-in
  capability, **off by default**. Senders call `Flush()` after writing.
- **Formats**: default `DXGI_FORMAT_B8G8R8A8_UNORM`; also RGBA8, `R10G10B10A2`,
  RGBA16-float, RGBA32-float. **All 4-channel.** (CUDA-D3D interop supports
  `R32G32B32A32_FLOAT`, `R16G16B16A16_FLOAT`, and 8-bit 4-channel; 3-channel and
  `*_TYPELESS`/sRGB-typed formats are **not** importable — pick a fully-typed
  4-channel format and match `cudaChannelFormatDesc` exactly.)
- **SDK**: C++, **BSD-2-Clause** (`github.com/leadedge/Spout2`). Two surfaces:
  - **`spoutDX`** — native D3D11 class. Key methods (verified against the header):
    - `bool OpenDirectX11(ID3D11Device* pDevice = nullptr);` — **accepts your own
      device** ← this is how we pin Spout to the CUDA adapter.
    - `ID3D11Device* GetDX11Device();` / `ID3D11DeviceContext* GetDX11Context();`
    - `bool SetSenderName(const char*);` / `bool SendTexture(ID3D11Texture2D*);`
    - `bool ReceiveTexture(ID3D11Texture2D** ppTexture);` / `bool IsUpdated();`
    - `HANDLE GetSenderHandle();` / `DXGI_FORMAT GetSenderFormat();`
      / `unsigned int GetSenderWidth()/Height();`
    - `void SetFrameSync(const char*);` / `bool WaitFrameSync(const char*, DWORD);`
      / `bool IsFrameNew();` / `void CloseDirectX11();`
  - **`SpoutLibrary`** — a COM-style vtable DLL (`GetSpout()` factory). GL-centric;
    dispatched through a vtable, **not** flat C — so neither surface is cleanly
    ctypes-able (see §6).

---

## 3. How cuda-link works (one paragraph)

Producer `cudaMalloc`s **linear** device memory, copies the frame in (D2D),
`cudaIpcGetMemHandle` (64-byte) + `cudaIpcEventHandle` (64-byte), and publishes
both + metadata (W/H/channels/dtype) into a named SHM segment. Consumer
`cudaIpcOpenMemHandle` → linear device pointer → zero-copy `torch`/`cupy` view.
Full detail in [`ARCHITECTURE.md`](../ARCHITECTURE.md). The defining property is
**linear memory shared CUDA↔CUDA with GPU-side event sync.**

---

## 4. The core impedance mismatch (and its irreducible cost)

Three mismatches the bridge must reconcile:

| | cuda-link | Spout |
|---|---|---|
| Handle world | `cudaIpcMemHandle` (**CUDA-only**) | DXGI/NT/KMT **shared handle** |
| Memory layout | **linear** (`cudaMalloc`, pitched) | **CUDA array** — opaque, swizzled/tiled |
| Sync | `cudaIpcEventHandle` | named mutex + frame event (keyed mutex opt-in) |

**The irreducible cost**: a texture is a **CUDA array**, not linear memory. You
cannot pointer-alias between them. So every crossing costs **one device-to-device
"de-swizzle" copy** — `cudaMemcpy2DToArray` (egress) / `cudaMemcpy2DFromArray`
(ingress), or an equivalent surface-write kernel. At 1080p BGRA (~8.3 MB) that's
~15–30 µs of on-GPU bandwidth — negligible per frame, but **real**, and it means
the honest headline is *"one GPU reformat + interop,"* not *"zero-copy into
Resolume."* (True zero-copy survives only on cuda-link's CUDA↔CUDA core.)

---

## 5. Bridge architecture

The bridge is the **only** component that touches both handle worlds. It owns a
D3D11 device on the CUDA-matched adapter, a CUDA context, and the de-swizzle copy.

### 5.0 Device/adapter affinity (do this first — it's the #1 failure mode)

CUDA↔D3D interop and Spout sharing both require **the same physical GPU**. Sequence:

1. Choose the CUDA device; read its LUID (`cudaDeviceProp.luid`).
2. Enumerate DXGI adapters; create the `ID3D11Device` on the adapter whose
   `DXGI_ADAPTER_DESC.AdapterLuid` matches.
3. `spout.OpenDirectX11(myDevice)` — Spout now shares on **our** adapter.
4. On multi-GPU, this guarantees no cross-adapter share (which silently fails or
   falls back through host). cuda-link is already single-GPU, so this is a
   one-time check, not a per-frame concern.

### 5.1 Direction A — `cuda-link → Spout` (egress)

**Simple path (2 copies, lowest risk — ship first):**

```cpp
// one-time
cudaSetDevice(gpu);
ID3D11Device* dev = CreateD3D11OnAdapter(luidOf(gpu));
spout.OpenDirectX11(dev);
spout.SetSenderName("cuda_link_out");
ID3D11Texture2D* tex = CreateTexture2D(W, H, fmt /* RGBA8/16F/32F */, SHARED);
cudaGraphicsD3D11RegisterResource(&res, tex, cudaGraphicsRegisterFlagsNone);

// per frame (src = cuda-link linear device ptr, from Importer or a torch tensor)
cudaGraphicsMapResources(1, &res, stream);
cudaArray_t arr;
cudaGraphicsSubResourceGetMappedArray(&arr, res, 0, 0);
cudaMemcpy2DToArray(arr, 0, 0, srcPtr, srcPitch, W*bpp, H, cudaMemcpyDeviceToDevice);
cudaGraphicsUnmapResources(1, &res, stream);   // map/unmap insert intra-process sync
cudaStreamSynchronize(stream);                  // ensure copy done before Spout reads
spout.SendTexture(tex);                          // Spout copies tex → its shared texture (copy #2)
```

`SendTexture` does Spout's own internal copy (under its mutex) into the sender's
shared texture. That's the second copy — accepted for v1 because Spout owns its
mutex/frame-event and we never touch its internals.

**Optimized path (1 copy — phase 2):** create the *sender's* shared texture via
the lower-level `SpoutDirectX`/`spoutDX` sender-create, register **that** texture
with CUDA, write the de-swizzle copy straight into it under a **keyed mutex**
(imported into CUDA as `cudaExternalSemaphoreHandleTypeKeyedMutex`), then advance
the frame event manually. Removes Spout's internal copy. Higher coordination risk
— only pursue if profiling shows the second copy matters.

### 5.2 Direction B — `Spout → cuda-link` (ingress)

**Via `ReceiveTexture` (simple):**

```cpp
spout.OpenDirectX11(dev);
spout.SetReceiverName("resolume_out");

// per frame
ID3D11Texture2D* recv = nullptr;
if (spout.ReceiveTexture(&recv) && recv) {
    if (spout.IsUpdated()) ReRegister(recv, &res);     // size/format change → re-register
    cudaGraphicsMapResources(1, &res, stream);
    cudaArray_t arr;
    cudaGraphicsSubResourceGetMappedArray(&arr, res, 0, 0);
    cudaMemcpy2DFromArray(dstPtr, dstPitch, arr, 0, 0, W*bpp, H, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &res, stream);
    // publish dstPtr through cuda-link Exporter → torch/cupy consumer gets a tensor
}
```

**Via external memory (1 copy, skips Spout's internal copy):** read the sender's
shared texture directly — `GetSenderHandle()` + `GetSenderFormat()` →
`cudaImportExternalMemory` (`...D3D11ResourceKmt` for default Spout, `...D3D11Resource`
if NT-handle mode) → `cudaExternalMemoryGetMappedMipmappedArray` → `cudaArray` →
`cudaMemcpy2DFromArray` into a cuda-link linear buffer. Detect handle type at
connect time.

### 5.3 Synchronization design

- **Simple paths (§5.1/§5.2 high-level)**: Spout handles its own mutex around its
  internal copy; we `cudaStreamSynchronize` our de-swizzle copy before
  `SendTexture` / after `ReceiveTexture`. New-frame signalling uses Spout's
  `IsFrameNew` / `SetFrameSync`/`WaitFrameSync`. No keyed mutex needed.
- **Optimized 1-copy paths**: we write directly into a texture a Spout *consumer*
  may read, so we must interlock — import the texture's **keyed mutex** as
  `cudaExternalSemaphoreHandleTypeKeyedMutex` and `cudaWait/SignalExternalSemaphoresAsync`
  around the copy (acquire key → copy → release key). Requires creating the shared
  texture with `D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX` and Spout in keyed mode.
- cuda-link's own `cudaIpcEventHandle` stays **internal to the CUDA↔CUDA leg**
  (Importer/Exporter) — it is *not* used across the Spout boundary.

### 5.4 Format & channel coercion

- Spout is 4-channel. cuda-link carries arbitrary channels/dtype. The bridge must
  coerce: RGBA8 ↔ `B8G8R8A8_UNORM`/`R8G8B8A8_UNORM`, fp16 ↔ `R16G16B16A16_FLOAT`,
  fp32 ↔ `R32G32B32A32_FLOAT`. **BGRA↔RGBA channel swap** and **row-origin flip**
  (D3D top-left vs GL bottom-left; Spout's `bInvert`) should be **fused into the
  de-swizzle copy** via a surface-write kernel rather than added as a second pass.
- Non-RGBA ML data (e.g. N-channel latents) **cannot** go through Spout — that is
  by design the cuda-link-native lane (keep it on the CUDA↔CUDA path).

---

## 6. Implementation options

| Option | What | Verdict |
|---|---|---|
| **Native pybind11 module (recommended)** | A small C++ extension linking `spoutDX` + CUDA runtime; owns the D3D11 device, the CUDA-D3D11 interop, and the de-swizzle copy; exposes `SpoutSender`/`SpoutReceiver` to Python. | **Chosen.** `spoutDX` gives native `SendTexture(ID3D11Texture2D*)` and `OpenDirectX11(myDevice)` for adapter matching. C++ where the C++ SDK lives; clean tensor-in/tensor-out Python API. |
| Sidecar process | A standalone exe doing IPC-open → interop → Spout, driven by CLI/flags; Python core unchanged. | Good for a **bridge mode** (connect an existing cuda-link IPC name to a Spout name without importing native code into the user's Python). Offer as a secondary entry point. |
| ctypes/cffi to SpoutLibrary | Bind the COM-style vtable from Python. | **Rejected as primary.** `SpoutLibrary` is GL-centric vtable dispatch and `spoutDX` is a C++ class — neither is a flat C ABI; vtable calls + `ID3D11*` marshalling from ctypes are brittle. A native module is simpler and faster. |

**Decision**: ship a **pybind11 native module** as the primary, plus a thin
**sidecar CLI** (`python -m cuda_link_spout.bridge`) that reuses it for the
no-native-import bridge-mode use case.

---

## 7. Proposed Python API

```python
from cuda_link_spout import SpoutSender, SpoutReceiver

# Egress: GPU tensor -> Spout (Resolume/UE/OBS/... drag it in as a source)
with SpoutSender("cuda_link_out", width=1024, height=1024, fmt="RGBA8") as tx:
    # accepts torch.Tensor / cupy.ndarray on GPU, or a cuda-link GpuFrame
    tx.send(tensor)            # one de-swizzle copy; returns when queued

# Ingress: Spout sender -> GPU tensor
with SpoutReceiver("resolume_out") as rx:
    frame = rx.receive()       # -> torch.Tensor on GPU (or None if no new frame)
    if frame is not None:
        out = model(frame)
```

```bash
# Bridge mode (sidecar) — wire an existing cuda-link IPC name to a Spout name,
# no native import in the user's own Python process:
python -m cuda_link_spout.bridge --ipc my_texture_ipc --spout cuda_link_out --dir out
python -m cuda_link_spout.bridge --spout resolume_out  --ipc ai_input_ipc   --dir in
```

Design rules: tensor-in/tensor-out (hide all D3D11/CUDA-interop), context-manager
lifecycle (mirror `Importer`/`Exporter`), honour the same `CUDALINK_*` env vars
where sensible, and surface adapter/format mismatches as clear Python exceptions.

---

## 8. Packaging & distribution

- **New package `cuda-link-spout`** (separate from the core wheel). Windows-only,
  ships a compiled `.pyd` (per CPython ABI) statically linking `spoutDX`; depends
  on the CUDA runtime already present for cuda-link.
- Core `cuda_link` wheel **unchanged** — pure Python, zero deps,
  [ADR-0002](../adr/0002-byte-identical-td-mirror.md)/[-0003](../adr/0003-library-install-sys-path-bootstrap.md)
  preserved. Optional extra `cuda-link[spout]` can pull it in for convenience.
- **License**: Spout2 is BSD-2-Clause — compatible with cuda-link's MIT. Bundle the
  Spout copyright notice.
- **Record an ADR** ("Optional native interop bridges; core stays pure Python") so
  the boundary is explicit and durable.

---

## 9. Testing strategy

- **TD as the round-trip harness**: TD speaks *both* Spout and cuda-link, so a
  single TD instance can validate every leg without a second app.
  - Egress: cuda-link tensor → `SpoutSender` → TD `Syphon Spout In TOP` → compare.
  - Ingress: TD `Spout Out TOP` → `SpoutReceiver` → tensor → compare.
- **Round-trip integrity**: known pattern → egress → ingress → assert bytes match
  per format (RGBA8 exact; fp16/fp32 exact; BGRA↔RGBA swap correct; no row flip).
- **Format matrix**: RGBA8 / RGBA16F / RGBA32F; BGRA vs RGBA; odd resolutions
  (pitch alignment); dynamic resize (Spout `IsUpdated` → re-register path).
- **Adapter matching**: assert the D3D11 device LUID == CUDA device LUID; force a
  mismatch in a test and confirm a clean error, not a silent host fallback.
- **Interop with real apps** (manual): Resolume source/output, OBS Spout plugin,
  UE Spout plugin — smoke-test each once.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Handle-type / adapter contract** (KMT vs NT mode; cross-adapter silent failure) — *highest probability* | Detect handle type at connect; read Spout's adapter index from its SHM; assert LUID match; round-trip on the exact GPU/driver before release. |
| **Format unsupported by CUDA interop** (3-ch, typeless, sRGB) | Restrict to fully-typed 4-channel formats; validate `GetSenderFormat()` and raise a clear error otherwise. |
| **The second copy** (high-level `SendTexture`/`ReceiveTexture`) | Accept for v1 (negligible at VJ resolutions); offer the 1-copy external-memory/keyed-mutex path in phase 2 only if profiled. |
| **Sync correctness on the 1-copy path** | Use keyed mutex (`...KeyedMutex` external semaphore); strictly paired acquire/release keys; never mix with a CUDA-only event across the boundary. |
| **Row origin / channel order bugs** | Fuse flip + BGRA↔RGBA into the surface-write kernel; cover in round-trip tests. |
| **Per-CPython-ABI build burden** | Native `.pyd` per Python minor; standard cibuildwheel matrix; Windows-only narrows it. |

---

## 11. Phased plan

1. **P0 — Egress, simple path.** pybind11 module: `SpoutSender.send(tensor)` via
   `cudaGraphicsD3D11RegisterResource` + `cudaMemcpy2DToArray` + `SendTexture`.
   Validate into TD/Resolume. *(The keystone — unlocks every app as a consumer.)*
2. **P1 — Ingress, simple path.** `SpoutReceiver.receive()` via `ReceiveTexture` +
   `cudaMemcpy2DFromArray` → cuda-link Exporter → torch tensor.
3. **P2 — Bridge-mode sidecar.** `python -m cuda_link_spout.bridge` wiring IPC↔Spout
   names, both directions.
4. **P3 — Optimization (only if profiled).** 1-copy external-memory / keyed-mutex
   paths; fused format/flip kernel.
5. **P4 — Packaging & ADR.** `cuda-link-spout` wheel matrix; ADR for the native
   boundary; docs + examples mirroring `INTEGRATION_EXAMPLES.md`.

---

## 12. Open questions

- Confirm whether `spoutDX::SendTexture` always performs an internal copy on the
  current SDK version, or can publish a caller-owned shared texture in place
  (decides whether the 1-copy optimized egress needs the lower-level API).
- Default output format: BGRA8 (Spout/VJ-native, one channel swap from RGBA) vs
  RGBA8 (cuda-link-native, no swap but some apps prefer BGRA)? Pick the default
  that minimizes total work for the common Resolume/OBS case.
- Should ingress reuse the existing `Exporter` verbatim, or a lighter internal
  publish path (the frame is already on-GPU)?
- Multi-GPU: do we ever need a Spout sender on a *different* GPU than the CUDA
  producer? (Currently no — single-GPU assumption holds; cross-GPU is a separate,
  deferred topic.)

---

## 13. CUDA 12 / 13 compatibility

The bridge (and cuda-link itself) must run on both **CUDA 12.x and CUDA 13.x**.
Findings (CUDA 13.0–13.3 release notes):

- **Runtime IPC survives.** `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` /
  `cudaIpcEventHandle` are **not** deprecated in CUDA 13. The "deprecated in 13,
  removed in 14" item is the **nv-p2p** kernel API set (GPUDirect RDMA for
  third-party drivers) — a different API cuda-link does **not** use. The core
  transport is safe.
- **Interop APIs unchanged.** `cudaGraphicsD3D11RegisterResource`,
  `cudaImportExternalMemory`, `cudaExternalMemoryGetMappedMipmappedArray`,
  external semaphores, and `cudaMemcpy2DToArray/FromArray` are all present and
  unchanged in CUDA 13 — the §5 bridge design holds verbatim.
- **ctypes structs are safe.** cuda-link does **not** materialize `cudaDeviceProp`,
  so CUDA 13's removal of deprecated `cudaDeviceProp` fields has **no effect**. The
  structs it does define (`cudaIpcMemHandle_t`/`cudaIpcEventHandle_t` = fixed
  64-byte opaque; `cudaMemcpy3DParms`; `cudaPointerAttributes`) are stable across
  12→13. *(Verify `cudaPointerAttributes` layout once on 13 to be safe.)*
- **The one required code change — the runtime loader.**
  `_load_cuda_runtime()` (`src/cuda_link/cuda_ipc_wrapper.py`) and its TD mirror
  (`td_exporter/CUDAIPCWrapper.py`) currently probe only `cudart64_12.dll`
  (+ `_11`/`_110` fallbacks) and hardcoded v12.x toolkit paths. **Add
  `cudart64_13.dll` to the bare-name list and `...\CUDA\v13.x\bin\` to the search
  paths.** Order: prefer the highest installed major that matches the process's
  existing CUDA (to avoid the error-400 "second cudart loaded alongside torch"
  hazard noted in `cuda_ipc_wrapper.py` — torch built against CUDA 13 will already
  have `cudart64_13` resident, so the loader should match it).
- **Bridge kernels target sm_75+.** CUDA 13 dropped offline compile for
  Maxwell/Pascal/Volta. The optional fused format/flip surface-write kernel (§5.4)
  should target **Turing (sm_75) and newer** — a non-issue for the modern RTX/pro
  GPUs this targets.
- **Deployment note.** CUDA 13 no longer bundles the Windows display driver with
  the toolkit; it also raises the minimum driver. Bump the documented driver
  requirement and keep the "install the driver separately" note in the deps docs.
- **Watch-item (not a blocker).** NVIDIA's docs continue to flag Windows IPC as
  "supported for compatibility, with a performance cost." cuda-link has
  empirically validated the legacy path on WDDM (ADR-0004); keep monitoring across
  13.x point releases.

**Net:** CUDA 13 support is **low-risk** — one loader change (mirrored TD-side),
one struct re-verification, an sm_75+ kernel target, and a test-matrix/driver-doc
bump. No architectural change to either cuda-link or the bridge.

## 14. Sources

- Spout2 SDK & license (BSD-2): `github.com/leadedge/Spout2`;
  `spoutDX.h` (verified: `OpenDirectX11(ID3D11Device*)`, `SendTexture`,
  `ReceiveTexture`, `GetSenderHandle/Format`, frame sync);
  `SpoutLibrary.h` (`GetSpout()` vtable factory).
- Spout DX11 shared-texture internals & KMT `uint32_t` handle:
  DeepWiki SpoutDirectX Core; `SpoutSenderNames.h`.
- CUDA interop: Runtime API — D3D11 Interop (`cudaGraphicsD3D11RegisterResource`,
  `cudaGraphicsSubResourceGetMappedArray`); External Resource Interop
  (`cudaImportExternalMemory`, `cudaExternalMemoryGetMappedMipmappedArray`,
  `cudaExternalSemaphoreHandleTypeKeyedMutex`); `cudaMemcpy2DToArray`/`FromArray`;
  Texture Object Mgmt (array opacity, pitch alignment). `docs.nvidia.com/cuda`.
- NVIDIA `cuda-samples` `simpleD3D11` (keyed-mutex interop pattern).
- App support: `resolume.com/support/en/syphonspout`,
  `docs.derivative.ca/Syphon_Spout_Out_TOP`, OBS/UE/Unity Spout plugins.
- Strategic context: [`plugin-expansion-analysis.md`](plugin-expansion-analysis.md).

*Some NVIDIA doc pages return HTTP 403 to automated fetch; API names/constraints
were cross-checked across the runtime/driver references, cuda-samples, and MS DXGI
docs. Re-verify exact constraint wording against a live page before implementation.*
