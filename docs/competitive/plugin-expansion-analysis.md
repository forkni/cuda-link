# Plugin & C++ Expansion Analysis — Connecting cuda-link to Resolume, Unreal, Spout & NDI

**Status**: Research / strategy brief (not a committed roadmap)
**Date**: 2026-06-25
**Scope**: Feasibility of expanding cuda-link beyond TouchDesigner↔Python into the
wider real-time-AI-VJ ecosystem (Resolume, Unreal Engine, OBS, Notch, Unity) and
the de-facto interchange protocols (Spout, NDI) — via optional plugins / native
(C++) bridge components.
**Premise from the request**: the [pure-Python / zero-dependency rule
(ADR-0006)](../adr/0006-stay-pure-python-no-rust.md) **may be relaxed** for an
*optional, separately-distributed* plugin layer — the core wheel stays pure
Python; native code is additive.

> **Companion note.** This brief was requested alongside a
> `daydream-scope-analysis.md` that is **not present in the repository** (it was
> referenced but never committed to this branch). The competitive framing below
> is reconstructed from the Daydream-Scope "advantages A–G" table supplied in the
> request plus first-party research into Daydream Scope's actual I/O. If the full
> scope analysis is recovered, cross-check §3 against it.

---

## 0. TL;DR — the one reframing that changes the strategy

There are **two different "zero-copy GPU sharing" problems**, and conflating them
is the trap:

| | **App ↔ App texture sharing** | **App ↔ ML-runtime tensor sharing** |
|---|---|---|
| Example | Scope → Resolume layer | TD frame → `torch.Tensor` in a diffusion process |
| Already solved by | **Spout** (sub-ms, no CPU, on every Windows VJ app) | **cuda-link** |
| What the consumer wants | a *sampleable D3D/GL texture* | a *CUDA device pointer / torch tensor* |
| cuda-link's edge | **none a VJ would perceive** — Spout is already GPU-speed | **decisive** — Spout into Python *still* needs a GL/DX→CUDA interop copy to become a tensor; cuda-link lands the frame as a tensor with zero such copy |

**Conclusion:** cuda-link should *not* try to make Resolume/Unreal "speak CUDA
IPC" — they never will, and Spout already gives them GPU-speed sharing. The
correct expansion is:

1. **Keep CUDA IPC as the premium core** for the GPU→ML boundary (its defensible,
   uncommoditised niche).
2. **Add a CUDA↔Spout bridge** as the *one keystone adapter*. Spout is the single
   interchange that simultaneously unlocks Resolume, Unreal, OBS, Notch, Unity,
   MadMapper, vvvv and TD — and it is architecturally the closest cousin to
   cuda-link (Windows, same-machine, GPU texture share).
3. Treat **NDI** as a *separate, cross-machine* egress adapter — useful for reach
   (row A), but it breaks the zero-copy/lossless story by construction and should
   never sit on the latency-critical path.
4. Native **Unreal** and **Resolume FFGL** plugins are *phase-2 optimisations* of
   the Spout route, not prerequisites.

Everything below is the evidence for, and the engineering shape of, that
conclusion.

---

## 1. The architectural reality (why a bridge is required, and what it costs)

cuda-link shares **linear** `cudaMalloc` device memory through **CUDA Runtime
IPC**: a 64-byte `cudaIpcMemHandle` + a 64-byte `cudaIpcEventHandle` published in
a named shared-memory segment (see [ARCHITECTURE.md](../ARCHITECTURE.md)). Two
hard facts constrain every possible expansion:

### Fact 1 — CUDA IPC is CUDA-only
`cudaIpcOpenMemHandle` can only be called by another **CUDA** process. A D3D /
OpenGL / Vulkan application (Resolume, Unreal) **cannot consume a
`cudaIpcMemHandle`**. Crossing into a graphics process requires that the *texture*
own a **DXGI / NT shared handle** (`IDXGIResource1::CreateSharedHandle` /
`ID3D12Device::CreateSharedHandle`) — a different handle world entirely.

### Fact 2 — linear memory ≠ a texture (the mandatory copy)
Textures are backed by **CUDA arrays** (`cudaArray`) — an *opaque, swizzled/tiled*
layout optimised for texture fetch. You cannot take a pointer into one, and you
cannot reinterpret linear memory as an array. **Every** bridge from cuda-link's
linear buffer to a sampleable texture therefore costs **one device-to-device
"de-swizzle" copy per frame** — `cudaMemcpy2DToArray` (out) /
`cudaMemcpy2DFromArray` (in), or an equivalent surface-write kernel.

> So the honest headline is **not** "zero-copy into Resolume." It is **"one
> on-GPU reformat copy + GPU interop"** — ~15–30 µs for a 1080p BGRA frame,
> negligible against a frame interval, but real and worth stating plainly. True
> zero-copy survives only on the CUDA↔CUDA path (cuda-link's existing core).

### The bridge chain (validated against NVIDIA docs)
A `cuda-link → graphics-app` bridge process must:

1. `cudaIpcOpenMemHandle(...)` → linear device pointer (existing path).
2. Create a **shareable** graphics texture on the **same GPU**:
   - D3D11: `CreateTexture2D` + `D3D11_RESOURCE_MISC_SHARED_NTHANDLE [| _KEYEDMUTEX]`
   - D3D12: `CreateCommittedResource` + `D3D12_HEAP_FLAG_SHARED`
3. Import that texture into CUDA — **modern path**:
   `cudaImportExternalMemory` → `cudaExternalMemoryGetMappedMipmappedArray` →
   `cudaArray_t`. (Legacy path: `cudaGraphicsD3D11RegisterResource` →
   `cudaGraphicsSubResourceGetMappedArray`.)
4. **Copy** linear → array: `cudaMemcpy2DToArray(..., cudaMemcpyDeviceToDevice)`
   (Fact 2), bracketed by cross-API sync (step 6).
5. Publish the **texture's DXGI/NT handle** downstream — hand to Unreal/Resolume
   directly, **or** feed Spout2 / NDI.
6. **Sync across the boundary** with a *shared* primitive, **not** `cudaIpcEvent`:
   - **D3D12 fence** as `cudaExternalSemaphoreHandleTypeD3D12Fence` (preferred,
     monotonic counter), or
   - **DXGI keyed mutex** as `cudaExternalSemaphoreHandleTypeKeyedMutex` (the
     classic Spout/`simpleD3D11` path).

The reverse (graphics → CUDA → IPC) is the mirror: import the app's shared
texture, `cudaMemcpy2DFromArray` into a linear buffer, then `cudaIpcGetMemHandle`
for downstream CUDA/torch consumers.

**Gotchas that will bite (all confirmed in NVIDIA / MS docs):**
- **Single GPU only** — imported memory & semaphores must be on the same adapter
  that created them; on multi-GPU/Optimus, pin the CUDA device to the texture's
  LUID. (cuda-link is already single-GPU, so this is fine.)
- **WDDM, not TCC** — TCC disables DXGI interop; the bridge GPU must be in WDDM.
- **Formats**: `R32G32B32A32_FLOAT` (fp32 RGBA) and `R16G16B16A16_FLOAT` (fp16)
  are importable; **3-channel and `*_TYPELESS`/sRGB-typed DXGI formats are not** —
  use fully-typed 4-channel and match `cudaChannelFormatDesc` exactly or the map
  call fails. (cuda-link is already RGBA-centric.)
- **D3D12 committed resources require `cudaExternalMemoryDedicated`** — common
  silent import failure if omitted.
- **NT handle ownership is not transferred** on import — you still `CloseHandle`;
  `CreateSharedHandle` is callable only once per resource.

*Primary sources:* CUDA Runtime API — External Resource Interop
(`cudaImportExternalMemory`, `…GetMappedMipmappedArray`, external semaphores);
CUDA Programming Guide — Graphics Interop (same-GPU constraint);
`cudaMemcpy2DToArray`/`FromArray`; NVIDIA `cuda-samples` `simpleD3D11`
(keyed mutex) and `simpleD3D12` (fence); MS DXGI `CreateSharedHandle` /
`IDXGIKeyedMutex`.

---

## 2. Precedent — this exact pattern already exists (and leaves a gap)

**SplatBus** (arXiv 2601.15431, Jan 2026; `github.com/RockyXu66/splatbus`) is the
strongest validation of cuda-link's whole approach. It shares contiguous CUDA
buffers using **the same primitives cuda-link uses** — `cudaIpcMemHandle` + an
IPC-capable `cudaIpcEventHandle` + metadata (resolution / format / pitch) — to
multiple viewer clients, and explicitly names **Unreal Engine, Unity, Blender and
OpenGL** as targets. Its shipped **Unity** client uses **CUDA-D3D11 interop** on
Windows (precisely the bridge chain in §1).

Two takeaways:
- The design is **proven**, not speculative — an independent project landed on the
  identical transport for a multi-engine consumer set.
- SplatBus names a UE client as a goal but **has not shipped one**, and bridges to
  **none** of the VJ tools (no Spout). **There is no off-the-shelf CUDA-IPC ↔
  Spout bridge, and no CUDA-IPC ↔ Unreal zero-copy plugin in existence.** That is
  the open niche this expansion would fill.

Adjacent precedents confirming feasibility: NVIDIA's own DLSS / Omniverse / RTXGI
UE plugins (deep RHI + CUDA-interop access), community `cuda_ue4_linux` /
`ue4-gpgpu-plugin` (CUDA-in-UE on the GPU), and NVIDIA dev-forum reports of
importing UE render targets into CUDA with no CPU round-trip.

---

## 3. Strategic positioning vs Daydream Scope (the A–G table)

Mapping the supplied Scope-advantage table to what this expansion actually moves:

| # | Scope advantage | Does plugin/C++ expansion close it? | Notes |
|---|---|---|---|
| **A** | Transport breadth (Spout, Syphon, NDI, WebRTC, DMX) | **Partially — the high-value 80%.** | A CUDA↔**Spout** bridge + an **NDI** egress adapter cover the two transports that matter for VJ reach. Syphon is macOS (cuda-link is Windows-only → N/A). WebRTC/DMX are out of scope for a GPU-frame library. |
| **B** | Cross-app reach (Resolume, Unity, Unreal, OBS) | **Yes — via one move.** | All four speak **Spout**. A single CUDA↔Spout bridge reaches Resolume, Unity, Unreal, OBS, Notch, MadMapper, vvvv *simultaneously*. This is the highest-leverage item in the whole analysis. |
| **C** | Distribution (desktop app + cloud templates) | **No.** | Product/packaging concern, orthogonal to the transport layer. Out of scope here. |
| **D** | Modality (native autoregressive video models) | **No.** | Model concern, not a transport concern. |
| **E** | Model breadth / plugin system | **No.** | Same — cuda-link is plumbing, not a model host. |
| **F** | Remote/cloud inference | **No** (and arguably anti-thesis). | cuda-link's value *is* local zero-copy. NDI/WebRTC adapters are the only cross-machine concession. |
| **G** | Polished onboarding | **No.** | UX concern. |

**Honest competitive read.** Daydream Scope *already ships Spout/Syphon* and is
candid that "the texture never leaves the GPU … under a millisecond" — i.e. **Scope
has already commoditised the app↔app GPU-sharing story that a naïve reading of
cuda-link's README claims as unique.** Where cuda-link remains genuinely
differentiated:

1. **The GPU → ML-framework boundary.** Spout hands you a *DX/GL texture*; to use
   it in StreamDiffusion you must do a GL/DX→CUDA interop copy to get a
   `torch.Tensor`. cuda-link hands you the **tensor directly, zero-copy**. Scope's
   "zero latency" is app-to-app; it does **not** close the
   texture→tensor boundary. This is the moat.
2. **Lossless full-precision** (fp32 / fp16 / arbitrary channels, ring buffer,
   GPU-side IPC-event sync) for ML *intermediates* (latents, depth, control
   signals) — not just 8-bit display textures.
3. **Decisive win over NDI** (which StreamDiffusionTD still uses *locally*): no
   encode/decode, no compression artifacts, no ~10–60 ms network tax.

**So the expansion's job is not to beat Spout — it is to *adopt* Spout as an
egress while keeping CUDA IPC as the premium ML-grade core.** Position: *"the
zero-copy bridge between a host app's GPU frames and a Python ML runtime's native
tensors — then Spout/NDI out to the rest of your rig."*

---

## 4. Per-target findings

### 4.1 Spout — the keystone (build this first)
- **Mechanism**: a shared **DirectX 11 texture**; the sender writes the share
  handle into named shared memory, receivers `OpenSharedResource`. OpenGL
  apps bridge via `WGL_NV_DX_interop`. Sync = a **named mutex** + a named
  frame-count event (keyed mutex is opt-in, off by default).
- **Handle type — the lucky alignment**: Spout's default stores `shareHandle` as a
  **`uint32_t`** → it is the *legacy global (KMT)* shared handle, which maps
  **exactly** onto CUDA's `cudaExternalMemoryHandleTypeD3D11ResourceKmt`. The
  interop path is real and direct. (NT-handle mode is an opt-in `bNThandle` +
  DX11.1 path — detect and handle both.)
- **SDK**: C++, **BSD-2-Clause** (`github.com/leadedge/Spout2`). Crucially there is
  a **stable C ABI — `SpoutLibrary`** — callable from ctypes/cffi or a small
  native module without C++ name-mangling. `spoutDX` gives native D3D11
  (`SendTexture(ID3D11Texture2D*)` / `ReceiveTexture(...)`).
- **App support**: TouchDesigner, Resolume, Notch, Unreal (plugin), OBS (plugin),
  Unity (KlakSpout), Max/MSP, Processing, MadMapper, vvvv — all native/standard.
- **Existing Python bridges** (`Python-SpoutGL`, `pyspout`, `Spout-for-Python`)
  all go through **OpenGL or CPU pixel buffers** — none bridge CUDA *device*
  memory. A CUDA-device ↔ Spout bridge would be **new work**.
- **Verdict: MODERATE.** Pieces fit better than expected (KMT↔KMT alignment, BSD
  license, C ABI, TD as a ready test harness). Not "Easy" only because it's novel
  at the CUDA-memory level and you give up literal zero-copy (one D2D array copy +
  format coercion).
- **Biggest risk**: the **shared-handle / device-affinity contract** — KMT vs
  NT-handle mode, and same-adapter requirement. Validate handle type, adapter
  index (Spout stores it in SHM) and a round-trip frame on the exact driver/GPU
  before committing.

### 4.2 Resolume — reachable today via Spout; FFGL is a phase-2 option
- **I/O**: **Spout in/out** is the GPU-native path (Spout *input is always on*;
  output via Output → Texture Sharing). NDI in/out exists but is network/CPU.
  Resolume runs **OpenGL** internally on Windows.
- **Design A (recommended)** — `cuda-link → CUDA-interop → Spout sender → Resolume
  Spout source`. Uses documented, always-on input; resilient to Resolume updates;
  the exact pattern the whole TD/StreamDiffusion scene already uses
  (TouchDiffusion, FluxRT). Capturing Resolume *output* is symmetric and trivial
  (enable Spout output, receive, interop back to CUDA).
- **Design B (phase-2)** — an **FFGL 2.x** source plugin (C++/OpenGL,
  `github.com/resolume/ffgl`) running *inside Resolume's GL context*, opening the
  CUDA IPC handle and doing CUDA-GL interop directly (`ProcessOpenGL` hands real
  `GLuint` handles). Saves one interop hop and removes the external Spout sender,
  **but** runs CUDA inside Resolume's process (crash blast-radius), demands exact
  same-GPU/context discipline, and adds an FFGL/version maintenance treadmill.
  Still cannot avoid the linear→array copy.
- **Resolume Wire**: ISF/GLSL only — **no native node SDK, no external-GPU-handle
  ingest**. Not a viable injection point.
- **Verdict**: feeding in and capturing out are both **feasible and fully on-GPU
  today via Spout (Design A)**. FFGL (Design B) only if profiling shows the Spout
  hop is a real bottleneck.

### 4.3 Unreal Engine — Spout shim (v1), native external-memory plugin (v2)
- **RHI**: UE5 on Windows defaults to **D3D12** (Nanite/Lumen/RT require it); D3D11
  is a fallback. So the bridge almost always talks D3D12 — the *cleanest* CUDA
  interop target.
- **RHI surface a plugin needs** (all exist): `RHIGetNativeDevice()` →
  `ID3D12Device*`; `FRHITexture::GetNativeResource()` → `ID3D12Resource*`;
  `RHICreateTexture2DFromResource(...)` to wrap a native texture back into a
  `FRHITexture` for materials/MediaTextures.
- **Architecture B (recommended native)** — **UE owns** a shared D3D12 texture
  (`D3D12_HEAP_FLAG_SHARED`) + a D3D12 fence; **cuda-link imports** via
  `cudaImportExternalMemory` (`…D3D12Resource`) + `cudaImportExternalSemaphore`
  (`…D3D12Fence`) and writes/reads directly. Idiomatic (UE stays the resource
  owner), lowest-risk sync, minimal copy. Cost: cuda-link's producer must learn a
  **second handle type** (Win32/D3D12 external memory) beyond `cudaIpcMemHandle` —
  a real but contained extension.
- **Architecture A** — CUDA stays source of truth, UE imports a CUDA-registered
  texture. Less idiomatic for UE devs.
- **v1 (Low–Medium)**: skip UE C++ entirely — `cuda-link → CUDA↔Spout shim →
  existing UE Spout plugin` (e.g. `UE5_Spout2_DX12`). Zero per-UE-version recompile
  on your side. This *is* the Spout keystone (§4.1) reused.
- **v2 (Medium–High)**: native plugin (Architecture B). Real C++/RHI work, render-
  /RHI-thread sync, and the **per-UE-version recompile + CUDA-version support
  matrix** treadmill (normal for niche GPU plugins — NVIDIA ships DLSS per-UE-
  version off Fab; a source-plugin drop-in avoids prebuilt binaries for v1).
- **Verdict**: clearly feasible; ship the Spout shim first, invest in the native
  external-memory plugin only when a true single-copy path is demanded.

### 4.4 NDI — a different niche (cross-machine), not part of the core
- **Nature**: network video-over-IP, **compressed** (SpeedHQ for "Full"; H.264/HEVC
  for HX/HX2/HX3), **host-memory-bound at the SDK boundary** — `p_data` is a *CPU
  pointer*, and NDI's own docs instruct you to "download it to system memory."
- **No GPU-direct ingress** of arbitrary device memory; GPU involvement is limited
  to codec engines (NVDEC) and optional pre-download color conversion. True
  GPU-direct networking is **Rivermax/GPUDirect**, a *separate* NVIDIA SDK NDI does
  not expose.
- **Latency/bandwidth**: ~10–60 ms (Full/HX3), 100–300 ms (HX2 long-GOP);
  ~100–150 Mbps at 1080p60 — vs a sub-ms local D2D copy. The **D2H copy is
  unavoidable** (cuda-link already measures ~1.3 ms D2H at 1080p fp32) and is the
  *cheapest* part of an NDI bridge (then color-convert + encode + network + decode).
- **License**: standard SDK royalty-free; Advanced SDK needs a vendor ID for
  commercial use.
- **Verdict**: support NDI only as an explicit **cross-machine egress/ingress
  adapter**, with the D2H+encode tax stated up-front. **The moment NDI is in the
  path, "zero-copy" and "lossless" are gone.** It belongs in transport-breadth
  (row A), never on the latency-critical core.

---

## 5. Proposed phased roadmap

> All phases are **optional, separately-distributed** components. The pure-Python
> ~30 KB core wheel and the classic Text-DAT `.tox` are untouched (ADR-0002 /
> -0003 preserved). Native code ships as its own artifact(s).

| Phase | Deliverable | Unlocks | Effort | Difficulty | Dependency footprint |
|---|---|---|---|---|---|
| **0** | **`cuda-link-spout` bridge** — sidecar (or module) doing CUDA-IPC ↔ D3D11 shared texture ↔ Spout, both directions | Resolume, Unreal, OBS, Notch, Unity, MadMapper, vvvv, TD — *all at once* | M | Moderate | Spout2 (BSD-2), D3D11; optional `SpoutLibrary` C ABI from Python |
| **1** | **`cuda-link-ndi` egress/ingress adapter** | Cross-machine reach; OBS/Resolume/UE over LAN | S–M | Low–Moderate | NDI SDK (royalty-free) + mandatory D2H |
| **2a** | **Native UE plugin** (External-Memory: UE-owned shared D3D12 texture + fence; cuda-link learns D3D12/Win32 handle import) | Single-copy UE path; in-editor live tensor texture | L | Med–High | UE C++ per-version build; CUDA external-memory API |
| **2b** | **Resolume FFGL 2.x source plugin** (in-GL-context CUDA-GL interop) | One fewer hop into Resolume; native layer-graph node | M–L | Med–High | FFGL SDK (C++/OpenGL), CUDA-GL |
| **3** | (Watch) Generic **DXGI/NT shared-handle export** from cuda-link's exporter (so any D3D app can import without a sidecar) | Direct app import; foundation for 2a/2b | M | Moderate | adds a second handle type to the exporter |

**Why Phase 0 is the keystone:** one bridge, built against the one protocol every
target shares, converts cuda-link from "TD↔Python only" to "reaches every Windows
VJ/engine app" — closing the bulk of rows A and B with a single artifact. Phases 2a/2b
are *latency optimisations* of that same path, justified only by profiling.

---

## 6. Dependency & distribution model (how to relax ADR-0006 cleanly)

The request authorises relaxing the zero-dependency rule "in that case as it would
be plugin expansion." The clean way to honour both that and ADR-0006:

- **Core (`cuda_link` wheel)**: unchanged — pure Python, zero required deps, ~30 KB,
  `.tox` mirror intact. Its *defensible niche* (CUDA↔CUDA, GPU→tensor) needs no
  native code.
- **Bridges**: ship as **separate optional packages/artifacts**
  (`cuda-link-spout`, `cuda-link-ndi`) and **separate plugin binaries**
  (`.uplugin`, `.dll` FFGL). They depend on Spout2/NDI/UE — but a user who only
  wants TD↔Python never installs them and never pays the dependency cost.
- This mirrors the existing dual-distribution split (consumer wheel vs TD `.tox`)
  and ADR-0006's own "narrow, optional native extension" escape hatch — extended
  from *performance* hot-paths to *interop* reach.
- **Recommendation**: record this as a new ADR (e.g. ADR-0007 — "Optional native
  interop bridges; core stays pure Python") so the boundary is explicit and the
  "should the core take a C++ dep?" question is not re-litigated. The answer is
  **no — the core stays pure; bridges are additive and opt-in.**

---

## 7. Consolidated risks

1. **Shared-handle / device-affinity contract (Spout)** — KMT vs NT-handle mode;
   same-adapter requirement; truncated `uint32_t` handle assumptions. *Highest-
   probability failure point.* Mitigate: detect handle type, read Spout's adapter
   index from SHM, round-trip on the exact GPU/driver early.
2. **The mandatory de-swizzle copy** — there is no literal zero-copy linear→texture
   path. Set expectations: "one GPU reformat + interop," not "zero-copy into
   Resolume." (Still excellent; just not the README's CUDA↔CUDA claim.)
3. **Cross-API sync correctness** — must use a *shared* primitive (D3D12 fence /
   keyed mutex), not `cudaIpcEvent`; mismatched keys deadlock; fence values must be
   strictly monotonic per frame.
4. **UE per-version treadmill** — RHI is not ABI-stable; native plugin needs a
   recompile per UE minor + a CUDA-version matrix. Source-plugin drop-in for v1
   sidesteps prebuilt binaries.
5. **FFGL in-process CUDA (Resolume Design B)** — a CUDA fault can crash Resolume;
   context/device discipline is unforgiving. Phase-2 only.
6. **NDI mis-positioning** — if marketed as part of the "zero-copy" story it
   undermines the brand; it is lossy + networked by construction. Keep it labelled
   as the cross-machine adapter.
7. **License hygiene** — Spout2 BSD-2 (fine), NDI SDK terms / Advanced-SDK vendor
   ID for commercial, Daydream Scope is **CC BY-NC-SA 4.0** (non-commercial — do
   **not** copy Scope code into cuda-link).

---

## 8. Recommendation

1. **Build Phase 0 (CUDA↔Spout bridge).** It is the single highest-leverage move:
   one moderate-difficulty artifact unlocks Resolume, Unreal, OBS, Notch, Unity and
   more, and is architecturally aligned with cuda-link. Use TD (which speaks *both*
   Spout and cuda-link) as the test harness.
2. **Reframe the positioning** around the GPU→ML-framework boundary (the
   uncommoditised moat), with Spout/NDI as egress to the wider rig — not as a
   claim to out-zero-copy Spout for app↔app sharing.
3. **Add Phase 1 (NDI)** for cross-machine reach, clearly labelled as the lossy/
   networked adapter.
4. **Defer Phases 2a/2b** (native UE / FFGL) until profiling proves the Spout hop
   is a real bottleneck for a real user.
5. **Record an ADR** for the optional-native-bridge boundary so the pure-Python
   core stays sacrosanct and the decision is durable.

---

## Appendix — key sources

**CUDA interop:** CUDA Runtime API — External Resource Interop
(`cudaImportExternalMemory`, `…GetMappedMipmappedArray`, external semaphores);
CUDA Programming Guide — Graphics Interop (same-GPU constraint); Runtime API
D3D11 / OpenGL interop; `cudaMemcpy2DToArray`/`FromArray`; NVIDIA `cuda-samples`
`simpleD3D11` (keyed mutex) & `simpleD3D12` (fence).
*docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html · github.com/NVIDIA/cuda-samples*

**Spout:** `github.com/leadedge/Spout2` (BSD-2; `SpoutLibrary` C ABI; `spoutDX`);
`SpoutSenderNames.h` (`uint32_t` share handle → KMT); DeepWiki SpoutDirectX Core.

**Resolume:** `resolume.com/support/en/syphonspout`; FFGL SDK
`github.com/resolume/ffgl` (FFGL 2.x, C++/OpenGL, `ProcessOpenGL`); Wire/ISF
`resolume.com/support/en/isf`; AI bridges `github.com/olegchomp/TouchDiffusion`,
`github.com/tensorforger/FluxRT`.

**Unreal:** `FRHITexture::GetNativeResource`, `RHIGetNativeDevice`,
`RHICreateTexture2DFromResource` (UE5 API docs); `github.com/GPUbrainStorm/UE5_Spout2_DX12`;
Epic NDI/Blackmagic media references; `developer.nvidia.com/rtx/dlss` (per-version distribution).

**NDI:** `docs.ndi.video` — What is NDI / Frame Types / Performance & Implementation /
Licensing; bandwidth white-paper; `developer.nvidia.com/networking/rivermax`.

**Precedent / ecosystem:** SplatBus — arXiv 2601.15431, `github.com/RockyXu66/splatbus`;
Daydream Scope — `github.com/daydreamlive/scope`, `blog.daydream.live/spout-and-syphon-in-scope-zero-latency-on-your-machine/`
(CC BY-NC-SA 4.0); StreamDiffusionTD — `dotsimulate.com/docs/streamdiffusiontd`
(uses NDI locally); ComfyStream — `github.com/livepeer/comfystream`.

*Several first-party pages (docs.nvidia.com, docs.daydream.live, dotsimulate.com)
returned HTTP 403 to automated fetch; their content was corroborated via search
excerpts and cross-referenced sources rather than quoted verbatim. Re-verify exact
API constraint wording against a live NVIDIA docs page before implementation.*
