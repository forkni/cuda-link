# PLAN-001: C++ Custom TOP sender/receiver inside TouchDesigner

**Status**: Proposed
**Date**: 2026-07-04
**Size**: L (4–6 weeks, 6 phases)
**Depends on**: soft — PLAN-002 (native CI muscle); ADR-0009 must land with Phase 0/1
**Related ADRs**: [ADR-0007](../adr/0007-spout-as-launcher-not-transport.md) (in-process-native prior),
[ADR-0009](../adr/0009-cpp-custom-top-in-process.md) (this plan's decision record)

---

## Goal & non-goals

**Goal**: replace the Python TD-side hot path with two native Custom TOPs, eliminating
the `top.cudaMemory()` staging alloc+copy and the per-frame Python scaffolding, while
remaining **byte-compatible** with the v0.5.0 SHM wire protocol
(`src/cuda_link/shm_protocol.py`) so every existing Python consumer works unchanged.

**Target**: sender cook cost **≤ 90 µs p50 at 1080p float32** (baseline ~182–200 µs).

**Non-goals**: replacing the Python `.tox` (it remains the zero-install reference
implementation); changing the wire protocol; touching the Python-process consumer.

## Baseline numbers (why this is worth it)

Measured in TD 2025.32820, RTX 4090, 1080p float32
(`benchmarks/results/td/cell_C_td/textport.txt`, `cell_D_td/textport.txt`):

| Component | Cost | What it is |
|---|---|---|
| `cudaMemory()` interop | 80–105 µs | TD Python API **allocates fresh CUDA memory and copies the texture** on every call ([TOP Class docs](https://docs.derivative.ca/TOP_Class)) |
| Ring-buffer memcpy | 28–46 µs | Second D2D copy into the IPC slot |
| Python scaffolding | ~40 µs | pre/post, SHM write, event record, flush probe — ctypes + interpreter on TD's main cook thread |
| **Total** | **~182–200 µs** | |

A C++ TOP reads the input texture **in place** via `getCUDAArray()` — no per-frame
allocation, no staging copy — so the pipeline drops from *two copies + one alloc + Python
dispatch* to *one copy* (cudaArray → linear IPC ring slot, still required because
`cudaIpcGetMemHandle` only works on `cudaMalloc`'d linear memory).

## Architecture decisions

### D1 — Two op types, two thin DLLs, one shared static core

`CudaLinkOutTOP.dll` (opType `Cudalinkout`, sender, `minInputs = maxInputs = 1`) and
`CudaLinkInTOP.dll` (opType `Cudalinkin`, receiver, 0 inputs). TD registers one custom
operator per plugin DLL via `FillTOPPluginInfo`, and the two roles have incompatible
input arity and cook semantics (sender: `cookEveryFrame` while Active; receiver: cook
driven by frame arrival). ~90% of the code lives in a static lib `cudalink_topcore`
(protocol writer/reader, ring management, IPC handle export/open, doorbell signal).
Matches TD idiom (`Syphon Spout In` / `Out` are separate operators).

*Rejected: one DLL with a `Mode` menu param — wrong input-arity semantics for one of the
two roles, muddier UX, no code-size benefit given the shared static lib.*

### D2 — Standalone CMake is the build of record; PluginBuilder is the dev loop

CMake ≥ 3.24, C++17, MSVC VS2022, `CMAKE_CUDA_ARCHITECTURES`, and **release CRT
enforced** (`CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL`) — a debug-CRT plugin silently
fails to load against TD's release runtime. Mirrors the `spout/` precedent and is
CI-buildable on a Windows runner without TD installed.

[IntentDev/PluginBuilder](https://github.com/IntentDev/PluginBuilder) (MIT) is the
recommended **inner dev loop**: CMake+Ninja hot-reload from inside TD with sub-second
rebuilds and a `CudaTOP` template — it defeats the DLL-locked-while-TD-open cycle.
It is *not* the build of record: it requires TD ≥ 2023.11600 + Ninja ≥ 1.12 and currently
targets CUDA 11.8 (needs adaptation for TD 2025 / CUDA 12.8).

*Rejected: per-sample `.vcxproj` like
[CustomOperatorSamples](https://github.com/TouchDesigner/CustomOperatorSamples) (no CI
story, `CUDA X.X.props` version churn); PluginBuilder-only (not CI-runnable).*

### D3 — Protocol parity via a TD-free, CUDA-free layout core + golden bytes

`shm_layout.hpp` / `ring_writer.cpp` / `ring_reader.cpp` operate on an opaque
`uint8_t*` region and opaque 64-byte handle blobs — no TD headers, no cudart. This makes
the wire-protocol code unit-testable **on the existing Ubuntu no-GPU CI**:

- **Golden-bytes tests, both directions**: `tests/golden/shm_v<version>.bin` is generated
  once from Python `shm_protocol.py`; a C++ unit test asserts the C++ writer reproduces
  it byte-for-byte, and a pytest asserts Python `shm_protocol` parses a dump produced by
  the Linux build of `tools/protocol_dump` (fixed magic `0x43495044`, synthetic handle
  patterns, deterministic metadata).
- **Publish-ordering contract test**: the C++ writer implements the same sequence as
  `publish_frame()` — slot payload → metadata → `write_idx` **last**, with
  `std::atomic_thread_fence(std::memory_order_release)` + atomic store (the C++
  equivalent of the Python `threading.Lock` release fence). A checker-thread unit test
  validates no torn publish.

### D4 — Dual CUDA build matrix keyed to TD versions

Plugin CUDA **major must match TD's bundled CUDA**
([TD CUDA docs](https://docs.derivative.ca/CUDA)):

| TD build series | Bundled CUDA | Artifact |
|---|---|---|
| 2023.1xxxx | 11.8 | `CudaLink{In,Out}TOP-cu118.dll` (sm_50…sm_75 + PTX) |
| 2025.3xxxx | 12.8 | `CudaLink{In,Out}TOP-cu128.dll` (sm_50…sm_90 + PTX) |

CMake option `CUDALINK_TOP_CUDA_MAJOR` selects the toolkit; the CUDA 11.8 leg wires
`-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` for current
MSVC. Runtime `setAPIVersion(TOPCPlusPlusAPIVersion)` gate plus a `cudaRuntimeGetVersion`
major check → error badge + refuse to cook on mismatch. The plugin does **not** ship
cudart — TD already loads a matching runtime.

### D5 — Repo layout: `cpp_top/` at repo root (sibling of `spout/`)

```text
cpp_top/
  CMakeLists.txt
  README.md                          # build, deploy, dev loop, gotchas
  vendor/td/2023/  vendor/td/2025/   # vendored TD CPlusPlus headers per matrix leg
                                     # (from Derivative's CustomOperatorSamples; attribution in README)
  src/core/    shm_layout.hpp  ring_writer.{h,cpp}  ring_reader.{h,cpp}
               ipc_ring.{h,cpp}  doorbell_win.{h,cpp}  cuda_check.h
  src/out_top/ CudaLinkOutTOP.{h,cpp}  array_copy.cu   # cudaArray -> linear
  src/in_top/  CudaLinkInTOP.{h,cpp}   linear_to_array.cu
  tools/protocol_dump/main.cpp
  tests/       test_layout.cpp  test_publish_ordering.cpp
```

### D6 — Copy technique: `cudaMemcpy2DFromArray` / `cudaMemcpy2DToArray` D2D

The byte-faithful path used by [DBraun/PyTorchTOP](https://github.com/DBraun/PyTorchTOP).
Surface-object kernels (`surf2Dread`/`surf2Dwrite`, per CannyEdgeTOP's `GpuUtils.cu`) are
reserved as the fallback **only if** the spike shows format conversion or Y-flip is
needed — TD's cudaArray origin is top-left; a raw memcpy preserves TD's orientation, so
flip policy must match what `top.cudaMemory()` produces today (verify in Phase 0 against
a Python receiver).

### D7 — Coexistence: the `.tox` is NOT deprecated

The Python `.tox` (CUDAIPCExtension + engines) remains the zero-install reference path
and the only path on TD builds without the plugin. The TOPs mirror the COMP's parameter
names exactly — `Ipcmemname`, `Active`, `Numslots` (2–5), `Cudadevice`, `Debug` — with
status via `getErrorString`/`getWarningString` badges + an Info DAT
(`getInfoDATEntries`), and stats via Info CHOP channels (`getInfoCHOPChan`), so swapping
is a node replacement. Interop is guaranteed by the shared wire protocol; the 4-way
matrix (C++/Py sender × C++/Py receiver) is an explicit Phase 4 test. Deprecation of the
`.tox` is deferred to a future ADR only if the plugin proves strictly superior over ≥ 2
TD release cycles.

## Known gotchas checklist (from official README + forums)

Sources: [CustomOperatorSamples README](https://github.com/TouchDesigner/CustomOperatorSamples/blob/main/README.md),
[Write a CPlusPlus TOP](https://docs.derivative.ca/Write_a_CPlusPlus_TOP),
[Custom Operators](https://docs.derivative.ca/Custom_Operators), Derivative forum.

- [ ] **Release CRT only** — TD ships release binaries; a debug-CRT plugin fails to load.
- [ ] **CUDA toolkit major must match TD's bundled CUDA** (11.8 → 2023.1x, 12.8 → 2025.3x).
- [ ] **`-allow-unsupported-compiler`** needed for CUDA 11.8 + current MSVC toolsets.
- [ ] **DLL locked while TD is open** — use PluginBuilder hot-reload or host the DLL in a
  generic *CPlusPlus TOP* node during development (it can reload; installed Custom OPs
  require a TD restart).
- [ ] **Dependency DLLs co-located** with the plugin DLL, else the OP silently never appears.
- [ ] **`opType` must be globally unique**, first char uppercase, `[a-z0-9]` after — or
  registration fails silently.
- [ ] **`setAPIVersion` gate**: return early if it fails (header/TD build mismatch).
- [ ] **`cudaArray*` is NULL outside `beginCUDAOperations()`** and invalid after
  `endCUDAOperations()` — never cache across frames; all CUDA work on the main cook
  thread inside the bracket.
- [ ] **Own `cudaStream_t`** (carried in `OP_CUDAAcquireInfo` / `TOP_CUDAOutputInfo`);
  never `cudaSetDevice` away from `TOP_Context::getCUDADeviceIndex()`.
- [ ] **`cookEveryFrame` vs `cookEveryFrameIfAsked`** — sender needs `cookEveryFrame`
  while Active; receiver should not burn cooks when idle.
- [ ] **`TOUCH_TEXT_CONSOLE=1`** for printf debugging (slow — strip for perf runs);
  attach the VS debugger by setting `TouchDesigner.exe` + the `.toe` as the debug command.
- [ ] **No C++ exceptions across the ABI boundary** — wrap every entry point; surface
  errors via `getErrorString` badges (see ADR-0009 mitigations).
- [ ] **Deployment**: `Documents/Derivative/Plugins` or per-project `Plugins/` folder;
  subfolders searched; no code signing required.

## Phases

### Phase 0 — Spike (S, ~3 days)

Build a SpectrumTOP-derived skeleton via standalone CMake against vendored 2025 headers;
load in TD 2025.3x; confirm the `getCUDAArray` → `beginCUDAOperations` →
`createCUDAArray` lifecycle and measure `cudaMemcpy2DFromArray` D2D at 1080p (expect
~20–40 µs). Verify channel order / Y-orientation against what `top.cudaMemory()`
produces today (feed both into a Python receiver, diff bytes).
**Exit**: pass-through TOP (input → linear staging → output) cooking at 60 fps; copy cost
logged. **Kill criterion**: array↔linear round trip alone > ~120 µs → target
unreachable; stop and record in this plan.

### Phase 1 — Protocol core + parity tests (M)

`src/core/` + `tools/protocol_dump` + C++ unit tests (doctest/Catch2, header-only) +
golden-bytes pytest. New CI job `cpp-top-protocol` on the existing Ubuntu runner: cmake
build of core+tools only (no CUDA), run C++ tests, then pytest cross-checks against
`shm_protocol.py`. **Exit**: bidirectional golden parity green in CI.

### Phase 2 — Sender (M)

`CudaLinkOutTOP`: parameters, ring alloc (`cudaMalloc` per slot at first cook /
resolution change), input array → slot via `cudaMemcpy2DFromArray` on own stream,
`cudaEventRecord` (IPC event), handles exported once per slot allocation, publish (D3
ordering), doorbell `SetEvent` (name: `Local\cudalink_db_<shm_name>`, matching
`_doorbell.py`). Resolution/format change → reallocate ring, bump `version`, re-export
handles (consumers already handle `VERSION_CHANGED`). **Exit**: existing Python
`importer.py` consumer receives frames unmodified; sender cook cost measured.

### Phase 3 — Receiver (M)

`CudaLinkInTOP`: SHM open/poll on cook, `cudaIpcOpenMemHandle` (cached per version),
`cudaStreamWaitEvent` on the slot IPC event, linear → output `createCUDAArray` via
`cudaMemcpy2DToArray`. Handles TD-starts-before-producer (reconnect loop mirroring the
importer's reconnect semantics) and producer restart (`version` bump → reopen handles).
**Exit**: Python `exporter.py` → C++ receiver renders correctly, incl. producer restart.

### Phase 4 — Parity, soak, bench (M)

4-way interop matrix (C++/Py sender × C++/Py receiver); 1-hour soak at 60 fps watching
GPU memory via `nvml_observer` + TD Performance Monitor; resolution sweep vs
`docs/BENCHMARKS.md` baselines; CUDA 11.8 leg build + smoke on TD 2023.1x.
**Exit**: sender ≤ 90 µs p50 at 1080p float32; zero leaks; no torn frames in soak.

### Phase 5 — Packaging & docs (S)

CI Windows job building both matrix legs, artifacts uploaded; deployment guide in
`cpp_top/README.md`; ADR-0009 flipped to Accepted; `docs/TOX_BUILD_GUIDE.md` gains a
"when to use which" table; CHANGELOG entry.

## Verification

- Golden-bytes parity runs on every PR (Ubuntu, no GPU, no TD).
- `verification/` script: C++ sender in TD → Python consumer printing per-frame latency;
  accept p50 ≤ 90 µs sender cook; notification latency unchanged (or PLAN-002 numbers if
  that landed).
- Soak: 1 h @ 60 fps, GPU memory delta 0 (nvml), TD alive, frame counter monotone.

## Risk register

| Risk | Sev | Mitigation |
|---|---|---|
| TD crash blast radius (ADR-0007 concern) | High | ADR-0009 mitigations: no exceptions across ABI, error badges + no-op cook on CUDA errors, release CRT, soak gate |
| CUDA major mismatch with TD build | High | Dual matrix (D4); runtime version check → error badge, refuse to cook |
| Protocol drift vs Python | High | Golden bytes both directions in CI (D3) |
| Pixel format / stride / Y-flip mismatch | Med | Phase 0 spike verifies against `cudaMemory()` output; kernel fallback reserved (D6) |
| DLL locked during dev | Low | PluginBuilder loop; generic CPlusPlus TOP host node |
| opType collision → silent registration failure | Low | Unique `Cudalinkout`/`Cudalinkin`; startup log check documented |
| Vendored TD header licensing | Low | Headers from Derivative's public samples repo; attribution in README; verify terms before commit |

## Reference material

- [Write a CPlusPlus TOP](https://docs.derivative.ca/Write_a_CPlusPlus_TOP) ·
  [TouchDesigner CUDA API Reference](https://docs.derivative.ca/TouchDesigner_CUDA_API_Reference) ·
  [Custom Operators](https://docs.derivative.ca/Custom_Operators) ·
  [TD CUDA versions](https://docs.derivative.ca/CUDA)
- [TouchDesigner/CustomOperatorSamples](https://github.com/TouchDesigner/CustomOperatorSamples)
  — SpectrumTOP (canonical modern CUDA TOP: `getCUDAArray`/`createCUDAArray`/begin-end
  bracket), CannyEdgeTOP (`.cu` kernels via `CUDA 12.8.props`, surface-object copies).
  The `CudaTOP` sample ships **inside the TD install** under `Samples/CPlusPlus`, with
  the authoritative headers.
- [IntentDev/PluginBuilder](https://github.com/IntentDev/PluginBuilder) (MIT) — hot-reload
  dev loop, `CudaTOP` template.
- [DBraun/PyTorchTOP](https://github.com/DBraun/PyTorchTOP) — `cudaMemcpy2DFromArray`
  D2D precedent (older API; migration contrast).
- [vinz9/CudaSortTOP](https://github.com/vinz9/CudaSortTOP) (MIT) — minimal
  cudaArray → linear → process → cudaArray skeleton (legacy API).
- [IntentDev/TopArray](https://github.com/IntentDev/TopArray) (MIT) — memory-layout math
  (component-first strides) for the Python side.
