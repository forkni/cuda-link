# C++ Custom TOP — CUDA / C++ Guidelines Verification Report

- **Branch verified**: `feat/cpp-custom-top` @ `9162a5f` ("feat: add BGRA wire flag, file-based debug logging, TD-in-the-loop benchmarks")
- **Scope**: everything under `cpp_top/src/` + `cpp_top/CMakeLists.txt` (~3,160 LOC, excluding the vendored TD headers)
- **Method**: full read of all 20 project source files, cross-checked against (a) the official NVIDIA CUDA Runtime API documentation and CUDA C++ Best Practices Guide, (b) the vendored `CPlusPlus_Common.h` / `TOP_CPlusPlusBase.h` contracts, and (c) the ISO C++ Core Guidelines rules called out in the project's research document.
- **Date**: 2026-07-06

## Verdict

The `cpp_top` tree is in **strong conformance** with both the NVIDIA CUDA best-practice rules and the C++ Core Guidelines rules that matter for this codebase. Every load-bearing CUDA claim embedded in the code's comments checked out against the official documentation, including two places where the research document itself was out of date (details below). No correctness bug was found. The findings that remain are one architectural limitation inherent to CUDA IPC (worth documenting for users), two consciously-documented deviations that survive scrutiny, and a handful of minor polish items.

---

## 1. Claims verified against official NVIDIA documentation

### 1.1 CUDA IPC on Windows — code is right, research doc was outdated ✅

The research document's caveat said classic CUDA IPC is **Linux-only**. That is no longer what the official docs say. The current CUDA Runtime API documentation (`group__CUDART__DEVICE`, CUDA 12.x) states:

> "IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. **IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost.** Users can test their device for IPC functionality by calling cudaDeviceGetAttribute with cudaDevAttrIpcEventSupport."

The code does **exactly** what the docs prescribe: both TOP constructors probe `cudaDeviceGetAttribute(..., cudaDevAttrIpcEventSupport, ...)` before any IPC call and latch a fatal error with a clear message if unsupported (`CudaLinkOutTOP.cpp:129-138`, `CudaLinkInTOP.cpp:96-105`). The "performance cost" caveat is a real, accepted trade-off of the whole design and is already the project's measured baseline.

### 1.2 Interprocess event flags ✅

Official docs: `cudaEventInterprocess` **must be specified along with** `cudaEventDisableTiming`. The sender creates every slot event with `cudaEventCreateWithFlags(&rawEvent, cudaEventDisableTiming | cudaEventInterprocess)` (`CudaLinkOutTOP.cpp:354`) — compliant, and `cudaEventDisableTiming` is also the Best Practices Guide's recommendation for pure-sync events.

### 1.3 IPC handle lifecycle pairing ✅

- Imported memory (`cudaIpcOpenMemHandle`) is closed with `cudaIpcCloseMemHandle`, never `cudaFree` (`CudaLinkInTOP.cpp:184`, `raii_handles.h:68-101` — the two guard types are deliberately non-interchangeable and say so).
- Imported events (`cudaIpcOpenEventHandle`) are freed with `cudaEventDestroy`, per docs (`CudaLinkInTOP.cpp:190`).
- `cudaIpcOpenMemHandle` is called with `cudaIpcMemLazyEnablePeerAccess`, matching NVIDIA's own `simpleIPC` sample (`CudaLinkInTOP.cpp:346`).
- The docs' undefined-behavior warning — using an imported handle after the exporter frees the original — is the exact rationale for the sender's shutdown-flag-first + handle-zeroing + 100 ms grace-period sequence in `teardown()` and `reallocate()` (`CudaLinkOutTOP.cpp:242-299, 464-499`). The ordering (signal → zero handles → grace → free) is correct.

### 1.4 2D memcpy signatures and units ✅

Official signatures confirmed:

- `cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)` — `width` is **in bytes**.
- `cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream)` — same.

Both call sites compute `rowBytes = width * num_comps * itemsize` and pass it as both pitch and transfer width with tightly-packed rows (`CudaLinkOutTOP.cpp:634-638`, `CudaLinkInTOP.cpp:520-524`). Correct parameter order, correct units, `cudaMemcpyDeviceToDevice`, explicit stream on every call.

### 1.5 Error-checking discipline (Best Practices Guide "systematically check every API call") ✅

Every CUDA call on the cook path and the build-up paths is checked: via `CUDALINK_CUDA_CHECK_BOOL`/`_FATAL` (with `__FILE__`/`__LINE__` capture, matching NVIDIA's `checkCudaErrors` pattern) or explicit `cudaError_t` tests with `cudaGetErrorString` in the latched error message. `cudaGetLastError()` after kernel launches is N/A — this pass launches **no kernels** (pure D2D copy path; the CMake comment correctly defers enabling the CUDA language until a kernel exists).

The **sticky vs non-sticky error** distinction from the research document is implemented better than most production code: hot-path failures latch `myFatal` and short-circuit all future cooks (with an accurate comment on why `cudaDeviceReset()` is forbidden inside TD's process), while cold-path failures (IPC open, allocation) stay retryable (`cuda_check.h:36-51`, both `execute()` preambles).

### 1.6 Hot-path audit (Best Practices "High Priority" items) ✅

Per-frame CUDA work is exactly:

| TOP | Per-cook CUDA calls | All async? | Stream |
|---|---|---|---|
| Out | `cudaMemcpy2DFromArrayAsync`, `cudaEventRecord`, `cudaMemcpy2DToArrayAsync` (pass-through) | yes | explicit non-blocking |
| In | `cudaStreamWaitEvent`, `cudaMemcpy2DToArrayAsync` | yes | explicit non-blocking |

- **No per-frame `cudaMalloc`/`cudaFree`** — allocation happens only in `reallocate()` on first cook / geometry / format / slot-count change; the ring of slots is pre-allocated. ✅
- **No `cudaDeviceSynchronize` / `cudaStreamSynchronize` anywhere on the frame path** — cross-process ordering is GPU-side via per-slot IPC events (`cudaEventRecord` → receiver `cudaStreamWaitEvent`), which is precisely the "prefer event-based synchronization" guidance. The comment block at `CudaLinkOutTOP.cpp:679-696` explaining why no CPU sync is needed (and what to fall back to if torn frames ever appear) is accurate. ✅
- **Default stream never used** — both TOPs create `cudaStreamNonBlocking` streams in their constructors, check the result (an unchecked failure would silently fall back to stream 0 — the code comments call this out), and pass the stream to every async call and to TD's `OP_CUDAAcquireInfo`/`TOP_CUDAOutputInfo`. The sender additionally requests greatest priority via `cudaStreamCreateWithPriority` with a documented, correct "hint only" caveat and a safe fallback. ✅
- **Import-once, not per-frame** — IPC handles are opened once per protocol version and cached (`openSlotHandlesIfNeeded`), re-opened only on `VERSION_CHANGED`. This satisfies the "share handles before the frame loop, never per frame" rule (IPC opens cost 100s of µs–ms). ✅
- `cudaMalloc` instead of `cudaMallocAsync`: **correct here, not a violation** — stream-ordered allocations cannot be exported with `cudaIpcGetMemHandle` (pool memory uses the separate `cudaMemPoolExportPointer` API), and the allocation is cold-path only. The research document's `cudaMallocAsync` recommendation targets per-frame allocation, which this code doesn't do.

### 1.7 TouchDesigner contract (vendored headers as primary source) ✅ with one documented deviation

- `getCUDAArray()` before `beginCUDAOperations()`, use after — matches `CPlusPlus_Common.h:967-971` verbatim. Same for `createCUDAArray()` (the fix comment at `CudaLinkOutTOP.cpp:591-599` records the live-observed `cudaErrorInvalidResourceHandle` that the wrong ordering produced).
- `beginCUDAOperations`/`endCUDAOperations` bracket is RAII-guarded (`CudaOpScope`), so **no exit path — early return, CUDA-check return, or exception — can leave the bracket unbalanced**, and `end` is only called if `begin` succeeded. This is exactly the R.1/CP.20 pattern the guidelines ask for.
- **Deviation (accepted)**: Derivative's wiki says *any* CUDA operation must sit inside the bracket; this code performs resource management (`cudaMalloc`, event creation, IPC export/open, teardown) outside it. The in-code justification (`CudaLinkOutTOP.cpp:306-318`) is well-argued from primary sources: the vendored header states the bracket exists "to ensure the order of operations between Vulkan and CUDA is properly managed" (`CPlusPlus_Common.h:677-678`), and TD's own bundled CudaTOP sample creates/destroys streams and surfaces outside the bracket. All calls remain on the main thread inside `execute()`/ctor/dtor, honoring the threading rule. Risk assessed as low; the rationale is recorded at the call site, which is the right way to carry a deviation.
- Exception safety at the ABI: every entry point that can allocate (`execute`, `setupParameters`, `CreateTOPInstance`, all Info CHOP/DAT/error-string callbacks) is wrapped in `try/catch(...)` so **no exception ever crosses the plugin ABI** — the Google-style rule the research document flags as essential for DLL boundaries. ✅

---

## 2. Findings

### F1 — Same-process IPC is impossible by design (document it) · **Medium / product limitation**

Official docs and NVIDIA forum guidance confirm `cudaIpcOpenMemHandle` **cannot open a handle exported by the same process**. Consequence: a `CudaLinkOutTOP` and `CudaLinkInTOP` placed in the **same TouchDesigner instance** can never connect — the receiver will fail in `openSlotHandlesIfNeeded()` every cook with a latched error. The failure is graceful (error badge + retry, no crash), but the symptom won't tell the user *why*. Recommend: (a) a note in the user-facing docs ("loopback within one TD process is not supported — use two TD instances or the Python peer"), and optionally (b) detecting `cudaErrorDeviceUninitialized`/`cudaErrorInvalidDevice` from a same-process open and substituting a targeted error message.

### F2 — `std::atomic_ref<uint64_t>` on a 4-byte-aligned field is formally UB · **Low / documented, accepted**

`VERSION_OFFSET = 4` gives the 8-byte `version` field 4-byte alignment; `atomic_ref<uint64_t>` requires 8 ([atomics.ref.generic]). The code (a) documents this exhaustively at `shm_layout.h:28-36`, (b) `static_assert`s `is_always_lock_free` at both use sites, and (c) correctly scopes the justification to x86-64 (cache-line-contained unaligned 8-byte access). Verified: the field spans bytes 4–11, entirely inside the first cache line. This is a wire-compatibility-forced deviation, safe on the shipped target. If the protocol ever gets a v0.6 bump, moving `version` to an 8-aligned offset would retire the UB; until then, the current guard rails are appropriate.

### F3 — `Sleep(100)` on the TD main thread in `teardown()`/`reallocate()` · **Low / cold path only**

The IPC-close grace period blocks the main thread ~6 frames at 60 fps on Active-off, name change, and resolution/format switches. It exists to avoid the documented use-after-free UB window for the receiver's imported handles, mirrors the Python exporter, and never runs on the steady-state frame path. Acceptable; a fully non-blocking alternative (deferred free-list drained on subsequent cooks) exists if hitchless live switching ever becomes a requirement — not needed for correctness.

### F4 — Teardown paths ignore CUDA return codes · **Low**

`teardown()`, `closeHandles()`, and the guard destructors call `cudaFree`/`cudaEventDestroy`/`cudaIpcCloseMemHandle` without checking results. For destructor-context cleanup this is conventional (nothing actionable can be done), but the Best Practices Guide's "check every call" rule would be fully satisfied by routing failures to `debugLog()` when Debug is on. Cosmetic hardening only.

### F5 — Dead macro: plain `CUDALINK_CUDA_CHECK` · **Trivial**

Only the `_BOOL` and `_FATAL` variants are used anywhere; the plain `void`-returning variant in `cuda_check.h:16-24` has no call sites. Either delete it or leave it as API symmetry — but if kept, a one-line "currently unused; for void-returning helpers" note would stop future readers from hunting for its users.

### F6 — `getInfoPopupString` lacks the `try/catch` its siblings have · **Trivial**

`info->setString(myStatus.c_str())` (both TOPs) is the one ABI entry point without the exception fence. `c_str()` is noexcept and `setString` is TD's own code, so real risk is negligible — but making it match the other seven callbacks costs two lines and removes the inconsistency.

### F7 — Receiver cannot verify the true size of imported IPC buffers · **Low / inherent residual risk**

`validateNumSlots`/`validateMetadata` treat all SHM content as untrusted (correct — named sections are openable by any process) and internally-consistency-check `data_size == expected_size()`. But no CUDA API exposes the actual allocation size behind an imported IPC pointer, so a malicious exporter could still declare dimensions larger than its real allocation and induce an out-of-bounds **device-side** read/write in the D2D copy. This is inherent to classic CUDA IPC (the Python peer has the same exposure); the existing validation already blocks all *accidental* corruption cases. Worth one sentence in the threat-model notes; no code change available that fixes it.

### F8 — Tooling gaps vs. the adopted standard · **Low / process**

Per the research document's enforcement stack: no `.clang-format`, no `.clang-tidy`, and no C++ CI exist yet at the repo root or under `cpp_top/` (golden-byte `topcore_tests` + `protocol_dump` are explicitly deferred in `CMakeLists.txt`). compute-sanitizer is currently low-value (no kernels, and memcheck can't attach to TD's process meaningfully for this plugin), but `clang-format` + `clang-tidy cppcoreguidelines-*,modernize-*,bugprone-*,performance-*` over `cpp_top/src/` is cheap and the code as written would pass with near-zero churn — it already follows the rules by hand.

---

## 3. C++ Core Guidelines conformance summary

| Rule | Status | Evidence |
|---|---|---|
| R.1 / R.11 RAII, no naked new/delete | ✅ | `raii_handles.h` guards for every owned resource class; the only `new`/`delete` pair is the TD factory contract (`CreateTOPInstance`/`DestroyTOPInstance`), which the SDK requires |
| C.20/C.21 Rule of Five | ✅ | All four guards: deleted copies, `noexcept` moves via `release()`; both TOP classes delete all four with a comment explaining why delete (not implement) is correct |
| F.6 `noexcept` | ✅ | All `core/` free functions, all guard moves/accessors; header comments justify each |
| `[[nodiscard]]` | ✅ | `acquire_slot`, guard `release()`/`get()`, all `SHMLayout` accessors |
| ES.47/48/49 `nullptr`, no C-style casts | ✅ | `static_cast`/`reinterpret_cast` throughout; the two `const_cast`s in `ring_reader.cpp` are load-only and documented |
| ES.45 no magic numbers | ✅ | `kIpcCloseGracePeriodMs`, `BITS_PER_BYTE`, `kNumInfoCHOPChans`, `kMaxReasonableSlots`, layout constants; remaining literals (97-frame cadence) are commented |
| Enum.3 `enum class` | ✅ | `SlotState` |
| SF.7/SF.8 header hygiene | ✅ | `#pragma once` everywhere; no `using namespace` in any header; using-declarations confined to `.cpp` files |
| E.13-ish: no exceptions across ABI | ✅ | `catch(...)` at every allocating ABI entry point (see F6 for the single trivial gap) |
| CP.20 RAII locks/brackets | ✅ | `CudaOpScope` |
| I.11/R.3 raw pointers non-owning at interfaces | ✅ | `core/` operates on caller-owned `uint8_t*`; ownership lives in guards/members only |

**Layering** also deserves a pass: `src/core/` is genuinely TD-free and CUDA-free (verified — no `cuda_runtime.h`/TD includes), which is what makes the deferred golden-byte parity tests testable standalone. `CMakeLists.txt` matches the research document's modern-CMake guidance where applicable (3.24+, `CUDAToolkit` imported targets, `MODULE` library, no `cuda_add_*`, CMP0091 + `/MD` rationale, CUDA-major pin matching TD 2025.3x → CUDA 12.8); `LANGUAGES CUDA`/`CUDA_ARCHITECTURES` are correctly absent since no `.cu` files exist yet.

## 4. Research-document corrections

Two claims in the research document should be amended so future work doesn't "fix" correct code:

1. **"CUDA IPC (classic runtime API) is Linux-only"** — outdated. Current official docs: supported on Windows (compatibility mode, performance-cost caveat, `cudaDevAttrIpcEventSupport` probe). The shipped design is documentation-compliant as-is.
2. **C++17 as the target standard** — `cpp_top` deliberately requires **C++20** (`std::atomic_ref` for the formally-correct cross-process acquire/release on `write_idx`/`version`). This is an upgrade, not a violation; the C++17 idioms the document recommends are all present anyway.

## Sources

- [CUDA Runtime API — Device Management (IPC functions)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)
- [CUDA Runtime API — Event Management](https://docs.nvidia.com/cuda/archive/9.1/cuda-runtime-api/group__CUDART__EVENT.html) (flag contract unchanged in 12.x)
- [CUDA Runtime API — Memory Management (2D memcpy signatures)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
- [CUDA Programming Guide — Interprocess Communication](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html)
- [NVIDIA forum — same-process IPC open is unsupported](https://forums.developer.nvidia.com/t/why-exporting-and-importing-cuda-ipc-handles-in-the-scope-of-the-same-linux-process-is-not-supported/252737)
- [NVIDIA cuda-samples — simpleIPC](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/simpleIPC/simpleIPC.cu)
- Vendored `cpp_top/vendor/td/2025/CPlusPlus_Common.h` / `TOP_CPlusPlusBase.h` (primary TD contract)
