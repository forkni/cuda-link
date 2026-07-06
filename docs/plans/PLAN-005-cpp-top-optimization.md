# PLAN-005: C++ Custom TOP optimization backlog (source-study driven)

**Status**: Proposed (research complete; nothing implemented)
**Date**: 2026-07-06
**Size**: S–M (items 1–3 are days; the Tier 2 kernel spike is 1–2 wk)
**Depends on**: [PLAN-001](PLAN-001-cpp-custom-top.md) (the C++ Custom TOP this plan optimizes)
**Scope**: `cpp_top/` (CudaLinkOutTOP, CudaLinkInTOP, ring_reader/writer, shm_layout) and `native/` (native_waiter)

**Sources studied:**

1. NVIDIA CUDA C++ Best Practices Guide — https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (plus CUDA C++ Programming Guide where the BPG defers to it)
2. Lei Mao's blog — https://leimao.github.io/article/ (~20 relevant articles)
3. PTX ISA in markdown — https://github.com/technillogue/ptx-isa-markdown (PTX ISA 9.1 + CUDA 13.1 runtime/driver API docs)
4. Aussie AI CUDA optimization techniques list — https://www.aussieai.com/blog/list-cuda-optimization-techniques

---

## TL;DR

The current C++ TOP design is structurally sound — all four sources independently validate the
zero-sync, non-blocking-stream, event-ordered architecture, and code inspection confirms none of
the classic hazards (default-stream calls, hidden syncs, blocking streams) are present.

The one big, proven, unimplemented optimization for the **current** code is **porting the CUDA
Graphs collapse to the C++ TOPs** — the Python side already measured −4 µs/frame from collapsing
3 WDDM submissions into 1, and the C++ Out TOP makes the identical 3-submission sequence per cook.
Beyond that: two cheap safety/perf wins (2 MiB-aligned IPC allocations, resize-hitch awareness) and
a complete, well-evidenced blueprint for a future fused copy/format-conversion kernel.

---

## 1. What the sources confirm the code already does right

| Design decision | Confirmation |
|---|---|
| No CPU sync in the hot path (`CudaLinkOutTOP.cpp:679` comment) | BPG §9.1: "CPU-to-GPU synchronization points imply a stall in the GPU's processing pipeline and should be used sparingly." Verified: zero `cudaStreamSynchronize` / `cudaDeviceSynchronize` in `cpp_top/`. |
| Non-blocking streams, creation checked (`CudaLinkOutTOP.cpp:155`, `CudaLinkInTOP.cpp:109`) | Lei Mao, *CUDA Default Stream*: any blocking-stream or legacy-default-stream activity serializes device-wide — critical inside TouchDesigner's process, which issues its own CUDA/Vulkan work. No default-stream calls found in the hot path. |
| Event-based cross-process ordering (record → IPC → `cudaStreamWaitEvent`) | PTX ISA §8.9.4 (memory consistency model): event record tasks formally *synchronize with* matching wait tasks and participate in causality order — no GPU-side fences needed. |
| CPU-side `std::atomic_ref` acquire/release over SHM `write_idx`/`version` | Correct host-side mirror of the same release/acquire discipline; matches the PTX-documented pattern semantics. |
| Stream priority treated as a hint (`CudaLinkOutTOP.cpp:143` comment) | Programming Guide: priorities cannot preempt in-flight work — they only matter at scheduling decision points. |
| `cudaEventDisableTiming \| cudaEventInterprocess` | Required combination per the Programming Guide; already used. |

**Structural fact to internalize (BPG §11.6):** producer and consumer are separate processes ⇒
separate CUDA contexts, which the GPU **time-slices**. Context switching — not API overhead — is
the floor on cross-process latency. Ring depth ≥ 2 is the correct mitigation and already exists.
MPS (NVIDIA's escape hatch) is Linux-only.

---

## 2. Tier 1 — actionable on the current C++ code (ranked)

### 2.1 Port CUDA Graphs to the C++ TOPs  ★ top item

*Sources: all four converge; NVIDIA PG "CUDA Graphs" (per-launch driver setup is paid per call);
Lei Mao PyTorch-CUDA-Graph-Capture (full capture beats partial); project's own Python benchmark.*

- **Why:** Out TOP submits 3 times per cook — `cudaMemcpy2DFromArrayAsync`
  (`CudaLinkOutTOP.cpp:635`), `cudaEventRecord` (`:642`), pass-through copy (`:668`).
  In TOP submits twice — `cudaStreamWaitEvent` (`CudaLinkInTOP.cpp:517`), `cudaMemcpy2DToArrayAsync`
  (`:521`). Each submission is a WDDM kernel-mode transition. The Python exporter's identical
  triplet collapsed 3→1 for −4.0 µs p50 (22 %) at 1080p.
- **How:**
  - Capture once per session; per cook update only changed pointers.
  - The interop `arr->cudaArray` can differ every cook and the ring slot rotates ⇒ use
    `cudaGraphExecMemcpyNodeSetParams` (the full-3D-params variant — the Python TD sender's
    `cudaGraphExecMemcpyNodeSetParams1D` does **not** cover array copies), or keep N per-slot
    `cudaGraphExec_t` instances.
  - Capture *everything* per cook (event record included; display copy too if it stays
    same-stream) — Lei Mao's full-vs-partial finding: every op left outside the graph costs a
    separate submission.
  - Follow the existing `CUDALINK_TD_USE_GRAPHS` pattern: env-gated
    (e.g. `CUDALINK_CPP_USE_GRAPHS`, default off initially), automatic fallback to the stream path
    on any capture/launch failure.
- **Verify live:** `cudaEventRecord` of an *interprocess* event inside graph capture on TD's
  bundled CUDA runtime and current drivers.
- **Expected gain:** a few µs per cook per TOP — meaningful against ~29–45 µs cook baselines.

### 2.2 Round IPC slot allocations up to a 2 MiB multiple

*Source: CUDA C++ Programming Guide §6.2.11 (IPC).*

- `cudaIpcGetMemHandle` shares the **entire underlying allocation block**, and `cudaMalloc` may
  sub-allocate from a larger block. NVIDIA's explicit recommendation: only IPC-share allocations
  whose size is rounded to 2 MiB.
- `bufferSize` at `CudaLinkOutTOP.cpp:342` is exact (e.g. 1080p RGBA uint8 = 8,294,400 B — not a
  2 MiB multiple). A one-line round-up in `reallocate()` closes a documented
  information-disclosure hazard between processes and incidentally guarantees the alignment the
  future vectorized kernel wants.
- Check the Python exporter for the same issue.

### 2.3 Treat `reallocate()`/`teardown()` stalls as a documented design constraint

*Sources: BPG §10.3; Aussie AI (cudaMallocAsync reminder).*

- `cudaMalloc`/`cudaFree` perform implicit **device-wide synchronization** — on a resolution
  change, TD's whole render pipeline hiccups beyond the intentional `Sleep(100)` grace period.
- The modern fix — stream-ordered pools (`cudaMemPoolCreate` + `cudaMemHandleTypeWin32` shareable
  handles + `cudaMemPoolExportPointer`) — is **incompatible with the 64-byte `cudaIpcMemHandle_t`
  wire protocol**: it would be a protocol revision, not a tweak.
- **Recommendation:** document as a known constraint; revisit only if users report resize hitches.

### 2.4 Pass-through display copy: measure-first micro-item

*Sources: Lei Mao CUDA-Stream / Kernel-Execution-Overlap; BPG §10.1.2.*

- Same-stream ops never overlap and the two copies read independent data — a second non-blocking
  stream (event-linked to the IPC copy) could overlap them.
- **But:** both copies are same-direction D2D and may serialize at the copy engine anyway, and the
  current event-record-before-passthrough ordering already protects receiver latency.
- The *real* fix is the Tier 2 fused kernel writing both destinations in one pass. Low priority;
  verify with Nsight under real TD render load before keeping any change.

### 2.5 Hygiene items

- **Document `CUDA_LAUNCH_BLOCKING=1`** in the troubleshooting guide as the field-debugging switch
  for localizing async CUDA errors — and warn it makes every launch synchronous **process-wide**
  (TD cook times explode while set; never ship enabled).
- **Debug-build GPU timing:** the `copy_us` Info CHOP channel measures CPU enqueue cost only (its
  comment admits this). For true GPU-side numbers, add a Debug-gated timing-enabled event pair
  (`cudaEventElapsedTime`, ~0.5 µs resolution, BPG §9.1.2). Production IPC events correctly stay
  `cudaEventDisableTiming`.
- **One submitting thread per stream** (Lei Mao multi-thread-vs-multi-stream benchmarks): currently
  true — keep it true when adding features; the native waiter thread must never issue work on a
  cook thread's stream.

---

## 3. Tier 2 — blueprint for the future fused copy/format-conversion kernel

Replaces `cudaMemcpy2D*` and folds in the BGRA swizzle / uint8↔float16 conversion currently
deferred to consumers (the `FLAGS_BGRA` wire-flag path). This is where the sources add the most
new knowledge.

### 3.1 Memory access pattern

- **16-byte vectorized accesses are the measured sweet spot** — one `uint4` = 4 BGRA pixels per
  thread. Evidence: Lei Mao's custom-memcpy benchmarks (RTX 3090, CUDA 12.0: 8/16-byte vector
  access improves effective bandwidth in almost all cases) + BPG §10.2.3.4 ("best performance with
  elements of size 8 or 16 bytes"; 16 B per-thread copies can bypass L1).
- **256-bit accesses (`.v8.b32`) require PTX 8.8 + sm_100+** — out of reach for sm_86/89; only a
  compile-time-dispatch curiosity on sm_120. Don't bother.
- **Alignment:** round row pitch to 128 B so every row of every slot supports `uint4` access at
  full transaction efficiency (misalignment costs ~10–20 % per BPG §10.2.1.3). The 2 MiB
  allocation rounding (§2.2) handles the base address.
- **Coalescing:** threadIdx.x walks contiguous pixels within a row; never stride by column
  (stride-2 halves efficiency; large strides degrade to 1/32 bandwidth — BPG §10.2.1.4). If a
  flip/transpose feature is ever added, protect **write** coalescing first (Lei Mao transpose
  benchmark) and only then introduce shared-memory staging.
- **No shared memory for the straight swizzle** — measured no benefit for pure copies (Lei Mao),
  no data reuse exists.
- **Streaming cache hints (optional):** `ld.global.nc` (`__ldg`) for the source, `.cs`
  (evict-first) stores — the documented anti-pollution recipe for one-shot frame data. Hints only,
  no correctness impact.

### 3.2 The cudaArray side

- The Vulkan-interop `cudaArray` is **texture/surface state space, not linear memory** — PTX
  `ld`/`st` cannot address it. A kernel must read it via a **texture object** (bonus: free
  hardware uint8→normalized-float conversion, 2D-locality-optimized cache — BPG §10.2.5) and/or
  write via surface objects. This decision dominates any instruction-level tuning.
- PTX proxies (§8.6): texture/surface accesses are a different proxy from generic ld/st — if a
  kernel ever mixes surface writes with generic reads of the same memory, a `fence.proxy` is
  required (cross-process ordering stays covered by the event API rules).

### 3.3 Conversion instructions

- **BGRA→RGBA swizzle:** one `__byte_perm()` (`prmt.b32`) per 4 bytes, in-register.
- **uint8↔fp16:** operate on packed `half2` (`__hadd2`-family intrinsics — 2 ops/instruction,
  BPG §12.1); unpack `uint4`→bytes with integer ops — `char`-typed arithmetic silently inserts
  int-conversion instructions.
- Keep pitch math in powers of two where possible (integer div/mod costs up to ~20 instructions).
- `-use_fast_math` / `-ftz=true` are safe here (no transcendentals; uint8-sourced data cannot hit
  denormal-sensitive ranges). Gate register creep in CI with `--ptxas-options=-v`.

### 3.4 Launch configuration

- 128–256 threads/block (multiple of 32); 2D blocks (e.g. 32×8) for texture 2D locality;
  grid-stride loops over rows; several independent 16 B loads in flight per thread
  (memory-level parallelism — BPG §12.2); warp-uniform edge handling (pad widths to 128 B or split
  the tail into its own branch/launch — BPG §13).
- Size grids via `cudaOccupancyMaxActiveBlocksPerMultiprocessor` at session setup (frame sizes
  vary per session); `__launch_bounds__` for forward compatibility. Occupancy is a means, not a
  goal, for a bandwidth-bound kernel.

### 3.5 Packaging

- **Ship fatbin SASS for sm_86, sm_89, sm_120 + PTX fallback** (BPG §17.3/§18.4): a missing cubin
  triggers driver JIT **at plugin load** — a visible TouchDesigner startup stall.
- Statically link cudart (default) — no system-runtime dependency.

### 3.6 Benchmarking methodology (applies to any A/B in this project)

- **Benchmark cold, not hot** (Lei Mao Hot-vs-Cold): production sees a fresh frame every cook, so
  kernel-vs-memcpy2D comparisons must flush L2 between iterations (write an L2-sized scratch
  buffer, ~72 MB on AD102) plus uncounted warm-ups — hot-cache numbers overstate vectorization
  wins.
- **Metric:** achieved GB/s = (bytes read + bytes written)/time vs device peak (BPG §9.2), plus
  CPU submit cost. Arithmetic intensity ≈ 0 ⇒ strictly memory-bound ⇒ after launch overhead is
  fixed, the only GPU-time lever is **moving fewer bytes** — an fp16 ring format is itself a 2×
  GPU-time optimization, convertible for free inside this kernel.

### 3.7 GPU-side signaling (only if kernels ever replace event-based sync)

- `ld.acquire.sys` / `st.release.sys` (CUDA C++: `cuda::atomic_ref<T, cuda::thread_scope_system>`)
  — available since sm_70, fine on all target GPUs.
- **Footguns from the PTX spec:** the flag must be a scalar ≤ 64-bit access — vector/128-bit
  accesses are *not* single atomics (§8.2.4); use `atom`, never `red`, for flag updates that must
  acquire (§8.11.1); prefer `.relaxed.sys`/`.acquire.sys` over `volatile` (§8.4.2, explicitly
  "may deliver better performance"); avoid RMW atomics on host-mapped memory across WDDM/PCIe
  (§8.1.1 atomicity caveat) — plain aligned strong loads/stores are fine.
- Multi-stream future: the **rendezvous-stream pattern** (Lei Mao) — one dedicated stream waits on
  all m producer events and records a single barrier event — keeps exactly one
  `cudaIpcEventHandle` in SHM regardless of internal stream count (m+n ops instead of m×n).

---

## 4. Tier 3 — explicitly rejected techniques (with reasons)

| Technique | Why rejected |
|---|---|
| Persistent kernel (resident copy kernel spinning on a flag, replacing per-frame launches) | WDDM TDR watchdog risk on a display GPU; permanently occupies SMs TouchDesigner's renderer needs; cross-process ordering gets much harder than `cudaStreamWaitEvent`. The one novel Aussie AI idea — rejected after assessment. |
| Zero-copy / mapped host memory for frame data | Every access rides PCIe; catastrophic at 60 fps on a discrete GPU (BPG §10.1.3: only for exactly-once coalesced access). |
| Shared-memory tiling for the swizzle kernel | Elementwise op, zero data reuse; measured no benefit for pure copies. Only needed if transpose/rotation is added. |
| Pinned host memory in the C++ hot path | No host leg exists; hot path is pure D2D. (Python D2H already uses it correctly.) |
| L2 persistent cache (`accessPolicyWindow` persisting) for frames | Write-once/read-once streaming data; set-aside L2 would steal cache from TD's renderer. Inverse idea — marking the ring range `cudaAccessPropertyStreaming` — is a legal, cheap, measure-first experiment. |
| TMA / `cp.async.bulk` | sm_90+ (not on sm_86/89), and **no global→global variant exists** — even on Blackwell a D2D frame copy would bounce via shared memory. |
| `mbarrier` for cross-process signaling | Must reside in CTA shared memory; `.cta`/`.cluster` scope only — structurally impossible for IPC. |
| `multimem.*`, WGMMA/tensor cores, cluster features | Data-center / multi-GPU / matrix-math features; no overlap with a texture-transfer plugin. |
| NUMA tuning, MPS | Linux-only (project is Windows-only). |
| Managed/unified memory | Not IPC-compatible; poor fit for WDDM. |

---

## 5. Source quality assessment

- **NVIDIA CUDA C++ Best Practices Guide** — the substantive backbone. Notably does **not** cover
  WDDM submission behavior, CUDA Graphs, stream priorities, or IPC (those live in the Programming
  Guide, cited above where used).
- **Lei Mao's blog** — the highest-value third-party source: measured benchmarks for vectorized
  copies, default-stream semantics, graph capture, and the hot/cold-cache methodology warning. No
  coverage of IPC, WDDM, acquire/release C++ atomics, or stream priorities.
- **ptx-isa-markdown** — genuinely useful greppable mirror of PTX ISA 9.1 + CUDA 13.1
  runtime/driver API docs. Caveats: broken INDEX links (grep-only navigation), a few missing
  memory-model parent sections (incl. the "morally strong" definition), version skew vs CUDA 12.x
  (each feature carries version gates, so checkable). Mostly relevant to the future kernel;
  formally validates the current event-based design.
- **Aussie AI list** — derivative recompilation of NVIDIA guidance, LLM-inference-oriented, with
  some technically wrong claims (e.g. `cudaPeekAtLastError` "avoids implicit synchronization" —
  neither it nor `cudaGetLastError` synchronizes; the real difference is Peek doesn't clear the
  sticky error). Net contribution: the persistent-kernel idea (rejected), the
  `cudaMallocAsync`-vs-resize-hitch reminder (§2.3), and the `CUDA_LAUNCH_BLOCKING` environment
  check. Weight NVIDIA guidance over this source wherever they overlap.
- **Access caveat:** aussieai.com and leimao.github.io block automated fetchers; those two analyses
  were reconstructed from search-index content of the exact pages rather than live fetches. The
  NVIDIA and PTX material — which carries the load-bearing recommendations — was read in full.

---

## 6. Suggested execution order

1. **2 MiB allocation rounding** (§2.2) — one line, closes a documented hazard. Do first.
2. **CUDA Graphs in the C++ TOPs** (§2.1) — env-gated, fallback-safe, biggest measured win
   available today. Benchmark with the cold-cache methodology (§3.6) and the existing 97-frame
   Debug bench log.
3. **Docs:** `CUDA_LAUNCH_BLOCKING` troubleshooting entry; resize-stall design note (§2.3, §2.5).
4. **Fused kernel spike** (§3) — behind a feature flag, judged in achieved GB/s cold, starting
   with the BGRA-uint8 1080p case where the swizzle currently burdens consumers.
5. **Measure-first extras:** pass-through copy on a second stream (§2.4);
   `cudaAccessPropertyStreaming` on the ring range (Tier 3 note).
