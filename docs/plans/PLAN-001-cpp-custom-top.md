# PLAN-001: C++ Custom TOP sender/receiver — comprehensive build plan

**Status**: Proposed (v2 — implementation-level; supersedes the v1 outline on
`claude/cuda-link-cpp-extension-f1gu9d`, reconcile at merge)
**Date**: 2026-07-05
**Size**: L (4–6 weeks, phases 0–5)
**Depends on**: PLAN-002 lessons (shipped, `feat/r5-native-wait-backend`); ADR-0009
lands with Phase 0/1
**Related ADRs**: [ADR-0007](../adr/0007-spout-as-launcher-not-transport.md) (in-process
prior), ADR-0009 (this plan's decision record), [ADR-0006](../adr/0006-stay-pure-python-no-rust.md),
[ADR-0002](../adr/0002-byte-identical-td-mirror.md) (mirror unaffected)

> **Verification note (2026-07-05)**: every API/version/gotcha claim in this document
> was fact-checked against the actual TD SDK headers
> (`TOP_CPlusPlusBase.h`/`CPlusPlus_Common.h` from
> [CustomOperatorSamples/SpectrumTOP](https://github.com/TouchDesigner/CustomOperatorSamples)),
> docs.derivative.ca, [IntentDev/PluginBuilder](https://github.com/IntentDev/PluginBuilder),
> official CUDA 12.9 headers, MicrosoftDocs sources, CPython source, CMake docs, and the
> C++ standard draft. Claims that could not be pinned to documentation are carried as
> explicit Phase-0 spike questions rather than assumptions.

---

## 1. Goal, targets, non-goals

**Goal**: two native Custom TOPs — `CudaLinkOutTOP` (sender) and `CudaLinkInTOP`
(receiver) — that replace the Python TD-side hot path while speaking the **byte-identical
v0.5.0 SHM protocol** (`src/cuda_link/shm_protocol.py`), so every existing Python
consumer/producer works unchanged, and the Python `.tox` remains a drop-in fallback.

**Quantified targets** (baseline: `benchmarks/results/td/cell_C_td` / `cell_D_td`,
TD 2025.32820, RTX 4090, 1080p float32, total ~182–200 µs/frame):

| Metric | Baseline (Python .tox) | Target (C++ TOP) |
|---|---|---|
| Sender cook cost p50, 1080p f32 | ~182–200 µs | **≤ 90 µs** |
| — `cudaMemory()` staging alloc+copy | 80–105 µs | **0 (eliminated)** |
| — Python scaffolding (pre/post/shm/ctypes) | ~40 µs | ~5 µs (native) |
| — ring D2D copy | 28–46 µs | ~28–46 µs (unchanged, GPU-bound) |
| Receiver cook cost | not separately profiled | no regression vs `copyCUDAMemory` path |
| float16 textures | rejected (TD `cudaMemory()` limitation) | **supported** (new capability) |

**Where the win comes from — and where it doesn't (R5 lesson)**: this plan's gains are
*copy/alloc elimination and native scaffolding*, which are real and language-addressable.
It promises **nothing** about cross-process notification latency — R5 measured that wake
latency is a Windows kernel scheduling floor (~65 µs p50 publish→detect) that no
in-process language choice moves. The receiver TOP is cook-paced anyway (TD cooks at
frame rate); it polls `write_idx` per cook and never blocks.

**Non-goals**: replacing the `.tox` (stays as reference + fallback); protocol changes;
touching the Python-process consumer; D2H anywhere in the plugin (ADR-0008); Spout/NDI
transport (ADR-0007 owns that scope).

## 2. Prerequisites & version matrix

| TD build series | Bundled CUDA | Plugin toolkit | MSVC | Artifact suffix |
|---|---|---|---|---|
| 2025.3xxxx | 12.8 | CUDA 12.8 | VS2022 (official samples pin toolset v142; both build in VS2022) | `-cu128` |
| 2023.1xxxx | 11.8 | CUDA 11.8 (with the v143 toolset add `-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` — CUDA < 12.4 rejects it otherwise) | VS2022 | `-cu118` |

Windows CUDA IPC status (CUDA 12.9 runtime reference, verbatim): "IPC functionality on
Windows is supported for compatibility purposes but not recommended as it comes with
performance cost"; the programming guide steers new code to the VMM APIs. This project
already weighed that and chose legacy IPC ([ADR-0004](../adr/0004-legacy-cuda-ipc-over-vmm.md));
the plugin inherits that decision unchanged.

- Plugin CUDA **major must match** TD's bundled runtime ([docs.derivative.ca/CUDA](https://docs.derivative.ca/CUDA)).
- SDK headers (`TOP_CPlusPlusBase.h`, `CPlusPlus_Common.h`) ship in the TD install under
  `Samples/CPlusPlus` and per-sample in
  [TouchDesigner/CustomOperatorSamples](https://github.com/TouchDesigner/CustomOperatorSamples).
  Vendor them per matrix leg under `cpp_top/vendor/td/{2023,2025}/` with attribution
  (verify Derivative's sample license text before commit).
- Primary dev target: TD 2025.3x + CUDA 12.8 (the installed pair on the dev box). The
  2023/11.8 leg is Phase-4 work, not day-1.
- Dev machine already proven: MSVC (`utils/build_native_wheel.cmd` detects VS Community),
  CUDA 12.8 toolkit (spout wheel build found it), CMake/scikit-build toolchain used twice.

## 3. Architecture decisions

### D1 — Two op types, two thin DLLs, one shared static core

- `CudaLinkOutTOP.dll` — opType `Cudalinkout`, label "CUDA Link Out",
  `minInputs = maxInputs = 1`, `executeMode = TOP_ExecuteMode::CUDA`.
- `CudaLinkInTOP.dll` — opType `Cudalinkin`, label "CUDA Link In", 0 inputs, CUDA mode.
- `cudalink_topcore` static lib holds ~90% of the logic (layout codec, ring
  writer/reader, IPC lifecycle, doorbell, dtype mapping, error formatting). One custom
  operator per DLL is TD's model; the two roles differ in input arity and cook
  semantics (sender: cook every frame while Active; receiver: cook every frame while
  Active but with a cheap NO_FRAME early-out, mirroring `TDReceiverEngine.has_new_frame()`).

*Rejected: one DLL + Mode param (wrong input arity for one role); shipping via the
generic CPlusPlus TOP host node (dev-only convenience — no OP Create menu identity).*

### D2 — Build: standalone CMake is the build of record; PluginBuilder is the dev loop

`cpp_top/CMakeLists.txt`, CMake ≥ 3.24 (policy **CMP0091 NEW** required — set before
`project()` — or `CMAKE_MSVC_RUNTIME_LIBRARY` is silently ignored), C++20 (for
`std::atomic_ref`, see D4), `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL` — this forces
`/MD` (release dynamic CRT) in **every** configuration including Debug, deliberately:
debug-CRT plugin DLLs failing to load in TD is a recurring community report
([forum: "Failed to load the .dll" for Debug .dll](https://forum.derivative.ca/t/resolved-failed-to-load-the-dll-for-debug-dll/175616));
the official README only documents the weaker fact that Debug configs may fail to
*compile* against TD's shipped release libs. `CMAKE_CUDA_ARCHITECTURES` per leg, option
`CUDALINK_TOP_CUDA_MAJOR=12|11`. Five targets: `cudalink_topcore` (static),
`CudaLinkOutTOP` + `CudaLinkInTOP` (MODULE), `protocol_dump` (console tool) and
`topcore_tests` (doctest) — the last two build **without CUDA or TD headers** so they
run on the Ubuntu CI runner.

**Dev inner loop — primary: the generic CPlusPlus TOP host node.** Derivative's
documented workflow: load the DLL into a `CPlusPlus TOP` during development — it can
unload/reload the DLL without restarting TD ("Unload Plugin" parameter,
[docs.derivative.ca/CPlusPlus_TOP](https://docs.derivative.ca/CPlusPlus_TOP));
installed Custom OPs are only detected at TD startup and hold the DLL lock until exit.
[IntentDev/PluginBuilder](https://github.com/IntentDev/PluginBuilder) (MIT, CMake+Ninja
hot reload from inside TD, ships a CudaTOP template) is worth an attempt as a faster
loop, **but expect porting work**: its README pins CUDA 11.8, it has no CUDA 12.x or TD
2025 support anywhere, and its last commit is 2024-07-08 — treat it as a Phase-0
experiment with the CPlusPlus-TOP workflow as the reliable fallback, not as
infrastructure this plan depends on.

### D3 — cudart linkage: dynamic, never static, never shipped (R5's `d0370aa` lesson, applied in-process)

The plugin links the CUDA runtime **dynamically** (`cudart64_12.dll` import lib). The
Windows loader resolves imports against the loaded-module list **by base name, before
any directory search** ([dynamic-link-library-search-order](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order)),
so inside the TD process the import binds to the cudart TD already loaded — one runtime
instance, one shared primary context. **Forbidden**: `cudart_static` (a second
in-process runtime instance; CUDA docs say events are driver-level handles shared via
the primary context, but per-instance error state diverges, mixing runtime *versions*
is only supported statically linked, and R5's `d0370aa` incident showed how easily
same-named/duplicate cudart situations produce silently-wrong event answers — don't
create the situation), and shipping any `cudart64_*.dll` beside the plugin (shadow-copy
risk). Guard at init: `cudaRuntimeGetVersion()` major ≠ expected → error badge +
refuse to cook.

### D4 — Protocol core: TD-free, CUDA-free, golden-byte-pinned

`src/core/shm_layout.hpp` transcribes the constants from `shm_protocol.py` **verbatim**
(the Python file stays the single source of truth; the header cites it and golden tests
enforce equality):

```text
PROTOCOL_MAGIC 0x43495044 ("CIPD")           SLOT_SIZE 128 (64B mem + 64B event handle)
header: magic u32@0, version u64@4,          shutdown_flag u8 @ 20+N*128
        num_slots u32@12, write_idx u32@16   metadata 20B @ 21+N*128:
SHM_HEADER_SIZE 20                             width u32, height u32, num_comps u32,
slots @ 20 + slot*128                          kind u8, bits u8, flags u16, data_size u32
                                             timestamp f64 @ 41+N*128
dtype kinds: 0=Signed 1=Unsigned 2=Float; FLAGS_BFLOAT16 0x0001, FLAGS_MONO_ALPHA 0x0002
```

- `ring_writer.{h,cpp}` — sender side, with the **C3 ordering contract**: slot payload →
  metadata → timestamp → `write_idx` stored **last** via
  `std::atomic_ref<uint32_t>(*idx).store(v, std::memory_order_release)` (reader:
  `std::atomic_ref` acquire load). A bare release *fence* + plain store — the naive
  transcription of `publish_frame()`'s fence-lock — is formally UB
  ([atomics.fences] requires atomic operations on both sides), even though it happens
  to compile correctly on x86-64 MSVC today. `atomic_ref<uint32_t>` is lock-free and
  address-free ([atomics.lockfree]), which is exactly what makes it valid on
  cross-process shared memory; alignment is guaranteed (`write_idx` at offset 16).
  This is why the core is C++20.
- `ring_reader.{h,cpp}` — receiver side: `acquire(last_write_idx, last_version)` →
  `{NO_FRAME, NEW_FRAME, SHUTDOWN, VERSION_CHANGED}` mirroring `acquire_slot()`;
  `read_slot = (write_idx - 1) % num_slots`.
- Handles are opaque 64-byte blobs at this layer — no cudart include anywhere in
  `src/core/`.

**Golden-byte parity, both directions** (the load-bearing test):

1. `tests/golden/shm_v050_3slots.bin` — generated once by a small script driving
   `shm_protocol.py` with fixed synthetic values (deterministic handle patterns,
   1920×1080 float32 4ch, version=7, write_idx=42, timestamp=1234.5); checked in.
2. C++ doctest: `ring_writer` reproduces that file byte-for-byte.
3. pytest (`tests/core/test_cpp_protocol_parity.py`): runs `protocol_dump --emit`,
   parses the output with `shm_protocol.py`, asserts every field; and feeds the golden
   file to `protocol_dump --verify` (ring_reader side).
4. Publish-ordering torture test: writer thread publishes 1e6 frames, checker thread
   asserts `write_idx` is never observed advanced with stale metadata (doctest, threads).

### D5 — Sender TOP design (`src/out_top/CudaLinkOutTOP.{h,cpp}`)

Modeled on SpectrumTOP's current-API flow:

```text
FillTOPPluginInfo: setAPIVersion(TOPCPlusPlusAPIVersion); executeMode=CUDA;
                   opType "Cudalinkout"; minInputs=maxInputs=1
ctor(TOP_Context*): store context; cudaStreamCreateWithFlags(nonblocking)
execute(TOP_Output*, OP_Inputs*, ...):
  1  read params (Active off -> clear status, return)
  2  in = inputs->getInputTOP(0); map OP_PixelFormat -> (dtype, comps) [table below]
     unsupported format -> warning badge + skip frame (mirror _is_unsupported_format)
  3  (re)allocate on first cook / resolution / format / numslots change:
       cudaMalloc x N slots; cudaEventCreateWithFlags(DisableTiming|Interprocess) x N;
       cudaIpcGetMemHandle + cudaIpcGetEventHandle x N;
       CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, hi, lo,
                          L"<shm_name>")   // name VERBATIM, unprefixed -- see note
       ring_writer: header (version+1) + slot handles + metadata;
       create doorbell (CreateEventW auto-reset, "Local\\cudalink_db_<name>")
  4  acquireInfo.stream = myStream; arr = in->getCUDAArray(acquireInfo, nullptr)
  5  beginCUDAOperations()
       cudaMemcpy2DFromArray(slot_ptr, rowBytes, arr->cudaArray, 0, 0,
                             rowBytes, height, cudaMemcpyDeviceToDevice, myStream)
       cudaEventRecord(slot_event, myStream)
       cudaStreamSynchronize(myStream)          // blocking by design, see below
     endCUDAOperations()
  6  ring_writer.publish(++write_idx); SetEvent(doorbell)
  7  Info CHOP stats (cook µs, copy µs, frames, slot)
destroy: shutdown_flag=1 + publish + SetEvent; free events/buffers; CloseHandle; stream destroy
```

**SHM naming interop (verified from CPython source)**: `multiprocessing.shared_memory.
SharedMemory(name=X)` passes the name **verbatim** as the `CreateFileMapping` tagname —
no `Local\` prefix, pagefile-backed (`INVALID_HANDLE_VALUE`), `PAGE_READWRITE`. The
mapping size is not stored anywhere: Python attachers `VirtualQuerySize` a mapped view,
so they observe a **page-rounded** size. Consequences: the C++ name must match the
Python `shm_name` exactly and unprefixed, and `ring_reader` must accept mapping size ≥
`layout.total_size` (the Python side already tolerates this).

**Blocking by default, by design**: the SDK guarantees the input array info "will
remain valid until execute() returns" and requires all CUDA operations to occur between
`beginCUDAOperations()`/`endCUDAOperations()` (declared on `OP_Context`, which
`TOP_Context` inherits) — nothing documents that TD synchronizes the user-supplied
stream at `endCUDAOperations()`. An async copy outliving the array's validity window is
the same source-lifetime race that forced the Python TD sender to blocking export in
v1.10.1 (CUDA 719). So v1 synchronizes before `endCUDAOperations()` — correctness
first; the copy is ~28–46 µs at 1080p, inside the ≤90 µs budget. Data point for
Phase 0: the shipped SpectrumTOP sample issues async stream work with **no** explicit
sync before `endCUDAOperations()`, implying TD orders it internally — but that is
inference from sample code, not documentation, so Phase 0 tests it empirically.
Double-buffered staging (only the staging copy synchronous) is a later optimization,
out of v1 scope.

**READY_LATE lesson applied**: neither side ever treats `write_idx` advancement as GPU
completion. Completion is only `cudaEventRecord` / `cudaStreamWaitEvent` /
`cudaEventQuery` on the slot event.

**Pixel-format table (v1 scope)** — exact `OP_PixelFormat` members verified from
`CPlusPlus_Common.h`:

| OP_PixelFormat | protocol dtype / comps | Notes |
|---|---|---|
| RGBA32Float=2 / RG32Float=6 / Mono32Float=5 / A32Float | float32 / 4·2·1·1 | |
| RGBA16Float=202 / RG16Float=201 / Mono16Float=200 / A16Float | float16 / 4·2·1·1 | **new capability** — Python `cudaMemory()` rejects all float16 variants |
| RGBA16Fixed=102 / RG16Fixed=101 / Mono16Fixed=100 / A16Fixed | uint16 / 4·2·1·1 | |
| RGBA8Fixed=1 / RG8Fixed=4 / Mono8Fixed=3 / A8Fixed=300 | uint8 / 4·2·1·1 | |
| BGRA8Fixed=0 | uint8 / 4 | channel order not carried by the protocol — documented, matches existing behavior |
| MonoA{8,16}Fixed=400… / MonoA{16,32}Float | dtype / 2 with `FLAGS_MONO_ALPHA` | mirrors the existing `monoalpha*` map in TDSender.py |
| RGB10A2Fixed=700 / RGB11Float | **rejected** | packed, no protocol representation — warning badge + skip (matches Python-path behavior) |

Y-orientation: Phase 0 byte-diffs against today's `cudaMemory()` output via a Python
consumer; if flipped, the fallback is a flip-during-copy surface-object kernel
(`array_copy.cu`, CannyEdgeTOP `GpuUtils.cu` pattern) — compiled only if needed.

### D6 — Receiver TOP design (`src/in_top/CudaLinkInTOP.{h,cpp}`)

```text
execute():
  1  Active? open SHM if absent (OpenFileMappingW; missing -> "Waiting for producer"
     info status, return — reconnect is simply "try again next cook")
  2  ring_reader.acquire(last_write_idx, last_version)
       NO_FRAME        -> return (previous output persists)
       SHUTDOWN        -> close handles, status badge, return
       VERSION_CHANGED -> cudaIpcCloseMemHandle all, reopen handles + metadata
                          (mirrors _refresh_on_version_change), fall through
       NEW_FRAME       -> read_slot = (write_idx-1) % N
  3  open-once-per-version: cudaIpcOpenMemHandle / cudaIpcOpenEventHandle (cached)
  4  outputInfo.stream = myStream; textureDesc from metadata (width/height/pixelFormat
     via dtype table); out = output->createCUDAArray(outputInfo, nullptr)
  5  beginCUDAOperations()
       cudaStreamWaitEvent(myStream, slot_event, 0)   // GPU-side wait, no CPU block
       cudaMemcpy2DToArray(out->cudaArray, 0, 0, slot_ptr, rowBytes, rowBytes, height,
                           cudaMemcpyDeviceToDevice, myStream)
     endCUDAOperations()
  6  last_write_idx = write_idx; stats
```

No CPU wait anywhere: `cudaStreamWaitEvent` is the R1-style GPU-side ordering
(cross-process use of imported events is explicitly documented: "cudaEventRecord,
cudaEventSynchronize, cudaStreamWaitEvent and cudaEventQuery may be used in either
process"), and TD's bracket serializes the stream against the Vulkan consumer of the
output texture. Resolution/format changes arrive via metadata and drive `textureDesc`
per cook — the Script-TOP `pending_resolution`/`pending_format` dance in
`script_top_callbacks.py` disappears entirely.

**Teardown ordering (CUDA-documented UB to avoid)**: using an imported IPC event after
the exporter destroys the original is undefined behavior, and `cudaEventQuery` on a
never-recorded event returns success (so it's only meaningful after a record). The
protocol already sequences this — producer sets `shutdown_flag=1` + publishes *before*
freeing events/buffers; the receiver treats SHUTDOWN as close-handles-first, and the
sender destructor follows the same order (step "destroy" in D5).

### D7 — Parameters, status, stats (parity with the COMP)

Custom parameters via `OP_ParameterManager`, page "CUDA Link", **same names** as the
.tox COMP so swaps are node replacements: `Ipcmemname` (string), `Active` (toggle),
`Numslots` (int 2–5, sender only), `Cudadevice` (int — guarded by
`TOP_Context::getCUDADeviceIndex()`; mismatch = error badge, never `cudaSetDevice`
away), `Debug` (toggle), `Reconnect` (pulse, receiver → `pulsePressed`).
Status text → `getErrorString` / `getWarningString` / `getInfoPopupString` (replaces
the `Status` par + `warning_emitter` Script TOP). Stats → `getInfoCHOPChan` (`frames`,
`cook_us`, `copy_us`, `write_idx`/`read_slot`, `ipc_version`) and `getInfoDATEntries`
(config table) — replaces `get_stats()`.

**Error policy (ADR-0009 mitigations, enforced in review)**: every entry point wrapped
`try/catch(...)` → error badge, no exception ever crosses the ABI; every CUDA call goes
through a `CUDA_CHECK` macro that latches an error string and turns the cook into a
no-op; IPC failures never fatal (receiver retries next cook); release CRT; own stream;
no `exit()`/`abort()`.

## 4. Repo layout

```text
cpp_top/
  CMakeLists.txt            # targets: cudalink_topcore, CudaLinkOutTOP, CudaLinkInTOP,
  CMakePresets.json         #          protocol_dump, topcore_tests; legs cu118/cu128
  README.md                 # build, deploy, dev loop (PluginBuilder), gotchas checklist
  vendor/td/2025/ …2023/    # TOP_CPlusPlusBase.h, CPlusPlus_Common.h (+ license note)
  src/core/                 # shm_layout.hpp ring_writer ring_reader ipc_ring
                            # doorbell_win cuda_check dtype_map
  src/out_top/              # CudaLinkOutTOP.{h,cpp} Parameters.{h,cpp} [array_copy.cu]
  src/in_top/               # CudaLinkInTOP.{h,cpp}  Parameters.{h,cpp}
  tools/protocol_dump/      # main.cpp (--emit / --verify golden modes)
  tests/                    # doctest: test_layout, test_ring_roundtrip,
                            #          test_publish_ordering, test_dtype_map
utils/build_cpp_top.cmd     # MSVC+CUDA detection clone of build_native_wheel.cmd
verification/verify_cpp_top.py           # Python consumer/producer driving the TOPs
tests/core/test_cpp_protocol_parity.py   # pytest side of the golden-byte contract
```

## 5. Phases

### Phase 0 — Spike (S, ~3 days) — go/no-go gate

SpectrumTOP-derived skeleton, standalone CMake, vendored 2025 headers, loaded in
TD 2025.3x. Answer with measurements:

1. `getCUDAArray` → `cudaMemcpy2DFromArray` → linear cost at 1080p (expect 20–40 µs).
2. Does `endCUDAOperations()` synchronize the user stream (is the explicit sync free)?
3. Y-orientation + BGRA/RGBA channel order vs today's `cudaMemory()` bytes.
4. float16 input arrives correctly as a cudaArray (validates the new-capability claim).
5. PluginBuilder adapted to CUDA 12.8 hot-reload, or fall back to restart cycle.

**Exit**: pass-through TOP cooks at 60 fps; numbers recorded in this doc.
**Kill**: array↔linear round trip alone > ~120 µs → target unreachable; stop, record,
keep the Python path.

### Phase 1 — Protocol core + parity in CI (M, ~1 wk)

`src/core/` + `protocol_dump` + doctest suite + golden files + pytest cross-check. New
CI job `cpp-top-protocol` (Ubuntu: cmake core+tools only, no CUDA, run doctests + the
parity pytest). ADR-0009 committed (Proposed).
**Exit**: bidirectional golden parity green in CI on every PR.

### Phase 2 — Sender (M, ~1 wk)

`CudaLinkOutTOP` per D5 on the real protocol core. **Exit**: existing **unmodified**
Python `importer.py` consumer (torch arm) receives correct frames at 60 fps;
`verification/verify_cpp_top.py --dir out` byte-diffs frames vs a reference pattern;
mid-run resolution/format switch → consumer sees VERSION_CHANGED and recovers; cook
cost measured via Info CHOP + TD Performance Monitor.

### Phase 3 — Receiver (M, ~1 wk)

`CudaLinkInTOP` per D6. **Exit**: unmodified Python `exporter.py` (async+graphs
default) drives the TOP; producer restart mid-run recovers via VERSION_CHANGED; TD
starting before the producer shows a clean waiting state and connects when SHM appears.

### Phase 4 — Interop matrix, soak, bench, second leg (M, ~1 wk)

- 4-way matrix: {C++ TOP, Python .tox} sender × {C++ TOP, Python .tox, Python process}
  receiver — every cell exchanges verified frames.
- 1-hour soak at 60 fps: GPU memory delta 0 (`nvml_observer`), no cook-time drift, TD
  alive, `write_idx` monotone.
- Bench vs the §1 baseline table — sender p50 ≤ 90 µs at 1080p f32, or a documented
  miss (PLAN-002 set the precedent: honest numbers either way).
- CUDA 11.8 leg: build + smoke on TD 2023.1x.

### Phase 5 — Packaging, docs, ADR flip (S)

`utils/build_cpp_top.cmd`; CI Windows job building both legs, artifacts uploaded;
deploy guide (`Documents/Derivative/Plugins` or per-project `Plugins/`, dependency
co-location, no code signing); TOX_BUILD_GUIDE "when to use which" table;
BENCHMARKS.md section; CHANGELOG (next R-number); ADR-0009 → Accepted.

## 6. Gotchas checklist (carried into cpp_top/README.md)

Release CRT only (`/MD`; debug-CRT load failures are community-reported, README
documents Debug compile breakage) · CUDA toolkit major = TD's bundled major ·
`-allow-unsupported-compiler` for v143 + CUDA < 12.4 · DLL locked while TD open →
CPlusPlus-TOP host node for dev (documented reload path), PluginBuilder as experiment ·
dependency DLLs co-located, never cudart · `opType` unique among loaded TOP plugins,
first char `[A-Z]` then `[a-z0-9]`, no spaces — else silent registration failure ·
`setAPIVersion` early-return guard (`TOPCPlusPlusAPIVersion = 12 |
(OP_CommonAPIVersion << 16)`) · `cudaArray*` null until `beginCUDAOperations()`
(declared on `OP_Context`); array-info valid only until `execute()` returns; all CUDA
ops inside the bracket; never cache across frames · own `cudaStream_t`, never
`cudaSetDevice` off `getCUDADeviceIndex(reserved)` · `cookEveryFrame` deliberate per
role · `TOUCH_TEXT_CONSOLE=1` for printf (strip for perf runs) · VS debugger:
TouchDesigner.exe as debug command + `.toe` argument · no heavy work in
`getInfoCHOPChan`/`getErrorString` (queried out-of-band) · no signing requirement
documented for plugin DLLs.

## 7. Risk register

| Risk | Sev | Mitigation |
|---|---|---|
| TD crash blast radius (ADR-0007 prior) | High | ADR-0009: no exceptions across ABI, CUDA_CHECK → badge + no-op cook, release CRT, 1 h soak gate |
| Protocol drift vs Python | High | D4 golden bytes both directions on every PR |
| Source-lifetime race (async copy vs cook-scoped array) | High | Blocking copy in v1 (D5); Phase 0 measures the real sync cost |
| CUDA major mismatch / second runtime instance | High | D3: dynamic link only, runtime version guard, never ship/static-link cudart |
| Y-flip / BGRA order mismatch | Med | Phase-0 byte-diff vs `cudaMemory()`; `array_copy.cu` flip kernel reserved |
| `endCUDAOperations` stream-sync semantics undocumented | Med | Phase 0 tests empirically (SpectrumTOP's no-sync pattern is the hint); explicit sync is the safe default |
| PluginBuilder unusable on CUDA 12.8/TD 2025 (unmaintained since 2024-07) | Low | Dev-loop experiment only; documented fallback is the CPlusPlus-TOP reload workflow |
| Vendored header licensing | Low | From Derivative's public samples; attribute; verify terms |
| Windows CI runner cost/flakiness | Low | Protocol tests stay on Ubuntu; Windows job is build-only |

## 8. Verification summary

- **Every PR (CI, no GPU)**: doctest core suite + golden-byte parity pytest; Windows
  build job (Phase 5+); existing ruff/pyrefly gates on the Python side.
- **Per phase on the dev box**: `verification/verify_cpp_top.py` (frame-byte diffs,
  restart/reconnect scenarios); TD Performance Monitor + Info CHOP cook numbers; nsys
  capture attributing kernels/copies to the plugin's own stream.
- **Release gate**: Phase-4 matrix + soak + bench table; both CUDA legs load in their
  TD versions.

## 9. Relationship to existing components

- `.tox` / `CUDAIPCExtension`: unchanged, not deprecated; docs steer new users to the
  plugin, existing patches keep working. Deprecation decision deferred ≥ 2 TD release
  cycles (future ADR).
- `td_exporter/` mirror (ADR-0002): untouched — the plugin does not import Python.
- `cuda-link-native` (R5): unrelated runtime paths; the installer may later gain
  `--cpp-top` plumbing to copy DLLs into a Plugins folder (Phase-5 decision, mirroring
  the `--native`/`--spout` polarity discussion).
- PLAN-003 (cuda.bindings): orthogonal. PLAN-004/ADR-0008: the plugin never does D2H.
