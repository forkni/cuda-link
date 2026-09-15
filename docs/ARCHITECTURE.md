# CUDA IPC Architecture

Technical specification of the SharedMemory protocol, ring buffer design, and GPU synchronization strategy.

---

## System Overview

The library supports **bidirectional** zero-copy GPU transfer: TD → Python (input capture) and Python → TD (AI output display).

### Direction A: TouchDesigner → Python (TD is Producer)

```
┌─────────────────────────────────────────┐
│   TouchDesigner Process (Producer)     │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ CUDAIPCExtension (Sender mode)    │ │
│  │                                   │ │
│  │  export_frame(top_op) every frame│ │
│  │    ↓                              │ │
│  │  top_op.cudaMemory() → GPU ptr   │ │
│  │    ↓                              │ │
│  │  cudaMemcpy D2D to ring buffer   │ │
│  │    ↓                              │ │
│  │  cudaEventRecord (GPU signal)    │ │
│  │    ↓                              │ │
│  │  Update write_idx in SharedMemory│ │
│  └───────────────────────────────────┘ │
│                 ↓                       │
└─────────────────┼───────────────────────┘
                  │
                  │ SharedMemory (v0.5.0 protocol)
                  │ [magic + version + handles + metadata]
                  │
┌─────────────────┼───────────────────────┐
│                 ↓                       │
│   Python Process (Consumer)            │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ Importer                          │ │
│  │                                   │ │
│  │  get_frame() or get_frame_numpy()│ │
│  │    ↓                              │ │
│  │  Read write_idx from SharedMemory│ │
│  │    ↓                              │ │
│  │  read_slot = (write_idx-1) % N   │ │
│  │    ↓                              │ │
│  │  Wait on IPC event (GPU-side)    │ │
│  │    ↓                              │ │
│  │  Return tensor/array (zero-copy) │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Direction B: Python → TouchDesigner (Python is Producer)

```
┌─────────────────────────────────────────┐
│   Python Process (Producer)            │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ Exporter                          │ │
│  │                                   │ │
│  │  export(GpuFrame(ptr, size))     │ │
│  │    ↓                              │ │
│  │  cudaMemcpy D2D to ring buffer   │ │
│  │    ↓                              │ │
│  │  cudaEventRecord (GPU signal)    │ │
│  │    ↓                              │ │
│  │  Update write_idx in SharedMemory│ │
│  └───────────────────────────────────┘ │
│                 ↓                       │
└─────────────────┼───────────────────────┘
                  │
                  │ SharedMemory (v0.5.0 protocol, same layout)
                  │
┌─────────────────┼───────────────────────┐
│                 ↓                       │
│   TouchDesigner Process (Consumer)     │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ CUDAIPCExtension (Receiver mode)  │ │
│  │                                   │ │
│  │  import_frame(script_top)        │ │
│  │    ↓                              │ │
│  │  Wait on IPC event (GPU-side)    │ │
│  │    ↓                              │ │
│  │  scriptTOP.copyCUDAMemory()      │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

Both directions share the **same v0.5.0 binary protocol** — the consumer is symmetric regardless of whether the producer is TD or Python.

---

## TD-Side Extension Architecture

The TouchDesigner extension (`CUDAIPCExtension`) uses a **facade-with-delegation** pattern to keep Sender and Receiver concerns in separate engine classes.

```
CUDAIPCExtension  (~300 LOC facade)
├── TDHost / RealTDHost         ← adapter: isolates all ownerComp.par.*, op(), cudaMemory() calls
├── TDSenderConfig              ← frozen dataclass: all CUDALINK_* env-var reads in one place
└── _engine: TDSenderEngine | TDReceiverEngine
      ├── TDSenderEngine (~415 LOC thin adapter) ← TD pixel-format bridging + GpuFrame bridge;
      │     delegates GPU alloc / SHM write / IPC export / CUDA graphs to Exporter
      └── TDReceiverEngine (~1070 LOC) ← owns SHM attach, IPC open, copyCUDAMemory calls
```

**Mode switching** tears down the current engine and constructs a fresh one — guaranteeing zero cross-mode state leak. The old engine's `cleanup()` is called first, which frees all CUDA resources before the new engine is constructed.

**TDHost seam**: all `ownerComp.par.*`, `ownerComp.op("ExportBuffer")`, `top.cudaMemory()`, and `scriptTOP.copyCUDAMemory()` calls go through `TDHost`/`TOPHandle` protocols. Tests inject `FakeTDHost`/`FakeTOPHandle` — no TD runtime required.

**textDAT binding**: every `.py` file in `td_exporter/` corresponds to a Text DAT inside the `CUDAIPCLink` Base COMP. Imports between them resolve within the COMP namespace (e.g., `from TDSender import TDSenderEngine` finds the `TDSender` sibling DAT). See `docs/TOX_BUILD_GUIDE.md` for the full assembly sequence.

**Two deployment modes** — `CUDALinkBootstrap` (a new Text DAT, the first import in `CUDAIPCExtension.py`) enables a choice at COMP init:

- **Library mode** (recommended): install `cuda_link` into an external folder with `install_td_library.cmd`. Ensure TouchDesigner can import it either by setting `CUDALINK_LIB_PATH` to that folder before launching TD or by adding the folder to TD Preferences → Python 32/64 bit Module Path. The bootstrap injects the environment-variable path when set, imports the installed package, and registers all 15 mirror module names as `sys.modules` aliases to the installed `cuda_link.*` submodules — so the 15 mirror Text DATs can be removed from the `.tox` entirely.
- **Fallback / classic mode**: if `cuda_link` cannot be imported or alias registration fails, the bootstrap falls back to the sibling Text DAT mirrors. All 15 mirror Text DATs must be present in the COMP (the original deployment story, unchanged). See ADR-0003 for rationale.

---

## SharedMemory Protocol

### Binary Layout (433 bytes for 3 slots)

```
┌─────────────────────────────────────────────────────────────┐
│ HEADER (20 bytes)                                           │
├─────────────────────────────────────────────────────────────┤
│ [0-3]     magic (uint32, little-endian)                     │
│           Protocol validation: 0x43495044 ("CIPD")          │
│ [4-11]    version (uint64, little-endian)                   │
│           Increments on producer re-initialization          │
│ [12-15]   num_slots (uint32, little-endian)                 │
│           Number of ring buffer slots (typically 3)         │
│ [16-19]   write_idx (uint32, little-endian)                 │
│           Atomic counter, increments every frame            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ SLOT 0 (128 bytes)                                          │
├─────────────────────────────────────────────────────────────┤
│ [20-83]     cudaIpcMemHandle_t (64 bytes)                   │
│             GPU memory handle for IPC transfer              │
│ [84-147]    cudaIpcEventHandle_t (64 bytes)                 │
│             GPU event handle for synchronization            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ SLOT 1 (128 bytes)                                          │
├─────────────────────────────────────────────────────────────┤
│ [148-211]   cudaIpcMemHandle_t (64 bytes)                   │
│ [212-275]   cudaIpcEventHandle_t (64 bytes)                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ SLOT 2 (128 bytes)                                          │
├─────────────────────────────────────────────────────────────┤
│ [276-339]   cudaIpcMemHandle_t (64 bytes)                   │
│ [340-403]   cudaIpcEventHandle_t (64 bytes)                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ FOOTER (29 bytes)                                           │
├─────────────────────────────────────────────────────────────┤
│ [404]       shutdown_flag (uint8)                           │
│             Producer sets to 1 on exit                      │
│ [405-424]   metadata (20 bytes)                             │
│             [405-408]   width (uint32)                      │
│             [409-412]   height (uint32)                     │
│             [413-416]   num_comps (uint32)                  │
│             [417]       format_kind (uint8)                 │
│                         cudaChannelFormatKind:              │
│                         0=Signed, 1=Unsigned, 2=Float       │
│             [418]       bits_per_comp (uint8) — 8/16/32/64  │
│             [419-420]   flags (uint16 LE)                   │
│                         bit 0: bfloat16 (kind=Float,bits=16)│
│                         bits 1-15: reserved=0               │
│             [421-424]   data_size (uint32)                  │
│                         Actual buffer size in bytes         │
│ [425-432]   timestamp (float64)                             │
│             Producer timestamp for latency measurement      │
└─────────────────────────────────────────────────────────────┘

Total: 20 + 3*128 + 1 + 20 + 8 = 433 bytes
```

### Variable Slot Count Formula

For `N` slots:

```
Total Size = 20 + (N × 128) + 1 + 20 + 8 bytes

shutdown_offset = 20 + (N × 128)
metadata_offset = 20 + (N × 128) + 1
timestamp_offset = 20 + (N × 128) + 1 + 20
```

**Examples**:

- 2 slots: 20 + 2×128 + 1 + 20 + 8 = 305 bytes, shutdown at [276]
- 3 slots: 20 + 3×128 + 1 + 20 + 8 = 433 bytes, shutdown at [404]
- 4 slots: 20 + 4×128 + 1 + 20 + 8 = 561 bytes, shutdown at [532]

---

## Ring Buffer Design

### Motivation

A single-buffer approach would require producer and consumer to synchronize on every frame, blocking each other. The ring buffer allows **pipelining**: producer writes to slot N while consumer reads from slot N-1.

### 3-Slot Pipeline Flow

```
Time →

Frame 0:
  Producer writes → Slot 0
  Consumer idle (no frames yet)

Frame 1:
  Producer writes → Slot 1
  Consumer reads ← Slot 0  (parallel)

Frame 2:
  Producer writes → Slot 2
  Consumer reads ← Slot 1  (parallel)

Frame 3:
  Producer writes → Slot 0  (wraps)
  Consumer reads ← Slot 2  (parallel)

Frame 4:
  Producer writes → Slot 1  (wraps)
  Consumer reads ← Slot 0  (parallel)
```

**Key insight**: Consumer is always **1 frame behind** producer, but this latency is negligible (~16ms at 60 FPS) and enables **zero blocking**.

### Slot Selection Logic

#### Producer (writes to current slot)

```python
write_idx += 1  # Increment before writing
slot = write_idx % NUM_SLOTS
# Write frame data to gpu_buffer[slot]
# Update write_idx in SharedMemory
```

#### Consumer (reads from previous slot)

```python
write_idx = read_from_shm()  # Current write_idx
if write_idx == 0:
    read_slot = 0  # Special case: no frames written yet
else:
    read_slot = (write_idx - 1) % NUM_SLOTS
# Read frame data from gpu_buffer[read_slot]
```

**Example sequence (3 slots)**:

| write_idx | Producer Slot | Consumer Slot |
|-----------|---------------|---------------|
| 0         | (init)        | 0 (special)   |
| 1         | 1             | 0             |
| 2         | 2             | 1             |
| 3         | 0 (wraps)     | 2             |
| 4         | 1             | 0             |
| 5         | 2             | 1             |

---

## GPU Synchronization

### Strategy: CUDA IPC Events

CUDA IPC events provide **GPU-side synchronization** without CPU involvement. This is critical for low-microsecond overhead.

#### Producer Side (Record Event)

```python
cuda.memcpy(dst=gpu_buffer[slot], src=top_cuda_ptr, count=size, kind=D2D)
cuda.record_event(ipc_event[slot])  # ← GPU-side signal, ~1-2μs
write_idx += 1
shm.buf[12:16] = struct.pack("<I", write_idx)
```

**Performance**: `cudaEventRecord` takes ~0.5-2μs, does NOT block CPU.

#### Consumer Side (Wait on Event)

```python
read_slot = (write_idx - 1) % NUM_SLOTS
cuda.wait_event(ipc_event[read_slot])  # ← GPU-side wait, ~0.5-2μs
tensor = tensors[read_slot]  # Zero-copy access, already valid
```

**Performance**: `cudaEventQuery` + `cudaStreamWaitEvent` take ~0.5-2μs combined.

### Producer-Stream Ordering (Cross-Stream Sync)

The IPC stream is created as **non-blocking and high-priority** (`StreamFlags.NON_BLOCKING`,
`CUDALINK_LIB_STREAM_PRIO=high`).  This means the IPC stream has no implicit FIFO ordering
relationship with any other stream, including the producer's default or compute stream.

#### The Race — What Happens Without Ordering

```
Producer stream:   [kernel writes to src_buffer] ...
IPC stream:                                        [D2D memcpy src→ring_slot]  ← reads before write!
```

If `export()` is called without arming ordering, the D2D memcpy on the IPC stream may execute
concurrently with (or before) the producer kernel writes, producing torn or gray frames — silently.
`CUDALINK_EXPORT_SYNC=1` does **not** fix this: it inserts a CPU-blocking sync *after* the memcpy,
not before it.

#### Two Equivalent APIs to Arm Ordering

**Option A — GpuFrame.producer_stream (per-frame):**

```python
stream_handle = torch.cuda.current_stream().cuda_stream   # PyTorch
# stream_handle = cupy.cuda.get_current_stream().ptr       # CuPy
exporter.export(GpuFrame(ptr=..., size=..., producer_stream=stream_handle))
```

**Option B — record_source_sync() (explicit, or one-time for same-stream callers):**

```python
exporter.record_source_sync(stream_handle)  # records event on producer stream
exporter.export(GpuFrame(ptr=..., size=...))
```

Both cause `stream_wait_event(ipc_stream, source_sync_event)` before the D2D memcpy, ordering
the copy after all previously enqueued work on `producer_stream`.  The CPU is never blocked.

#### Synchronous Writers

If you fill the source buffer via a *synchronous* H2D copy (e.g., `cudaMemcpy` with
`cudaMemcpyHostToDevice`), the write is complete before the Python call returns.  You may pass
`producer_stream=0` (the CUDA legacy default stream) to declare ordering without a real kernel
stream:

```python
cuda.memcpy(dst=staging_ptr, src=host_buf, count=nbytes, kind=1)  # H2D, synchronous
exporter.export(GpuFrame(ptr=staging_ptr, size=nbytes, producer_stream=0))
```

#### Enforcement

| Mode | Behaviour |
|------|-----------|
| Default (`CUDALINK_REQUIRE_SOURCE_SYNC=0`) | `logger.warning` once per exporter instance |
| Strict (`CUDALINK_REQUIRE_SOURCE_SYNC=1`) | `ValueError` raised immediately |

Set `require_source_sync=True` in `ExportPolicy` or `CUDALINK_REQUIRE_SOURCE_SYNC=1` to enforce
ordering at call sites during development.

#### Export-Sync Default and Coexistence Safety

**Python `Exporter` API:** `export_frame()` defaults to **async** (`CUDALINK_EXPORT_SYNC`
unset or `0`). The CUDA IPC event provides correct cross-process GPU ordering; coexistence
safety relies on explicit per-engine streams + producer-stream ordering, not blocking export.

**TD Sender (v1.10.1+):** defaults to **blocking** (`CUDALINK_EXPORT_SYNC` unset → `1`).
The TD source is TD's cook-scoped TOP texture (`cm.ptr`) — an externally-owned, transient
pointer reclaimed the instant the cook returns.  Async export lets TD reclaim the source
while the queued D2D copy is still executing → reads freed memory → CUDA 719.
`CUDALINK_EXPORT_SYNC=0` opts into async only when the caller guarantees the source buffer
outlives the copy (e.g. a stable device ring buffer passed into TD via `copyCUDAMemory`).

**Critical distinction — ordering vs. lifetime:**

- `record_source_sync` / `producer_stream` / `_arm_same_stream_ordering` are *pre-copy
  ordering* primitives: ensure the source is fully written before the copy starts.
  They do **not** guarantee the source outlives the queued read.
- `CUDALINK_EXPORT_SYNC=1` is the **source-lifetime guard**: CPU-blocks until the D2D
  finishes, so the source is provably safe to release when `export()` returns.

Three distinct stream hazards are each addressed by the correct mechanism:

- **Receiver teardown TDR** (fixed v1.4.1) — eliminated by dedicated, persistent per-engine
  streams (`CUDALINK_TD_PERSIST_STREAM=1`, default). No sender-side sync needed.
- **Producer-side cross-stream race** (fixed v1.9.0) — eliminated by `record_source_sync` /
  `require_source_sync` (see above). These ordering primitives guarantee the source is fully
  written before the copy *starts* — they do not keep the source alive past the queued D2D read.
- **TD Sender source-buffer lifetime race** (fixed v1.10.1, CUDA 719) — the TD source is TD's
  transient cook-scoped TOP texture (`cm.ptr`): TD reclaims it when the cook returns. Async
  export can delay the IPC-stream copy past cook exit → reads freed memory.  Fixed by **TD
  Sender blocking by default** (post-copy `stream_synchronize`): the source is provably safe to
  release when `export()` returns.  `CUDALINK_EXPORT_SYNC=0` opts back into async only when the
  caller guarantees the source buffer outlives the copy.

See `docs/PROFILING.md §8 EXPORT_SYNC defaults` for the full hazard analysis and timeline.

---

### Fallback: CPU Synchronization

If IPC events are unavailable (older CUDA versions), fall back to CPU sync:

#### Producer

```python
if frame_count % 10 == 0:  # Only sync every 10 frames
    cuda.synchronize()  # ← Blocks CPU until GPU idle
```

#### Consumer

```python
torch.cuda.synchronize()  # ← Blocks CPU until GPU idle
```

**Performance**: `cudaDeviceSynchronize()` takes ~10-50μs, **10-25x slower** than IPC events.

---

### CUDA Graphs (v1.10.0+)

When `CUDALINK_USE_GRAPHS=1` (default for the Python `Exporter`; `CUDALINK_TD_USE_GRAPHS` for the
TD Sender — default off in TD), the per-frame export path captures a CUDA graph once and replays
it on every subsequent frame.  The graph folds `stream_wait_event` + `cudaMemcpyAsync` +
`cudaEventRecord` into a **single WDDM submission** (`cudaGraphLaunch`) instead of three separate
kernel dispatches.

**Effect**: 3 WDDM submissions/frame → 1.  Measured gain at 1080p uint8 standalone:
−4.0 µs p50 (22%) from 17.9 µs (async, no graphs) to 13.9 µs (async+graphs).  See
[docs/BENCHMARKS.md](BENCHMARKS.md) for the full 2×2 matrix.

Graph replay is transparent: on the first frame after a geometry or dtype change the graph is
automatically re-captured, then replayed for subsequent frames.  Auto-fallback to
`cudaMemcpyAsync` occurs on capture or launch failure.

---

### R1: torch GPU-side wait opt-in (v1.11.0)

When `CUDALINK_TORCH_GPU_WAIT=1`, the torch consumer path replaces the host-side
`cudaEventSynchronize` on the IPC producer event with a GPU-resident dependency edge via
`cudaStreamWaitEvent` on the consumer stream.  This eliminates the CPU round-trip that WDDM
batches into a multi-hundred-microsecond stall.

```
Default (CUDALINK_TORCH_GPU_WAIT=0):
  cudaEventSynchronize(ipc_event[slot])              <- CPU blocks until GPU signal, ~50-200 us WDDM-batched

R1 GPU-side wait (CUDALINK_TORCH_GPU_WAIT=1):
  cudaStreamWaitEvent(consumer_stream, ipc_event[slot])  <- GPU dependency, ~0.5-2 us CPU overhead
  <tensor copy enqueued after the event on consumer_stream>
```

Gain: ~50–200 µs/frame depending on WDDM batch timing; safe alongside CUDA Graphs.

**Adaptive latch** (`CUDALINK_TORCH_GPU_WAIT_ADAPTIVE=1`): monitors per-frame event-wait time and
permanently enables GPU-side wait after three consecutive frames exceed the heuristic threshold —
no manual `CUDALINK_TORCH_GPU_WAIT=1` required.  One-way: once latched it stays enabled for the
session lifetime.

> **Scope**: torch frame path only (`get_frame()` → `torch.Tensor`).  CuPy and numpy paths are
> unaffected.

---

### R2: Win32 Named-Event Doorbell (v1.11.0)

Opt-in consumer-idle signaling primitive (`CUDALINK_DOORBELL=1`, must be set on **both** producer
and consumer).  Replaces the consumer's poll-sleep idle loop with a blocking `WaitForSingleObject`
call on a Win32 auto-reset named event, eliminating busy-wait CPU usage between frames.

**Flow**:

```
Producer (after advancing write_idx):
  SetEvent(doorbell_handle)                      <- ~0.02-0.10 ms, cook-thread safe, no FPS dip

Consumer (idle -- get_frame() returned NO_FRAME):
  importer.wait_for_doorbell(timeout_secs=2.0)   <- blocks on WaitForSingleObject, wakes on SetEvent
```

**Lost-wakeup guard**: before blocking, `wait_for_doorbell` re-checks `write_idx`.  If the
producer advanced `write_idx` between the `get_frame()` poll and the `WaitForSingleObject` call,
the function returns immediately — preventing a stall of up to `timeout_secs`.

**Measured gains** (TD Sender → Python subprocess, 60 FPS):

- CPU: ~3-4% → ~0.3-1%
- Notify latency p95: ~10× tighter (0.02–0.10 ms vs 0.04–1.80 ms poll baseline)
- Teardown: IPC handles close in ~0.6 ms (no 2 s hang, no orphaned handles)

**Scope and limitations**:

- Windows-only (`CreateEventW` / `WaitForSingleObject`).
- Single consumer only — one `SetEvent` releases exactly one waiter.
- `TDReceiver.py` (in-TD COMP on the cook thread) is **excluded** from doorbell wiring; it returns
  immediately on `NO_FRAME` and must not block in a cook context.
- Env var propagates across `subprocess.Popen` boundaries.

---

## Lifecycle

### Phase 1: Initialization

**Producer**:

1. Allocate N GPU buffers (`cuda.malloc()`)
2. Create IPC handles for each buffer (`cuda.ipc_get_mem_handle()`)
3. Create IPC events for each slot (`cuda.create_ipc_event()`)
4. Create SharedMemory (size = 20 + N*128 + 29)
5. Write version=1, num_slots=N, write_idx=0, all handles to SharedMemory

**Consumer**:

1. Open SharedMemory (retry with backoff if not ready)
2. Read version, num_slots from header
3. For each slot:
   - Read IPC memory handle (64 bytes)
   - Open handle with `cuda.ipc_open_mem_handle()` → GPU pointer
   - Read IPC event handle (64 bytes)
   - Open event with `cuda.ipc_open_event_handle()` → event
4. Create zero-copy tensor views (if torch available)

**Note on pixel format compatibility** (empirically probed, TD 2025.32820 — see `verification/results/cuda_memory_probe_20260510_090919.json`):

| Category | Formats | `cudaMemory()` behaviour |
|---|---|---|
| **Supported** (no conversion) | 8/16-bit fixed and 32-bit float in R/RG/RGBA/A variants | Returns correct `uint8` / `uint16` / `float32` buffer |
| **Rejected outright** | All 4 float16 variants (R/RG/RGBA/A); 10-bit RGB / 2-bit Alpha fixed | Raises exception; Sender skips frame + tints component **yellow** via `parent().color`; resumes when upstream format is corrected |
| **Silent corruption** | 11-bit float (RGB) | `cudaMemory()` "succeeds" but returns `dataType=uint8, numComps=4` (raw byte layout of the 32-bit packed word, NOT the 11:11:10 float semantic); treated as unsupported — same skip + yellow-tint policy |

The Sender extension (`_is_unsupported_format`) detects all six problematic formats by substring match on the pixel-format string. On detection each bad frame:

1. Calls `set_warning_status(msg)` (sets `parent().color` to amber-yellow — visible on the COMP node body, idempotent).

The frame is skipped, and `clear_status()` is called as soon as the upstream format is corrected (restoring the original COMP color). Engine-fatal errors (init failure, IPC handle failure, GPU alloc failure) instead call `set_error_status(msg)`, which tints the COMP red and emits a red `addScriptError` badge. No auto-conversion is performed; fix the upstream source TOP instead. Supported formats: `uint8`, `uint16` (fixed), `float32` in R/RG/RGBA/A channel configurations.

> **Why tint-only (empirically confirmed):** TD has no anytime-yellow-badge API — `addScriptWarning` does not exist; `addWarning` raises `tdError: Cannot set warning outside of cook` when called from `onFrameEnd` (a post-cook lifecycle callback, confirmed via `verification/results/probe_addwarning_*`); `addScriptError` is always red. A child Script TOP's `onCook` IS a valid cook context where `addWarning` succeeds, but TD does not propagate the child warning to the parent COMP boundary tile — neither via `comp.warnings()` nor as a visual badge (confirmed via `verification/results/probe_cook_context_*`). The COMP body tint is therefore the only COMP-local warning surface available.

### Phase 2: Steady State (Per-Frame)

**Producer** (~2-5µs IPC overhead, plus GPU D2D copy):

```
get TOP's cudaMemory() → src_ptr
slot = write_idx % NUM_SLOTS
cudaMemcpy D2D (src_ptr → gpu_buffer[slot])  ← GPU work, scales with frame size
cudaEventRecord(ipc_event[slot])             ← ~0.5-2µs
write_idx += 1
shm.buf[12:16] = struct.pack("<I", write_idx) ← ~0.5µs
```

**Consumer** (~1-3µs overhead):

```
write_idx = struct.unpack("<I", shm.buf[12:16])  ← ~0.5µs
read_slot = (write_idx - 1) % NUM_SLOTS
cudaStreamWaitEvent(ipc_event[read_slot])       ← ~0.5-2µs (GPU-side)
return tensors[read_slot]                        ← Zero-copy, 0µs
```

**Total IPC primitive overhead**: ~3-8µs per frame (producer + consumer). Full `export_frame()` with EXPORT_SYNC=1 (TD Sender default, v1.10.1+; Python `Exporter` defaults to async) includes GPU D2D completion: p50 24 µs (512×512) → 106 µs (1080p) → 357 µs (4K) float32 RGBA on RTX 4090 / PCIe 4.0. Python `Exporter` async+graphs p50: 13.9 µs (1080p uint8). See [docs/BENCHMARKS.md](BENCHMARKS.md) for the full breakdown.

### Phase 3: Re-initialization

**Trigger**: Producer detects resolution change (e.g., TOP resolution changed).

**Producer**:

1. Free old GPU buffers
2. Re-allocate with new size
3. Create new IPC handles
4. Increment version in SharedMemory (e.g., 1 → 2)
5. Write new handles

**Consumer**:

1. Detect version change: `new_version != stored_version`
2. Close old IPC handles
3. Re-read num_slots (may have changed)
4. Open new IPC handles
5. Re-create tensor views
6. Update stored_version

**Performance**: Re-initialization takes ~50-100μs (one-time cost), does NOT disrupt frame flow.

### Phase 4: Shutdown

**Producer**:

1. Set `shm.buf[shutdown_offset] = 1`
2. Close SharedMemory (but don't unlink - consumer may still need it)
3. Free GPU buffers

**Consumer**:

1. Detect shutdown flag: `shm.buf[shutdown_offset] == 1`
2. Close IPC handles
3. Close SharedMemory
4. Clean up resources

---

## Error Handling

### Producer Crashes Mid-Frame

**Scenario**: TD crashes after `cudaMemcpy` but before `cudaEventRecord`.

**Impact**: On the default CPU-side wait paths, the consumer waits until
`ImportSpec.timeout_ms` (5 seconds by default) and then returns
`ImportOutcome.TIMEOUT`; it does not wait indefinitely.

**Mitigation**: Handle `ImportOutcome.TIMEOUT` and inspect the producer's logs.
The opt-in torch GPU-side wait (`CUDALINK_TORCH_GPU_WAIT=1`, or an explicit
consumer stream) and the CuPy path enqueue the event wait on the GPU instead;
those paths intentionally do not report `TIMEOUT`, so a hung producer can
stall the consumer stream. Exception: if a ring slot carries no IPC event at
all, both GPU-side paths fall back to the CPU wait so the read stays ordered —
in that case `TIMEOUT` is still reachable.

### Consumer Crashes

**Scenario**: Python process terminates without calling `cleanup()`.

**Impact**: IPC handles remain open, GPU memory not released by consumer.

**Mitigation**: OS cleans up IPC handles automatically on process exit. Producer can detect stale consumer via SharedMemory timestamps (future enhancement).

### Version Mismatch

**Scenario**: Consumer opens handles for version 1, producer re-initializes to version 2.

**Impact**: Consumer reads from stale GPU buffers.

**Detection**: Consumer checks `shm.buf[4:12]` (version) every frame. If changed, triggers re-initialization.

### SharedMemory Corruption

**Scenario**: Manual editing or concurrent access corrupts SharedMemory.

**Impact**: Undefined behavior, likely crashes.

**Prevention**: Use dedicated `Ipcmemname` per exporter instance, avoid manual access.

### Cross-Process Error Attribution

**Important**: `cudaPeekAtLastError` / `cudaGetLastError` only inspect the CUDA context of the **calling process**. A GPU memory fault or async kernel error in the **producer** process will **not** propagate to the consumer process via the IPC event mechanism.

**What the consumer observes**: a delayed or absent IPC event (timeout in `_wait_for_slot`), not a CUDA error code.

**Where the error surfaces**: the producer's own per-frame sticky-error check (`check_sticky_error`, controlled by `CUDALINK_STICKY_ERROR_CHECK`, default ON) will catch the fault on the next producer frame and raise there.

**Debugging guideline**: when the consumer reports a timeout or stall, check the **producer process logs first** — the root fault is almost always on the producer side.

---

## Performance Characteristics

### Overhead Breakdown (1080p RGBA float32, 3 slots)

| Operation | Time | Location | Blocking |
|-----------|------|----------|----------|
| `cudaMemcpy D2D` | 60-80μs | Producer | GPU-async |
| `cudaEventRecord` | 0.5-2μs | Producer | CPU-non-blocking |
| `write_idx update` | 0.5μs | Producer | CPU |
| `read write_idx` | 0.5μs | Consumer | CPU |
| `cudaStreamWaitEvent` | 0.5-2μs | Consumer | GPU-async |
| **Total CPU overhead** | **~3-8μs** | Both | - |

### Measured Benchmarks

RTX 4090 / PCIe 4.0 x16 / Windows 11 / driver 596.36. Full tables and per-resolution breakdowns: **[docs/BENCHMARKS.md](BENCHMARKS.md)**.

**`export_frame()` standalone p50 (isolated — no consumer process, EXPORT_SYNC=1):** 24 µs (512×512) → 106 µs (1080p f32) → 357 µs (4K f32).
**Python `Exporter` async+graphs (default):** 13.9 µs p50 at 1080p uint8 (−31.3 µs / −69% vs 45.2 µs blocking baseline). CUDA graphs collapse 3 WDDM submissions/frame to 1. See [docs/BENCHMARKS.md](BENCHMARKS.md) for the full 2×2 async/sync × graphs/no-graphs matrix.

**IPC roundtrip p50 (two separate processes, graphs=off):** export 662–1483 µs; `get_frame_numpy()` 0.38–5.03 ms; IPC notify ~136–286 µs (resolution-independent signaling latency).

### Throughput Limits

**Theoretical max FPS** (ignoring application logic; isolated export, EXPORT_SYNC=1):

```
FPS_max = 1 / export_frame_p50
        = 1 / 106 us   (1080p f32)  ~= 9,400 FPS
        = 1 / 357 us   (4K f32)     ~= 2,800 FPS
```

**Practical limit** (with 60 FPS TD cook + 16ms AI model inference):

```
FPS_actual = min(TD_FPS, 1 / inference_time)
           = min(60, 1 / 0.016)
           = 60 FPS
```

**Latency** (producer write -> consumer read, bench_sweep + bench_d2h_streams, 1080p f32):

```
Latency ~= IPC_notify + D2H_copy
        ~= 136 us + 1,320 us
        ~= 1.5 ms
```

This latency is **imperceptible** for real-time applications.

---

## Comparison: CUDA IPC vs CPU SharedMemory

CUDA IPC zero-copies GPU memory across processes; CPU SharedMemory adds two memcpys (GPU->CPU on the producer, CPU->GPU on the consumer). Numerical impact at typical resolutions:

**1920x1080 float32 RGBA (31.6 MB/frame):**

| Metric | CPU SharedMemory | CUDA-Link | Speed-up |
|--------|------------------|-----------|----------|
| Producer write | 2.60 ms | 117 us (bench_graphs) / 1483 us (bench_sweep) | 1.8x – 22.2x |
| Consumer read (D2H) | 2.48 ms | 1.32 ms (bench_d2h_streams) / ~5.0 ms (bench_sweep) | 0.5x – 1.9x |
| End-to-end | 5.37 ms | ~1.5 ms (IPC notify 136 us + D2H 1.32 ms) | ~3.6x |

**512x512 float32 RGBA (4 MB/frame):**

| Metric | CPU SharedMemory | CUDA-Link | Speed-up |
|--------|------------------|-----------|----------|
| Producer write | 361 us | 22 us (bench_graphs) | 16.4x |
| Consumer read (D2H) | 350 us | 0.18 ms (bench_d2h_streams) | 1.9x |
| End-to-end | 1.02 ms | ~0.35 ms (IPC notify 172 us + D2H 0.18 ms) | ~2.9x |

**Methodology notes:**

- CPU SharedMemory numbers are from prior measurements on an unspecified earlier RTX-class system, PCIe 4.0. CUDA-Link numbers are from RTX 4090 / driver 596.36 / PCIe 4.0 x16. PCIe generation matches; D2H bandwidth comparison is meaningful. Producer-side write is GPU-independent (CPU->SHM is CPU-bound).
- `bench_graphs` measures pure isolated `export_frame()` (single producer, no consumer). `bench_sweep` measures `export_frame()` under concurrent consumer load — cross-process WDDM contention inflates the wall-clock sample. Use the bench_graphs figure for "what does CUDA-Link cost in the integrated path"; use the bench_sweep figure for "what does the spawn-process Python-only roundtrip look like."
- For zero-copy GPU consumers (`get_frame()` -> torch tensor, `get_frame_cupy()`), the read column collapses to <5 us and the gap widens further. The D2H comparison only applies when the consumer needs CPU data.

**Architectural differences:**

| Property | CUDA-Link | CPU SharedMemory |
|---|---|---|
| Memory copies | 0 (GPU D2D only) | 2 (GPU->CPU, CPU->GPU) |
| Sync primitive cost | 3-8 us | N/A |
| Platform support | Windows only | Cross-platform |
| Zero-copy read mode | `get_frame()` / `get_frame_cupy()` | not available |

**TouchOUT / Spout**: not measured. The original project README referenced TD-side loggers for these baselines but no scripts were ever committed. A comparison would require building those loggers from scratch against a live TD instance.

**TD->Python note**: When a live TouchDesigner sender is the producer, `cudaIpcOpenMemHandle` requires TD's CUDA runtime instance. Loading a second CUDA runtime (e.g. `cudart64_12.dll`) in TD's process causes error 400 — see `td_exporter/CUDAIPCWrapper.py` `_load_cuda_runtime()` for details.

**Conclusion**: CUDA-Link is 2x – 3.4x faster end-to-end than CPU SharedMemory at typical resolutions and 4x – 19x faster on the producer side. Use CPU SharedMemory for cross-platform or non-CUDA workflows.

---

## Architectural Decisions

### Why Legacy IPC Over VMM API

> This decision is recorded canonically in
> [`docs/adr/0004-legacy-cuda-ipc-over-vmm.md`](adr/0004-legacy-cuda-ipc-over-vmm.md).
> The summary below is retained for inline reference.

This project uses CUDA Runtime API IPC (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`)
rather than the Driver API Virtual Memory Management (VMM) approach
(`cuMemCreate` / `cuMemExportToShareableHandle`).

**Rationale**: We share **linear memory** (`cudaMalloc`), not **textures** (`cudaArray`).
TouchDesigner's `top_op.cudaMemory()` returns linearized pixel data. The VMM API's
advantages — texture layout preservation, SECURITY_ATTRIBUTES, virtual address
manipulation — solve problems this project does not have.

| Factor | Legacy IPC (Chosen) | VMM API (Rejected) |
|--------|--------------------|--------------------|
| Code complexity | ~600 lines | ~1,500+ lines |
| Allocation | 1 step (`cudaMalloc`) | 4 steps (create + reserve + map + access) |
| API level | Runtime API (automatic context) | Driver API (manual context) |
| Performance | ~3-8μs IPC overhead | Same for linear D2D |
| TD compatibility | Proven | Unvalidated |

**Validation**: The IPC roundtrip sweep confirms legacy IPC works on Windows WDDM with CUDA 12.x. CuPy and dora-rs also use this approach.

**When VMM would be needed**: If sharing `cudaArray` objects directly (opaque
texture memory with swizzled layout) without linearization.

See [`docs/adr/0004-legacy-cuda-ipc-over-vmm.md`](adr/0004-legacy-cuda-ipc-over-vmm.md) for
the full decision record, including the rejected VMM alternative and the one condition that
would reopen the decision. A proof-of-concept VMM probe lives at
`scripts/probe/driver_api_ipc_probe.py`.

---

## Recent Optimizations (v1.10.x)

### Pipelined Device-to-Host Copy (P5, v1.10.0)

`CUDALINK_D2H_PIPELINED=1` (default off) enables a double-buffer pipeline that enqueues the
**next** D2H copy asynchronously while the consumer processes the **current** frame.  The copy
for frame *N+1* overlaps with consumer CPU work on frame *N*, hiding transfer latency.

- First `get_frame_numpy()` returns `ImportOutcome.NO_FRAME` (priming).
- Steady-state adds +1 frame of latency; the gain is `≈ d2h_copy_time` when
  `consumer_work > d2h_copy_time` (1080p break-even ~0.38 ms; 4K ~1.3 ms).
- On teardown and reconnect, the pipeline drains the in-flight copy (`NumpyBuffers.drain()`)
  and re-primes — the first frame after `RECONNECTING` always returns `NO_FRAME`, matching
  fresh-session behavior (v1.10.3).

### Windowed Telemetry (v1.10.3)

Both the TD Sender's `export=… µs avg` and the TD Receiver's `copy=… µs avg` summary lines
now report a **windowed (~150-frame) average** rather than a lifetime cumulative mean.  This
is implemented via `ReportWindow.add_sample()` / `avg_and_reset()` in `_profile.py`
(`FrameProfile.py` in TD), shared by both engines.

The windowed figure reflects current-session performance rather than converging monotonically
from first-frame startup; it resets each time the format changes.

### Hot-Path Allocation Reduction (P8, v1.10.2)

`get_frame()` (`_TorchBackend`) and `get_frame_cupy()` (`_CupyBackend`) cache their backend
instance and update `_stream` in-place instead of constructing a new object per call.
`AcquireResult` (returned by `acquire_slot`) is now a `NamedTuple` instead of a `@dataclass`,
eliminating the per-call `__dict__` allocation on the consumer hot path.

### Receiver Idle-Cook Skip (P11, v1.10.2)

`TDReceiverEngine.has_new_frame()` reads the SHM `write_idx`, `version`, and `shutdown` fields
before `import_buffer.cook(force=True)`.  When the producer has not written a new frame, the
cook is skipped entirely — saving one Script-TOP cook and all its Python overhead per idle TD
frame.  Effect is zero when producer ≥ TD frame rate; significant for slow producers (e.g. 30 FPS
AI inference into a 60 FPS TD project).

### v1.11.0: R1 GPU-side wait + R2 Doorbell + R3/R4 hygiene

See the dedicated subsections in **GPU Synchronization → R1** and **GPU Synchronization → R2** above for full architecture detail.  Summary:

- **R1** (`CUDALINK_TORCH_GPU_WAIT=1`): ~50–200 µs/frame gain on torch path by replacing host-side `cudaEventSynchronize` with a GPU-resident `cudaStreamWaitEvent`.  Adaptive auto-enable latch available via `CUDALINK_TORCH_GPU_WAIT_ADAPTIVE=1`.
- **R2** (`CUDALINK_DOORBELL=1`): Win32 named-event replaces poll-sleep on the consumer idle path; CPU ~3-4% → ~0.3-1%, notify latency p95 ~10× tighter.  Single-consumer, Windows-only.
- **R3**: pinned-memory f16 fallback in CuPy wait path — prevents dtype mismatch on f16 producers.

---

## Design Decisions

See `docs/adr/` for the full Architecture Decision Record index:

- **ADR-0001** — Port-adapter deepening (testable seams without real GPU)
- **ADR-0002** — Byte-identical TD mirror files
- **ADR-0003** — Library-mode bootstrap via `sys.path` injection
- **ADR-0004** — Legacy CUDA IPC over VMM driver API
- **ADR-0005** — Static typing hardening (per-file suppression policy, no category blankets)
- **ADR-0006** — Stay pure-Python (Rust `cuda-oxide`/`cudarc` evaluated and deferred; packaging conclusion superseded by ADR-0012)
- **ADR-0007** — Spout as a sidecar-launcher COMP, not an embedded transport
- **ADR-0008** — No native work on the D2H readback path
- **ADR-0009** — Accept in-process native code inside TD as a C++ Custom TOP (Proposed)
- **ADR-0012** — Fold the native extension into the core wheel (supersedes ADR-0006's packaging conclusion)
- **ADR-0013** — Prebuilt wheel distribution: Windows-only, cp311 native + py3-none-any fallback; end-user machines never compile

---

## Future Enhancements

1. **Adaptive slot count**: Automatically increase slots under high load.
2. **Multi-consumer support**: Multiple Python processes reading from one producer.
3. **Timeout on IPC event wait**: Prevent consumer hang if producer crashes mid-frame.

---

**Last Updated**: 2026-08-11
**Version**: 1.12.2
