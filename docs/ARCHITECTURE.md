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
│  │ CUDAIPCImporter                   │ │
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
│  │ CUDAIPCExporter                   │ │
│  │                                   │ │
│  │  export_frame(gpu_ptr, size)     │ │
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

#### Producer (writes to current slot):

```python
write_idx += 1  # Increment before writing
slot = write_idx % NUM_SLOTS
# Write frame data to gpu_buffer[slot]
# Update write_idx in SharedMemory
```

#### Consumer (reads from previous slot):

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

#### Producer Side (Record Event):

```python
cuda.memcpy(dst=gpu_buffer[slot], src=top_cuda_ptr, count=size, kind=D2D)
cuda.record_event(ipc_event[slot])  # ← GPU-side signal, ~1-2μs
write_idx += 1
shm.buf[12:16] = struct.pack("<I", write_idx)
```

**Performance**: `cudaEventRecord` takes ~0.5-2μs, does NOT block CPU.

#### Consumer Side (Wait on Event):

```python
read_slot = (write_idx - 1) % NUM_SLOTS
cuda.wait_event(ipc_event[read_slot])  # ← GPU-side wait, ~0.5-2μs
tensor = tensors[read_slot]  # Zero-copy access, already valid
```

**Performance**: `cudaEventQuery` + `cudaStreamWaitEvent` take ~0.5-2μs combined.

### Fallback: CPU Synchronization

If IPC events are unavailable (older CUDA versions), fall back to CPU sync:

#### Producer:
```python
if frame_count % 10 == 0:  # Only sync every 10 frames
    cuda.synchronize()  # ← Blocks CPU until GPU idle
```

#### Consumer:
```python
torch.cuda.synchronize()  # ← Blocks CPU until GPU idle
```

**Performance**: `cudaDeviceSynchronize()` takes ~10-50μs, **10-25x slower** than IPC events.

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

**Note on pixel format compatibility**: TouchDesigner 2025 (CUDA 12.8) rejects `rgba16float` formats from `cudaMemory()`. The Sender extension automatically detects this and sets a permanent `dtype_converter` Transform TOP (wired before ExportBuffer) to `rgba32float`, skipping one transition frame. Supported formats without conversion: `uint8`, `uint16` (fixed), `float32`.

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

**Total IPC primitive overhead**: ~3-8µs per frame (producer + consumer). Full `export_frame()` with EXPORT_SYNC=1 (default) includes GPU D2D completion: p50 42 µs (512×512) → 400 µs (4K) float32 RGBA on RTX 4090 / PCIe 4.0. See `bench_graphs.py` for resolution breakdown.

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

**Impact**: Consumer waits indefinitely on IPC event.

**Mitigation**: Consumer uses timeout on `wait_event` (future enhancement). Currently, consumer will hang - restart required.

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

Produced by `benchmarks/bench_graphs.py` and `benchmarks/bench_sweep.py` (2000 frames, EXPORT_SYNC=1, RTX 4090, driver 596.36, Windows 11, PCIe 4.0 x16).

**`export_frame()` wall-clock (bench_graphs, isolated -- no consumer process):**

| Resolution | Graphs off p50 | Graphs on p50 |
|---|---|---|
| 512x512 f32 | 42 us | 45 us |
| 1280x720 f32 | 59 us | 60 us |
| 1920x1080 f32 | 138 us | 133 us |
| 3840x2160 f32 | 400 us | 404 us |

With EXPORT_SYNC=1, GPU D2D copy time dominates both paths; CUDA Graphs provides <5% wall-clock difference at 1080p. In async mode (no EXPORT_SYNC), graphs reduce submission overhead from ~15.7 us to ~4.7 us at 1080p (WDDM transitions: 3 -> 2), but production default is EXPORT_SYNC=1.

**IPC roundtrip (bench_sweep, spawn-process producer + consumer, graphs=off):**

| Resolution | dtype | export p50 | get_numpy p50 | IPC notify p50 |
|---|---|---|---|---|
| 512x512 | f32 | 1316 us | 1435 us | 259 us |
| 512x512 | u8 | 970 us | 448 us | 255 us |
| 1280x720 | f32 | 1743 us | 4599 us | 297 us |
| 1280x720 | u8 | 1042 us | 1263 us | 254 us |
| 1920x1080 | f32 | 585 us | 1995 us | 240 us |
| 1920x1080 | u8 | 1234 us | 2650 us | 264 us |
| 3840x2160 | f32 | 566 us | 5848 us | 300 us |
| 3840x2160 | u8 | 566 us | 2585 us | 227 us |

- **export p50**: producer `export_frame()` wall-clock with concurrent consumer process (higher than isolated bench_graphs numbers due to cross-process WDDM contention).
- **get_numpy p50**: consumer `get_frame_numpy()` D2H copy wall-clock.
- **IPC notify p50**: producer write_idx update -> consumer SHM detection; ~250 us resolution-independent (ring-buffer notification latency, not GPU copy time).

Full results in `benchmarks/results/sweep_latest.csv`.

### Throughput Limits

**Theoretical max FPS** (ignoring application logic; bench_graphs isolated export, EXPORT_SYNC=1):

```
FPS_max = 1 / export_frame_p50
        = 1 / 138 us   (1080p f32)  ~= 7,200 FPS
        = 1 / 400 us   (4K f32)     ~= 2,500 FPS
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
        ~= 240 us + 1,350 us
        ~= 1.6 ms
```

This latency is **imperceptible** for real-time applications.

---

## Comparison: CUDA IPC vs CPU SharedMemory

CUDA IPC zero-copies GPU memory across processes with no CPU transit. Architectural differences vs CPU SharedMemory:

| Property | CUDA IPC | CPU SharedMemory |
|--------|----------|------------------|
| Memory copies | 0 (GPU D2D only) | 2 (GPU->CPU, CPU->GPU) |
| export_frame() p50 (1080p f32) | 138 us (bench_graphs) | n/a |
| get_frame_numpy() p50 (1080p f32) | ~2.0 ms D2H (bench_sweep) | n/a |
| D2H throughput (1080p f32, bench_d2h_streams) | 1.35 ms at 23.7 GB/s | memcpy limited |
| IPC sync primitives only | 3-8 us | N/A |
| Platform support | Windows only | Cross-platform |

**Read overhead note**: `get_frame_numpy()` performs a GPU-to-CPU copy. The zero-copy GPU modes (`get_frame()` -> torch tensor, `get_frame_cupy()`) have negligible read overhead and are the recommended path for AI pipelines.

**TD->Python note**: When a live TouchDesigner sender is the producer, `cudaIpcOpenMemHandle` requires TD's CUDA runtime instance. Loading a second CUDA runtime (e.g. `cudart64_12.dll`) in TD's process causes error 400 — see `td_exporter/CUDAIPCWrapper.py` `_load_cuda_runtime()` for details.

**Conclusion**: CUDA IPC eliminates CPU-side data movement entirely. Use CPU SharedMemory for cross-platform or non-CUDA workflows.

---

## Architectural Decisions

### Why Legacy IPC Over VMM API

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

**Validation**: `benchmarks/bench_sweep.py` confirms legacy IPC works on
Windows WDDM with CUDA 12.x. CuPy and dora-rs also use this approach.

**When VMM would be needed**: If sharing `cudaArray` objects directly (opaque
texture memory with swizzled layout) without linearization.

See `References/CUDA IPC Texture Transfer Windows.txt` for full analysis.

---

## Future Enhancements

1. **Adaptive slot count**: Automatically increase slots under high load.
2. **Multi-consumer support**: Multiple Python processes reading from one producer.
3. **Timeout on IPC event wait**: Prevent consumer hang if producer crashes mid-frame.

---

**Last Updated**: 2026-02-26
**Version**: 1.4.0
