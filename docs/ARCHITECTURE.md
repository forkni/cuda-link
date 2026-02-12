# CUDA IPC Architecture

Technical specification of the SharedMemory protocol, ring buffer design, and GPU synchronization strategy.

---

## System Overview

```
┌─────────────────────────────────────────┐
│   TouchDesigner Process (Producer)     │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ CUDAIPCExporter Extension         │ │
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
                  │ SharedMemory (593 bytes)
                  │ [version + handles + metadata]
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
│  │  Calculate read_slot = (write_idx-1) % N │
│  │    ↓                              │ │
│  │  Wait on IPC event (GPU-side)    │ │
│  │    ↓                              │ │
│  │  Return tensor/array (zero-copy) │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

---

## SharedMemory Protocol

### Binary Layout (625 bytes for 3 slots)

```
┌─────────────────────────────────────────────────────────────┐
│ HEADER (20 bytes)                                           │
├─────────────────────────────────────────────────────────────┤
│ [0-3]     magic (uint32, little-endian)                     │
│           Protocol validation: 0x43495043 ("CIPC")          │
│ [4-11]    version (uint64, little-endian)                   │
│           Increments on producer re-initialization          │
│ [12-15]   num_slots (uint32, little-endian)                 │
│           Number of ring buffer slots (typically 3)         │
│ [16-19]   write_idx (uint32, little-endian)                 │
│           Atomic counter, increments every frame            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ SLOT 0 (192 bytes)                                          │
├─────────────────────────────────────────────────────────────┤
│ [20-147]    cudaIpcMemHandle_t (128 bytes)                  │
│             GPU memory handle for IPC transfer              │
│ [148-211]   cudaIpcEventHandle_t (64 bytes)                 │
│             GPU event handle for synchronization            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ SLOT 1 (192 bytes)                                          │
├─────────────────────────────────────────────────────────────┤
│ [212-339]   cudaIpcMemHandle_t (128 bytes)                  │
│ [340-403]   cudaIpcEventHandle_t (64 bytes)                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ SLOT 2 (192 bytes)                                          │
├─────────────────────────────────────────────────────────────┤
│ [404-531]   cudaIpcMemHandle_t (128 bytes)                  │
│ [532-595]   cudaIpcEventHandle_t (64 bytes)                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ FOOTER (29 bytes)                                           │
├─────────────────────────────────────────────────────────────┤
│ [596]       shutdown_flag (uint8)                           │
│             Producer sets to 1 on exit                      │
│ [597-616]   metadata (20 bytes)                             │
│             [597-600]   width (uint32)                      │
│             [601-604]   height (uint32)                     │
│             [605-608]   num_comps (uint32)                  │
│             [609-612]   dtype_code (uint32)                 │
│                         0=float32, 1=float16, 2=uint8       │
│             [613-616]   data_size (uint32)                  │
│                         Actual buffer size in bytes         │
│ [617-624]   timestamp (float64)                             │
│             Producer timestamp for latency measurement      │
└─────────────────────────────────────────────────────────────┘

Total: 20 + 3*192 + 1 + 20 + 8 = 625 bytes
```

### Variable Slot Count Formula

For `N` slots:

```
Total Size = 20 + (N × 192) + 1 + 20 + 8 bytes

shutdown_offset = 20 + (N × 192)
metadata_offset = 20 + (N × 192) + 1
timestamp_offset = 20 + (N × 192) + 1 + 20
```

**Examples**:
- 2 slots: 20 + 384 + 1 + 20 + 8 = 433 bytes, shutdown at [404]
- 3 slots: 20 + 576 + 1 + 20 + 8 = 625 bytes, shutdown at [596]
- 4 slots: 20 + 768 + 1 + 20 + 8 = 817 bytes, shutdown at [788]

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

CUDA IPC events provide **GPU-side synchronization** without CPU involvement. This is critical for sub-microsecond overhead.

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
4. Create SharedMemory (size = 16 + N*192 + 1)
5. Write version=1, num_slots=N, write_idx=0, all handles to SharedMemory

**Consumer**:
1. Open SharedMemory (retry with backoff if not ready)
2. Read version, num_slots from header
3. For each slot:
   - Read IPC memory handle (128 bytes)
   - Open handle with `cuda.ipc_open_mem_handle()` → GPU pointer
   - Read IPC event handle (64 bytes)
   - Open event with `cuda.ipc_open_event_handle()` → event
4. Create zero-copy tensor views (if torch available)

### Phase 2: Steady State (Per-Frame)

**Producer** (~2-5μs overhead):
```
get TOP's cudaMemory() → src_ptr
slot = write_idx % NUM_SLOTS
cudaMemcpy D2D (src_ptr → gpu_buffer[slot])  ← GPU work, ~60-80μs for 1080p
cudaEventRecord(ipc_event[slot])             ← ~0.5-2μs
write_idx += 1
shm.buf[12:16] = struct.pack("<I", write_idx) ← ~0.5μs
```

**Consumer** (~1-3μs overhead):
```
write_idx = struct.unpack("<I", shm.buf[12:16])  ← ~0.5μs
read_slot = (write_idx - 1) % NUM_SLOTS
cudaStreamWaitEvent(ipc_event[read_slot])       ← ~0.5-2μs (GPU-side)
return tensors[read_slot]                        ← Zero-copy, 0μs
```

**Total overhead**: ~3-8μs per frame (producer + consumer)

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

**Detection**: Consumer checks `shm.buf[0:8]` (version) every frame. If changed, triggers re-initialization.

### SharedMemory Corruption

**Scenario**: Manual editing or concurrent access corrupts SharedMemory.

**Impact**: Undefined behavior, likely crashes.

**Prevention**: Use dedicated `Ipcmemname` per exporter instance, avoid manual access.

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

### Throughput Limits

**Theoretical max FPS** (ignoring application logic):

```
FPS_max = 1 / (memcpy_time + sync_overhead)
        ≈ 1 / (70μs + 3μs)
        ≈ 13,700 FPS
```

**Practical limit** (with 60 FPS TD cook + 16ms AI model inference):

```
FPS_actual = min(TD_FPS, 1 / inference_time)
           = min(60, 1 / 0.016)
           = 60 FPS
```

**Latency** (producer write → consumer read):

```
Latency = 1 frame delay + GPU sync time
        = 16.7ms (at 60 FPS) + 2μs
        ≈ 16.7ms
```

This latency is **imperceptible** for real-time applications.

---

## Comparison: CUDA IPC vs CPU SharedMemory

| Metric | CUDA IPC | CPU SharedMemory | Speedup |
|--------|----------|------------------|---------|
| Per-frame overhead | ~3-8μs | ~1.5ms | **187-500x** |
| Memory copies | 0 (GPU-GPU) | 2 (GPU→CPU, CPU→GPU) | ∞ |
| Latency | ~16.7ms (1 frame) | ~16.7ms + 1.5ms | 1.09x |
| Setup cost | ~50-100μs | ~10μs | 0.1-0.2x |
| Platform support | Windows only | Cross-platform | - |

**Conclusion**: CUDA IPC is **200-500x faster** per-frame, but Windows-only. Use CPU SharedMemory for cross-platform or non-CUDA workflows.

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
| Performance | <2μs overhead | Same for linear D2D |
| TD compatibility | Proven | Unvalidated |

**Validation**: `benchmarks/test_cuda_ipc_windows.py` confirms legacy IPC works on
Windows WDDM with CUDA 12.x. CuPy and dora-rs also use this approach.

**When VMM would be needed**: If sharing `cudaArray` objects directly (opaque
texture memory with swizzled layout) without linearization.

See `References/CUDA IPC Texture Transfer Windows.txt` for full analysis.

---

## Future Enhancements

1. **Adaptive slot count**: Automatically increase slots under high load.
2. **Multi-consumer support**: Multiple Python processes reading from one producer.
3. **Bidirectional IPC**: Consumer → producer feedback (e.g., AI result overlay).

---

**Last Updated**: 2026-02-10
**Version**: 1.1.0
