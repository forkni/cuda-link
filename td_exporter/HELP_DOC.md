# CUDA-Link — Component Help

> **Name:** CUDA-Link
> **Description:** Zero-copy GPU texture sharing via CUDA IPC
> **Author:** forkni (forkni@gmail.com)

**Zero-copy GPU texture sharing between TouchDesigner and external Python processes using CUDA Inter-Process Communication (IPC).**

---

## Overview

CUDAIPCLink transfers GPU textures between TouchDesigner and a Python process without copying data through CPU memory. Texture data stays on the GPU at all times — only a small control packet (~433 bytes) is exchanged through OS shared memory to coordinate access.

The component operates in two modes: **Sender** (TouchDesigner exports textures to Python) and **Receiver** (Python sends frames back into TouchDesigner). Both directions use the same underlying protocol, so two TouchDesigner instances can also communicate directly with each other.

Per-frame overhead is typically **0.5–2 µs** — roughly 750× faster than copying textures through CPU shared memory (~1.5 ms at 1080p).

---

## How It Works

### Sender Mode — TD → Python

1. Each frame, the component calls `top_op.cudaMemory()` to get a raw GPU pointer to the upstream texture.
2. That texture is copied into a pre-allocated ring buffer slot on the GPU using `cudaMemcpyAsync` (device-to-device, never touching CPU memory).
3. A CUDA IPC event is recorded on that slot — a lightweight GPU-side signal (~1 µs).
4. A shared memory channel is updated with the current slot index and a producer timestamp.
5. The Python process reads the slot index, waits on the GPU event (without blocking the CPU), and accesses the texture as a zero-copy `torch.Tensor` or `cupy.ndarray`.

### Receiver Mode — Python → TD

1. An external Python process allocates GPU buffers, writes IPC handles into shared memory, and signals via CUDA IPC events.
2. On each frame start, the component reads the IPC handles from shared memory, waits on the GPU event, and copies the data into a Script TOP via `copyCUDAMemory()`.
3. The result is a live TD texture that updates every frame with the Python process's output.

### Ring Buffer Architecture

The component maintains **N independent GPU buffer slots** (N = `Numslots`, default 3). The producer writes into the current slot while the consumer simultaneously reads from the previous slot. This pipeline prevents either side from ever waiting on the other:

```
Frame 0:  Producer → Slot 0   Consumer idle
Frame 1:  Producer → Slot 1   Consumer ← Slot 0
Frame 2:  Producer → Slot 2   Consumer ← Slot 1
Frame 3:  Producer → Slot 0   Consumer ← Slot 2  (wraps)
```

The consumer is always one frame behind the producer. At 60 FPS this is ~16 ms — negligible for real-time AI pipelines.

### Shared Memory Protocol

The shared memory channel carries only control data (no pixel data):

| Field | Size | Purpose |
|-------|------|---------|
| Magic number | 4 B | Protocol validation (`CIPD`) |
| Version counter | 8 B | Increments on sender re-init; receiver detects reconnection |
| Slot count | 4 B | Number of ring buffer slots |
| Write index | 4 B | Current producer slot (atomic counter) |
| IPC mem handle × N | 128 B each | GPU memory handle per slot |
| IPC event handle × N | 64 B each | GPU sync event handle per slot |
| Shutdown flag | 1 B | Reasserted to 0 every frame; set to 1 on exit |
| Texture metadata | 20 B | Width, height, components, dtype, buffer size |
| Producer timestamp | 8 B | `perf_counter()` for latency measurement |

Total for 3 slots: **433 bytes**.

### Lazy Initialization

GPU resources (buffer allocation, IPC handle creation, shared memory setup) are not allocated when `Active` is toggled on. Initialization happens on the **first frame** after activation. This avoids startup overhead and allows resolution to be detected automatically from the live texture.

If the sender is not yet running, the receiver retries connection with **exponential backoff** (doubling the wait interval up to ~2 seconds between attempts), then keeps retrying silently.

### Automatic Re-initialization

If the upstream texture resolution or format changes, the component detects the mismatch on the next frame, tears down the existing buffers, and re-initializes with the new dimensions. This takes ~50–100 µs (one-time) and is transparent to the connected Python process.

---

## Parameters

### Active
**Type:** Toggle | **Default:** On

Master enable/disable switch for the CUDA IPC pipeline.

- **On:** The component initializes GPU resources on the first frame and processes every frame thereafter.
- **Off:** All GPU work stops immediately. `export_frame()` and `import_frame()` return without doing anything. Calling cleanup frees all GPU buffers, destroys IPC events, closes shared memory, and (in Sender mode) signals shutdown to connected consumers. The `Numslots` parameter is re-enabled for editing in Sender mode.
- **Toggling On** does not re-initialize immediately — GPU resources are re-created lazily on the next frame callback.
- Hot-swappable: can be toggled at any time without restarting TouchDesigner.

---

### Mode
**Type:** Menu | **Default:** Sender | **Options:** Sender / Receiver

Sets the direction of data flow.

- **Sender:** This component is the producer. It captures the upstream texture each frame, copies it into the GPU ring buffer, and makes it available to an external Python process (or another TD instance in Receiver mode).
- **Receiver:** This component is the consumer. It reads GPU frames produced by an external Python process (using `CUDAIPCExporter`) and imports them into a Script TOP for use in the TD network.

Switching modes triggers a full cleanup of the current state and lazy re-initialization on the next frame. In Receiver mode, the `Numslots` parameter is locked and read-only — the slot count is determined by the sender's shared memory protocol and automatically reflected in the parameter display.

---

### Ipcmemname
**Type:** String | **Default:** `cudalink_output_ipc`

The name of the OS shared memory segment used to exchange GPU handles between the sender and receiver.

Both sides **must use the exact same name**. On Windows, this maps to a named `CreateFileMapping` kernel object.

Changing this parameter while active triggers a full cleanup and reconnection:
- In Sender mode: re-initializes on the next frame export.
- In Receiver mode: immediately resets the retry counter and attempts to connect on the next frame start (without waiting through the current backoff interval).

Use different names to run multiple independent sender/receiver pairs simultaneously in the same TouchDesigner session.

---

### Numslots
**Type:** Integer Menu | **Default:** 3 | **Options:** 2 / 3 / 4

Number of ring buffer slots in the GPU pipeline.

- **Higher values** (e.g., 4) reduce the chance of producer/consumer contention when frame processing takes variable time. Each additional slot uses one full texture worth of GPU memory (`ceil(W × H × C × sizeof(dtype) / 2 MiB) × 2 MiB`).
- **Lower values** (e.g., 2) reduce GPU memory usage at the cost of slightly increased contention risk.
- **3 slots (default)** is sufficient for the vast majority of use cases.

**Lock behavior:**
- Only editable when `Mode = Sender` and `Active = Off`.
- Locked automatically when `Active` is turned On.
- In Receiver mode: always locked. The actual slot count is read from the sender's shared memory and displayed here for reference.

Changing this parameter while active is silently ignored. Changing it while inactive triggers a cleanup and lazy re-initialization on the next frame.

---

### Debug
**Type:** Toggle | **Default:** Off

Enables verbose performance logging to the TouchDesigner Textport.

- **Off:** Only critical errors and state changes are logged.
- **On:** every ~97 frames, prints an average timing breakdown:
  - `cudaMemory` — OpenGL→CUDA interop time
  - `memcpy` — D2D memcpy enqueue time
  - `record` — IPC event record time
  - `total` — full `export_frame()` wall-clock time
  - `GPU memcpy` — actual GPU elapsed time measured via CUDA timing events (only available if Debug was On at initialization)
  - `sync mode` — whether GPU-event synchronization or CPU-sync fallback is active

The first frame after initialization always prints a detailed timing diagnostic regardless of this setting.

Hot-swappable: can be toggled at runtime without affecting the pipeline. However, GPU timing events (`cudaEventElapsedTime`) are only created during initialization. If Debug is turned On after the component is already running, CPU-side timing is enabled immediately but the `GPU memcpy` metric will not appear until the next full cleanup/re-init cycle.

---

### Hide Built-In
**Type:** Toggle | **Default:** Off

Hides the built-in TouchDesigner parameter pages (Common, Extensions) from the parameter dialog, leaving only the CUDA IPC page visible.

- **Off:** All parameter pages are shown — Common, Extensions, and CUDA IPC.
- **On:** Only the CUDA IPC parameter page is shown. Built-in pages are not deleted; they are just hidden from the UI. Toggling Off restores them immediately.

Hot-swappable: takes effect instantly without restarting or reinitializing the component. The setting is also applied automatically at component load time.

Use this when distributing the component to end-users who should not need to interact with TD's built-in parameters.

---

## Quick Start

### TD → Python (Sender mode)

1. Drop `CUDAIPCLink_v0.x.x.tox` into your TD network.
2. Wire your source TOP into the component's input.
3. Set **Mode** = `Sender`.
4. Set **Ipcmemname** to a unique name, e.g. `my_pipeline`.
5. Toggle **Active** = On.
6. In Python, install `cuda-link` and connect:
   ```python
   from cuda_link import CUDAIPCImporter
   importer = CUDAIPCImporter(shm_name="my_pipeline")
   frame = importer.get_frame()          # torch.Tensor on GPU (zero-copy)
   frame_np = importer.get_frame_numpy() # numpy array (CPU copy)
   ```

### Python → TD (Receiver mode)

1. In Python, create an exporter:
   ```python
   from cuda_link import CUDAIPCExporter
   exporter = CUDAIPCExporter(shm_name="ai_output", width=1920, height=1080)
   exporter.export_frame(gpu_tensor)
   ```
2. Drop the component into TD and set **Mode** = `Receiver`.
3. Set **Ipcmemname** to the same name (`ai_output`).
4. Toggle **Active** = On. The receiver will connect automatically once the Python exporter is running.

---

## Performance Reference

| Operation | Typical Time | Notes |
|-----------|-------------|-------|
| Per-frame IPC overhead | 0.5–2 µs | GPU event record + `write_idx` update |
| First-frame initialization | 50–100 µs | One-time GPU buffer allocation + IPC handle creation |
| D2D texture copy (1080p RGBA float32) | 60–80 µs | Runs fully on GPU |
| Receiver `copyCUDAMemory` into TD (1080p) | ~3 ms | Includes CUDA→OpenGL interop inside TD |
| D2H numpy copy (1080p RGBA float32) | 400–600 µs | Only when using `get_frame_numpy()` |

**Baseline comparison:** CPU SharedMemory at 1080p RGBA float32 costs ~1.5 ms per frame — roughly **750× slower** than CUDA IPC.

---

## Troubleshooting

**Receiver stays in "waiting for sender" state**
- Confirm the sender is running and `Active` is On before starting the receiver.
- Verify `Ipcmemname` is identical on both sides (case-sensitive).
- Check the Textport for retry messages — the receiver uses exponential backoff up to ~2 seconds between attempts.

**"Stale SharedMemory" or version mismatch logged**
- The sender was restarted while the receiver is still holding old IPC handles. Toggle the receiver's `Active` Off → On to force reconnection.

**"Protocol magic mismatch" error**
- Another process is using the same `Ipcmemname` for a different purpose. Change `Ipcmemname` to a unique value.

**GPU memory not freed after deactivation**
- `cudaFree` of ring buffer slots is deferred briefly after cleanup (a 100 ms grace period) to allow the consumer to finish its current frame. This is normal behavior.

**`Numslots` is greyed out**
- In Sender mode: toggle `Active` Off first to edit slot count.
- In Receiver mode: slot count is controlled by the sender and cannot be set locally.

**Debug shows high `cudaMemory` time (>0.5 ms)**
- This is the OpenGL→CUDA interop step inside TouchDesigner's `top_op.cudaMemory()` call and is not controllable by this component. It is normal for large textures or when the GPU is under heavy load.

---

## Requirements

- **OS:** Windows 10 / 11 (CUDA IPC handle sharing is Windows-only)
- **CUDA:** 12.x (tested with 12.4)
- **GPU:** NVIDIA, CUDA compute capability 3.5 or higher
- **TouchDesigner:** 2022.x or later
- **Python (consumer side):** 3.9+, `cuda-link` package (`pip install cuda-link`)
