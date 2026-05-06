# cuda-link

Zero-copy GPU texture transfer between TouchDesigner and Python processes using CUDA IPC.

## Overview

This component enables **zero-copy GPU texture sharing** between TouchDesigner and Python processes using CUDA Inter-Process Communication (IPC). It eliminates CPU memory copies for real-time AI pipelines, video processing, and other GPU-accelerated workflows.

### Key Features

- **Zero-copy GPU transfer** - Textures stay on GPU, no CPU memory copies
- **Bidirectional IPC** - TD → Python (input capture) AND Python → TD (AI output display)
- **Low-overhead IPC** - `export_frame()` 42–400 µs p50 (512×512 → 4K float32, EXPORT_SYNC=1); `get_frame_numpy()` D2H 0.2–5.5 ms p50 (PCIe 4.0 ~23 GB/s); IPC notification ~250 µs cross-process (see [Benchmarks](#benchmarks))
- **Ring buffer architecture** - N-slot pipeline prevents producer/consumer blocking
- **GPU-side synchronization** - CUDA IPC events eliminate CPU polling
- **Triple output modes** - PyTorch tensors (GPU, zero-copy), CuPy arrays (GPU, zero-copy), or numpy arrays (CPU, D2H copy)
- **Production-ready** - Tested at 30+ FPS for hours, handles dynamic resolution changes

### Performance

Measured on RTX 4090 / PCIe 4.0 x16 / Windows 11 / driver 596.36. All Python-side. Scripts in `benchmarks/`.

| Operation | p50 | Notes |
|-----------|-----|-------|
| `export_frame()` — 512×512 RGBA float32 | 42 µs | Standalone, EXPORT_SYNC=1; GPU D2D + stream_synchronize |
| `export_frame()` — 1080p RGBA float32 | 138 µs | Standalone, EXPORT_SYNC=1 |
| `export_frame()` — 4K RGBA float32 | 400 µs | Standalone, EXPORT_SYNC=1 |
| `get_frame_numpy()` D2H — 512×512 float32 | 0.23 ms | Standalone, ~18 GB/s |
| `get_frame_numpy()` D2H — 1080p float32 | 1.35 ms | Standalone, ~24 GB/s PCIe 4.0 |
| `get_frame_numpy()` D2H — 4K float32 | 5.5 ms | Standalone, ~23 GB/s PCIe 4.0 |
| `get_frame()` / `get_frame_cupy()` GPU | <5 µs | Zero-copy tensor/array view, no D2H |
| IPC notification latency | ~250–300 µs | Producer publish → consumer detect (cross-process) |
| Initialization | ~50–100 µs | One-time IPC handle opening |

## Requirements

- **OS**: Windows 10/11 (CUDA IPC is Windows-only)
- **CUDA**: 12.x (tested with 12.4)
- **GPU**: NVIDIA GPU with CUDA compute capability 3.5+
- **TouchDesigner**: 2022.x or later (for producer side)
- **Python**: 3.9+ (for consumer side)

### Python Dependencies

**Required**: None (pure ctypes CUDA wrapper)

**Optional**:

- `torch>=2.0` - For zero-copy GPU tensor output (recommended for AI pipelines)
- `cupy-cuda12x>=12.0` - For zero-copy GPU array output (CuPy/JAX workflows)
- `numpy>=1.21` - For CPU array output (for OpenCV, etc.)

## Quick Start

### 1. TouchDesigner Side (Exporter)

**Option A: Use the .tox component** (recommended)

1. Drag `CUDAIPCLink_v1.0.1.tox` into your TD network
2. Wire your source TOP to the `input` In TOP
3. Set `Ipcmemname` parameter (e.g., `"my_texture_ipc"`)
4. Enable `Active` toggle

**Option B: Build from source**

See [`docs/TOX_BUILD_GUIDE.md`](docs/TOX_BUILD_GUIDE.md) for step-by-step assembly.

### 2. Python Side (Importer)

#### Install the package

```bash
# Option A: Build wheel and install (recommended — portable, no source needed):
cd C:\path\to\CUDA_IPC
build_wheel.cmd                             # Builds dist\cuda_link-0.7.3-py3-none-any.whl

pip install "dist\cuda_link-0.7.3-py3-none-any.whl[torch]"   # PyTorch GPU tensors
pip install "dist\cuda_link-0.7.3-py3-none-any.whl[cupy]"    # CuPy GPU arrays
pip install "dist\cuda_link-0.7.3-py3-none-any.whl[numpy]"   # NumPy CPU arrays
pip install "dist\cuda_link-0.7.3-py3-none-any.whl[all]"     # All output modes

# Option B: Editable install from source (for development — changes apply immediately):
pip install -e ".[torch]"
pip install -e ".[all]"

# From PyPI (coming soon):
# pip install cuda-link[torch]
```

#### Use in your Python script

```python
from cuda_link import CUDAIPCImporter

# Initialize (use same name as TD's Ipcmemname parameter)
importer = CUDAIPCImporter(
    shm_name="my_texture_ipc",
    shape=(1080, 1920, 4),  # height, width, channels (RGBA) — or None for auto-detect
    dtype="float32",         # "float32", "float16", or "uint8" — or None for auto-detect
    debug=False,
    timeout_ms=5000.0,       # Wait up to 5s for producer to appear (default)
)

# Option 1: Get torch.Tensor (GPU, zero-copy)
if importer.is_ready():
    tensor = importer.get_frame()  # torch.Tensor on GPU, shape (1080, 1920, 4)
    # Use directly in AI model:
    # output = model(tensor)

# Option 2: Get numpy array (CPU, involves D2H copy)
if importer.is_ready():
    array = importer.get_frame_numpy()  # numpy.ndarray on CPU
    # Use in OpenCV, PIL, etc.:
    # cv2.imwrite("frame.png", array)

# Option 3: Get CuPy array (GPU, zero-copy)
if importer.is_ready():
    cupy_arr = importer.get_frame_cupy()  # cupy.ndarray on GPU
    # Use in CuPy/JAX workflows

# Context manager (recommended — ensures cleanup on exit)
with CUDAIPCImporter(shm_name="my_texture_ipc") as importer:
    for _ in range(100):
        tensor = importer.get_frame()

# Manual cleanup
importer.cleanup()
```

### 3. Python → TouchDesigner (AI Output)

Send AI-generated frames **back to TD** for display:

```python
from cuda_link import CUDAIPCExporter

exporter = CUDAIPCExporter(
    shm_name="ai_output_ipc",   # Must match TD Receiver's Ipcmemname parameter
    height=512, width=512,
    channels=4, dtype="uint8",
    num_slots=2,                 # Ring buffer slots (double-buffering)
)
exporter.initialize()

# Export each AI-generated frame (~10-20μs overhead at 512x512)
exporter.export_frame(
    gpu_ptr=output_tensor.data_ptr(),
    size=output_tensor.nbytes,
)

exporter.cleanup()
```

On the TD side, set `CUDAIPCExtension` **Mode** to `Receiver` with matching `Ipcmemname`.

## Architecture

```
Direction A: TD (Producer) → Python (Consumer)
──────────────────────────────────────────────
CUDAIPCExtension (Sender)          CUDAIPCImporter
  │ export_frame(top_op)             │ get_frame() / get_frame_numpy()
  │ cudaMemcpy D2D → ring buffer    │ Waits on IPC event
  └─→ SharedMemory ←─────────────────┘

Direction B: Python (Producer) → TD (Consumer)
───────────────────────────────────────────────
CUDAIPCExporter                    CUDAIPCExtension (Receiver)
  │ export_frame(gpu_ptr, size)     │ import_frame(script_top)
  │ cudaMemcpy D2D → ring buffer    │ copyCUDAMemory()
  └─→ SharedMemory ←─────────────────┘

Both directions share the same v0.5.0 binary protocol.
```

### Ring Buffer (3 Slots)

The system uses a 3-slot ring buffer to allow producer and consumer to work in parallel:

- **Slot 0**: Producer writes frame N
- **Slot 1**: Producer writes frame N+1 while consumer reads frame N
- **Slot 2**: Producer writes frame N+2 while consumer reads frame N+1
- Wraps back to Slot 0 for frame N+3

This prevents blocking - producer never waits for consumer, consumer is always 1 frame behind.

### SharedMemory Protocol (433 bytes for 3 slots)

```
[0-3]     magic "CIPD" (4B)       - Protocol validation (0x43495044)
[4-11]    version (8B)             - Increments on TD re-initialization
[12-15]   num_slots (4B)           - Number of ring buffer slots (3)
[16-19]   write_idx (4B)           - Current write index (atomic counter)

Per slot (128 bytes each):
[20+slot*128 : 84+slot*128]   cudaIpcMemHandle_t (64B)  - GPU memory handle
[84+slot*128 : 148+slot*128]  cudaIpcEventHandle_t (64B) - GPU event handle

[20+NUM_SLOTS*128]        shutdown_flag (1B)   - Reasserted to 0 every frame; set to 1 on exit
[21+NUM_SLOTS*128]        metadata (20B)       - width/height/num_comps/dtype/buffer_size
[41+NUM_SLOTS*128]        timestamp (8B)       - Producer perf_counter() for latency
```

For 3 slots: `20 + (3 × 128) + 1 + 20 + 8 = 433 bytes`

## Documentation

- **[TOX Build Guide](docs/TOX_BUILD_GUIDE.md)** - Step-by-step .tox assembly in TouchDesigner
- **[Architecture](docs/ARCHITECTURE.md)** - Protocol spec, ring buffer design, GPU sync
- **[Integration Examples](docs/INTEGRATION_EXAMPLES.md)** - TD→PyTorch, TD→OpenCV, multi-stream

## Testing

Run the full test suite:

```bash
cd C:\path\to\CUDA_IPC

# Protocol tests (no CUDA needed)
pytest tests/test_shm_protocol.py -v

# Unit tests (requires CUDA)
pytest tests/test_cuda_ipc_wrapper.py -v

# All tests
pytest tests/ -v

# Skip slow multi-process tests
pytest tests/ -v -m "not slow"
```

## Benchmarks

All results on RTX 4090 / PCIe 4.0 x16 / Windows 11 / driver 596.36. RGBA (4-channel) frames.

### export_frame() — CUDA Graphs A/B (`bench_graphs.py`)

Single-process, EXPORT_SYNC=1 (CPU waits for GPU D2D completion), 2000 frames:

```
Resolution    Graphs off (p50 µs)   Graphs on (p50 µs)
----------    -------------------   ------------------
512x512                      41.7                 45.3
1280x720                     59.0                 59.6
1920x1080                   138.3                133.3
3840x2160                   400.2                404.0
```

With EXPORT_SYNC=1 the GPU D2D copy dominates; CUDA Graphs saves WDDM submission transitions but the net wall-clock difference is small (<5%). The Graphs path stays on by default for consistency with async workflows.

```bash
python benchmarks/bench_graphs.py --frames 2000 --sizes 512 1280 1920 3840
```

### get_frame_numpy() D2H — stream count (`bench_d2h_streams.py`)

Standalone D2H copy, no IPC overhead, 2000 frames:

```
Resolution    1 stream p50 (ms)   2 streams p50 (ms)   1 stream GB/s
----------    -----------------   ------------------   -------------
512x512                    0.23                 0.22            17.7
1280x720                   0.62                 0.62            22.6
1920x1080                  1.35                 1.35            23.7
3840x2160                  5.54                 5.55            23.1
```

PCIe 4.0 saturates at ~23–24 GB/s. Single stream is sufficient; `CUDALINK_D2H_STREAMS=1` (default) is optimal for this platform.

```bash
python benchmarks/bench_d2h_streams.py --frames 2000 --streams 1 2 --sizes 512 1280 1920 3840
```

### Full IPC roundtrip sweep (`bench_sweep.py`)

Two separate Python processes (producer + consumer), 500 warmup + 2000 measurement frames at 60 FPS. `export p50` and `get_numpy p50` are inflated vs standalone because both processes share PCIe bandwidth concurrently. `IPC notify p50` measures producer-publish → consumer-detects-write_idx (signaling latency, resolution-independent).

```
Resolution    dtype     Graphs   export p50 (µs)   get_numpy p50 (ms)   IPC notify p50 (µs)
----------    -------   ------   ---------------   ------------------   -------------------
512x512       float32   off               1316                 1.44                     259
512x512       float32   on                1076                 1.45                     257
512x512       uint8     off                970                 0.45                     255
1280x720      float32   off               1743                 4.60                     297
1920x1080     float32   off                585                 2.00                     240
1920x1080     uint8     off               1234                 2.65                     265
3840x2160     float32   off                566                 5.85                     300
3840x2160     uint8     off                566                 2.59                     227
```

Full results (all 16 cells, CSV + JSON): `benchmarks/results/sweep_latest.csv` / `sweep_latest.json`.

```bash
python benchmarks/bench_sweep.py          # full 16-cell sweep (~12 min)
python benchmarks/bench_sweep.py --quick  # smoke test, 1 cell (~1 min)
```

### vs CPU SharedMemory

End-to-end at typical resolutions (float32 RGBA), CUDA-Link vs UT_SharedMem-class CPU SharedMemory baseline (PCIe 4.0):

```
Resolution    Method              Producer write   Consumer read   E2E
----------    ----------------    --------------   -------------   ---------
1920x1080     CPU SharedMemory          2.60 ms         2.48 ms     5.37 ms
1920x1080     CUDA-Link                  138 µs         1.35 ms     ~1.6 ms      (~3.4x faster E2E)
512x512       CPU SharedMemory           361 µs          350 µs     1.02 ms
512x512       CUDA-Link                   42 µs         0.23 ms    ~0.49 ms      (~2.1x faster E2E)
```

Producer write is 4–19x faster (no CPU transit). With zero-copy GPU consumers (`get_frame()` / `get_frame_cupy()`), the read path collapses to <5 µs and the end-to-end gap widens further. **TouchOUT and Spout** baselines were never measured — see methodology notes in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#comparison-cuda-ipc-vs-cpu-sharedmemory) for the full hardware caveats and source data.

### Performance Tuning (env vars)

| Variable | Default | Effect |
|---|---|---|
| `CUDALINK_USE_GRAPHS` | `1` | CUDA Graphs for `export_frame()` (Python-side `CUDAIPCExporter`). Collapses the `stream_wait_event + memcpy_async + record_event` triplet into a single `cudaGraphLaunch`, cutting WDDM kernel-mode transitions from 3 → 2 per frame. With EXPORT_SYNC=1 (default) the GPU D2D copy dominates wall-clock time and the net savings are small (<5% at 1080p on PCIe 4.0); see `bench_graphs.py` for measured A/B. Set to `0` to revert to the legacy stream path (e.g., if a driver version rejects graph capture). |
| `CUDALINK_TD_USE_GRAPHS` | `0` | CUDA Graphs for the TouchDesigner-side `CUDAIPCExtension` Sender. Same mechanism as `CUDALINK_USE_GRAPHS`, gated independently because TD ships `cudart64_110.dll` and the per-frame `cudaGraphExecMemcpyNodeSetParams1D` API requires CUDA 11.3+. Auto-disabled on older runtimes (probed via `cudaRuntimeGetVersion` at `initialize()`). Off by default pending TD-side soak; flip to `1` to enable. |
| `CUDALINK_D2H_STREAMS` | `1` | Number of parallel streams for `get_frame_numpy()` D2H copy. Values `2`/`4` may help on PCIe 3.0 systems or GPUs with dual DMA engines; on PCIe 4.0 a single stream already saturates the bus (~23–24 GB/s). Check `nvidia-smi -q \| findstr "Async Engines"` before tuning. |
| `CUDALINK_EXPORT_SYNC` | `1` | Block CPU on the IPC stream after each `export_frame()`. Default on — load-bearing for concurrent topologies (prevents cycle-2 first-settle TDR cascade when a TD Sender shares a process with a TD Receiver, Phase 3.6 confirmed). Cost equals GPU D2D copy time: ~42 µs (512×512) → ~400 µs (4K) float32 RGBA on PCIe 4.0 (see `bench_graphs.py`). Set to `0` to opt out for low-latency single-producer async scenarios. |
| `CUDALINK_ACTIVATION_BARRIER` | `1` | Python-lib side of the cross-process activation barrier (F9). Reads a tiny SHM counter each `export_frame()` and skips publishing while a TD Sender is in its WDDM-saturating init window. No-op in single-pair topologies (counter stays at 0); gracefully skipped if the SHM segment is absent. Set to `0` to opt out. |
| `CUDALINK_TD_ACTIVATION_BARRIER` | `1` | TD-side counterpart of `CUDALINK_ACTIVATION_BARRIER` — increments the same SHM counter around Sender `initialize()` so the Python producer backs off. Same no-op / graceful-absence behaviour. Set to `0` to opt out. |
| `CUDALINK_TD_PERSIST_STREAM` | `1` | Skip `stream_destroy` in TD Sender `cleanup()` so the IPC CUDA stream survives `deactivate`→`reactivate` cycles (F8). Free in single-pair (no deactivation ever happens); load-bearing in concurrent — without it, stream recreate on each reactivation collides with in-flight Receiver work, doubling first-settle `post=` latency (Phase 3.6 confirmed). Set to `0` to opt out. |
| `CUDALINK_TD_STREAM_PRIO` | `normal` | CUDA stream priority for the TD Sender's IPC stream. Default `normal` is safe for both single-pair and concurrent topologies — in single-pair only one stream exists per process so priority is moot; in concurrent, equal priorities prevent WDDM contention accumulation across reactivation cycles (high/high contention produces non-recovering cycle-3 shutdowns, Phase 3.6 Step C confirmed). Set to `high` only for explicit single-pair lowest-latency optimisation. |
| `CUDALINK_EXPORT_FLUSH_PROBE` | `1` | Insert a non-blocking `cudaStreamQuery(ipc_stream)` after `check_sticky_error` when `EXPORT_SYNC=0`. Forces WDDM-deferred CUDA submissions to drain each frame, preventing Windows Task Manager's 3D-engine counter from inflating when true compute load (per NVML) is low. NVML readings are unchanged — purely cosmetic/observability. Set to `0` to disable. |
| `CUDALINK_EXPORT_PROFILE` | `0` | Enable fine-grained per-region sub-timers in `export_frame()` and emit a `[PROFILE] pre=…us interop=…us post=…us memcpy=…us record=…us sync=…us sticky=…us flush_probe=…us shm=…us unacc=…us` line every 97 frames. Force-enables `verbose_performance` (TD) / `debug` (lib). Diagnostic-only; negligible overhead when on, zero when unset. |


## Troubleshooting

### "SharedMemory not found"

**Cause**: Python importer started before TD exporter initialized.

**Solution**: Ensure TD's `CUDAIPCExporter` is active before starting Python process. If starting both together, use `timeout_ms` to give the producer time to initialize:

```python
importer = CUDAIPCImporter(shm_name="my_project_ipc", timeout_ms=10000.0)  # Wait up to 10s
```

### "CUDA IPC overhead unexpectedly high"

**Cause**: In standalone Python processes (WDDM), `export_frame()` with EXPORT_SYNC=1 typically measures 42–400 µs p50 (512×512 → 4K float32 RGBA, RTX 4090 / PCIe 4.0). Values 2–5× higher than these baselines may indicate GPU driver overhead, context contention, or PCIe bandwidth sharing with other D2H workloads.

**Solution**: Run `python benchmarks/bench_graphs.py` for standalone export latency and `python benchmarks/bench_sweep.py --quick` for a multiprocess IPC roundtrip baseline on your hardware.

### "Version mismatch" or stale frames

**Cause**: TD re-exported IPC handles (network reset, resolution change).

**Solution**: The importer automatically detects version changes and re-opens handles. No action needed.

### GPU memory leak

**Cause**: Importer not cleaned up properly.

**Solution**: Use the context manager pattern for automatic cleanup:

```python
with CUDAIPCImporter(shm_name="my_project_ipc") as importer:
    # importer.cleanup() is called automatically on exit
    tensor = importer.get_frame()
```

Or call `importer.cleanup()` explicitly in a `finally` block.

## Distribution

cuda-link uses a **dual distribution model** to support both use cases:

### For Python Consumers (StreamDiffusion, AI/ML pipelines)

#### Method 1: Build wheel (recommended — portable, installs into any environment)

```bash
git clone https://github.com/forkni/cuda-link.git
cd cuda-link

# Run the build script (uses PEP 517 isolated build via python -m build)
build_wheel.cmd
# Output: dist\cuda_link-0.7.3-py3-none-any.whl  (~30 KB)

# Install into any Python environment — conda, venv, system Python, TouchDesigner Python:
pip install "dist\cuda_link-0.7.3-py3-none-any.whl[torch]"   # PyTorch GPU tensors
pip install "dist\cuda_link-0.7.3-py3-none-any.whl[cupy]"    # CuPy GPU arrays
pip install "dist\cuda_link-0.7.3-py3-none-any.whl[numpy]"   # NumPy CPU arrays
pip install "dist\cuda_link-0.7.3-py3-none-any.whl[all]"     # All output modes

# Force reinstall to update:
pip install --force-reinstall "dist\cuda_link-0.7.3-py3-none-any.whl[torch]"
```

The wheel is a self-contained archive — copy it anywhere and install without needing the source tree.

#### Method 2: Editable install from source (for development)

```bash
git clone https://github.com/forkni/cuda-link.git
cd cuda-link
pip install -e ".[torch]"   # Changes to src/cuda_link/ apply immediately, no rebuild needed
pip install -e ".[all]"     # All output modes
```

#### Method 3: From PyPI (coming soon)

```bash
# pip install cuda-link[torch]
```

**Usage**:

```python
from cuda_link import CUDAIPCImporter

importer = CUDAIPCImporter(shm_name="my_project_ipc")
tensor = importer.get_frame()  # torch.Tensor, GPU zero-copy
```

The `cuda-link` package contains only the **consumer-side** Python code (`src/cuda_link/`). The TouchDesigner extension is distributed separately.

### For TouchDesigner Integration

**Option A: Use the .tox component** (recommended)

Drag `CUDAIPCLink_v1.0.1.tox` into your TouchDesigner network from the project root.

> **Older versions:** Previous `.tox` releases are available as downloadable assets on the
> [GitHub Releases page](https://github.com/forkni/cuda-link/releases) — pick the tag
> matching the TouchDesigner build you target.

**Option B: Build from source**

Follow the manual build guide at [`docs/TOX_BUILD_GUIDE.md`](docs/TOX_BUILD_GUIDE.md) to assemble the `.tox` from `td_exporter/` source files.

The TouchDesigner extension (`td_exporter/`) is **not included in the pip package** because it uses TD-specific APIs (`parent()`, `op()`, `me`, COMP-scoped imports) that cannot run outside TouchDesigner.

### Use Cases

| Use Case | TD Side | Python Side |
|----------|---------|-------------|
| **TD → Python** (StreamDiffusion, AI pipelines) | `.tox` Sender mode | `pip install dist\cuda_link-*.whl[torch]` |
| **Python → TD** (AI output display) | `.tox` Receiver mode | `pip install dist\cuda_link-*.whl[torch]` |
| **TD → TD** (two instances communicating) | `.tox` on both sides | Not needed |

Both sides communicate through the 433-byte SharedMemory protocol — zero import dependencies between TD and Python code.

---

## Changelog

### v0.7.3
Maintainability & performance: merged fast/debug method pairs into single methods with local debug flag, cached operator lookups with lazy fallback, lazy CuPy import, pre-cached f16 views per slot, export size validation, named `CUDAError.NOT_READY` constant.

### v0.7.2
Performance optimizations: cached SHM offsets, pre-compiled struct objects, debug-path elimination, TD hot-path cleanup.

### v0.7.1
Initial public release with dual Sender/Receiver mode, ring buffer architecture, and triple output modes (PyTorch, CuPy, NumPy).

---

## License

MIT License - See LICENSE file

## Credits

Original implementation by Forkni (forkni@gmail.com).
Extracted and refactored from the StreamDiffusionTD project.
