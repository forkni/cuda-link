# cuda-link

Zero-copy GPU texture transfer between TouchDesigner and Python processes using CUDA IPC.

## Overview

This component enables **zero-copy GPU texture sharing** between TouchDesigner and Python processes using CUDA Inter-Process Communication (IPC). It eliminates CPU memory copies, achieving low-microsecond IPC overhead (~3-8µs CPU-side) for real-time AI pipelines, video processing, and other GPU-accelerated workflows.

### Key Features

- **Zero-copy GPU transfer** - Textures stay on GPU, no CPU memory copies
- **Bidirectional IPC** - TD → Python (input capture) AND Python → TD (AI output display)
- **Low-overhead IPC** - ~3-8μs CPU-side for IPC sync primitives; ~117μs for `export_frame()` in Python (WDDM); ~10-20μs within TouchDesigner's CUDA context; TD→Python e2e: avg 0.57ms at 512x512 (measured, 60 FPS); vs ~2.6ms for CPU SharedMemory at 1080p (see [Benchmarks](#benchmarks))
- **Ring buffer architecture** - N-slot pipeline prevents producer/consumer blocking
- **GPU-side synchronization** - CUDA IPC events eliminate CPU polling
- **Triple output modes** - PyTorch tensors (GPU, zero-copy), CuPy arrays (GPU, zero-copy), or numpy arrays (CPU, D2H copy)
- **Production-ready** - Tested at 30+ FPS for hours, handles dynamic resolution changes

### Performance

| Operation | Time | Notes |
|-----------|------|-------|
| IPC sync primitives | ~3-8μs | cudaEventRecord + write_idx + cudaStreamWaitEvent (CPU-side only) |
| export_frame() in TouchDesigner | ~10-20μs | Within TD's CUDA context at 512x512 |
| export_frame() Python process | ~117-120μs | Standalone Python, WDDM kernel overhead |
| get_frame() TD→Python (consumer) | avg 30µs, p95 69µs | 512x512 float32, cudaStreamWaitEvent + tensor view (measured, 60 FPS) |
| E2E latency TD→Python | avg 0.57ms, p95 1.06ms | 512x512 float32, producer write → consumer read (measured, 300 frames @ 60 FPS) |
| D2H copy (720p uint8) | ~300-500μs | numpy output mode, PCIe bandwidth dependent |
| D2H copy (1080p uint8) | ~1-2ms | numpy output mode, PCIe bandwidth dependent |
| D2H copy (1080p float32) | ~4ms | numpy output mode, ~31.6 MB (measured) |
| Initialization | ~50-100μs | One-time IPC handle opening |
| Theoretical max FPS | 10,000+ | Limited by GPU pipeline depth, not IPC overhead |

**Baseline comparison** (measured, 1080p float32): CPU SharedMemory write averages ~2.6ms per frame. CUDA IPC `export_frame()` averages ~117µs — **~22x faster write**, **~5.8x lower E2E latency** (0.93ms vs 5.37ms). See [Benchmarks](#benchmarks) for full comparison table.

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

1. Drag `CUDAIPCExporter.tox` into your TD network
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
build_wheel.cmd                             # Builds dist\cuda_link-0.6.6-py3-none-any.whl

pip install "dist\cuda_link-0.6.6-py3-none-any.whl[torch]"   # PyTorch GPU tensors
pip install "dist\cuda_link-0.6.6-py3-none-any.whl[cupy]"    # CuPy GPU arrays
pip install "dist\cuda_link-0.6.6-py3-none-any.whl[numpy]"   # NumPy CPU arrays
pip install "dist\cuda_link-0.6.6-py3-none-any.whl[all]"     # All output modes

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

### SharedMemory Protocol (625 bytes for 3 slots)

```
[0-3]     magic "CIPC" (4B)       - Protocol validation (0x43495043)
[4-11]    version (8B)             - Increments on TD re-initialization
[12-15]   num_slots (4B)           - Number of ring buffer slots (3)
[16-19]   write_idx (4B)           - Current write index (atomic counter)

Per slot (192 bytes each):
[20+slot*192 : 148+slot*192]   cudaIpcMemHandle_t (128B) - GPU memory handle
[148+slot*192 : 212+slot*192]  cudaIpcEventHandle_t (64B) - GPU event handle

[20+NUM_SLOTS*192]        shutdown_flag (1B)   - Producer sets to 1 on exit
[21+NUM_SLOTS*192]        metadata (20B)       - width/height/num_comps/dtype/buffer_size
[41+NUM_SLOTS*192]        timestamp (8B)       - Producer perf_counter() for latency
```

For 3 slots: `20 + (3 × 192) + 1 + 20 + 8 = 625 bytes`

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

The `benchmarks/` directory contains a reproducible benchmark suite comparing CUDA IPC against CPU SharedMemory and NumPy array transfer without TouchDesigner.

### Tier 1: Pure Python (no TD required)

```bash
# Default: 512x512, 300 frames @ 60fps
python benchmarks/compare_all.py

# Specific resolution
python benchmarks/compare_all.py --resolution 1080p --frames 500

# Sweep all standard resolutions (512x512, 720p, 1080p, 4K)
python benchmarks/compare_all.py --sweep

# Save raw comparison data as JSON
python benchmarks/compare_all.py --resolution 1080p --save-json
```

### Results — Tier 1: Pure Python (no TD, 1080p float32 RGBA, RTX GPU, Windows 11, 300 frames @ 60fps)

```
Method               Write avg    Read avg     E2E Latency    Notes
-------------------  -----------  -----------  -------------  ----------------
CUDA IPC             117µs        4.15ms       0.93ms         GPU zero-copy
CPU SharedMem (Py)   2.60ms       2.48ms       5.37ms         CPU memcpy
NumPy Transfer (Py)  2.48ms       2.58ms       5.35ms         numpy+SHM
```

**Read column**: CUDA IPC `get_frame_numpy()` incurs a D2H copy (~4ms for 31.6 MB float32). Zero-copy GPU modes (`get_frame()`, `get_frame_cupy()`) have negligible read overhead.

Run individual benchmarks for per-frame CSV and full percentile stats:

```bash
python benchmarks/benchmark_roundtrip.py --resolution 1080p --frames 500 --csv results.csv
python benchmarks/benchmark_cpu_sharedmem.py --resolution 1080p --frames 500
python benchmarks/benchmark_numpy_transfer.py --resolution 1080p --frames 500 --with-copy
```

### Results — Tier 2: TD Sender → Python Receiver (512x512 float32 RGBA, RTX 4060 Laptop, Windows 11, 300 frames @ 60fps)

Measured with `benchmark_comparison.py` against a live TouchDesigner sender:

```
Metric                    avg      p50      p95      p99      min      max
-----------------------  -------  -------  -------  -------  -------  -------
E2E latency (ms)          0.57     0.58     1.06     1.20     0.019    1.422
get_frame() call (us)      30       21       69       --       12       --
```

- **E2E latency**: time from TD producer writing the frame (producer timestamp) to Python consumer returning it
- **get_frame()**: consumer-side only — `cudaStreamWaitEvent` enqueue + tensor view return; excludes periodic handle-reopen events (~13 out of 300 frames reopen a handle at ~500µs each, excluded from avg above)
- **FPS**: 60.0 sustained, 0 skipped frames
- **Zero CPU blocking**: GPU synchronization via `cudaStreamWaitEvent` enqueue (~0.5-2µs) — does not wait for GPU

### Tier 2: TD-Integrated Benchmarks (other methods)

Run `python benchmarks/benchmark_td_metrics.py` with TouchDesigner active to measure Shared Mem Out TOP, Touch Out/In, and Spout cook times and E2E latency. TD-side logger scripts: `benchmarks/td_touchout_logger.py`, `benchmarks/td_spout_logger.py`.

## Troubleshooting

### "SharedMemory not found"

**Cause**: Python importer started before TD exporter initialized.

**Solution**: Ensure TD's `CUDAIPCExporter` is active before starting Python process. If starting both together, use `timeout_ms` to give the producer time to initialize:

```python
importer = CUDAIPCImporter(shm_name="my_project_ipc", timeout_ms=10000.0)  # Wait up to 10s
```

### "CUDA IPC overhead > 500μs"

**Cause**: In standalone Python processes (WDDM), `export_frame()` typically measures ~117-120μs (async D2D enqueue + IPC sync + WDDM kernel transitions). Within TouchDesigner's CUDA context it is ~10-20μs. Values above 500μs may indicate GPU driver overhead or context contention.

**Solution**: Run `python benchmarks/compare_all.py --resolution 1080p --frames 200` to measure actual overhead and compare against CPU SharedMemory on your hardware.

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
git clone https://github.com/forkni/cuda-ipc.git
cd cuda-ipc

# Run the build script (uses PEP 517 isolated build via python -m build)
build_wheel.cmd
# Output: dist\cuda_link-0.6.6-py3-none-any.whl  (~30 KB)

# Install into any Python environment — conda, venv, system Python, TouchDesigner Python:
pip install "dist\cuda_link-0.6.6-py3-none-any.whl[torch]"   # PyTorch GPU tensors
pip install "dist\cuda_link-0.6.6-py3-none-any.whl[cupy]"    # CuPy GPU arrays
pip install "dist\cuda_link-0.6.6-py3-none-any.whl[numpy]"   # NumPy CPU arrays
pip install "dist\cuda_link-0.6.6-py3-none-any.whl[all]"     # All output modes

# Force reinstall to update:
pip install --force-reinstall "dist\cuda_link-0.6.6-py3-none-any.whl[torch]"
```

The wheel is a self-contained archive — copy it anywhere and install without needing the source tree.

#### Method 2: Editable install from source (for development)

```bash
git clone https://github.com/forkni/cuda-ipc.git
cd cuda-ipc
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

Drag `CUDAIPCExporter.tox` into your TouchDesigner network from the project root.

**Option B: Build from source**

Follow the manual build guide at [`docs/TOX_BUILD_GUIDE.md`](docs/TOX_BUILD_GUIDE.md) to assemble the `.tox` from `td_exporter/` source files.

The TouchDesigner extension (`td_exporter/`) is **not included in the pip package** because it uses TD-specific APIs (`parent()`, `op()`, `me`, COMP-scoped imports) that cannot run outside TouchDesigner.

### Use Cases

| Use Case | TD Side | Python Side |
|----------|---------|-------------|
| **TD → Python** (StreamDiffusion, AI pipelines) | `.tox` Sender mode | `pip install dist\cuda_link-*.whl[torch]` |
| **Python → TD** (AI output display) | `.tox` Receiver mode | `pip install dist\cuda_link-*.whl[torch]` |
| **TD → TD** (two instances communicating) | `.tox` on both sides | Not needed |

Both sides communicate through the 625-byte SharedMemory protocol — zero import dependencies between TD and Python code.

---

## License

MIT License - See LICENSE file

## Credits

Original implementation by Forkni (forkni@gmail.com).
Extracted and refactored from the StreamDiffusionTD project.
