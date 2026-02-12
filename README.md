# cuda-link

Zero-copy GPU texture transfer between TouchDesigner and Python processes using CUDA IPC.

## Overview

This component enables **zero-copy GPU texture sharing** between TouchDesigner and Python processes using CUDA Inter-Process Communication (IPC). It eliminates CPU memory copies, achieving sub-microsecond per-frame overhead for real-time AI pipelines, video processing, and other GPU-accelerated workflows.

### Key Features

- **Zero-copy GPU transfer** - Textures stay on GPU, no CPU memory copies
- **Sub-microsecond overhead** - ~0.5-2μs per frame (vs ~50-500μs for CPU SharedMemory)
- **Ring buffer architecture** - 3-slot pipeline prevents producer/consumer blocking
- **GPU-side synchronization** - CUDA IPC events eliminate CPU polling
- **Triple output modes** - PyTorch tensors (GPU, zero-copy), CuPy arrays (GPU, zero-copy), or numpy arrays (CPU, D2H copy)
- **Production-ready** - Tested at 60 FPS for hours, handles dynamic resolution changes

### Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Per-frame overhead | < 2μs | GPU event record + write_idx update |
| Initialization | ~50-100μs | One-time IPC handle opening |
| D2D memcpy (1080p RGBA float32) | ~60-80μs | GPU texture copy |
| D2H copy (1080p RGBA float32) | ~400-600μs | Only for numpy output mode |
| Theoretical max FPS | 10,000+ | Limited only by GPU pipeline depth |

**Baseline comparison**: CPU SharedMemory requires ~1.5ms per frame for 1080p, **~750x slower** than CUDA IPC.

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

#### Install the package:

```bash
# From PyPI (when published):
pip install cuda-link[torch]   # For PyTorch tensors (recommended)
pip install cuda-link[cupy]    # For CuPy arrays
pip install cuda-link[numpy]   # For numpy arrays only
pip install cuda-link[all]     # All output modes

# For development (local install):
cd C:\path\to\CUDA_IPC
pip install -e ".[torch]"      # Editable install with PyTorch
```

#### Use in your Python script:

```python
from cuda_link import CUDAIPCImporter

# Initialize (use same name as TD's Ipcmemname parameter)
importer = CUDAIPCImporter(
    shm_name="my_texture_ipc",
    shape=(1080, 1920, 4),  # height, width, channels (RGBA)
    dtype="float32",         # "float32", "float16", or "uint8"
    debug=False
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

# Cleanup
importer.cleanup()
```

## Architecture

```
TouchDesigner Process              Python AI Process
─────────────────────              ──────────────────
CUDAIPCExporter (TD Extension)     CUDAIPCImporter (Python)
  │                                   │
  │ export_frame(top_op)              │ get_frame() / get_frame_cupy() / get_frame_numpy()
  │   ↓                                │   ↑
  │ top_op.cudaMemory()               │ torch.as_tensor() / cupy.ndarray / cuda.memcpy()
  │   ↓                                │   ↑
  │ cudaMemcpy D2D to ring buffer     │ Waits on IPC event
  │   ↓                                │   ↑
  │ cudaEventRecord (GPU signal)      │ Returns tensor/array
  │   ↓                                │
  │ Write write_idx to SharedMemory   │
  │   ↓                                │
  └─→ SharedMemory (617 bytes) ←──────┘
      [IPC handles + sync metadata]
```

### Ring Buffer (3 Slots)

The system uses a 3-slot ring buffer to allow producer and consumer to work in parallel:

- **Slot 0**: Producer writes frame N
- **Slot 1**: Producer writes frame N+1 while consumer reads frame N
- **Slot 2**: Producer writes frame N+2 while consumer reads frame N+1
- Wraps back to Slot 0 for frame N+3

This prevents blocking - producer never waits for consumer, consumer is always 1 frame behind.

### SharedMemory Protocol (617+ bytes for 3 slots)

```
[0-3]     magic "CIPC" (4B)       - Protocol validation (0x43495043)
[4-11]    version (8B)             - Increments on TD re-initialization
[12-15]   num_slots (4B)           - Number of ring buffer slots (3)
[16-19]   write_idx (4B)           - Current write index (atomic counter)

Per slot (192 bytes each):
[20+slot*192 : 148+slot*192]   cudaIpcMemHandle_t (128B) - GPU memory handle
[148+slot*192 : 212+slot*192]  cudaIpcEventHandle_t (64B) - GPU event handle

[20+NUM_SLOTS*192]  shutdown_flag (1B)  - Producer sets to 1 on exit
[21+NUM_SLOTS*192]  metadata (20B)      - width/height/num_comps/dtype/buffer_size
```

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

## Benchmarking

```bash
cd C:\path\to\CUDA_IPC\benchmarks

# Quick benchmark
python benchmark_cuda_ipc.py --frames 1000 --events

# Specific resolution
python benchmark_cuda_ipc.py --frames 1000 --resolution 1920x1080 --events

# Check Windows CUDA IPC viability
python test_cuda_ipc_windows.py

# Stress test (10 minutes at 60 FPS)
python benchmark_cuda_ipc.py --fps 60 --duration 600 --events
```

## Troubleshooting

### "SharedMemory not found"

**Cause**: Python importer started before TD exporter initialized.

**Solution**: Ensure TD's `CUDAIPCExporter` is active before starting Python process.

### "CUDA IPC overhead > 100μs"

**Cause**: Windows CUDA IPC may have high overhead on some driver/GPU combinations.

**Solution**: Run `python benchmarks/test_cuda_ipc_windows.py` to verify viability. If overhead is consistently > 100μs, consider using Phase 1 (CPU SharedMemory) instead.

### "Version mismatch" or stale frames

**Cause**: TD re-exported IPC handles (network reset, resolution change).

**Solution**: The importer automatically detects version changes and re-opens handles. No action needed.

### GPU memory leak

**Cause**: Importer not cleaned up properly.

**Solution**: Always call `importer.cleanup()` or use context manager pattern (future enhancement).

## Distribution

cuda-link uses a **dual distribution model** to support both use cases:

### For Python Consumers (StreamDiffusion, AI/ML pipelines)

**Install via pip**:
```bash
pip install cuda-link[torch]   # For PyTorch GPU tensors (recommended)
pip install cuda-link[cupy]    # For CuPy GPU arrays
pip install cuda-link[numpy]   # For numpy CPU arrays
pip install cuda-link[all]     # All output modes
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
| **TD → Python** (StreamDiffusion, AI pipelines) | `.tox` component | `pip install cuda-link[torch]` |
| **TD → TD** (two instances communicating) | `.tox` on both sides | Not needed |

Both sides communicate through the 617-byte SharedMemory protocol — zero import dependencies between TD and Python code.

---

## License

MIT License - See LICENSE file

## Credits

Extracted and refactored from the StreamDiffusionTD project.

Original CUDA IPC implementation: StreamDiffusion Performance Team (2025-2026)

Standalone component: Claude Code (2026)
