<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo_w.png">
    <source media="(prefers-color-scheme: light)" srcset="logo_b.png">
    <img alt="cuda-link" src="logo_b.png" width="160">
  </picture>
</p>

# cuda-link

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: Windows](https://img.shields.io/badge/Platform-Windows-blue.svg)](https://www.microsoft.com/windows)
[![codecov](https://codecov.io/gh/forkni/cuda-link/graph/badge.svg)](https://codecov.io/gh/forkni/cuda-link)

Zero-copy GPU texture transfer between TouchDesigner and Python processes using CUDA IPC.

## Overview

This component enables **zero-copy GPU texture sharing** between TouchDesigner and Python processes using CUDA Inter-Process Communication (IPC). It eliminates CPU memory copies for real-time AI pipelines, video processing, and other GPU-accelerated workflows.

### Key Features

- **Zero-copy GPU transfer** - Textures stay on GPU, no CPU memory copies
- **Bidirectional IPC** - TD → Python (input capture) AND Python → TD (AI output display)
- **Low-overhead IPC** - Sub-100 µs GPU-side export at up to 1080p, microsecond-scale D2H when needed (see Performance below and [docs/BENCHMARKS.md](docs/BENCHMARKS.md))
- **Ring buffer architecture** - N-slot pipeline prevents producer/consumer blocking
- **GPU-side synchronization** - CUDA IPC events eliminate CPU polling
- **Triple output modes** - PyTorch tensors (GPU, zero-copy), CuPy arrays (GPU, zero-copy), or numpy arrays (CPU, D2H copy)
- **Production-ready** - Tested at 30+ FPS for hours, handles dynamic resolution changes

### Performance

Measured on RTX 4090 / PCIe 4.0 x16 / Windows 11 / driver 596.36. All Python-side.

| Operation | p50 | Notes |
|-----------|-----|-------|
| `export_frame()` — 512×512 RGBA float32 | 24 µs | Standalone, EXPORT_SYNC=1; GPU D2D + stream_synchronize |
| `export_frame()` — 1080p RGBA uint8, async+graphs (Python Exporter default) | 13.9 µs | −31.3 µs (69%) vs blocking baseline 45.2 µs; **TD Sender default = blocking (v1.10.1+)** ≈ 45 µs; see [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |
| `export_frame()` — 1080p RGBA float32 | 106 µs | Standalone, EXPORT_SYNC=1 |
| `export_frame()` — 4K RGBA float32 | 357 µs | Standalone, EXPORT_SYNC=1 |
| `get_frame_numpy()` D2H — 512×512 float32 | 0.18 ms | Standalone, ~22 GB/s |
| `get_frame_numpy()` D2H — 1080p float32 | 1.32 ms | Standalone, ~24 GB/s PCIe 4.0 |
| `get_frame_numpy()` D2H — 4K float32 | 5.7 ms | Standalone, ~21 GB/s PCIe 4.0 |
| `get_frame()` / `get_frame_cupy()` GPU | <5 µs | Zero-copy tensor/array view, no D2H |
| IPC notification latency | ~136–286 µs | Producer publish → consumer detect (cross-process) |
| Initialization | ~50–100 µs | One-time IPC handle opening |

## Requirements

- **OS**: Windows 10/11 (CUDA IPC is Windows-only)
- **CUDA**: 12.x (tested with 12.4)
- **GPU**: NVIDIA GPU with CUDA compute capability 3.5+
- **TouchDesigner**: 2022.x or later (for producer side)
- **Python**: 3.11+ recommended (for consumer side; matches TouchDesigner's bundled interpreter). The pure-Python fallback wheel also installs on 3.9+ — see [Distribution](#distribution).

### Python Dependencies

**Required**: None (pure ctypes CUDA wrapper)

**Optional**:

- `torch>=2.0` - For zero-copy GPU tensor output (recommended for AI pipelines)
- `cupy-cuda12x>=12.0` - For zero-copy GPU array output (CuPy/JAX workflows)
- `numpy>=1.21` - For CPU array output (for OpenCV, etc.)

## Quick Start

### 1. TouchDesigner Side (Exporter)

**Option A: Use the .tox component** (recommended)

1. Drag `TOXES/CUDAIPCLink_v1.12.1.tox` into your TD network
2. Wire your source TOP to the `input` In TOP
3. Set `Ipcmemname` parameter (e.g., `"my_texture_ipc"`)
4. Enable `Active` toggle

The component displays its transfer state in the read-only **Status** custom parameter:
`"<W>x<H> <dtype> <ch>ch"` during active transfer, `"WARNING: ..."` or `"ERROR: ..."` on
faults, and `"Idle"` when inactive. A `warning_emitter` Script TOP inside the COMP also
shows a local warning badge when the component is open. See [`td_exporter/HELP_DOC.md`](td_exporter/HELP_DOC.md)
for per-parameter documentation.

**Option B: Build from source**

See [`docs/TOX_BUILD_GUIDE.md`](docs/TOX_BUILD_GUIDE.md) for step-by-step assembly.

**Option C: Library mode (cleaner .tox — fewer Text DATs)**

For a leaner `.tox` (package installed once into a Python environment TD can see, instead
of 14 mirror Text DATs), see [Distribution → For TouchDesigner Integration](#for-touchdesigner-integration).

### 2. Python Side (Importer)

#### Install the package

cuda-link ships **prebuilt wheels** — no compiler needed on your machine (see
[ADR-0013](docs/adr/0013-prebuilt-wheel-distribution.md)). Two supported targets: a
system Python 3.11 install, or a dedicated venv (e.g. StreamDiffusionTD's pinned
3.11.9). Never pip-install directly into TouchDesigner's own bundled Python.

```bash
# Guided installer (recommended) — auto-downloads the wheel matching your
# target interpreter's version from GitHub Releases:
cd C:\path\to\CUDA_IPC
python scripts\install_td_library.py --mode 2 --venv D:\path\to\your\venv
# or: python scripts\install_td_library.py   (interactive menu — pick your target)
```

> Manual wheel download, editable source installs, and PyPI: see
> [Distribution → For Python Consumers](#for-python-consumers-streamdiffusion-aiml-pipelines).

#### Use in your Python script

```python
from cuda_link import Importer, ImportSpec, ImportOutcome

importer = Importer.open(
    ImportSpec(
        shm_name="my_texture_ipc",
        shape=(1080, 1920, 4),  # height, width, channels (RGBA) — or None for auto-detect
        dtype="float32",         # "float32", "float16", "uint8" — or None for auto-detect
        timeout_ms=5000.0,       # Wait up to 5s for producer to appear (default)
    )
)

# Option 1: Get torch.Tensor (GPU, zero-copy)
result = importer.get_frame()
if result.outcome is ImportOutcome.NEW_FRAME:
    tensor = result.frame  # torch.Tensor on GPU, shape (1080, 1920, 4)
    # Use directly in AI model:
    # output = model(tensor)

# Option 2: Get numpy array (CPU, involves D2H copy)
result = importer.get_frame_numpy()
if result.outcome is ImportOutcome.NEW_FRAME:
    array = result.frame  # numpy.ndarray on CPU
    # Use in OpenCV, PIL, etc.:
    # cv2.imwrite("frame.png", array)

# Option 3: Get CuPy array (GPU, zero-copy)
result = importer.get_frame_cupy()
if result.outcome is ImportOutcome.NEW_FRAME:
    cupy_arr = result.frame  # cupy.ndarray on GPU
    # Use in CuPy/JAX workflows

# Context manager (recommended — ensures cleanup on exit)
with Importer.open(ImportSpec(shm_name="my_texture_ipc")) as importer:
    for _ in range(100):
        result = importer.get_frame()
        if result.outcome is ImportOutcome.NEW_FRAME:
            tensor = result.frame

# Explicit cleanup
importer.close()
```

> ▶ Runnable version: [examples/03_td_to_pytorch_pipeline.py](examples/03_td_to_pytorch_pipeline.py) — spawns its own demo producer, so no TouchDesigner is needed to try it.

### 3. Python → TouchDesigner (AI Output)

Send AI-generated frames **back to TD** for display:

```python
from cuda_link import Exporter, FrameSpec, GpuFrame

with Exporter.open(
    FrameSpec(
        shm_name="ai_output_ipc",  # Must match TD Receiver's Ipcmemname parameter
        height=512, width=512,
        channels=4, dtype="uint8",
        num_slots=2,               # Ring buffer slots (double-buffering)
    )
) as exporter:
    # Export each AI-generated frame (~10-20μs overhead at 512x512).
    # producer_stream arms cross-stream ordering so the D2D copy on the
    # non-blocking IPC stream is ordered after your kernel writes.
    # PyTorch: torch.cuda.current_stream().cuda_stream
    # CuPy:    cupy.cuda.get_current_stream().ptr
    stream_handle = torch.cuda.current_stream().cuda_stream
    exporter.export(GpuFrame(
        ptr=output_tensor.data_ptr(),
        size=output_tensor.nbytes,
        producer_stream=stream_handle,
    ))
```

On the TD side, set `CUDAIPCExtension` **Mode** to `Receiver` with matching `Ipcmemname`.

> ▶ Runnable version: [examples/07_python_to_td_exporter_and_benchmark.py](examples/07_python_to_td_exporter_and_benchmark.py) — includes a per-frame `export()` cost benchmark (p50/p95/p99).

## Architecture

```
Direction A: TD (Producer) → Python (Consumer)
──────────────────────────────────────────────
CUDAIPCExtension facade
  └── TDSenderEngine (thin TD adapter)   Importer
        │ cuda_memory() → GpuFrame         │ get_frame() / get_frame_numpy()
        │ delegates to Exporter            │ Waits on IPC event
        └─→ SharedMemory ←─────────────────┘

Direction B: Python (Producer) → TD (Consumer)
───────────────────────────────────────────────
Exporter                           CUDAIPCExtension facade
  │ export(GpuFrame(ptr, size))      └── TDReceiverEngine
  │ cudaMemcpy D2D → ring buf             │ import_frame(script_top)
  └─→ SharedMemory ←──────────────────────┘ copyCUDAMemory()

Both directions share the same v0.5.0 binary protocol.
```

The TD extension uses a **facade-with-delegation** pattern: `CUDAIPCExtension` (~300 LOC) holds either a `TDSenderEngine` or `TDReceiverEngine` and delegates all work to it. `TDSenderEngine` is a thin TD-only adapter (~415 LOC) over the canonical `Exporter` — it owns pixel-format bridging, the `cuda_memory()`→`GpuFrame` translation, dynamic geometry reopen, and `HolderBarrier` lifecycle; all GPU ring-buffer logic delegates to `Exporter`. Mode switches replace the engine entirely — zero cross-mode state leak. All TouchDesigner runtime access (`ownerComp.par.*`, `top.cudaMemory()`, `copyCUDAMemory()`) goes through the `TDHost`/`TOPHandle` adapter seam, making the engine logic testable without a TD runtime.

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
- **[Runnable Examples](examples/README.md)** - 8 standalone, heavily-commented scripts (`examples/`) — each spawns its own demo producer, so they run without TouchDesigner

## Testing

The suite lives in `tests/` split into five purpose-named packages:

```
tests/
  core/         protocol layer — SHM layout, format negotiation, ports, activation barriers
  cuda/         CUDA runtime seam — IPC wrapper, errcheck, handle guards, NVML, probe scripts
  td/           TouchDesigner integration — TDHost, Sender, Receiver, bootstrap, config
  integration/  end-to-end pipeline, round-trip data integrity, deprecation shims
  support/      tooling — env checks, console shutdown, wrapper-sync drift guard
```

Run the full test suite:

```bash
cd C:\path\to\CUDA_IPC

# Run a whole category (no CUDA needed)
pytest tests/core/ -v

# Single-file examples
pytest tests/core/test_shm_protocol.py -v      # protocol tests (no CUDA needed)
pytest tests/cuda/test_cuda_ipc_wrapper.py -v  # CUDA unit tests (requires GPU)

# All tests
pytest tests/ -v

# Skip slow multi-process tests
pytest tests/ -v -m "not slow"
```

### Randomized ordering

[pytest-randomly](https://github.com/pytest-dev/pytest-randomly) is active via `addopts` — every
run uses a fresh seed and tests must be order-independent. To reproduce a specific failure:

```bash
# --randomly-seed prints at the top of each run (e.g. "Using --randomly-seed=12345")
pytest tests/ -m "not requires_cuda" -p randomly --randomly-seed=12345
```

Disable for a single run: add `-p no:randomly`.

### Coverage gate

```bash
pytest tests/ -m "not requires_cuda" --cov=cuda_link --cov-report=term-missing
```

The gate is `fail_under = 76` (branch-coverage-aware, baseline 79.46% measured 2026-07-12)
in `[tool.coverage.report]` in `pyproject.toml`. For line-level inspection add
`--cov-report=html` and open `htmlcov/index.html`.

### Test doubles

Shared fakes are consolidated in `tests/fakes/` and resolved automatically via `pythonpath`
in `pyproject.toml`. Import them directly:

```python
from fakes import FakeShmAdapter, FakeTDHost, FakeTOPHandle, make_connected_importer
```

## Benchmarks

All results on RTX 4090 / PCIe 4.0 x16 / Windows 11 / driver 596.36. RGBA (4-channel) frames.

Key highlights:

- **`export_frame()` Python Exporter async+graphs** — **13.9 µs p50** at 1080p uint8 (RTX 4090 / PCIe 4.0); −27.3 µs (60%) vs blocking baseline from eliminating `cudaStreamSynchronize`; −4.0 µs (22%) more from CUDA graph collapsing 3 WDDM submissions → 1; full A→D = −31.3 µs (69%). **TD Sender default = blocking since v1.10.1** (Cell A ≈ 45 µs / Cell C ≈ 29 µs @ 1080p uint8; prevents source-buffer lifetime race → CUDA 719). Blocking baseline range: 24–357 µs p50 (512×512 → 4K float32, EXPORT_SYNC=1).
- **`get_frame_numpy()` D2H** — 0.18 ms p50 (512×512) → 5.7 ms (4K) at ~22–24 GB/s PCIe 4.0. With opt-in `CUDALINK_D2H_PIPELINED=1` and 5 ms consumer work: 1080p −6% / −309 µs, 4K −20% / −1276 µs.
- **Full IPC roundtrip** — IPC notification latency ~136–286 µs cross-process (resolution-independent signaling).
- **vs CPU SharedMemory** — ~3.4× faster E2E at 1080p, ~2.1× at 512×512. Producer write 4–19× faster (no CPU transit). With `get_frame()` / `get_frame_cupy()` (zero-copy), the consumer read collapses to <5 µs.
- **Receiver hot-path optimizations (v1.10.2)** — receiver skips idle Script-TOP cooks when no new frame is queued (P11, reduces observable cook counts); import-hot-path `_TorchBackend`/`_CupyBackend` allocations cached across calls (P8, reduces per-frame GC pressure).

Full tables, per-resolution breakdowns, and CUDA Graphs A/B comparison: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**

### Performance Tuning (env vars)

The six variables most consumers reach for:

| Variable | Default | Effect |
|---|---|---|
| `CUDALINK_EXPORT_SYNC` | `0` (Python API) / `1` (TD Sender) | Block CPU on the IPC stream after each `export_frame()`. **Python `Exporter` API default: `0` (async)** — the CUDA IPC event provides correct cross-process GPU ordering. **TD Sender default: `1` (blocking)** — the TD source is a transient cook-scoped TOP texture reclaimed right after the cook, so the D2D read must finish first (CUDA 719 guard; see CHANGELOG 1.10.1). |
| `CUDALINK_DOORBELL` | `0` | R2 Win32 named-event doorbell (opt-in). Producer signals a Win32 event after each `publish_frame()`; consumer blocks on it instead of poll-sleeping. Sub-300 µs notify latency, ~zero idle CPU. Single-consumer, Windows-only. |
| `CUDALINK_WAIT_BACKEND` | `auto` | R5 native notification-wait backend (compiled into the core wheel on Windows, [ADR-0012](docs/adr/0012-native-extension-in-core-wheel.md)). `auto` uses it when available with silent fallback; `python` forces the pre-R5 path; `native` skips the `auto` gate. Target: p50 < 10 µs vs 136–286 µs poll-sleep baseline. |
| `CUDALINK_D2H_PIPELINED` | `0` | Opt-in pipelined D2H for `get_frame_numpy()` — overlaps the copy with consumer CPU work at the cost of +1 frame latency. Gain: 1080p −6% / 4K −20% with 5 ms consumer work. |
| `CUDALINK_IMPORT_RECONNECT` | `1` | Enable automatic reconnect in the importer when the SHM segment disappears (e.g., producer restart). Set to `0` to raise immediately on a missing segment. |
| `CUDALINK_TD_USE_GRAPHS` | `0` | CUDA Graphs for the TouchDesigner-side `CUDAIPCExtension` Sender — collapses 3 WDDM kernel-mode submissions into 1. Auto-disabled on CUDA runtimes older than 11.3; falls back to the legacy stream path on any capture or launch failure. |

> Full reference (~40 variables, including TD-internal, profiling, and adaptive-barrier
> knobs): [docs/ENV_VARS.md](docs/ENV_VARS.md).

For GPU-timeline profiling (Nsight Systems / Nsight Compute / compute-sanitizer) see [docs/PROFILING.md](docs/PROFILING.md).


## Troubleshooting

### "SharedMemory not found"

**Cause**: Python importer started before TD exporter initialized.

**Solution**: Ensure the TD component is active before starting the Python process. If starting both together, use `timeout_ms` to give the producer time to initialize:

```python
from cuda_link import Importer, ImportSpec
importer = Importer.open(ImportSpec(shm_name="my_project_ipc", timeout_ms=10000.0))  # Wait up to 10s
```

### "CUDA IPC overhead unexpectedly high"

**Cause**: In standalone Python processes (WDDM), `export_frame()` with EXPORT_SYNC=1 typically measures 24–357 µs p50 (512×512 → 4K float32 RGBA, RTX 4090 / PCIe 4.0). Values 2–5× higher than these baselines may indicate GPU driver overhead, context contention, or PCIe bandwidth sharing with other D2H workloads.

**Solution**: Compare against the baseline numbers in [docs/BENCHMARKS.md](docs/BENCHMARKS.md), which documents exact reproduction commands using the tracked `scripts/profiling/` tooling (e.g. `python scripts/profiling/profile_export.py --frames 2000 --export-profile`).

### "Version mismatch" or stale frames

**Cause**: TD re-exported IPC handles (network reset, resolution change).

**Solution**: The importer automatically detects version changes and re-opens handles. No action needed.

### GPU memory leak

**Cause**: Importer not cleaned up properly.

**Solution**: Use the context manager pattern for automatic cleanup:

```python
from cuda_link import Importer, ImportSpec, ImportOutcome
with Importer.open(ImportSpec(shm_name="my_project_ipc")) as importer:
    # importer.close() is called automatically on exit
    result = importer.get_frame()
    if result.outcome is ImportOutcome.NEW_FRAME:
        tensor = result.frame
```

Or call `importer.close()` explicitly in a `finally` block.

## Distribution

cuda-link uses a **dual distribution model** to support both use cases:

### For Python Consumers (StreamDiffusion, AI/ML pipelines)

cuda-link is **Windows-only** and ships as **prebuilt wheels** — see
[ADR-0013](docs/adr/0013-prebuilt-wheel-distribution.md). Every tagged release publishes
two wheel assets:

- `cuda_link-<version>-cp311-cp311-win_amd64.whl` — Python 3.11 (covers both
  StreamDiffusionTD's pinned 3.11.9 and TouchDesigner's bundled 3.11.10), carrying the
  compiled `_native_waiter` wait-backend accelerator.
- `cuda_link-<version>-py3-none-any.whl` — every other Python version. Pure Python; the
  native accelerator's benefit is marginal (<1–5%), so this loses almost nothing.

No local compiler is required for either. Two supported install scenarios in practice: a
system Python 3.11 install, or a dedicated venv (e.g. StreamDiffusionTD's).

#### Method 1: Guided installer (recommended)

```bash
git clone https://github.com/forkni/cuda-link.git
cd cuda-link

# Auto-downloads the wheel matching your target interpreter's version from
# GitHub Releases — no --wheel or --build flag needed for the common case:
python scripts\install_td_library.py --mode 2 --venv D:\path\to\your\venv    # dedicated venv
python scripts\install_td_library.py --mode 4 --python C:\Python311\python.exe  # system Python
python scripts\install_td_library.py   # interactive menu

# Force reinstall to update:
python scripts\install_td_library.py --mode 2 --venv D:\path\to\your\venv
```

#### Method 2: Manual download from GitHub Releases

```bash
# Download the wheel matching your interpreter from
# https://github.com/forkni/cuda-link/releases, then:
pip install "cuda_link-1.12.1-cp311-cp311-win_amd64.whl[torch]"   # PyTorch GPU tensors
pip install "cuda_link-1.12.1-cp311-cp311-win_amd64.whl[cupy]"    # CuPy GPU arrays
pip install "cuda_link-1.12.1-cp311-cp311-win_amd64.whl[numpy]"   # NumPy CPU arrays
pip install "cuda_link-1.12.1-cp311-cp311-win_amd64.whl[all]"     # All output modes

# Force reinstall to update:
pip install --force-reinstall "cuda_link-1.12.1-cp311-cp311-win_amd64.whl[torch]"
```

#### Method 3: Editable install from source (for development)

```bash
git clone https://github.com/forkni/cuda-link.git
cd cuda-link
pip install -e ".[torch]"   # Changes to src/cuda_link/ apply immediately, no rebuild needed
pip install -e ".[all]"     # All output modes
```

#### Method 4: Build from source (dev-only, requires MSVC)

```bash
git clone https://github.com/forkni/cuda-link.git
cd cuda-link
utils\build_wheel.cmd            # Native wheel — requires an MSVC C++17 toolchain
utils\build_wheel.cmd nowaiter   # Fallback wheel — no compiler needed
# Then: python scripts\install_td_library.py --build --mode 2 --venv D:\path\to\your\venv
```

End users should not need this method — see Method 1 and 2 above.

#### Method 5: From PyPI (coming soon)

```bash
# pip install cuda-link[torch]
```

**Usage**:

```python
from cuda_link import Importer, ImportSpec, ImportOutcome

importer = Importer.open(ImportSpec(shm_name="my_project_ipc"))
result = importer.get_frame()
if result.outcome is ImportOutcome.NEW_FRAME:
    tensor = result.frame  # torch.Tensor, GPU zero-copy
```

The `cuda-link` package contains only the **consumer-side** Python code (`src/cuda_link/`). The TouchDesigner extension is distributed separately.

### For TouchDesigner Integration

**Option A: Use the .tox component** (recommended)

Drag `TOXES/CUDAIPCLink_v1.12.1.tox` into your TouchDesigner network.

> **Older versions:** Previous `.tox` releases are available as downloadable assets on the
> [GitHub Releases page](https://github.com/forkni/cuda-link/releases) — pick the tag
> matching the TouchDesigner build you target.

**Option B: Build from source**

Follow the manual build guide at [`docs/TOX_BUILD_GUIDE.md`](docs/TOX_BUILD_GUIDE.md) to assemble the `.tox` from `td_exporter/` source files.

**Option C: Library mode (cleaner .tox — fewer Text DATs)**

Install `cuda_link` into a Python environment TouchDesigner can see. The `CUDALinkBootstrap`
DAT then loads the package automatically — the 14 mirror Text DATs (Env, SHMProtocol,
Exporter, Importer, …) are no longer needed in the `.tox`. Run the multi-target installer
(one-time):

```bat
install_td_library.cmd             REM interactive menu — auto-downloads the matching
                                    REM prebuilt wheel; no build step needed (see ADR-0013)
```

**Install modes** (`python scripts/install_td_library.py --help`):

| Mode | Flag | Description |
|------|------|-------------|
| 1 | `--target DIR` | Install into a custom folder; set `CUDALINK_LIB_PATH=DIR` before launching TD |
| 2 | `--venv DIR` | Install into an existing venv that TD is configured to use |
| 3 | `--conda ENV` | Install into a conda environment |
| 4 | `--python EXE` | Install into a parallel Python; auto-writes TD Preferences — no env var needed |
| 5 | `--td-python EXE` | Install directly into TD's bundled Python (`app.pythonExecutable`) |

**Mode 4 (recommended for most setups):** auto-discovers both the registered system Python
(`py -3`) and the TouchDesigner install path; sets `Python64 Path` in TD Preferences so
library mode activates on the next TD launch with zero env-var configuration.

```bat
REM Non-interactive mode 4 (auto-discover Python + TD):
install_td_library.cmd --mode 4 --non-interactive

REM Dry-run to preview what would be written:
install_td_library.cmd --mode 4 --dry-run
```

The `TDHost`/`TDConfig`/`TDSender`/`TDReceiver` glue DATs remain in the COMP unchanged.
If `CUDALINK_LIB_PATH` is unset and mode 4 was not used, the bootstrap no-ops and the
classic mirror DATs take over silently. See [`docs/TOX_BUILD_GUIDE.md`](docs/TOX_BUILD_GUIDE.md)
for full instructions.

The TouchDesigner extension (`td_exporter/`) is **not included in the pip package** because it uses TD-specific APIs (`parent()`, `op()`, `me`, COMP-scoped imports) that cannot run outside TouchDesigner.

### Use Cases

| Use Case | TD Side | Python Side |
|----------|---------|-------------|
| **TD → Python** (StreamDiffusion, AI pipelines) | `.tox` Sender mode | `install_td_library.py --mode 2/4` (see above) |
| **Python → TD** (AI output display) | `.tox` Receiver mode | `install_td_library.py --mode 2/4` (see above) |
| **TD → TD** (two instances communicating) | `.tox` on both sides | Not needed |

Both sides communicate through the 433-byte SharedMemory protocol — zero import dependencies between TD and Python code.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full history.

---

## Contributing

1. Fork the repo and branch off **`development`** (`master` is release-only, promoted via
   [`.github/workflows/merge-development-to-main.yml`](.github/workflows/merge-development-to-main.yml)).
2. `pip install -e ".[all]"` for an editable install with every output backend.
3. Run the checks CI enforces before opening a PR: `pytest tests/ -v` (see
   [Testing](#testing) for the coverage gate) and `pyrefly` for static typing.
4. Use [conventional-commit](https://www.conventionalcommits.org/) messages
   (`feat:`, `fix:`, `docs:`, …).
5. Open the PR against `development`. CI runs tests, typecheck, and docs validation, and
   Claude's automated reviewer leaves feedback.

Use plain `git`/`gh` for your fork — the `scripts/git/*` wrappers in this repo are the
maintainer's local automation, not a contributor requirement.

---

## License

MIT License - See LICENSE file

## Credits

Original implementation by Forkni (forkni@gmail.com).
Extracted and refactored from the StreamDiffusionTD project.
