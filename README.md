<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo_w.png">
    <source media="(prefers-color-scheme: light)" srcset="logo_b.png">
    <img alt="cuda-link" src="logo_b.png" width="160">
  </picture>
</p>

# cuda-link

[![codecov](https://codecov.io/gh/forkni/cuda-link/branch/development/graph/badge.svg)](https://codecov.io/gh/forkni/cuda-link)

Zero-copy GPU texture transfer between TouchDesigner and Python processes using CUDA IPC.

## Overview

This component enables **zero-copy GPU texture sharing** between TouchDesigner and Python processes using CUDA Inter-Process Communication (IPC). It eliminates CPU memory copies for real-time AI pipelines, video processing, and other GPU-accelerated workflows.

### Key Features

- **Zero-copy GPU transfer** - Textures stay on GPU, no CPU memory copies
- **Bidirectional IPC** - TD → Python (input capture) AND Python → TD (AI output display)
- **Low-overhead IPC** - `export_frame()` 13.9–357 µs p50 (512×512 → 4K float32, Python Exporter async+graphs); **TD Sender defaults to blocking** (45–357 µs, v1.10.1+); `get_frame_numpy()` D2H 0.18–5.7 ms p50 (PCIe 4.0 ~22–24 GB/s); IPC notification ~136–286 µs cross-process (see [docs/BENCHMARKS.md](docs/BENCHMARKS.md))
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

1. Drag `TOXES/CUDAIPCLink_v1.12.0.tox` into your TD network
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

Install `cuda_link` into a Python environment TouchDesigner can see. The `CUDALinkBootstrap`
DAT then loads the package automatically — the 14 mirror Text DATs (Env, SHMProtocol,
Exporter, Importer, …) are no longer needed in the `.tox`. Run the multi-target installer
(one-time):

```bat
utils\build_wheel.cmd              REM build dist\cuda_link-1.12.0-py3-none-any.whl
install_td_library.cmd             REM interactive menu — choose one of 5 install modes
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

### 2. Python Side (Importer)

#### Install the package

```bash
# Option A: Build wheel and install (recommended — portable, no source needed):
cd C:\path\to\CUDA_IPC
utils\build_wheel.cmd                       # Builds dist\cuda_link-1.12.0-py3-none-any.whl

pip install "dist\cuda_link-1.12.0-py3-none-any.whl[torch]"   # PyTorch GPU tensors
pip install "dist\cuda_link-1.12.0-py3-none-any.whl[cupy]"    # CuPy GPU arrays
pip install "dist\cuda_link-1.12.0-py3-none-any.whl[numpy]"   # NumPy CPU arrays
pip install "dist\cuda_link-1.12.0-py3-none-any.whl[all]"     # All output modes

# Option B: Editable install from source (for development — changes apply immediately):
pip install -e ".[torch]"
pip install -e ".[all]"

# From PyPI (coming soon):
# pip install cuda-link[torch]
```

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

The gate is `fail_under = 72` (branch-coverage-aware, baseline 74.65% measured 2026-06-29)
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

| Variable | Default | Effect |
|---|---|---|
| `CUDALINK_REQUIRE_SOURCE_SYNC` | `0` | Raise `ValueError` in `export()` when no producer-stream ordering has been armed (no `producer_stream` in `GpuFrame` and `record_source_sync()` not called). Default `0` emits a `logger.warning` once per exporter instance instead. Set to `1` to enforce ordering at call sites (recommended for new integrations). See *Producer-stream ordering* below. |
| `CUDALINK_STRICT_DEVICE` | `0` | Raise `ValueError` in `export()` / `record_source_sync()` when the source pointer or stream belongs to a different CUDA device than the exporter. Default `0` logs an error but continues. |
| `CUDALINK_LIB_STREAM_PRIO` | `high` | CUDA stream priority for the Python-lib IPC stream. Default `high` minimises GPU scheduling latency for the D2D copy. Set to `normal` to disable high-priority scheduling (e.g., to avoid priority inversion with other high-priority streams). |
| `CUDALINK_BARRIER_STALE_NS` | `5000000000` | Staleness threshold in nanoseconds for the activation barrier's cross-process SHM counter. Frames are skipped while a Sender is settling; a counter older than this value is treated as stale and ignored. Default is 5 seconds. |
| `CUDALINK_TORCH_GPU_WAIT` | `0` | GPU-side wait for `get_frame()` (torch backend, R1 opt-in). When `1`, issues `cudaStreamWaitEvent` on `torch.cuda.current_stream()` instead of CPU-polling; the tensor is valid in stream order, not at return. `ImportOutcome.TIMEOUT` is unreachable on this path — a hung producer stalls the stream. Default `0` preserves the existing CPU-spin/sleep behaviour and timeout detection. |
| `CUDALINK_TORCH_GPU_WAIT_ADAPTIVE` | `0` | Auto-promote to GPU-side wait (torch backend) once real sleep-blocking is detected at runtime (R1-adaptive). Monitors the cpu-spin/sleep ratio over a sliding window; latches into `cudaStreamWaitEvent` mode for the rest of the session when the sleep fraction exceeds `CUDALINK_GPU_WAIT_ADAPTIVE_SLEEP_PCT` within a window of `CUDALINK_GPU_WAIT_ADAPTIVE_WINDOW` frames. One-way — never reverts. Carries the same `ImportOutcome.TIMEOUT`-unreachable consequence as `CUDALINK_TORCH_GPU_WAIT`. Effective at 30 fps (high sleep rate); stays in cpu-spin at 60 fps (zero sleep). |
| `CUDALINK_GPU_WAIT_ADAPTIVE_WINDOW` | `120` | Frame window size for the adaptive gpu-wait detector (frames counted before a tumbling reset if threshold not reached). Only effective when `CUDALINK_TORCH_GPU_WAIT_ADAPTIVE=1`. |
| `CUDALINK_GPU_WAIT_ADAPTIVE_SLEEP_PCT` | `5` | Sleep-fraction threshold (integer percent of window) that triggers the adaptive gpu-wait latch. A value of `5` means ≥5 % of frames in the window must have actually slept (fallen through the spin budget) before the latch engages. Only effective when `CUDALINK_TORCH_GPU_WAIT_ADAPTIVE=1`. |
| `CUDALINK_WAIT_SPIN_US` | `200` | Spin-wait window in microseconds for the importer's slot-available poll before yielding. Increase on systems with high OS scheduling jitter to reduce wake-up latency; decrease to reduce CPU burn on low-latency pipelines. |
| `CUDALINK_D2H_STREAM_PRIO` | `normal` | CUDA stream priority for the importer's D2H copy stream. Default `normal`; set to `high` to prioritise the consumer's D2H transfer over background GPU work. |
| `CUDALINK_ALLOW_PAGEABLE_FALLBACK` | `0` | Allow fallback to pageable (non-pinned) host memory when `cudaHostAlloc` fails. Default `0` raises instead. Useful on systems where pinned memory is exhausted by other processes. |
| `CUDALINK_IMPORT_RECONNECT` | `1` | Enable automatic reconnect in the importer when the SHM segment disappears (e.g., producer restart). Set to `0` to raise immediately on a missing segment. |
| `CUDALINK_IMPORT_RECONNECT_MAX_ATTEMPTS` | `20` | Maximum reconnect attempts before the importer raises. Only effective when `CUDALINK_IMPORT_RECONNECT=1`. |
| `CUDALINK_STICKY_ERROR_CHECK` | `1` | Check for sticky (latched) CUDA errors via `cudaPeekAtLastError` after each export frame. Default `1`; set to `0` to skip the check (saves one CUDA API call per frame on fault-free paths). |
| `CUDALINK_USE_GRAPHS` | `1` | CUDA Graphs for `export()` (Python-side `Exporter`). Collapses the `stream_wait_event + memcpy_async + record_event` triplet into a single `cudaGraphLaunch`, cutting WDDM kernel-mode transitions from 3 → 1 per frame. With EXPORT_SYNC=0 (async, Python Exporter default) the gain is −4.0 µs p50 (22%) at 1080p; with EXPORT_SYNC=1 (blocking, TD Sender default) the GPU D2D copy dominates and savings are small (<5%). See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the full 2×2 matrix. Set to `0` to revert to the legacy stream path (e.g., if a driver version rejects graph capture). |
| `CUDALINK_TD_USE_GRAPHS` | `0` | CUDA Graphs for the TouchDesigner-side `CUDAIPCExtension` Sender. Same mechanism as `CUDALINK_USE_GRAPHS`, gated independently because TD ships `cudart64_110.dll` and the per-frame `cudaGraphExecMemcpyNodeSetParams1D` API requires CUDA 11.3+. Auto-disabled on older runtimes (probed via `cudaRuntimeGetVersion` at `initialize()`). Disabled by default. Set to `1` to opt in; falls back to the legacy `cudaMemcpyAsync` stream path automatically on any capture or launch failure. |
| `CUDALINK_D2H_STREAMS` | `1` | Number of parallel streams for `get_frame_numpy()` D2H copy. Values `2`/`4` may help on PCIe 3.0 systems or GPUs with dual DMA engines; on PCIe 4.0 a single stream already saturates the bus (~23–24 GB/s). Check `nvidia-smi -q \| findstr "Async Engines"` before tuning. |
| `CUDALINK_D2H_PIPELINED` | `0` | Opt-in pipelined D2H for `get_frame_numpy()`. Enqueues the current slot's D2H copy and returns the previous frame, adding +1 frame latency in exchange for overlapping copy with consumer CPU work. First call returns `NO_FRAME`. On reconnect (v1.10.3), the pipeline drains and re-primes — one additional `NO_FRAME` per reconnect event. Gain measured at 1080p −6% (−309 µs) / 4K −20% (−1276 µs) with 5 ms consumer work. Only beneficial when consumer loop time > D2H copy time (~0.38 ms at 1080p, ~1.32 ms at 4K). Default `0` (opt-in). |
| `CUDALINK_EXPORT_SYNC` | `0` (Python API) / `1` (TD Sender) | Block CPU on the IPC stream after each `export_frame()`. **Python `Exporter` API default: `0` (async)** — the CUDA IPC event provides correct cross-process GPU ordering; coexistence safety relies on explicit per-engine streams + producer-stream ordering. **TD Sender default: `1` (blocking)** — the TD source is TD's transient cook-scoped TOP texture (`cm.ptr`) reclaimed immediately after the cook, so the D2D read must complete before that happens (CUDA 719 source-lifetime guard; see CHANGELOG 1.10.1). Set `CUDALINK_EXPORT_SYNC=0` in the TD Sender only when the source buffer is guaranteed to outlive the async copy. |
| `CUDALINK_ACTIVATION_BARRIER` | `1` | Python-lib side of the cross-process activation barrier (F9). Reads a tiny SHM counter each `export_frame()` and skips publishing while a TD Sender is in its WDDM-saturating init window. No-op in single-pair topologies (counter stays at 0); gracefully skipped if the SHM segment is absent. Set to `0` to opt out. |
| `CUDALINK_TD_ACTIVATION_BARRIER` | `1` | TD-side counterpart of `CUDALINK_ACTIVATION_BARRIER` — increments the same SHM counter around Sender `initialize()` so the Python producer backs off. Same no-op / graceful-absence behaviour. Set to `0` to opt out. |
| `CUDALINK_DOORBELL` | `0` | R2 Win32 named-event doorbell (single consumer, opt-in, default OFF). When `1` the producer (`Exporter`) creates a Win32 auto-reset named event and signals it after each `publish_frame()`; the consumer (`Importer.wait_for_doorbell()`) blocks on the event instead of poll-sleeping on `NO_FRAME`. Expected: sub-300 µs notify latency and ~zero idle CPU while waiting. Must be set on **both** producer and consumer. **Single-consumer only** — auto-reset wakes exactly one waiter. **Windows-only** — silently disabled on Linux/macOS (poll-sleep default is preserved). Default `0`. |
| `CUDALINK_WAIT_BACKEND` | `auto` | R5 native notification-wait backend, compiled into the core wheel on Windows (see [ADR-0012](docs/adr/0012-native-extension-in-core-wheel.md)). `auto` uses the native backend when the extension was compiled in, on Windows, and a CUDA runtime is already loaded in-process — with silent fallback to the Python spin/sleep path on any resolution failure. `python` always uses the pre-R5 path exactly. `native` skips the `auto` environment gate (same silent-fallback semantics otherwise). Activating the native backend forces the doorbell open for that connection even if `CUDALINK_DOORBELL=0` — the native backend's block phase cannot reach its <10 µs target without it. Target: p50 < 10 µs, p95 < 50 µs (vs 136–286 µs p50 poll-sleep baseline). |
| `CUDALINK_TD_PERSIST_STREAM` | `1` | Skip `stream_destroy` in TD Sender `cleanup()` so the IPC CUDA stream survives `deactivate`→`reactivate` cycles (F8). Free in single-pair (no deactivation ever happens); load-bearing in concurrent — without it, stream recreate on each reactivation collides with in-flight Receiver work, doubling first-settle `post=` latency (Phase 3.6 confirmed). Set to `0` to opt out. |
| `CUDALINK_TD_STREAM_PRIO` | `normal` | CUDA stream priority for the TD Sender's IPC stream. Default `normal` is safe for both single-pair and concurrent topologies — in single-pair only one stream exists per process so priority is moot; in concurrent, equal priorities prevent WDDM contention accumulation across reactivation cycles (high/high contention produces non-recovering cycle-3 shutdowns, Phase 3.6 Step C confirmed). Set to `high` only for explicit single-pair lowest-latency optimisation. |
| `CUDALINK_EXPORT_FLUSH_PROBE` | `1` | Insert a non-blocking `cudaStreamQuery(ipc_stream)` after `check_sticky_error` when `EXPORT_SYNC=0`. Forces WDDM-deferred CUDA submissions to drain each frame, preventing Windows Task Manager's 3D-engine counter from inflating when true compute load (per NVML) is low. NVML readings are unchanged — purely cosmetic/observability. Set to `0` to disable. |
| `CUDALINK_EXPORT_PROFILE` | `0` | Enable fine-grained per-region sub-timers in `export_frame()` and emit a `[PROFILE] pre=…us interop=…us post=…us memcpy=…us record=…us sync=…us sticky=…us flush_probe=…us shm=…us unacc=…us` line every 97 frames. Force-enables `verbose_performance` (TD) / `debug` (lib). Diagnostic-only; negligible overhead when on, zero when unset. |
| `CUDALINK_RECEIVER_REPORT_EVERY` | `150` | How often the TD Receiver COMP emits its Debug timing summary line (`Frame N \| FPS \| shape=… \| latency=… ms \| copy=… µs avg`) to the Textport when **Debug** is ON. The `copy=` figure is a **windowed (~150-frame) average** (v1.10.3), not a lifetime cumulative mean — it resets with each report window. Default 150 matches `example_receiver_python.py`'s `REPORT_EVERY`. Increase to reduce log volume; decrease for higher-frequency monitoring. No effect when Debug is Off. |
| `CUDALINK_SENDER_REPORT_EVERY` | `150` | How often the TD Sender COMP emits its Debug timing summary line (`Frame N \| FPS \| shape=… \| export=… µs avg`) to the Textport when **Debug** is ON. The `export=` figure is a **windowed (~150-frame) average** (v1.10.3), not a lifetime cumulative mean — it resets with each report window. Mirrors the receiver's report cadence; same tuning advice applies. No effect when Debug is Off. |
| `CUDALINK_NVTX` | `0` | Enable NVTX range annotations on top-level phase boundaries (`cudalink.exporter.flush_probe`, `cudalink.receiver.import_frame`, `cudalink.receiver.event_wait`, etc.) for Nsight Systems GPU timeline correlation. Zero overhead when off. Set to `1` before running any `nsys` capture; see [docs/PROFILING.md](docs/PROFILING.md). |
| `CUDALINK_NVTX_VERBOSE` | `0` | Enable additional sub-operation NVTX ranges (sticky-error check, D2A copy submit, SHM header read) inside the top-level phase ranges. Only useful for deep per-frame breakdown captures; implies `CUDALINK_NVTX=1`. |
| `CUDALINK_TD_GRAPHS_DEFERRED` | `0` | Defer CUDA Graph capture to after the second `export_frame()` call (TD Sender). Avoids a first-frame graph-capture stall in latency-sensitive topologies where the graph build cost would be visible. |
| `CUDALINK_TD_INIT_PACE` | `0` | Throttle the TD Sender init sequence to reduce WDDM saturation during activation windows (experimental). Adds a small sleep between consecutive CUDA API calls at `initialize()` time; useful when concurrent Sender+Receiver activation produces WDDM kernel-mode queue backpressure. |
| `CUDALINK_TD_BARRIER_SETTLE_FRAMES` | `30` | Number of frames the TD activation barrier counter remains armed after a Sender `initialize()` completes, giving the Python producer time to back off before publishing resumes. Increase if your Python producer's poll loop is slower than 30 frames at your target rate; decrease for tighter single-pair topologies. |
| `CUDALINK_NVML` | `0` | Append NVML GPU telemetry (utilization %, clocks MHz, PCIe Tx/Rx MB/s, temperature °C, power W, throttle reasons) to the 97-frame periodic stats line emitted by `CUDALINK_EXPORT_PROFILE`. Requires `nvidia-ml-py` (`pip install nvidia-ml-py`). Zero overhead when off. |

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

**Solution**: Compare against the baseline numbers in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Contributors with a local clone may reproduce using `python benchmarks/bench_graphs.py` (standalone) or `python benchmarks/bench_sweep.py --quick` (multiprocess).

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

#### Method 1: Build wheel (recommended — portable, installs into any environment)

```bash
git clone https://github.com/forkni/cuda-link.git
cd cuda-link

# Run the build script (uses PEP 517 isolated build via python -m build)
utils\build_wheel.cmd
# Output: dist\cuda_link-1.12.0-py3-none-any.whl  (~30 KB)

# Install into any Python environment — conda, venv, system Python, TouchDesigner Python:
pip install "dist\cuda_link-1.12.0-py3-none-any.whl[torch]"   # PyTorch GPU tensors
pip install "dist\cuda_link-1.12.0-py3-none-any.whl[cupy]"    # CuPy GPU arrays
pip install "dist\cuda_link-1.12.0-py3-none-any.whl[numpy]"   # NumPy CPU arrays
pip install "dist\cuda_link-1.12.0-py3-none-any.whl[all]"     # All output modes

# Force reinstall to update:
pip install --force-reinstall "dist\cuda_link-1.12.0-py3-none-any.whl[torch]"
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
from cuda_link import Importer, ImportSpec, ImportOutcome

importer = Importer.open(ImportSpec(shm_name="my_project_ipc"))
result = importer.get_frame()
if result.outcome is ImportOutcome.NEW_FRAME:
    tensor = result.frame  # torch.Tensor, GPU zero-copy
```

The `cuda-link` package contains only the **consumer-side** Python code (`src/cuda_link/`). The TouchDesigner extension is distributed separately.

### For TouchDesigner Integration

**Option A: Use the .tox component** (recommended)

Drag `TOXES/CUDAIPCLink_v1.12.0.tox` into your TouchDesigner network.

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

See [CHANGELOG.md](CHANGELOG.md) for the full history.

---

## License

MIT License - See LICENSE file

## Credits

Original implementation by Forkni (forkni@gmail.com).
Extracted and refactored from the StreamDiffusionTD project.
