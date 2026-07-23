# CUDA-Link — Component Help

> **Name:** CUDA-Link
> **Description:** Zero-copy GPU texture sharing via CUDA IPC
> **Author:** forkni (<forkni@gmail.com>)

**Zero-copy GPU texture sharing between TouchDesigner and external Python processes using CUDA Inter-Process Communication (IPC).**

---

## Overview

CUDAIPCLink transfers GPU textures between TouchDesigner and a Python process without copying data through CPU memory. Texture data stays on the GPU at all times — only a small control packet (~433 bytes) is exchanged through OS shared memory to coordinate access.

The component operates in two modes: **Sender** (TouchDesigner exports textures to Python) and **Receiver** (Python sends frames back into TouchDesigner). Both directions use the same underlying protocol, so two TouchDesigner instances can also communicate directly with each other.

Per-frame *coordination* overhead (GPU event record + `write_idx` update) is typically **0.5–2 µs**. End-to-end, CUDA-Link is measured at **~3.4× faster** than copying textures through CPU shared memory (1080p RGBA float32: ~1.6 ms CUDA-Link vs ~5.4 ms CPU SharedMemory) — with the **producer-side write 4–19× faster**, and a zero-copy GPU consumer (`torch.Tensor`/`cupy.ndarray` via `get_frame()`) collapsing the read side to **under 5 µs**. See [Performance Reference](#performance-reference) below and `docs/BENCHMARKS.md` for full methodology.

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
| IPC mem handle × N | 64 B each | GPU memory handle per slot |
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

If the Python consumer has `CUDALINK_D2H_PIPELINED=1` enabled, it drains the in-flight D2H copy and re-primes the double-buffer on reconnect — the first `get_frame_numpy()` call of the new session returns `NO_FRAME` (priming), matching fresh-session behavior.

---

## Parameters

### Active

**Type:** Toggle | **Default:** On

Master enable/disable switch for the CUDA IPC pipeline.

- **On:** The component initializes GPU resources on the first frame and processes every frame thereafter (in Receiver mode, the cook is skipped on idle frames where the producer has not written a new frame — `v1.10.2+`).
- **Off:** All GPU work stops immediately. `export_frame()` and `import_frame()` return without doing anything. Calling cleanup frees all GPU buffers, destroys IPC events, closes shared memory, and (in Sender mode) signals shutdown to connected consumers. The `Numslots` parameter is re-enabled for editing in Sender mode.
- **Toggling On** does not re-initialize immediately — GPU resources are re-created lazily on the next frame callback.
- Hot-swappable: can be toggled at any time without restarting TouchDesigner.

---

### Mode

**Type:** Menu | **Default:** Sender | **Options:** Sender / Receiver

Sets the direction of data flow.

- **Sender:** This component is the producer. It captures the upstream texture each frame, copies it into the GPU ring buffer, and makes it available to an external Python process (or another TD instance in Receiver mode).
- **Receiver:** This component is the consumer. It reads GPU frames produced by an external Python process (using `Exporter`) and imports them into a Script TOP for use in the TD network.

Switching modes triggers a full cleanup of the current state and lazy re-initialization on the next frame. In Receiver mode, the `Numslots` parameter is locked and read-only — the slot count is determined by the sender's shared memory protocol and automatically reflected in the parameter display.

---

### Ipcmemname

**Type:** String | **Default:** `cudalink_ipc_TD>>Python` (Sender mode) / `cudalink_ipc_Python>>TD` (Receiver mode)

The name of the OS shared memory segment used to exchange GPU handles between the sender and receiver.

Both sides **must use the exact same name**. On Windows, this maps to a named `CreateFileMapping` kernel object.

Changing this parameter while active triggers a full cleanup and reconnection:

- In Sender mode: re-initializes on the next frame export.
- In Receiver mode: immediately resets the retry counter and attempts to connect on the next frame start (without waiting through the current backoff interval).

Use different names to run multiple independent sender/receiver pairs simultaneously in the same TouchDesigner session.

---

### Numslots

**Type:** Integer Menu | **Default:** 3 | **Supported range:** 2–10 (2–5 recommended/tested)

Number of ring buffer slots in the GPU pipeline.

- **Hard bounds:** values outside 2–10 are rejected outright (below 2 there is no double-buffering; above 10 overflows internal fixed-size tables). Values 6–10 work but are less exercised in testing than 2–5.
- **Higher values** (e.g., 4–5) reduce the chance of producer/consumer contention when frame processing takes variable time. Each additional slot uses one full texture worth of GPU memory (`ceil(W × H × C × sizeof(dtype) / 2 MiB) × 2 MiB`).
- **Lower values** (e.g., 2) reduce GPU memory usage at the cost of slightly increased contention risk.
- **3 slots (default)** is sufficient for the vast majority of use cases.

**Lock behavior:**

- Only editable when `Mode = Sender` and `Active = Off`.
- Locked automatically when `Active` is turned On.
- In Receiver mode: always locked. The actual slot count is read from the sender's shared memory and displayed here for reference.

Changing this parameter while active is silently ignored. Changing it while inactive triggers a cleanup and lazy re-initialization on the next frame.

---

### Status

**Type:** String (read-only) | **Default:** `Idle`

Live status display — updated every frame while the component is active. Cannot be edited.

| Value | Meaning |
|-------|---------|
| `Idle` | Component is inactive (`Active = Off`) or no transfer in progress. |
| `<W>x<H> <dtype> <ch>ch` | Active transfer — e.g. `1920x1080 uint8 4ch`. Updated after each successful frame. |
| `WARNING: <msg>` | Non-fatal issue — e.g. `WARNING: unsupported pixel format '11:11:10'`. Frames are skipped until resolved. The COMP node body tints yellow and `warning_emitter` shows a local badge. |
| `ERROR: <msg>` | Fatal engine error (GPU/IPC init failure). The COMP node body tints red. Toggle `Active` Off → On to recover after fixing the underlying cause. |

---

### Debug

**Type:** Toggle | **Default:** Off

Enables verbose performance logging to the TouchDesigner Textport. Behaviour differs by mode.

- **Off:** Only critical errors and state changes are logged (both modes).

#### Sender mode

- **On:** emits two kinds of output:

  **Per-frame `EXPORT_PROFILE` breakdown** (every 97 frames, only when `CUDALINK_EXPORT_PROFILE=1` is also set):

  ```
  Frame 97 [PROFILE] memcpy=52.3us record=3.1us sync=41.8us sticky=1.2us flush_probe=0.0us shm=2.4us unacc=4.6us total=105.4us
  ```

  - `memcpy` — D2D memcpy enqueue time
  - `record` — IPC event record time
  - `sync` — `stream_synchronize` blocking time (TD Sender defaults to blocking sync, `CUDALINK_EXPORT_SYNC=1`, for the source-buffer-lifetime guard — see Troubleshooting)
  - `sticky` — sticky CUDA-error check (`CUDALINK_STICKY_ERROR_CHECK`)
  - `flush_probe` — non-blocking stream-query probe (only runs when sync is off; `CUDALINK_EXPORT_FLUSH_PROBE`)
  - `shm` — shared-memory publish (`publish_frame`) time
  - `unacc` — unaccounted time (total minus the sum of the above)
  - `total` — full `export_frame()` wall-clock time, windowed-averaged over the same 97 frames

  **Windowed summary line** (every 150 frames, configurable via `CUDALINK_SENDER_REPORT_EVERY`, always active when Debug=On — no `EXPORT_PROFILE` required):

  ```
  [CUDAIPCExtension:Sender] Frame  150 |  59.4 FPS | shape=(1080, 1920, 4) dtype=uint8 | export=45.2 µs avg (write_idx=150)
  ```

  The `export=` figure is a **windowed (~150-frame) average** that resets with each report;
  it reflects the current session's performance, not a lifetime mean.

#### Receiver mode

- **On:** every 150 frames (configurable via `CUDALINK_RECEIVER_REPORT_EVERY` env var), prints a
  per-frame summary line:

  ```
  [CUDAIPCExtension:Receiver] Frame  150 |  60.4 FPS | shape=(1080, 1920, 4) dtype=uint8 | latency=10.09 ms | copy=129.2 µs avg (slot=2, write_idx=231)
  ```

  - **FPS** — frames consumed ÷ wall time since the first consumed frame
  - **shape** — texture dimensions in numpy H×W×C order
  - **latency** — `now − producer_timestamp`; valid when sender and receiver run on the same machine (TD→TD and Python→TD setups use `time.perf_counter` which is system-wide on Windows)
  - **copy** — windowed (~150-frame) average of `copyCUDAMemory` wall time, covering only the frames in the current report window (in sync with the windowed FPS beside it); analogous to `get_frame= µs avg` in the standalone Python receiver
  - **slot / write_idx** — ring buffer diagnostics, sampled as a **1-in-N snapshot** of the
    sender's free-running counter (`slot = (write_idx − 1) % Numslots`). Because the producer
    advances `write_idx` slightly faster than the consumer reads (latest-frame-wins design),
    the sampled `slot` lands on a different value almost every report and looks non-sequential
    — this is expected sampling aliasing, **not** uneven slot usage. On the actual data path
    every slot is written and read in equal `0→1→2…` rotation; the per-frame `[DIAG]` lines
    printed for the first 5 frames after each re-init show that clean consecutive rotation.

Hot-swappable in both modes: can be toggled at runtime without affecting the pipeline.

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

1. Drop `TOXES/CUDAIPCLink_v1.12.1.tox` into your TD network.
2. Wire your source TOP into the component's input.
3. Set **Mode** = `Sender`.
4. Set **Ipcmemname** to a unique name, e.g. `my_pipeline`.
5. Toggle **Active** = On.
6. In Python, install `cuda_link` (see below) and connect:

   ```python
   from cuda_link import Importer, ImportSpec, ImportOutcome
   importer = Importer.open(ImportSpec(shm_name="my_pipeline"))
   result = importer.get_frame()          # ImportResult; .frame is torch.Tensor (zero-copy)
   result_np = importer.get_frame_numpy() # ImportResult; .frame is numpy array (CPU copy)
   ```

**Installing `cuda_link`:** the package is **not published on PyPI** — install the prebuilt
wheel via `scripts/install_td_library.py` (downloads from GitHub Releases, or resolves a
local `dist/` wheel if you built one). The native wheel ships a compiled extension but
**end-users need no MSVC/C++ build toolchain** — the wheel is prebuilt per release.

**Library mode (fewer Text DATs in the .tox):** run `install_td_library.cmd` once to install
`cuda_link` into a Python environment that TouchDesigner can see. The `CUDALinkBootstrap` DAT
inside the component will then load the package automatically — no `CUDALINK_LIB_PATH` setup
required when using TD Preferences mode (mode 4). Run `python scripts/install_td_library.py --help`
to see all five install modes. **Mode 5 (TD's own Python) is deprecated** — prefer mode 2
(dedicated venv) or mode 4 (system/parallel Python).

### Python → TD (Receiver mode)

1. In Python, create an exporter:

   ```python
   from cuda_link import Exporter, FrameSpec, GpuFrame
   import torch
   exporter = Exporter.open(FrameSpec(shm_name="ai_output", width=1920, height=1080))
   # Pass producer_stream so the D2D copy is ordered after your kernel writes.
   # PyTorch: torch.cuda.current_stream().cuda_stream
   exporter.export(GpuFrame(
       ptr=gpu_tensor.data_ptr(),
       size=gpu_tensor.nbytes,
       producer_stream=torch.cuda.current_stream().cuda_stream,
   ))
   ```

2. Drop the component into TD and set **Mode** = `Receiver`.
3. Set **Ipcmemname** to the same name (`ai_output`).
4. Toggle **Active** = On. The receiver will connect automatically once the Python exporter is running.

**Optional: Pipelined D2H (`CUDALINK_D2H_PIPELINED=1`)** — overlaps the device-to-host copy
with the consumer's CPU work. Disabled by default. First `get_frame_numpy()` returns `NO_FRAME`
(priming); re-primes on reconnect (+1-frame latency in steady state). See README Performance
Tuning table for break-even thresholds (1080p ~0.38 ms workload, 4K ~1.3 ms).

---

## Advanced / Opt-In Tuning

CUDA-Link ships a **prebuilt native extension** (`_native_waiter`, compiled into the core
wheel on Windows — end-users need no MSVC/C++ toolchain) plus several opt-in environment
variables for consumer-side wait behavior. The headline ones:

| Variable | Purpose |
|----------|---------|
| `CUDALINK_WAIT_BACKEND` | Native notification-wait backend (`auto` / `python` / `native`). Target p50 < 10 µs vs the ~136–286 µs poll-sleep baseline. Honest caveat: benchmarked as **not measurably faster** than the doorbell in practice — it ships because it never regresses, not because it's proven faster. |
| `CUDALINK_DOORBELL` | Win32 named-event doorbell wake (single consumer, Windows-only). Sub-300 µs notify latency, ~zero idle CPU while waiting. Must be set on both producer and consumer. |
| `CUDALINK_TORCH_GPU_WAIT` | GPU-side wait for `get_frame()` (torch backend) — issues `cudaStreamWaitEvent` instead of CPU spin/sleep. Trade-off: `ImportOutcome.TIMEOUT` becomes unreachable on this path. |

This is not the full list — CUDA-Link exposes roughly 40 `CUDALINK_*` tuning variables in
total (D2H streaming, activation barriers, CUDA Graphs, NVTX profiling, etc.). See the
**Performance Tuning** table in the project `README.md` for the complete reference.

---

## Performance Reference

| Operation | Typical Time | Notes |
|-----------|-------------|-------|
| Per-frame coordination overhead | 0.5–2 µs | GPU event record + `write_idx` update — not the data transfer itself |
| Cross-process IPC notify latency | ~136–286 µs | Poll-sleep baseline; `CUDALINK_WAIT_BACKEND=native` targets p50 < 10 µs |
| First-frame initialization | 50–100 µs | One-time GPU buffer allocation + IPC handle creation |
| `export_frame()` (1080p RGBA float32) | ~106 µs | Standalone Python `Exporter`, EXPORT_SYNC=1, RTX 4090 — full D2D copy + IPC record, runs on GPU |
| Receiver `copyCUDAMemory` into TD (1080p) | ~130 µs | Typical measured value (see the Receiver Debug example above); includes CUDA→OpenGL interop inside TD |
| D2H numpy copy (1080p RGBA float32) | ~1.3 ms | Only when using `get_frame_numpy()`; avoided entirely by `get_frame()` (zero-copy GPU tensor) |

**Baseline comparison:** CPU SharedMemory at 1080p RGBA float32 costs **~5.4 ms** end-to-end
per frame vs CUDA-Link's **~1.6 ms** — CUDA-Link is **~3.4× faster end-to-end**, with the
producer-side write alone 4–19× faster. Numbers from `docs/BENCHMARKS.md`
(RTX 4090 / PCIe 4.0 x16 / driver 596.36).

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

**Sender `export=` windowed average is higher than expected**

- The `top_op.cudaMemory()` OpenGL→CUDA interop call happens *before* the timed `export_frame()` region and is not broken out as a separate Debug metric — it is not controllable by this component and is normal for large textures or when the GPU is under heavy load. If `export=` itself (or the `[PROFILE]` `memcpy=`/`sync=` fields) is high, that points to the D2D copy or blocking sync instead.

**Consumer crashes with CUDA 719 / `cudaErrorLaunchFailure` after receiving IPC frames**

- This indicates a producer-side source-buffer lifetime race: the D2D memcpy read the TD
  texture after TD reclaimed it. From v1.10.1 the TD Sender **blocks by default** (post-copy
  `stream_synchronize`) so the source is live until `export()` returns — upgrade to
  `CUDAIPCLink_v1.10.1.tox` to fix this. If you are on v1.10.0 and cannot upgrade immediately,
  set `CUDALINK_EXPORT_SYNC=1` in the environment that launches TouchDesigner as a stopgap.
  See CHANGELOG 1.10.1 for the full root-cause analysis. Upgrade to
  `CUDAIPCLink_v1.12.1.tox` (current) to have the fix and all subsequent fixes included.

---

## Requirements

- **OS:** Windows 10 / 11 (CUDA IPC handle sharing is Windows-only)
- **CUDA:** 11.x, 12.x, or 13.x runtime (the loader prefers 13.x/12.x, tested with 12.4 and 12.8; 11.x accepted as a fallback for systems that haven't migrated)
- **GPU:** NVIDIA, CUDA compute capability 3.5 or higher
- **TouchDesigner:** 2022.x or later
- **Python (consumer side):** 3.11+ recommended (matches TouchDesigner's bundled interpreter). The prebuilt native wheel targets **cp311 (Python 3.11)**; a `py3-none-any` pure-Python fallback wheel also installs on 3.9+ interpreters if needed. Not published on PyPI — see [Quick Start](#quick-start) for install instructions.
