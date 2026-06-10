# Profiling Guide

Operational guide for the `compute-sanitizer → nsys → ncu` workflow in cuda-link.

---

## 0. Tool Selection

| Tool | Purpose | When to use |
|---|---|---|
| `compute-sanitizer` | Correctness gate: memory errors, race conditions, synchronization | Always first — before any perf work |
| `nsys` | System timeline: which streams collide and when, cross-process boundary | After correctness is confirmed |
| `ncu` | Kernel deep dive: Speed-of-Light, memory bandwidth, warp stalls | After `nsys` narrows to specific kernels |

cuda-link's multi-stream topology (Sender `ipc_stream`, Receiver `_rx_stream`, Importer
`_numpy_stream`, optional `_d2h_streams[1..n-1]`, cudart graph stream) means timeline-level
visibility is required to diagnose interaction between streams. CPU-side `CUDALINK_EXPORT_PROFILE`
timers capture aggregate latency but cannot show GPU-side serialisation. nsys shows both.

---

## 0.5. Step 0: Napkin Math (Mandatory Pre-work)

Before opening any profiler, calculate expected GPU time and compare to measured API call latency.
This prevents misinterpreting WDDM enqueue overhead as a kernel performance problem.

### cuda-link IPC D2D copy: worked example

Payload: 1 MB (typical HD frame slice). GPU: RTX-class with ~600 GB/s device bandwidth.

```
Expected GPU kernel time = 1 MB / 600 GB/s ≈ 1.7 µs
Expected GPU kernel time = 1 MB / 3,350 GB/s (H100 HBM3) ≈ 0.3 µs
```

**Measured v3 nsys results (producer-side CUDA API trace) — historical baseline; see §8 for current v5 measurements:**

| CUDA API call | Median | Interpretation |
|---|---:|---|
| `cudaGraphLaunch` | ~30 µs | CPU-side WDDM API enqueue — **not GPU execution time** |
| (D2D copy on GPU) | ~0.3 µs (calculated) | Actual kernel at 3.35 TB/s — healthy |
| `cudaStreamSynchronize` | **~444 µs** | **WDDM batch-flush wait** — dominant cost |
| Sender slot p50 | 562 µs | Sum of above + Python overhead |

**Key conclusion:** The 30 µs `cudaGraphLaunch` is the Windows kernel transition to submit the
GPU work item via WDDM. The GPU kernel itself completes in under 2 µs. The 444 µs
`cudaStreamSynchronize` is the CPU blocking on the WDDM batch being submitted and acknowledged
by the GPU driver — this is a WDDM scheduling artifact, not a kernel efficiency problem.

`ncu` profiles GPU kernel execution. It will correctly show the D2D copy is memory-bandwidth
bound at ~80–95% SOL. That is a healthy result. It does **not** explain the 444 µs sync.
To investigate that, use nsys WDDM lane analysis (see §4 and §6).

---

## 1. Setup

### Install the `nvtx` package

```bash
pip install nvtx
```

Without this package the `CUDALINK_NVTX` env var is a no-op (the shim degrades silently).

### Nsight tool locations on Windows

Default after CUDA 12 install:

```
C:/Program Files/NVIDIA Corporation/Nsight Systems <version>/target-windows-x64/nsys.exe
C:/Program Files/NVIDIA Corporation/Nsight Compute <version>/ncu.exe
```

Ensure both are on `PATH` before running the scripts:

```powershell
# In PowerShell
$env:PATH += ";C:/Program Files/NVIDIA Corporation/Nsight Systems 2024.6.1/target-windows-x64"
$env:PATH += ";C:/Program Files/NVIDIA Corporation/Nsight Compute 2024.3.2"
```

### Clock control on WDDM

WDDM (Windows Display Driver Model) does **not** support `nvidia-smi -pm 1` persistent mode.
All runner scripts use `--clock-control base` instead, which tells ncu to use the boost clock
without locking it. Results are comparable across runs on the same machine but may vary more
than on bare-metal Linux. Do not attempt `nvidia-smi -ac` on WDDM — it has no effect and
masks the issue.

---

## 2. Workflow

### Step 1 — Correctness gate

```powershell
./scripts/profiling/run_compute_sanitizer.ps1
# expect: exit 0, no memcheck violations
# output: memcheck.log in your working directory
```

Fix any violations before proceeding. CUDA IPC uses cross-process device pointers; a single
out-of-bounds write can corrupt another process's GPU allocation without triggering a page fault.

### Step 2 — System timeline

```powershell
$env:CUDALINK_NVTX = "1"
$env:CUDALINK_NVTX_VERBOSE = "1"
./scripts/profiling/run_nsys.ps1
# output: run.nsys-rep in your local results directory
#         analyze.txt (anti-pattern report)
```

Run the automatic diagnostic report before opening the GUI:

```powershell
nsys analyze run.nsys-rep
# writes analyze.txt — scan for anti-patterns first
```

Open the report:

```powershell
nsys-ui run.nsys-rep
```

**What to look for:**
- NVTX ranges aligned with CUDA kernel activity on the correct stream
- No unexpected serialisation between `ipc_stream` and `_rx_stream` in the steady state
- `cudalink.sender.export_frame.*` and `cudalink.receiver.import_frame.*` do not overlap when
  `CUDALINK_TD_STREAM_PRIO=normal` and `CUDALINK_TD_PERSIST_STREAM=1` (see §4 below)

### Step 3 — Kernel deep dive

```powershell
./scripts/profiling/run_ncu.ps1          # Sender export path
./scripts/profiling/run_ncu_receiver.ps1 # Receiver import path
# output: sender.ncu-rep, receiver.ncu-rep in your local results directory
```

Open in the Nsight Compute GUI:

```powershell
ncu-ui sender.ncu-rep
```

**SOL classification — use this table before drawing conclusions:**

| SM% | DRAM% | Classification | Primary next step |
|---|---|---|---|
| ≥60% | any | Compute-bound | ComputeWorkloadAnalysis + warp stalls |
| any | ≥60% | Memory-bandwidth bound | MemoryWorkloadAnalysis → Sectors/Request |
| <30% | <30% but Memory %SOL ≥60% | Internal congestion (shared/L1) | L1/shared hit rate, bank conflicts |
| <40% | <40% | Latency-bound | WarpStateStats, instruction-level stalls |

**For the IPC D2D copy specifically:** expect DRAM ≥80% (memory-bandwidth bound). A healthy
kernel will show near-peak bandwidth at very short duration. See §0.5 for why this does not
explain the dominant `cudaStreamSynchronize` latency.

Check the **Speed-of-Light** roofline: the IPC D2D `memcpy_async` should be memory-bandwidth
bound (~90%+ of HBM/GDDR bandwidth on typical payloads). If SOL is unexpectedly low, check
for coalescing issues in **Memory Workload Analysis → Sectors/Request**.

**Two-pass ncu workflow (reduces WDDM TDR risk):**

Run two separate invocations rather than `--set full` in a single session:

- **HW pass (default, safe):** `SpeedOfLight` + `MemoryWorkloadAnalysis` — collects hardware
  counters only. Runs in seconds. Use this first to classify the bottleneck. This is what
  `run_ncu.ps1` does with no arguments.

- **SW pass (higher TDR risk):** `--set full` or explicit `SourceCounters,InstructionStats,
  WarpStateStats` — requires SW-patched replay, significantly more replay passes per kernel.
  Keep `--launch-count 1` to limit exposure. Run as a second invocation only when the HW pass
  identifies a kernel worth drilling into:

  ```powershell
  ./scripts/profiling/run_ncu.ps1 -Set full   # SW pass, sender path
  ```

The `--import-source yes` flag (already in both runners) is required for the Source/SASS
correlation tab in ncu-ui and for the `ncu_report` Python API.

---

## 3. NVTX Annotation Taxonomy

All ranges are emitted at the **Python process** level (shim: `src/cuda_link/_nvtx.py`) and
at the **TouchDesigner COMP** level (shim: `td_exporter/NVTXShim.py`). Both shims read
`CUDALINK_NVTX` and `CUDALINK_NVTX_VERBOSE` once at import; zero overhead when unset.

| Phase | Top-level range | Sub-ranges (verbose only) | Color |
|---|---|---|---|
| TD Sender export | `cudalink.sender.export_frame.slot<N>` | `cudalink.sender.memcpy`, `cudalink.sender.record_event` | green |
| Python Exporter | `cudalink.exporter.slot<N>` | `cudalink.exporter.memcpy`, `cudalink.exporter.record_event`, `cudalink.exporter.flush_probe`, `cudalink.exporter.shm_write` | green |
| TD Receiver import | `cudalink.receiver.import_frame.slot<N>` | `cudalink.receiver.event_wait` | blue |
| Python Importer GPU | `cudalink.importer.get_frame.slot<N>` | `cudalink.importer.event_wait` | purple |
| Python Importer numpy | `cudalink.importer.get_frame_numpy.slot<N>` | `cudalink.importer.event_wait`, `cudalink.importer.d2h_copy` | orange |

**Slot identity, not frame number** — slot count is bounded (1–3), frame count is unbounded.
This keeps the `nsys stats` symbol table compact and timeline labels readable.

**Sub-ranges** appear only when `CUDALINK_NVTX_VERBOSE=1`. They nest inside the top-level range
for the same phase and are safe to enable in profiling sessions but add Python overhead per
context manager; disable for production.

### Cross-process timeline

`bench_sweep.py` spawns producer and consumer as separate processes. nsys produces one
`.nsys-rep` per process (they share wall-clock). In nsys-ui tile the two reports to align
their timelines:

```
File → Open → run.nsys-rep        (producer/TD process)
File → Open in Same Window → run_1.nsys-rep  (consumer process)
```

The four NVTX phases (sender, exporter, receiver, importer) should interleave on alternating
stream lanes, not stack vertically (which would indicate stream priority contention).

---

## 4. Multi-Stream Topology and Load-Bearing Flags

Two env flags are required for stable concurrent topology. They were discovered by subtractive
probing (see `SESSION_LOG.md` Phase 3.6 for the full timeline).

### Correct topology (flags set)

```
SET CUDALINK_TD_STREAM_PRIO=normal
SET CUDALINK_TD_PERSIST_STREAM=1
```

On the nsys timeline, the Sender export stream (`ipc_stream`) and Receiver import stream
(`_rx_stream`) run concurrently on separate GPU SM lanes. `cudalink.sender.export_frame.*`
and `cudalink.receiver.import_frame.*` NVTX ranges run in parallel without head-of-line
blocking. Post-settle latency stabilises within 3 frames.

### Regression signature (flags wrong)

```
SET CUDALINK_TD_STREAM_PRIO=high
SET CUDALINK_TD_PERSIST_STREAM=0
```

On the nsys timeline you will see:

1. `ipc_stream` and `_rx_stream` CUDA kernels stacking vertically (single SM lane, serialised)
   — symptom of high-priority contention accumulation across WDDM scheduling epochs.
2. A multi-second gap after Receiver reactivation before `cudalink.receiver.import_frame.*`
   resumes — symptom of cold submission queue stalls from `PERSIST_STREAM=0` (stream
   destroyed/recreated on each deactivate/activate cycle while Sender is in flight).
3. CPU-side `CUDALINK_EXPORT_PROFILE` shows `post=` latency growing monotonically across
   reactivation cycles (non-recovering). This is the same pattern as the Phase 3.6
   Step-C cycle-3 shutdown.

This deliberate regression is useful for validating that the NVTX instrumentation can see
the failure mode that motivated this work.

> **Note:** `scripts\probes\v4_capture_*.cmd` isolate the `PERSIST_STREAM=0` flag only
> (`STREAM_PRIO` is left at the correct value `normal`). A separate v4b run with both
> `STREAM_PRIO=high` and `PERSIST_STREAM=0` is needed to observe symptom #1 (stream stacking).
> v4 data is sufficient for NVTX coverage validation of symptoms #2 and #3.

---

## 5. WDDM-Specific Caveats

### TDR risk during ncu sessions

Nsight Compute replays kernels multiple times for counter collection. On WDDM, long replay
sessions can starve the display driver and trigger a TDR (Timeout Detection and Recovery)
reset. Mitigations already baked into the runner scripts:

- `--launch-count 5` — bounds the number of kernels replayed per invocation
- `--launch-skip 5` — skips warmup kernels
- `--replay-mode kernel` — **required** for CUDA Graph workloads: replays one kernel at a time.
  Range/application replay re-executes the entire CUDA Graph per pass (NVIDIA Nsight Compute
  Kernel Profiling Guide §2.2.4). On the IPC export path this re-launches the producer→consumer
  handoff graph, invalidating the cross-process IPC handle and producing corrupt or missing
  counter data. Kernel replay avoids re-launching the graph entirely.

If ncu still TDRs, reduce `--launch-count` to 2 or 3 in the script.

### `cudart64_12.dll` preference

cuda-link auto-detects the CUDA runtime DLL. Never hard-code a DLL path or force a version via
`PATH` reordering — this broke compatibility with CUDA 11.x in a prior incident. Trust the
detection logic in `CUDAIPCWrapper.py`.

### CUPTI single-subscriber rule

CUPTI (the profiling API used by both nsys and ncu) allows only **one subscriber per process**.
If a downstream consumer script imports `torch.profiler` and leaves it active, running nsys or
ncu against that process will produce incomplete traces or CUPTI errors. Stop `torch.profiler`
before attaching Nsight tools. This applies even though cuda-link itself does not use PyTorch —
the constraint is per-process, not per-library.

### Output file already exists — silent redirect to `%TEMP%`

When the target `.nsys-rep` path already exists, nsys on Windows silently redirects output to
`%TEMP%\nsys-Inter\` with no warning or error. The capture appears to complete successfully but
the file at the expected path is unchanged; the new report is in a temp location that gets wiped
on reboot.

**Symptom**: timestamp on `run.nsys-rep` not updated after a re-run; new report found under
`C:\Users\<user>\AppData\Local\Temp\nsys-<user>\`.

**Fix**: always pass `--force-overwrite=true` to `nsys profile`:

```powershell
nsys profile --force-overwrite=true --trace=cuda,nvtx,wddm --output "$out/run" ...
```

`run_nsys.ps1` and `run_nsys.sh` both include this flag. For the TD-pipeline cross-process
capture (where nsys attaches to a process you don't control via a wrapper script), pass it
explicitly on the command line, e.g.:

```
nsys profile --force-overwrite=true --output td_pipeline_producer/producer ...
```

### Parallel IPC consumers under nsys profiling

**Symptom:** A second TD receiver (or any second process calling
`cudaIpcOpenMemHandle`) that connects while the producer or first consumer is
being profiled by nsys with `--trace=cuda` sees:

```
[CUDAIPCExtension:Receiver] Slot N: cudaIpcOpenMemHandle failed:
    invalid resource handle (error 400: UNKNOWN_ERROR_400)
```

The producer's `write_idx` is actively incrementing — the IPC memory is live.
In normal (non-nsys) runs the same topology works without error.

**Root cause (status: triage-level, not fully attributed):**

nsys with `--trace=cuda` installs a CUDA driver-level interception hook in the
profiled process. This hook alters the driver-context metadata associated with
IPC export handles. A second process opening the same handle may receive an
invalid or stale handle descriptor because the driver-context tag used by the
first process has been remapped by the nsys hook.

Three hypotheses (probe matrix, in priority order):

| Probe | Action | If passes → | If fails → |
|---|---|---|---|
| P1 | Use `--trace=nvtx` only on the first process (drop `cuda,wddm`) | nsys `cuda` hook is the cause; file nsys bug report | P2 |
| P2 | Run both processes under nsys (`--output` different dirs for each) | Shared nsys hook context resolves it; workaround is dual-nsys | H2/H3 |
| P3 | HWS off + nsys on (requires reboot) | HWS=2 + nsys combination is the cause | Multi-factor |

**Known workaround:**

- Do **not** attach a second consumer while any process in the IPC topology is
  under `nsys --trace=cuda`.
- For topology testing with multiple consumers, run without nsys. Alternatively,
  profile all consumers simultaneously (each with its own `--output` path) rather
  than attaching one at a time.

**Status:** Known nsys instrumentation interaction. Not a cuda-link bug. No fix
in `TDConfig.py` or `CUDAIPCWrapper.py` — the IPC export handle is conformant;
the issue is in nsys's CUDA hook affecting cross-process handle visibility.

---

## 6. `CUDALINK_EXPORT_PROFILE` ↔ NVTX: What Each Measures

Both instruments coexist. They are complementary, not duplicates.

| Dimension | `CUDALINK_EXPORT_PROFILE` (CPU timers) | NVTX (GPU timeline) |
|---|---|---|
| **Enable** | `SET CUDALINK_EXPORT_PROFILE=1` | `SET CUDALINK_NVTX=1` |
| **What it measures** | CPU-side elapsed time per `export_frame()` sub-operation (enqueue cost only — async ops record enqueue latency, not GPU execution time) | GPU kernel launch and completion on the nsys/ncu timeline (actual GPU work duration) |
| **Granularity** | Rolling average logged every 97 frames to Python logger | Per-call range visible in nsys-ui |
| **Cross-process** | Per-process only | Both processes visible in tiled nsys-ui |
| **When to trust** | `memcpy=`, `record=`, `shm=` for diagnosing CPU scheduling jitter and SHM write latency | GPU-side for diagnosing stream contention, PCIe saturation, kernel occupancy |

**Key bridging rule**: if `EXPORT_PROFILE` shows `memcpy=40µs` but the nsys GPU timeline shows
the D2D copy taking 200µs, the gap is CPU-GPU enqueue latency (WDDM batch submission delay).
This is normal on WDDM. The nsys timeline is ground truth for actual GPU execution time.

If `EXPORT_PROFILE` shows `flush_probe=` growing across frames, it means `cudaStreamQuery`
is blocking on an unsubmitted WDDM batch — the deferred-submission accumulation that F8
(`PERSIST_STREAM=1`) was designed to prevent during reactivation. Correlate with the nsys
timeline: if the Sender stream shows no GPU activity during that window, the batch was buffered
by WDDM and `flush_probe` is flushing it.

---

## 7. WDDM Hardware-Accelerated GPU Scheduling (GPU-P)

### What it is

Hardware-Accelerated GPU Scheduling (HWS, also called GPU-P) moves GPU work scheduling from
the CPU-side WDDM kernel driver into the GPU hardware scheduler. On standard WDDM, the CPU
driver batches GPU submissions and delivers completion notifications on a ~600–700 µs heartbeat.
With HWS enabled, the GPU hardware processes the queue directly, reducing batch latency to
~50–100 µs.

**Effect on cuda-link:**

| Metric | WDDM software scheduling | WDDM GPU-P (HWS on) |
|---|---|---|
| Producer `cudaStreamSynchronize` p50 | ~617 µs (v4 baseline) | Expected ~50–100 µs |
| Consumer `import_frame` outlier max | ~36.5 ms (WDDM queue gap) | Expected < 5 ms |
| WDDM Copy engine max queue entry | 116 ms (v4 baseline) | Expected < 20 ms |

### How to toggle

1. Open **Settings → System → Display → Graphics → Default graphics settings**.
2. Toggle **"Hardware-accelerated GPU scheduling"** on or off.
3. **Reboot required** — the change does not take effect until restart.

Or via registry (requires reboot):
```
HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers
  HwSchMode  REG_DWORD  0 = disabled, 2 = enabled
```

### Runbook: v4 HWS comparison

Run the same protocol as v4 twice — once with HWS off (verify `HwSchMode=0`) and once with
HWS on (`HwSchMode=2`). Use the standard v4 capture scripts:

```cmd
scripts\probes\run_v4_regression_capture.cmd
```

After each capture, run `v4_analyze.cmd` and compare:

1. `producer_cuda_api_sum.csv` — `cudaStreamSynchronize` avg should drop from ~630 µs to < 100 µs.
2. `td_consumer_wddm_queue_sum.csv` — engine 6 (Copy) max queue entry should drop from > 100 ms to < 20 ms.
3. `td_pipeline_v4_findings.md` — consumer `import_frame` max outlier should drop from ~36 ms to < 5 ms.

### Self-documenting captures

The exporter emits the current HWS state as an NVTX startup range at initialization:

```
cudalink.startup.hws_mode=<value>
```

where `<value>` is `0` (software scheduling), `2` (hardware scheduling), or `unknown`
(non-Windows or registry key absent). This range appears in the nsys timeline and in
`nsys stats --report nvtx_sum`, so every archived `.nsys-rep` is self-documenting about
the WDDM scheduling mode in effect during that capture.

---

## 8. Async Export Path for Python-Sender Topologies

### When this applies

Standalone Python-sender deployments where no TD-Sender process shares the CUDA
context. Validated in the v5 nsys capture (findings documented in `td_pipeline_v5_findings_extended.md` in contributor archives).

> **Note — TD Sender users:** The **TD Sender** (TouchDesigner `CUDAIPCExtension` Sender
> COMP) defaults to **blocking** export as of v1.10.1 because its source is TD's transient
> cook-scoped TOP texture (`cm.ptr`), which TD reclaims immediately after the cook.  Async
> export returns before the D2D copy reads the source, causing reads-freed-memory (CUDA 719)
> under a loaded consumer.  This section applies only to standalone Python `Exporter` callers
> with a **persistent, caller-owned** source buffer.  See CHANGELOG 1.10.1 and ADR-0001.

### Flags

```cmd
SET CUDALINK_EXPORT_SYNC=0
SET CUDALINK_EXPORT_FLUSH_PROBE=1
```

These complement the standard topology flags from §4 and work best after enabling
HWS=2 (§7).

### Measured trade-off (v4 blocking → v5 async, same HWS=2 machine)

| Side | v4 baseline (`EXPORT_SYNC=1`) | v5 async (`EXPORT_SYNC=0 + FLUSH_PROBE=1`) | Δ |
|---|---|---|---|
| Producer `cudaStreamSynchronize` avg | 629.8 µs | **absent** | −629.8 µs (−100%) |
| Producer `flush_probe` NVTX avg | absent | 6.1 µs | +6.1 µs (replacement cost) |
| Producer slot p50 (`export_frame`) | 693.7 µs | **90.6 µs** | −603 µs (−87%) |
| Producer slot p99 | 2,997 µs | 225.5 µs | −2,771 µs (−92%) |
| Consumer `event_wait` p50 | 19.6 µs | 38.8 µs | +19.2 µs (redistributed wait) |
| Consumer `import_frame` p50 | 182.7 µs | 157.7 µs | −25 µs |
| Effective producer FPS | 58.7 | ~60 | +1.3 FPS |
| Net per-frame savings (producer − consumer overhead) | — | — | **~−584 µs** |

**How it works:** `EXPORT_SYNC=0` replaces `cudaStreamSynchronize` (~630 µs blocking
GPU fence) with a `cudaStreamQuery` poll loop (`cudalink.exporter.flush_probe`, avg
6.1 µs). The producer updates the SHM header as soon as `cudaStreamQuery` returns
success — before the D2D event fires on the GPU timeline. The consumer receives the
update earlier and calls `cudaStreamWaitEvent` to wait for the event itself, which
increases consumer `event_wait` by ~19 µs on average. The wait is redistributed,
not eliminated.

### `EXPORT_SYNC` defaults by deployment path

The default differs between the two producer implementations:

| Producer | Default | Rationale |
|---|---|---|
| **Library exporter** (`cuda_link.Exporter`) | `EXPORT_SYNC=0` (async) | Python senders are standalone with a caller-owned persistent source buffer; IPC events provide correct cross-process ordering. Falls back to blocking sync automatically when no IPC event exists for a slot. |
| **TD Sender** (`TDConfig.py`, `TDSender`) | `EXPORT_SYNC=1` (blocking, **default since v1.10.1**) | The TD source is TD's transient cook-scoped TOP texture (`cm.ptr`): TD reclaims it when the cook returns. Async export lets TD recycle the source while the IPC-stream copy is still queued → reads freed memory → CUDA 719 under a loaded consumer. Blocking ensures the D2D read completes before the cook exits. Set `CUDALINK_EXPORT_SYNC=0` to opt back into async **only** with a guaranteed-stable source. See CHANGELOG 1.10.1. |

> **This section (§8) applies only to the standalone Python `Exporter` callers described above.**
> For TD Sender deployments, note that `CUDALINK_EXPORT_SYNC=0` is an explicit opt-out from
> the safety-blocking default — not a recommended production configuration unless the source
> buffer lifetime is guaranteed by the caller.

**Coexistence safety — two distinct stream hazards, each fixed by the correct mechanism:**

1. **Receiver-side teardown TDR** (fixed since v1.4.1, `0918914`/F8 `0556197`): WDDM
   held stale CUDA↔D3D11 interop registrations when the receiver stream and IPC handles
   were torn down across deactivate→reactivate cycles → `DXGI_ERROR_DEVICE_REMOVED` → TDR.
   Fixed by **dedicated, persistent per-engine streams** (`CUDALINK_TD_PERSIST_STREAM=1`,
   default; `b7d51c1`, F7/F8 `3f6d1c2`/`0556197`). A blunt per-frame
   `cudaStreamSynchronize` on the sender never addressed this teardown-lifecycle hazard.

2. **Producer-side cross-stream race** (fixed since v1.9.0, `d2d4674` / SD `346a59f`):
   cuda-link's high-priority non-blocking IPC stream can race the producer's default-stream
   pack kernels → half-written BGRA buffer → torn gray frame. Fixed by **producer-stream
   ordering** (`record_source_sync` / `require_source_sync`; see "Producer-Stream Ordering"
   in `ARCHITECTURE.md`).

3. **TD Sender source-buffer lifetime race** (fixed since v1.10.1, CUDA 719): TD's TOP
   texture pointer (`cm.ptr`) is valid only within the cook frame. Async export can delay
   the IPC-stream copy past cook exit → reads freed memory. Fixed by **TD Sender blocking
   by default** (`_resolve_export_sync(None) → True`). The `record_source_sync` ordering
   primitive guarantees the copy starts after the source is fully written but does NOT keep
   the source live past the D2D read; only blocking export closes the lifetime window for a
   transient source. See CHANGELOG 1.10.1 and ADR-0001.

### Prerequisites

1. **HWS=2 (§7).** The async path moves the WDDM flush wait from the producer
   to the consumer's `event_wait`. With HWS=0 the consumer's event_wait tail
   still sees WDDM epoch gaps (~116 ms in v4). Enable HWS=2 and reboot before
   evaluating this config.
2. **NVTX enabled.** Use `SET CUDALINK_NVTX=1` and confirm
   `cudalink.exporter.flush_probe` ranges appear in nsys (target avg ~6 µs).
3. **Capture ≥ 60 s.** Producer slot p50 should stabilise in the 80–100 µs
   range. If it doesn't drop below 200 µs, check that both env vars are set
   in the process that launched the Python sender.

### Verification

Run `v5_analyze.cmd` after a capture and check:

```
nsys stats --report cuda_api_sum producer.nsys-rep | findstr cudaStreamSynchronize
```

Should return **no rows** — `cudaStreamSynchronize` must be absent from the
producer when async mode is active.
