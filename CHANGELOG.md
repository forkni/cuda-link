# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] — 2026-05-09

### Fixed

- **TDReceiver.import_frame() reconnect crash** — corrected two remaining calls to
  `self.cleanup_receiver()` (left over from the v1.2.0 engine-split refactor) to call
  `self.cleanup()`. Without this, the receiver crashed with `AttributeError:
  'TDReceiverEngine' object has no attribute 'cleanup_receiver'` when the sender shut
  down or restarted mid-session. Note: this fix was already mentioned in the v1.2.0
  entry's `### Fixed` section but landed in commit `05cda3a` after the logical v1.2.0
  release. v1.2.1 is the first tagged release where it ships.
  (`td_exporter/TDReceiver.py`)
- **`example_sender_python.py` Unicode banner** — replaced Unicode box-drawing
  characters with ASCII to prevent encoding errors on Windows console code pages other
  than UTF-8. (commit `30f7f2a`)

### Added — Diagnostics & Profiling Infrastructure

- **`scripts/profiling/v4_*` and `v5_*` capture runners** — cmd.exe scripts that
  launch Nsight Systems against the TD pipeline with pinned consumer/producer process
  pairs. v4 baseline (EXPORT_SYNC=1) and v5 (async-flush-probe: `EXPORT_SYNC=0` +
  `EXPORT_FLUSH_PROBE=1` + HWS=2) recipes validated end-to-end. Each runner emits
  standard Nsight summary CSVs via `nsys stats`. Includes `v4_analyze.cmd`
  post-capture decomposition. (commits `c41fda0`, `2e6a20f`, `18420de`, `c3f4e19`,
  plus reliability fixes `df0ec67`, `727634c`, `7862916`, `63f6036`, `112b620`)
- **`scripts/profiling/v5b_slot0_outlier_mine.py`** — sqlite3-only Python script that
  mines the v5 consumer nsys SQLite to classify every `import_frame.slot0` outlier
  (>2 ms) into one of four hypotheses: H5 (D2A WDDM stall), H4 (event_wait blocking),
  H2 (SHM poll wait / preemption), H1 (producer write-bias). Used to attribute the
  residual 30 ms slot0 outlier to a single `cudaMemcpy2DToArrayAsync` CPU-side WDDM
  fence. (commit `8cadef1`, documented in
  `benchmarks/results/nsys/td_pipeline_v5_findings_extended.md §G`)
- **WDDM queue capture flags** in nsys runners (`--trace=cuda,nvtx,wddm`,
  WDDM_QUEUE_PACKET / DMA_PACKET event capture) — adds GPU command-buffer queue-depth
  visibility for diagnosing WDDM scheduling-epoch gaps. (commit `c41fda0`)
- **HWS state probe** — Python helper that reads
  `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\HwSchMode` and reports
  whether hardware-accelerated GPU scheduling is active. Used to verify the HWS=2
  prerequisite before v5 captures. (commit `c3f4e19`)

### Added — Documentation

- **`docs/PROFILING.md` §5: "Parallel IPC consumers under nsys profiling"** — triage
  note for the `cudaIpcOpenMemHandle err 400` failure mode when a second TD receiver
  attaches while a first is profiled under `nsys --trace=cuda`. Documented as a known
  nsys CUDA-driver-hook interaction (not a cuda-link bug) with workaround.
  (commit `89546e8`)
- **`docs/PROFILING.md` §8: "Async Export Path for Python-Sender Topologies"** —
  flag-set documentation for `CUDALINK_EXPORT_SYNC=0` + `CUDALINK_EXPORT_FLUSH_PROBE=1`
  in standalone Python-sender deployments. Includes measured trade-off table (producer
  slot p50: 693.7 µs → 90.6 µs, −87 %; consumer event_wait +19 µs redistributed),
  rationale for not flipping the global default, and HWS=2 prerequisite.
  (commit `89546e8`)
- **`docs/PROFILING.md` napkin math + SOL classification table + two-pass workflow +
  CUDA Graph note** — restructured to lead with quick-decision tooling (when to use
  nsys vs ncu) before diving into recipes. (commit `b40c943`)
- **`docs/PROFILING.md` silent-redirect trap note** — documents that cmd.exe `>`
  silently swallows `nsys export` errors; `--force-overwrite=true` and exit-code
  checks added to all runners. (commit `5cebac6`)
- **`benchmarks/results/nsys/td_pipeline_v5_findings_extended.md`** — v5 capture
  analysis, async-flush-probe + HWS=2 validation (six acceptance criteria passed),
  and §G slot0 outlier root-cause attribution table. (commit `41d9a14`, F1 closure
  `8cadef1`)

### Changed

- **nsys / ncu runners hardened** — `--force-overwrite=true` on nsys;
  `--cache-control all`, `--import-source yes`, and `--set` flag on ncu. Eliminates
  silent-redirect failures and stale-cache misattributions. (commits `5cebac6`,
  `404917e`, `2e6a20f`)
- **`analyze_td_pipeline.py`** — gained CLI path arguments so the same script
  analyses v4 and v5 captures without env-var hardcoding. (commit `2e6a20f`)
- **TD installation path** bumped to `32820` in capture runners. (commit `18420de`)
- **`pyproject.toml` dev-dep**: `nvtx>=0.2` added so contributors can run profiling
  scripts without a separate `pip install nvtx`. (commit `77af0b7`)

### Internal

- **CGW PreToolUse guardrail** — `cgw-pre-bash.sh` intercepts `git reset --hard` /
  `git push --force` to non-PR branches before they execute. Local-only; not shipped
  in the package. (commit `0a14945`)
- **Example .toe project** bumped to `CUDA_Link_Example.45.toe`; new
  `Test_TD_Receiver1.toe` added for parallel-receiver topology testing.
  (commits `1fbbb61`, `fec21ca`, `72e1354`)
- **`TOXES/CUDAIPCLink_v1.2.0.tox`** added; `v1.1.0.tox` retired per `.gitignore`
  "only latest binary on main" policy. (commit `89a2b91`)

---

## [1.2.0] — 2026-05-07

### Added

- NVTX annotations on Sender/Receiver/Exporter/Importer phase boundaries
  (env-gated via `CUDALINK_NVTX=1` / `CUDALINK_NVTX_VERBOSE=1`; zero cost when off).
- `scripts/profiling/` — runner scripts for compute-sanitizer, nsys, and ncu
  (Windows `.cmd`/`.ps1` + POSIX `.sh` parity).
- `docs/PROFILING.md` — operational guide for Nsight workflow,
  WDDM caveats, and `EXPORT_PROFILE` ↔ NVTX bridging.

### Changed

- **TD extension refactored into facade + engine split** (`CUDAIPCExtension.py` → ~300 LOC facade;
  new `TDSender.py` / `TDReceiver.py` engine classes). `TDSenderEngine` owns all Sender-mode GPU
  resources; `TDReceiverEngine` owns all Receiver-mode resources. Mode switches tear down the old
  engine and construct a fresh one — zero cross-mode state leak. Public API (`export_frame`,
  `import_frame`, `switch_mode`, etc.) is unchanged; existing `.tox` callback templates work
  without modification.
- **`TDHost` adapter seam** (`TDHost.py`) isolates all `ownerComp.par.*`, `op(...)`,
  `top.cudaMemory()`, and `copyCUDAMemory()` calls from engine logic. Tests inject `FakeTDHost`
  / `FakeTOPHandle` — no TD runtime required.
- **`TDSenderConfig` frozen dataclass** (`TDConfig.py`) centralises all 11 `CUDALINK_*`
  environment-variable reads. Constructed once at extension init; engines read only
  `self._config.<field>`.
- **`docs/TOX_BUILD_GUIDE.md`** updated: Component Structure diagram now shows all eight Text DATs
  (`CUDAIPCWrapper`, `ActivationBarrier`, `NVMLObserver`, `TDHost`, `TDConfig`, `TDSender`,
  `TDReceiver`, `CUDAIPCExporter`); Step 3 expanded with per-DAT assembly instructions.
- **`docs/ARCHITECTURE.md`** and **`README.md`** Architecture section updated to document the
  facade-with-delegation layout and TDHost seam.
- **`CONTEXT.md`** created at repo root with canonical vocabulary for the new architecture.

### Fixed

- **Facade mode-gating** — `CUDAIPCExtension.import_frame()` / `export_frame()` now
  return `False` when called in the wrong mode instead of dispatching to an engine that
  lacks the method. Added `_check_deferred_cleanup()` and `update_receiver_resolution()`
  delegations so `callbacks_template.py` / `script_top_callbacks.py` continue to work
  without modification after the engine split. (`td_exporter/CUDAIPCExtension.py`)

- **`TDReceiver.import_frame()` reconnect crash** — two internal calls to the removed
  `self.cleanup_receiver()` (left over from the engine-split refactor) now correctly
  call `self.cleanup()`. Without this, the receiver crashed with `AttributeError:
  'TDReceiverEngine' object has no attribute 'cleanup_receiver'` the moment the sender
  shut down or restarted mid-session. (`td_exporter/TDReceiver.py`)

## [1.1.0] — 2026-05-06

### Added

- **`benchmarks/bench_sweep.py`** — full IPC roundtrip sweep (16 cells: 4 resolutions
  × 2 dtypes × graphs on/off). Two spawn-process workers (producer + consumer)
  exercise the IPC path end-to-end at 60 FPS, capture per-cell `export_us`,
  `get_numpy_us`, `e2e_us` (IPC notify), and `throughput_gbs` percentiles, and
  write `benchmarks/results/sweep_{timestamp}.{csv,json}` plus
  `sweep_latest.{csv,json}` for doc-update reproducibility. Validated on
  RTX 4090 / driver 596.36 / PCIe 4.0 x16 (2026-05-06).
- **CPU SharedMemory comparison block** in `README.md` and detailed comparison
  tables in `docs/ARCHITECTURE.md` — concrete speed-ups vs the original
  UT_SharedMem-class baseline at 1080p (~3.4× E2E) and 512×512 (~2.1× E2E),
  with 4–19× advantage on producer write. TouchOUT/Spout baselines are
  explicitly flagged as never measured.

- **CUDA Graphs for `export_frame()`** — `CUDAIPCExporter` now captures the
  per-frame `memcpy_async` into a 1-node CUDA Graph on first use and replays it
  via `cudaGraphLaunch` each frame. This cuts WDDM kernel-mode transitions from 3
  to 2 per frame, reducing CPU submission overhead by ~70% at 1080p float32
  (15.7 µs → 4.7 µs mean, measured async). Enabled by default; set
  `CUDALINK_USE_GRAPHS=0` to revert to the legacy stream path. Falls back
  automatically if graph capture or launch fails at runtime.
  (`src/cuda_link/cuda_ipc_exporter.py`, `src/cuda_link/cuda_ipc_wrapper.py`)

- **CUDA Graphs for TouchDesigner Sender** — the TD-side `CUDAIPCExtension`
  (Sender mode) gains the same graph capture path, gated by
  `CUDALINK_TD_USE_GRAPHS` (default `0`, opt-in pending soak). Probes the
  loaded cudart version via `cudaRuntimeGetVersion`; auto-disabled if the
  runtime is older than 11.3 (the `cudaGraphExecMemcpyNodeSetParams1D` API).
  (`td_exporter/CUDAIPCExtension.py`, `src/cuda_link/cuda_ipc_wrapper.py`
  adds `cudaRuntimeGetVersion` binding + `get_runtime_version()` helper)

- **Multi-stream D2H for `get_frame_numpy()`** — opt-in via
  `CUDALINK_D2H_STREAMS=N` (default `1`). Splits the D2H copy across N
  independent non-blocking streams. No throughput gain on PCIe 4.0 (single
  stream already saturates ~23–24 GB/s); may help on PCIe 3.0 or GPUs with dual
  DMA engines. (`src/cuda_link/cuda_ipc_importer.py`)

- **`cudaHostAllocPortable` for pinned D2H buffer** — the `get_frame_numpy()`
  pinned host allocation now uses `cudaHostAlloc` with `cudaHostAllocPortable`
  (flag `0x01`), making it accessible from any CUDA context in the process.
  Relevant when PyTorch, CuPy, or other runtimes are loaded alongside
  `cuda-link`. No throughput change; robustness improvement only.
  (`src/cuda_link/cuda_ipc_importer.py:875`)

- **Python lib gains `CUDALINK_EXPORT_PROFILE` + `CUDALINK_EXPORT_FLUSH_PROBE`** —
  the Python-side `CUDAIPCExporter.export_frame()` now reads the same two diagnostic
  env vars as the TD extension. `CUDALINK_EXPORT_PROFILE=1` enables fine-grained
  per-region sub-timers (`sync`, `sticky`, `flush_probe`) and emits a `[PROFILE]` line
  every 97 frames; force-enables `debug=True`. `CUDALINK_EXPORT_FLUSH_PROBE`
  inserts a non-blocking `cudaStreamQuery(ipc_stream)` after `check_sticky_error`
  when `EXPORT_SYNC=0`. Closes a long-standing instrumentation asymmetry between
  the TD extension and the Python lib. (`src/cuda_link/cuda_ipc_exporter.py`)

### Changed

- **Concurrent-topology load-bearing flags now default-on (Phase 4 / 4.1)** —
  `CUDALINK_EXPORT_SYNC`, `CUDALINK_ACTIVATION_BARRIER`,
  `CUDALINK_TD_ACTIVATION_BARRIER`, and `CUDALINK_TD_PERSIST_STREAM` flip from
  opt-in to default-on; `CUDALINK_TD_STREAM_PRIO` default flips `"high"` →
  `"normal"`; the experimental `CUDALINK_TD_INIT_CLEAR_STICKY` (F4) is removed
  entirely (never observed firing in 3.6 subtractive probe). Net effect: the
  validated Python-producer + TD-Sender concurrent topology now requires **zero
  env vars** to run safely. Each flag's load-bearing role was confirmed via
  Phase 3.6 step-by-step subtractive probes (2026-05-06, branch
  `feat/cuda-graphs-d2h-streams`). Set any flag to `0` to opt out.

- **`CUDALINK_EXPORT_FLUSH_PROBE` default flipped `"0"` → `"1"`** (both TD extension
  and Python lib). Phase 3 measurement (2026-05-04, RTX 30/40, 1080p RGBA8): the
  ~12 µs/frame `cudaStreamQuery` collapses Windows Task Manager's 3D-engine reading
  from ~65 % to ~7 % on rigs where WDDM defers GPU command submission, *without*
  the ~130 µs/frame cost of a full `cudaStreamSynchronize` (which `EXPORT_SYNC=1`
  pays). NVML true compute load is unchanged across all three settings — confirms
  the high Task Manager reading was a queue-depth artefact, not real load. The
  earlier v0.9.0 changelog entry calling this knob "diagnostic-only — hypothesis
  refuted" reflected an earlier rig where the artefact did not reproduce; the
  WDDM behaviour is rig- and driver-dependent. Set `CUDALINK_EXPORT_FLUSH_PROBE=0`
  to restore the prior default. (`td_exporter/CUDAIPCExtension.py`,
  `src/cuda_link/cuda_ipc_exporter.py`)

### Fixed

- **CUDA Graphs build crash on cudart 11.0–11.8** — replaced `cudaGraphInstantiate`
  (3-arg ABI stable only on CUDA 12.0+) with `cudaGraphInstantiateWithFlags` (stable
  3-arg API since CUDA 11.4). The prior binding called the 12.0 3-arg form against
  11.x DLLs that export the 5-arg form, producing an access violation
  (`0xFFFFFFFFFFFFFFFF`) under TD's subprocess PATH. Gate raised from cudart `>= 11.3`
  to `>= 11.4` to match the true floor of all graph APIs in use.
  (`src/cuda_link/cuda_ipc_wrapper.py`, `src/cuda_link/cuda_ipc_exporter.py`,
  `td_exporter/CUDAIPCExtension.py`)

- **cudart DLL preference** — `cudart64_12.dll` is now preferred over `cudart64_110.dll`
  in the by-name search list. TouchDesigner 2025+ ships both in `bin/`; `cudart64_12.dll`
  is the primary CUDA 12.x runtime TD itself uses; `cudart64_110.dll` is a legacy 11.x
  ABI compat shim. Preferring 12.x also improves process-wide cudart sharing with PyTorch.
  (`src/cuda_link/cuda_ipc_wrapper.py`)

- **Receiver second-activation freeze on Windows WDDM** — `cleanup_receiver()` no longer
  calls `cudaStreamSynchronize` before teardown. The synchronize was itself the cause of
  a 5+ second hang during cleanup (measured: 5406.9 ms), exceeding the Windows WDDM TDR
  threshold and triggering the NVIDIA driver reset popup ("An error occured trying to
  output to a Window") on the next `Active=True` toggle. `cudaStreamDestroy` releases the
  stream asynchronously once in-flight work completes and does not block the calling
  thread, so it cannot trigger TDR.
  (`td_exporter/CUDAIPCExtension.py`)

### Internal

- Test suite now resolves `cuda_link` from this repo's `src/` regardless of any
  previously installed `cuda_link` editable package in site-packages (`pyproject.toml`
  `pythonpath = ["src"]` + `tests/conftest.py` `sys.path.insert`).

---

## [1.0.1] — 2026-05-03

### Added

- **NVML `driver_model` field** — `NVMLObserver.snapshot()` now reports the active
  Windows driver model (`"WDDM"`, `"TCC"`, or `"MCDM"`) when running on Windows,
  using `nvmlDeviceGetCurrentDriverModel`. The key is absent on Linux (call raises
  `NVMLError_NotSupported` and is suppressed). Useful for diagnosing why a TCC-mode
  GPU exhibits different latency characteristics than the typical WDDM consumer setup.

### Internal / Docs

- `docs/ARCHITECTURE.md` — new "Cross-Process Error Attribution" subsection under
  Error Handling. Documents that `cudaPeekAtLastError`/`cudaGetLastError` only
  inspect the calling process's CUDA context — a producer-side GPU fault surfaces
  to the consumer as an IPC event timeout, not a CUDA error code. Debugging
  guideline: when consumer reports a stall, check producer logs first.
- `src/cuda_link/cuda_ipc_wrapper.py` — `malloc_host` docstring notes that
  this project is single-GPU by construction (`get_cuda_runtime` rejects a second
  device); multi-GPU usage would require `cudaHostAlloc` with `cudaHostAllocPortable`
  for cross-device visibility (Handbook §5.1).
- `.gitignore` — `scripts/git/`, `.githooks/`, `.gemini/` (deleted), and
  `cgw.conf.example` are now local-only / untracked. Removes 43 files from the
  index without touching working-tree state. Fresh clones no longer receive these
  developer-tooling paths.
- `build_wheel.cmd` — hardened Windows Python interpreter selection: prefers the
  `py -3` launcher to bypass Microsoft Store stubs, rejects `WindowsApps`
  reparse-point Python, and enforces `requires-python = ">=3.9"` from
  `pyproject.toml` with a clear error instead of cryptic build failures
  downstream. Build behavior on healthy Python ≥3.9 environments is unchanged.

[1.2.1]: https://github.com/forkni/cuda-link/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/forkni/cuda-link/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/forkni/cuda-link/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/forkni/cuda-link/compare/v1.0.0...v1.0.1

## [1.0.0] — 2026-05-02

### BREAKING CHANGES

- **Wire protocol incompatible with v0.9.x** — `PROTOCOL_MAGIC` bumped from `0x43495043`
  ("CIPC") to `0x43495044` ("CIPD"). Old senders/receivers will fail-fast at the magic check
  with "Protocol magic mismatch" and refuse to operate. Update both TD extension and Python
  package together.

### Changed

- **dtype encoding redesigned** — the 4-byte `dtype_code` enum at metadata+12 is replaced
  by a CUDA-aligned self-describing encoding: `format_kind` (uint8, `cudaChannelFormatKind`),
  `bits_per_component` (uint8), `flags` (uint16, bit 0 = bfloat16). Sender derives
  `bits_per_component` from `data_size / (W*H*C)` (authoritative — can no longer be silently
  wrong). Receiver validates `W*H*C*(bits/8) == data_size` and refuses init on mismatch.
  Fixes a bug where TD's `CUDAMemoryShape.dataType` could misreport the dtype (float32 for
  a uint8 buffer), causing a 4× size mismatch and "Source memory size is not large enough"
  errors at non-square resolutions like 576×1024 and 1550×288.

## [0.9.0] — 2026-04-23

### Added

- `CUDALINK_EXPORT_PROFILE=1` env var (TD extension, default OFF): enables 9 fine-grained
  per-region sub-timers in `export_frame` (`pre_interop`, `interop`, `post_interop`,
  `memcpy`, `record`, `sync`, `sticky_check`, `flush_probe`, `shm_publish`, `unacc`);
  columns appended to the 97-frame periodic stats line as `[PROFILE] pre=…us …`.
  Force-enables `verbose_performance` when set. Zero overhead when unset (~400 ns/frame
  when on). Used to close out the `export_frame` ~4190 µs gap diagnostic
  (SESSION_LOG 2026-04-23).
- `CUDALINK_EXPORT_FLUSH_PROBE=1` env var (TD extension, default OFF): inserts a
  non-blocking `cudaStreamQuery(ipc_stream)` after `check_sticky_error` when
  `EXPORT_SYNC=0`. Per CUDA Handbook p3/pg56. Diagnostic-only — retained on-tree for
  future use; the WDDM-batching hypothesis it was designed to test was refuted by data.

### Changed

- **TD: `CUDALINK_EXPORT_SYNC` default flipped `"1"` → `"0"`** (`td_exporter/CUDAIPCExtension.py`).
  Saves ~295 µs/frame of redundant CPU blocking; correctness is already guaranteed by the
  receiver's `cudaStreamWaitEvent(ipc_events[slot])`. Set `CUDALINK_EXPORT_SYNC=1` to restore
  prior behavior. Diagnostic details: SESSION_LOG 2026-04-23 (`export_frame` gap analysis,
  A/B/C experiment). This aligns TD's default with the Python lib default (`"0"`).

### Fixed

- `src/cuda_link/__init__.py` `__version__` bumped from stale `"0.7.3"` to `"0.9.0"`
  (was not updated during the v0.8.0 release; now in sync with `pyproject.toml`).

## [0.8.0] — 2026-04-23

### Added

- Configurable CUDA device index (`device: int = 0`) on `CUDARuntimeAPI`, `CUDAIPCExporter`,
  `CUDAIPCImporter`, and the TD extension (`Cudadevice` parameter). The `get_cuda_runtime()`
  singleton now raises `RuntimeError` when re-requested with a conflicting device.
- `NVMLObserver` (new module `src/cuda_link/nvml_observer.py`) — pull-based GPU telemetry:
  gpu/mem utilization, SM & memory clocks, PCIe Tx/Rx throughput, temperature, power draw,
  and decoded throttle reasons. Ref-counted `nvmlInit`/`nvmlShutdown`; context-manager
  friendly. Attach via `exporter.attach_nvml_observer(obs)` / `importer.attach_nvml_observer(obs)`.
- TD extension surfaces NVML metrics in the 97-frame periodic stats line when
  `CUDALINK_NVML=1` — appends `| [NVML] gpu=…% mem=…% sm=…MHz pcie_tx=…kbps
  pcie_rx=…kbps temp=…C power=…W` (plus `throttle=…` when non-empty).
- Sticky-error checking: `cudaPeekAtLastError` binding with `peek_at_last_error()` /
  `check_sticky_error(context)` helpers on `CUDARuntimeAPI`; called automatically after
  `export_frame()` and `get_frame_numpy()`. Opt out via `CUDALINK_STICKY_ERROR_CHECK=0`.
- Pinned-memory secondary fallback: `cudaHostRegister` path page-locks an
  `np.empty` buffer before falling back to pageable memory; tracks
  `pinned_memory_available: bool`.
- Priority IPC stream: `ipc_stream` created via `cudaStreamCreateWithPriority` at the
  device's greatest priority. New ctypes bindings for `cudaDeviceGetStreamPriorityRange`
  and `cudaStreamCreateWithPriority`.
- Spin-then-sleep wait loop in `_wait_for_slot`: tight spin for `CUDALINK_WAIT_SPIN_US` µs
  (default 200), then sleep-poll phase. Counters exposed via `get_stats()`.
- Bounded deferred-free watchdog: `cudaFree` / `cudaEventDestroy` in `cleanup()` run in
  daemon threads with `join(timeout=0.5)` to prevent WDDM hangs when the peer crashes.
- Windows high-resolution timer: `_HighResTimer` context manager calls
  `winmm.timeBeginPeriod(1)` around the `_wait_for_slot` polling loop, dropping the
  effective sleep floor from ~15 ms to ~1 ms.
- `CUDA_LAUNCH_BLOCKING=1` preflight warning logged at `CUDARuntimeAPI` init (~30× slowdown).
- `scripts/sync_td_wrapper.py` — keeps `td_exporter/CUDAIPCWrapper.py` and
  `td_exporter/NVMLObserver.py` byte-identical to canonical sources.
  Hooked into `build_wheel.cmd` step [1.5]; CI drift guard via `tests/test_wrapper_sync.py`.
- Git tooling expansion (14 new `scripts/git/*.sh` helpers): `bisect_helper`,
  `branch_cleanup`, `changelog_generate`, `clean_build`, `create_pr`, `create_release`,
  `push_validated`, `rebase_safe`, `repo_health`, `setup_attributes`, `stash_work`,
  `sync_branches`, `undo_last`, `configure`; plus a new `.githooks/pre-push` hook and
  `cgw.conf` config system.
- Test coverage: `test_nvml_observer.py` (11 cases), `test_wait_for_slot_busywait.py` (8),
  `test_device_affinity.py`, `test_cuda_ipc_exporter_python.py`. Suite: **122 passed, 2 skipped**
  (up from ~80 at v0.7.3).

### Changed

- **BREAKING (runtime default):** `CUDALINK_EXPORT_SYNC` defaults to **OFF** on the Python
  library's `export_frame()` hot path — `cudaStreamSynchronize` is no longer called
  automatically. Saves ~13–100 µs/frame. The TD extension still defaults **ON**. Set
  `CUDALINK_EXPORT_SYNC=1` to restore pre-v0.8.0 Python behavior.
- Silent `cudaMallocHost` failure escalated from `debug` to `warning`.
- `nvml` optional dependency switched from deprecated `pynvml>=11.5` to the official
  **`nvidia-ml-py>=12.535`**. The top-level `import pynvml` statement is unchanged —
  both packages expose the same module. Users with `pynvml` manually installed should
  `pip uninstall pynvml` before `pip install -e ".[nvml]"` to avoid namespace ambiguity.
- `CUDA_Link_Example.toe` updated — bundles `NVMLObserver` Text DAT inside both
  Sender (`/project1/CUDAIPCLink_to_Touchdesigner`) and Receiver
  (`/project1/CUDAIPCLink_from_Python`) components. Required for TD's sibling-import
  resolver to find `NVMLObserver` when loading the extension.
- `td_exporter/CUDAIPCWrapper.py` regenerated to mirror all wrapper changes (byte-identical
  to `src/cuda_link/cuda_ipc_wrapper.py`).

### Fixed

- `.claude/settings.json` PreToolUse Bash hook path corrected
  (`F:/RD_PROJECTS/...` → `D:/cuda-link/...`).

### Internal / Docs

- Explanatory comment at `src/cuda_link/cuda_ipc_importer.py` documenting the `getattr`
  fallback pattern in `cleanup()` for `__del__`-time partial-init safety.
- `docs/OPT_1_implementation_PLAN.md` moved to local-only (untracked).

[0.8.0]: https://github.com/forkni/cuda-link/compare/v0.7.3...v0.8.0
