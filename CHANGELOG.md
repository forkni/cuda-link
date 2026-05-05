# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
