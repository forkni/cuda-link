# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.12.2] - 2026-08-10

### Fixed

- **`cudaPointerAttributes` undersized against CUDA 13.x** (`cuda_runtime_types.py`) — the
  struct grew a `long reserved[8]` field in CUDA 13.x driver_types.h (56 bytes vs. 24 in
  12.x), but the ctypes binding still declared the 24-byte 12.x layout and the ABI guard
  hard-checked that literal. The loader probes 13.3 before 12.x, so `pointer_get_attributes()`
  — on the per-frame device-affinity path — was passing an undersized out-parameter to a
  13.x cudart. Extended the struct with the `reserved` field and raised the guard to 56;
  over-sizing is safe against both runtime generations, so no version branching is needed.
- **`cudaDevAttrAsyncEngineCount` documented as attribute 4** (`cuda_ipc_wrapper.py`) — the
  real value is 40; 4 is `cudaDevAttrMaxBlockDimZ`. Latent (no production caller queried it
  yet), but corrected before it could be relied on.
- **IPC capability probe aborted initialization on CUDA 11.x** (`cuda_ipc_wrapper.py`) —
  `check_ipc_capability()`, added earlier in this release and called unconditionally from
  `Exporter.open()`, queries `cudaDevAttrIpcEventSupport` (125), which CUDA 12.0 introduced.
  An 11.x runtime — still probed by the loader, and what TouchDesigner ships as
  `cudart64_110.dll` — returns an invalid-attribute error that the strict `errcheck` turned
  into a hard failure. The probe is now version-gated and its query is non-fatal, so a failed
  probe degrades to a logged diagnostic; only an explicit `cudaDevAttrIpcEventSupport == 0`
  still raises.

### Added

- **`check_ipc_capability()`** on the `CudaPort` protocol, `CTypesCUDAAdapter`, and
  `FakeCUDAAdapter`, called once from `Exporter.open()` before any IPC handle is minted.
  Queries `cudaDevAttrIpcEventSupport` and logs an informational note about the
  Windows-WDDM-vs-NVIDIA-documented-support gap (see ADR-0004) rather than hard-failing,
  since this project's whole premise is that legacy CUDA IPC works there in practice; raises
  only when the driver reports IPC flatly unsupported (`cudaDevAttrIpcEventSupport == 0`).

### Docs

- Corrected the inverted CUDA-IPC platform-support claim in `README.md` and
  `td_exporter/HELP_DOC.md` — NVIDIA documents the legacy IPC API as Linux/Windows-TCC only;
  this project runs it on Windows WDDM, which is validated in practice (ADR-0004) but
  outside NVIDIA's documented support envelope.
- Fixed the stale 14-mirror-DAT count (actual: 15) across `README.md`, `ARCHITECTURE.md`,
  the ADRs, and `TOX_BUILD_GUIDE.md` — the latter previously instructed builders to add "14
  mirror DATs," silently omitting `Doorbell`.

## [1.12.1] - 2026-07-12

### Fixed

- **NVML ref-count leak on failed `start()`** (`nvml_observer.py`) — a failure after
  `acquire()` (e.g. `nvmlDeviceGetHandleByIndex` raising) never released the ref, so
  `nvmlInit()` was never paired with `nvmlShutdown()` and repeated failures compounded
  the leak. `start()` now only releases a ref it actually acquired — calling `release()`
  unconditionally would have wrongly `nvmlShutdown()`'d an NVML that was never
  successfully initialized by this call.
- **`TDReceiverEngine.initialize_receiver()` leaked its locally-opened `shm_handle`**
  when a raise (e.g. from `create_stream_with_priority`) landed after the validation
  guards but before `connection_committed` was set — a window outside both the guards'
  own `close()` calls and `_cleanup_partial`'s coverage. Leaked on every failed attempt
  and every backoff retry; `shm_handle` is now tracked independently and closed in the
  outer `except` whenever the connection was never committed.
- **ABI/size guards used bare `assert`** (`cuda_runtime_types.py`), stripped under
  `python -O` — a ctypes struct-layout drift would import silently instead of failing
  loudly. Replaced with an explicit `_abi_guard()` helper that raises `RuntimeError`
  unconditionally, regardless of `-O`.
- **`example_sender_launcher.py` spawned a bare `"python"` subprocess**, relying on
  `PATH` order — inside a TouchDesigner Execute DAT this risks silently launching TD's
  own bundled interpreter instead of the sender's env. Now reuses the receiver
  launcher's resolver pattern (`CUDALINK_SENDER_PYTHON_EXE` env var → Windows `py -3`
  → `shutil.which("python")`), hardened further: if nothing resolves, `onStart()`
  prints a clear error and does not spawn, instead of risking an opaque `ImportError`
  in the wrong interpreter. (The resolver itself is Python 3.9-safe via
  `from __future__ import annotations`.)
- **`TDHost.param_value`/`is_active` caught only `AttributeError`**, unlike the sibling
  `find_top()` — broadened both to `(AttributeError, RuntimeError)` so a TD param in an
  error state fails safe instead of propagating on the per-frame path.

Two other findings from the same audit (partial-connect CUDA IPC handle leak; a
doorbell default disagreement) were investigated and confirmed to be false positives /
already handled — no code change needed.

## [1.12.0] - 2026-07-06

### Added

- **Runnable `examples/` directory.** 8 standalone, heavily-commented scripts
  (`examples/01_fake_adapter_tour.py` through `examples/08_low_cpu_doorbell_consumer.py`,
  plus shared `_common.py`) that each spawn their own demo producer process, so they run
  end-to-end on a single machine without TouchDesigner. These are the runnable counterparts
  of the embedded-code workflows in [docs/INTEGRATION_EXAMPLES.md](docs/INTEGRATION_EXAMPLES.md);
  see [examples/README.md](examples/README.md) for the reading order.
- **CUDA 13 runtime support in the cudart loader.**  `CUDARuntimeAPI._load_cuda_runtime()`
  now probes for `cudart64_13*.dll` alongside the existing CUDA 12.x candidates, with the
  Windows `winerror 126` ("module not found") hint message reordered so CUDA 13 is checked
  first — matching the actual probe order and avoiding a misleading hint on CUDA 13-only
  systems.
- **Native notification-wait backend for the consumer wait loop**
  (`ImportPolicy.wait_backend`: `"auto"` default | `"python"` | `"native"`; env
  `CUDALINK_WAIT_BACKEND`). Replaces the pure-Python spin/sleep poll in
  `Importer._wait_for_slot` with a native C++/pybind11 call that spins on `cudaEventQuery`
  then blocks on the doorbell's Win32 event; enabling it forces the doorbell open even when
  `CUDALINK_DOORBELL` is unset. Compiled into the core `cuda-link` wheel on Windows (via
  `scikit-build-core`), degrading to pure Python off Windows or without an MSVC toolchain — see
  [ADR-0006](docs/adr/0006-stay-pure-python-no-rust.md) and
  [ADR-0012](docs/adr/0012-native-extension-in-core-wheel.md). In benchmarking it did not
  measurably beat the plain doorbell (the dominant cost is the Windows kernel's own
  `WaitForSingleObject` wake floor), so it ships as an opt-in that never regresses against
  doorbell-only and is fully reversible (`CUDALINK_WAIT_BACKEND=python`).

### CI / Test hardening

- **Coverage gate**: `--cov` enforced in CI (`fail_under = 72`, branch-coverage-aware,
  baseline 74.65% measured 2026-06-29), with XML report upload to Codecov and a
  `coverage-main-*` artifact per Python version.
- **Randomized test ordering** (`pytest-randomly`) to catch order-dependent test bugs;
  verified order-independent across repeated randomized seeds.
- **Test suite reorg**: `--import-mode=importlib`, consolidated shared fakes under
  `tests/fakes/`, dropped `sys.path` hacks and from-conftest imports in favor of
  `pythonpath` (`pyproject.toml`) and `strict-markers`.
- **CI infra**: bumped `actions/setup-python`/`actions/upload-artifact` to their Node-24
  releases (v5→v6); removed the Gemini CI workflows entirely; refreshed the
  branch-protection `ALLOWED_DOCS` allowlist.

### Fixed

- **`bench_r1_wait.py`'s accept-gate measured the wrong metric.** The
  PASS/MISS check compared `get_frame()` total wall-clock time (spin + tensor
  materialization + Python overhead) against the native-waiter accept gate, which is
  defined on publish→detect notification latency — a metric this script
  never measures. Replaced the gate line with a detection-profile line
  (`avg_spin_us`/pacing-block breakdown) and a pointer to
  `bench_doorbell.py`'s new native arm as the authoritative venue.
- **`bench_doorbell.py`'s poll/doorbell arms did not pin `wait_backend`**,
  defaulting to `ImportPolicy`'s own `"auto"` — with the native extension
  available, the "doorbell" arm would silently engage the native backend,
  contaminating the poll-vs-doorbell comparison (same bug class as the
  `bench_r1_wait.py` fix above). Every arm now pins `wait_backend`
  explicitly.
- **`bench_r1_wait.py`'s per-arm frame budget was still insufficient after
  its initial fix.** `OVERHEAD_FRAMES=100` (~1.67s at 60fps) let 4/5 arms
  succeed but the 5th (cupy, last in sequence) still exhausted the shared
  producer's budget before its turn — real per-arm overhead (subprocess
  spawn + import + CUDA context init + SHM discovery) runs closer to 4–5s.
  Bumped to `overhead_frames=400`.
- **Native waiter: torn-frame race in the block phase's `write_idx`
  pre-check.** `native_waiter.cpp`'s block phase pre-checked the SHM
  `write_idx` before `cudaEventQuery` and returned success (`READY_LATE`,
  bucketed as a success by `importer.py`) as soon as `write_idx` advanced
  past the value captured at wait-start — treating "a newer frame landed"
  as equivalent to "this slot's own event fired." Under the Python
  `Exporter`'s async default (`export_sync=False`), `write_idx` can advance
  several frames ahead of a still-in-flight D2D copy on *this* slot, so a
  consumer could read a torn (still-copying) GPU buffer. Not reachable via
  the TD Sender (blocking export since v1.10.1 guarantees the copy is done
  before publish); reachable via the Python `Exporter`'s async default under
  GPU load. `cudaEventQuery` is now the only valid "ready" exit from the
  block phase — the doorbell wake (or, absent a doorbell, a bounded `Sleep`)
  is purely a hint to re-check sooner, never a substitute. `READY_LATE`
  stays in the status enum for ABI stability but is never emitted again.
  Added `READY_POLL` alongside (block phase caught it with no doorbell
  available, vs. `READY_DOORBELL` genuinely having one in play) — a related
  telemetry-accuracy fix bundled in since it's the same code region. New
  `tests/core/test_native_state_machine.py` drives the real C++ state machine
  with injected Python fakes (via a new test-only `wait_slot_test()` pybind
  entry point) to regression-test this, including the specific "`write_idx`
  advances while this slot's event stays not-ready" scenario the bug needed.
- **Prebuilt wheel distribution — end-user machines no longer need MSVC.** The
  ADR-0012 fold (earlier in this release) turned `cuda-link` into a compiled
  wheel but left an install-time gap: `install_td_library.py` silently
  auto-built via `utils\build_wheel.cmd` whenever no matching wheel was
  cached, and `CMakeLists.txt`'s `project(cuda_link LANGUAGES CXX)` forced
  compiler detection even for the `BUILD_NATIVE_WAITER=OFF` fallback — both
  defeating the "works everywhere" intent for StreamDiffusionTD artists'
  machines, which almost never have Visual Studio installed. Fixed: CI
  (`release.yml`, new `windows-latest` job) now builds and publishes both a
  native `cp311-cp311-win_amd64` wheel and a compiler-free `py3-none-any`
  fallback as GitHub Release assets; `install_td_library.py` resolves a wheel
  per install target (probing that target's own Python version, not the
  installer's) and only falls back to a local build behind an explicit
  `--build` flag; `CMakeLists.txt` now defers compiler detection to inside
  the `BUILD_NATIVE_WAITER` branch (`project(... LANGUAGES NONE)` +
  `enable_language(CXX)`), so the fallback build is genuinely compiler-free.
  Mode 5 (installing into TouchDesigner's own bundled Python) is now marked
  deprecated in the installer's menu, steering to mode 2 (venv) or mode 4
  (system Python) instead. See
  [ADR-0013](docs/adr/0013-prebuilt-wheel-distribution.md). This also
  corrects the `e57d1fc` fold commit's message, which implied no further
  MSVC-toolchain work was needed for end users — that assumption was wrong,
  and this entry is the correction of record (the pushed commit itself is
  not being rewritten).

## [1.11.0] - 2026-06-12

### Added

- **R2: Win32 named-event doorbell — opt-in kernel wake for subprocess receivers**
  (`CUDALINK_DOORBELL=1`, default OFF, Windows-only, single-consumer).  Replaces the
  1 ms poll-sleep in `Importer.wait_for_doorbell()` with a `WaitForSingleObject` on a
  Win32 auto-reset named event (`Local\cudalink_db_<shm_name>`).  Producer calls
  `_doorbell.signal()` (non-blocking `SetEvent`, ~µs) after every `publish_frame()`
  and once more on shutdown so the consumer wakes immediately on teardown.  Lost-wakeup
  safe: an early-out in `wait_for_doorbell()` checks `write_idx` before blocking, so
  a signal that arrives between the NO_FRAME check and the wait is never missed.
  Tested Python↔Python (`bench_doorbell.py`): CPU ~3-4% → ~0.3-1%, notify-latency
  p95 ~10× tighter.  TD-verified on the `TD>>Python` channel (TD Sender COMP →
  python receiver subprocess): doorbell fires 1:1 with frames at 60 FPS, latency
  0.02–0.10 ms (vs 0.04–1.80 ms poll baseline), cook-thread safe (no FPS dip from
  `SetEvent`), env propagates across TD's `subprocess.Popen` boundary, and teardown
  is clean (IPC handles closed in ~0.6 ms, no 2 s hang, no orphaned handle).
  The in-TD COMP receiver (`TDReceiver.py`) runs on the main cook thread and returns
  immediately on NO_FRAME — wiring the doorbell there is explicitly deferred.
- **R1: torch GPU-side wait — opt-in `cudaStreamWaitEvent` on the consumer GPU before
  tensor copy** (`CUDALINK_TORCH_GPU_WAIT=1`, default OFF).  Eliminates the
  host-side `cudaEventSynchronize` on the IPC producer event when torch is the frame
  mode, replacing it with a GPU-resident dependency edge.  Gains ~50–200 µs
  throughput per frame depending on WDDM batch timing; safe to enable alongside
  CUDA Graphs.
- **R1 adaptive auto-enable latch** (`CUDALINK_TORCH_GPU_WAIT_ADAPTIVE=1`).  Monitors
  per-frame event-wait time; if three consecutive frames exceed the heuristic
  threshold the latch fires and permanently enables GPU-side wait for the session —
  no manual `CUDALINK_TORCH_GPU_WAIT=1` required.  One-way: once enabled it stays
  enabled for the session lifetime.

### Performance

- **R3: pinned-memory f16 fallback.**  When `get_frame_numpy()` is called with a
  float16 frame and CUDA cannot allocate pinned (page-locked) host memory for the
  direct D2H path, the importer now falls back to a pageable staging copy rather
  than raising.  Avoids unexpected `RuntimeError` under memory pressure.
- **R4: hygiene — `graphs_deferred` flag and docstring cleanup.**  Internal
  `CUDAGraphsMixin` state tracking for deferred graph builds; no behavior change.

### Fixed

- **In-TD Sender COMP did not honor `CUDALINK_DOORBELL`.**  `TDSenderEngine.initialize()`
  built `ExportPolicy(...)` by hand without `doorbell=`, so the env switch had zero
  effect on the in-TD sender (`self._doorbell` stayed `None` → signal skipped →
  receiver always timed out).  Added `doorbell=env_bool("CUDALINK_DOORBELL", default=False)`.
- **`sync_td_wrapper.py` rewriter dropped all but the last stem on comma-separated
  `from . import A, B` lines.**  The regex captured `A, B` as one group and emitted a
  single `import B as A, B` statement.  Rewriter now splits on comma and emits one
  `import X as X` line per stem (black-compatible, avoids E702).
- **`sync_td_wrapper.py` emitted `E702` multi-statement lines for comma-import stems.**
  Companion fix: each stem gets its own physical line.
- **`_event_to_int` type error on `cupy` wait path.**  `c_void_p.value` is `int | None`
  (returns `None` for a null pointer); the fast-path `int(x.value)` raised `TypeError`
  on null handles.  Fixed with `x.value or 0` pattern; the residual `c_uint64`
  overload mismatch suppressed via pyrefly sub-config.
- **`bench_doorbell.py` summary line raised `UnicodeEncodeError` on cp1252 terminals.**
  Unicode `→` arrow replaced with ASCII `->`.

## [1.10.3] - 2026-06-11

### Fixed

- **`Exporter.open()` crashed on non-Windows platforms (CI red since 1.10.0).**
  `_read_hws_mode()` (P2 HWS warning) wrapped `import winreg` in `except OSError`,
  but a missing module raises `ModuleNotFoundError` (an `ImportError`) — so every
  `Exporter.open()` on Linux/macOS raised, and all 40 failures in the no-GPU CI
  suite traced to this line.  The handler now catches `(ImportError, OSError)` and
  degrades to `"unknown"`.  Locked by
  `tests/core/test_exporter_python.py::test_read_hws_mode_returns_unknown_without_winreg`.
- **Pipelined D2H left an async copy in flight across teardown/reconnect (latent
  CUDA 719).**  With `CUDALINK_D2H_PIPELINED=1` every `get_frame_numpy()` returns
  with one un-synchronized copy enqueued on the D2H stream.  `NumpyBuffers.close()`
  freed the pinned-host destination and `Importer._reinitialize()` closed the IPC
  source mapping without draining first.  New `NumpyBuffers.drain()` synchronizes
  the stream before either endpoint is released — the importer-side counterpart of
  the 1.10.1 sender-side STEP 1b fix.
- **Pipelined D2H surfaced a stale frame after producer restart.**  When the format
  was unchanged, `_reinitialize()` kept the double-buffer with `priming=False`, so
  the first frame of the new session was the dead session's last frame replayed as
  `NEW_FRAME`.  Reconnect now re-primes: the first call after `RECONNECTING`
  returns `NO_FRAME`, matching fresh-session behavior.  Both pipelined fixes are
  locked by `tests/integration/test_cpu_roundtrip.py::test_pipelined_drains_and_reprimes_on_reconnect`
  and new drain/rotate unit tests in `tests/core/test_pipelined_d2h.py`.
- **Malformed `CUDALINK_SENDER_REPORT_EVERY` / `CUDALINK_RECEIVER_REPORT_EVERY`
  crashed engine init.**  Both engines parsed the env var with a raw `int(...)`;
  they now use `Env.env_int()`, which falls back to the default (150) on
  non-numeric values.
- **TD Sender reported negative FPS and inflated export-avg after a geometry/dtype
  change.**  When `export_frame()` reopened the `Exporter` for a format change,
  the new exporter reset its `frame_count` to 0 while `ReportWindow.last_frame`
  retained the previous session's value (e.g. ~16 500).  The first post-change
  `Frame 150` report computed `(150 − 16500) / dt` → −1030.8 FPS, and divided the
  old session's cumulative export wall time by the tiny new count → 34 382.8 µs
  avg.  Both figures were observed in production after a uint8 → float32 live switch
  (v1.10.3 release candidate, confirmed by post-fix logs).  Fix:
  `_reset_report_window()` (already called in `cleanup()`) is now also called
  immediately after `Exporter.open()` in the format-change reopen block, resetting
  `ReportWindow` and the export-time accumulator so the first windowed report of the
  new session is always valid.  Data path unaffected; telemetry only.  Locked by
  `tests/td/test_tdsender_report.py::test_format_change_reset_prevents_negative_fps`
  and `::test_reset_report_window_clears_state`.

### Changed

- **Shared `ReportWindow` helper for windowed FPS reporting.**  The session-baseline
  / window seed / advance bookkeeping previously duplicated between
  `TDSenderEngine` (`_tx_*`) and `TDReceiverEngine` (`_rx_*`) now lives in
  `_profile.py` (`FrameProfile.py` in TD), used by both engines.
- `TDReceiverEngine.has_new_frame()` reads the SHM header via `SHMProtocol`'s
  precompiled-`Struct` helpers (`read_write_idx` / `read_version`) instead of
  per-call `struct.unpack_from` with duplicated offsets.
- `tests/td/test_td_config.py::test_export_sync_resolution_table` now tests the
  actual resolver (`TDSenderEngine._resolve_export_sync`) and documents the
  blocking-by-default mapping; its previous docstring still described the pre-1.10.1
  async default while asserting raw config storage.
- Removed a stale "implementation pending" comment on `ImportPolicy.d2h_pipelined`
  (the P5 implementation shipped in 1.10.2) and documented the post-reconnect
  re-priming contract in the `get_frame_numpy()` docstring.
- **Periodic `export=… µs avg` (Sender) and `copy=… µs avg` (Receiver) are now
  windowed, not lifetime.**  Both debug summary lines previously computed a
  lifetime cumulative average (`total_time / frame_count`), which converged
  slowly toward the session median and drifted visibly over long runs (e.g.
  export-avg climbing from 366 µs at frame 150 to ~990 µs at frame 5000 in a
  stable session).  The figure now reflects only the frames in the current report
  window (~150 frames), in sync with the windowed FPS printed beside it.
  Implementation: `ReportWindow` gained `add_sample(value)` and `avg_and_reset()`;
  `TDSenderEngine` and `TDReceiverEngine` dropped their bespoke `_export_total_s` /
  `_copy_total_s` lifetime accumulators in favour of the shared mechanism.  Locked
  by `tests/td/test_report_window.py` (7 cases) and
  `tests/td/test_tdsender_report.py::test_consecutive_reports_use_windowed_fps_and_avg`.

## [1.10.2] — 2026-06-11

### Performance

- **P11 — Receiver skips idle Script-TOP cooks.** `TDReceiverEngine.has_new_frame()`
  does a cheap SHM `write_idx` / version / shutdown probe before
  `import_buffer.cook(force=True)` in `callbacks_template.py:onFrameStart`.  When
  the producer has not written a new frame the cook is skipped entirely — one
  Script-TOP cook and all its Python overhead saved per idle TD frame.  Zero effect
  when producer ≥ TD rate; large effect for slow producers (e.g. 30 FPS AI inference
  into a 60 FPS TD project).  Covered by `tests/td/test_tdreceiver_has_new_frame.py`.
- **P8 — Fewer per-frame allocations on the import hot path.**
  `get_frame()` (`_TorchBackend`) and `get_frame_cupy()` (`_CupyBackend`) now cache
  their backend instance and update `_stream` in-place instead of constructing a new
  object on every call, matching the existing `_NumpyBackend` pattern.
  `AcquireResult` is now a `NamedTuple` instead of a `@dataclass`, eliminating the
  per-call `__dict__` allocation inside `acquire_slot` on the consumer hot path.
  Covered by `tests/core/test_backend_reuse.py`.

## [1.10.1] — 2026-06-10

### Fixed

- **TD Sender producer-side async source-buffer lifetime race (CUDA 719).** In 1.10.0,
  `export_frame()` defaulted to async export (`CUDALINK_EXPORT_SYNC` unset → async).  The
  TD source is TD's cook-scoped TOP texture (`cm.ptr`) — reclaimed the instant the cook
  returns.  Under a loaded consumer (e.g. SD's TRT inference saturating the GPU) the queued
  async D2D read executed *after* TD recycled the source, reading freed memory → sticky CUDA
  `cudaErrorLaunchFailure` (719), cascading through the consumer's CUDA context.
  **Root cause (confirmed by Loop A — `CUDALINK_EXPORT_SYNC=1` eliminated the crash):**
  Hypothesis #2 — producer-side async source-buffer lifetime race.  `record_source_sync` /
  `_arm_same_stream_ordering` is a *pre-copy ordering* primitive only (ensures the source is
  fully written before the copy *starts*) — it does **not** guarantee the source outlives the
  queued async read.  The only mechanism that closes that window is a post-copy
  `stream_synchronize(ipc_stream)`.
  **Fix:** The **TD Sender now blocks by default** — `CUDALINK_EXPORT_SYNC` unset (or `None`)
  resolves to blocking, matching the 1.9.0 behavior.  Explicit `CUDALINK_EXPORT_SYNC=0` opts
  back into async for callers that own a persistent source buffer with explicit lifetime
  management.  The async default introduced in 1.10.0 is correct for the Python `Exporter`
  API (caller owns a stable buffer + passes `producer_stream`) but unsafe for the TD Sender
  whose source is externally-owned and transient.
  Regression seam: `TDSenderEngine._resolve_export_sync` (GPU-free, testable static method);
  locked by `tests/td/test_tdsender_export_sync_default.py`.
- **`Exporter.close()` races in-flight async D2D copies (latent CUDA 719).**
  `_do_cleanup()` previously destroyed IPC events/stream and freed GPU ring-buffer slots
  without first synchronizing the IPC stream.  An async export returning before the D2D
  completes (geometry/dtype-change reopen, or any explicit `exporter.close()` call) could
  free the destination buffers under a running copy.  Fix: a new STEP 1b in `_do_cleanup`
  issues `stream_synchronize(ipc_stream)` before any teardown of GPU resources.

## [1.10.0] — 2026-06-10

### Added

- **Sender-side periodic stats log** (`CUDALINK_SENDER_REPORT_EVERY`, default 150) —
  mirrors the receiver's Debug summary line; prints `Frame N | FPS | shape | dtype |
  export=… µs avg (write_idx=…)` to the Textport every N published frames when the
  Sender COMP's **Debug** parameter is ON. Uses engine-local wall-clock timers,
  independent of `CUDALINK_EXPORT_PROFILE`.
- **Opt-in pipelined device-to-host copy** (`ImportPolicy.d2h_pipelined` /
  `CUDALINK_D2H_PIPELINED`, default off) — double-buffers the D2H copy so the
  next frame's transfer overlaps the consumer's compute on the current frame.
  Wins only when consumer work exceeds copy time (1080p ~6% / −305 µs, 4K
  ~20% / −1235 µs at 5 ms consumer work; ~0% at 512²). First `get_frame` returns
  a priming `NO_FRAME`; steady-state is +1-frame latency. New standalone harness
  `scripts/profiling/bench_d2h_pipelined.py` (spawn producer/consumer, sweeps
  512²/1080p/4K, nsys-verifies D2H↔CPU overlap).
- **pyrefly type-checking extended to `td_exporter/`** (ADR-0005 follow-up) —
  `pyrefly check` (project mode) now covers the td_exporter engine files
  (`TDHost`, `TDSender`, `TDReceiver`, `TDConfig`, `CUDAIPCExtension`,
  `CUDALinkBootstrap`, `benchmark_timestamp`, `_td_fakes`) and all 14
  auto-generated mirrors. Pure-TD glue scripts (callbacks, example launchers)
  are excluded. The `td` module is handled via `replace-imports-with-any`;
  bare ambient TD globals (`op`, `run`, `CUDAMemoryShape`) are resolved through
  `if TYPE_CHECKING: from _td_builtins import …` stubs. CI gate (`typecheck.yml`)
  updated to project mode and `td_exporter/**` added to path triggers.
- **Static type-checking CI gate** (`.github/workflows/typecheck.yml`) — runs
  `pyrefly check` (project mode: `src/cuda_link/` + `td_exporter/`) on every
  PR/push touching the package or its config. Previously pyrefly ran only as a
  skippable local pre-commit hook checking `src/cuda_link/` only.
- **Pytest CI job** (`.github/workflows/tests.yml`) — runs the `not requires_cuda`
  suite on Python 3.10–3.12. The suite is green with or without torch installed
  (torch-dependent tests use `pytest.importorskip`), so torch is a best-effort
  CPU install. Previously there was no CI test workflow at all.
- **GPU-free Protocol drift guard** (`tests/core/test_exporter_port.py::test_ctypes_adapter_api_covers_protocol_without_gpu`)
  — asserts `CUDARuntimeAPI` implements every `CudaPort` member. The
  `CTypesCUDAAdapter.__getattr__` delegation otherwise hides such drift from
  both the (suppressed) type checker and the `requires_cuda` conformance test.
- **ADR-0006** (`docs/adr/0006-stay-pure-python-no-rust.md`) — records the
  decision to keep cuda-link pure Python rather than rewrite in Rust
  (`cuda-oxide`/`cudarc`), with the performance, TD-embedded-CPython, and
  deployment evidence, so the question is not re-litigated from scratch.
- **ADR-0005** (`docs/adr/0005-static-typing-hardening.md`) — records the scoped
  type-suppression policy and the "no package-wide category blanket" rule.

### Changed

- **`export_frame()` is async end-to-end by default.** The producer no longer
  issues a per-frame blocking `cudaStreamSynchronize`; the IPC event + async
  flush-probe carry ordering. Blocking export is opt-in via
  `CUDALINK_EXPORT_SYNC=1`. Coexistence safety (TD Sender + Receiver in one
  process) comes from explicit, persistent per-engine streams plus
  producer-stream ordering (`record_source_sync` / `require_source_sync`), not
  from blocking export — now documented in `docs/PROFILING.md §8` and
  `docs/ARCHITECTURE.md`.
- **CUDA graph folds `stream_wait_event` + `record_event` into the captured
  graph** — the per-frame export now issues a single WDDM submission
  (`cudaGraphLaunch`) instead of three.
- **`[tool.pyrefly]` hardened from one package-wide suppression blanket to
  precise per-file scoping.** Added `python-platform = "win32"` (Windows-only
  project — resolves `ctypes.windll`/`WINFUNCTYPE` on any host) and
  `replace-imports-with-any` for the optional GPU deps (torch/cupy/pynvml/nvtx/
  ml_dtypes). The 9-category blanket over all of `src/cuda_link/**` is replaced
  by per-file `sub-config` blocks scoped to the modules with accepted ctypes /
  optional-import idioms; every other module (`shm_protocol.py`, the `_port`
  Protocols, `_env`, `_profile`, `_console`, `cuda_runtime_types.py`, …) is now
  fully type-checked.

### Fixed

- **`FakeCUDAAdapter` pointer coalescing** (`_cuda_adapters.py`) — `c_void_p.value`
  (`int | None`) is now coalesced to `int` before dict/list operations, removing
  a latent `None`-key path in the test fake and letting the module type-check
  cleanly.
- **Torch-fragile tests made CI-safe** — `test_torch_buffers_int8/int16_succeeds`
  now use `pytest.importorskip("torch")`, and the stale `test_get_frame_without_torch`
  (which asserted a removed "torch is required" contract via the unconnected
  deprecated `CUDAIPCImporter`) was modernized to assert the current graceful
  `None` return and renamed `test_get_frame_on_unconnected_importer_returns_none`.

### Removed

- **Dead export-sync receiver-topology auto-select branch** and the process-local
  receiver registry in `td_exporter/CUDAIPCExtension.py`
  (`_active_receiver_count` / `_register_receiver` / `_unregister_receiver` /
  `receiver_active_in_process`). The tri-state `export_sync=None` "auto-select
  blocking when a receiver coexists in-process" path was structurally unreachable
  in the standard multi-COMP `.toe`: each COMP compiles its own
  `CUDAIPCExtension` module, so the counter is per-COMP-isolated and a sender's
  copy is always 0. `None` now simply means async (the validated production
  default). Two registry unit tests removed; the export-sync resolution table
  test rewritten to the two-state (`True` → blocking, `None`/`False` → async)
  contract.

### Performance

- **Async export −24.7 µs p50 (59%)** @ 1080p uint8 (42.1 → 17.4 µs) — eliminates
  the producer-side `cudaStreamSynchronize`.
- **CUDA graphs −3.8 µs p50 (22%)** (17.4 → 13.6 µs) — 3 WDDM submissions/frame → 1.
- **Combined async+graphs −28.5 µs p50 (68%)** @ 1080p (42.1 → 13.6 µs).
- **Pipelined D2H** (opt-in) overlaps copy with consumer work: 1080p ~6%, 4K ~20%
  when `work > copy`.

## [1.9.0] — 2026-06-06

### Added

- **Unordered-export safeguard** — `Exporter.export()` now warns (once per instance) when no
  producer-stream ordering has been armed via `GpuFrame.producer_stream` or
  `record_source_sync()`. The D2D memcpy on the non-blocking high-priority IPC stream is silently
  unordered without this, which can produce torn/gray frames downstream (confirmed by a real
  StreamDiffusion → TouchDesigner regression). Set `ExportPolicy(require_source_sync=True)` /
  `CUDALINK_REQUIRE_SOURCE_SYNC=1` to raise `ValueError` instead of warning.

- **Env-var documentation sweep** — ten `CUDALINK_*` variables that were used in `src/cuda_link`
  but absent from the README Performance Tuning table are now documented: `REQUIRE_SOURCE_SYNC`,
  `STRICT_DEVICE`, `LIB_STREAM_PRIO`, `BARRIER_STALE_NS`, `WAIT_SPIN_US`, `D2H_STREAM_PRIO`,
  `ALLOW_PAGEABLE_FALLBACK`, `IMPORT_RECONNECT`, `IMPORT_RECONNECT_MAX_ATTEMPTS`,
  `STICKY_ERROR_CHECK`.

- **Env-var docs invariant test** (`tests/support/test_env_var_docs.py`) — asserts every
  `CUDALINK_*` variable referenced by `env_bool`/`env_int`/`env_str` in `src/cuda_link/*.py`
  appears in the README Performance Tuning section. Prevents silent drift going forward.

- **Producer-stream ordering documentation** — new "Producer-Stream Ordering" subsection in
  `docs/ARCHITECTURE.md`; `producer_stream` added to all code snippets in `README.md`,
  `docs/INTEGRATION_EXAMPLES.md`, and `td_exporter/HELP_DOC.md`.

### Fixed

- **All in-repo `export()` call sites now declare producer-stream ordering**:
  - `td_exporter/example_sender_python.py` — passes `producer_stream=0` (synchronous H2D fill
    is already complete on the host; stream 0 declares ordering explicitly).
  - `td_exporter/TDSender.py` — calls `record_source_sync(ipc_stream)` once at `Exporter.open()`
    and on each format-change reopen; both `cuda_memory()` calls already pass
    `stream=ipc_stream.value`, so same-stream FIFO ordering is the correct declaration.
  - `scripts/profiling/profile_export.py` — calls `record_source_sync(0)` once before the timing
    loop so the measurement includes the per-frame `stream_wait_event` overhead (representative
    of real ordered usage).

### Internal

- **Unified `_ordering_armed` predicate** — both the CUDA Graphs path and the legacy stream path
  now gate `stream_wait_event` on a single `_ordering_armed` property
  (`_source_sync_recorded and source_sync_event is not None`). Eliminates a redundant wait on a
  never-recorded event in the legacy path when ordering was never armed.

## [1.8.1] — 2026-06-01

### Added

- **Debug timing summary in Receiver mode** (`25a0514`, `fc45578`) — `TDReceiverEngine` now
  emits a full per-frame timing line to the Textport when **Debug** is ON, every 150 frames
  (configurable via `CUDALINK_RECEIVER_REPORT_EVERY`). Example output:
  ```
  [CUDAIPCExtension:Receiver] Frame  150 |  60.4 FPS | shape=(1080, 1920, 4) dtype=uint8 | latency=10.09 ms | copy=129.2 µs avg (slot=2, write_idx=231)
  ```
  Fields: **FPS** (frames consumed ÷ wall time since first frame); **latency** (`now −
  producer_timestamp` — valid on single-machine TD→TD / TD→Python setups); **copy** (running
  average of `copyCUDAMemory` wall time — the receiver's analogue of `get_frame= µs avg` in the
  standalone Python receiver); **shape** (numpy H×W×C order). Output stops when Debug is toggled
  Off. Format matches `td_exporter/example_receiver_python.py`.

### Internal

- **Test suite reorganized into subfolders** (`d319703`) — the flat `tests/` directory of 37
  `test_*.py` files is now grouped into five purpose-named packages: `core/` (12 — protocol layer),
  `cuda/` (6 — CUDA runtime seam), `td/` (11 — TouchDesigner integration), `integration/` (5 —
  end-to-end pipeline), `support/` (3 — tooling). Each subfolder is a Python package
  (`__init__.py`). History preserved via `git mv`. 11 files that computed the repo root via
  `Path(__file__).parent.parent` were deepened by one level to
  `Path(__file__).parent.parent.parent`. `README.md` testing section and ADR cross-references
  updated to reflect new paths. `pytest --collect-only` count unchanged: 667 before and after.
- **Gemini CI — headless workspace trust** (`eb30d76`) — added `GEMINI_CLI_TRUST_WORKSPACE: 'true'`
  to all four `run-gemini-cli` steps (`gemini-review`, `gemini-invoke`, `gemini-triage`,
  `gemini-scheduled-triage`). Without it, Gemini CLI exits with "not running in a trusted directory"
  in every CI run. The dispatch fork-gate (`head.repo.fork == false`) and `GITHUB_TOKEN: ''` in the
  triage steps remain as the primary trust controls.
- **Missing `requires_cuda` mark** (`f122d88`) — `test_connect_with_nonexistent_shm_enters_waiting_state`
  in `tests/integration/test_cuda_ipc_importer.py` was missing `@pytest.mark.requires_cuda` despite
  calling `connect()`, which loads the CUDA runtime. The CI filter `-m "not requires_cuda"` did not
  exclude it, causing a pre-existing failure on every Linux runner run.
- **GitHub Actions — Node 24 upgrade** (`2ebe90c`) — `actions/checkout@v4` → `v6` (all six
  non-Gemini workflows) and `actions/setup-python@v5` → `v6`; both `v6` releases use Node 24 and
  resolve the Node 20 deprecation warning active since 2026-06-16. Type checker in the lint job
  swapped from `mypy` to `pyrefly` — mypy produced 140 false positives on this codebase (Windows
  ctypes, optional deps) that pyrefly handles correctly via the project config.

## [1.8.0] — 2026-05-31

### Fixed

- **`DtypeCodec.decode` mono+alpha itemsize** (`0526096`) — `FLAGS_MONO_ALPHA` (bit 1) is a
  channel-semantic flag, not a dtype selector. The old `_DECODE_TABLE` lookup keyed on the exact
  `(kind, bits, flags)` triple caused `monoalpha8fixed` → `"float32"` fallback (4× oversized
  itemsize) and `monoalpha16fixed` → 2× oversized. Fix: mask `flags & FLAGS_BFLOAT16` before the
  lookup so only the dtype-relevant bfloat16 bit participates. Also repairs the same latent
  mis-decode in the Direction-A `Importer.Format.from_shm` path.
- **`TDReceiverEngine` — `CUDAMemoryShape.dataType` scalar type** (`3c4f54c`) — `Format.numpy_dtype`
  is a `np.dtype` instance (e.g. `np.dtype('uint8')`), but `CUDAMemoryShape.dataType` requires the
  numpy scalar *type* (`numpy.uint8`). Fixed with `.type` at both `initialize_receiver` and
  `_refresh_on_version_change` call sites. Without this, every non-float32 format (uint8, uint16)
  crashed with `td.tdError: dataType must be numpy.uint8, numpy.uint16, numpy.float32` on init.
- **`TDReceiverEngine` — SHM exported-pointer loop** (`3c4f54c`) — `_shm_mv = memoryview(shm_handle.buf)`
  in the new metadata-read path held a mmap exported pointer as a named local. When the dtype error
  above propagated up, Python kept the `initialize_receiver` frame alive in the exception traceback
  across cook cycles, blocking `shm_handle.close()` with `BufferError: cannot close exported pointers
  exist` and locking the receiver into a crash loop. Fixed by snapshotting the SHM as `bytes` first
  (`memoryview(bytes(shm_handle.buf))`).
- **`TDSenderEngine.dev_ptrs` post-cleanup** (`9be8212`) — after `cleanup()`, `dev_ptrs` and
  `ipc_handles` now return `[]` (empty) rather than `[None] * num_slots` (null stubs). A new
  `_closed: bool` flag tracks post-cleanup state so the pre-init stub (`[None] * num_slots`) and
  post-cleanup state (`[]`) are correctly distinguished. Resolved `test_sender_cleanup_resets_all_state`.
- **CUDA DLL load error diagnostics** (`faf55cc`) — `_load_cuda_runtime` now captures the last
  `OSError` from each search tier (absolute-path and bare-name) and surfaces the `winerror` code
  in the final `RuntimeError`. When `winerror == 126` ("module not found"), a targeted hint directs
  the user to run `dumpbin /dependents` or `Dependencies.exe` to identify the missing *dependent*
  DLL, since WinError 126 does not identify cudart itself.
- **`Importer.attach_nvml_observer` — permanently-dead NVML stats branch** (`e9f22bc`) — the NVML
  stats reporting branch in the frame-receive hot path was unreachable due to a missing call site.
  Root cause: `Importer` had no method to attach a live `NVMLObserver` after construction. Fix adds
  `Importer.attach_nvml_observer(observer)` (public) and wires the stats branch correctly.

### Added

- **`Metadata.expected_size`** (`0526096`) — new pure property `width * height * num_comps *
  (bits_per_comp // 8)` on the `Metadata` frozen dataclass. Single authoritative home for the
  frame-size invariant that was previously duplicated in `TDReceiver.initialize_receiver` and
  `_refresh_on_version_change`.
- **`Format.from_metadata(cls, md)`** (`831031a`) — new constructor on the `Format` value object
  in `importer.py` that builds a `Format` from an already-read `Metadata`, avoiding a second SHM
  read. `Format.from_shm` now delegates to it.
- **`TDSenderEngine.frame_count` property** (`9be8212`) — exposes the underlying
  `Exporter.frame_count` (0 when no exporter is active).
- **`Importer.attach_nvml_observer(observer)`** (`e9f22bc`) — new public method to bind a live
  `NVMLObserver` to a connected `Importer` after construction, enabling per-frame GPU telemetry
  without rebuilding the importer.
- **`test_every_bound_func_has_argtypes`** (`faf55cc`) — new test in
  `tests/test_errcheck_coverage.py` asserts that every explicitly-bound cudart function has
  `argtypes` set. Without `argtypes`, pointer arguments are silently truncated to 32-bit int on
  64-bit platforms (ctypes reference). 23 new parametrized decode-matrix cases added to
  `tests/test_shm_protocol.py`.
- **`test_every_errcheck_func_is_typed`** (`123e518`) — reverse-direction errcheck coverage guard:
  every function in `_strict_funcs` must also have `restype=c_int` in
  `_setup_function_signatures`. Closes the mirror-image drift path (errcheck'd but untyped →
  silent 64-bit pointer truncation).

### Changed

- **Default IPC channel names** (`78bd051`) — `Ipcmemname` defaults renamed for directional
  clarity: `cudalink_output_ipc` → `cudalink_ipc_TD>>Python` (Sender mode, TD-to-Python);
  `cudalink_input_ipc` → `cudalink_ipc_Python>>TD` (Receiver mode, Python-to-TD). Only projects
  relying on the hardcoded defaults need updating; explicit names set via the `Ipcmemname`
  parameter are unaffected.

### Refactored

- **`TDReceiver` frame-format unified on `Format`** (`831031a`) — `FormatDescriptor` (a
  hand-rolled near-duplicate of `importer.Format`) removed from `TDReceiver.py`. Both the
  Direction-A `Importer` and the Direction-B `TDReceiverEngine` now share one frozen value object
  for parsed frame format. Two hand-rolled `struct.unpack` + dtype-ladder blocks replaced with
  `Metadata.read_from + Format.from_metadata`. `_to_td_pixel_format` renamed to
  `td_format_string(fmt: Format)`; `FLAGS_MONO_ALPHA` preserved so `monoalpha*` par.format strings
  and Info/Status messages are unchanged. No new `.tox` DAT: `Importer` and `SHMProtocol` already
  ship in both deployment modes.
- **Honest `CudaPort` seam, named CUDA enums, `SHMLayout` handle split** (`787853a`) — C3
  refactor: `CudaPort` protocol now exposes only the operations `Exporter` genuinely needs;
  `cudaMemcpyKind` and related enums promoted to named constants; `SHMLayout` handle-offset logic
  extracted to a dedicated helper.
- **`TDReceiverConfig` honest type** (`7732c36`) — C4 refactor: `TDReceiverEngine` config
  dataclass fields typed to their actual domain types (not bare `Any`), surfacing latent type
  errors at definition time rather than at runtime.
- **`_release_connection_state()` extracted** (`d607eb2`) — C2 slice: deduplicates the importer
  teardown sequence (previously duplicated across the normal-close and error-recovery paths) into
  one method.
- **TD-global leak closed behind `TDHost` seam** (`f6e294d`) — C5 refactor: TD runtime globals
  accessed directly by engine code now route through the `TDHost` protocol; `make_cuda_shape`
  factory added.
- **`make_bare_runtime_api()` shared test factory** (`af1c610`) — C6 refactor: GPU-free
  `CUDARuntimeAPI` construction factored into `tests/fakes.py` so multiple test modules share one
  canonical factory instead of duplicating the setup.

### Docs

- **ADR-0004 — Legacy CUDA Runtime IPC over VMM driver API** (`a08f864`) — records why
  `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` is used instead of the VMM driver API; prevents
  future re-investigation of a validated, empirically-settled decision.
- **ADR index** (`c4a2e16`) — `docs/adr/README.md` added; dead `MIGRATION_v1.5.md` cross-links
  removed from `ARCHITECTURE.md`, `INTEGRATION_EXAMPLES.md`, and `MIGRATION_v1.6.md`; ADR-0003
  and ADR-0004 added to `CONTEXT.md` architecture-reference list.
- **IPC handle docstring corrected** (`75fd590`) — `ipc_get_mem_handle` docstring said
  "128 bytes"; corrected to 64 bytes (the actual `cudaIpcMemHandle_t = c_byte * 64` size; 128 was
  the total SHM slot, not the handle).

### Internal

- `pyproject.toml` version: `1.7.2 → 1.8.0`.
- `build_wheel.cmd` moved from repo root to `utils/` (`0c4c8d3`); `scripts/install_td_library.py`
  and `README.md` updated. Use `utils\build_wheel.cmd` (or double-click from `utils\`).
- TOX artifact `TOXES/CUDAIPCLink_v1.8.0.tox` — built and saved separately by the user.

## [1.7.2] — 2026-05-30

### Fixed

- **Sender dtype switch (float32→uint8)** — Producer now correctly detects dtype-shrink transitions
  via `pixelFormatName` and copies only the dtype-derived front region of the stale allocation
  (`GpuFrame(size=spec.data_size)`), resolving permanent stop-send + frozen Status after a
  float32→8-bit switch. The previous 5-commit chain (`cddfdae`–`70da086`) introduced a permanent
  skip-export trap; this was corrected in `e724719` using the proven v1.5.1 copy-front-region approach.
- **Consumer ImportBuffer format for float32 non-RGBA sources** — `needs_format_update` flag now
  fires for all non-`rgba32float` float32 sources (e.g. `monoalpha32float`, `rg32float`). Previously
  skipped for all float32 sources, causing Script TOP to stay at the default `rgba32float` allocation
  even when the source was a different float32 variant (`e31bad6`).

### Added

- **`FLAGS_MONO_ALPHA = 0x0002`** wire metadata bit — distinguishes 2-channel mono+alpha sources
  from genuine RG sources in the SHM metadata `flags` field. Fits in 15 spare bits of the existing
  uint16 `flags` field; `METADATA_SIZE` unchanged at 20 bytes. Backward-compatible: old Consumers
  see `flags=0` and fall back to `rg*` format (`54cb7df`).
- **`FrameSpec.extra_flags`** — new `int = 0` field on `FrameSpec` (both `ExporterPort.py` and
  `_exporter_port.py`) lets callers OR extra protocol flags into `Exporter._write_metadata_to_shm`.
- **TD-native format names in Status and `par.format`** — both Producer Status and Consumer Status
  now show the TD pixel-format menu name (e.g. `rgba8fixed`, `monoalpha32float`) rather than the
  numpy dtype + channel-count string.
- **`cudaMemory(pixelFormat=)` kwarg infrastructure** — `RealTOPHandle.cuda_memory()` now accepts
  an optional `pixel_format` string and forwards it as the `pixelFormat=` keyword to TD's
  `cudaMemory()`, enabling future format-conversion requests (`22b9022`).

### Internal

- **`pyproject.toml` version**: `1.7.1` → `1.7.2`.
- **TOX artifact**: `TOXES/CUDAIPCLink_v1.7.2.tox` to be built separately.
  `v1.7.1` retained per versioned-binary tracking policy.

## [1.7.1] — 2026-05-30

### Refactored

- **ctypes robustness sweep** (`a2baea9`) — eight correctness and safeguard
  improvements cross-checked against the Python ctypes reference documentation:
  - ABI size asserts for all graph param structs (`cudaPos==24`,
    `cudaPitchedPtr==32`, `cudaExtent==24`, `cudaMemcpy3DParms==160`) catch
    layout drift at import time.
  - `NumpyBuffers.close()`: null `self.buffer` after `free_host` to prevent
    use-after-free on the CUDA-pinned host-memory alias.
  - `_log_dll_path`: consume `ctypes.get_last_error()` on `GetModuleFileNameW`
    failure; guard `dll._handle` with `getattr` for forward-compatibility.
  - `_HighResTimer.__enter__`: check `timeBeginPeriod(1)` return code; log on
    `TIMERR_NOCANDO`.
  - `_winmm` loader: removed unused `use_last_error=True` (winmm timer APIs
    return status directly, not via `GetLastError`).
  - Removed dead `cudaGetLastError` argtypes binding; `cudaPeekAtLastError` is
    the correct non-destructive sticky-error read.
  - `probe_producer`: decode `cudaGetErrorString` bytes → str (was embedding
    `b'...'` repr in error messages).
  - `_load_cuda_runtime`: documented `winmode` asymmetry between absolute-path
    and bare-name DLL-load tiers.

### Tests

- **`tests/test_errcheck_coverage.py`** (`a2baea9`, 3 tests): drift-prevention
  guard — every `c_int`-restype cudart function has `errcheck` installed;
  `cudaGetLastError` asserted unbound.
- **`tests/test_numpybuffers_close.py`** (`8b74ab0`, 5 tests): `NumpyBuffers.close()`
  UAF guard — pinned path nulls `buffer`, pageable path preserves it, idempotency,
  and `host_unregister` branch.
- **`tests/test_probe_error_decode.py`** (`8b74ab0`, 2 tests): `probe_producer`
  error-string decode — verifies str in message and `None` guard for NULL returns.

### Internal

- **`pyproject.toml` version**: `1.7.0` → `1.7.1`.
- **TOX artifact**: `TOXES/CUDAIPCLink_v1.7.1.tox` to be built separately.
  `v1.7.0` retained per versioned-binary tracking policy.

## [1.7.0] — 2026-05-29

### Added

- **Library-mode bootstrap (`CUDALinkBootstrap`)** — `td_exporter/CUDALinkBootstrap.py`
  injects `CUDALINK_LIB_PATH` onto `sys.path` and registers all 14 mirror module names
  in `sys.modules` as aliases to the installed `cuda_link` submodules. When active the
  `.tox` needs only 6 core Text DATs (instead of 20). Falls back silently if the package
  is not installed. Must be the first Text DAT in the COMP.

- **Multi-target installer** (`scripts/install_td_library.py`, `install_td_library.cmd`)
  — 5 install modes: (1) system Python site-packages, (2) user site-packages, (3) conda
  active env, (4) TD-Preferences–managed Python path, (5) custom target directory.
  Mode 4 auto-sets the installed path in TD's Python site-packages preference so library
  mode activates immediately on the next TD launch without any env-var configuration.

- **Auto-discover system Python and TouchDesigner installations** in installer modes 4
  and 5 — the installer probes registry keys and well-known install paths to locate the
  Python interpreter registered by the Windows Python Launcher (`py -3`) and TouchDesigner
  (`HKCU\Software\Derivative\TouchDesigner\InstallDir`) without requiring manual path input.

- **TD startup pop-up** — first-run setup dialog shown at TouchDesigner startup when
  `cuda_link` is not yet installed in the active Python environment.

- **ADR-0003**: `docs/adr/0003-library-install-sys-path-bootstrap.md` documents the
  library-mode bootstrap design decisions and trade-offs.

- **`Format.layout_differs_from`** — new predicate on the `Format` value object that
  returns `True` only when the resolved shape or dtype has changed, letting receivers
  avoid a full teardown/reinit for metadata-only updates. Tests: `test_format_layout.py`.

- **`MIGRATION_v1.6.md`** (`docs/MIGRATION_v1.6.md`) — migration guide for callers
  upgrading from v1.5.x to v1.6.x API changes.

### Changed

- **Folded `warning_emitter_callbacks.py` into `script_top_callbacks.py`** — both Script
  TOPs inside the `.tox` (`ImportBuffer` and `warning_emitter`) now share one Callbacks
  DAT. `onCook` dispatches by `scriptOp.name`; behaviour is unchanged. Removes one source
  file and one DAT from the component.

- **`example_receiver_python.py` crash-logging wrapper** — unhandled exceptions now write
  a full traceback to `receiver_error.log` next to the script, print it to stdout, and
  pause with "Press Enter to close this window" before exiting — so errors are never lost
  when the subprocess window closes automatically.

- **`CUDAAdapters` / `CTypesCUDAAdapter` / `FakeCUDAAdapter` renamed** for naming
  consistency (`CudaAdapters` → `CUDAAdapters`, `CTypesCudaAdapter` → `CTypesCUDAAdapter`,
  `FakeCudaAdapter` → `FakeCUDAAdapter`). The `td_exporter/` mirror (`CUDAAdapters.py`)
  and the `_ALIAS_MAP` entry in `CUDALinkBootstrap` are updated. Out-of-tree code that
  imported the old names must use the new names.

- **`get_frame*` consumers unified behind `_FrameBackend` seam** in `Importer` — the
  torch / numpy / cupy dispatch logic is concentrated in one place. Behaviour is unchanged;
  this is an internal refactor that eliminates duplicate per-backend branches.

- **`PAIRS` is now the single source of truth** for TD mirror registration in
  `scripts/sync_td_wrapper.py`. `CUDALinkBootstrap._ALIAS_MAP` is verified against
  `PAIRS` at test time (`tests/test_td_bootstrap.py::test_alias_map_covers_all_pairs`).

- **Example `.toe` artifacts** — `CUDA_Link_Example.85.toe` added, `v81` retired.

### Fixed

- **`cudaIpcMemHandle_t`/`cudaIpcEventHandle_t` class-identity mismatch** in
  `CUDARuntimeAPI.ipc_open_mem_handle` and `ipc_open_event_handle` — ctypes validates
  `Structure` arguments by class identity, not structural equivalence. In TD's bare-name
  import namespace two independent imports of `cuda_link.cuda_runtime_types` (e.g.
  library-mode package + a leftover mirror Text DAT) create two distinct class objects;
  handles built with one were rejected by `argtypes` bound to the other, raising
  `ArgumentError: expected cudaIpcMemHandle_t instance instead of cudaIpcMemHandle_t`
  every cook. Fixed by adding an `isinstance` guard that calls
  `cudaIpcMemHandle_t.from_buffer_copy(bytes(handle))` to normalize the handle into the
  wrapper's own class before the cudart call. Covers both the TD→TD and TD→Python paths.
  Regression test: `tests/test_ipc_open_mem_handle_guard.py` (6 GPU-free cases).

- **`CUDALinkBootstrap` import guarded** in `CUDAIPCExtension.py` —
  `import CUDALinkBootstrap` is now wrapped in `contextlib.suppress(ImportError)` so the
  extension loads cleanly in environments where the bootstrap module is absent (e.g.
  classic-mode comps that pre-date the bootstrap DAT).

- **`TDSenderEngine.export_frame` lazy auto-init restored** — the deferred
  `_lazy_init_on_first_frame()` call was accidentally removed in commit `8eebda5`; the
  sender now correctly defers CUDA allocation until the first frame export.

- **`build_wheel.cmd` blocking pause suppressed** when called as a subprocess by the
  installer — `PAUSE` is now skipped when `CI` or `CUDALINK_NO_PAUSE` is set, preventing
  the installer from hanging.

- **Site-packages detection corrected** in installer modes 2/3/4; activation simplified
  to write directly to TD Preferences instead of intermediate env-var files.

- **`python.exe` auto-resolved** from directory paths — installer modes 4/5 now append
  `python.exe` when the resolved path points to a directory rather than an executable.

- **Indented inline relative imports** now correctly rewritten by `sync_td_wrapper.py`
  (previously only top-level `from .X import` lines were matched).

- **`_console.py` registered in `CANONICAL_ONLY`** in `sync_td_wrapper.py` — the module
  was added in `eb3c8d3` but not registered; the sync script now correctly skips it when
  generating TD mirror stubs.

### Removed

- `td_exporter/warning_emitter_callbacks.py` — folded into `script_top_callbacks.py`.
- `docs/ctypes-audit-v1.5.md`, `docs/MIGRATION_v1.5.md`,
  `docs/perf/graphs-benchmark-v1.5.md`, `docs/perf/nsight-v1.5.md`,
  `docs/agents/domain.md`, `docs/agents/issue-tracker.md`,
  `docs/agents/triage-labels.md` — stale v1.5 artifacts cleaned up.

### Tests

- `tests/test_td_bootstrap.py` — 15 cases covering library-mode bootstrap alias
  registration, fallback mode, and alias-map / PAIRS sync guard.
- `tests/test_format_layout.py` — `Format.layout_differs_from` coverage.
- `tests/test_ipc_open_mem_handle_guard.py` — 6 GPU-free regression tests for the
  class-identity guard in `ipc_open_mem_handle` and `ipc_open_event_handle`.

### Internal

- **`pyproject.toml` version**: `1.6.0` → `1.7.0`.
- **TOX artifact**: `TOXES/CUDAIPCLink_v1.7.0.tox` to be built separately.
  `v1.6.0` retained per versioned-binary tracking policy.

## [1.6.0] — 2026-05-29

### Changed

- **`CUDALINK_EXPORT_SYNC` default `1` → `0` (sync-free by default)** — The per-frame
  `cudaStreamSynchronize` on the IPC stream was redundant: the CUDA IPC event recorded
  at export time already provides correct cross-process GPU ordering on the consumer.
  The CPU-blocking sync is no longer imposed by default, reducing per-frame latency on
  the producer side. Set `CUDALINK_EXPORT_SYNC=1` to restore the old behaviour if you
  need it (recommended for concurrent TD Sender+Receiver topologies and when
  `CUDALINK_USE_GRAPHS=0`). `TDConfig` default is **unchanged** (TD Sender still
  defaults to sync-blocking for TDR-cascade safety in shared-process topologies).

### Added

- **`DtypeCodec` backend accessors** — `DtypeCodec.typestr(dtype)`,
  `DtypeCodec.numpy_name(dtype)`, and `DtypeCodec.cupy_name(dtype)` expose the
  per-dtype backend representations (CAI typestr, NumPy dtype name, CuPy dtype name)
  through a single sealed codec. Adding a new dtype is now one row in `_DtypeEntry` —
  no other file changes. `numpy_name` returns `None` for bfloat16 (use `ml_dtypes`);
  `cupy_name` returns `None` for unsupported types.

- **`Importer.from_connection` public seam** — `Importer.from_connection(spec, policy,
  conn, fmt, *, cuda=None, …)` is now a documented public API that wraps an
  already-open `IPCConnection` into a connected `Importer`. Intended for GPU-free tests
  and callers that build a connection out-of-band.

- **`fakes.make_connected_importer` canonical test factory** — `tests/fakes/__init__.py`
  provides `make_connected_importer(…)` and `make_fake_ipc_connection(…)` as shared
  test fixtures, eliminating the scattered `object.__new__` / private-attribute
  injection patterns across the three Importer test files. Accepts `last_write_idx`,
  `debug`, `cupy`, and `numpy` parameters for scenario setup without private poking.

- **`Importer.from_connection` `last_write_idx` parameter** — pass `last_write_idx=N`
  to construct an Importer that treats frame index `N` as already-consumed.

- **`cuda_link._console` module** — private helper with
  `install_console_ctrl_handler(prefix, on_cleanup, *, defer_close)` and
  `run_with_watchdog(fn, timeout_s, label, prefix)`. Centralises the Windows
  `SetConsoleCtrlHandler` setup and daemon-thread cleanup watchdog pattern used by both
  example subprocess scripts.

- **Profiling scripts** — `scripts/profiling/` gains `ncu_pipeline.py` (ncu capture
  pipeline for per-region kernel analysis) and `analyze_td_pipeline.py` (A/B Nsight
  Systems capture analysis), both used in the Phase G profiling campaign that confirmed
  the default-SYNC-off decision.

### Refactored

- **`TDSenderEngine` collapsed onto canonical `Exporter`** (ADR-0001 step 7) —
  `td_exporter/TDSender.py` shrinks from ~1 280 lines to ~415 lines. All GPU ring-buffer
  management, graph capture, SHM writes, and publish logic are now delegated to the
  auto-derived `td_exporter/Exporter.py` mirror. `TDSenderEngine` retains only what is
  genuinely TD-specific: pixel-format rejection, the `cuda_memory()`→`GpuFrame` bridge,
  dynamic geometry reopen, and `HolderBarrier` lifecycle.

- **`CudaPort` / `ImporterCudaPort` unified** — `ImporterCudaPort` is now a public
  `CudaPort` alias (the full union of importer-exclusive methods was added to `CudaPort`
  in v1.6). `CTypesCUDAAdapter` body replaced with `__getattr__` delegation — no more
  explicit one-line forwarder list, so name drift becomes structurally impossible.

- **`_DTYPE_TABLE` consolidated via `_DtypeEntry` NamedTuple** — the three inline dtype
  maps (`TorchBuffers.typestr_map`, `CupyBuffers.dtype_map`, `_numpy_dtype_for` if-chain)
  that were previously spread across the codebase were deleted; all dtype lookups now
  route through the single `DtypeCodec` codec backed by `_DtypeEntry` rows.

- **`acquire_slot` / `publish_frame` ownership sealed** — all SHM-read/write operations
  outside `shm_protocol.py` were eliminated; the C3 ordering guarantee (shutdown flag
  visible before write_idx) is now exclusively enforced in one place.

### Fixed

- **`example_sender_python.py` migrated to `Exporter.open/export/close` API** — the
  example script was still using the removed `CUDAIPCExporter` class. It now uses the
  v1.5.0 `Exporter.open()` / `exporter.export(GpuFrame(...))` / `exporter.close()` API.

- **`Format.__eq__` sentinel comparison in `_reinitialize`** — the old full-field
  `__eq__` returned `False` when an override-derived `Format` (all-zero `kind`/`bits`/
  `flags` sentinels) was compared with a SHM-derived one carrying real non-zero values
  for the same `shape` + `dtype`. This triggered a spurious `NumpyBuffers` teardown on
  the first `_reinitialize` even when the shape and dtype were unchanged. Fixed by
  comparing only `shape` + `dtype_str` in `_reinitialize`.

- **`SetConsoleCtrlHandler` argtype drift in sender** — `example_sender_python.py` was
  using a raw untyped `ctypes.windll.kernel32` call (missing the `argtypes = [c_void_p,
  BOOL]` declaration present in the receiver since v1.5.1). Unified via the shared
  `_console` helper.

## [1.5.1] — 2026-05-22

### Added

- **TD→Python receiver example** (`td_exporter/example_receiver_python.py` +
  `td_exporter/example_receiver_launcher.py`) — complete example for receiving RGBA
  frames from a TouchDesigner `CUDAIPCLink_to_Python` Sender COMP into a Python
  subprocess. Mirrors the existing Python→TD sender pair. Launch via the Execute DAT
  or run `python td_exporter/example_receiver_python.py` directly.

- **Receiver example env-var overrides** — all receiver knobs configurable without
  editing the script:

  | Variable | Default | Effect |
  |---|---|---|
  | `CUDALINK_RECEIVER_SHM_NAME` | `cudalink_input_ipc` | IPC channel name |
  | `CUDALINK_RECEIVER_DEVICE` | `0` | GPU device index |
  | `CUDALINK_RECEIVER_TIMEOUT_MS` | `5000` | Frame-wait timeout ms |
  | `CUDALINK_RECEIVER_REPORT_EVERY` | `150` | Frames between status prints |
  | `CUDALINK_SENDER_REPORT_EVERY` | `150` | Frames between sender status prints |
  | `CUDALINK_RECEIVER_FRAME_MODE` | `torch` | Frame-fetch mode: `numpy` / `torch` / `cupy` |

- **Per-call `get_frame()` timing in receiver example** — each status line now
  includes a running `get_frame avg` µs figure. On exit a perf-summary line prints
  `mode=<mode>  get_frame avg/min/max µs  (n=<frames>)` for direct A/B comparison
  across frame modes.

- **`CUDALINK_RECEIVER_PYTHON_EXE` env var + Windows Python Launcher auto-detection**
  in the receiver Execute DAT launcher. At DAT load time the launcher queries
  `py -3 -c "import sys; print(sys.executable)"` to resolve the registered system
  Python 3 path, ensuring third-party packages (torch, cupy) are found. Override
  with `CUDALINK_RECEIVER_PYTHON_EXE=<full path>` before launching TouchDesigner.
  The resolved path is printed on each `onStart()`.

### Fixed

- **`SetConsoleCtrlHandler` argtype crash in example scripts** — `c_void_p` is now
  used for the `PHANDLER_ROUTINE` argument position (previously `_HandlerRoutine`,
  a strict `WINFUNCTYPE` subclass). `ctypes` rejected `None` for a strict `WINFUNCTYPE`
  argtype, causing a module-import crash when `SetConsoleCtrlHandler(None, False)` was
  called to re-enable Ctrl+C in the new process group. Fixed in both
  `example_sender_python.py` and `example_receiver_python.py`.

- **Graceful fallback to numpy when torch/cupy is not installed** — the receiver
  example previously exited with a hard `RuntimeError` if `get_frame()` could not
  find `torch`/`cupy`. It now pre-flights the import and silently falls back to
  `get_frame_numpy()` with a one-line warning, keeping the example functional in
  environments without GPU ML libraries.

- **Receiver launcher uses system Python, not PATH `python`** — the Execute DAT
  previously called `["python", script]`, which could resolve to TD's bundled Python
  (no third-party packages). Auto-detection via `py -3` now finds the registered
  system Python 3 at DAT load time.

### Changed

- **TD-side CUDA Graphs disabled by default** (`TDSenderConfig.use_graphs=False`,
  `CUDALINK_TD_USE_GRAPHS` default `"1"` → `"0"`). Per-frame receiver timing shows
  the graph-launch path provides negligible benefit at WDDM-bound 60 FPS
  (`cudaMemory ≈ 97 µs` with or without graphs at 1920×1080 uint8). Set
  `CUDALINK_TD_USE_GRAPHS=1` to opt back in. Reverses the v1.5.0 default-ON
  decision; prior benchmark analysis is preserved in
  `docs/perf/graphs-benchmark-v1.5.md`.

- **Receiver example defaults to `CUDALINK_RECEIVER_FRAME_MODE=torch`** (zero-copy
  GPU tensor path). Previously defaulted to `numpy` (D2H copy, ≈ 4 ms avg at
  1920×1080 uint8). Set to `numpy` for the D2H path or `cupy` for CuPy zero-copy.

## [1.5.0] — 2026-05-21

### Breaking changes

- **`CUDAIPCExporter` removed** — The `CUDAIPCExporter` class has been removed. Use
  `Exporter.open(FrameSpec(...))` from `cuda_link.exporter` instead. See
  `docs/MIGRATION_v1.5.md` for a side-by-side migration guide.

- **`CUDAIPCImporter.__init__` no longer auto-initializes.** (S3) Callers must explicitly
  invoke `.connect()` after construction, or use the `CUDAIPCImporter.from_connected()`
  classmethod for one-shot construction. The context manager (`with CUDAIPCImporter(...) as imp:`)
  auto-connects on entry and is unaffected.

  Migration:

  ```python
  # Before (v1.4.x)
  imp = CUDAIPCImporter(shm_name="my_shm")

  # After — one-shot (equivalent to old behaviour)
  imp = CUDAIPCImporter.from_connected(shm_name="my_shm")

  # After — two-step (when construction and connection happen at different times)
  imp = CUDAIPCImporter(shm_name="my_shm")
  ...
  imp.connect()
  ```

- **Pinned-memory allocation failure now raises by default.** (C6) When `cudaMallocHost`
  fails, the importer raises `RuntimeError` with a diagnostic message instead of silently
  falling back to `cudaHostRegister` → pageable memory. To restore the prior silent fallback,
  set `CUDALINK_ALLOW_PAGEABLE_FALLBACK=1`.

### Added

- **`Exporter` module** (`src/cuda_link/exporter.py`) — deep, testable replacement for
  the inlined CUDA IPC logic in `CUDAIPCExporter`. Three-method public surface:
  `Exporter.open(spec, *, policy, cuda)` / `export(frame)` / `close()`. `open()` either
  returns a fully-initialised exporter or raises — no half-states. `close()` is
  idempotent. `export()` returns `FrameOutcome` instead of raising on backpressure.

- **`FrameSpec` dataclass** — frozen value object capturing resolution, dtype, slot count,
  and device. Replaces positional constructor arguments.

- **`ExportPolicy` dataclass** — frozen value object for all export flags
  (`export_sync`, `use_graphs`, `flush_probe`, `strict_device`, `barrier_enabled`,
  `high_priority_stream`, `export_profile`). Replaces env-var-only configuration.
  Env vars are still respected via `ExportPolicy.from_env()`. Named presets:
  `ExportPolicy.for_testing()` and `ExportPolicy.low_latency()`.

- **`GpuFrame` dataclass** — typed wrapper for `(ptr, size)` passed to `export()`.

- **`FrameOutcome` enum** — `PUBLISHED` / `SKIPPED_BARRIER` / `SKIPPED_NOT_READY` /
  `FAILED`. Replaces the `bool` return from `export_frame()`.

- **`CudaPort` Protocol** (`_exporter_port.py`) — structural-typing seam between the
  `Exporter` and the CUDA runtime. Enables injecting a test double at the
  one-seam boundary without touching SHM, time, or logging.

- **`CTypesCUDAAdapter`** and **`FakeCUDAAdapter`** (`_cuda_adapters.py`) — production
  and in-memory test adapters satisfying `CudaPort`. `FakeCUDAAdapter` requires no GPU
  or ctypes DLL; tracks allocations and supports failure injection. Both symbols now
  import directly from `_cuda_adapters` — the `_exporter_adapters` re-export shim has
  been removed.

- All five new symbols exported from `cuda_link.__init__`:
  `Exporter`, `FrameSpec`, `ExportPolicy`, `GpuFrame`, `FrameOutcome`.

- **`HolderBarrier` port+adapter deepening** — `HolderBarrier` now consumes a
  `HolderShmPort` (production: `RealShmAdapter`, test: `FakeShmAdapter`) so its
  acquire/release lifecycle can be exercised without real `SharedMemory`. Mirrors
  the `CheckerBarrier` deepening from the same session. `RealShmAdapter` satisfies
  both `BarrierShmPort` and `HolderShmPort` structurally.

- **`slot_names(prefix, n=10)`** helper in `_nvtx.py` — consolidates three duplicate
  slot-name tuple declarations in `exporter.py` and `importer.py` into one location.

- **`Exporter.open(..., barrier=...)` kwarg** — accepts an optional
  `barrier: CheckerBarrier | None` for injecting a pre-configured `CheckerBarrier`
  in tests. Production callers pass nothing; the exporter constructs the default.

- `CUDALINK_D2H_STREAM_PRIO=high` opt-in env var allocates the importer's D2H streams at
  high priority, mirroring `CUDALINK_LIB_STREAM_PRIO` on the exporter side. Default:
  `"normal"`. (S9)

- **TD-side `[GRAPHS_INIT]` diagnostic log** — `TDSenderEngine.__init__` now logs
  `_use_graphs`, `config.use_graphs`, and the `CUDALINK_TD_USE_GRAPHS` env var
  value at startup, confirming graph state on each component activation.

- **`docs/ctypes-audit-v1.5.md`** — full audit of all 49 CUDA runtime bindings and
  Win32 helpers against the Python ctypes documentation. No drift found; all argtypes,
  restypes, pointer semantics, and struct layouts verified correct.

- **`docs/perf/graphs-benchmark-v1.5.md`** — A/B/C/D benchmark for CUDA Graphs ON/OFF
  (Python and TD sides). Cell B (Python graphs ON) shows −3.4 % median latency vs
  Cell A (OFF); no regression in any soak metric. Decision: both `CUDALINK_USE_GRAPHS`
  and `CUDALINK_TD_USE_GRAPHS` remain default ON.

### Changed

- **TD-side CUDA Graphs enabled by default** (`TDSenderConfig.use_graphs=True`,
  `CUDALINK_TD_USE_GRAPHS` default `"0"` → `"1"`). The graph-capture path is
  byte-identical to the proven Python sender path (shared `cuda_graphs.py` mixin) and
  ships with three auto-fallback sites that silently revert to `cudaMemcpyAsync` on
  capture or launch failure. Set `CUDALINK_TD_USE_GRAPHS=0` to opt out. Brings TD sender
  to default parity with the Python sender (`CUDALINK_USE_GRAPHS` defaults to `1`).

- **`CUDAIPCWrapper` / `cuda_ipc_wrapper` docstring** updated: runtime requirement now
  reads "CUDA 11.x or 12.x runtime (cudart64_12.dll preferred; cudart64_11.dll /
  cudart64_110.dll accepted as fallback)" to accurately reflect the probe order.

- **`CUDALINK_*` env reads consolidated behind `_env` helpers** — all scattered
  `os.getenv`/`os.environ.get` calls in `src/cuda_link/` now flow through
  `_env.env_bool()`, `_env.env_int()`, and `_env.env_str()`. Each helper reads
  `os.environ` at call time so `monkeypatch.setenv` works reliably in tests without
  import-order constraints. `CUDAIPCImporter.connect()` no longer duplicates env reads —
  delegates to `ImportPolicy.from_env()`.

- **`TDSenderConfig.from_env()` now routes all `CUDALINK_*` reads through the
  `Env.env_bool` / `env_int` / `env_str` helpers**, matching `ExportPolicy.from_env()`.
  **Behaviour change**: `CUDALINK_EXPORT_SYNC`, `CUDALINK_TD_PERSIST_STREAM`, and
  `CUDALINK_TD_ACTIVATION_BARRIER` now use strict `"1"` = enabled semantics
  (previously permissive: anything except `"0"` was enabled). Set the env var
  to `"1"` or `"0"` explicitly. `CUDALINK_TD_BARRIER_SETTLE_FRAMES` now falls
  back to default `30` on non-numeric input instead of raising.

- **`CUDAIPCExtension.reconfigure_and_reinit(field_name, new_value)`** — new method
  that consolidates the cleanup → set-field → reconnect cycle used by `parexecute_callbacks`.
  `handle_ipcmemname_change` and `handle_numslots_change` now delegate to this method,
  removing ~20 LOC of duplicated reinit logic.

- **`CheckerBarrier`** now consumes a `BarrierShmPort` (production: `RealShmAdapter`,
  test: `FakeShmAdapter` in `tests/conftest.py`), enabling coverage without real
  `SharedMemory`. Hot-path method renamed `evaluate() -> CheckerOutcome`
  (DISABLED / NO_SKIP / SKIP_ACTIVE / SKIP_STALE / SHM_ABSENT). The existing
  `should_skip_publish() -> bool` wrapper is kept for callers that only need a bool.
  Mirror to `td_exporter/ActivationBarrier.py` stays byte-identical.

- **`_nvtx.py` rewritten to module-import-time detection** — the `_ensure_init()` lazy
  initialiser and associated globals are replaced by a single
  `_NVTX_STATE = _detect_nvtx()` call at import time. All five public functions now
  read `_NVTX_STATE.available` directly.

### Fixed

- **cudart DLL probe order** — `cudart64_12.dll` is now tried first, falling back to
  `cudart64_11.dll` then `cudart64_110.dll`. The previous order (`110 → 12 → 11`) was a
  stale artifact of the W1 WDDM bisect investigation (closed 2026-04-23, REFUTED).
  Synced to `td_exporter/CUDAIPCWrapper.py`.

- **`nvml_observer` `pynvml` `FutureWarning` suppressed** — `import pynvml` is now
  wrapped in `warnings.catch_warnings()` with a targeted `filterwarnings` anchored on
  the deprecation-banner text. Other `FutureWarning`s still surface; global warning
  state is unchanged. Docstring also clarifies the `pynvml` / `nvidia-ml-py` naming
  ambiguity. Synced to `td_exporter/NVMLObserver.py`.

- **Redundant "Loaded CUDA runtime" log on reconnect** — `TDReceiverEngine` and
  `TDSenderEngine` now log this line only once per engine lifetime.

- **`Exporter.export()` no longer catches `ValueError`** in its broad exception handler.
  Strict-mode violations (`strict_device=True`, wrong pointer type or wrong device) now
  propagate to the caller as documented.

### Removed

- **`_exporter_adapters.py` re-export shim** — `CTypesCUDAAdapter` and `FakeCUDAAdapter`
  now import directly from `_cuda_adapters`. The `_exporter_adapters.py ↔ ExporterAdapters.py`
  sync pair removed from `scripts/sync_td_wrapper.py`.

- **`src/cuda_link/debug_utils.py`** — dead code with zero importers.

### Hardening

- Declared `argtypes`/`restype` on all Win32 helper calls: `kernel32.GetModuleFileNameW`,
  `winmm.timeBeginPeriod`, `winmm.timeEndPeriod`. `kernel32` and `winmm` now loaded with
  `use_last_error=True` via process-local `WinDLL` handles, eliminating global
  `ctypes.windll` usage. (C7)
- Full-path CUDA DLL fallback now passes `winmode=0` to prevent DLL hijacking via the
  process search order. (C8)
- DLL-loader `OSError` catches now log `e.winerror` at DEBUG to distinguish WinError 126
  (DLL not found) from 193 (wrong bitness). (C9)

### Tests

- Rewrote `tests/test_device_affinity.py` and the write-ordering section of
  `tests/test_cuda_ipc_exporter_python.py` to use `Exporter.open(..., cuda=FakeCUDAAdapter())`
  instead of `object.__new__(CUDAIPCExporter)` followed by ~25 hand-populated private
  attributes. Both files now run without a GPU.

- Added `tests/test_activation_barrier_holder.py` — 23 no-SHM tests for `HolderBarrier`
  via `FakeShmAdapter`. Covers acquire, arm_settle_countdown, tick_and_maybe_release,
  force_release, close, and end-to-end lifecycle.

- Removed spurious `@pytest.mark.requires_cuda` from 4 CPU-only tests in
  `tests/test_cuda_ipc_importer.py`.

- Fixed stale `tests/test_wrapper_sync.py` docstring: replaced the hardcoded pair
  listing with a pointer to `sync_td_wrapper.PAIRS` as the authoritative source.

## [1.4.2] — 2026-05-18

### Fixed

- **C2 — CUDA stream-capture mode coexistence with TensorRT / CuPy / PyTorch CUDA Graphs**
  (`src/cuda_link/cuda_ipc_exporter.py:591`, `td_exporter/TDSender.py:507`):
  Both CUDA Graph build sites previously used `cudaStreamCaptureModeGlobal` (mode=0),
  which marks any concurrent `cudaStreamBeginCapture` in the **entire process** as
  invalid. When cuda-link is co-resident with TensorRT (e.g. StreamDiffusion ControlNet),
  TRT's own `cudaStreamEndCapture` returned `cudaErrorStreamCaptureInvalidated (901)`.
  Changed to `cudaStreamCaptureModeRelaxed` (mode=2): independent captures no longer
  invalidate each other; cuda-link's graph build is synchronous at init so there is no
  concurrent enqueue on the captured stream during build.

  Docstring in `cuda_graphs.py` / `CUDAGraphs.py` `stream_begin_capture` updated to
  explain all three modes and when each is appropriate.

  Regression test: `tests/test_graph_coexistence_capture.py`
  (`@pytest.mark.requires_cuda`).

### Internal

- **`pyproject.toml` version**: `1.4.1` → `1.4.2`.
- **TOX artifact**: `TOXES/CUDAIPCLink_v1.4.2.tox` to be built separately.
  `v1.4.1` retained per versioned-binary tracking policy.

---

## [1.4.1] — 2026-05-10

### Added

- **Read-only `Status` custom parameter** on the CUDAIPCLink COMP surfaces engine state at a glance. Displays `"WARNING: <msg>"` / `"ERROR: <msg>"` during fault conditions; `"<W>x<H> <dtype> <ch>ch"` (e.g. `"1920x1080 float32 4ch"`) on each successful frame; `"Idle"` when the component is inactive or clear. Wired via the new `TDHost.set_info_status()` protocol method and `_write_status_par()` dedup helper (`td_exporter/TDHost.py`); call sites in `TDSenderEngine` (after a successful `cuda_memory()` fetch) and `TDReceiverEngine` (after `copyCUDAMemory`). Per-frame dedup avoids hammering the par on steady-state transfers. Regression test: `tests/test_tdhost_status_par.py` (12 cases).
- **`warning_emitter` Script TOP child** driven by COMP storage emits a local `addWarning` badge INSIDE the COMP alongside the COMP-body tint. Provides a second visible warning surface when the user opens the COMP network. New file `td_exporter/warning_emitter_callbacks.py`. State-transition-gated to avoid redundant force-cooks every bad frame. Regression test: `tests/test_tdhost_warning_emitter.py`.

### Fixed

- **Stale COMP tint at extension boot** — `RealTDHost.__init__` now invokes `_reset_stale_tint()`, which detects a COMP saved in a warning/error-tinted state, restores the default grey colour, clears script errors, and unstores the `"cuda_link_status_msg"` key. Previously the `.tox` booted yellow if it had been saved mid-warning (e.g. during development). Regression test: `tests/test_tdhost_init_reset.py`.
- **Active toggle off now resets COMP to grey and writes `"Idle"` to Status par** — `parexecute_callbacks.py:handle_active_change` deactivation branch calls `ext._host.clear_status()` after `ext.cleanup()`. Previously the COMP stayed tinted and the Status par retained the last frame's resolution/dtype or warning string after deactivation.
- **Status-par dtype label showed `"<class 'numpy.uint8'>"` instead of `"uint8"`** — TD's `shape.dataType` is the numpy TYPE class, not a dtype instance (no `.name` attribute). Fixed by chaining `getattr(_dt, "name", None) or getattr(_dt, "__name__", str(_dt))` in both `TDSender.py` and `TDReceiver.py`.
- **Unsupported-pixel-format warning message shortened** — `TDSender.py:783` now emits `f"unsupported pixel format {src_fmt!r}"` instead of a verbose 3-line instructional string. The full instructional text remains in the per-episode `_log()` call (textport output). Status par now displays `"WARNING: unsupported pixel format '11:11:10'"` — readable at a glance in the Custom Pars panel.
- **`clear_status()` sentinel renamed `"OK"` → `"Idle"`** — better semantic match for "no transfer in progress, no warning"; used in Status par and `warning_emitter` state.
- **`test_activation_barrier` failures on Windows when a live TD session holds the named SHM segment** — Windows SHM lifetime is handle-bound; `SharedMemory.unlink()` is a no-op while another process holds an open handle, so the test fixture's cleanup silently failed and subsequent counter assertions read production-accumulated values (e.g. `assert 73 == 3`). Fixed in `tests/test_activation_barrier.py`: `cleanup_barrier` fixture now re-zeros the SHM header in-place if external cleanup fails, giving each test a deterministic zero state; `test_open_or_create_raises_when_missing_and_no_create` conditionally skips with a descriptive reason when an external holder is detected.

### Internal

- **`pyproject.toml` version**: `1.4.0` → `1.4.1`.
- **`docs/BENCHMARKS.md` added** — consolidates every benchmark table previously embedded in `README.md` §Performance / §Benchmarks (RTX 4090 / PCIe 4.0 / driver 596.36 numbers). `README.md` §Benchmarks is now a concise summary linking here.
- **`benchmarks/` moved to local-only** — gitignored and removed from the git index (`git rm --cached -r`). Scripts remain on contributor disks; numbers are preserved in `docs/BENCHMARKS.md`.
- **`CONTEXT.md` moved to local-only** — gitignored and removed from the git index. Architecture vocabulary is covered by `docs/ARCHITECTURE.md`.
- **Test gate**: `272 passed, 3 skipped` (pure-Python suite; CUDA-marked tests auto-skip on CUDA-less hosts; `test_activation_barrier` SHM-external-holder test skips when a live TD session is running).
- **Mirror invariant** preserved across five module pairs: `shm_protocol.py`, `cuda_ipc_wrapper.py`, `activation_barrier.py`, `cuda_runtime_types.py`, `cuda_graphs.py` ↔ their `td_exporter/` twins.
- **TOX artifact**: `TOXES/CUDAIPCLink_v1.4.1.tox` to be built by the user separately. `v1.4.0` retained per versioned-binary tracking policy.

---

## [1.4.0] — 2026-05-10

### Changed

- **Unsupported pixel formats now tint the component yellow instead of auto-converting** (**breaking**) — `TDSenderEngine` no longer routes float16, 10:10:10:2, and 11:11:10 pixel formats through a `dtype_converter` Transform TOP. Instead, when `cudaMemory()` would reject or silently corrupt the source format, the sender skips frames and tints the component node body **yellow** (`parent().color`) every bad frame; the first bad frame per episode also logs once. The tint clears automatically as soon as the upstream source TOP format is corrected and transfer resumes. Engine-fatal errors (IPC/GPU init failure) tint the component **red** and emit a red `addScriptError` badge. Users who previously relied on silent auto-conversion must fix the upstream source TOP's Pixel Format parameter. The `dtype_converter` Transform TOP inside the `.tox` is now dead weight; rebuild without it as `CUDAIPCLink_v1.4.0.tox` (wire `input → ExportBuffer` directly). API rename: `_needs_format_conversion` → `_is_unsupported_format`; `TDHost.add_warning`/`clear_warning` → `set_warning_status`/`set_error_status`/`clear_status`. Regression test: `tests/test_tdsender_format_warning.py`.

### Fixed

- **Yellow tint no longer sticks after a bad-format episode is resolved** (Bug 1 — H1 confirmed) — `RealTDHost` previously cached `parent().color` at construction time; if the `.tox` was saved while tinted, or the extension re-initialised after a bad frame had already run, `_default_color` was stored as the warning colour and `clear_status()` faithfully restored it to yellow. `_default_color` is now captured lazily on the first `set_warning_status` / `set_error_status` call and excluded from the cache when it equals a managed colour, falling back to TD's default node grey. Regression test: `tests/test_tdhost_default_color_cache.py`.
- **Dead `cooking_op=me` plumbing removed** (Bug 2 — H5 confirmed; H1–H4 and cook-context-propagation theory disproved) — Phase E shipped `callbacks_template.py:onFrameEnd` passing `cooking_op=me` to `export_frame()`, which called `me.addWarning(msg)` to emit a yellow badge on the Execute DAT. Live TD probes (`verification/results/probe_addwarning_*`, `verification/results/probe_cook_context_*`) disproved every plausible path: H5 (`tdError: Cannot set warning outside of cook` — `addWarning` is illegal in `onFrameEnd`) blocked the Execute DAT path; the child-Script-TOP-cook path (`addWarning` succeeds in `onCook` per H5a, confirmed) was blocked by H5b (TD does not propagate the child warning to the parent COMP boundary tile — neither `comp.warnings()` aggregates it, nor does a visual badge appear). The COMP body tint is the only COMP-local warning surface. All `cooking_op` plumbing removed: `TDSender.export_frame` signature, `CUDAIPCExtension.export_frame` wrapper, `callbacks_template.py:onFrameEnd`. The `_FakeCookingOp` helper and the three `test_cooking_op_*` tests locked in the wrong contract and are dropped; replaced by two host-boundary regression tests (`test_export_frame_bad_format_returns_false_and_warns_host`, `test_export_frame_good_format_after_bad_calls_clear_status`) in `tests/test_tdsender_format_warning.py`.

### Internal

- **`pyproject.toml` version**: `1.3.0` → `1.4.0`.
- **TOX artifact**: `TOXES/CUDAIPCLink_v1.4.0.tox` to be built from the updated component tree (no `dtype_converter` Transform TOP — wire `input → ExportBuffer` directly per `docs/TOX_BUILD_GUIDE.md`). Prior `.tox` versions (`v1.2.1`, `v1.3.0`) retained per versioned-binary tracking policy.
- **Test gate**: `247 passed, 2 skipped` (pure-Python suite; CUDA-marked tests auto-skip on CUDA-less hosts via `tests/conftest.py:34-50`).
- **Mirror invariant** preserved across five module pairs: `shm_protocol.py`, `cuda_ipc_wrapper.py`, `activation_barrier.py`, `cuda_runtime_types.py`, `cuda_graphs.py` ↔ their `td_exporter/` twins.

## [1.3.0] — 2026-05-09

### Changed — Internal Architecture (Deepening pass)

- **SHM protocol extracted to standalone module** — `SHMProtocol.py` now owns the v0.5.0 binary layout (20-byte header, 128-byte slots, shutdown flag, metadata, timestamp). Mirror pair `src/cuda_link/shm_protocol.py` ↔ `td_exporter/SHMProtocol.py` is byte-identical (commit `0ee8cbf`, Deepening A).
- **TDReceiverEngine deepened with value objects** — `ReceiverConnection`, `FormatDescriptor`, and `RetryState` replace ad-hoc state attributes; retry logic concentrated in one place (commit `3ceab97`, Deepening B).
- **TDHost seam activated in engines** — `TDSenderEngine` and `TDReceiverEngine` now talk to TouchDesigner exclusively through the `TDHost` / `TOPHandle` adapter protocol; `RealTDHost` and `FakeTDHost` are interchangeable for tests (commit `1c9aa98`, Deepening C).
- **CUDAIPCImporter deepened** — split into `IPCConnection` + `Format` + per-backend value objects, replacing shallow attribute access with an explicit lifecycle (commit `e2cdb45`, Deepening D, retires deferred item #6).
- **Activation barrier extracted into dataclasses** — `ProducerActivationBarrier` (`src/cuda_link/cuda_ipc_exporter.py:107`) and `SenderActivationBarrier` (`td_exporter/TDSender.py:65`) replace 5 scattered `_barrier_*` attrs per consumer. Lifecycle now flows through `from_env` / `from_config` / `acquire` / `arm_settle_countdown` / `tick_and_maybe_release` / `force_release` / `should_skip_publish` / `close`. Behavior preserved; log line origins shift but shape unchanged. `CUDALINK_ACTIVATION_BARRIER` default remains `"1"` (commit `d67c143`, Deferred #7).
- **CUDA wrapper split into types + graphs** — `cuda_runtime_types.py` (ctypes structs, type aliases, `CUDAError`, `CUDART_GRAPHS_MIN_VERSION`) and `cuda_graphs.py` (`CUDAGraphsMixin`, 13 graph-lifecycle methods) extracted from the 1455-LOC `cuda_ipc_wrapper.py`. `CUDARuntimeAPI(CUDAGraphsMixin)` keeps the public API byte-stable; callers (`cuda_ipc_exporter`, `cuda_ipc_importer`, `TDSender`, `TDReceiver`, tests) migrated off the wrapper shim to import types from the canonical location. Mirror invariant extended: `cuda_runtime_types.py` ↔ `td_exporter/CUDARuntimeTypes.py` and `cuda_graphs.py` ↔ `td_exporter/CUDAGraphs.py`. Wrapper shrinks from 1455 → ~660 LOC (commits `415f7b2`, `7a4b5cb`, `67f20a2`; merge `43fd4b9`; Deferred #3).

### Removed — Internal

- **7 test-only `@property` bridges deleted from `CUDAIPCExtension` facade** — approximately 50 test sites in `tests/test_extension_characterization.py` and `tests/test_cuda_ipc_exporter.py` rewritten to access `ext._engine.X` / `exporter._engine.X` directly. Three production bridges (`shm_name`, `num_slots`, `verbose_performance` at `td_exporter/CUDAIPCExtension.py:226-244`) are retained intentionally for parexecute-DAT callbacks. Out-of-tree consumers that touched the deleted bridges will need to migrate to `_engine` access (commit `4ad2a5c`, Deferred #5).

### Fixed

- **TDSender writes correct dtype metadata when format changes mid-stream** — `TDSenderEngine._write_metadata_to_shm()` now uses `cuda_mem.data_type` (`_detected_numpy_dtype`) as the authoritative source for `format_kind` and `bits_per_comp`, falling back to the allocation-size ratio only when the dtype is unavailable. Previously, when the source TOP's pixel format changed from RGBA32F to RGBA8 while TD's reported `cuda_mem.size` remained at the float32-sized allocation (padded/atlas buffer), the ratio `data_size / pixel_count = 4 bytes/pixel` produced `bits=32, kind=Float` even though the actual data was uint8. The receiver then correctly read this wrong metadata and decoded every uint8 byte as float32, producing `-3.4028e+38` (FLT_MIN) garbage in normalized views and the visual output appearing frozen. Regression tests: `tests/test_tdsender_dtype_metadata.py`.
- **TDReceiver refreshes dtype in-place on SHM version bump** — The `VERSION_CHANGED` handler in `TDReceiverEngine.import_frame()` now calls the new `_refresh_on_version_change()` method before falling back to `cleanup()`. The refresh re-reads the 20-byte metadata block from SHM, rebuilds `self._format` and `self._cached_shape` (including `dataType`), and advances `ipc_version` — all without closing the SHM handle, IPC events, or stream. For genuine sender re-inits (new IPC handle bytes in SHM), the refresh also closes stale IPC imports and opens the new ones, preserving the SHM connection. This mirrors `CUDAIPCImporter._reinitialize`. The previous heavy `cleanup()`-and-reinit path remains as the fallback when the refresh returns False (corrupt metadata or failed handle re-import). Regression tests: `tests/test_tdreceiver_dtype_refresh.py`.
- **TDSender metadata invariant with padded GPU allocations** — `TDSenderEngine._write_metadata_to_shm()` now writes `data_size = W*H*C*(bits/8)` (active-region size) instead of `self.data_size` (GPU allocation size). Previously, when TD reported the same `cuda_mem.size` across frames with different dimensions (padded or atlas allocations), the metadata-only-update branch updated `width`/`height`/`channels` but left `data_size` at the old allocation value. The new pixel count no longer divided evenly into the old allocation, `bits` fell back to 32, and the receiver invariant `W*H*C*(bits/8) == data_size` failed — causing the receiver to log `"Metadata size invariant failed … Sender/receiver protocol mismatch."` and loop indefinitely in `"Waiting for sender"`. In normal TD operation (no padding) the active-region size equals the allocation size, so there is no regression for existing users. Matches `CUDAIPCExporter` behaviour and the v1.0.0 protocol spec. Latent since `ddd8f0f`; surfaced by the v1.0.0 strict receiver invariant (`6afc49d`). Regression test: `tests/test_tdsender_metadata_only_update.py` (commit `a23b768`).
- **Progress columns suppressed when `EXPORT_PROFILE` is off** — `avg_total` and `avg_memcpy` no longer print misleading `0.0 µs` when profiling is disabled (commit `8658dbe`).
- **Protocol constants now route directly from `SHMProtocol`** in `CUDAIPCExtension` (commit `eb8b443`).
- **10-bit and 11-bit pixel formats routed through `dtype_converter`** — Empirical probe (`verification/results/cuda_memory_probe_20260510_090919.json`, TD 2025.32820) found two production bugs in `_CUDA_UNSUPPORTED_PIXEL_FORMATS` (`td_exporter/TDSender.py:58`): (Bug A) 10-bit RGB / 2-bit Alpha fixed was rejected outright by `cudaMemory()` with "Source TOP has unsupported pixel format." — frames skipped forever, no auto-conversion fallback; (Bug B) 11-bit float (RGB) "succeeded" but returned `dataType=uint8, numComps=4` (raw 32-bit packed word byte layout, not the 11:11:10 float semantic) — receiver decoded garbage with no error log. Widened rejection substring set with `"10-bit"`, `"10bit"`, `"11-bit"`, `"11bit"`; all six problematic formats now route through `dtype_converter → rgba32float`. Regression test: `tests/test_tdsender_format_rejection.py`.

### Docs

- **Refreshed v1.2.1 benchmark numbers and aligned cross-doc citations** (commit `9f98a72`).

### Internal

- **`pyproject.toml` version**: `1.2.1` → `1.3.0`.
- **TOX artifact**: `TOXES/CUDAIPCLink_v1.3.0.tox` rebuilt to include `CUDARuntimeTypes` and `CUDAGraphs` textDATs from the wrapper split. `TOXES/CUDAIPCLink_v1.2.1.tox` retained per versioned-binary tracking policy.
- **Example `.toe` snapshots**: `CUDA_Link_Example.50.toe` (pre-wrapper-split, initial v1.3.0) retained as historical; `CUDA_Link_Example.51.toe` captures the post-wrapper-split COMP state. `Test_TD_Receiver1*.toe` and `SESSION_LOG.md` gitignored (commit `e0a5f49`).
- **Mirror invariant** preserved across five module pairs: `shm_protocol.py`, `cuda_ipc_wrapper.py`, `activation_barrier.py`, `cuda_runtime_types.py`, `cuda_graphs.py` ↔ their `td_exporter/` twins.
- **Test gate**: `212 passed, 2 skipped` (pure-Python suite; CUDA-marked tests auto-skip on CUDA-less hosts via `tests/conftest.py:34-50`).

---

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
- **`example_sender_python.py` progress-line cosmetics** — `avg_total` and
  `avg_memcpy` columns are now suppressed when `CUDALINK_EXPORT_PROFILE=0`
  (the default). Previously the two columns always printed as `0.0 µs`, which was
  misleading. The leading `export=` wall-clock figure (always meaningful) is unaffected.
  (commit `8658dbe`)

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
- **Benchmark refresh (v1.2.1, 2026-05-09)** — re-ran `bench_sweep.py` (full 16-cell),
  `bench_graphs.py` (4 resolutions), and `bench_d2h_streams.py` (4 resolutions) on
  RTX 4090 / driver 596.36 / PCIe 4.0 x16 / Windows 11 under v1.2.1 defaults
  (EXPORT_SYNC=1, CUDALINK_USE_GRAPHS=1). Updated README, ARCHITECTURE, and
  INTEGRATION_EXAMPLES citations. Notable changes vs v1.2.0 sweep: isolated
  `export_frame()` p50 improved (512×512: 42→22 µs, 1080p: 138→117 µs, 4K: 400→367 µs);
  IPC notify p50 tightened (~250→~136–286 µs range); D2H bench_d2h_streams largely
  unchanged. Fresh `sweep_latest.csv` / `sweep_latest.json` committed (16-cell,
  2026-05-09 03:23).

### Internal

- **CGW PreToolUse guardrail** — `cgw-pre-bash.sh` intercepts `git reset --hard` /
  `git push --force` to non-PR branches before they execute. Local-only; not shipped
  in the package. (commit `0a14945`)
- **Example .toe project** bumped to `CUDA_Link_Example.45.toe`; new
  `Test_TD_Receiver1.toe` added for parallel-receiver topology testing.
  (commits `1fbbb61`, `fec21ca`, `72e1354`)
- **`TOXES/CUDAIPCLink_v1.2.0.tox`** added; `v1.1.0.tox` retired per `.gitignore`
  "only latest binary on main" policy. (commit `89a2b91`)
- **`TOXES/CUDAIPCLink_v1.2.1.tox`** added (58 630 B, built 2026-05-09);
  `v1.2.0.tox` retired from git index per the same policy. Updated
  `docs/TOX_BUILD_GUIDE.md` filename and build-date references.

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

[1.8.0]: https://github.com/forkni/cuda-link/compare/v1.7.2...v1.8.0
[1.7.2]: https://github.com/forkni/cuda-link/compare/v1.7.1...v1.7.2
[1.7.1]: https://github.com/forkni/cuda-link/compare/v1.7.0...v1.7.1
[1.7.0]: https://github.com/forkni/cuda-link/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/forkni/cuda-link/compare/v1.5.1...v1.6.0
[1.5.1]: https://github.com/forkni/cuda-link/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/forkni/cuda-link/compare/v1.4.2...v1.5.0
[1.4.2]: https://github.com/forkni/cuda-link/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/forkni/cuda-link/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/forkni/cuda-link/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/forkni/cuda-link/compare/v1.2.1...v1.3.0
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
