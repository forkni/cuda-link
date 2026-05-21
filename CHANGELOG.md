# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **`CUDAIPCExporter`** removed as scheduled in the v1.6.0 deprecation notice. Use
  `Exporter.open(FrameSpec(...))` instead. See `docs/MIGRATION_v1.6.md` (migration
  window closed in v1.7.0).
- `src/cuda_link/debug_utils.py` — dead code with zero importers; removed.

### Changed

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

## [1.6.0] — 2026-05-20

### Breaking changes (deprecations)

- **`CUDAIPCExporter` is deprecated** and will be **removed in v1.7.0**. Use
  `Exporter.open(FrameSpec(...))` from `cuda_link.exporter` instead. Existing code
  continues to work via a compatibility shim that emits `DeprecationWarning`. See
  `docs/MIGRATION_v1.6.md` for a side-by-side migration guide.

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

- **`CTypesCudaAdapter`** (`_exporter_adapters.py`) — production adapter that wraps
  `CUDARuntimeAPI` and satisfies `CudaPort`.

- **`FakeCudaAdapter`** (`_exporter_adapters.py`) — in-memory test adapter. No GPU, no
  ctypes DLL required. Tracks allocations (`adapter.allocations`), supports failure
  injection (`fail_on_malloc_count`, `fail_on_stream_create`, `fail_on_event_create`),
  and simulates CUDA Graphs as no-ops. Used in all unit tests for device-affinity and
  export-outcome coverage.

- All five new symbols exported from `cuda_link.__init__`:
  `Exporter`, `FrameSpec`, `ExportPolicy`, `GpuFrame`, `FrameOutcome`.

### Fixed

- `Exporter.export()` no longer catches `ValueError` in its broad exception handler.
  Strict-mode violations (`strict_device=True`, wrong pointer type or wrong device)
  now propagate to the caller as documented, instead of being silently converted to
  `FrameOutcome.FAILED`.

### Tests

- Rewrote `tests/test_device_affinity.py` and the write-ordering section of
  `tests/test_cuda_ipc_exporter_python.py` to use `Exporter.open(..., cuda=FakeCudaAdapter())`
  instead of `object.__new__(CUDAIPCExporter)` followed by ~25 hand-populated private
  attributes. Both test files now run without a GPU.

## [1.5.0] — 2026-05-19

### Breaking changes

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

- `CUDALINK_D2H_STREAM_PRIO=high` opt-in env var allocates the importer's D2H streams at
  high priority, mirroring `CUDALINK_LIB_STREAM_PRIO` on the exporter side. Default:
  `"normal"`. (S9)

### Hardening

- Declared `argtypes`/`restype` on all Win32 helper calls: `kernel32.GetModuleFileNameW`,
  `winmm.timeBeginPeriod`, `winmm.timeEndPeriod`. `kernel32` and `winmm` now loaded with
  `use_last_error=True` via process-local `WinDLL` handles, eliminating global
  `ctypes.windll` usage. (C7)
- Full-path CUDA DLL fallback now passes `winmode=0` to prevent DLL hijacking via the
  process search order. (C8)
- DLL-loader `OSError` catches now log `e.winerror` at DEBUG to distinguish WinError 126
  (DLL not found) from 193 (wrong bitness). (C9)

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
