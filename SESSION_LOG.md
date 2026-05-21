# Session Log

## Quick Navigation

- [2026-05-21 - Graphs Both Ends, pynvml Suppression & Runtime Cleanup Bundle](#2026-05-21---graphs-both-ends-pynvml-suppression--runtime-cleanup-bundle)
- [2026-05-20 - CheckerBarrier Port+Adapter Deepening](#2026-05-20---checkerbarrier-portadapter-deepening)

---

### Session: 2026-05-21 - Graphs Both Ends, pynvml Suppression & Runtime Cleanup Bundle

**Primary Achievement**: Enabled CUDA Graphs by default on both the Python and TD sender paths, diagnosed and fixed the `use_graphs=False` TD-side root cause, suppressed the pynvml `FutureWarning` at the import site, and landed a four-item post-log-analysis cleanup bundle — all soak-confirmed end-to-end.

#### Key Accomplishments
- Diagnosed `graphs=OFF` on TD side: `TDSenderConfig.use_graphs` defaulted to `False` and read `CUDALINK_TD_USE_GRAPHS` (not `CUDALINK_USE_GRAPHS`); added `[GRAPHS_INIT]` diagnostic log that surfaced the root cause immediately at next TD startup
- Flipped `TDSenderConfig.use_graphs` default `False → True` and `CUDALINK_TD_USE_GRAPHS` env default `"0" → "1"` — soak confirmed: 8972 frames Python→TD + TD→Python at 58.7 FPS avg with `graphs=ON` on both sides; format change (uint8 → float32) and two reconnect cycles all clean
- Four-item cleanup bundle: cudart DLL probe order reversed to prefer `cudart64_12.dll` (reverted stale W1/WDDM bisect artifact), `[GRAPHS_INIT]` diagnostic log, `nvml_observer` docstring clarifying `pynvml`/`nvidia-ml-py` naming collision, one-shot `"Loaded CUDA runtime"` log flag on TDSender + TDReceiver
- Suppressed `pynvml` `FutureWarning` at import site using `warnings.catch_warnings()` + targeted `filterwarnings(message=r"The pynvml package is deprecated.*", category=FutureWarning)` — filter is narrow (deprecation-banner text only) and scoped to the import block; TD textport confirmed clean
- Consolidated all `CUDALINK_*` env reads in `src/cuda_link/` behind `_env.env_bool()` / `_env.env_int()` / `_env.env_str()` helpers; `monkeypatch.setenv` now works reliably in tests
- Consolidated the `parexecute` reinit dance into `reconfigure_and_reinit()` helper
- Retired `CUDAIPCExporter` shim (v1.7.0 removal, scheduled in v1.6.0 deprecation notice)
- Deleted dead `debug_utils.py` (zero importers)
- Refreshed example `.toe` (NVMLObserver + TDConfig DATs); pushed 10 commits to `origin/release/cuda-link-v1.6.0`

#### Technical Details
The TD graphs root cause was purely a "landed opt-in pending soak" default — the graph code path itself was byte-identical to the proven Python sender path (shared `cuda_graphs.py` mixin, same `cudaStreamCaptureModeRelaxed` C2 fix). Three auto-fallback sites (`TDSender.py:401/474/863`) silently degrade to `cudaMemcpyAsync` on any capture or launch failure. Flipping the default required changing both the dataclass default and the `from_env()` fallback simultaneously, since `from_env()` always runs at TD startup and would have overridden the dataclass default to `False` even after the flip.

The pynvml suppression is effective only when `nvml_observer` is the first importer of `pynvml` in the process — which holds on the TD path (extension init runs before any torch-loading operator). In pure-Python contexts where torch imports `pynvml` first, the warning fires once before our filter installs; documented in the updated docstring.

Log analysis also surfaced a float32 GPU memcpy bandwidth anomaly: 8 MB uint8 copies at ~420 GB/s (device-memory-limited) but 32 MB float32 copies at ~36–73 GB/s (PCIe-class). Hypothesis: 32 MB × 3 IPC slots (96 MB total) exceeds WDDM device-resident budget and is host-staged. Noted; not a graphs regression.

#### Files Modified
- `src/cuda_link/nvml_observer.py` — Added `import warnings`; wrapped `import pynvml` in `catch_warnings()` + targeted `filterwarnings`; updated docstring Note
- `td_exporter/NVMLObserver.py` — Byte-identical mirror regenerated via `sync_td_wrapper.py`
- `td_exporter/TDConfig.py` — `use_graphs` default `False → True`; `CUDALINK_TD_USE_GRAPHS` env default `"0" → "1"`
- `td_exporter/TDSender.py` — `[GRAPHS_INIT]` diagnostic log; one-shot `_runtime_load_logged` flag
- `td_exporter/TDReceiver.py` — One-shot `_runtime_load_logged` flag
- `src/cuda_link/cuda_ipc_wrapper.py` — cudart DLL probe order: `12 → 11 → 110`; docstring updated
- `td_exporter/CUDAIPCWrapper.py` — Mirror regenerated
- `src/cuda_link/_env.py` — `env_bool()` / `env_int()` / `env_str()` helpers
- `src/cuda_link/exporter.py` — All `CUDALINK_*` reads migrated to `_env` helpers
- `README.md` — `CUDALINK_TD_USE_GRAPHS` updated to reflect default-ON
- `CHANGELOG.md` — `[Unreleased]` entries: Added, Fixed (×4), Changed (×2)
- `CUDA_Link_Example.toe` — Refreshed NVMLObserver + TDConfig DATs

---

### Session: 2026-05-20 - CheckerBarrier Port+Adapter Deepening

**Primary Achievement**: Applied ADR-0001 Port+Adapter template to the Checker side of `activation_barrier.py`, introducing `BarrierShmPort`, `CheckerOutcome`, and `RealShmAdapter` to enable no-SHM unit testing of the activation-barrier hot path.

#### Key Accomplishments
- Ran the full `/improve-codebase-architecture` skill: explored 9 deepening candidates across `src/cuda_link/`, `td_exporter/`, and `tests/`; grilled Candidate 4 (activation barrier) through 5 design decisions before writing the plan
- Deepened `CheckerBarrier` onto a `BarrierShmPort` Protocol — `evaluate() -> CheckerOutcome` replaces the collapsed `should_skip_publish() -> bool`; bool wrapper kept for backwards compat
- Added `RealShmAdapter` (production, delegates to unchanged module-level SHM-IO functions) and `FakeShmAdapter` (test double in `tests/conftest.py`)
- Wrote 21 new tests in `test_activation_barrier_checker.py` covering all 5 `CheckerOutcome` values, lazy-attach semantics, log throttling, and error-swallowing paths — all without real `SharedMemory`
- Preserved byte-identical TD mirror: `td_exporter/ActivationBarrier.py` sha256-equals canonical (no relative imports introduced)
- Fixed two `test_device_affinity.py` tests that had been patching the now-removed `should_skip_publish` call site; updated to patch `evaluate` returning `CheckerOutcome.SKIP_ACTIVE`
- Committed and pushed to `release/cuda-link-v1.6.0`

#### Technical Details
The key design constraint was the byte-identical TD mirror: adding `_activation_barrier_port.py` + `_activation_barrier_adapters.py` as separate files would have introduced relative imports and broken the `sync_td_wrapper.py` byte-identical check. Resolution: collapse `BarrierShmPort` + `RealShmAdapter` + `CheckerOutcome` into `activation_barrier.py` itself, keeping `FakeShmAdapter` in `tests/conftest.py`. This preserves the invariant that the TD-side `ActivationBarrier.py` is a verbatim copy of the canonical, while still giving the `BarrierShmPort` seam that enables fake-driven testing.

`HolderBarrier` (TD-Sender side) was intentionally left untouched — it will likely evaporate when Candidate 1 (TDSender collapse onto the mirrored `Exporter`) is implemented.

#### Files Modified
- `src/cuda_link/activation_barrier.py` — Added `CheckerOutcome` enum, `BarrierShmPort` Protocol, `RealShmAdapter` adapter; refactored `CheckerBarrier` to use Port + return enum
- `src/cuda_link/exporter.py` — Hot-path call updated from `should_skip_publish()` to `evaluate().should_skip`
- `td_exporter/ActivationBarrier.py` — Regenerated byte-identical mirror via `sync_td_wrapper.py`
- `td_exporter/Exporter.py` — Regenerated mirror (reflects exporter.py hot-path change)
- `tests/conftest.py` — Added `FakeShmAdapter` dataclass
- `tests/test_activation_barrier_checker.py` — New: 21 no-SHM tests for `CheckerBarrier` + `CheckerOutcome`
- `tests/test_device_affinity.py` — Fixed 2 mocks: `should_skip_publish` → `evaluate` returning `CheckerOutcome.SKIP_ACTIVE`
- `CHANGELOG.md` — Added `[Unreleased]` entry

---
