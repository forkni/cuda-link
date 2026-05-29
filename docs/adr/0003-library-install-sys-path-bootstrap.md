# ADR-0003: Library-install sys.path bootstrap (adopt alternative 1C)

**Status**: Accepted (implemented 2026-05-29)
**Date**: 2026-05-29
**Supersedes**: The "1C — TD path-shim" rejected alternative in ADR-0002 (now adopted).
**Applies to**: `td_exporter/CUDALinkBootstrap.py`, `td_exporter/CUDAIPCExtension.py`,
`install_td_library.cmd`, `tests/test_td_bootstrap.py`, `docs/TOX_BUILD_GUIDE.md`.

---

## Context

ADR-0002 (`byte-identical-td-mirror`) introduced `scripts/sync_td_wrapper.py` to keep 14
PascalCase mirror modules in `td_exporter/` in sync with their canonical `src/cuda_link/`
counterparts. These mirrors exist because TouchDesigner loads sibling Text DATs by **bare
module name** from a flat COMP namespace, not from a package on `sys.path`.

ADR-0002 explicitly evaluated and deferred **alternative 1C**:

> *A loader Text DAT adds `cuda_link` to `sys.path`, eliminating the need for
> `td_exporter/Exporter.py` entirely. Rejected because it changes the deployment story
> (TD users currently drop Text DATs into their COMP without any external `cuda_link`
> installation). May be revisited when v2.0 drops the drop-in-Text-DAT guarantee.*

The library-install deployment story has now been established as the primary path. Users can
install `cuda_link` into a stable external folder using `install_td_library.cmd`, point
`CUDALINK_LIB_PATH` at it, and omit the 14 mirror Text DATs from the `.tox` entirely.

## Decision

Adopt alternative 1C as an **additive library mode** alongside the existing classic/fallback
mode (ADR-0002 mirrors retained):

1. A new `td_exporter/CUDALinkBootstrap.py` Text DAT is added to the COMP.
2. It is the **first import** in `CUDAIPCExtension.py` (line 18, before `import contextlib`).
3. At COMP init it:
   a. Reads `CUDALINK_LIB_PATH` from the environment and injects it onto `sys.path`.
   b. Imports `cuda_link` from the installed package.
   c. Registers each of the 14 mirror module names (`Env`, `SHMProtocol`, `Exporter`, …) in
      `sys.modules` as aliases to the corresponding installed submodule
      (`cuda_link._env`, `cuda_link.shm_protocol`, `cuda_link.exporter`, …).
   d. If any step fails (package not installed, import error), silently no-ops — the existing
      sibling mirror Text DATs handle all bare-name imports as before.

4. The 14 mirror modules in `td_exporter/` and `scripts/sync_td_wrapper.py` are kept
   **unchanged**. The fallback / classic deployment (all DATs in the COMP, no install) remains
   fully functional without any modifications.

### Why sys.modules aliasing rather than editing the glue files

The "1B" alternative (try/except in each glue import) was rejected in ADR-0002 because it
leaks TD-environment awareness into canonical source. The equivalent here — adding try/except
blocks to `TDConfig`, `TDSender`, `TDReceiver` — has the same problem plus a concrete scale
issue: `TDReceiver.py` imports 15+ symbols from `SHMProtocol` (including the private
`_ST_BBH`). Reproducing that list across both import styles would be fragile and noisy.

`sys.modules` aliasing keeps the 4 glue files byte-for-byte identical to today. The bootstrap
is a single 80-line module that centralises all dual-mode logic. The alias map is explicitly
cross-referenced against `PAIRS` by `tests/test_td_bootstrap.py::test_alias_map_covers_all_pairs`.

### Why this is safe for TD's embedded Python

`cuda_link` is a **pure-ctypes, zero-required-dependency, `py3-none-any`** wheel. There is no
ABI surface, no build step, no C extension. It imports cleanly in TD's embedded Python 3.9/3.11
with no conflict risk. The `__init__.py` torch/cupy/numpy imports are guarded `*_AVAILABLE`
flags — `tests/test_cuda_ipc_importer.py::test_get_frame_without_torch` confirms the package
loads with no optional deps installed.

### Why not subprocess / separate process (the StreamDiffusion model)

The StreamDiffusion integration runs the heavy ML stack out-of-process (subprocess + venv +
OSC bridge) because torch/diffusers/CUDA would conflict with TD's bundled Python. That model
does NOT apply here — the TD sender/receiver code must run **in-process** because it calls
`top.cudaMemory()` and `scriptTOP.copyCUDAMemory()`, which are live TD runtime calls. An
out-of-process design is architecturally incompatible with these operations.

## Alias map

| TD bare name (key) | Installed submodule (value) |
|---|---|
| `Env` | `cuda_link._env` |
| `FrameProfile` | `cuda_link._profile` |
| `CUDAIPCWrapper` | `cuda_link.cuda_ipc_wrapper` |
| `CUDARuntimeTypes` | `cuda_link.cuda_runtime_types` |
| `CUDAGraphs` | `cuda_link.cuda_graphs` |
| `NVMLObserver` | `cuda_link.nvml_observer` |
| `SHMProtocol` | `cuda_link.shm_protocol` |
| `ActivationBarrier` | `cuda_link.activation_barrier` |
| `NVTXShim` | `cuda_link._nvtx` |
| `ExporterPort` | `cuda_link._exporter_port` |
| `ImporterPort` | `cuda_link._importer_port` |
| `CUDAAdapters` | `cuda_link._cuda_adapters` |
| `Exporter` | `cuda_link.exporter` |
| `Importer` | `cuda_link.importer` |

## Consequences

**Positive:**
- Primary `.tox` drops 14 mirror Text DATs (from ~24 to ~10). `.tox` size shrinks significantly.
- `CUDAIPCExtension.py`, `TDSender.py`, `TDReceiver.py`, `TDConfig.py` are byte-for-byte
  unchanged — zero regression risk in the glue layer.
- `install_td_library.cmd` gives users a one-step, no-venv install path.
- The fallback is transparent and automatic: unset `CUDALINK_LIB_PATH` and the classic
  deployment just works.

**Negative / trade-offs:**
- One new file in the COMP (`CUDALinkBootstrap`) and one new import line in
  `CUDAIPCExtension.py`.
- The bootstrap's alias map must be kept in sync with `PAIRS`; the drift-guard test enforces
  this.
- If `CUDALINK_LIB_PATH` is set but points to a stale install (wrong version), library mode
  activates with the wrong submodule versions. Users should re-run `install_td_library.cmd`
  after upgrading. Future work could add version checks.
- The 14 mirror Text DATs remain in the repo (needed for fallback and for the `.tox` assembly
  guide). They continue to be auto-generated by `sync_td_wrapper.py` — the fallback cost is
  maintenance of the sync script, which is unchanged.
