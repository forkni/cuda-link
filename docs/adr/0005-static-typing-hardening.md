# ADR-0005: Static type-checking hardening (scoped suppression + CI gate)

**Status**: Accepted
**Date**: 2026-06-07
**Applies to**: `pyproject.toml` (`[tool.pyrefly]`),
`.github/workflows/typecheck.yml`, `.pre-commit-config.yaml` (pyrefly hook),
all modules in `src/cuda_link/`.

---

## Context

This decision arose from exploring whether to rewrite `cuda-link` in Rust for
performance/reliability. That investigation concluded Rust is not justified —
performance is hardware-bound (PCIe/GPU; ~4% of a frame is Python overhead) and
a rewrite would forfeit the pure-Python, zero-dependency, drop-into-TD
deployment story. The one real reliability gap Rust would have addressed — the
weakly-typed ctypes seam — is addressable in Python far more cheaply, which is
what this ADR does.

Two concrete weaknesses existed before this change:

1. **One package-wide suppression blanket.** `[tool.pyrefly]` disabled 9 error
   categories (`missing-attribute`, `bad-argument-type`, `not-callable`,
   `no-matching-overload`, `bad-assignment`, …) via a single
   `matches = "src/cuda_link/**"` sub-config. This silenced the few genuinely
   ctypes-heavy modules **and** the well-typed logic modules
   (`shm_protocol.py`, the activation barrier, the `_port` Protocols), so a real
   type error in clean code merged unnoticed.
2. **No CI gate.** `pyrefly` ran only as a local pre-commit hook
   (`always_run: false`), so type errors could land via any push that skipped
   pre-commit.

A win32 evaluation (`pyrefly check --python-platform win32`) showed the true
error surface is dominated by **deliberate, documented idioms**, not bugs:

- **Optional-import-as-`None`** — `torch`/`numpy`/`cupy`/`pynvml` bound to `None`
  under `try/except` and guarded at runtime (the `NoneType has no attribute …`
  cluster in `importer.py`, `nvml_observer.py`).
- **ctypes `c_void_p` ↔ `int` pointer duality** (`exporter.py`, `cuda_graphs.py`).
- **Dynamic `CUDARuntimeAPI` delegation** via `CTypesCUDAAdapter.__getattr__`,
  an explicit design choice (see `_cuda_adapters.py` — "without maintaining a
  mechanical list of one-line forwarders").
- **Optional-until-attached** `SharedMemory` buffers.

## Decision

Replace the blanket with a **check-strictly-by-default, suppress-narrowly**
posture:

1. **`python-platform = "win32"`** — cuda-link is Windows-only; this resolves
   `ctypes.windll`/`WINFUNCTYPE` and platform-gated code on any host (incl. the
   Linux CI runner), so the host OS no longer changes results.
2. **`replace-imports-with-any`** for `torch`, `cupy`, `pynvml`, `nvtx`,
   `ml_dtypes` — the genuinely optional, runtime-guarded GPU deps. CI needs no
   CUDA toolchain; the runtime `None`-guards remain authoritative. `numpy` is
   **not** stubbed (it ships real stubs and stays checked).
3. **Per-file `[[tool.pyrefly.sub-config]]` blocks** that each disable only the
   categories a specific module triggers under an accepted idiom above. Every
   other module — `shm_protocol.py`, `_exporter_port.py`, `_importer_port.py`,
   `_env.py`, `_profile.py`, `_console.py`, `_nvtx.py`, `cuda_runtime_types.py`,
   `__init__.py` — is now **fully type-checked**.
4. **CI gate** — `.github/workflows/typecheck.yml` runs `pyrefly check
   src/cuda_link/` on every PR/push touching the package or its config.
5. **GPU-free Protocol drift guard** — `test_ctypes_adapter_api_covers_protocol_without_gpu`
   asserts `CUDARuntimeAPI` implements every `CudaPort` member. The
   `__getattr__` delegation hides drift from both the (suppressed) checker and
   the existing `requires_cuda` conformance test; this guard catches it without
   a GPU.

Policy going forward: **no new package-wide category blanket.** New suppressions
must be a narrow per-file sub-config or a line-level `# type: ignore[code]`
(the pattern already used in `nvml_observer.py`, `cuda_ipc_wrapper.py`).

## Rejected alternatives

- **Rewrite the ctypes seam (or library) in Rust.** Negligible performance
  upside; forfeits the pure-Python wheel; the TD producer side is locked to
  TouchDesigner's embedded CPython. The compile-time-safety argument is real but
  does not outweigh the deployment cost.
- **Add explicit one-line forwarders to `CTypesCUDAAdapter`** to make it
  statically satisfy `CudaPort`. Directly contradicts the documented design in
  `_cuda_adapters.py`. The runtime drift-guard test covers the same risk without
  the boilerplate.
- **Refactor the optional-import-`None` idiom** across `importer.py`/
  `nvml_observer.py`. High churn in a production file for false positives that a
  narrow per-file suppression handles cleanly.

## Consequences

- Type regressions in the clean logic modules are now caught locally and in CI.
- The accepted ctypes/optional idioms remain ergonomic but are suppressed
  precisely, per file, with a documented rationale.
- The per-file category lists were derived from a win32 evaluation; the first CI
  run is authoritative and the lists should be tightened/loosened from it.

## Follow-ups

- **Pytest CI job — done** (`.github/workflows/tests.yml`). Runs the
  `not requires_cuda` suite on Python 3.10–3.12. The previously torch-fragile
  tests were made robust: `test_torch_buffers_*` use `pytest.importorskip("torch")`,
  and the stale `test_get_frame_without_torch` (which asserted a removed
  "torch is required" contract through the unconnected deprecated wrapper) was
  modernized to assert the current graceful `None` return. The suite is green
  both with and without torch installed, so torch is a best-effort CPU install
  in CI. Extending the matrix to Python 3.9 is left for when a 3.9 runner is
  validated.
- **Extend type checking to `td_exporter/` — done.** `pyrefly check` (project
  mode) now covers `td_exporter/` engine files and the 14 generated mirrors.
  Pure-TD glue scripts (callbacks, example launchers) are excluded via
  `project-excludes`. Bare TD ambient globals (`op`, `run`, `CUDAMemoryShape`)
  are resolved through `if TYPE_CHECKING: from _td_builtins import …` stubs in
  the two non-mirror files that reference them (`CUDAIPCExtension.py`,
  `TDHost.py`). The `td` module (COMP, TOP, ui) is handled via
  `replace-imports-with-any = ["td", "td.*"]`. Mirror files' suppressions are
  keyed to `td_exporter/<name>.py` paths (not their src twins) so the
  mirror invariant is respected. CI gate updated to use project mode and added
  `td_exporter/**` to path triggers. Pre-commit hook widened to
  `^(src/cuda_link|td_exporter)/.*\.py$`.

## Reopen condition

Revisit if pyrefly is replaced, if the optional-dependency strategy changes, or
if the excluded pure-TD glue scripts (`callbacks_template.py`,
`parexecute_callbacks.py`, `script_top_callbacks.py`, example launchers) are
brought under checking (which would require per-file handling of the full
bare TD-global surface in those files).
