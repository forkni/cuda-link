# ADR-0012: Fold the native extension into the core wheel

**Status**: Accepted
**Date**: 2026-07-07
**Applies to**: packaging (`pyproject.toml`, root `CMakeLists.txt`), the R5 wait-backend seam
(`src/cuda_link/_wait_backend.py`, `_native_loader.py`, `_cpp/native_waiter.cpp`), and the
deployment model documented in [ADR-0002](0002-byte-identical-td-mirror.md) and
[ADR-0003](0003-library-install-sys-path-bootstrap.md). Supersedes
[ADR-0006](0006-stay-pure-python-no-rust.md)'s "narrow optional extension" framing.

---

## Context

[ADR-0006](0006-stay-pure-python-no-rust.md) kept `cuda-link` pure Python, with a single
escape hatch: "a narrow, optional PyO3 [or equivalent] extension for a consumer-side hot
path, and only if profiling later proves Python overhead material." PLAN-002 (R5,
2026-07-04) exercised that hatch as a separately-distributed C++/pybind11 sidecar package,
`cuda-link-native` (`native/`), installed alongside the core wheel and defaulting **on** in
`install_td_library.py`.

Two things changed since:

1. **The performance justification for a *separate* package evaporated.** The 2026-07-06
   timer-nap fix (commit `431794e`) made the native block phase nap on
   `CREATE_WAITABLE_TIMER_HIGH_RESOLUTION` — the same primitive CPython ≥3.11's
   `time.sleep` uses — bringing native and pure-Python paths to measured parity. A
   two-wheel install story no longer buys anything over one wheel.

2. **The roadmap now requires more native code, deliberately.** Future development
   includes C++ native TouchDesigner operators and a Spout bridge (see
   [ADR-0007](0007-spout-as-launcher-not-transport.md) and
   [ADR-0009](0009-cpp-custom-top-in-process.md) for the in-process-native precedent this
   already set on the producer side) — the project can no longer stay pure Python without
   sacrificing that work. This is a conscious decision to supersede ADR-0006's framing:
   absorb the native extension into the single `cuda-link` core wheel, rather than retire
   it or keep it as an optional sidecar, so future native work has one home.

## Decision

**Fold `cuda-link-native` into `cuda-link`.** One wheel, whose build compiles
`_native_waiter` on Windows (via root `CMakeLists.txt` + `scikit-build-core` +
`pybind11`) and degrades to a pure-Python wheel everywhere else — the same
`BUILD_NATIVE_WAITER`-gated CMake early-return the standalone package used, just
installing into `cuda_link/` directly instead of a separate `cuda_link_native/`
top-level package.

- `native/src/cuda_link_native/_backend.py` → `src/cuda_link/_wait_backend.py`
- `native/src/cuda_link_native/_native.py` → `src/cuda_link/_native_loader.py`
- `native/src/cuda_link_native/_cpp/native_waiter.cpp` → `src/cuda_link/_cpp/native_waiter.cpp`
- `native/CMakeLists.txt` + `native/pyproject.toml` → merged into the root equivalents
- `native/tests/*` → `tests/core/` and `tests/cuda/` (main test tree)
- `cuda_link_native` as an importable package name ceases to exist
- `install_td_library.py`'s `--native`/`--no-native` two-wheel logic is removed — there is
  only one wheel to resolve and install now
- The `importer.py` seam keeps its deferred `try/except ImportError` around
  `from cuda_link._native_loader import load_native_backend` — TD `.tox` (Text-DAT) mode
  has no pip install step and must keep degrading gracefully exactly as before

## Rejected alternatives

- **Keep the two-wheel sidecar as-is.** Was the ADR-0006 escape-hatch default until now;
  rejected because the performance gap that justified an *optional* extra install step
  closed (native/Python parity post-timer-nap-fix), while the roadmap now calls for more
  native code that needs a stable home — not more sidecar packages.
- **Retire the native extension entirely, back to pure-Python-only.** Rejected per explicit
  direction: future C++ TouchDesigner operators and the Spout bridge require native code as
  a first-class citizen of the build, not something to walk back from.
- **A new namespace/subpackage split (`cuda_link.native`) instead of a full fold.** Rejected
  as unnecessary complexity — `scripts/sync_td_wrapper.py`'s import rewriter only supports
  single-level relative imports, so flat modules directly under `src/cuda_link/` (absolute
  dotted imports in the seam) are simplest and already match how the TD mirror system
  expects the tree to be shaped.

## Consequences

- Single install/build/CI path: `pip install .` (or `utils\build_wheel.cmd`) produces one
  wheel; on Windows with an MSVC C++17 toolchain it is a platform wheel
  (`cp3XX-cp3XX-win_amd64`) carrying the compiled extension, otherwise a pure-Python wheel.
  Neither installers nor CI need to reason about "core" vs. "native" wheels anymore.
  ADR-0006's headline property — the *library* stays free of runtime dependencies — is
  unchanged; what changes is that the **build toolchain** is no longer optional in the same
  way once native TD operators land, since those will not degrade as gracefully as this
  wait-backend accelerator does.
- `requires_native`-marked tests live in the main `tests/` tree and are deselected on CI
  (`ubuntu-latest`, which never compiles the Windows-only extension) exactly as the old
  `native-tests` CI job's tests were — no CI coverage regression from the fold.
- Future native TouchDesigner operators and the Spout bridge (per the roadmap direction
  that motivated this ADR) build on the same root `CMakeLists.txt` + `scikit-build-core`
  scaffolding this fold establishes, rather than each spinning up its own sidecar package.
- ADR-0006 remains correct about *why* a full Rust/native rewrite was rejected (hardware-
  bound performance, TD-embedded producer constraint); this ADR only supersedes its
  packaging conclusion for the specific, narrow extension that hatch already permitted.

## Reopen condition

Revisit if a future native component turns out to need a genuinely different install
lifecycle than the core library (e.g. a large CUDA Toolkit build dependency the core wheel
must not require) — that would be grounds for a *new* sidecar, not un-folding this one.
