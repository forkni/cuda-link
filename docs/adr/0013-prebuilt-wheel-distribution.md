# ADR-0013: Prebuilt wheel distribution — Windows-only, cp311 native + py3-none-any fallback

**Status**: Accepted
**Date**: 2026-07-07
**Applies to**: `.github/workflows/release.yml`, `scripts/install_td_library.py`,
`utils/build_wheel.cmd`, root `CMakeLists.txt`. Closes a gap left by
[ADR-0012](0012-native-extension-in-core-wheel.md).

---

## Context

[ADR-0012](0012-native-extension-in-core-wheel.md) folded the separately-distributed
`cuda-link-native` sidecar into the single `cuda-link` core wheel, compiled via
`scikit-build-core` + `pybind11` + MSVC. That ADR's "Consequences" section framed install
as `pip install .` (or `utils\build_wheel.cmd`) **run on the user's own machine** — it did
not address what happens when that machine has no C++ toolchain.

It turned out that gap was load-bearing:

- `BUILD_NATIVE_WAITER` defaults **ON** on Windows (`${WIN32}`), and until this ADR's
  companion CMake fix, `project(cuda_link LANGUAGES CXX)` forced MSVC detection at
  configure time regardless of that option's value — so even the pure-Python fallback
  build failed without a compiler installed.
- `scripts/install_td_library.py`'s `resolve_wheel()` **auto-built** via
  `utils/build_wheel.cmd` whenever `dist/` (git-ignored) had no matching wheel — the
  default state on a fresh checkout — silently requiring MSVC at *install* time, not just
  *build* time.
- End users are **StreamDiffusionTD** artists, not cuda-link developers. Their machines
  almost never have Visual Studio installed. `cuda_link` runs in the StreamDiffusionTD
  **receiver** process, which by convention uses a dedicated **Python 3.11.9 venv** — never
  TouchDesigner's own bundled 3.11.10 interpreter (pip-installing into that is a documented
  anti-pattern; see mode 5 below).

Compiling on the end-user machine was never actually a requirement — it was an
unintended side effect of ADR-0012 not distinguishing "the project builds a compiled
wheel" from "every install runs that build locally."

## Decision

**Ship prebuilt binary wheels; end-user machines never compile.** `cuda-link` is
Windows-only (per ADR-0004/ADR-0012's driver-level constraints), so distribution targets
a single platform:

- **CI (`release.yml`, `windows-latest`)** builds and publishes **two** wheels per
  release, both attached as GitHub Release assets:
  - **Native**: `cuda_link-<version>-cp311-cp311-win_amd64.whl` — `BUILD_NATIVE_WAITER`
    ON (its default), carrying the compiled `_native_waiter` accelerator. A single
    `cp311` wheel imports on any 3.11.x patch (the ABI tag is minor-version-scoped, so it
    covers both StreamDiffusionTD's pinned 3.11.9 and TD's bundled 3.11.10 without a
    separate build per patch release).
  - **Fallback**: `cuda_link-<version>-py3-none-any.whl` — built with
    `-C cmake.define.BUILD_NATIVE_WAITER=OFF -C wheel.py-api=py3 -C wheel.platlib=false`,
    no compiler involved. `wheel.py-api=py3` sets the Python/ABI tag; `wheel.platlib=false`
    is what actually earns the `any` platform tag — scikit-build-core defaults to platlib
    (hence `win_amd64`) whenever `wheel.cmake` is true, regardless of `py-api`, since it
    doesn't auto-detect that the `BUILD_NATIVE_WAITER=OFF` CMake configuration installs no
    platform-specific files. Covers every other interpreter version. The 2026-07-06
    timer-nap fix already brought
    native/Python wait-path parity to within a marginal <1–5%, so this fallback loses
    almost nothing functionally — see ADR-0012's Context.
  - CI's first real compile-and-run of the C++ state machine: after building the native
    wheel, install it and run `pytest tests/core/test_native_state_machine.py -m
    requires_native` — that file needs only the compiled extension, no GPU/cudart
    (contrast `tests/cuda/test_native_smoke.py`, which needs a loaded cudart and still
    can't run on GPU-less runners).
- **`scripts/install_td_library.py`** resolves a wheel **per install target**, not once
  under its own interpreter: each mode (venv, system Python, conda, etc.) now probes its
  *target* interpreter's version after determining that target, then picks
  `cp311-cp311-win_amd64` for a 3.11 target or `py3-none-any` otherwise — fixing a
  latent tag-blind bug where `_find_wheel()` picked the newest-mtime wheel in `dist/`
  with no regard for ABI tag, so a stray fallback wheel could silently mask the native
  path in a 3.11 environment. Resolution order: `--wheel <path>` override → a
  tag-matched wheel already in `dist/` → auto-download the matching GitHub Release asset
  for the installed `__version__` → (only with the new `--build` flag) compile locally
  via `utils\build_wheel.cmd`. Without `--build`, a target with no available wheel exits
  with an actionable message instead of silently invoking a build.
- **Two supported install scenarios**: a system Python 3.11 install (mode 4), and a
  StreamDiffusionTD-style venv pinned to Python 3.11.9 (mode 2). Both resolve the native
  wheel automatically.
- **Mode 5 (install into TD's own bundled Python) is deprecated** — kept functional, but
  the installer now prints a warning steering to mode 2 or mode 4. Pip-installing into
  TD's interpreter directly remains a known anti-pattern independent of this ADR; this
  distribution rework is a natural point to start discouraging it.
- **`utils/build_wheel.cmd`** is now explicitly a dev/CI tool, not an end-user step. It
  gained a `vswhere`-based MSVC preflight (fails fast with actionable remediation —
  install VS Build Tools, use `nowaiter`, or grab a prebuilt wheel — instead of a
  cryptic CMake configure error) and a `nowaiter`/`--fallback` argument that passes
  `-C cmake.define.BUILD_NATIVE_WAITER=OFF -C wheel.py-api=py3 -C wheel.platlib=false`
  through to `python -m build`.
- **Root `CMakeLists.txt`** changed `project(cuda_link LANGUAGES CXX)` to
  `project(cuda_link LANGUAGES NONE)` with `enable_language(CXX)` moved inside the
  `if(BUILD_NATIVE_WAITER)` branch, after the early `return()` for the OFF case. This
  makes the fallback build genuinely compiler-free — previously even
  `-DBUILD_NATIVE_WAITER=OFF` failed on a machine with no MSVC, because `project()`
  detected a compiler before the option was ever read. `WIN32`/`UNIX`/`APPLE` are still
  set correctly by `project()` with `LANGUAGES NONE` (they reflect the target platform,
  not compiler availability), so the option's `${WIN32}` default is unaffected.

## Rejected alternatives

- **Bootstrap a compiler on the end-user machine** (e.g. auto-install VS Build Tools,
  bundle a portable MSVC/clang-cl). Rejected: heavyweight, slow, fragile across Windows
  versions/permissions, and unnecessary — a `cp311` platform wheel is a solved
  distribution mechanism that needs no local toolchain at all.
- **PyPI publishing.** Rejected for this pass — the installer + GitHub Release assets
  already cover both target scenarios, and PyPI adds packaging/versioning overhead
  (yanking, trusted publishing setup) with no benefit yet, since `cuda-link` isn't
  intended for `pip install cuda-link` from a stranger's requirements.txt today. Revisit
  if that changes.
- **Multiple native wheels across the full Python matrix (3.9–3.12).** Rejected: the two
  supported scenarios (system Python 3.11, StreamDiffusionTD's pinned 3.11.9 venv) are
  both `cp311`. Building native wheels for versions neither supported scenario uses would
  add CI cost and release-asset surface for no real-world benefit — those targets already
  get a fully-functional fallback wheel.
- **Statically set `wheel.py-api = "py3"` / `wheel.platlib = false` in `pyproject.toml`'s
  `[tool.scikit-build]`.** Rejected — that block applies to every build, so it would also
  mislabel the native build's own wheel tag (forcing it to `py3-none-...` or purelib,
  losing the `cp311-cp311-win_amd64` tag the compiled extension needs). The per-invocation
  `-C wheel.py-api=py3 -C wheel.platlib=false` config-settings, used only on fallback build
  invocations (`build_wheel.cmd nowaiter`, CI's fallback step), achieve the same result
  without corrupting the native build's tag.

## Consequences

- End-user machines — StreamDiffusionTD artists' venvs and system Python installs —
  never need MSVC. `pip install`-adjacent flows always resolve a wheel that matches
  their interpreter, either the accelerated native build or the fully-functional
  fallback.
- `install_td_library.py --build` remains available for cuda-link developers on an MSVC
  dev box, or anyone who wants to build from source deliberately — it is no longer the
  default, silent behavior.
- CI gains its first genuine compile-and-exercise of the C++ state machine
  (`test_native_state_machine.py` under `-m requires_native`) on every tagged release
  and on-demand via `workflow_dispatch` — closing the coverage gap ADR-0012 noted (that
  marker was previously only ever run manually on a Windows dev box).
- `docs/README.md`'s install section now leads with "download the prebuilt wheel" (or
  let the installer auto-fetch it) rather than "build locally."
- The `e57d1fc` commit message (the ADR-0012 fold) stated no MSVC toolchain was
  available on end-user machines as if that were already handled; it wasn't, at the time
  it was written. That commit is already pushed and is not being rewritten — this ADR and
  the CHANGELOG entry for this change are the correction of record.

## Reopen condition

Revisit if the supported-scenario set grows beyond system Python 3.11 / StreamDiffusionTD
3.11.9 (e.g. a third pinned Python version for a different downstream integration) —
that would mean adding another native wheel to the CI build matrix, not changing this
distribution mechanism itself. Revisit PyPI publishing if `cuda-link` ever needs to be
installable as a transitive dependency of another public package.
