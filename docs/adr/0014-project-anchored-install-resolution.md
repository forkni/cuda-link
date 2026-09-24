# ADR-0014: Project-anchored install resolution with a mirror version stamp

**Status**: Accepted
**Date**: 2026-09-23
**Amends**: ADR-0003 (discharges its "future work could add version checks" line and demotes
`CUDALINK_LIB_PATH` from the only source to one layer of several).
**Applies to**: `td_exporter/CUDALinkBootstrap.py`, `scripts/sync_td_wrapper.py`,
`tests/td/test_td_bootstrap.py`, `docs/TOX_BUILD_GUIDE.md`, `docs/ENV_VARS.md`.

---

## Context

ADR-0003 made library mode depend on exactly one machine-global input: `CUDALINK_LIB_PATH`
(or, equivalently, whatever `cuda_link` TouchDesigner's global module path already exposes).
That was enough for one project on one machine. It is not enough once cuda-link is installed
more than once:

- The library is used from StreamDiffusionTD (`D:\dev\SDTD_040_Beta`, its own venv) while the
  system Python 3.11 site-packages carries a second, older copy on the same machine. Any TD
  process resolves `cuda_link` to whichever copy comes first on `sys.path`, regardless of which
  project the `.toe` belongs to. Observed: the repo and the mirrors inside the `.tox` were at
  1.12.2 while TD imported 1.12.1.
- Nothing compared the installed package's `__version__` with the mirrors the `.tox` shipped.
  ADR-0003 recorded this as a known risk ("stale install / wrong version") and deferred it.
- Before ADR-0002's relative-import rule was enforced for every mirrored module (commit
  `be95662`), two mirrors could even resolve *their own* dependencies against the wrong copy,
  which is how two `cudaIpcMemHandle_t` classes ended up in one process.
- StreamDiffusionTD had already forked the bootstrap into a 255-line layered resolver with a
  rival-install guard and `last_error` reporting; `CUDAIPCExtension.py` was reading
  `last_error` from a module that, in this repo, never defined it.

## Decision

`CUDALinkBootstrap` becomes a **project-anchored resolver** with three rules.

1. **Layered lookup, first match wins.** The layers are tried lazily, in this order, and a
   later layer is never evaluated once an earlier one has resolved:
   1. an explicit folder passed to `_bootstrap(basefolder=...)` (host integrations);
   2. the `Libpath` custom parameter on the COMP or any ancestor COMP;
   3. `<project.folder>/cuda_link`, `<project.folder>/StreamDiffusion`, then
      `<project.folder>` itself;
   4. `CUDALINK_LIB_PATH` (ADR-0003 compatibility, now a fallback rather than the source);
   5. whatever `sys.path` already provides (TD Preferences module path, pip).

   Every root is probed as `<root>/venv/Lib/site-packages`, `<root>/.venv/Lib/site-packages`,
   `<root>/Lib/site-packages` and `<root>` itself, so a project folder holding a venv, the
   venv itself, a `pip install --target` folder and a bare site-packages directory all work
   without configuration.

2. **Version stamp, checked before import.** `scripts/sync_td_wrapper.py` writes
   `MIRROR_VERSION = "<cuda_link.__version__>"` into the bootstrap alongside the mirror sync,
   and `--check` (the pre-commit hook) fails when the stamp is stale. At resolution time the
   candidate's `__version__` is read from its `__init__.py` on disk; an install whose version
   differs from `MIRROR_VERSION` is **rejected without being imported** and the next candidate
   is tried. Strict equality is deliberate: the mirrors in the `.tox` and the package they alias
   must be the same code, or the ctypes handle classes and the SHM protocol constants can
   silently diverge.

3. **Rival install is a hard stop.** If a `cuda_link` (or any bare alias that is a package
   copy) is already loaded from a different folder than the selected install, resolution aborts
   with `_RivalInstallError`, `last_error` names both folders and tells the user to restart
   TouchDesigner. It does not fall through to later layers, because the only thing a later
   layer could do is silently accept the copy the user did not ask for. A bare alias name
   already owned by a classic mirror Text DAT (one imported ahead of the bootstrap) is refused
   the same way: aliasing the remaining names would leave the COMP with the mirror's copy of
   one module and the package's copy of the rest, the same two-class failure by another route.
   `last_error` names the DAT and the remedy (make the bootstrap the first import, or drop the
   mirror DATs). The same stop applies when nothing was loaded but importing the selected
   install lands somewhere else because an import hook ahead of `sys.path` (an editable
   install's redirecting finder, such as scikit-build-core's) serves the name; `last_error`
   then names the hook, and the modules that import pulled in are dropped again.

When every layer fails, the module records why in `last_error` (one clause per layer, e.g.
`Libpath parameter: not set; CUDALINK_LIB_PATH: cuda_link 1.12.1 under C:\... does not match
the component's 1.12.2`), stays inactive, and the COMP falls back to its mirror DATs exactly as
before. `CUDAIPCExtension` already surfaces `last_error` as the yellow status text.

The module keeps the unconditional run at Text DAT load time and exports `_active`,
`last_error`, `resolved_site_packages` and `MIRROR_VERSION`. The shape matches the
StreamDiffusionTD fork closely enough that the fork can shrink to a thin wrapper passing its
Base Folder as `basefolder`.

## Rejected alternatives

- **Keep `CUDALINK_LIB_PATH` as the only input and document "set it per project".** A user
  environment variable is process-global; two `.toe` files on one machine cannot disagree
  about it, and TD's own preference path always wins when it is set.
- **Warn on version mismatch but activate anyway.** That is the current failure mode with a
  log line attached; the mismatch still produces two copies of the runtime types in one
  process. Rejecting keeps the COMP on its byte-identical mirrors, which are known to match.
- **Compatible-range check (same major.minor).** Not worth the ambiguity: the mirrors are
  generated from the exact same tree as the wheel, so equality is both the simplest rule and
  the only one that is actually guaranteed.
- **Import first, then compare `cuda_link.__version__`.** Importing is the side effect being
  guarded against (module cache, ctypes bindings, `sys.path` order); reading the file is free.
- **Also read StreamDiffusionTD's `%APPDATA%` config JSON.** Host-specific; the `basefolder`
  argument covers it from the host's side without this repo knowing about the file.

## Consequences

**Positive:**

- Each project resolves its own install; a second copy elsewhere on the machine is either
  ignored (wrong version) or refused loudly (already loaded), never mixed in.
- `last_error` finally exists in the repo bootstrap, so the extension's yellow status text
  says which layer failed and why, instead of "Base Folder not set" for every cause.
- ADR-0003's open risk (stale install activates with wrong submodule versions) is closed.
- `CUDALINK_LIB_PATH` keeps working unchanged for existing setups.

**Negative / trade-offs:**

- A version bump now touches one more generated line; forgetting to run the sync script fails
  the pre-commit hook and the `test_mirror_version_stamp_matches_package_version` guard.
- Library mode is stricter: an install that used to activate at 1.12.1 against 1.12.2 mirrors
  now falls back to classic mode until it is upgraded. That is the intended behaviour.
- The `Libpath` parameter has to be added to the COMP (see `docs/TOX_BUILD_GUIDE.md` Step 2).
- Residual gap, out of scope here: the mirror `Importer.py` still probes
  `cuda_link._native_loader` / `cuda_link._wait_backend` with a guarded absolute import (they
  are `CANONICAL_ONLY`, pip-only per ADR-0012). In classic mode that probe still resolves
  against whichever `cuda_link` happens to be importable. A follow-up should route it through
  the same version check.

## Reopen condition

Revisit if cuda-link ever needs to support more than one package version inside one TD process
(for example two COMPs pinned to different releases), or if the mirror DATs are dropped
entirely so there is no longer a second copy to keep in lock-step.
