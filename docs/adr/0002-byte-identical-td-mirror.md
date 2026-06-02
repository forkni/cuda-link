# ADR-0002: Byte-identical TD mirror + sync-script rewrite mode

**Status**: Accepted (implemented 2026-05-20) — see ADR-0003 for the v2.0 adoption of alternative 1C.
**Date**: 2026-05-20
**Applies to**: `scripts/sync_td_wrapper.py`, `td_exporter/`, `tests/support/test_wrapper_sync.py`.

---

## Context

TouchDesigner loads Python modules from sibling Text DATs inside a COMP, not from a package on `sys.path`. This means:

- `td_exporter/` files cannot use relative imports (`from ._foo import …`).
- Derived `td_exporter/` files must import siblings by **bare module name** (`from Foo import …`).
- Any module that does `from . import _nvtx` in its canonical source needs the import rewritten to `import NVTXShim` in the derived TD copy.

The six original byte-identical pairs (`cuda_ipc_wrapper`, `cuda_runtime_types`, `cuda_graphs`, `nvml_observer`, `shm_protocol`, `activation_barrier`) work because none of them use relative imports — they can be byte-copied verbatim.

The deepened `Exporter` (v1.5.0) and `Importer` (v1.5.x) use relative imports (`from ._exporter_port import …`, `from ._importer_port import …`, etc.). This blocked byte-identical mirroring and was the explicit reason the "TDSender collapse + byte-identical `Exporter.py` mirror" step was deferred.

## Decision

Extend `sync_td_wrapper.py` with a **rewrite mode** for pairs where the canonical source uses relative imports:

1. The `PAIRS` list gains a third field, `mode: Literal["byte_identical", "rewrite_relative"]`.
2. For `mode="byte_identical"` pairs, the script copies verbatim (existing behaviour).
3. For `mode="rewrite_relative"` pairs, the script applies a line-level regex transform **before writing** the derived file:
   - `from ._name import ...` → `from <DerivedName> import ...`
   - `from .name import ...` → `from <DerivedName> import ...`
   - `from . import name` → `import <DerivedName>`
   - Multi-line continuations (after a `(\n`) are left untouched — only the `from .X import (` opener is rewritten.
   - Any other relative pattern (`from .. import`, `from ...`) → script exits with an error.
4. The `NAMES` mapping in the script encodes the `canonical_stem → derived_stem` table explicitly (not inferred from filename). This handles irregular cases (`_nvtx → NVTXShim`).
5. `tests/support/test_wrapper_sync.py` uses two check modes:
   - **byte-identical pairs**: compare SHA-256 of canonical and derived (existing).
   - **rewrite pairs**: re-run the sync script in-process and compare the stdout-derived content with the on-disk derived file (no SHA-256; derived is not byte-identical to canonical by design).
6. The pre-commit hook (`scripts/sync_td_wrapper.py --check`) dispatches per-pair based on mode; no external change required.

## Rejected alternatives

**1B — dual-import `try/except` preamble in canonical source**: Each relative import becomes `try: from ._foo import X; except ImportError: from Foo import X`. Rejected because it leaks TD-environment awareness into canonical source; 8+ try/except blocks per file obscure the module's actual dependencies.

**1C — TD path-shim (no derived files for exporter/importer)**: A loader Text DAT adds `cuda_link` to `sys.path`, eliminating the need for `td_exporter/Exporter.py` entirely. **Adopted in ADR-0003 (2026-05-29)** as `CUDALinkBootstrap.py`, which injects `CUDALINK_LIB_PATH` onto `sys.path` and registers `sys.modules` aliases for all 14 mirror names. The drop-in-Text-DAT fallback is preserved alongside it.

## Consequences

**Positive**:
- Canonical source stays clean — no TD-environment awareness, no try/except noise.
- The adapter (sync script) absorbs the transform. Canonical → derived is a deterministic, auditable, re-runnable operation.
- `TDSender` collapse (ADR-0001 step 7) is now unblocked: once `Exporter.py` exists as a derived module in `td_exporter/`, `TDSenderEngine` can collapse to a thin TD-COMP adapter over `Exporter`.
- CI and pre-commit reject hand-edits to derived files.

**Negative / trade-offs**:
- Derived files are no longer byte-identical to canonical — the SHA-256 invariant is replaced by "re-run and diff" for `rewrite_relative` pairs. This is weaker but sufficient: the derived file is still fully determined by the canonical source + the transform.
- The `NAMES` mapping must be kept in sync with the actual filenames. If a canonical file is renamed, the mapping must be updated or `--check` will fail.

## Naming convention

Private canonical modules use a leading underscore (`_exporter_port.py`). Their TD derivatives drop the underscore and PascalCase the stem:

| Canonical | Derived |
|---|---|
| `exporter.py` | `Exporter.py` |
| `importer.py` | `Importer.py` |
| `_exporter_port.py` | `ExporterPort.py` |
| `_importer_port.py` | `ImporterPort.py` |
| `_cuda_adapters.py` | `CUDAAdapters.py` |
| `_profile.py` | `FrameProfile.py` (already exists) |
| `_nvtx.py` | `NVTXShim.py` (already exists, irregular) |

`FrameProfile.py` and `NVTXShim.py` currently ship as byte-identical pairs (they have no relative imports). If they ever gain relative imports, their mode switches to `rewrite_relative` without a name change.
