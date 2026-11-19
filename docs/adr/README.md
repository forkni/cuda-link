# Architecture Decision Records

This directory contains ADRs for load-bearing design decisions in `cuda-link`. Each ADR records
why a decision was made, what alternatives were considered, and under what conditions the decision
should be revisited. See [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md#design-decisions)
§Design Decisions for a richer set of cross-links.

| # | Title | Status | Date |
|---|-------|--------|------|
| [0001](0001-port-adapter-deepening.md) | Port + Adapter deepening template | Accepted | 2026-05-20 |
| [0002](0002-byte-identical-td-mirror.md) | Byte-identical TD mirror + sync-script rewrite mode | Accepted | 2026-05-20 |
| [0003](0003-library-install-sys-path-bootstrap.md) | Library-install sys.path bootstrap (1C) | Accepted | 2026-05-29 |
| [0004](0004-legacy-cuda-ipc-over-vmm.md) | Legacy CUDA Runtime IPC over VMM driver API | Accepted | 2026-05-31 |
| [0005](0005-static-typing-hardening.md) | Static type-checking hardening (scoped suppression + CI gate) | Accepted | 2026-06-07 |
| [0006](0006-stay-pure-python-no-rust.md) | Stay pure-Python — do not rewrite in Rust | Superseded by 0012 | 2026-06-07 |
| [0008](0008-skip-native-d2h.md) | No native work on the D2H readback path | Accepted | 2026-07-04 |
| [0009](0009-cpp-custom-top-in-process.md) | Accept in-process native code inside TD as a C++ Custom TOP | Proposed | 2026-07-04 |
| [0010](0010-park-out-top-graphs.md) | Park the Out TOP CUDA-Graphs path (array-source node update rejected) | Accepted | 2026-07-06 |
| [0011](0011-in-top-graphs-keyed-cache.md) | Port CUDA Graphs to the In TOP via a keyed graph-exec cache | Accepted | 2026-07-06 |
| [0012](0012-native-extension-in-core-wheel.md) | Fold the native extension into the core wheel | Accepted | 2026-07-07 |
| [0013](0013-prebuilt-wheel-distribution.md) | Prebuilt wheel distribution — Windows-only, cp311 native + py3-none-any fallback | Accepted | 2026-07-07 |
| [0014](0014-project-anchored-install-resolution.md) | Project-anchored install resolution with a mirror version stamp (amends 0003) | Accepted | 2026-09-23 |
| [0015](0015-shm-protocol-v0-6-slot-meta.md) | SHM protocol v0.6 — per-slot metadata, 8-aligned version, sender process ID | Proposed | 2026-11-16 |

## Numbering gaps (0007)

This number is not missing: the ADR was written on a feature branch that has not merged,
and the number stays reserved so references (for example in ADR-0009 and ADR-0012) keep
resolving. It lands with its branch.

| # | Title | Branch |
|---|-------|--------|
| 0007 | Spout as launcher, not transport | `feat/spout-bridge` |

## Adding a new ADR

Copy the structure from an existing ADR (title / status / date / applies-to / context / decision /
rejected alternatives / consequences / reopen condition). Number sequentially. Add a row to the
table above and a bullet to the ADR list in
[`docs/ARCHITECTURE.md`](../ARCHITECTURE.md#design-decisions) §Design Decisions.
