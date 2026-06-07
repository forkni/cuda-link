# Architecture Decision Records

This directory contains ADRs for load-bearing design decisions in `cuda-link`. Each ADR records
why a decision was made, what alternatives were considered, and under what conditions the decision
should be revisited. See `CONTEXT.md §Architecture references` for a richer set of cross-links.

| # | Title | Status | Date |
|---|-------|--------|------|
| [0001](0001-port-adapter-deepening.md) | Port + Adapter deepening template | Accepted | 2026-05-20 |
| [0002](0002-byte-identical-td-mirror.md) | Byte-identical TD mirror + sync-script rewrite mode | Accepted | 2026-05-20 |
| [0003](0003-library-install-sys-path-bootstrap.md) | Library-install sys.path bootstrap (1C) | Accepted | 2026-05-29 |
| [0004](0004-legacy-cuda-ipc-over-vmm.md) | Legacy CUDA Runtime IPC over VMM driver API | Accepted | 2026-05-31 |
| [0005](0005-static-typing-hardening.md) | Static type-checking hardening (scoped suppression + CI gate) | Accepted | 2026-06-07 |

## Adding a new ADR

Copy the structure from an existing ADR (title / status / date / applies-to / context / decision /
rejected alternatives / consequences / reopen condition). Number sequentially. Add a row to the
table above and a bullet to `CONTEXT.md §Architecture references`.
