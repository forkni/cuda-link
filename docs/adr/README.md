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
| [0006](0006-stay-pure-python-no-rust.md) | Stay pure-Python — do not rewrite in Rust | Accepted | 2026-06-07 |
| [0007](0007-spout-as-launcher-not-transport.md) | Expose Spout as a sidecar-launcher COMP, not an embedded transport | Accepted | 2026-06-27 |
| [0008](0008-skip-native-d2h.md) | No native work on the D2H readback path | Accepted | 2026-07-04 |
| [0009](0009-cpp-custom-top-in-process.md) | Accept in-process native code inside TD as a C++ Custom TOP | Proposed | 2026-07-04 |
| [0010](0010-park-out-top-graphs.md) | Park the Out TOP CUDA-Graphs path (array-source node update rejected) | Accepted | 2026-07-06 |
| [0011](0011-in-top-graphs-keyed-cache.md) | Port CUDA Graphs to the In TOP via a keyed graph-exec cache | Accepted | 2026-07-06 |

## Adding a new ADR

Copy the structure from an existing ADR (title / status / date / applies-to / context / decision /
rejected alternatives / consequences / reopen condition). Number sequentially. Add a row to the
table above and a bullet to the ADR list in
[`docs/ARCHITECTURE.md`](../ARCHITECTURE.md#design-decisions) §Design Decisions.
