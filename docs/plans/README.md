# Implementation Plans

Forward-looking implementation plans for cuda-link. Unlike ADRs (which record decisions
already made), a plan describes **work not yet done**: its goal, baseline numbers,
architecture decisions, phase breakdown with exit criteria, verification, and risks.
Plans graduate: when a plan is executed, its load-bearing decisions get recorded as ADRs
and the plan's status flips to Done (or Killed, with the measured reason).

## Index

Recommended execution order is by dependency and risk, **not** by number:

| Order | Plan | Title | Size | Status | Depends on |
|-------|------|-------|------|--------|------------|
| 1 | [PLAN-004](PLAN-004-d2h-tuning.md) | D2H: skip native work, tune configuration | S (days) | Proposed | — |
| 2 | [PLAN-002](PLAN-002-native-waiter.md) | Native notification waiter for the Python consumer | M (1–2 wk) | Implemented (this branch; accept gate MISS documented — see the doc) | — |
| 3 | [PLAN-003](PLAN-003-cuda-bindings-adapter.md) | Optional `cuda.bindings` adapter (benchmark-gated) | M (1–2 wk) | Proposed | — |
| 4 | [PLAN-001](PLAN-001-cpp-custom-top.md) | C++ Custom TOP sender/receiver inside TouchDesigner | L (4–6 wk) | Proposed (v2, verified — this branch) | soft: PLAN-002 |
| 5 | [PLAN-005](PLAN-005-cpp-top-optimization.md) | C++ Custom TOP optimization backlog (CUDA Graphs port, 2 MiB IPC alignment, fused copy kernel) | S–M | Proposed (research complete) | PLAN-001 |

Rationale for the order:

- **PLAN-004 first** — pure docs/benchmark/env-default work, zero risk; its ADR-0008
  sharpens the "where native pays off" argument the other three plans cite.
- **PLAN-002 second** — smallest native surface, clones the proven `spout/` build
  pattern, improves every existing Python consumer, and builds native-wheel CI muscle
  before the big plugin work.
- **PLAN-003 third or in parallel** — independent; expected win is modest, so it carries
  an explicit kill criterion and "rejected, with numbers" is a valid outcome.
- **PLAN-001 last** — biggest win, biggest surface; benefits from PLAN-002's CI
  experience, and its protocol-core golden tests are designed so later native components
  can share the layout constants.

## Related ADRs

- [ADR-0006](../adr/0006-stay-pure-python-no-rust.md) — the pure-Python prior all four
  plans engage with (its escape-hatch clause is what PLAN-002/003 exercise).
- [ADR-0007](../adr/0007-spout-as-launcher-not-transport.md) — the in-process-native
  blast-radius prior PLAN-001 must answer (see ADR-0009).
- [ADR-0008](../adr/0008-skip-native-d2h.md) — decision record produced by PLAN-004.
- [ADR-0009](../adr/0009-cpp-custom-top-in-process.md) — decision record for PLAN-001's
  in-process posture (Proposed until the Phase 0 spike validates).
