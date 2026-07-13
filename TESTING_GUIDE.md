# Testing Guide

Canonical record of the cuda-link test suite: how to run it, its current quality
baseline, and the phased hardening campaign that produced it. Future contributors
(and AI assistants) should read this before changing the suite, and keep it current
when they do.

- Suite layout: `tests/{core,cuda,integration,support,td}` next to `src/` and `td_exporter/`.
- Config: single source of truth in `pyproject.toml` (`[tool.pytest.ini_options]`,
  `[tool.coverage.*]`). No `pytest.ini`, no `sys.path` hacks in `conftest.py`.
- GPU (`requires_cuda`) and native (`requires_native`) tests are **deliberately
  local-only** — this is a public repo and self-hosted GPU runners are excluded by
  policy. CI runs the deselected suite on ubuntu (Python 3.10/3.11/3.12).

## Running the suite

```bash
# Fast loop (what CI runs, minus coverage):
python -m pytest tests/ -m "not requires_cuda and not requires_native" -q

# With coverage (CI invocation — always whole-package, never a dotted submodule):
python -m pytest tests/ -m "not requires_cuda and not requires_native" --cov=cuda_link -q

# Full suite including GPU/native (local machine with CUDA + built extension):
python -m pytest tests/ -q
```

Test order is randomized every run (`pytest-randomly`, fresh seed). Reproduce an
ordering failure with `pytest --randomly-seed=<N>` (the seed is printed in the header).

## Baseline metrics (2026-07-12, post-campaign)

| Metric | Value |
|--------|-------|
| Tests (no-GPU selection) | 1179 passed + 3 skipped (Windows; 1112 pre-Phase-D) · ubuntu CI runs the same selection (Windows-only tests skip there) |
| Full-suite wall clock | ~19.4 s |
| Branch coverage, clean run | 99.10 % (`--cov=cuda_link`) · 79.46 % combined incl. `td_exporter` |
| Coverage gate (`fail_under`) | 76 (honest floor: ⌊79.46⌋ − 3; one-way ratchet — bump only upward) |
| Order independence / flakiness | 20/20 consecutive randomized-order runs green |
| Slowest test | 3.63 s (subprocess `-O` import test; by design — see Phase B) |
| Mutation score | see Phase D table below |

## Campaign record (2026-07-12)

The suite was audited and hardened per a structured campaign (structure → order
independence → speed → smells → mutation testing). Phases 0–3 of the standard
playbook were already satisfied before the campaign — recorded here for completeness:

- **Structure/config (Phases 0–1): already compliant.** `--import-mode=importlib`,
  `pythonpath = [".", "src", "td_exporter", "tests"]`, `--strict-markers
  --strict-config`, all markers registered, clean `conftest.py`. No unit/slow tiering:
  the whole suite runs in ~19 s, below the 30 s fast-loop threshold that would justify it.
- **Coverage gate (Phase 3): ratcheted this cycle.** `fail_under` 55 → 76 after the
  coverage push landed (commit `f7bfb7f`, cuda_link 75 % → 99.10 % branch coverage).
  The gate is measured against the weaker combined invocation (bare `--cov`, which
  includes TD-only `td_exporter` files at 79.46 %) so both invocations pass.
  **Ratchet rule:** when coverage genuinely improves, bump `fail_under`; never lower it.
- **Snapshot testing (Phase 4): skipped deliberately.** The codebase has no complex
  pure outputs (reports, rendered trees, serialized artifacts) that are hard to assert
  by hand — struct layouts and telemetry dicts are asserted directly.

### Phase A — Order independence + flakiness

20/20 consecutive full-suite runs with randomized order (fresh seed each run), all
green: 1112 passed per run, ~19 s each (one run hit 57 s under machine contention —
still green). No ordering dependencies, no flakes, nothing to fix.

### Phase B — Speed profile

`--durations=25` over the no-GPU suite. Slowest: 3.63 s and 3.51 s — the two
`test_module_imports_cleanly_under_dash_o*` tests in
`tests/cuda/test_cuda_runtime_types.py`, which spawn a fresh `python -O` subprocess
by design (they verify import-time ABI guards survive `-O`). Third place is 0.61 s;
everything else ≤ 0.21 s. Nothing crosses the 5 s threshold that would warrant
`@pytest.mark.slow`, so no tiering was added (deferral principle).

### Phase C — Structural smells

Audit of ~40 bare float `==` assertions, mock `side_effect` lists, and filesystem use:

- **Float `==`: 3 converted, rest kept exact deliberately.** The three assertions on
  division-derived values in `tests/support/test_profile_cov.py` (`FrameProfile.avg`,
  `ReportWindow.fps`) now use `pytest.approx`. Every other float comparison is
  **exact by construction** and stays `==`: zero-guard branches that `return 0.0`
  literally (importer wait paths, fps dt≤0 guards), passthrough assignments of
  exactly-representable constants (`1.5`, `4.2`, `5000.0`), and struct pack/unpack
  round-trips of doubles (exact for representable values). Converting those to
  `approx` would weaken the assertions for no benefit.
- **`side_effect` lists: audited safe.** All three occurrences
  (`test_importer.py`, `test_wait_for_slot_busywait.py`) are function-local, freshly
  built per test, and padded against extra calls (`[0.0] + [10.0] * 20`) — no
  cross-test iterator-exhaustion risk.
- **Filesystem: no production-dir pollution.** `test_installer_staleness.py` writes
  only under pytest's `tmp_path`. Known wart: that file inserts `scripts/` into
  `sys.path` at module level to import `install_td_library` (scripts/ is not a
  package and not on `pythonpath`); acceptable as-is.

### Phase D — Mutation testing (cosmic-ray)

Mutation testing measures whether tests *catch bugs*, not just execute lines. Its
payoff is inversely proportional to mock density, so only pure, deterministic,
zero-/near-zero-mock modules are targeted:

| Target | Why chosen | Dedicated tests (mocks) |
|--------|-----------|-------------------------|
| `src/cuda_link/shm_protocol.py` | pure struct/layout logic, 0 ctypes | `test_shm_protocol.py` — 115 tests (0) |
| `src/cuda_link/activation_barrier.py` | struct + `SharedMemory` (deterministic local IPC) | `test_activation_barrier{,_checker,_holder,_cov}.py` — 67 tests (~0) |
| `src/cuda_link/_profile.py` | pure arithmetic/formatting | `test_profile_cov.py`, `test_report_window.py` — 27 tests (0) |

**Excluded (boundary code):** `cuda_runtime_types.py`, `_native_loader.py`,
`cuda_graphs.py` — ctypes struct definitions and DLL/GPU calls. Their tests mock at
the ctypes/DLL boundary (classicist-compliant), so a mutation score there would
measure the mocks, not the logic. They form the de-mocking backlog: a module
graduates to a mutation target only when its mock density approaches zero.

Workflow (Windows; note the **absolute interpreter path** in each `cr-*.toml` —
bare `python` resolves through the App Paths registry to a different environment):

```bash
python -m cosmic_ray.cli init cr-<target>.toml cr-<target>.sqlite
python -m cosmic_ray.cli baseline cr-<target>.toml     # must pass on unmutated source
python -m cosmic_ray.cli exec cr-<target>.toml cr-<target>.sqlite   # sequential, long
cr-report cr-<target>.sqlite
```

The `cr-*.sqlite` session databases are regenerated artifacts and gitignored.
`cr-activation_barrier.toml` must never run concurrently with another pytest run or
a live TD session — its tests use the fixed-name activation-barrier SHM segment.

**Results (2026-07-12):**

| Target | Mutants | Killed | Genuine survivors | Suppressed (equivalent) | Score |
|--------|---------|--------|-------------------|-------------------------|-------|
| `_profile.py` | 133 | 133 | 0 | 0 | **100 %** |
| `shm_protocol.py` | 506 (1 incompetent) | 453 | 1 | 51 | **99.78 %** |
| `activation_barrier.py` | 349 (5 incompetent) | 333 | 2 | 9 | **99.4 %** |

Score = killed / (killed + genuine survivors), after excluding incompetent mutants
and suppressing proven-equivalent ones. Target: ≥ 85 % per module.

The raw exec run left 196 survivors (17 + 106 + 73). Triage added **66 kill-tests**
(11 in `test_profile_cov.py`, 17 in `test_shm_protocol.py`, 38 in
`test_activation_barrier_cov.py`), each named `test_..._kills_<operator>_mutant` and
**transiently verified**: the mutant diff was applied to the source, the named test
confirmed to fail, and the source restored — per mutant, no exceptions.

**Equivalence suppression:** 30 source lines carry a trailing `# pragma: no mutate`
(24 in `shm_protocol.py`, 6 in `activation_barrier.py`), each with a one-line proof
comment above it (e.g. lazy PEP 563 annotations where `str | None` is never
evaluated, bit-disjoint offset arithmetic where `+`/`|`/`^` coincide, single-element
tuple `[0]`/`[-1]`, Protocol stub bodies, Enum `is`/`==`). `cr-filter-pragma` skips
*every* mutant on a pragma'd line (173 + 61 SKIPPED respectively, verified against
the session DBs), so a killable mutant sharing a line with an equivalent one gets
collaterally suppressed — such cases were kill-test-verified *before* the pragma was
added and are flagged in the source comment (e.g. `activation_barrier.py`
reserved-bytes line).

**Survivor ledger (the 3 non-suppressed survivors):**

| Module | Line (fn) | Mutant | Why accepted |
|--------|-----------|--------|--------------|
| `shm_protocol.py` | `acquire_slot`, `write_idx == 0 or ...` | `==` → `<=` on `write_idx == 0` | Equivalent: `write_idx` is an unsigned counter, so `<= 0` ≡ `== 0`. Left **un**-pragma'd deliberately — a line pragma would also skip the 4 kill-tested mutants sharing the line. Counted against the score instead. |
| `activation_barrier.py` | `_log_stale`, `/ 1e9` | NumberReplacer ×2 (`1e9` ± 1) | A ~1e-9 relative error in the ns→s conversion is invisible in the `%.1f`-formatted staleness log at any realistic timestamp magnitude — not reasonably testable. |

**Re-run trigger:** before releases, or after major refactors of a target module.

### Test-quality review (post-Phase-D)

A scoped review of the suite (test correctness / tautology only, 27 files) found 4
blocking and 13 important findings — mock-heavy tests that asserted a call happened
without pinning its arguments, byref out-params never written through, caplog
assertions that matched no logger, and spy-less cleanup paths. All 17 were fixed:
memcpy/stream arguments are now pinned exactly, ctypes byref out-params are filled
with sentinels and asserted, caplog assertions bind logger name + level, and
`_do_cleanup` dispatch is spy-verified (`autospec` wrap). Known accepted limitation:
the STEP6 `if dev_ptr:` full-guard-deletion mutant in `exporter.py` is partially
masked because frees run on daemon threads; the strengthened assertion still catches
inversion and wrong-slot regressions.

## Deferral table — tooling we deliberately did NOT add

| Tooling | Trigger that would justify it | Current state |
|---------|-------------------------------|---------------|
| `pytest-xdist -n auto` | fast loop > ~5 min | 19 s — far below |
| `pytest-split` sharding | per-runner > ~10 min after xdist | n/a |
| Test tiering (unit/slow dirs) | fast loop > 30 s | 19 s |
| `@pytest.mark.slow` markers | individual test > 5 s | slowest is 3.63 s |
| Snapshot testing (syrupy) | complex pure outputs hard to assert by hand | none exist |
| New mutation targets | module's mock density approaches zero | backlog above |
| Self-hosted GPU runner | **never** — public repo, excluded by policy | GPU tests local-only |
