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
To pin the order (bisecting, mutation runs) use a fixed `--randomly-seed=<N>`, **not**
`-p no:randomly`: `addopts` passes `--randomly-dont-reset-seed`, which is an unknown argument
once the plugin is off, so pytest exits with a usage error.

Replaying past fixes (`replay_fixes.py` from the harden-test-suite skill) needs one extra step on
this repo. The scikit-build-core editable install (`_editable_skbc_cuda_link`) maps `cuda_link` to
the main checkout ahead of `PYTHONPATH`, so the throwaway worktree's `src/` would be ignored and the
control run goes red. Put a directory holding this `sitecustomize.py` on `PYTHONPATH` (the script
appends it after the worktree paths):

```python
import sys

sys.meta_path[:] = [f for f in sys.meta_path if "_editable_skbc" not in type(f).__module__]
```

and pass the venv interpreter as an **absolute** path in `--test-command`, since the command runs
inside the worktree.

## Current baseline (2026-11-18, Phase J audit)

**Supersedes the 2026-10-22 table below.** Measured on `feat/orbbec-v0.6` at `35d207d` (86 commits
after Phase I: SHM protocol v0.6 slot meta, producer/consumer doorbell defaults, Receiver
Framewaitms), before any Phase J change. **Linux** (cloud container, Python 3.12, `uv sync --locked
--extra dev`, CPU torch), not the Windows box of earlier audits: the Win32-only paths skip here, so
the coverage figures are not directly comparable with the table below (see the platform note).

| Metric | Value |
|--------|-------|
| Tests (no-GPU selection) | 1717 passed + 18 skipped, 68 deselected (1492 `def test_`) |
| Skips | 18, all platform with a stated reason: 10 Win32 named events (`_doorbell`), 6 Win32 console handler (`_console`), 2 compiled native module. Ratio 1.0 %, under `max_ratio = 0.02` |
| No-GPU wall clock | 18.4-18.9 s (10 runs); 34 s with `--cov --cov-branch` |
| Slowest tests | 4.5 s and 1.8 s (the two subprocess `-O` import tests; by design); next 0.6 s |
| Branch coverage, clean run | 91.72 % combined (`--cov`, 5671 stmts, 425 miss), gate 91 passes. `cuda_link` 97 % (99 % without the Win32-only `_doorbell`/`_console`), `td_exporter` 84 % |
| Patch coverage of the v0.6 work | 94.4 % of the 468 executable lines added since `cb7e951` (the Phase I record) |
| Order independence / flakiness | 10/10 randomized runs green, identical counts; `rank_failures.py`: no failing test |
| Past-fix recall (`replay_fixes.py --last 12`) | **5/5 = 100 %** caught; 7 skipped (no test change, cpp-only, or merge). Needs the editable-finder bypass below |
| Test smells (AST scanner) | **33 vs baseline 19: ratchet red.** 14 new findings, all in v0.6 tests: 7 `parametrize-no-ids`, 4 `bitwise-assert`, 2 `overprotective`, 1 `no-assert`. Fixed in J1 |
| Functions over CRAP 30 | `cuda_link`: `Exporter.export` 47 (CC 47, 100 % covered), `_do_cleanup` 32, `_console._ctrl_handler` 30 (Win32-only, 0 % on Linux). `td_exporter`: `TDSender.export_frame` 68.1, `_refresh_on_version_change` 52.2, `import_frame` 48.6, launcher `onExit` 42 ×2, `initialize_receiver` 34.5 |
| C901 (ruff, max 43) | passes; `export` still at 43, the ceiling |
| Logic in test bodies | 412 `if/for/while/try` lines (368 at Phase I); top files unchanged |
| Sleeps | 1 real sleep in the no-GPU suite (`test_exporter_cov_cleanup.py`, past a 0.5 s join deadline) + 3 polls in the GPU roundtrip; the other `sleep` hits are `mock_time.sleep` asserts |
| Code pollution (`TESTING`, `is_test`) | none in `src/` or `td_exporter/` |

Platform note: on Windows the 16 Win32 skips run and `_doorbell.py`/`_console.py` are covered, so
the combined figure there is higher. The `fail_under = 91` margin on Linux is 0.7 points. CI's own
leg measures `--cov=cuda_link` only (97 % here), so it is not at risk.

Audit findings that need a change to a gate or config file, so they are **not** in J1 (they go in
their own PR; see the review rule on gate files). All six are handled in J3 below:

- **Coverage measures fake packages.** `source = ["cuda_link", ...]` matches by package name, so the
  throwaway `cuda_link/` trees that the bootstrap tests build under `tmp_path` show up as 0-statement
  rows, and `coverage lcov`/`json` fail with "No source for code" unless run with `-i`. The total is
  unaffected. Fix: `source_pkgs`/`source` with paths, or `omit = ["*/pytest-of-*/*"]`.
- **`-p no:randomly` cannot be used.** `addopts` passes `--randomly-dont-reset-seed`, which is an
  unknown argument once the plugin is off. Pin order with `--randomly-seed=<N>` instead (the bisect
  recipe in the harden-test-suite skill uses `-p no:randomly`; on this repo use the seed form).
- **`replay_fixes.py` needs the scikit-build-core finder removed.** The editable install
  (`_editable_skbc_cuda_link`) maps `cuda_link` to the main checkout ahead of `PYTHONPATH`, so a
  worktree's `src/` is ignored. Without a bypass the control run goes red (the fix commit's tests run
  against HEAD's code). Bypass used here: a `sitecustomize.py` on `PYTHONPATH` that removes the
  `_editable_skbc` finder from `sys.meta_path`; checked by importing a fake package from a scratch
  `src/` with and without it.
- **Ruff pins disagree.** `uv.lock` resolves ruff 0.17.4, `.pre-commit-config.yaml` pins
  `ruff-pre-commit` v0.14.11, and the deferral table below said 0.17.0.
- **Scanner upgrade available.** The skill's `scan_test_smells.py` adds a report-only `real-clock`
  smell (5 hits here, none flaky in 10 runs: bounds such as `t_before <= host_timestamp <=
  perf_counter()`). Report-only, so vendoring it needs no baseline change.
- **CI passes no `--cov-fail-under=0`**, so the local `fail_under` also gates CI's `--cov=cuda_link`
  leg. Harmless today (97 % vs 91).

## Previous baseline (2026-10-22, Phase I audit)

(Historical — see the current baseline above.) Measured on `development` at `1da5339` (PRs #79-#87 Measured on `development` at `1da5339` (PRs #79-#87
merged), Python 3.12.11, Windows, before any Phase I change.

| Metric | Value |
|--------|-------|
| Tests (no-GPU selection) | 1415 passed + 1 skipped, 68 deselected |
| No-GPU wall clock | ~23 s (10 runs, 22.6-26.2 s); 28 s with `--cov --cov-branch` |
| Slowest tests | 3.9 s and 3.8 s (the two subprocess `-O` import tests; by design); next is 0.6 s |
| Branch coverage, clean run | 90.76 % combined (`--cov`, 5364 stmts, 448 miss) |
| Coverage gate (`fail_under`) | 85 at the audit (raised in I1) |
| Coverage by module | `cuda_link` 99 %; `TDReceiver.py` 72 % (173 miss), `CUDAIPCExtension.py` 88 %, `CUDALinkBootstrap.py` 89 %, `TDSender.py` 91 %, `TDHost.py` 92 % |
| Order independence / flakiness | 10/10 randomized runs green, identical counts; `rank_failures.py` found no failing test |
| Test smells (AST scanner) | 19, equal to `.smell-baseline.json` (ratchet OK) |
| Functions over CRAP 30 | `cuda_link`: 2 (`Exporter.export` 46, `_do_cleanup` 32, both 100 % covered, so complexity alone). `td_exporter`: 6, worst `TDReceiverEngine.initialize_receiver` **244** (CC 33, 42 % covered), `parexecute_callbacks.handle_mode_change` and `handle_numslots_change` 72 each (0 %), `TDSenderEngine.export_frame` 68, `_refresh_on_version_change` 52, `import_frame` 44 |
| C901 over 15 | 10 (`Exporter.export` 43, `_do_cleanup` 26, `TDSender.export_frame` 34, `initialize_receiver` 28, ...) |
| Mutation score | at the audit: last measured 2026-07-12 (`_profile` 100 %, `shm_protocol` 99.78 %, `activation_barrier` 99.4 %), stale after #82/#83. Re-measured in I2: `shm_protocol` and `activation_barrier` both 100 % with no genuine survivors |
| Tests / test LOC | 1295 `def test_` · 26.9k test LOC vs 8.5k `src/cuda_link` LOC |
| Logic in test bodies | 368 `if/for/while/try` lines; top: `test_cpu_roundtrip.py` 36, `test_td_bootstrap.py` 25 |
| Sleeps / private patch targets | 4 `time.sleep` · 24 patches of `_private` names |
| Code pollution (`TESTING`, `is_test`) | none in `src/` or `td_exporter/` |

**Full local suite (GPU + native) was red at the audit: 6 failed, 1477 passed, 1 skipped, ~115 s.**
CI never sees these because it deselects `requires_cuda`.

- `test_ctypes_adapter_satisfies_protocol` and `..._importer_protocol` (2): stale tests. On
  Python 3.12+ `isinstance(adapter, Protocol)` uses `inspect.getattr_static`, which skips the
  adapter's `__getattr__` delegation, so it is `False` until every member has been cached. The
  adapter itself is fine. Fixed in I1 with a `hasattr` member check.
- `test_roundtrip.py` (4): environmental. The venv holds `torch 2.14.1+cpu` (no CUDA), so the
  consumer's torch backend fails with "Cannot initialize CUDA without ATen_cuda library" after the
  IPC slots and events have opened. `uv.lock` resolves torch from PyPI, whose Windows wheels are
  CPU-only, so a `uv sync` replaces a CUDA build. Needs a CUDA torch in the local venv; not a code
  regression. Product note: the Importer reports this as "Producer not yet available" and retries,
  which hides a permanent local failure.

## Baseline metrics (2026-10-19, post Phase H2)

(Historical — see the current baseline above.)

| Metric | Value |
|--------|-------|
| Tests (no-GPU selection) | 1327 passed + 2 skipped (Windows; 63 deselected) · ubuntu CI runs same selection |
| No-GPU selection wall clock | ~18 s (H1, without coverage) · ~22 s with `--cov=cuda_link --cov-branch` (H2) |
| Full suite incl. GPU/native (local) | 1386 passed + 3 skipped, ~29 s (was ~76 s before H1; H1 figure, full suite not re-run in H2) |
| Branch coverage, clean run | 99.35 % (`--cov=cuda_link`, no-GPU selection) · 88.21 % combined incl. `td_exporter` (full suite) |
| Coverage gate (`fail_under`) | 85 (unchanged; one-way ratchet. See F4.) |
| Order independence / flakiness | 10/10 no-GPU + 3/3 full-suite randomized runs green (audit), 3/3 full + 1 no-GPU green after H1, 3/3 no-GPU seeds green after H2 |
| Slowest test | 2.9 s (the two subprocess `-O` import tests; by design — see Phase B). GPU roundtrips ~2.1 s each, were 22.5 s |
| Test smells (AST scanner) | 19 total (was 77 at the H0 audit, 73 on the H2 base); ratcheted by committed `.smell-baseline.json` (see H2) |
| Skipped/xfail ratio | 1.1 % (13/1224) — gated under 2.0 % threshold |
| Complexity ceiling (`C901 max-complexity`) | 43 (unchanged). See F3 and Phase H audit for CRAP. |

## Baseline metrics (2026-10-12, post Phase G)

(Historical — see the current baseline above.)

| Metric | Value |
|--------|-------|
| Tests (no-GPU selection) | 1322 passed + 3 skipped (Windows; 64 deselected: 61 requires_cuda, 3 requires_native) · ubuntu CI runs same selection |
| Full-suite wall clock | ~22.3 s |
| Branch coverage, clean run | 99.45 % (`--cov=cuda_link`) · 87.10 % combined incl. `td_exporter` |
| Coverage gate (`fail_under`) | 85 (honest floor: ⌊87.10⌋ − 1.5, rounded down; one-way ratchet — bump only upward. See F4.) |
| Warnings | 0 (strict `filterwarnings = ["error"]` enforced with targeted compatibility deprecation ignores) |
| Order independence / flakiness | 20/20 consecutive randomized-order runs green — skip count stable 3 in every run |
| Slowest test | 3.63 s (subprocess `-O` import test; by design — see Phase B) |
| Test smells (AST scanner) | 77 total (0 sleep smells, down from 82 smells / 1 sleep; see Phase G) |
| Skipped/xfail ratio | 1.1 % (14/1247 disabled, 59 platform-gated excluded) — gated under 2.0 % threshold |
| Complexity ceiling (`C901 max-complexity`) | 43, ratcheted downward-only from unenforced. See F3. |
| Type check errors | 0 errors (15 baseline-suppressed legacy diagnostics in `pyrefly-baseline.json`) |

## Baseline metrics (2026-08-20, post Phase E/F)

(Historical — see the current baseline above. Retained for the Phase E–F record below.)

| Metric | Value |
|--------|-------|
| Tests (no-GPU selection) | 1196 passed + 3 skipped (Windows) · ubuntu CI runs the same selection (Windows-only tests skip there) |
| Full-suite wall clock | ~17.2 s |
| Branch coverage, clean run | 99.14 % (`--cov=cuda_link`) · 87.10 % combined incl. `td_exporter`, post the F4 untestable-demo-script omit |
| Coverage gate (`fail_under`) | 85 (honest floor: ⌊87.10⌋ − 1.5, rounded down; one-way ratchet — bump only upward. See F4.) |
| Order independence / flakiness | 20/20 consecutive randomized-order runs green, **re-verified after the Phase E fix** — skip count a stable 3 in every run (was 3 or 4, order-dependent, before Phase E) |
| Slowest test | 3.63 s (subprocess `-O` import test; by design — see Phase B) |
| Mutation score | see Phase D table below (unchanged by Phase E/F — no mutation targets touched) |
| Complexity ceiling (`C901 max-complexity`) | 43, ratcheted downward-only from unenforced. See F3. |

## Baseline metrics (2026-07-12, post-campaign)

(Historical — see the current baseline above. Retained for the Phase A–D record below.)

| Metric | Value |
|--------|-------|
| Tests (no-GPU selection) | 1179 passed + 3 skipped (Windows; 1112 pre-Phase-D) · ubuntu CI runs the same selection (Windows-only tests skip there) |
| Full-suite wall clock | ~19.4 s |
| Branch coverage, clean run | 99.10 % (`--cov=cuda_link`) · 79.46 % combined incl. `td_exporter` |
| Coverage gate (`fail_under`) | 76 (honest floor: ⌊79.46⌋ − 3; one-way ratchet — bump only upward) |
| Order independence / flakiness | 20/20 consecutive randomized-order runs green — **later found false on re-audit; see Phase E** |
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

## Campaign record — Phase E/F (2026-08-20)

The prior campaign's **"20/20 consecutive randomized-order runs green"** claim (Phase A above)
was false when re-audited: 4 of 13 randomized-order runs failed, and three checklist steps from
the audit skill (Step 0 triage, Phase 0.5 prune, Phase 3.5 complexity/CRAP gate) had never been
executed. This campaign kills the flake at its root cause and closes those three gaps.

### Phase E — Kill the activation-barrier flake

**Root cause:** every barrier test shared the single fixed-name SHM segment
`cudalink_activation_barrier`. On Windows, `SharedMemory.unlink()` is a no-op and segment
lifetime is handle-bound, not name-bound — a not-yet-GC'd handle elsewhere in the process kept
the name alive, so `SharedMemory(name=..., create=True)` intermittently raised `FileExistsError`
in `test_open_or_create_initializes_full_header_to_zero`. This is the **external shared-state
corruption** anti-pattern: a shared out-of-process resource, not a timing race, so the fix is
hermeticity, not retries.

A second test papered over the same defect with a runtime `pytest.skip()`, producing an
**order-dependent skip** — 4 skipped instead of a stable 3 depending on what ran before it.

Fix:

- Added `isolated_barrier_shm` fixture (`tests/conftest.py`) — redirects `activation_barrier`'s
  module-global `SHM_NAME` to a unique `cudalink_barrier_test_<uuid8>` segment per test via
  `monkeypatch.setattr`, with teardown that closes/unlinks it if still present.
- Both `tests/core/test_activation_barrier.py` and `tests/core/test_activation_barrier_cov.py`
  adopt it through a thin autouse wrapper (`_isolate_barrier`), replacing the old module-local
  `_cleanup()` / `cleanup_barrier` fixture. `_isolate_barrier` is scoped to these two modules
  only — no new autouse fixture leaks into the rest of the suite.
- Deleted the order-dependent `pytest.skip()` guard in `test_activation_barrier.py` — with a
  guaranteed-absent unique segment, the skip's premise (segment sometimes already exists) can no
  longer occur, so the bare `with pytest.raises(FileNotFoundError): open_or_create(create=False)`
  now always exercises the real path.

**Verified:** 20/20 consecutive randomized-order full-suite runs green, skip count a stable
**3** in every run (proving the order-dependent skip is gone, not just hidden by luck).
`cr-activation_barrier.toml`'s concurrency warning was updated to match: the two SHM-touching
test files are now hermetic against concurrent pytest runs (each test gets its own segment
name); the only surviving caution is against running concurrently with a **live TD session**,
which opens the real fixed-name segment through production code paths that are never
monkeypatched — a genuinely different entry point onto the same name.

### Phase F — Close three unrecorded audit gaps

#### F1 — Test-value triage (Step 0)

Classifying each module by risk quadrant, anchored to measured numbers, so future coverage and
mutation work aims at the right targets instead of picking by file size or mock density:

| Module | Quadrant | Evidence |
|--------|----------|----------|
| `_exporter_port.py`, `_importer_port.py`, `_env.py` | **Unit-test hard** | Zero direct *and* transitive mocks; 61 / 25 / 20 tests. Prime untapped mutation targets. |
| `activation_barrier.py`, `shm_protocol.py`, `_profile.py` | **Unit-test hard** | Already mutation-tested (99.4 % / 99.78 % / 100 %) — see Phase D table above. |
| `td_exporter/TDReceiver.py` | **Refactor first** | 51 % covered *and* CC 35 / 24 / 23 in `initialize_receiver` / `_refresh_on_version_change` / `import_frame` → est. CRAP(`initialize_receiver`) ≈ 179. The genuine risk hotspot; deferred (see table below). |
| `src/cuda_link/exporter.py::export` | **Refactor first** | CC 43 (ruff) — over the CRAP-30 ceiling on complexity alone, even at 100 % coverage (CRAP = 43 at cov=1). |
| `td_exporter/CUDAIPCExtension.py` | **Integration-test briefly** | Facade dispatching to two engines; many collaborators, low per-method complexity. |
| Demo / launcher / benchmark scripts | **Don't test** | Not runnable outside TouchDesigner → drives F4's coverage-omit list. |

#### F2 — Phase 0.5 prune

1. **48 redundant `sys.path.insert` calls removed, across 21 files.** All targeted the repo
   root, `src/`, or `td_exporter/` — already provided by `pythonpath = [".", "src",
   "td_exporter", "tests"]` in `pyproject.toml`. Kept the 2 legitimate ones
   (`tests/support/test_wrapper_sync.py`, `tests/support/test_installer_staleness.py`), which
   insert `scripts/` — not on `pythonpath`. This makes the "no `sys.path` hacks" claim in this
   guide's intro actually true, not just true of `conftest.py`.
2. **37 unpinned interaction assertions reclassified**, across 14 files (`assert_called_once()`
   / `assert_called()` with no argument pinning; the other 3 of the original 40 grep hits were
   comment-only, not code). Each was judged individually against the mock/stub principle — a
   mock verifies an outgoing effect; asserting an interaction with a stub is the anti-pattern:
   - Most were pinned to `assert_called_once_with(...)` with the real argument values, traced
     through production code (e.g. `close_spy.assert_called_once_with()`,
     `fake_drv.cuInit.assert_called_once_with(0)`).
   - Opaque `byref(...)` ctypes out-params with no derivable literal were pinned with
     `unittest.mock.ANY` instead of left unpinned (e.g.
     `cudaGraphExecMemcpyNodeSetParams.assert_called_once_with(graph_exec, node, ANY)`).
   - **7 kept as bare `assert_called_once()`**, each followed by a manual
     `call_args[0][i].value ==` check — `c_int`/`c_void_p` ctypes simple types have no
     value-based `__eq__`, so `assert_called_once_with(c_int(3))` can never pass. Matches the
     files' own pre-existing convention (`tests/core/test_exporter_cov_graphs.py`,
     `tests/cuda/test_wrapper_cov_methods.py` — 2 of the 6 there had no follow-up check
     originally and were strengthened with one).
   - Zero interactions were deleted outright — every occurrence was a genuine mock (an outgoing
     effect worth verifying), not a misused stub.
3. **`fake_exporter_open` hoisted** from 4 duplicated `tests/td/*.py` copies into
   `tests/fakes/__init__.py`, dispatching to each requesting module's own `_FakeExporter` via
   `request.module._FakeExporter` (the 4 local doubles differ slightly — grow-safety's tracks
   an extra `export_calls` list — so the fixture stays generic rather than picking one copy as
   canonical). Required 4 new `per-file-ignores` (`F811`) in `pyproject.toml`, scoped to exactly
   the 4 consuming files — pyflakes doesn't recognize the imported-fixture-as-parameter-name
   pytest idiom and flags a false redefinition.
4. **The imperative `pytest.xfail()`** at `test_graph_coexistence_capture.py` converted to
   `@pytest.mark.xfail(strict=True, reason=...)` + an explicit `raise AssertionError(...)` in
   the error branch (which the decorator catches as XFAIL); the `pytest.skip(...)` branch for
   "driver serialized the captures" is unchanged. Now strict-checked: if the underlying C2
   regression is ever fixed, this test starts failing loudly (XPASS) instead of silently
   staying green.
5. **`pytest-mock` — no-op.** Zero `mocker.` usages found, and it was never actually declared
   in `dev` deps in the first place (only `pytest`, `pytest-randomly`, `pytest-cov` are) —
   nothing to remove.

#### F3 — Phase 3.5: complexity + CRAP baseline

Added `C901` to `[tool.ruff.lint] select` with `[tool.ruff.lint.mccabe] max-complexity = 43` — a
**downward-only ratchet** set at today's true worst offender (measured directly, not estimated),
so it produces zero new failures today while making any new function worse than the current
worst a hard error.

McCabe complexity (ruff), production hotspots ≥ 20:

| Location | CC (ruff) |
|----------|-----------|
| `src/cuda_link/exporter.py:534 export` | **43** |
| `td_exporter/TDReceiver.py:638 initialize_receiver` | 35 |
| `td_exporter/TDSender.py:510 export_frame` | 30 |
| `src/cuda_link/exporter.py:780 _do_cleanup` | 26 |
| `td_exporter/TDReceiver.py:1095 _refresh_on_version_change` | 24 |
| `td_exporter/TDReceiver.py:386 import_frame` | 23 |

**46 total** violations at `max-complexity=10` (measured; supersedes an earlier estimate of 37):
11 in shipping product code, 6 duplicated in the generated `Exporter.py`/`Importer.py` mirrors,
9 in `scripts/profiling/*`, 2 in `scripts/install_td_library.py`, 2 in
`td_exporter/example_*_python.py`, 7 in `tests/td/*` helpers, 1 in
`.claude/hooks/git-commit-enforcer.py`, 8 in `examples/*.py` (files 02–08) — the last two
categories weren't in the original estimate, which never scanned `.claude/hooks/` or
`examples/`. None of the 46 are touched by this campaign; that refactor stays deferred (below).

Tooling for this pass (`cosmic-ray`, `radon`, `crap4py`) installs via `pip install -e ".[quality]"`
— a separate extra from `dev`, kept out of it deliberately so a resolver failure in these
manual, local-only analysis tools can never break CI's Python 3.9 leg (`ci.yml` installs the
locked `dev` extra only). crap4py's only PyPI release is
0.1.1 and it requires Python ≥3.10, so it's marker-gated (`python_version >= '3.10'`) and simply
skipped on 3.9 rather than failing the install — the CRAP table below can only be regenerated on
3.10+.

CRAP pass (`crap4py src/cuda_link --lcov lcov.info --max-crap 30`, `src/cuda_link` only — 100 %
line coverage everywhere in that package, so CRAP reduces to radon's own complexity number for
every entry): **2 functions exceed the 30 ceiling**, both already known hotspots —

| Function | comp (radon) | CRAP |
|----------|---------------|------|
| `Exporter.export` | 46 | **46.0** |
| `Exporter._do_cleanup` | 32 | **32.0** |

Next tier (comp 12–17, all under the ceiling, all `importer.py`/`exporter.py`):
`CupyBuffers.build` / `NumpyBuffers.build` / `TorchBuffers.build` / `Importer._open_ipc_slots`
(17), `Exporter._initialize` / `Importer._wait_for_slot` / `Importer.get_stats` (14),
`Exporter._build_export_graphs` / `Importer._resolve_format` (12).

Note: radon's `comp` (46 for `export`, 32 for `_do_cleanup`) runs higher than ruff's McCabe
count (43, 26 respectively) for the same two functions — a different metric, same discrepancy
pattern already seen between ruff and the code-search index's `complexity_score`. **Ruff's
number is the one the `max-complexity` ratchet is set against**; CRAP is a stop condition on
coverage-chasing, not a second complexity authority.

This is a **stop condition, not a discovery instrument** — `export` and `_do_cleanup` were
already the two hotspots flagged in F1's triage and in the pre-existing deferral table below;
CRAP just confirms coverage can't buy their way under 30 without a complexity reduction. Not
fixed here — recorded as the trigger condition in the deferral table.

#### F4 — Honest coverage denominator + ratchet

Added a second, separately-commented block to `[tool.coverage.run] omit` in `pyproject.toml`
covering `td_exporter/example_receiver_python.py`, `example_sender_python.py`,
`callbacks_template.py`, `benchmark_timestamp.py` — kept distinct from the pre-existing 15-file
mirror block (omitted for a different reason: double-counting, not untestability). All four run
only inside a live TD process, are not importable standalone, and have zero tests referencing
them (F1's "Don't test" quadrant). Deliberately **not** added:
`example_receiver_launcher.py` / `example_sender_launcher.py` (38 % each, exercised by F5's
tests), `parexecute_callbacks.py` (32 %), `script_top_callbacks.py` (98 %) — tests genuinely
exercise these, so omitting them would hide real gaps instead of an artifact of the denominator.

Re-measured clean-run baseline post-omit: **87.10 %** combined (vs. the plan's ≈86.6 %
projection — measured, not assumed). Ratcheted `fail_under` **76 → 85**: `⌊87.10⌋ − 1.5`,
rounded down. This replaces the old `⌊baseline⌋ − 3` convention with a tighter ~1.5-point
margin — the wider 3-point margin existed partly to absorb exactly the 4 files just removed
from the denominator, so keeping it unchanged after they're gone would leave slack that
catches nothing. Verified both CI invocations still pass against the new gate: combined
(bare `--cov`) 87.10 %, `--cov=cuda_link` 99.14 %.

#### F5 — Receiver-launcher parity fix

`td_exporter/example_receiver_launcher.py` was at 0 % coverage and carried a known, previously
undiagnosed-as-fixed bug: its final Python-interpreter-resolution fallback unconditionally
returned the literal string `"python"`, so on a machine where nothing actually resolved on
`PATH`, `subprocess.Popen` would raise deep inside the subprocess with an opaque error instead
of failing clearly at launch time. `example_sender_launcher.py` had already been hardened
against exactly this; the receiver launcher had not.

Fix: ported the sender's pattern — `_find_python_exe() -> str | None` now checks
`shutil.which("python")` for the bare fallback and returns `None` if it doesn't resolve;
`onStart()` checks for `None` first and prints an actionable error (env var + PATH guidance)
without spawning anything and without touching the TD `project` global (which doesn't exist
outside a live TD process). Added `tests/td/test_example_receiver_launcher.py`, a direct mirror
of the existing sender test (4 tests total between the two files, all passing). The sender
test's docstring — which had documented the receiver's bug as unfixed — now points to the new
test instead.

### Phase G — CI/CD Pipeline Modernization & Test Hardening (2026-10-12)

Executed a systematic hardening campaign using the `/setup-ci-cd-pipeline` and `/harden-test-suite`
skills to bring the project into strict compliance with modern Python CI/CD standards, reproducible
locking, warning hygiene, and smell elimination.

#### G1 — Virtual environment & lockfile pinning

- Pinned exact versions for all 78 production, test, and quality dependencies in `uv.lock` via `uv lock`.
- Synchronized local environment via `uv sync --extra dev --extra quality` (`torch 2.14.1`, `numpy 2.5.4`, `ruff 0.17.0`, `pyrefly 1.3.2`).
- Updated `.pre-commit-config.yaml` to pin `ruff v0.17.0` matching `uv.lock`.

#### G2 — Git attributes & line endings

- Hardened `.gitattributes` to explicitly enforce `eol=lf` across all text sources (`*.py`, `*.md`,
  `*.yml`, `*.yaml`, `*.toml`, `*.json`, `*.jsonc`), `eol=crlf` for Windows command scripts (`*.cmd`,
  `*.ps1`, `*.bat`), and marked binary distributions and native assets (`*.whl`, `*.dll`, `*.pyd`,
  `*.toe`, `*.tox`, `*.sqlite`) as `-text`.
- Eliminates CRLF/LF churn and checkout diff hazards across cross-platform CI runners.

#### G3 — Pytest & warning strictness

- Enabled `xfail_strict = true` in `pyproject.toml` so unexpected passes trigger immediate CI failures.
- Added strict warnings policy: `filterwarnings = ["error"]` with targeted ignores for PyTorch/CUDA
  driver deprecation (`pynvml`) and the backward-compatibility `CUDAIPCImporter` shim.
- Registered standard markers: `smoke`, `integration`, and `quarantine`.
- Added quarantine collection hook in `tests/conftest.py` enforcing `reason`, `issue`, and unexpired
  `expires` date (YYYY-MM-DD) for any quarantined tests.

#### G4 — Skipped-test drift ratchet

- Implemented `scripts/check_skipped_tests.py` using stdlib-only AST parsing to count disabled tests
  vs platform/hardware-gated tests.
- Configured `[tool.check_skipped_tests] max_ratio = 0.02` in `pyproject.toml`.
- Active ratio measured at 1.1% (14 disabled / 1,247 collected, 59 platform-gated excluded), passing well
  under the 2.0% ceiling. Integrated into pre-commit and CI lint stages.

#### G5 — Type checking baseline

- Created `pyrefly-baseline.json` via `pyrefly check --generate-baseline`.
- Establishes a zero-error baseline (15 legacy warnings suppressed), allowing `pyrefly check --baseline`
  to run as a non-breaking blocker in both pre-commit and CI.

#### G6 — 3-Tier GitHub Actions pipeline

- **`ci.yml`**: Parallel lint and test matrix. Built once via universal fallback wheel (`py3-none-any.whl`),
  provenance recorded in `BUILD_INFO.txt`, promoted artifact verified via SHA256 checksums, installed
  via `uv pip install --no-deps`, smoke tested, and exercised via `uv run --no-sync pytest`.
- **`wheels.yml`**: Windows MSVC native wheel build (`cibuildwheel`/`scikit-build-core`) triggering on tags
  and manual dispatch with download-and-test verification.
- **`nightly.yml`**: Weekly deep CI run: the matrix legs PR CI does not cover (py3.13 on both OSes,
  Windows py3.10), a dependency lockfile refresh canary (`uv lock --upgrade`) and a quarantine test
  watcher. A gate job skips the matrix when the default branch has not moved since the last green
  scheduled run, and skips the quarantine watcher when no test is quarantined.

#### G7 — Test hardening & AST smell remediation

- Remediated test smells:
  - Eliminated `sleep` smell in `tests/support/test_console_shutdown.py` by switching to cancelable `threading.Event().wait(5.0)`.
  - Resolved `DeprecationWarning` in `tests/integration/test_importer_deprecation.py` with `pytest.deprecated_call()`.
  - Added assertions to `tests/core/test_numpybuffers_close.py` and `tests/core/test_importer_port.py`.
  - Configured sandboxed `CUPY_CACHE_DIR` isolation in `tests/cuda/test_cuda_runtime_types.py`.
- AST test smells reduced from 82 to 77 (sleep smell count: 0).
- Suite health: **1,322 passed, 3 skipped, 64 deselected in 22.3s, 0 warnings, 99.45% package branch coverage**.

## Campaign record — Phase H (2026-10-19)

### H0 — Audit (full suite, GPU included)

First audit that timed the **full** suite (`-m "requires_native or not requires_native"`), not
only the no-GPU selection. Findings:

| Metric | Measured |
|--------|----------|
| Full suite (GPU) | 1386 passed + 3 skipped, 75.8–82 s |
| Combined branch coverage | 88.21 % (5189 stmts, 555 miss) |
| Random order | 10/10 no-GPU + 3/3 full-suite seeds green |
| Slowest test | 22.5 s (`test_roundtrip_data_integrity`, GPU) — invisible to the no-GPU durations |
| CRAP > 30 (crap4py, measured) | 8 functions; worst `TDReceiverEngine.initialize_receiver` **715** (CC 40, 25 % covered) and `_refresh_on_version_change` 343.5 — replaces the F1 estimate of ≈179 |
| Smells | 77, unchanged; no `.smell-baseline.json` committed |

One random-order run failed (two `-O` subprocess tests exited `0xC0000142`, DLL init failed). It
was a false alarm: a stopped loop was still running, so two suites ran at once. The seed passes
alone and the clean rerun was 13/13 green.

Five follow-up phases came out of it, each on its own branch off `development`: H1 roundtrip
hermeticity and speed, H2 smell triage plus baseline, H3 mutation on the port modules, H4 the
TDReceiver Humble-Object split (after the 2026-10-19 block-review fixes merge), and H5
coverage-report hygiene (tmp copies of `cuda_link` show up as 0-statement files).

### H1 — Roundtrip hermeticity and speed

**Root cause of the 22 s tests.** The producer in `tests/integration/test_roundtrip.py` exported
all its frames on fixed sleeps, often before the consumer had finished spawning and initializing
CUDA. The consumer then sat out its 20 s deadline waiting for frames that never came, and the
test still passed because it asserted only "at least one frame".

**Hermeticity gap.** The producer workers called `Exporter.open` with the default policy, so every
export checked the production `cudalink_activation_barrier` segment. A live TD sender holding that
barrier made the exports skip. With the barrier held by a stand-in for a TD sender
(`HolderBarrier.acquire`), the old `test_roundtrip_exporter_importer` failed with "Consumer
received 0 frames" after 22.3 s.

**Fix.**

- Producer and consumer run in lockstep through three spawn-context Events (`ready`,
  `frame_read`, `done`). The producer waits for each frame to be read before it exports the next,
  so no fixed sleep is left.
- Each producer points `activation_barrier.SHM_NAME` at a per-test segment before opening its
  exporter. `test_integration.py`, which runs the real `Exporter` in-process, now uses the
  existing `isolated_barrier_shm` fixture module-wide.
- Assertions are now exact: every frame arrives (`frames == num_frames`, was `>= 1`), and the
  metadata auto-detect test also checks that the auto-detected importer received a frame.

| Measure | Before | After |
|---------|--------|-------|
| 4 roundtrip tests | 22.6 + 22.5 + 6.4 + 4.4 = 55.9 s | ~2.1 s each, ~8.2 s total |
| `test_roundtrip.py` + `test_integration.py` | 57.8 s | ~10 s |
| Full suite (GPU) | 75.8 s | 29.2–33.4 s (3 random seeds) |
| No-GPU selection | ~18 s | 18.3 s (unchanged; these tests need CUDA) |
| Roundtrip + integration with the production barrier held | old roundtrip fails, 0 frames | 8 passed in 9.8 s |
| Combined branch coverage | 88.21 % | 88.21 % |

This closes the Phase E caution about running concurrently with a live TD session for these two
integration files: they no longer touch the production barrier segment. Other files that call
`Exporter.open` (core exporter tests with fakes, `test_cpu_roundtrip.py`, a few `tests/cuda`
files) were not audited for this in H1, so the general "no pytest beside a live TD session" rule
still stands. The roundtrip tests still use real CUDA IPC, so they stay `requires_cuda` and
local-only.

### H2 — Smell triage and baseline

`scripts/scan_test_smells.py` (AST scanner) over `tests/`. The H2 base (H1 branch) had 73 smells.
The 77 from the audit included four early-exit tests H1 had already fixed. Each no-assert and
early-exit test was rewritten to assert an observable effect, or deleted when it had no
regression protection. No test was skipped or xfailed to drop the count.

| Smell | H2 base | After |
|-------|---------|-------|
| no-assert | 30 | 0 |
| early-exit (`pytest.skip`/`return` mid-test) | 9 | 0 |
| parametrize-no-ids | 15 | 0 |
| multi-mock-assert | 2 | 2 |
| overprotective | 11 | 11 |
| platform-check | 4 | 4 |
| bitwise-assert | 2 | 2 |
| **Total** | **73** | **19** |
| Test functions scanned | 1227 | 1224 |
| Skipped/xfail ratio | 1.1 % (14/1227) | 1.1 % (13/1224) |

**Deleted.** Each of these had no regression protection.

- `test_fake_adapter_graph_exec_memcpy_node_noop`
- `test_fake_adapter_restore_context_is_callable`
- `test_fake_adapter_synchronize_noop`

  These three only called a no-op on the fake adapter.
- `test_abi_guard_passes_for_correct_size` duplicated every other import of the module. The
  mismatch test is now parametrized over `too-small`/`too-large`.
- `test_mode0_invalidates_concurrent_capture` was a strict xfail behind a skip, so it could never
  fail. The mode-2 half stays, and the contract is pinned GPU-free by the new
  `test_build_export_graphs_captures_every_slot_in_relaxed_mode`. That test asserts every slot is
  captured with `StreamCaptureMode.RELAXED`.

**Rewritten.**

- **No-assert tests** now assert the effect they exercised: log output via `capsys`/`caplog`,
  freed slots counted on double close, D2D memcpy pattern round-tripped, event query after
  synchronize, and barrier state.
- **Backend caching** (`test_backend_reuse.py`):
  - The old tests re-implemented the caching inline. The new ones drive `get_frame*` twice and
    assert the same backend instance is reused with `_stream` refreshed.
  - A missing library raises before anything is cached.
- **`test_adaptive_torch_backend_uses_gpu_wait_when_latched`** no longer skips without torch. A
  fake `torch` module is injected, and the test asserts the outgoing `stream_wait_event` args to
  prove the stream handle is cached, rather than stub interactions.
- **`test_get_frame_numpy_without_numpy`** always skipped where numpy is installed, and would have
  failed where it isn't (an unconnected facade returns `None`). It now patches `NUMPY_AVAILABLE`
  on a connected facade.
- **`test_td_bootstrap.py`**: four `if not bootstrap._active: pytest.skip(...)` became
  precondition asserts.
- **Parametrize ids** were added to all 15 parametrizations. A duplicate row in
  `test_shm_protocol.py` (float16 was identical to rgba16fixed) was merged into one row.

**Hand mutants.** Five source mutants were each killed by the rewritten test aimed at them:

- the `_stream` refresh dropped
- the cupy backend rebuilt every call
- the capture mode changed to 0
- the torch wait stream re-read every call
- the facade swallowing the numpy `RuntimeError`

**Gate.** `.smell-baseline.json` is committed, and the scanner exits 1 when any category grows.

- **Pre-commit:** the `scan-test-smells` local hook runs on `tests/**.py` and baseline changes.
- **CI:** the `smell-ratchet` step in the lint job, aggregated like the skip-drift check.

`--baseline` is passed explicitly in both places. The `[tool.scan_test_smells]` fallback in
`pyproject.toml` needs `tomllib` (Python 3.11+). An injected no-assert test failed the gate with
`no-assert increased 0 -> 1`.

**Verification.** No-GPU selection on seeds 1729, 424242 and 90210: 1327 passed, 2 skipped,
63 deselected each, ~22.4 s with coverage. `--cov=cuda_link` branch coverage is 99.35 %
(3154 stmts, 22 miss), equal to the H1 baseline. `fail_under` stays at 85.

## Campaign record — Phase I (2026-10-22)

Audit numbers are in the current baseline above. Four phases, each on its own branch off
`development`: I1 ratchet and baseline, I2 mutation re-run on `shm_protocol` and
`activation_barrier`, I3 `TDReceiver.initialize_receiver` tests, I4 mutation on the port modules.

### I1 — Ratchet and baseline

| Measure | Before | After |
|---------|--------|-------|
| `fail_under` | 85 | 89 (`floor(90.76) - 1.5`, rounded down) |
| Gate run at the new floor | n/a | "Required test coverage of 89.0% reached. Total coverage: 90.76%" |
| Red tests in the full local suite | 6 | 4 (the 2 stale protocol tests fixed; the 4 roundtrip tests need a CUDA torch) |
| No-GPU selection | 1415 passed + 1 skipped | 1415 passed + 1 skipped (unchanged) |

The stale "TDReceiver.py at 51 %" comment in `pyproject.toml` now says 72 %. No test was skipped,
`xfail`ed or deleted. Both repaired tests fail when a protocol member is missing from the adapter
(the same members `test_ctypes_adapter_api_covers_protocol_without_gpu` guards without a GPU).

### I2 — Mutation re-run on `shm_protocol` and `activation_barrier`

Both modules changed in #82/#83 after the 2026-07-12 run. Whole-module cosmic-ray run, then a
second full run after the fixes to confirm the final score.

| Target | Measure | Before (2026-07-12 run) | First re-run | After fixes |
|--------|---------|-------------------------|--------------|-------------|
| `shm_protocol.py` | Mutants | 506 | 506 | 506 |
| | Survivors, genuine / pragma'd | 1 / 51 | 1 / 51 | 0 / 52 |
| | Score | 99.78 % | 99.78 % | **100 %** |
| `activation_barrier.py` | Mutants | 349 | 438 | 434 |
| | Survivors, genuine / pragma'd | 2 / 9 | 18 / 9 | 0 / 13 |
| | Score | 99.4 % | 95.8 % | **100 %** |
| No-GPU selection | Tests | 1415 passed | n/a | 1423 passed, 1 skipped |

What changed:

- **Kill-tests, 8 new** in `test_activation_barrier_cov.py`: `BarrierHeaderError` default and
  keyword-only marker, field indices in the header-mismatch message, the `last_change_ns` offset,
  the `bump_skip` u32 wrap, the stale-age divisor (two rounding cases), and the `_header_warned`
  field flags (added in #83, so never mutated before). Each was verified by applying the mutant by
  hand and confirming the test fails (13 mutants, all killed).
- **Proven-equivalent, now pragma'd:** `write_idx == 0` vs `<= 0` and the two `uninitialised`
  `== 0` checks (unsigned u32 fields), `fields[1] != VERSION` vs `is not` (interned small int), and
  `[0]` vs `[-1]` on a one-field unpack (moved into a `_read_u32` helper so one pragma covers its
  three call sites). A line pragma also hides other mutants on the same line; the proof comments
  note the existing tests that kill them.
- **Tooling fix:** the `cr-*.toml` `test-command` pointed at a Python 3.11 install that imported a
  stale `cuda_link` from site-packages. It now uses the project venv and
  `--randomly-seed=1`, so a mutant gets the same verdict on every run. `cr-profile.toml` still
  has the old path and was not re-run.
- The mutant count differs from the 2026-07-12 run because the modules changed.

Score = killed / (killed + genuine survivors); pragma'd survivors are excluded.

### I3 — `TDReceiverEngine.initialize_receiver` and the par-exec handlers

The audit's three worst CRAP entries. Two new files under `tests/td/`:
`test_tdreceiver_initialize.py` (29 tests) and `test_parexecute_handlers.py` (38 tests). The
receiver tests run the real `initialize_receiver` against an in-memory SharedMemory built with
the real `SHMLayout` and `Metadata` (a fake CUDA runtime is the only double); one test uses a
real `SharedMemory` segment. The handler tests record calls on a stub extension. Every
parametrize carries `ids=`.

| Measure | Before | After |
|---------|--------|-------|
| No-GPU selection | 1423 passed + 1 skipped | 1490 passed + 1 skipped (+67) |
| Branch coverage, combined | 90.76 % | **93.39 %** (5366 stmts, 324 miss) |
| `TDReceiver.py` | 72 % (173 miss) | **82 %** (111 miss) |
| `parexecute_callbacks.py` | 35 % | **95 %** |
| `initialize_receiver` CRAP | 244 (CC 33, 42 % covered) | **33.5** (92.1 % covered) |
| `handle_mode_change` / `handle_numslots_change` CRAP | 72 / 72 (0 %) | **8.0 / 8.0** (100 %) |
| `fail_under` | 89 | 91 (`floor(93.39) - 1.5`, rounded down) |
| Smell scanner | 19 | 19 (ratchet OK) |

CRAP re-measured with `crap4py td_exporter` on an lcov export of the same coverage run. The
remaining `initialize_receiver` figure is complexity: CC 33 is the floor until the function is
split.

Agent-written-test acceptance, by hand-applied mutants (one anchor swapped in the source, the
new tests run, the original bytes restored):

| Target | Mutants | Killed | Survivors |
|--------|---------|--------|-----------|
| `initialize_receiver` | 26 | 24 | 2, proven equivalent |
| `parexecute_callbacks` handlers | 24 | 24 | 0 |

The two `initialize_receiver` survivors are the `width == 0` and `height == 0` guards. Once the
size invariant passes with non-zero bits per component, a zero width or height forces
`data_size == 0`, which the `buffer_size == 0` check in the same condition already rejects. One
mutant (`num_slots` upper bound 11) survived the first pass and led to a boundary test (10
accepted, 11 rejected).

Still uncovered in `initialize_receiver`: two defensive header reads that cannot fail once the
length check passed, the numpy-missing import branch, the bfloat16 fallback, and one loop-exit
branch. The handlers' remaining misses are the no-op stubs for unused callbacks. All tests pass
under four `pytest-randomly` seeds.

### I4 — Mutation testing on `_env`, `_exporter_port` and `_importer_port`

The deferred "ready now" targets: no direct or transitive mocks, whole-module cosmic-ray runs.
Before the new tests, the existing default and `from_env` tests covered only 5-6 fields of each
policy, so most survivors were unchecked defaults and env fallbacks.

| Target | Measure | Before | After |
|--------|---------|--------|-------|
| `_env.py` | Mutants | 67 | 67 |
| | Survivors, genuine / pragma'd | 3 / 11 | 0 / 11 |
| | Score | 94.6 % | **100 %** |
| `_exporter_port.py` | Mutants | 114 | 114 |
| | Survivors, genuine / pragma'd | 29 / 50 | 0 / 50 |
| | Score | 54.7 % | **100 %** |
| `_importer_port.py` | Mutants | 88 | 88 |
| | Survivors, genuine / pragma'd | 41 / 1 | 0 / 1 |
| | Score | 52.9 % | **100 %** |
| No-GPU selection | Tests | 1490 passed + 1 skipped | 1549 passed + 1 skipped (+59) |
| Branch coverage, combined | | 93.39 % | 93.39 % (unchanged: these lines already ran) |
| Smell scanner | | 19 | 19 (ratchet OK) |

What changed:

- **Kill-tests.** `test_exporter_port.py` and `test_importer_port.py` now compare every field of
  `ExportPolicy` / `ImportPolicy` against a literal table taken from `docs/ENV_VARS.md` (the
  backoff ladder is documented only on the field): the dataclass defaults, `from_env()` with every
  variable unset, one test per variable setting `"1"` and `"0"` that must change only its own
  field, and both presets field by field. Also added: `FrameSpec.extra_flags`, the
  `barrier_stale_ns` value, `LIB_STREAM_PRIO` / `D2H_STREAM_PRIO` against `normal`, `high`, an
  unknown value and empty (separates `!=` from `<` / `>`), and `D2H_STREAMS` of `0` / `-3`
  clamping to 1. `test_env.py` gains a keyword-only-default check for the three typed helpers.
- **Proven-equivalent, pragma'd.** Annotation operator mutants (`int | None` becomes
  `int + None`; never evaluated under `from __future__ import annotations`), the default arguments
  of `CudaPort` Protocol stubs (never called; the real defaults live in the adapters), and
  `default=1` vs `0` inside `max(1, env_int(...))` for `D2H_STREAMS`. A line pragma hides every
  mutant on its line; the justification sits next to each.
- Verified by re-running cosmic-ray on the two port modules after the tests and pragmas landed
  (`cr-filter-pragma` applied before `cr-report`); `_env` was verified earlier in the phase. The
  `src` and `td_exporter` diff is comments only; the mirror was re-synced.
- New configs `cr-env.toml`, `cr-exporter_port.toml`, `cr-importer_port.toml` use the project venv
  and `--randomly-seed=1`.

Score = killed / (killed + genuine survivors); pragma'd survivors are excluded.

## Campaign record — Phase J (2026-11-18)

Audit numbers are in the current baseline above. The v0.6 work (86 commits) landed with good patch
coverage (94.4 %) and a 100 % past-fix recall, but pushed the smell ratchet from 19 to 33. J1 brings
it back without touching the baseline file; J2 re-runs mutation testing on `shm_protocol.py`; J3 handles the gate and config findings.

### J1 — Smell ratchet back to the baseline (test code only)

The 14 findings new since Phase I (found by scanning `cb7e951`'s `tests/` with the same scanner and
diffing the lists; the 19 baseline findings are unchanged). No production file, `conftest.py`,
baseline or gate was edited.

| Finding | Tests | Verdict | Change |
|---------|-------|---------|--------|
| `parametrize-no-ids` ×7 | `test_classify_frame_step` (13 rows), the two `GpuFrame` range/type tables, `test_sender_invalid_streamid_is_zero`, `test_frame_delta_wraps`, `test_shmlayout_formula_matches_golden_and_python`, the Framewaitms cook decision | real (K 5.8) | `ids=` naming each case. `test_classify_frame_step` keeps plain tuples, so `test_cpp_frame_step_cases_match_python_table` still reads the table from `pytestmark` |
| `bitwise-assert` ×4 | range limits, frame-mod parity, sender/receiver frame wrap | the shifts stood for named values | `_U32_MAX`/`_U64_MAX`; `INFO_CHOP_FRAME_MOD + 17` and `3 * INFO_CHOP_FRAME_MOD + 9`; the C++ parity test asserts the parsed shift `"24"` and the literal `16_777_216` |
| `overprotective` ×2 | `m is not None` (frame-mod parity), `msg is not None` (`check_ipc_capability` degrade) | real (K 4.9) | guard deleted; the next line fails anyway |
| `no-assert` ×1 | `test_script_top_slot_meta_cook_never_raises` | half real | split: `..._swallows_a_failing_cook` asserts the outgoing `cook(force=True)` happened before the error was swallowed (a mock on a command, allowed); the two no-CHOP cases keep "returns without raising" as their whole contract, with `# smell: ignore[no-assert]` and the reason in the docstring |

The range-limit test now checks all three fields at their maximum instead of `stream_id` alone.

| Measure | Before | After |
|---------|--------|-------|
| Smell scanner total | 33 (ratchet red) | **19** (= `.smell-baseline.json`, ratchet OK) |
| No-GPU selection | 1717 passed + 18 skipped | 1719 passed + 18 skipped (+2 from the split) |
| Branch coverage, combined (Linux) | 91.72 % | 91.72 % |
| Touched files under seeds 1, 2, 3 | n/a | 143 passed + 8 skipped each |

Break and revert (source edited, touched tests run, source restored from git):

| Mutant | Old tests | New tests |
|--------|-----------|-----------|
| `_cook_slot_meta`: `except Exception` narrowed to `(AttributeError, LookupError)` | 1 failed | **2 failed** (the new cook-raises case also catches it) |
| `INFO_CHOP_FRAME_MOD = 1 << 23` | failed | failed (`test_info_chop_frame_mod_matches_cpp`) |

### J2 — Mutation re-run on `shm_protocol.py`

`shm_protocol.py` changed by 185+/54- lines since the I2 run (v0.6 header with `sender_pid`, the
32-byte slot meta, `classify_frame_step`, `protocol_mismatch_message`). Whole-module cosmic-ray
8.9.0 run on Linux, test command = the module's four direct test files (below), `--randomly-seed=1`,
`cr-filter-pragma` before `exec`.

| Measure | I2 (2026-10-22) | J2 first run | After kill-tests |
|---------|-----------------|--------------|------------------|
| Mutants | 506 | 774 (112 pragma'd, 662 run) | 774 |
| Killed | | 639 | **651** |
| Survivors, genuine / proven equivalent | 0 / 52 pragma'd | 12 / 11 | **0** / 11 |
| Score | 100 % | 98.2 % | **100 %** |
| No-GPU selection | | 1719 passed + 18 skipped | 1726 passed + 18 skipped (+7) |
| C++ `topcore_tests` (core-only build) | | 304 checks | 306 checks (+2) |

Score = killed / (killed + genuine survivors); proven-equivalent survivors are excluded. Of the 639
first-run kills, 612 are test failures, 20 are syntax-invalid mutants (an operator in place of the
keyword-only `*`), 6 are `x ** 0xFFFFFFFF` timeouts and 1 is a `metadata_offset << 23` allocation
the kernel OOM-killed; each is a real detection. "After" was checked by re-applying each of the 23
survivor diffs from the session database to the source, running the test command, and restoring the
file from git; a full second session was not run, because the change only adds tests and cannot
un-kill a mutant.

**Kill-tests (7), each from the spec, each with the mutant it kills:**

| Test | Mutants killed (line: operator) | What it pins |
|------|---------------------------------|--------------|
| `test_publish_frame_without_slot_meta_writes_not_set` | 518, 520: `NumberReplacer` ×4 (`producer_frame`/`stream_id` defaults to 1 and -1) | slot meta not passed reads back as 0, the `SlotMeta` "not set" |
| `test_publish_frame_keeps_slot_meta_at_its_range_limits` | 539: `BitAnd_Mod` (`& U64_MAX` -> `% U64_MAX`) | u64/u64/u32 maxima are written unchanged |
| `test_is_same_process_sender_lower_sender_pid_is_another_process` | 470: `Eq_LtE` | a sender pid below our own pid is another process |
| `test_protocol_mismatch_message_zeroed_segment_is_not_v05` | 475: `Eq_LtE` | the v0.6 magic is below the v0.5 one, so `<=` named a zeroed segment "v0.5" |
| `test_protocol_mismatch_message_names_v05_magic_read_from_the_segment` | 475: `Eq_Is` | the magic comes from a buffer read, an equal int that is not the constant object |
| `classify_frame_step` row `next_from_odd_producer_frame` (5, 101, 6, 102) | 266: `Add_BitOr`, `Add_BitXor` | every existing row had an even previous frame, where `\| 1` and `^ 1` equal `+ 1` |
| `classify_frame_step` row `same_write_idx_producer_frame_jump` (5, 100, 5, 102) | 266: `Eq_LtE` | a skip needs write_idx step 1, not step 0 |

The two table rows were added to the C++ twin's `test_frame_step_classification` as well, because
`test_cpp_frame_step_cases_match_python_table` requires the two tables to be equal. The C++ side had
the same hole: with `write_step <= 1` in `frame_step.h` the old C++ tests pass (304 checks, 0
failures); with the new rows that mutant and `(prev_producer_frame | 1)` each fail one check.
All new Python tests pass under seeds 1, 2 and 3.

**Proven equivalent, left unpragma'd (11).** cosmic-ray's pragma is whole-line only
(`.*#.*pragma:.*no mutate.*`). On lines 266 and 470 it would also hide the mutants the new tests kill,
and on 162, 178 and 300 it would hide the `+`/`*` mutants that are killed today, so none was added.
The next run will show these 11 again; this table is the reference:

| Line | Mutant(s) | Why equivalent |
|------|-----------|----------------|
| 162 `slot_offset`, 178 `shutdown_offset` | `24 + n*160` -> `\|`, `^` (×4) | `n*160` is a multiple of 32 (bits 5 and up); 24 is bits 3 and 4: no shared bit for any `n >= 0`, so `+`, `\|` and `^` agree |
| 300 `Metadata.pack_into` | `offset + 4` -> `\|`, `^` (×2) | `metadata_offset = 25 + 160 * num_slots`: bit 2 is always clear. (`+8`, `+12`, `+16` do overlap and are killed) |
| 248, 251 `FrameStep` | `FIRST = -1`, `PRODUCER_SKIP = 4` | no code reads `.value` (searched `src/` and `td_exporter/`); the mutants that collide with another member become aliases and are killed |
| 261 | `prev_write_idx == 0` -> `<= 0` | write_idx is a u32 read, never negative |
| 266 | `write_step == 1` -> `>= 1` | the `> 1` return above leaves `write_step` in {0, 1} |
| 266 | `prev_producer_frame != 0` -> `> 0` | producer_frame is a u64 read, never negative |
| 470 | `sender_pid != 0` -> `> 0` | sender_pid is a u32 read, never negative |

Tooling: the committed `cr-shm_protocol.toml` ran only `test_shm_protocol.py`, which would report
every slot-meta and `classify_frame_step` mutant as a survivor. Its `test-command` now lists the four
direct test files (`test_shm_protocol`, `test_slot_meta_fields`, `test_golden_shm_fixtures`,
`test_cpp_protocol_parity`); the Windows interpreter path is unchanged. The `td/` tests import the
`td_exporter` mirror, which a run does not mutate, and the importer/integration tests are mock-heavy,
so neither is in the set. Production code is unchanged in J2.

### J3 — Gate and config findings

The six findings listed under the current baseline. Config, tooling and docs only: no production
code, test or `conftest.py` change, and no gate value moved (`fail_under` stays 91, the smell
baseline stays 19). Committed on its own, apart from J1/J2, so it can be reviewed or split onto
its own branch.

| Finding | Decision | Change |
|---------|----------|--------|
| Coverage measures the fake `cuda_link/` trees under `tmp_path` | fixed | `[tool.coverage.run] source` names `src/cuda_link` as a path instead of the package name `cuda_link` |
| `-p no:randomly` is a usage error | documented, config kept | "Running the suite" says to pin order with `--randomly-seed=<N>`. `--randomly-dont-reset-seed` stays: dropping it would reseed `random` before every test, which changes what the suite runs, and no reason for it was recorded |
| `replay_fixes.py` and the scikit-build-core finder | documented | the `sitecustomize.py` bypass and the absolute `--test-command` interpreter, in "Running the suite" |
| Ruff pins disagree | fixed | `.pre-commit-config.yaml` `ruff-pre-commit` v0.14.11 -> **v0.17.4**, equal to `uv.lock` (the tag exists upstream). `ruff check .` and `ruff format --check .` under 0.17.4: clean |
| Newer smell scanner | vendored | `scripts/scan_test_smells.py` now has the report-only `real-clock` smell (5 hits, out of the total and the baseline). Formatted to repo style. Ruff's C419 fix ("unnecessary list comprehension in `any()`") is wrong in two places, because `taint()` has side effects and a generator would stop at the first hit, so they carry a `noqa` with that reason. The upstream skill copy has the same two lines |
| CI passes no `--cov-fail-under=0` | **kept as is, on purpose** | `codecov.yml` records the choice: the project floor is enforced *inside* CI by `fail_under`, and Codecov patch coverage (80 %) is the per-change gate. The harden-test-suite skill's default (a local-only ratchet) does not apply here |

| Measure | Before | After |
|---------|--------|-------|
| Combined branch coverage (Linux, bare `--cov`) | 91.72 % (5671 stmts, incl. 29 fake-tree statements counted as covered) | **91.69 %** (5642 stmts); gate 91 passes |
| `coverage lcov` / `coverage json` | fail without `-i` | work |
| CI-style `--cov=cuda_link` | 96.87 % | 96.87 % (unchanged: CI measures the installed wheel by package name, so its 29 fake-tree statements remain, +0.03 points) |
| Smell scanner | 19, OK | 19, OK (`real-clock` 5, report only) |
| `ruff check .` / `ruff format --check .` | clean | clean |
| No-GPU selection | 1726 passed + 18 skipped | 1726 passed + 18 skipped |

The 0.03-point drop is the honest denominator, not lost coverage. The Linux margin over `fail_under`
is now 0.69 points; on Windows the Win32 paths run and the figure is higher.

### Still open

| Item | What | Notes |
|------|------|-------|
| Accepted gap | `RealTDHost.abs_frame` (new, 0 %) and `script_chop_callbacks.onCook` | `td`-bound shells (Humble Object edge); covered by the in-TD run, not the committed suite. The logic they drive (`slot_meta_channels`, stamping) is tested through `FakeTDHost` |

## Deferral table — tooling we deliberately did NOT add

| Tooling | Trigger that would justify it | Current state |
|---------|-------------------------------|---------------|
| `pytest-xdist -n auto` | fast loop > ~5 min | no-GPU 18 s, full suite ~29 s — far below |
| `pytest-split` sharding | per-runner > ~10 min after xdist | n/a |
| Test tiering (unit/slow dirs) | fast loop > 30 s | no-GPU 18 s (full suite ~29 s, local-only) |
| `@pytest.mark.slow` markers | individual test > 5 s | slowest is 2.9 s (was 22.5 s for the GPU roundtrips before H1) |
| Snapshot testing (syrupy) | complex pure outputs hard to assert by hand | none exist |
| New mutation targets | module's mock density approaches zero | backlog above |
| Self-hosted GPU runner | **never** — public repo, excluded by policy | GPU tests local-only |
| TD-layer coverage push | Done in I3 | `TDReceiver.py` 51 % → 82 %; `initialize_receiver` CRAP 244 → 33.5. Next gaps: `TDSender.export_frame` (CRAP 68), `_refresh_on_version_change` (52), `import_frame` (44) |
| `Exporter.export` (CC 43) / `_do_cleanup` (CC 26) Humble-Object split | Complexity ratchet driven below 26 | Both over the CRAP-30 ceiling on complexity alone — see F3 |
| Mutation targets `_exporter_port.py`, `_importer_port.py`, `_env.py` | Done in I4 | All three at 100 % (pragma'd equivalents excluded) |
| Hypothesis for `DtypeCodec.encode`/`decode` | Round-trip fuzz coverage desired | Currently hand-enumerated across 8 `@parametrize` lists |
| Ruff version drift | Before the `C901` ratchet is trusted long-term | Pinned in Phase G, drifted by Phase J (`.pre-commit-config.yaml` v0.14.11 vs `uv.lock` 0.17.4); re-aligned in J3 at 0.17.4. Bump both together |
| 24 mock tokens in `tests/integration/test_cuda_ipc_exporter.py` | Next mock-density audit | 3rd-highest in the suite, in a nominally *integration* test |
