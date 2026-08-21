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

## Current baseline (2026-08-20, post Phase E/F)

**Supersedes the 2026-07-12 table below.** That table's "20/20 consecutive randomized-order
runs green" claim was re-audited on 2026-08-20 and found **false** (4 failures in 13 runs) —
see Phase E in the campaign record. The number here is freshly re-earned, not re-asserted.

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
manual, local-only analysis tools can never break CI's Python 3.9 gate (`branch-protection.yml`
installs `.[dev]`; `tests.yml` never installs an extra at all). crap4py's only PyPI release is
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
| TD-layer coverage push (`TDReceiver.py` 51 %) | Next campaign | Largest genuine coverage gap, highest CRAP (est. ≈179) — see F1 |
| `Exporter.export` (CC 43) / `_do_cleanup` (CC 26) Humble-Object split | Complexity ratchet driven below 26 | Both over the CRAP-30 ceiling on complexity alone — see F3 |
| Mutation targets `_exporter_port.py`, `_importer_port.py`, `_env.py` | Ready now, deferred for scope | Confirmed zero direct *and* transitive mocks — see F1 |
| Hypothesis for `DtypeCodec.encode`/`decode` | Round-trip fuzz coverage desired | Currently hand-enumerated across 8 `@parametrize` lists |
| Ruff version drift | Before the `C901` ratchet is trusted long-term | `.pre-commit-config.yaml` pins `v0.14.11`, CI (`branch-protection.yml`) pins `0.14.13`, dev extra unpinned, local is `0.14.10` |
| 24 mock tokens in `tests/integration/test_cuda_ipc_exporter.py` | Next mock-density audit | 3rd-highest in the suite, in a nominally *integration* test |
