# Phase J test hardening: status and plan

Status of the Phase J hardening campaign on `feat/orbbec-v0.6` (harden-test-suite skill, run on
2026-11-18), what is left, and the message for the agent that finishes it. Measurements, tables and
reasons for every decision are in [TESTING_GUIDE.md](../TESTING_GUIDE.md) under
"Current baseline (2026-11-18, Phase J audit)" and "Campaign record — Phase J"; this page is the
summary and the to-do list.

## Where things stand

The campaign ran in a Linux cloud container (Python 3.12, CPU torch, no GPU, no TouchDesigner).
Everything that can be checked there is done. Everything below "Remaining" needs the Windows GPU
machine or a decision from the maintainer.

| Phase | Commit | Result |
|-------|--------|--------|
| J0 audit | (recorded in J1) | 10/10 randomized runs green; past-fix recall 5/5; v0.6 patch coverage 94.4 %; smell ratchet **red** (33 vs 19) |
| J1 smell ratchet | `25d95b4` | the 14 new v0.6 smells fixed in test code only; scanner back to 19 |
| J2 mutation `shm_protocol.py` | `03c53e4` | 98.2 % -> **100 %**; 7 kill-tests; two `classify_frame_step` rows mirrored in the C++ `topcore_tests` (the C++ twin had the same hole) |
| J3 gate/config | `c420e7a`, fix `64b43ed` | coverage `source` by path; ruff pre-commit pin = lock (0.17.4); newer smell scanner vendored; `-p no:randomly` and `replay_fixes.py` recipes documented; CI's `fail_under` kept on purpose (`codecov.yml`) |
| J4 Hypothesis properties | `941ef2e` | `tests/core/test_protocol_properties.py` (15 tests) on slot meta, `classify_frame_step`, `GpuFrame`; derandomized `ci` profile in `tests/conftest.py` |
| J5 mutation port modules | `2cc3b6b` | `_exporter_port.py` 95.5 % -> **100 %**, `_importer_port.py` **100 %** |
| J6 prune | `d1b9c64`, `2674017` | 11 duplicate tests deleted, 4 rewritten, 1 kept; smell baseline locked at 18 |

No commit changes product code under `src/` or `td_exporter/`. Test, tooling and gate changes are
in separate commits so they can be reviewed or split apart.

Measured at the end, on Linux:

| Metric | Value |
|--------|-------|
| No-GPU selection | 1730 passed + 18 skipped (18 are Win32-only, with reasons) |
| Branch coverage, combined (`--cov`) | 91.69 % (gate 91); `cuda_link` alone 96.87 % |
| Randomized order | 10/10 green on the final tree (`rank_failures.py`: no failing test) |
| Mutation score | `shm_protocol`, `_exporter_port`, `_importer_port`: 100 % (13 proven-equivalent survivors documented, not pragma'd) |
| Smell scanner | 18 = baseline; `real-clock` 5 (report only) |
| C++ `topcore_tests` (core-only build) | 306 checks, 0 failures |
| ruff check / format, skip ratio | clean / clean / 1.0 % |

## Remaining

### A. Needs the Windows GPU machine

**Status 2026-11-18 (Windows, RTX 4090):** A1 to A5 are done, and all of them passed. Results are
in TESTING_GUIDE.md, "Phase J, Windows follow-up". A1 had 1815 passed and 0 failed. A2 measured
93.25 % and the gate stays at 91. A3 found 0 genuine survivors. A3 also found that the committed
configs hard-code the main checkout's interpreter. A4 gave 306 checks and 0 failures with MSVC.
In A5, `producer_frame` in TD followed the sender's absFrame and no error showed. Only B6 (PR
layout, maintainer) and the C items stay open.

1. **Full suite including GPU and native tests.** Not run since v0.6. The Phase I audit had 4
   `test_roundtrip.py` failures from a CPU-only torch in the venv; `uv sync` swaps a CUDA torch for
   the PyPI one, so install the CUDA build by hand and use `uv run --no-sync`.
2. **Windows coverage figure.** The Linux 91.69 % skips the Win32 paths (`_doorbell.py`,
   `_console.py`), so it is not comparable with Phase I's 93.39 %. Measure on Windows, record it next
   to the Linux one, and raise `fail_under` only if the Windows-and-Linux floor allows it (rule in
   `pyproject.toml`: `floor(x) - 1.5`, never lowered without a written reason). The Linux margin is
   0.69 points, so a raise must still pass the Linux CI run.
3. **cosmic-ray with the committed configs.** J2/J5 ran on Linux with scratch copies of the configs.
   The committed `cr-shm_protocol.toml` and `cr-exporter_port.toml` now list more test files and
   `--hypothesis-profile=ci`; run them as committed and confirm 100 % (genuine survivors 0). The
   expected proven-equivalent survivors are listed in TESTING_GUIDE.md (12 for `shm_protocol`, 1 for
   `_exporter_port`); anything else is a finding.
4. **C++ tests on MSVC.** The two new `test_frame_step_classification` rows passed with GCC
   (core-only build); run the `core-only` (or `msvc-release`) preset's tests on Windows.
5. **In-TouchDesigner check of `RealTDHost.abs_frame`** (new in v0.6, 0 % covered, accepted gap):
   open a sender with the `.tox`, confirm the receiver's `slot_meta` CHOP shows `producer_frame`
   following the sender's absolute frame (mod 2^24), and that no error badge appears.

### B. Maintainer decision

1. **PR layout.** `feat/orbbec-v0.6` has no PR. A PR into `development` carries the v0.6 feature
   commits and the 8 Phase J commits. Options: one PR with the Phase J commits listed separately;
   or revert J1-J6 on the branch (revert commits, no force-push) and re-apply them on their own
   branch and PR. The skill prefers gate changes (`c420e7a`, `2674017`, the J4 `conftest.py` and
   `pyproject.toml` lines) apart from the code they judge.

### C. Deferred, not blocking

1. `TDReceiverEngine.import_frame` CRAP rose 44 -> 48.6 in v0.6; `Exporter.export` is at the ruff
   C901 ceiling (43). Both stay on the Humble-Object split item in the deferral table.
2. Hypothesis for `DtypeCodec.encode`/`decode` (still hand-enumerated); the profile now exists.
3. Upstream skill: the two `any([...])` lines in `scan_test_smells.py` need a `noqa: C419` with the
   side-effect reason (applied in this repo's copy), and the bisect recipe's `-p no:randomly` does
   not work with `--randomly-dont-reset-seed` in `addopts`.

## Follow-up message for the next agent

Paste this into a Claude Code session on the Windows GPU machine, in this repository, with the
harden-test-suite skill installed:

```text
Continue the Phase J test-hardening campaign on branch feat/orbbec-v0.6 (cuda-link).
Read docs/TESTING_PHASE_J_PLAN.md first, then the "Current baseline (2026-11-18, Phase J
audit)" and "Campaign record — Phase J" sections of TESTING_GUIDE.md. Use the
harden-test-suite skill. Phases J1-J6 ran on Linux without a GPU; your job is the
Windows-only items A1-A5 of the plan. Do not change product code, and do not lower any
gate.

1. Venv: make sure torch is the CUDA build (uv sync replaces it with the CPU wheel), then
   use `uv run --no-sync` for everything below.
2. A1: run the full suite, GPU and native included: `uv run --no-sync pytest tests/ -q`.
   For any failure, decide whether it is environmental (driver, torch build) or a
   regression from v0.6, and say which, with the error.
3. A2: measure combined branch coverage with the CI selection:
   `uv run --no-sync pytest tests/ -m "not requires_cuda and not requires_native and not integration and not quarantine" --cov --cov-branch -q`.
   Record it next to the Linux 91.69 %. Propose a fail_under change only if the rule in
   pyproject.toml allows it on both platforms; put any change in its own commit.
4. A3: for cr-shm_protocol.toml, cr-exporter_port.toml and cr-importer_port.toml, as
   committed: cosmic-ray init <cfg> <cfg>.sqlite; cr-filter-pragma <cfg>.sqlite;
   cosmic-ray baseline <cfg>; cosmic-ray exec <cfg> <cfg>.sqlite; cr-report <cfg>.sqlite.
   Compare every survivor with the proven-equivalent lists in TESTING_GUIDE.md (J2, J5).
   A survivor not on those lists gets a kill-test written from the spec, under the skill's
   acceptance rule (named mutant killed, no recomputed expectation, three seeds).
5. A4: build and run the C++ core tests with the core-only preset in cpp_top/.
6. A5: in TouchDesigner, check RealTDHost.abs_frame end to end (sender .tox -> receiver
   slot_meta CHOP shows producer_frame following the sender's absolute frame mod 2^24, no
   error badge). Report what you saw; this is the verify step for the one accepted gap.
7. Record each result as a new row or section in TESTING_GUIDE.md ("Phase J, Windows
   follow-up"), commit per item, and push to feat/orbbec-v0.6. Do not open a PR: the PR
   layout (plan item B6) is the maintainer's decision.
Report at the end: each item as done / failed (with evidence) / blocked (with reason).
```
