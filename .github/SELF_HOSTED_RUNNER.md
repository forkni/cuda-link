# Self-Hosted GPU Runner Setup

`gpu-tests.yml` runs the `requires_cuda` pytest suite on a self-hosted Windows
machine with an NVIDIA GPU — the tests that hosted CI permanently deselects.
This document covers registering the runner and the security posture that makes
a self-hosted runner safe on a **public** repository.

## Why label-gated

Anyone can open a PR against a public repo. Without gating, a fork PR could run
arbitrary code on the runner machine. The `gpu-tests` job therefore only runs
when:

- a maintainer applies the **`gpu-ci`** label to the PR (a per-PR trust
  decision, made after reading the diff), or
- a maintainer triggers **workflow_dispatch** manually.

Pushes to an already-labeled PR re-run the job (`synchronize` event), so
**remove the label before requesting changes from an untrusted contributor**,
and re-apply it after reviewing their new commits.

## Machine prerequisites

- Windows 10/11 with an NVIDIA GPU and a CUDA toolkit installed
  (`cudart64_*.dll` discoverable — the suite loads it via ctypes)
- Python 3.10+ on `PATH`
- No MSVC required: the GPU suite imports from `src/` via pytest's
  `pythonpath` config, it does not build the native extension
  (`requires_native` stays covered by `native-build.yml` and `release.yml`)

## Registering the runner

1. GitHub → `forkni/cuda-link` → **Settings → Actions → Runners → New
   self-hosted runner** → Windows x64. Follow the download/configure snippet it
   shows; when running `config.cmd`, add the custom label:

   ```text
   ./config.cmd --url https://github.com/forkni/cuda-link --token <TOKEN> --labels gpu
   ```

   The runner then carries `self-hosted`, `Windows`, `X64`, `gpu` — matching
   the workflow's `runs-on: [self-hosted, Windows, gpu]`.

2. Run it. Two options:
   - **Interactive** (`run.cmd`): inherits your user environment — CUDA_PATH
     and PATH behave exactly as in your shells. Simplest; stops when you log
     out.
   - **Windows service** (`svc install` + `svc start`): survives reboots, but
     runs under a service account whose environment is the *machine-level* env,
     not your user env. If the suite fails with "CUDA runtime not available",
     check that `CUDA_PATH`/`PATH` are set at machine level (same failure mode
     as the TouchDesigner stale-env issue: a parent process with a stale or
     different environment).

3. Create the gating label once (already done if `gh label list` shows it):

   ```bash
   gh label create gpu-ci --color B60205 --description "Run GPU tests on the self-hosted runner"
   ```

## Repository security settings (verify once)

Settings → Actions → General:

- **Fork pull request workflows**: "Require approval for all outside
  collaborators" — first-time/outside PRs then need a maintainer click before
  *any* workflow runs, a second gate in front of the label.
- Leave "Allow all actions" narrowed if practical; every action in this repo's
  workflows is SHA-pinned regardless.

## Operational notes

- **Timeout**: the job has `timeout-minutes: 30` so a hung CUDA call can't wedge
  the runner forever.
- **Queued jobs when the machine is off**: a labeled PR with the runner offline
  sits "Queued" until the runner comes back or GitHub expires the job (24 h).
  Apply the label only when the runner is up.
- **Hygiene**: the job creates and deletes a per-run venv; the checkout action
  cleans the workdir between runs. For stronger isolation, register the runner
  with `--ephemeral` (accepts one job, then deregisters — needs re-registration
  per job, so only worth it with outside automation).
- The runner executes whatever the labeled PR's `gpu-tests.yml` says — the
  label is a decision to trust *that diff*, including its workflow changes.
