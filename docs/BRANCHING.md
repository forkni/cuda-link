# Branching & Promotion

This repo integrates through a single long-lived branch pair: `development` → `master`.
`master` never receives direct work — it only ever moves forward via a promotion merge
from `development`.

## The development-first rule

**Open pull requests against `development`, never `master`.** `[R7]` in
[`.charlie/instructions/code-style.md`](../.charlie/instructions/code-style.md) states
this explicitly, and [`.github/dependabot.yml`](../.github/dependabot.yml) enforces it
mechanically — every Dependabot version-update PR sets `target-branch: development`, so
even automated dependency bumps reach `master` through the normal promotion, not a
side door.

If a PR is opened against `master` by mistake, retarget it to `development` rather than
merging it directly.

(Dependabot **security** updates are the one exception: they ignore `target-branch` and
always target the repository's default branch, per Dependabot's own behavior — not
something this repo's config controls.)

## How promotion actually runs

Promotion is a manual, deliberate action — not something that fires on every merge to
`development`. It runs via the
[`Merge Development to Master`](../.github/workflows/merge-development-to-master.yml)
GitHub Actions workflow (`workflow_dispatch` only, so it never triggers on push or PR
events).

**Prerequisites:**

- `development` is green — every PR that should be in the promotion has already merged
  and passed CI.
- No one is mid-flight on a `development` PR you'd rather not include; the workflow
  promotes whatever `development` currently points at, not a pinned commit.

**Inputs:**

- `create_backup` (default `true`) — tags `master` at its pre-merge tip
  (`backup-master-before-merge-<timestamp>`) before touching it, so a bad promotion is a
  `git reset --hard <tag>` away from undone.
- `dry_run` (default `false`) — when `true`, prints the file diff and commit log between
  `master` and `development` without merging or pushing anything. Use this first when
  you're unsure what a promotion will bring in.

**What it does, in order:** checks out `master`, verifies `.gitattributes` exists,
optionally tags a backup, then runs `git merge --no-ff development`. A local-only-file
guard checks the merge result doesn't accidentally introduce `CLAUDE.md`, `MEMORY.md`,
`GEMINI.md`, or the `TD_RAG/` / `Backup/` directories into `master` — if it does, the
merge commit is rolled back (`git reset --hard HEAD~1`) and the job fails before
pushing. Only on success does it push `master`.

## The `--no-ff` consequence

Promotion always creates a merge commit, and that commit exists **only on `master`** —
it is never merged back into `development`. This means
`git rev-list --left-right --count origin/master...origin/development` legitimately
reports something like `1  4`, not `0  0`, immediately after a clean promotion: the `1`
is that merge commit (master-only), and the right-hand count is whatever new commits
have landed on `development` since. Neither side being zero is not drift — it's the
expected shape of a one-way `--no-ff` promotion. Don't "fix" it by merging `master` back
into `development`.

## Merge strategy

[`.gitattributes`](../.gitattributes) sets exactly one merge strategy:
`CHANGELOG.md merge=union`, so changelog entries added independently on both branches
combine instead of conflicting. Nothing else in this repo uses a custom merge driver —
earlier revisions of `.gitattributes` carried a set of `merge=ours` rules for docs that
never actually existed in this repository's history, plus an inert `merge=diff3` block
(`diff3` is a *conflict display style*, not a merge driver — it's registered
per-invocation via `-c merge.conflictStyle=...` in
[`scripts/git/merge_with_validation.sh`](../scripts/git/merge_with_validation.sh), not
through `.gitattributes`). Both were removed as dead weight; see the CHANGELOG entry for
this cleanup.

Confirm what's actually active at any time with:

```bash
git check-attr merge -- CHANGELOG.md docs/ARCHITECTURE.md src/cuda_link/exporter.py
```

Only `CHANGELOG.md` should report `union`; everything else reports `unspecified`.
