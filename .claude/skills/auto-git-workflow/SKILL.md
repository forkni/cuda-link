---
name: auto-git-workflow
description: Automated git workflow behavioral rules — always use project scripts, protect local files, handle conflicts properly
---

# Auto Git Workflow Skill

## 🎯 On Activation

**When this skill loads:**

1. Acknowledge: "Auto git workflow skill active. I'll use project scripts for all git operations and protect local-only files."
2. Wait for the user's actual task
3. Apply the guidance below to all git operations in this session

**DO NOT**: Explore or analyze this skill document, launch agents to investigate the skill, or treat this as a request for information about git workflows.

---

## Purpose

Ensures all git operations in this project follow established patterns:
- Use project scripts (`scripts/git/*.sh` or `scripts/git/batch/*.bat`) instead of raw git commands
- Protect local-only files from accidental commits
- Handle merge conflicts automatically when safe
- Maintain comprehensive logging
- Follow conventional commit message format

---

## ⚠️ Core Rules (MANDATORY)

### Rule #1: NEVER Use Raw `git commit`

**Always use:**
```bash
./scripts/git/commit_enhanced.sh [flags] "commit message"
```

**NEVER use:**
```bash
git commit -m "message"  # ❌ WRONG
```

**Why**: `commit_enhanced.sh` provides:
- Automatic lint validation (Python + Markdown)
- Local-only file protection (CLAUDE.md, MEMORY.md, etc.)
- Branch-specific validation (no tests on main)
- Commit message format checking
- Comprehensive logging

### Rule #2: Use `--no-venv` When No Virtual Environment

**Check for `.venv` directory:**
- If `.venv` exists → use script normally
- If `.venv` missing → add `--no-venv` flag

**Example:**
```bash
# No .venv in project
./scripts/git/commit_enhanced.sh --no-venv "feat: add new feature"
```

The `--no-venv` flag uses system-installed `ruff` instead of `.venv/Scripts/ruff.exe`.

### Rule #3: NEVER Commit Local-Only Files

**Files that MUST NEVER be committed:**

Core local-only files:
- `CLAUDE.md` — Claude Code project instructions (local-only)
- `MEMORY.md` — Claude Code auto-memory (session-specific)
- `GEMINI.md` — Gemini AI config (if present)
- `BASH_STYLE_GUIDE.md`, `BATCH_STYLE_GUIDE.md`, `SHELL_STYLE_GUIDE.md`
- `SLASH_COMMAND_TESTING_PROTOCOL.md`
- `clean_pycache.bat`, `clean_pycache.cmd`

Directories:
- `_archive/` — Old code not subject to current standards
- `benchmark_results/` — Local benchmark output
- `analysis/` — Local analysis results
- `logs/` — Execution logs
- `scripts/git/` — Git automation scripts (varies by project)

**Before any commit:** Verify these are not staged:
```bash
git diff --cached --name-only | grep -E "(CLAUDE\.md|MEMORY\.md|_archive|benchmark_results|logs)"
```

If found, unstage before committing.

### Rule #4: Chain Git Commands to Prevent Lock Files

**Problem**: Separate Bash tool calls can create lock file race conditions.

**Solution**: Chain related git commands with `&&`:

```bash
# ✅ CORRECT - single chained command
git add . && git commit -m "message" && git push origin development

# ❌ WRONG - separate calls allow lock file race
# Call 1: git add .
# Call 2: git commit -m "message"  # Lock can appear here!
# Call 3: git push origin development
```

**Lock file handling**: If `.git/index.lock` found, remove it before git operations:
```bash
rm -f .git/index.lock && git add . && git commit -m "message"
```

---

## 📋 Available Scripts Reference

### Shell Scripts (Git Bash / Linux / macOS — PRIMARY)

**Commit workflow:**
- `./scripts/git/commit_enhanced.sh [flags] "message"` — Enhanced commit with validation
  - Flags: `--non-interactive`, `--skip-md-lint`, `--no-venv`, `--staged-only`, `--interactive`

**Lint validation:**
- `./scripts/git/check_lint.sh` — Pre-commit validation (3 tools: ruff check, ruff format, markdownlint)
- `./scripts/git/fix_lint.sh` — Auto-fix lint issues

**Branch management:**
- `./scripts/git/validate_branches.sh` — Validate branch state before operations
- `./scripts/git/merge_with_validation.sh [--non-interactive]` — Merge development → main with validation
- `./scripts/git/rollback_merge.sh` — Rollback merge using backup tag

**Other:**
- `./scripts/git/cherry_pick_commits.sh` — Selective commit picking
- `./scripts/git/merge_docs.sh` — Documentation merging
- `./scripts/git/install_hooks.sh` — Git hooks installation

### Batch Scripts (Windows CMD — Fallback)

Located in `scripts/git/batch/`:
- `commit_enhanced.bat --non-interactive "message"`
- `check_lint.bat`
- `fix_lint.bat`
- `merge_with_validation.bat --non-interactive`
- `rollback_merge.bat`
- etc.

**Environment detection:**
```bash
echo $OSTYPE
# Git Bash: "msys"
# Linux: "linux-gnu"
# macOS: "darwin"
# Empty: Windows cmd.exe
```

Use `.sh` scripts for Git Bash/Linux/macOS (primary). Use `.bat` scripts only if in Windows cmd.exe.

---

## 🚀 `commit_enhanced.sh` Flags

Complete flag reference for the commit script:

| Flag | Purpose | When to Use |
|------|---------|-------------|
| `--non-interactive` | Skip all prompts, use defaults | Claude Code (auto-detected), CI/CD, automation |
| `--skip-md-lint` | Skip markdown lint checks | Python-only projects, known MD errors |
| `--no-venv` | Use system tools (no .venv) | Projects without virtual environment |
| `--staged-only` | Commit pre-staged files only | Selective commits, incremental work |
| `--interactive` | Force interactive mode | Debugging, manual control |

**Auto-detection**: When no TTY (Claude Code context), script automatically enables `--non-interactive` + `--skip-md-lint`.

**Environment variables:**
- `CLAUDE_GIT_NON_INTERACTIVE=1` — Force non-interactive
- `CLAUDE_GIT_SKIP_MD_LINT=1` — Force skip markdown lint
- `CLAUDE_GIT_STAGED_ONLY=1` — Use pre-staged files only
- `CLAUDE_GIT_NO_VENV=1` — Use system tools

**Examples:**
```bash
# Auto-detected mode (typical for Claude Code)
./scripts/git/commit_enhanced.sh "feat: add feature"

# Explicit flags (optional)
./scripts/git/commit_enhanced.sh --no-venv "feat: add feature"

# Selective commit
git add src/specific_file.py
./scripts/git/commit_enhanced.sh --staged-only "fix: specific bug"
```

---

## 🔒 Branch Rules

### Main Branch

**Allowed:**
- Production-ready code
- Documentation (README.md, ARCHITECTURE.md, etc.)
- Core source (`src/`, `td_exporter/`)
- Essential configs (pyproject.toml, .gitignore)

**NOT allowed:**
- `tests/` directory
- `pytest.ini`
- Development-only docs (TESTING_GUIDE.md, GIT_WORKFLOW.md)
- Development tools (benchmark scripts, analysis tools)

**Validation**: `commit_enhanced.sh` automatically checks and blocks test files on main.

### Development Branch

**Allowed:**
- Everything (tests, dev docs, benchmarks, analysis tools, etc.)

**Standard workflow:**
1. Work on `development` branch
2. Commit and test there
3. Merge to `main` via `./scripts/git/merge_with_validation.sh`
4. Push both branches

---

## 🔀 Conflict Resolution

### Modify/Delete Conflicts (EXPECTED)

**Status code:** `DU` (deleted by us, modified by them)

**When:** Merging development → main when test files exist on development but not on main.

**Action:** Auto-resolve (already handled by `merge_with_validation.sh`):
```bash
git status --short | grep "^DU " | awk '{print $2}' | xargs -r git rm
git commit --no-edit
```

**These are expected** — don't treat as errors.

### Content Conflicts (UNEXPECTED)

**Status code:** `UU` (both modified)

**When:** Same file modified differently on both branches.

**Action:** STOP workflow, require manual resolution:
```bash
# Edit files to resolve conflicts
git add <resolved-files>
git commit

# Or abort merge
git merge --abort
git checkout development
```

**Never auto-resolve content conflicts** — they require human review.

---

## 📊 Logging

All git automation scripts generate logs in `logs/` directory.

**Log naming pattern:**
- `logs/commit_enhanced_YYYYMMDD_HHMMSS.log`
- `logs/check_lint_YYYYMMDD_HHMMSS.log`
- `logs/merge_with_validation_YYYYMMDD_HHMMSS.log`

**Log format:**
```
========================================
[SECTION NAME] Started: HH:MM:SS
========================================
[tool output]
[SECTION NAME] Ended: HH:MM:SS (Xs) - PASSED/FAILED
```

**Analysis reports** (optional):
- `logs/commit_enhanced_analysis_YYYYMMDD_HHMMSS.log`

Logs are automatically excluded from commits via `.gitignore`.

---

## 🚨 Error Recovery

### Lint Failures

**When:** `check_lint.sh` or `commit_enhanced.sh` reports lint errors.

**Python lint (ruff):**
```bash
./scripts/git/fix_lint.sh  # Auto-fix
./scripts/git/check_lint.sh  # Verify
```

**Markdown lint (markdownlint):**
- Ignore errors in `CLAUDE.md` / `MEMORY.md` (local-only files, won't be committed)
- For production MD files: fix manually or use `--skip-md-lint` flag

### Push Failures

**Possible causes:**
- Network issues
- Authentication failure
- Remote has diverged
- Branch protection rules

**Action:**
1. Check error output
2. If remote diverged: `git pull --rebase origin <branch>` then retry push
3. If auth failure: check credentials
4. Otherwise: resolve manually

### Merge Rollback

**When:** Need to undo a merge to main.

**Action:**
```bash
./scripts/git/rollback_merge.sh
```

This script:
- Resets main to pre-merge state using backup tag (`pre-merge-backup-YYYYMMDD_HHMMSS`)
- Preserves backup tag for reference
- Returns to development branch
- Logs all actions

**Manual rollback** (if script unavailable):
```bash
git checkout main
git reset --hard pre-merge-backup-YYYYMMDD_HHMMSS
git checkout development
```

### No Changes to Commit

**When:** `git diff --quiet && git diff --cached --quiet` succeeds (exit code 0).

**Action:** Display "⚠ No changes to commit" and stop workflow (not an error).

---

## 📝 Commit Message Format

**Conventional commit format** (enforced by `commit_enhanced.sh`):

| Prefix | Use Case | Example |
|--------|----------|---------|
| `feat:` | New feature | `feat: add CUDA IPC zero-copy texture transfer` |
| `fix:` | Bug fix | `fix: resolve memory leak in ring buffer cleanup` |
| `docs:` | Documentation | `docs: update TOX build guide with screenshots` |
| `chore:` | Maintenance | `chore: update .gitignore for logs directory` |
| `test:` | Test changes | `test: add integration tests for IPC protocol` |
| `refactor:` | Code refactoring | `refactor: extract ring buffer logic to separate class` |
| `style:` | Code style | `style: apply ruff format to importer` |
| `perf:` | Performance | `perf: optimize D2D memcpy for 4K textures` |

**Format validation:** Script warns if commit message doesn't follow format, but allows override in interactive mode.

---

## 🔧 Environment-Specific Behavior

### Git Bash / Linux / macOS (Primary)

- Use `.sh` scripts from `scripts/git/`
- Auto-detection works (checks for TTY)
- Shell-native execution (no wrappers needed)

### Windows cmd.exe (Alternative)

- Use `.bat` scripts from `scripts/git/batch/`
- Must explicitly pass flags (`--non-interactive`, `--skip-md-lint`)
- Execute with: `scripts\git\batch\commit_enhanced.bat --non-interactive "message"`

### Claude Code Context

- Automatically detected as non-interactive (no TTY)
- Both `--non-interactive` and `--skip-md-lint` enabled by default
- Can force interactive with `--interactive` flag (for debugging)

---

## 🎯 Quick Decision Tree

**When you need to commit code:**

```
Do you have a .venv directory?
├─ Yes → ./scripts/git/commit_enhanced.sh "feat: message"
└─ No  → ./scripts/git/commit_enhanced.sh --no-venv "feat: message"

Are CLAUDE.md or MEMORY.md staged?
├─ Yes → git reset HEAD CLAUDE.md MEMORY.md (unstage them first)
└─ No  → Proceed with commit

Did lint checks fail?
├─ Yes → Run ./scripts/git/fix_lint.sh then retry
└─ No  → Commit proceeds

Are you on the main branch trying to commit tests?
├─ Yes → ERROR: Tests not allowed on main
└─ No  → Proceed
```

**When you need to merge to main:**

```
./scripts/git/merge_with_validation.sh --non-interactive
```

This handles:
- Pre-merge validation
- Automatic backup tag creation
- Modify/delete conflict resolution (test files)
- Content conflict detection (stops for manual resolution)

**When you need to rollback:**

```
./scripts/git/rollback_merge.sh
```

---

## Summary

This skill ensures git operations in this project are:
- ✅ Script-based (not raw git commands)
- ✅ Protected (local-only files never committed)
- ✅ Validated (lint checks pass before commit)
- ✅ Logged (comprehensive audit trail)
- ✅ Safe (conflicts handled appropriately)
- ✅ Conventional (proper commit message format)

**Remember**: The `/auto-git-workflow` command executes the full commit→push→merge→push workflow. This skill provides the underlying behavioral rules applied to ANY git operation you perform in this project.
