# Automated Git Workflow - Claude Orchestration Guide

**Purpose**: Step-by-step instructions for Claude Code to execute automated commit→push→merge→push workflows with comprehensive logging.

**For detailed troubleshooting**: See [GIT_WORKFLOW.md](./GIT_WORKFLOW.md)

---

## ⚠️ CRITICAL: Environment Requirements

**Directory Structure**:

- **Shell scripts (.sh)**: `scripts/git/` - **PRIMARY for automated workflows**
- **Batch scripts (.bat)**: `scripts/git/batch/` - Manual Windows CMD execution only

| Environment | Recommended | Reason |
|-------------|-------------|--------|
| **Git Bash / MSYS** | Use `.sh` scripts (PRIMARY) | Cross-platform, automated workflows |
| **Linux / macOS** | Use `.sh` scripts (PRIMARY) | Native bash support |
| **Windows CMD** | Use `.bat` scripts (manual) | Batch script support, fallback option |

**Batch Scripts Limitation**:

Batch scripts (.bat) MUST be run from Windows CMD, NOT Git Bash. Variable scoping fails in Git Bash.

**Error if run from Git Bash**:

```
[ERROR] This script must be run from Windows CMD, not Git Bash
Detected environment: MSYSTEM=MSYS
```

**Solutions** (in order of preference):

1. **Use shell scripts (PRIMARY):**

   ```bash
   # Git Bash / Linux / macOS (PREFERRED)
   ./scripts/git/commit_enhanced.sh "feat: Your commit message"
   ```

2. **Use Windows CMD with batch scripts (manual):**

   ```bash
   # CORRECT - Message as separate argument:
   cmd.exe /c "scripts\git\batch\commit_enhanced.bat" "feat: Your commit message"

   # WRONG - Message inside quoted command (causes empty output):
   cmd.exe /c "scripts\git\batch\commit_enhanced.bat \"feat: message\""
   ```

3. Use direct git commands with lint checks (see Fallback Workflow below)

**Why this matters**: Git Bash cannot properly execute Windows batch scripts. Variable scoping fails, causing empty `%LOGFILE%` redirections that create accidental files named "Commit" and "Workflow" in the project root.

### Fallback Workflow (When Batch Scripts Fail)

**Preferred**: Use shell scripts instead of batch scripts:

```bash
# Use shell scripts (RECOMMENDED)
./scripts/git/commit_enhanced.sh "feat: Your message"
```

If shell scripts are unavailable and `cmd.exe /c` approach doesn't work, follow these steps **IN ORDER** to preserve lint validation:

1. **Run lint checks first:**

   ```bash
   cmd.exe /c "scripts\git\batch\check_lint.bat"
   ```

2. **Fix lint issues if any:**

   ```bash
   cmd.exe /c "scripts\git\batch\fix_lint.bat"
   ```

3. **Only then commit with chained commands:**

   ```bash
   git add . && git commit -m "feat: Your message"
   ```

**⚠️ CRITICAL:** NEVER skip steps 1-2 when falling back to raw git commands. Lint validation is mandatory.

---

## User Request Pattern

```
"Commit and push to development, then merge to main following AUTOMATED_GIT_WORKFLOW.md"
```

---

## Workflow Overview

1. Create comprehensive log file
2. Execute 5 phases sequentially
3. Capture all output
4. Monitor and fix issues
5. Provide complete audit trail

**Log Location**: `logs/workflow_commit_merge_YYYYMMDD_HHMMSS.log`

**Format Reference**: Logs use structured format with section headers, timestamps, durations, and summary tables (see Enhanced Logging Infrastructure below)

---

## Enhanced Logging Infrastructure

All git automation scripts now use comprehensive structured logging provided by 5 core functions in `scripts/git/_common.sh`:

### 1. `log_section_start(section_name, log_path)`

Creates timestamped section headers for each validation or operation step.

**Output Format**:

```
========================================
[SECTION NAME] Started: HH:MM:SS
========================================
```

**Internal State**: Sets `SECTION_START_TIME` for duration calculation

### 2. `log_section_end(section_name, log_path, exit_code, error_count)`

Closes sections with duration and status tracking.

**Output Format**:

```
[SECTION NAME] Ended: HH:MM:SS (Xs) - PASSED
# or
[SECTION NAME] Ended: HH:MM:SS (Xs) - FAILED (N errors)
```

**Parameters**:

- `exit_code`: 0 = PASSED, non-zero = FAILED
- `error_count`: Optional, number of errors detected (defaults to 0)

### 3. `run_tool_with_logging(tool_name, log_path, command...)`

Executes lint tools with automatic output capture and error counting.

**Features**:

- Captures both stdout and stderr (`2>&1`)
- Counts structured errors (e.g., `file.py:10:5: E501 line too long`)
- Sets `TOOL_OUTPUT` and `TOOL_ERROR_COUNT` variables
- Automatically logs section start/end with durations

**Example Usage**:

```bash
run_tool_with_logging "RUFF CHECK" "$logfile" \
    "${PYTHON_BIN}/ruff" check . --extend-exclude "tests/test_data"
```

### 4. `log_summary_table(log_path, results...)`

Generates formatted summary tables at the end of multi-tool workflows.

**Output Format**:

```
========================================
[ERROR SUMMARY]
========================================
Tool           Status   Errors   Duration
----           ------   ------   --------
Ruff           PASSED   0        1s
Format         PASSED   0        0s
Markdownlint   PASSED   0        2s
```

**Input Format**: Array of strings: `"ToolName:STATUS:ErrorCount:DurationSeconds"`

### 5. `run_git_with_logging(section_name, log_path, git_args...)`

Executes git commands with full output capture and logging.

**Features**:

- Logs the exact git command being executed
- Captures full git output (both stdout and stderr)
- Sets `GIT_OUTPUT` and `GIT_EXIT_CODE` variables
- Includes section headers and duration tracking

**Example Usage**:

```bash
run_git_with_logging "GIT STATUS" "$logfile" status --short
# Logs: Command: git status --short
# Then captures and logs git's output
```

### Scripts Using Enhanced Logging

All git automation scripts now implement comprehensive logging:

**Complete Rewrites** (full structured logging):

- `check_lint.sh` - Pre-commit validation with 3 Python tools (ruff check, ruff format, markdownlint)
- `fix_lint.sh` - Auto-fix with file modification tracking

**Enhanced Scripts** (section headers + git logging):

- `validate_branches.sh` - Branch state validation
- `merge_with_validation.sh` - Development → main merge
- `rollback_merge.sh` - Merge rollback operations
- `cherry_pick_commits.sh` - Selective commit picking
- `merge_docs.sh` - Documentation merging
- `install_hooks.sh` - Git hooks installation

**Log Output Characteristics**:

- Section headers with `========` separators
- Per-section timestamps (HH:MM:SS format)
- Duration tracking (seconds elapsed)
- Pass/fail status with error counts
- Full tool/git output captured
- Summary tables for multi-step operations

---

## STEP 1: Initialize Workflow Log

### Actions

1. Create logs/ directory if not exists
2. Generate timestamp: `YYYYMMDD_HHMMSS`
3. Create log file: `logs/workflow_commit_merge_TIMESTAMP.log`
4. Write header to log

### Log Header Template

```
=== [Workflow Title] Commit & Merge Workflow ===
Start Time: YYYY-MM-DD HH:MM:SS

================================================================================
                    [WORKFLOW NAME - descriptive title]
================================================================================

Objective: [Brief description of changes being committed]

Changes Summary:
- [Summary line 1]
- [Summary line 2]

Files to Commit (X total):
=================================

New Files (X):
1. path/to/file (size) - Description

Modified Files (X):
1. path/to/file - Description
   - Change detail 1
   - Change detail 2

================================================================================
                          WORKFLOW EXECUTION LOG
================================================================================
```

### Implementation

- Use Bash tool with output redirection
- Write header immediately (don't wait until end)
- Append to log file throughout execution using `>>`

---

## STEP 2: Verify Repository State

### Actions

1. **Check for lock files**: `ls -la .git/*.lock 2>/dev/null`
2. Get current branch: `git branch --show-current`
3. Check working directory: `git status --short`
4. Verify no uncommitted changes that would block checkout
5. Log initial state to audit trail

### Log Output Template

```
================================================================================
[INITIAL STATE VERIFICATION]
Time: HH:MM:SS

Lock File Check:
  .git/index.lock: ❌ Not present (clean)

Current Branch: development
Working Directory Status:
  [git status --short output, or "No changes" if clean]

State: ✅ Clean working directory
```

### If Uncommitted Changes Detected

```
================================================================================
[INITIAL STATE VERIFICATION]
Time: HH:MM:SS

Current Branch: feature/some-branch
Working Directory Status:
  M  path/to/modified/file.py
  ??  path/to/untracked/file.py

State: ⚠️ Uncommitted changes detected

Options:
  - Include in commit: These changes will be staged and committed
  - Stash first: git stash (retrieve later with git stash pop)
  - Abort: Exit workflow to handle manually
```

### Branch Verification Logic

**If on development branch with clean state:**

- Log: "✅ Already on development branch, clean state"
- Continue to Phase 1

**If on different branch:**

- Log: "Current branch: [branch_name]"
- Log: "Will checkout to development branch"
- Check for uncommitted changes first

**If uncommitted changes exist:**

- Log all modified/untracked files
- Default action: Include changes in commit (most common case)
- Log: "Including uncommitted changes in this workflow"

### Error Conditions

**Cannot checkout (uncommitted changes would be overwritten):**

- Log error details
- Suggest: `git stash` or `git commit`
- Abort workflow

### Lock File Detection and Removal

**IMPORTANT**: Always check for and remove stale lock files before git operations.

```bash
# Check for lock files
if [ -f ".git/index.lock" ]; then
    echo "⚠️ Git lock file detected: .git/index.lock"

    # Check if any git process is running
    tasklist | findstr -i git
    if [ $? -ne 0 ]; then
        echo "No git processes running - lock file is stale"
        echo "Removing stale lock..."
        rm -f .git/index.lock
        echo "✓ Stale lock file removed"
    else
        echo "Git process still running - wait for completion"
        exit 1
    fi
fi
```

**Log output when lock found:**

```
Lock File Check:
  .git/index.lock: ⚠️ Found (0 bytes, stale)
  Git processes: None running
  Action: Removed stale lock file
  ✓ Lock file cleared
```

### Best Practice: Chain Git Commands

**CRITICAL**: To prevent lock file issues between Bash tool calls, always chain related git commands.

**❌ Avoid** (separate Bash calls create lock windows):

```bash
# Call 1
git add .
# Lock can appear here!
# Call 2
git commit -m "message"
# Lock can appear here!
# Call 3
git push origin development
```

**✅ Use** (single chained command):

```bash
git add . && git commit -m "message" && git push origin development
```

**Why this matters:**

- Claude Code's git monitoring checks repository state between Bash calls
- IDE extensions (VS Code) may also query git status
- Chaining prevents any process from grabbing the lock mid-workflow

**Also chain lock removal with the first git operation:**

```bash
rm -f .git/index.lock && git add . && git commit -m "message" && git push origin development
```

---

## STEP 3: Analyze Staged Changes

### Actions

1. Run `git status --short`
2. Run `git diff --cached --name-status`
3. Run `git diff --cached --stat`
4. Count new vs modified files
5. Describe each file's purpose
6. Write to log under "Files to Commit"

### Example Output

```
Files to Commit (4 total):
=================================

Modified Files (4):
1. docs/GIT_WORKFLOW.md - Added automated workflow documentation
   - New section: Automated Workflows (Non-Interactive Mode)
   - Updated Quick Commands Reference
2. scripts/git/commit_enhanced.bat - Added --non-interactive flag support
   - Auto-stages all changes in non-interactive mode
   - Auto-fixes lint issues automatically
3. scripts/git/merge_with_validation.bat - Added --non-interactive flag
   - Added flag for consistency (already non-interactive)
4. scripts/git/check_lint.bat - Added mandatory logging infrastructure
   - Creates timestamped log files
   - Records all validation results
```

---

## STEP 4: Execute Phase 1 - Pre-Commit Validation

### Log Phase Header

```
================================================================================
[PHASE 1: PRE-COMMIT VALIDATION]
Status: IN PROGRESS
Time: HH:MM:SS
```

### Actions

1. Ensure on development branch: `git checkout development`
2. Run lint check:
   - Shell (PRIMARY): `./scripts/git/check_lint.sh`
   - Batch (fallback): `scripts/git/batch/check_lint.bat`
3. Capture output (3 Python tools: ruff check, ruff format, markdownlint)
4. Log each tool's result

**Note**: Lint checks automatically skip `tests/test_data/` and `_archive/` directories (see Phase 2 for details on automatic exclusions)

### If Lint Errors Found

- Run fix lint:
  - Shell (PRIMARY): `./scripts/git/fix_lint.sh`
  - Batch (fallback): `scripts/git/batch/fix_lint.bat`
- Capture what was fixed
- Re-run check lint to verify
- Log auto-fix results

### Log Output Template

**New Structured Format** (using enhanced logging functions):

```
========================================
[RUFF CHECK] Started: 22:47:09
========================================
All checks passed!
[RUFF CHECK] Ended: 22:47:09 (0s) - PASSED


========================================
[RUFF FORMAT CHECK] Started: 22:47:09
========================================
All checks passed!
[RUFF FORMAT CHECK] Ended: 22:47:10 (1s) - PASSED


========================================
[MARKDOWNLINT CHECK] Started: 22:47:10
========================================
markdownlint-cli2 v0.14.0 (markdownlint v0.35.0)
Finding: **/*.md !node_modules !.venv
Linting: 28 file(s)
Summary: 0 error(s)
[MARKDOWNLINT CHECK] Ended: 22:47:12 (2s) - PASSED


========================================
[ERROR SUMMARY]
========================================
Tool           Status   Errors   Duration
----           ------   ------   --------
Ruff           PASSED   0        0s
Format         PASSED   0        1s
Markdownlint   PASSED   0        2s

Total Duration: 3s
✅ All lint checks passed!
```

**Key Features of New Format**:

- Section headers with `========` separators for each tool
- Start timestamp for each tool check
- Full tool output captured (not summarized)
- End timestamp with duration and status
- Summary table at the end with all results
- Error counts shown when failures occur

### Error Handling

- **If auto-fix fails**: Log issue, markdown errors are acceptable for CLAUDE.md/MEMORY.md
- **For CLAUDE.md/MEMORY.md markdown errors**: Ignore (local-only files, won't be committed)
- **For production file errors**: Must fix or abort

---

## STEP 5: Execute Phase 2 - Commit to Development

### Log Phase Header

```
================================================================================
[PHASE 2: COMMIT TO DEVELOPMENT]
Status: IN PROGRESS
Time: HH:MM:SS
```

### Actions

1. Run commit enhanced:
   - Shell (PRIMARY): `./scripts/git/commit_enhanced.sh "commit message"`
   - Batch (fallback): `scripts/git/batch/commit_enhanced.bat --non-interactive "commit message"`
   - Note: Flags auto-enabled when running through Claude Code (no TTY)

**Available flags for commit_enhanced**:

- `--non-interactive`: Skip all prompts, use sensible defaults
- `--skip-md-lint`: Skip markdown lint checks (Python lint always runs)
- `--interactive`: Force interactive mode even without TTY (for debugging)
- `--staged-only`: Use pre-staged files only, skip auto-staging (enables selective commits)

**Auto-Detection (v0.6.2+)**:

- When running without a TTY (Claude Code, CI/CD, pipes), both `--non-interactive` and `--skip-md-lint` are enabled automatically
- Use `--interactive` to force prompts in automated contexts
- Environment variables: `CLAUDE_GIT_NON_INTERACTIVE=1`, `CLAUDE_GIT_SKIP_MD_LINT=1`, `CLAUDE_GIT_STAGED_ONLY=1` for explicit control

**Combined usage**:

```bash
# Shell (PRIMARY) - Auto-detection makes flags optional
./scripts/git/commit_enhanced.sh "commit message"

# Shell with explicit flags (optional)
./scripts/git/commit_enhanced.sh --non-interactive --skip-md-lint "commit message"

# Batch (fallback) - Still requires explicit flags
scripts/git/batch/commit_enhanced.bat --non-interactive --skip-md-lint "commit message"
```

**Selective commits with --staged-only** (v0.8.1+):

By default, commit_enhanced auto-stages all changes with `git add .`. Use `--staged-only` to commit only pre-staged files:

```bash
# Example: Commit only feature implementation files
git add mcp_server/*.py search/config.py tests/unit/search/*.py
./scripts/git/commit_enhanced.sh --staged-only --non-interactive "feat: implement new feature"

# Example: Commit only documentation updates
git add README.md docs/*.md CHANGELOG.md
./scripts/git/commit_enhanced.sh --staged-only --non-interactive "docs: update documentation"

# Example: Environment variable control
export CLAUDE_GIT_STAGED_ONLY=1
git add specific_file.py
./scripts/git/commit_enhanced.sh "fix: specific bug fix"
```

**Use cases for --staged-only**:

- Splitting related changes into separate logical commits
- Committing only tested files while keeping work-in-progress unstaged
- Incremental commits during development (commit completed work, keep exploratory code unstaged)

**Automatic Exclusions** (applies to all git automation scripts):

All lint tools automatically exclude test fixtures and archived files:

- **Test Data**: `tests/test_data/` - Contains intentional lint errors for testing
- **Archives**: `_archive/` - Old code not subject to current standards

**Implementation details**:

```batch
# Ruff uses --extend-exclude (applies to both check and format)
ruff check . --extend-exclude "tests/test_data" --extend-exclude "_archive"
ruff format . --extend-exclude "tests/test_data" --extend-exclude "_archive"

# Markdownlint has no special flags needed
```

These exclusions are applied in:

- `scripts/git/commit_enhanced.bat` (2 locations)
- `scripts/git/check_lint.bat` (2 locations)
- `scripts/git/fix_lint.bat` (2 locations)

**Why this matters**: Test fixtures often contain intentional errors (e.g., B904 violations for testing error handling). These exclusions prevent false positives during validation.

2. Capture commit details from output
3. Get commit hash: `git log -1 --format="%H"`
4. Get commit stats: `git log -1 --stat`

### Log Output Template

```
[1/2] Staging files...
  Command: git add [all modified files]
  ✓ All files staged successfully

[2/2] Creating commit...
  Command: git commit -m "commit message"

  Pre-commit hooks executed:
    ✓ Privacy protection: No local-only files detected
    ✓ Code quality: Checks passed

  Commit Details:
    Hash: abc1234567890
    Author: username <email@example.com>
    Date: Day Mon DD HH:MM:SS YYYY -0400
    Branch: development

  Files changed: 4
    - New: 0 files
    - Modified: 4 files
    - Insertions: 250 lines
    - Deletions: 45 lines

  ✓ Commit created successfully

Phase 2 Summary:
  ✓ Files staged: 4/4
  ✓ Commit created: abc1234
  ✓ Pre-commit hooks: PASSED
  Status: COMPLETED
  Time: HH:MM:SS
```

---

## STEP 6: Execute Phase 3 - Push to Development Remote

### Log Phase Header

```
================================================================================
[PHASE 3: PUSH TO DEVELOPMENT REMOTE]
Status: IN PROGRESS
Time: HH:MM:SS
```

### Actions

1. Run: `git push origin development`
2. Capture push result
3. Get remote URL: `git remote get-url origin`

### Log Output Template

```
[1/1] Pushing to origin/development...
  Command: git push origin development

  Push Result:
    Remote: https://github.com/username/repo.git
    Branch: development
    Commits: old_hash..new_hash
    Status: SUCCESS

  ✓ Development branch pushed successfully

Phase 3 Summary:
  ✓ Remote updated: origin/development
  ✓ Commit new_hash now on GitHub
  Status: COMPLETED
  Time: HH:MM:SS
```

---

## STEP 7: Execute Phase 4 - Merge Development → Main

### Log Phase Header

```
================================================================================
[PHASE 4: MERGE DEVELOPMENT → MAIN]
Status: IN PROGRESS
Time: HH:MM:SS
```

### Actions

1. Create backup tag: `git tag pre-merge-backup-YYYYMMDD_HHMMSS`
2. Switch to main: `git checkout main`
3. Run merge:
   - Shell (PRIMARY): `./scripts/git/merge_with_validation.sh --non-interactive`
   - Batch (fallback): `scripts/git/batch/merge_with_validation.bat --non-interactive`
   - OR if script has issues: `git merge development --no-ff -m "Merge development into main - Description"`
4. Handle conflicts if any
5. Capture merge results

**Note**: `merge_with_validation` internally executes:

- `validate_branches` - Pre-merge validation (checks branch state, uncommitted changes)
- Creates backup tag automatically
- Handles modify/delete conflicts for excluded files
- Validates documentation against CI policy

### Conflict Handling

**Modify/Delete Conflicts** (expected for test files):

- Auto-resolve with `git rm tests/**`
- Log which files were removed
- These are development-only files excluded from main

**Content Conflicts** (unexpected):

- Log details
- Cannot auto-resolve
- Abort workflow and notify user

### Log Output Template

```
[1/4] Creating backup tag...
  Command: git tag pre-merge-backup-20251005_204530
  ✓ Backup tag created successfully

[2/4] Switching to main branch...
  Command: git checkout main
  ✓ Switched to main branch

[3/4] Executing merge...
  Command: git merge development --no-ff -m "Merge development into main - Non-interactive workflow"

  Merge Conflicts Detected:
    ⚠️  CONFLICT (modify/delete): tests/integration/test_glsl_without_embedder.py
    ⚠️  CONFLICT (modify/delete): tests/integration/test_mcp_functionality.py
    ⚠️  CONFLICT (modify/delete): tests/integration/test_token_efficiency_workflow.py
    ⚠️  CONFLICT (modify/delete): tests/unit/test_imports.py

  Conflict Resolution:
    Note: Test files are development-only (excluded from main branch)
    Action: Remove test files from merge
    Command: git rm [4 test files]
    ✓ Conflicts resolved successfully

[4/4] Completing merge commit...
  Command: git commit --no-edit

  Pre-commit hooks executed:
    ✓ Privacy protection: No local-only files detected
    ✓ Code quality: Checks passed

  Merge Commit Details:
    Hash: def5678901234
    Message: "Merge development into main - Non-interactive workflow"
    Date: Day Mon DD HH:MM:SS YYYY -0400
    Branch: main

  Files merged: 11 (production files only)
    - New: 3 files
    - Modified: 8 files

  Files excluded: 4 test files (development-only)
    - tests/integration/test_glsl_without_embedder.py
    - tests/integration/test_mcp_functionality.py
    - tests/integration/test_token_efficiency_workflow.py
    - tests/unit/test_imports.py

  ✓ Merge commit created successfully

Phase 4 Summary:
  ✓ Backup tag: pre-merge-backup-20251005_204530
  ✓ Conflicts resolved: 4 test files removed (expected)
  ✓ Merge commit: def5678
  ✓ Production files merged: 11/11
  Status: COMPLETED
  Time: HH:MM:SS
```

---

## STEP 8: Execute Phase 5 - Push Main to Remote

### Log Phase Header

```
================================================================================
[PHASE 5: PUSH MAIN TO REMOTE]
Status: IN PROGRESS
Time: HH:MM:SS
```

### Actions

1. Run: `git push origin main`
2. Capture push result

### Log Output Template

```
[1/1] Pushing to origin/main...
  Command: git push origin main

  Push Result:
    Remote: https://github.com/username/repo.git
    Branch: main
    Commits: old_hash..new_hash
    Status: SUCCESS

  ✓ Main branch pushed successfully

Phase 5 Summary:
  ✓ Remote updated: origin/main
  ✓ Commit new_hash now on GitHub
  Status: COMPLETED
  Time: HH:MM:SS
```

---

## STEP 9: Write Workflow Completion Summary

### Log Completion Section

```
================================================================================
                           WORKFLOW COMPLETION
================================================================================

Workflow Status: ✅ SUCCESS

Timeline:
  Phase 1 (Pre-commit validation): 20:30:00 - 20:30:20 (20s)
  Phase 2 (Commit to development): 20:30:20 - 20:31:50 (90s)
  Phase 3 (Push development): 20:31:50 - 20:32:20 (30s)
  Phase 4 (Merge to main): 20:32:20 - 20:43:28 (668s)
  Phase 5 (Push main): 20:43:28 - 20:43:50 (22s)

  Total Duration: 830 seconds (~13.8 minutes)

Final State:
  ✅ Development branch: Commit abc1234 pushed to GitHub
  ✅ Main branch: Merge commit def5678 pushed to GitHub
  ✅ All lint checks: PASSED
  ✅ All pre-commit hooks: PASSED
  ✅ Production files merged: 11/11
  ✅ Test files excluded: 4/4 (development-only)

Deliverables:
  - 3 bash scripts created (check_lint.sh, fix_lint.sh, validate_branches.sh)
  - logs/ directory established with gitignore protection
  - Documentation updated (CHANGELOG.md, GIT_WORKFLOW.md, README.md)
  - Configuration enhanced (pyproject.toml per-file-ignores)
  - Lint issues resolved (9 Python files auto-fixed)
  - Git Bash compatibility achieved

End Time: YYYY-MM-DD HH:MM:SS

================================================================================
                          VERIFICATION & LOGGING
================================================================================

Workflow Log: ✅ logs/workflow_commit_merge_20251005_204530.log
Analysis Report: 📊 logs/workflow_commit_merge_analysis_20251005_204530.md (optional)

Next Actions:
1. Verify GitHub repository state
2. Monitor CI/CD pipeline (if applicable)
3. Test bash scripts in Git Bash environment

================================================================================
```

---

## STEP 10: Report to User

### User Message Template

```
✅ Workflow completed successfully!

📋 Comprehensive log: logs/workflow_commit_merge_TIMESTAMP.log

Summary:
- Development: Committed abc1234, pushed to remote
- Main: Merged def5678, pushed to remote
- Total duration: ~13.8 minutes

You can verify with:
  git log --oneline -5
  git log --graph --oneline -10
```

---

## Error Handling Guidelines

### Lint Errors

**Auto-fixable (ruff check, ruff format)**:

- Run `fix_lint.bat`
- Log what was fixed
- Re-stage files
- Continue workflow

**Not auto-fixable (markdownlint)**:

- Check if errors are in CLAUDE.md or MEMORY.md
- If yes: Ignore (local-only files)
- If no (production files): Log error, ask user to fix manually

### Merge Conflicts

**Modify/Delete (tests/ directory)**:

- Expected behavior
- Auto-resolve with `git rm`
- Log resolution
- Continue workflow

**Content Conflicts (both modified)**:

- Unexpected
- Cannot auto-resolve
- Log conflict details
- Abort workflow
- Notify user to resolve manually

### Push Failures

**Remote rejected**:

- Log error
- Check if remote has new commits
- User must pull and retry

**Authentication failure**:

- Log error
- User must configure credentials
- Cannot continue automatically

### General Errors

For any error:

1. Log full error output
2. Provide context (which phase, which command)
3. Suggest resolution steps
4. Reference GIT_WORKFLOW.md for detailed troubleshooting
5. Mark workflow as FAILED in completion summary

### Rollback Procedures

If the workflow fails or issues are discovered after merge:

**Use rollback script:**

```bash
# Shell (PRIMARY)
./scripts/git/rollback_merge.sh

# Batch (fallback)
scripts/git/batch/rollback_merge.bat
```

This script:

- Resets main branch to pre-merge state using backup tag
- Preserves the backup tag for reference
- Returns to development branch
- Logs all rollback actions

**Manual rollback (if script unavailable):**

```bash
# Return to pre-merge state
git checkout main
git reset --hard pre-merge-backup-TIMESTAMP

# Return to development
git checkout development
```

**When to rollback:**

- CI/CD pipeline fails after push
- Unexpected behavior discovered in main
- Need to add more changes before merge

---

## Technical Implementation Notes

### Writing to Log File

**Method 1: Direct echo redirection**

```bash
echo "Message" >> logs/workflow_commit_merge_TIMESTAMP.log
```

**Method 2: Capture command output**

```bash
scripts/git/check_lint.bat 2>&1 | tee -a logs/workflow_commit_merge_TIMESTAMP.log
```

Note: `tee` not available in Windows cmd.exe, use PowerShell or capture then append

**Method 3: Store output, then log**

```bash
output=$(scripts/git/check_lint.bat 2>&1)
echo "$output"  # Display to user
echo "$output" >> logs/workflow_commit_merge_TIMESTAMP.log  # Log to file
```

### Timestamp Format

Use: `YYYYMMDD_HHMMSS`

**Generate timestamp**:

```bash
# Windows batch
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do set mydate=%%c%%a%%b
for /f "tokens=1-2 delims=/: " %%a in ('time /t') do set mytime=%%a%%b
set TIMESTAMP=%mydate%_%mytime%

# PowerShell
powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"

# Git Bash
date +%Y%m%d_%H%M%S
```

### Capturing Git Command Output

**Commit hash**:

```bash
git log -1 --format="%H"  # Full hash
git log -1 --format="%h"  # Short hash
```

**Commit stats**:

```bash
git log -1 --stat  # File changes with insertions/deletions
git log -1 --shortstat  # Summary only
```

**Commit range for push**:

```bash
git log origin/branch..branch --oneline  # Commits to be pushed
```

---

## Reference Documentation

- **Full workflow guide**: [GIT_WORKFLOW.md](./GIT_WORKFLOW.md)
- **Troubleshooting**: GIT_WORKFLOW.md § "🔧 Troubleshooting"
- **Script details**: GIT_WORKFLOW.md § "🚀 Workflow Scripts"
- **Conflict resolution**: GIT_WORKFLOW.md § "🔀 Understanding Modify/Delete Conflicts"
- **Log format example**: `logs/bash_scripts_commit_merge_20251005_185155.log`

---

## Success Criteria Checklist

- [ ] Log file created at start of workflow
- [ ] Header written with file summary
- [ ] All 5 phases executed in order
- [ ] Each phase logged with full output
- [ ] Timeline tracked (start/end times per phase)
- [ ] Errors handled and logged appropriately
- [ ] Final status summary written
- [ ] User notified with log location
- [ ] Log matches format: 200-300 lines, comprehensive audit trail

---

## Example Complete Workflow

```
User: "Commit and push to development, then merge to main following AUTOMATED_GIT_WORKFLOW.md"

Claude Actions:
1. Create logs/workflow_commit_merge_20251005_204530.log
2. Write header with file changes summary
3. Check for lock files → remove if stale → verify repository state → log
4. Analyze staged changes → count files → log descriptions
5. Execute Phase 1: check_lint.sh (or .bat fallback) → capture output → log
6. Execute Phase 2: commit_enhanced.sh --non-interactive (or .bat fallback) → capture → log
7. Execute Phase 3: git push origin development → capture → log
8. Execute Phase 4: merge_with_validation.sh (or .bat fallback) → handle conflicts → log
9. Execute Phase 5: git push origin main → capture → log
10. Write timeline summary
11. Write final status: SUCCESS
12. Notify user: "✅ Workflow completed! 📋 logs/workflow_commit_merge_20251005_204530.log"
```

**Best Practice**: Chain git commands in single Bash calls to prevent lock file issues:

```bash
rm -f .git/index.lock && git add . && git commit -m "message" && git push origin development
```

**Result**: Comprehensive 250+ line log file matching `bash_scripts_commit_merge_20251005_185155.log` format

---

**End of Automated Workflow Guide**
