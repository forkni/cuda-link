# Automated Git Workflow

Execute automated commit→push→merge→push workflow using project scripts (token-efficient).

**Invocation**: `/auto-git-workflow`

**Key**: Suppress all output except errors. Show brief summary at end.

---

## Environment Detection

**Check execution environment first**:

```bash
echo $OSTYPE
# Git Bash / WSL: "msys" or "linux-gnu"
# macOS: "darwin"
# If variable empty: Windows cmd.exe
```

**Use appropriate workflow**:

- Git Bash / Linux / macOS → **Section A** (uses .sh scripts + direct git)
- Windows cmd.exe → **Section B** (uses .bat scripts)

**Why this matters**: Git Bash cannot execute .bat files directly. The `cmd.exe /c` wrapper fails silently (see GIT_WORKFLOW.md lines 1054-1117).

---

## Section A: Git Bash / Linux / macOS (Primary)

**Token-efficient execution**: All commands suppress output unless errors occur.

### ⚠️ CRITICAL - Bash Tool Compatibility

Due to Claude Code's Bash tool command parsing limitations:

- **Execute each numbered step as a SEPARATE Bash call**
- **DO NOT combine steps** with `&&` or `;` operators
- **DO NOT use complex patterns** like `VAR=$(cmd) && other_cmd`
- **Check exit codes** between steps
- **Store output before using** it in subsequent commands

Each step below must be a single, simple Bash tool invocation.

### Execution Pattern

For each phase:

1. Read the step number
2. Execute ONLY that step in one Bash call
3. Check the result/exit code
4. Decide whether to continue based on result
5. Move to next step

**Do NOT attempt to combine multiple steps into one Bash call.**

---

### Phase 1: Pre-commit Validation

Execute these commands in sequence (one Bash call per step):

**Step 1.1: Switch to development branch**

```bash
git checkout development >/dev/null 2>&1
```

**Step 1.2: Check for changes**

```bash
git diff --quiet && git diff --cached --quiet
```

- If exit code 0: No changes, display "⚠ No changes to commit", stop workflow
- If exit code 1: Changes exist, continue to step 1.3

**Step 1.3: Run lint check (suppress output)**

```bash
./scripts/git/check_lint.sh >/dev/null 2>&1
```

- If exit code 0: Lint passed, skip to Phase 2
- If exit code ≠ 0: Continue to step 1.4

**Step 1.4: Auto-fix lint issues**

```bash
./scripts/git/fix_lint.sh
```

**Step 1.5: Re-verify lint (show output this time)**

```bash
./scripts/git/check_lint.sh
```

- Ignore markdown errors in CLAUDE.md/MEMORY.md (local-only files, won't be committed)
- If still fails: Stop workflow
- If passes: Continue to Phase 2

---

### Phase 2: Commit to Development

Execute these commands in sequence (one Bash call per step):

**Step 2.1: Stage all changes**

```bash
git add .
```

**Step 2.2: Check for local-only files**

```bash
git diff --cached --name-only | grep -E "(CLAUDE\.md|MEMORY\.md|_archive|benchmark_results)"
```

- If found (exit code 0):
  - Display error: "✗ ERROR: Local-only files are staged!"
  - Show files found
  - Display: "Remove with: git reset HEAD CLAUDE.md MEMORY.md _archive/ benchmark_results/"
  - Stop workflow
- If not found (exit code 1): Continue to step 2.3

**Step 2.3: Get current branch**

```bash
git branch --show-current
```

- Store result for step 2.4

**Step 2.4: If on main, check for test files**

```bash
git diff --cached --name-only | grep "^tests/"
```

- Only check if step 2.3 returned "main"
- If found on main (exit code 0): Display "✗ ERROR: Test files staged on main branch", stop workflow
- If not found (exit code 1) or not on main: Continue to step 2.5

**Step 2.5: Create commit**

```bash
git commit -m "feat: Your descriptive commit message" >/dev/null 2>&1
```

- Replace message with appropriate conventional commit format (feat:, fix:, docs:, chore:, test:)
- If fails: Show error by running without output suppression, then stop workflow

**Step 2.6: Capture commit info for final report**

```bash
git log -1 --format="%h %s"
```

- Store result for final report (hash and message)

---

### Phase 3: Push Development

Execute these commands in sequence (one Bash call per step):

**Step 3.1: Push to development (suppress output)**

```bash
git push origin development >/dev/null 2>&1
```

- If exit code 0: Continue to Phase 4
- If exit code ≠ 0: Continue to step 3.2

**Step 3.2: If push failed, show error**

```bash
git push origin development
```

- Display full error output
- Stop workflow

---

### Phase 4: Merge to Main

Execute these commands in sequence (one Bash call per step):

**Step 4.1: Switch to main**

```bash
git checkout main >/dev/null 2>&1
```

**Step 4.2: Attempt merge (suppress output)**

```bash
git merge development --no-ff -m "Merge development into main" >/dev/null 2>&1
```

- If exit code 0: Clean merge, skip to step 4.7
- If exit code ≠ 0: Conflicts detected, continue to step 4.3

**Step 4.3: Check for modify/delete conflicts (expected)**

```bash
git status --short | grep "^DU "
```

- If found (exit code 0): Modify/delete conflicts (EXPECTED), continue to step 4.4
- If not found (exit code 1): Continue to step 4.5

**Step 4.4: Auto-resolve modify/delete conflicts**

```bash
git status --short | grep "^DU " | awk '{print $2}' | xargs -r git rm
```

- Display: "⚠ Resolving expected test file conflicts..."
- Then continue to step 4.6

**Step 4.5: Check for content conflicts (unexpected)**

```bash
git status --short | grep "^UU "
```

- If found (exit code 0):
  - Display: "✗ Content conflicts require manual resolution:"
  - Show conflicted files
  - Display: "Abort with: git merge --abort && git checkout development"
  - Stop workflow
- If not found: Merge issue unclear, stop workflow

**Step 4.6: Complete merge commit**

```bash
git commit --no-edit >/dev/null 2>&1
```

- If fails: Display "✗ Failed to complete merge", stop workflow
- If succeeds: Continue to step 4.7

**Step 4.7: Capture merge commit hash**

```bash
git log -1 --format="%h"
```

- Store result for final report

---

### Phase 5: Push Main

Execute these commands in sequence (one Bash call per step):

**Step 5.1: Push to main (suppress output)**

```bash
git push origin main >/dev/null 2>&1
```

- If exit code 0: Continue to step 5.3
- If exit code ≠ 0: Continue to step 5.2

**Step 5.2: If push failed, show error**

```bash
git push origin main
```

- Display full error output
- Stop workflow

**Step 5.3: Return to development**

```bash
git checkout development >/dev/null 2>&1
```

---

### Final Report

Execute these commands in sequence (one Bash call per step):

**Step 6.1: Get development commit info**

```bash
git log development -1 --format="%h %s"
```

- Store hash and message

**Step 6.2: Get main commit hash**

```bash
git log main -1 --format="%h"
```

- Store hash

**Step 6.3: Display summary**

Use stored values from steps 2.6, 4.7, 6.1, and 6.2:

```
✅ Workflow complete

Development: [hash] "[message]"
Main: [hash] merged & pushed
```

---

## Section B: Windows cmd.exe (Alternative)

**Environment**: Windows Command Prompt only
**Scripts**: Use .bat files with --non-interactive flag (already token-efficient)

### Phase 1: Pre-commit Validation

```batch
git checkout development
scripts\git\check_lint.bat
```

If lint errors:

```batch
scripts\git\fix_lint.bat
```

### Phase 2: Commit to Development

```batch
scripts\git\commit_enhanced.bat --non-interactive "feat: Descriptive commit message"
```

**Note**: Script automatically:

- Stages all changes
- Validates local-only files
- Checks branch-specific rules
- Suppresses prompts

### Phase 3: Push Development

```batch
git push origin development
```

### Phase 4: Merge to Main

```batch
scripts\git\merge_with_validation.bat --non-interactive
```

**Note**: Script automatically:

- Runs pre-merge validation
- Creates backup tag
- Handles modify/delete conflicts
- Validates documentation CI policy

### Phase 5: Push Main

```batch
git push origin main
```

### Final Report

Show brief summary:

```
✅ Workflow complete

Development: [hash] pushed
Main: [merge-hash] merged & pushed
```

---

## Error Handling

### Lint Failures

**When**: `check_lint.sh` exits with error

**Action**:

```bash
./scripts/git/fix_lint.sh
./scripts/git/check_lint.sh
```

**Ignore**: Markdown errors in CLAUDE.md/MEMORY.md (local-only files)

**Reference**: GIT_WORKFLOW.md lines 1972-2216

### Batch Script Fails in Git Bash

**Symptom**: `.bat` file doesn't execute or fails silently

**Cause**: Git Bash cannot run Windows batch files

**Solution**: Use Section A (Git Bash workflow) instead

**Reference**: GIT_WORKFLOW.md lines 1054-1117

### Local-Only Files Staged

**Symptom**: CLAUDE.md, MEMORY.md, _archive/, or benchmark_results/ are staged

**Solution**:

```bash
git reset HEAD CLAUDE.md MEMORY.md _archive/ benchmark_results/
```

**Reference**: GIT_WORKFLOW.md lines 677-683

### Modify/Delete Conflicts

**Symptom**: Test files show "DU" status during merge

**Status**: ✅ **EXPECTED** (see GIT_WORKFLOW.md lines 745-842)

**Resolution**: Already handled automatically in Phase 4

### Content Conflicts

**Symptom**: Files show "UU" status (both modified)

**Action**: Manual resolution required

```bash
# Edit files to resolve
git add <resolved-files>
git commit

# Or abort
git merge --abort
git checkout development
```

**Reference**: GIT_WORKFLOW.md lines 1288-1355

### Push Failures

**Possible causes**:

- Authentication issues
- Network problems
- Branch protection rules
- Remote has diverged

**Action**: Check error output, resolve manually

---

## Quick Reference

| Environment | Scripts Available | Workflow |
|-------------|-------------------|----------|
| Git Bash | .sh scripts + direct git | Section A |
| Linux | .sh scripts + direct git | Section A |
| macOS | .sh scripts + direct git | Section A |
| Windows cmd.exe | .bat scripts | Section B |

**Fallback**: If all scripts fail, use direct git commands (see GIT_WORKFLOW.md)

---

## Token Efficiency

**Output Suppression**:

- ✅ All successful commands: `>/dev/null 2>&1`
- ✅ Only show errors: Remove suppression on retry
- ✅ Final summary: 4 lines maximum

**Typical token savings**: 90% reduction vs full output (14 lines vs 170 lines)
