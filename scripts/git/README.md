# Git Automation Scripts

**Purpose**: Automated git workflows for safe commits, merges, and releases with comprehensive validation and logging.

**Location**: `scripts/git/`

**Note**: Shell scripts (.sh) are **PRIMARY** for automated Git workflows. Batch scripts (.bat) are in `scripts/git/batch/` for manual Windows CMD execution.

---

## Directory Structure

```
scripts/git/
├── *.sh                 # Shell scripts (PRIMARY - for automation)
│   ├── _common.sh
│   ├── check_lint.sh
│   ├── commit_enhanced.sh
│   └── ...
└── batch/               # Batch scripts (for manual Windows CMD)
    ├── _common.bat
    ├── check_lint.bat
    ├── commit_enhanced.bat
    └── ...
```

**When to use each:**
- **Shell scripts (.sh)**: Automated workflows, Git Bash, Linux, macOS
- **Batch scripts (.bat)**: Manual execution in Windows CMD only

---

## Quick Start

### Most Common Operations

```bash
# Check lint status (Python + Markdown)
./scripts/git/check_lint.sh              # Git Bash/Linux/macOS (PRIMARY)
scripts\git\batch\check_lint.bat         # Windows CMD (manual)

# Auto-fix lint issues
./scripts/git/fix_lint.sh                # Git Bash/Linux/macOS (PRIMARY)
scripts\git\batch\fix_lint.bat           # Windows CMD (manual)

# Safe commit with validation
./scripts/git/commit_enhanced.sh "feat: Your message"     # Git Bash/Linux/macOS (PRIMARY)
scripts\git\batch\commit_enhanced.bat "feat: Your message"      # Windows CMD (manual)

# Validate branches before merge
./scripts/git/validate_branches.sh       # Git Bash/Linux/macOS (PRIMARY)
scripts\git\batch\validate_branches.bat  # Windows CMD (manual)

# Safe merge development → main
./scripts/git/merge_with_validation.sh   # Git Bash/Linux/macOS (PRIMARY)
scripts\git\batch\merge_with_validation.bat    # Windows CMD (manual)
```

---

## Script Index

| Script | Shell (.sh) | Batch (.bat) | Purpose |
|--------|-------------|--------------|---------|
| **Core Workflows** |
| `commit_enhanced` | `scripts/git/` | `scripts/git/batch/` | Enhanced commit workflow with validation, logging, and lint checks |
| `merge_with_validation` | `scripts/git/` | `scripts/git/batch/` | Safe merge dev→main with auto-conflict resolution and backup |
| `validate_branches` | `scripts/git/` | `scripts/git/batch/` | Pre-merge branch validation (uncommitted changes, branch state) |
| **Quality & Testing** |
| `check_lint` | `scripts/git/` | `scripts/git/batch/` | Run lint validation (ruff, black, isort, markdownlint) |
| `check_shell` | `scripts/lint/` | *(none)* | Shell script validation using ShellCheck |
| `fix_lint` | `scripts/git/` | `scripts/git/batch/` | Auto-fix lint issues with verification |
| **Specialized Operations** |
| `cherry_pick_commits` | `scripts/git/` | `scripts/git/batch/` | Cherry-pick specific commits from development to main |
| `merge_docs` | `scripts/git/` | `scripts/git/batch/` | Documentation-only merge from development to main |
| `rollback_merge` | `scripts/git/` | `scripts/git/batch/` | Emergency rollback for merge operations |
| **Setup & Utilities** |
| `install_hooks` | `scripts/git/` | `scripts/git/batch/` | Install pre-commit hooks from .githooks/ |
| `_common` | `scripts/git/` | `scripts/git/batch/` | Shared utility functions (timestamps, logging, exclusions) |

---

## Detailed Descriptions

### commit_enhanced

**Purpose**: Enhanced commit workflow with comprehensive validations and mandatory logging.

**Features**:

- Stages changes with validation
- Runs lint checks (Python + Markdown)
- Auto-fix option for lint errors
- Protects local-only files (CLAUDE.md, MEMORY.md, _archive)
- Branch-specific validations (main vs development)
- Creates timestamped log files
- Conventional commit format validation

**Usage**:

```bash
# Git Bash/Linux/macOS (PRIMARY)
./scripts/git/commit_enhanced.sh "feat: Add semantic search"
./scripts/git/commit_enhanced.sh --non-interactive "fix: Bug fix"
./scripts/git/commit_enhanced.sh --skip-md-lint "docs: Update README"

# Windows CMD (manual execution)
scripts\git\batch\commit_enhanced.bat "feat: Add semantic search"
scripts\git\batch\commit_enhanced.bat --non-interactive "fix: Bug fix"
scripts\git\batch\commit_enhanced.bat --skip-md-lint "docs: Update README"
```

**Flags**:

- `--non-interactive`: Skip all prompts, use sensible defaults
- `--skip-md-lint`: Skip markdown lint checks (Python lint always runs)

**Output**: `logs/commit_enhanced_YYYYMMDD_HHMMSS.log`

---

### merge_with_validation

**Purpose**: Safe merge from development to main with validation and automatic conflict resolution.

**Features**:

- Pre-merge validation (branch state, uncommitted changes)
- Creates automatic backup tag (`pre-merge-backup-TIMESTAMP`)
- Auto-resolves modify/delete conflicts for test files
- Validates documentation against CI policy
- Comprehensive logging with analysis report
- Rollback instructions on failure

**Usage**:

```bash
# Git Bash/Linux/macOS (PRIMARY)
./scripts/git/merge_with_validation.sh
./scripts/git/merge_with_validation.sh --non-interactive

# Windows CMD (manual execution)
scripts\git\batch\merge_with_validation.bat
scripts\git\batch\merge_with_validation.bat --non-interactive
```

**Flags**:

- `--non-interactive`: Skip all prompts (for automation)

**Output**:

- `logs/merge_with_validation_YYYYMMDD_HHMMSS.log`
- `logs/merge_with_validation_analysis_YYYYMMDD_HHMMSS.md`
- Backup tag: `pre-merge-backup-YYYYMMDD_HHMMSS`

**Conflict Handling**:

- Modify/delete conflicts (test files): Auto-resolved (removed from main)
- Content conflicts: Manual resolution required

---

### validate_branches

**Purpose**: Pre-merge branch validation to ensure clean state.

**Features**:

- Verifies current branch is development or main
- Checks for uncommitted changes
- Analyzes branch relationship (commits ahead/behind)
- Warns if development has no new commits vs main
- Warns if main is ahead of development

**Usage**:

```bash
# Git Bash/Linux/macOS (PRIMARY)
./scripts/git/validate_branches.sh

# Windows CMD (manual execution)
scripts\git\batch\validate_branches.bat
```

**Exit codes**:

- 0: Validation passed
- 1: Validation failed (uncommitted changes, wrong branch, etc.)

---

### check_lint

**Purpose**: Run comprehensive lint validation on codebase.

**Features**:

- Runs 4 lint tools in sequence
- Automatic exclusions for test data and archives
- Cross-platform support (Windows/Linux/macOS)
- Clear pass/fail reporting

**Lint Tools**:

1. **ruff** - Python linter (fast, comprehensive)
2. **black** - Python formatter (check mode)
3. **isort** - Python import sorter (check mode)
4. **markdownlint** - Markdown linter

**Usage**:

```bash
# Git Bash/Linux/macOS (PRIMARY)
./scripts/git/check_lint.sh

# Windows CMD (manual execution)
scripts\git\batch\check_lint.bat
```

**Exclusions** (automatic):

- `tests/test_data/` - Intentional lint errors for testing
- `_archive/` - Historical code not subject to current standards
- `node_modules/`, `.venv/`, `benchmark_results/`, `logs/`

**Exit codes**:

- 0: All checks passed
- 1: One or more checks failed

---

### check_shell

**Purpose**: Static analysis of bash scripts using ShellCheck.

**Features**:

- Validates all `.sh` files in `scripts/` directory
- Detects common bash pitfalls and anti-patterns
- Project-local ShellCheck installation (`tools/bin/shellcheck.exe`)
- Graceful degradation if ShellCheck not installed
- Excludes SC2154 (sourced variable warnings from `_common.sh`)

**Checks For**:

- Unquoted variables that could cause word-splitting
- Unsafe `cd` commands without error handling
- POSIX compliance issues
- Command substitution style
- Unused variables
- Syntax errors

**Usage**:

```bash
# Git Bash/Linux/macOS
./scripts/lint/check_shell.sh
```

**Note**: No batch equivalent - Shell scripts only run in Git Bash/Unix environments.

**Integration**: See `BASH_STYLE_GUIDE.md` Section 5.4 for ShellCheck error codes and fixes.

**Exit codes**:

- 0: All scripts passed ShellCheck
- 1: One or more scripts have issues

---

### fix_lint

**Purpose**: Auto-fix lint issues with verification.

**Features**:

- Runs auto-fix for all lint tools
- Checks return values before reporting success
- Runs final verification after fixes
- Cross-platform support

**Usage**:

```bash
# Git Bash/Linux/macOS (PRIMARY)
./scripts/git/fix_lint.sh
./scripts/git/fix_lint.sh --non-interactive

# Windows CMD (manual execution)
scripts\git\batch\fix_lint.bat
scripts\git\batch\fix_lint.bat --non-interactive
```

**Flags**:

- `--non-interactive`: Skip all prompts

**Process**:

1. Run ruff --fix
2. Run black (reformat)
3. Run isort (sort imports)
4. Run markdownlint --fix
5. Run check_lint.sh for verification

---

### cherry_pick_commits

**Purpose**: Cherry-pick specific commits from development to main.

**Features**:

- Interactive commit selection
- Pre-validation (branch state)
- Automatic backup tag creation
- Warns about development-only files
- Shows commit details before applying

**Usage**:

```bash
# Git Bash/Linux/macOS (PRIMARY)
./scripts/git/cherry_pick_commits.sh

# Windows CMD (manual execution)
scripts\git\batch\cherry_pick_commits.bat
```

**Workflow**:

1. Run validation
2. Switch to main branch
3. Show recent development commits (last 20)
4. Prompt for commit hash
5. Show commit details
6. Warn if commit modifies test files
7. Create backup tag
8. Cherry-pick the commit

**Output**: Backup tag `pre-cherry-pick-YYYYMMDD_HHMMSS`

---

### merge_docs

**Purpose**: Documentation-only merge from development to main.

**Features**:

- Merges ONLY docs/ directory changes
- Warns about non-documentation changes
- Creates backup tag
- Interactive confirmation

**Usage**:

```bash
# Git Bash/Linux/macOS (PRIMARY)
./scripts/git/merge_docs.sh

# Windows CMD (manual execution)
scripts\git\batch\merge_docs.bat
```

**Process**:

1. Check for docs/ changes
2. Warn if non-docs changes exist
3. Create backup tag
4. Checkout docs/ from development
5. Commit with descriptive message

**Output**: Backup tag `pre-docs-merge-YYYYMMDD_HHMMSS`

---

### rollback_merge

**Purpose**: Emergency rollback for merge operations.

**Features**:

- Interactive rollback target selection
- Shows available backup tags
- Warns before destructive operations
- Returns to development branch after rollback

**Usage**:

```bash
# Git Bash/Linux/macOS (PRIMARY)
./scripts/git/rollback_merge.sh

# Windows CMD (manual execution)
scripts\git\batch\rollback_merge.bat
```

**Rollback Options**:

1. Latest pre-merge backup tag (recommended)
2. Previous commit (HEAD~1)
3. Specific commit hash
4. Cancel

**Safety**:

- Requires confirmation (type "ROLLBACK")
- Checks for uncommitted changes
- Only runs from main branch

---

### install_hooks

**Purpose**: Install pre-commit hooks from .githooks/ directory.

**Features**:

- Copies hook files to .git/hooks/
- Sets executable permissions
- Validates hook files exist
- Creates logs

**Usage**:

```bash
# Git Bash/Linux/macOS (PRIMARY)
./scripts/git/install_hooks.sh

# Windows CMD (manual execution)
scripts\git\batch\install_hooks.bat
```

**Installed Hooks**:

- **pre-commit**: File validation + code quality checks
  - Prevents committing local-only files (CLAUDE.md, MEMORY.md)
  - Validates documentation files
  - Checks code quality (Python)
  - Offers auto-fix for lint errors

**Output**: `logs/install_hooks_YYYYMMDD_HHMMSS.log`

---

### _common

**Purpose**: Shared utility functions for all git automation scripts.

**Features**:

- Platform-independent timestamp generation
- Centralized logging infrastructure
- Consistent lint exclusion patterns
- Cross-platform Python path detection (Windows/Linux/macOS)

**Functions** (Batch):

- `GetTimestamp` - Sets TIMESTAMP variable
- `InitLogging` - Creates log files with timestamps
- `GetLintExclusions` - Sets exclusion patterns for lint tools

**Functions** (Shell):

- `get_timestamp()` - Sets timestamp variable
- `init_logging()` - Creates log files with timestamps
- `get_lint_exclusions()` - Sets exclusion patterns for lint tools
- `get_python_path()` - Detects Windows/Linux Python paths
- `log_message()` - Logs to both console and file
- `log_section_start()` - Start section with timestamp header
- `log_section_end()` - End section with duration and status
- `run_tool_with_logging()` - Run lint tool with output capture and timing
- `run_git_with_logging()` - Run git command with output capture and timing
- `log_summary_table()` - Format structured summary table

**Usage**: Sourced by other scripts (not run directly)

```bash
# Batch
call "%~dp0_common.bat" :GetTimestamp
call "%~dp0_common.bat" :InitLogging "script_name"

# Shell
source "${SCRIPT_DIR}/_common.sh"
get_timestamp
init_logging "script_name"
```

---

## Platform Notes

### Shell Scripts (.sh files) - PRIMARY

**Location**: `scripts/git/*.sh`

**Requirements**:

- Bash shell (Git Bash on Windows, native on Linux/macOS)
- Execute permissions (scripts are already marked executable)

**Usage**:

```bash
# Git Bash / Linux / macOS (PRIMARY for automation)
./scripts/git/commit_enhanced.sh "feat: Your message"
./scripts/git/merge_with_validation.sh
./scripts/git/check_lint.sh
```

**Cross-Platform Support**: All .sh scripts detect platform automatically:

- Windows: Uses `.venv/Scripts/*.exe`
- Linux/macOS: Uses `.venv/bin/*`

**When to use**: Automated workflows, continuous integration, Claude Code automation

---

### Batch Scripts (.bat files) - Manual Execution Only

**Location**: `scripts/git/batch/*.bat`

**Requirements**:

- Windows Command Prompt (cmd.exe)
- Git for Windows

**⚠️ CRITICAL**: Batch scripts MUST be run from Windows CMD, NOT Git Bash.

**Why**: Git Bash cannot properly execute Windows batch scripts. Variable scoping fails, causing errors.

**Error if run from Git Bash**:

```
[ERROR] This script must be run from Windows CMD, not Git Bash
Detected environment: MSYSTEM=MSYS
```

**Usage**:

```cmd
REM Windows CMD only (manual execution)
scripts\git\batch\commit_enhanced.bat "feat: Your message"
scripts\git\batch\merge_with_validation.bat
scripts\git\batch\check_lint.bat
```

**When to use**: Manual operations in Windows CMD environment, fallback when shell scripts have issues

---

## Logging

All scripts create comprehensive timestamped log files in `logs/` directory with structured section headers, timing information, and error details.

**Format**: `logs/{script_name}_{timestamp}.log`

**Example**:

```
logs/commit_enhanced_20251207_203045.log
logs/merge_with_validation_20251207_204530.log
logs/merge_with_validation_analysis_20251207_204530.md
logs/validate_branches_20251220_160000.log
logs/check_lint_20251220_160100.log
logs/fix_lint_20251220_160200.log
```

**Log Structure** (Enhanced as of 2025-12-20):

```
=========================================
Script Name Log
=========================================
Start Time: 2025-12-20 16:00:00
Working Directory: /path/to/project

========================================
[SECTION NAME] Started: 16:00:01
========================================
Command: git checkout main
Switched to branch 'main'
[SECTION NAME] Ended: 16:00:02 (1s) - PASSED

========================================
[ERROR SUMMARY]
========================================
Tool          Status   Errors   Duration
----          ------   ------   --------
Ruff          PASSED   0        1s
Black         PASSED   0        2s
Isort         PASSED   0        1s
Markdownlint  PASSED   0        3s

Total: 0 errors

End Time: 2025-12-20 16:00:10
Total Duration: 10s
```

**Features**:

- **Section Headers**: Each operation (git command, lint tool) has timestamped start/end markers
- **Duration Tracking**: Every section shows execution time in seconds
- **Error Capture**: Full output from failed commands (file paths, line numbers, error codes)
- **Summary Tables**: Structured overview of all operations with pass/fail status
- **Agent-Parseable**: Consistent format enables automated log analysis

**What Gets Logged**:

- All git command output (checkout, merge, commit, reset, cherry-pick)
- Complete lint tool errors (ruff, black, isort, markdownlint)
- File paths and line numbers for failures
- Validation results with specific error messages
- Timestamps for every major operation
- Total execution time

**Retention**: Logs are preserved for audit trail. Clean up manually if needed.

---

## Related Documentation

- **Git Workflow Guide**: `docs/GIT_WORKFLOW.md`
- **Automated Git Workflow**: `docs/AUTOMATED_GIT_WORKFLOW.md`
- **Bash Style Guide**: `BASH_STYLE_GUIDE.md`
- **Batch Style Guide**: `BATCH_STYLE_GUIDE.md`
- **Shell Style Guide**: `SHELL_STYLE_GUIDE.md`

---

## Troubleshooting

### Lint Errors

**Problem**: Lint checks fail in commit_enhanced or check_lint.

**Solution**:

1. Run `./scripts/git/fix_lint.sh` (or `.bat` on Windows)
2. Review auto-fixes: `git diff`
3. Revert if needed: `git checkout .`
4. Re-run lint check

### Merge Conflicts

**Problem**: Merge fails with conflicts.

**Solution**:

- **Modify/delete conflicts** (test files): Auto-resolved by merge_with_validation
- **Content conflicts**: Resolve manually, then `git add` and `git commit`
- **Abort merge**: `git merge --abort` and checkout development

### Rollback Needed

**Problem**: Need to undo a merge.

**Solution**:

1. Run `./scripts/git/rollback_merge.sh` (or `.bat`)
2. Select rollback option (usually option 1: latest backup tag)
3. Confirm by typing "ROLLBACK"
4. Force push to remote: `git push origin main --force-with-lease`

### Lock File Issues

**Problem**: `.git/index.lock` exists.

**Solution**:

```bash
# Check for git processes
tasklist | findstr git  # Windows
ps aux | grep git       # Linux/macOS

# If none running, remove lock file
rm -f .git/index.lock
```

---

## Best Practices

1. **Always run lint checks** before committing
2. **Use commit_enhanced** instead of raw `git commit` for safety
3. **Create backup tags** before risky operations (automatic in merge scripts)
4. **Review logs** after automated workflows for audit trail
5. **Test on development** branch before merging to main
6. **Use --non-interactive** flag for CI/CD automation

---

## Version History

**Latest**: 2025-12-20

- **Major Logging Enhancement**: Comprehensive structured logging across all scripts
  - Added section headers with timestamps for all operations
  - Implemented duration tracking (per-tool and total execution time)
  - Added structured summary tables with error counts
  - Full git command output capture (checkout, merge, commit, reset, cherry-pick)
  - Enhanced lint tool error reporting (file:line:col:error format)
  - Added 5 new logging functions to _common.sh:
    - `log_section_start()` - Start section with timestamp
    - `log_section_end()` - End section with duration and status
    - `run_tool_with_logging()` - Lint tool execution with capture
    - `run_git_with_logging()` - Git command execution with capture
    - `log_summary_table()` - Format summary tables
- Scripts enhanced: validate_branches, merge_with_validation, rollback_merge, cherry_pick_commits, merge_docs, install_hooks
- All logs now agent-parseable for automated analysis

**Previous**: 2025-12-07

- Added cross-platform support (.sh scripts for Git Bash/Linux/macOS)
- Improved error handling and return value checks
- Added main functions to all shell scripts
- Fixed pipe-to-while loop bugs
- Enhanced lint validation with arrays

**Previous**: 2025-12-03

- Initial release with .bat scripts for Windows
- Core workflows: commit, merge, validate
- Lint automation: check, fix
- Specialized: cherry-pick, docs-only merge, rollback

---

**Maintained by**: Claude Code (Automated Git Workflows)
**Issues**: Report to project maintainer
