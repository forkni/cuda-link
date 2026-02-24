#!/usr/bin/env bash
# commit_enhanced.sh
# Enhanced commit workflow with comprehensive validations and mandatory logging
# Shell script port of commit_enhanced.bat for Git Bash compatibility
#
# Usage: ./commit_enhanced.sh [options] "commit message"
#
# Options:
#   --non-interactive  Skip all prompts, use sensible defaults (for automation)
#   --skip-md-lint     Skip markdown lint checks (Python lint always runs)
#   --interactive      Force interactive mode even without TTY (for debugging)
#   --staged-only      Use pre-staged files only, skip auto-staging
#   --no-venv          Use system-installed tools instead of .venv (ruff, markdownlint-cli2)
#
# Auto-detection:
#   When no TTY is detected (Claude Code, CI/CD, pipes), both --non-interactive
#   and --skip-md-lint are enabled automatically.
#
# Environment variables:
#   CLAUDE_GIT_NON_INTERACTIVE=1  Force non-interactive mode
#   CLAUDE_GIT_SKIP_MD_LINT=1     Force skip markdown lint
#   CLAUDE_GIT_STAGED_ONLY=1      Use pre-staged files only
#   CLAUDE_GIT_NO_VENV=1          Use system tools instead of .venv

# Note: NOT using 'set -e' to allow comprehensive error handling

# Change to project root (two levels up from scripts/git/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || {
  echo "ERROR: Cannot find project root" >&2
  exit 1
}

# Source shared utilities
source "$SCRIPT_DIR/_common.sh"

# ========================================
# Helper Functions
# ========================================

generate_analysis_report() {
  # Generate comprehensive analysis report
  cat > "$reportfile" << EOF
# Enhanced Commit Workflow Analysis Report

**Workflow**: Enhanced Commit
**Date**: $(date)
**Branch**: $current_branch
**Status**: [OK] SUCCESS

## Summary
Successfully committed changes with full validation and logging.

## Files Committed

$(git diff HEAD~1 --name-status 2>/dev/null)

## Commit Details

$(git log -1 --pretty=format:"- **Hash**: %H%n- **Message**: %s%n- **Author**: %an%n- **Date**: %ad%n" 2>/dev/null)

## Validations Passed

- [OK] No local-only files committed (CLAUDE.md, MEMORY.md, _archive)
- [OK] Branch-specific validations passed
- [OK] Code quality checks passed
- [OK] Conventional commit format validated

## Logs

- Execution log: \`$logfile\`
- Analysis report: \`$reportfile\`

EOF

  echo "End Time: $(date)" >> "$logfile"
  log_message "" "$logfile"
  log_message "======================================" "$logfile"
  log_message "[REPORT] Analysis Report: $reportfile" "$logfile"
  log_message "======================================" "$logfile"
}

unstage_local_only_files() {
  # Unstage local-only files that should never be committed
  # Mirrors .gitignore "Local-only content" section and development-only files
  local files_to_unstage=(
    # Core local-only files (.gitignore lines 234-247)
    "CLAUDE.md"
    "MEMORY.md"
    "GEMINI.md"
    "BASH_STYLE_GUIDE.md"
    "BATCH_STYLE_GUIDE.md"
    "SHELL_STYLE_GUIDE.md"
    "SLASH_COMMAND_TESTING_PROTOCOL.md"
    "clean_pycache.bat"

    # Local-only directories
    "_archive/"
    "benchmark_results/"
    "analysis/"
    "logs/"

    # Development tools (.gitignore lines 284-285)
    "tools/benchmark_models.py"
    "tools/summarize_audit.py"

    # Development-only tests (.gitignore lines 287-295)
    "tests/unit/test_evaluation.py"
    "tests/unit/test_token_efficiency.py"
    "tests/unit/test_tool_handlers.py"
    "tests/integration/test_complete_workflow.py"
    "tests/integration/test_mcp_functionality.py"
    "tests/integration/test_semantic_search.py"
    "tests/integration/test_token_efficiency_workflow.py"
    "tests/integration/test_graph_search.py"

    # Maintenance scripts (.gitignore lines 306-310)
    "scripts/cleanup_powershell_refs_apply.py"
    "scripts/cleanup_powershell_refs_preview.py"
    "scripts/batch/cleanup_powershell_references.bat"
    "scripts/batch/cleanup_powershell_references_DRYRUN.bat"

    # Local development test files (.gitignore lines 297-300)
    "tests/test_mmap_cleanup.py"
    "tests/toon_format_test_results.txt"
    "tests/toon_format_understanding_test.md"

    # Audit reports
    "audit_reports/"
  )

  for file in "${files_to_unstage[@]}"; do
    git reset HEAD "$file" 2>/dev/null
  done
}

# ========================================
# Main Function
# ========================================

main() {
  # ========================================
  # Parse Command Line Arguments
  # ========================================

  local non_interactive=0
  local skip_md_lint=0
  local staged_only=0
  local no_venv=0
  local skip_python_lint=0
  local commit_msg_param=""

  # Auto-detect non-interactive mode when no TTY (Claude Code, CI/CD, pipes)
  if [[ ! -t 0 ]]; then
    non_interactive=1
    skip_md_lint=1
  fi

  # Environment variable overrides (explicit enable)
  [[ "${CLAUDE_GIT_NON_INTERACTIVE:-0}" == "1" ]] && non_interactive=1
  [[ "${CLAUDE_GIT_SKIP_MD_LINT:-0}" == "1" ]] && skip_md_lint=1
  [[ "${CLAUDE_GIT_STAGED_ONLY:-0}" == "1" ]] && staged_only=1
  [[ "${CLAUDE_GIT_NO_VENV:-0}" == "1" ]] && no_venv=1

  # Parse flags and message
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --non-interactive)
        non_interactive=1
        shift
        ;;
      --skip-md-lint)
        skip_md_lint=1
        shift
        ;;
      --interactive)
        # Force interactive mode even without TTY (for debugging)
        non_interactive=0
        skip_md_lint=0
        shift
        ;;
      --staged-only)
        staged_only=1
        shift
        ;;
      --no-venv)
        no_venv=1
        shift
        ;;
      *)
        commit_msg_param="$1"
        shift
        ;;
    esac
  done

  # ========================================
  # Initialize Mandatory Logging
  # ========================================

  # Initialize logging using shared utility
  init_logging "commit_enhanced"

  # Validate logging initialization
  if [[ -z "$logfile" ]] || [[ -z "$reportfile" ]]; then
    echo "[ERROR] Failed to initialize logging" >&2
    exit 1
  fi

  # Initialize log file
  {
    echo "========================================="
    echo "Enhanced Commit Workflow Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Branch: $(git branch --show-current)"
    echo ""
  } > "$logfile"

  # Log initialization message
  log_message "=== Enhanced Commit Workflow ===" "$logfile"
  log_message "" "$logfile"
  log_message "[LOG] Workflow Log: $logfile" "$logfile"
  log_message "" "$logfile"

  # Get cross-platform Python paths
  if [[ ${no_venv} -eq 1 ]]; then
    # Use system-installed tools (must be on PATH)
    PYTHON_BIN=""
    echo "[--no-venv] Using system-installed tools (no .venv)"
    # Verify ruff is available on PATH
    if ! command -v ruff &>/dev/null; then
      echo "[!] WARNING: ruff not found on PATH - Python lint checks will be skipped"
      skip_python_lint=1
    fi
  else
    if ! get_python_path; then
      exit 1
    fi
  fi

  # [1/6] Get current branch
  local current_branch
  current_branch=$(git branch --show-current)
  echo "Current branch: $current_branch"
  echo ""

  # [2/6] Check for uncommitted changes
  echo "[2/6] Checking for changes..."

  # Check for unstaged changes
  git diff --quiet
  local has_unstaged=$?

  # Check for staged changes
  git diff --cached --quiet
  local has_staged=$?

  if [[ ${has_unstaged} -eq 0 ]] && [[ ${has_staged} -eq 0 ]]; then
    echo "[!] No changes to commit"
    echo "Working directory is clean"
    exit 0
  fi

  if [[ ${has_unstaged} -ne 0 ]]; then
    echo "Unstaged changes detected"
    echo ""
    echo "Unstaged files:"
    git diff --name-status
    echo ""

    if [[ ${staged_only} -eq 1 ]]; then
      echo "[--staged-only] Using pre-staged files only"
      echo "[!] Unstaged files will be ignored"
    elif [[ ${non_interactive} -eq 1 ]]; then
      echo "[Non-interactive mode] Auto-staging all changes..."
      git add .
      if [[ $? -ne 0 ]]; then
        echo "[X] Failed to stage files" >&2
        exit 1
      fi
      unstage_local_only_files
      echo "[OK] All changes staged"
    else
      read -p "Stage all changes? (yes/no): " stage_all
      if [[ "$stage_all" == "yes" ]]; then
        git add .
        if [[ $? -ne 0 ]]; then
          echo "[X] Failed to stage files" >&2
          exit 1
        fi
        unstage_local_only_files
        echo "[OK] All changes staged"
      else
        echo ""
        echo "Please stage changes manually:"
        echo "  git add <files>"
        echo "Then run this script again"
        exit 1
      fi
    fi
  else
    echo "[OK] All changes already staged"
  fi
  echo ""

  # [3/6] Validate staged files
  echo "[3/6] Validating staged files..."

  # Remove local-only files from staging (safety check)
  unstage_local_only_files

  # Check for local-only files
  local found_local_files=0

  if git diff --cached --name-only | grep -iq "^CLAUDE\.md$"; then
    echo "[X] ERROR: CLAUDE.md is staged (should be local-only)" >&2
    found_local_files=1
  fi

  if git diff --cached --name-only | grep -iq "^MEMORY\.md$"; then
    echo "[X] ERROR: MEMORY.md is staged (should be local-only)" >&2
    found_local_files=1
  fi

  if git diff --cached --name-only | grep -iq "_archive"; then
    echo "[X] ERROR: _archive/ is staged (should be local-only)" >&2
    found_local_files=1
  fi

  if [[ ${found_local_files} -eq 1 ]]; then
    echo "" >&2
    echo "Please remove these files from staging" >&2
    exit 1
  fi

  # Branch-specific validations
  if [[ "$current_branch" == "main" ]]; then
    echo "Validating main branch commit..."

    # Check for test files
    if git diff --cached --name-only | grep -q "^tests/"; then
      echo "[X] ERROR: Test files staged on main branch" >&2
      echo "Tests should only be on development branch" >&2
      echo "" >&2
      echo "Staged test files:" >&2
      git diff --cached --name-only | grep "^tests/" >&2
      echo "" >&2
      exit 1
    fi

    # Check for pytest.ini
    if git diff --cached --name-only | grep -q "pytest.ini"; then
      echo "[X] ERROR: pytest.ini staged on main branch" >&2
      echo "This file should only be on development branch" >&2
      exit 1
    fi

    # Check for development-only docs
    if git diff --cached --name-only | grep -q "docs/TESTING_GUIDE.md"; then
      echo "[X] ERROR: TESTING_GUIDE.md staged on main branch" >&2
      echo "This doc should only be on development branch" >&2
      exit 1
    fi

    echo "[OK] No development-only files detected"
  fi

  echo "[OK] Staged files validated"
  echo ""

  # [4/7] Code quality check
  echo "[4/7] Checking code quality..."
  log_section_start "PYTHON LINT CHECK" "$logfile"

  # Get shared lint exclusion patterns
  get_lint_exclusions

  # Set ruff command path based on venv mode
  local ruff_cmd
  if [[ -n "$PYTHON_BIN" ]]; then
    ruff_cmd="$ruff_cmd"
  else
    ruff_cmd="ruff"
  fi

  # Helper: Get staged Python files for --staged-only mode
  get_staged_python_files() {
    git diff --cached --name-only --diff-filter=ACMR -- '*.py' | tr '\n' ' '
  }

  # Convert exclusions to arrays for proper word splitting
  local -a ruff_check_args ruff_format_args

  # shellcheck disable=SC2206
  ruff_check_args=(${RUFF_CHECK_EXCLUDE})
  # shellcheck disable=SC2206
  ruff_format_args=(${RUFF_FORMAT_EXCLUDE})

  # Determine what to lint (entire directory or staged files only)
  local lint_target="."
  local staged_py_files=""

  if [[ ${staged_only} -eq 1 ]]; then
    staged_py_files=$(get_staged_python_files)
    if [[ -z "$staged_py_files" ]]; then
      echo "[--staged-only] No Python files staged - skipping Python lint check" | tee -a "$logfile"
      python_lint_error=0
    else
      lint_target="$staged_py_files"
      echo "[--staged-only] Checking only staged Python files:" | tee -a "$logfile"
      echo "  $staged_py_files" | tee -a "$logfile"
    fi
  fi

  # Always run Python lint checks (ruff check + ruff format) - capture output
  local python_lint_error=0
  local ruff_output format_output

  # Run lint checks on target (skip if no Python files staged in --staged-only mode, or if ruff unavailable)
  if [[ ${skip_python_lint} -eq 1 ]]; then
    echo "  [--no-venv] ruff not on PATH - skipping Python lint" | tee -a "$logfile"
  elif [[ ${staged_only} -eq 0 ]] || [[ -n "$staged_py_files" ]]; then
    # shellcheck disable=SC2086
    ruff_output=$("$ruff_cmd" check $lint_target "${ruff_check_args[@]}" 2>&1) || python_lint_error=1
    if [[ -n "$ruff_output" ]] && [[ "$ruff_output" != *"All checks passed"* ]]; then
      echo "[RUFF ERRORS]" | tee -a "$logfile"
      echo "$ruff_output" | tee -a "$logfile"
    fi

    # shellcheck disable=SC2086
    format_output=$("$ruff_cmd" format --check $lint_target "${ruff_format_args[@]}" 2>&1) || python_lint_error=1
    if [[ -n "$format_output" ]] && [[ "$format_output" == *"would reformat"* ]]; then
      echo "[FORMAT ERRORS]" | tee -a "$logfile"
      echo "$format_output" | tee -a "$logfile"
    fi
  fi

  log_section_end "PYTHON LINT CHECK" "$logfile" "$python_lint_error"

  # Check markdownlint unless --skip-md-lint is set
  local md_lint_error=0
  local md_output
  if [[ ${skip_md_lint} -eq 0 ]]; then
    log_section_start "MARKDOWN LINT CHECK" "$logfile"
    md_output=$(markdownlint-cli2 $MD_PATTERNS 2>&1) || md_lint_error=1
    if [[ -n "$md_output" ]]; then
      echo "[MARKDOWNLINT ERRORS]" | tee -a "$logfile"
      echo "$md_output" | tee -a "$logfile"
    fi
    log_section_end "MARKDOWN LINT CHECK" "$logfile" "$md_lint_error"
  else
    echo "  [--skip-md-lint] Skipping markdown lint checks" | tee -a "$logfile"
  fi

  # Handle lint errors
  if [[ ${python_lint_error} -eq 1 ]] && [[ ${skip_python_lint} -eq 0 ]]; then
    echo "[!] Python lint errors detected"
    echo ""

    if [[ ${non_interactive} -eq 1 ]]; then
      echo "[Non-interactive mode] Auto-fixing Python lint issues..." | tee -a "$logfile"
      log_section_start "AUTO-FIX PYTHON LINT" "$logfile"

      # Run auto-fix tools and capture output
      local ruff_fix_output format_fix_output

      # shellcheck disable=SC2086
      ruff_fix_output=$("$ruff_cmd" check --fix $lint_target "${ruff_check_args[@]}" 2>&1)
      if [[ -n "$ruff_fix_output" ]]; then
        echo "[RUFF FIXES]" | tee -a "$logfile"
        echo "$ruff_fix_output" | tee -a "$logfile"
      fi

      # shellcheck disable=SC2086
      format_fix_output=$("$ruff_cmd" format $lint_target "${ruff_format_args[@]}" 2>&1)
      if [[ -n "$format_fix_output" ]]; then
        echo "[FORMAT FIXES]" | tee -a "$logfile"
        echo "$format_fix_output" | tee -a "$logfile"
      fi

      log_section_end "AUTO-FIX PYTHON LINT" "$logfile" "0"

      echo ""
      if [[ ${staged_only} -eq 1 ]]; then
        echo "[--staged-only] Skipping auto-restaging (manual staging required)" | tee -a "$logfile"
      else
        echo "Restaging fixed files..." | tee -a "$logfile"
        git add .
        if [[ $? -ne 0 ]]; then
          echo "[X] Failed to stage fixed files" | tee -a "$logfile" >&2
          exit 1
        fi
        unstage_local_only_files
        echo "[OK] Fixed files staged" | tee -a "$logfile"
      fi

      # Re-check Python lint
      log_section_start "RE-CHECK PYTHON LINT" "$logfile"
      python_lint_error=0
      local recheck_output
      # shellcheck disable=SC2086
      recheck_output=$("$ruff_cmd" check $lint_target "${ruff_check_args[@]}" 2>&1) || python_lint_error=1
      if [[ -n "$recheck_output" ]]; then
        echo "$recheck_output" | tee -a "$logfile"
      fi
      log_section_end "RE-CHECK PYTHON LINT" "$logfile" "$python_lint_error"

      if [[ ${python_lint_error} -eq 1 ]]; then
        echo "[X] Python lint errors remain after auto-fix" | tee -a "$logfile" >&2
        exit 1
      fi
    else
      read -p "Auto-fix Python lint issues? (yes/no): " fix_lint
      if [[ "$fix_lint" == "yes" ]]; then
        log_section_start "AUTO-FIX PYTHON LINT" "$logfile"

        # Run auto-fix tools and capture output
        local ruff_fix_output format_fix_output

        # shellcheck disable=SC2086
        ruff_fix_output=$("$ruff_cmd" check --fix $lint_target "${ruff_check_args[@]}" 2>&1)
        if [[ -n "$ruff_fix_output" ]]; then
          echo "[RUFF FIXES]" | tee -a "$logfile"
          echo "$ruff_fix_output" | tee -a "$logfile"
        fi

        # shellcheck disable=SC2086
        format_fix_output=$("$ruff_cmd" format $lint_target "${ruff_format_args[@]}" 2>&1)
        if [[ -n "$format_fix_output" ]]; then
          echo "[FORMAT FIXES]" | tee -a "$logfile"
          echo "$format_fix_output" | tee -a "$logfile"
        fi

        log_section_end "AUTO-FIX PYTHON LINT" "$logfile" "0"

        echo ""
        if [[ ${staged_only} -eq 1 ]]; then
          echo "[--staged-only] Skipping auto-restaging (manual staging required)" | tee -a "$logfile"
        else
          echo "Restaging fixed files..." | tee -a "$logfile"
          git add .
          if [[ $? -ne 0 ]]; then
            echo "[X] Failed to stage fixed files" | tee -a "$logfile" >&2
            exit 1
          fi
          unstage_local_only_files
          echo "[OK] Fixed files staged" | tee -a "$logfile"
        fi
      else
        echo ""
        echo "To see Python lint errors, run: $ruff_cmd check ."
        read -p "Continue commit with Python lint errors? (yes/no): " continue_anyway
        if [[ "$continue_anyway" != "yes" ]]; then
          echo "Commit cancelled - fix Python lint errors first"
          exit 1
        fi
      fi
    fi
  elif [[ ${md_lint_error} -eq 1 ]]; then
    echo "[!] Markdown lint warnings detected (non-blocking)"
    echo "  Run: markdownlint-cli2 \"*.md\" \"docs/**/*.md\" to see details"
    echo "  Use --skip-md-lint to suppress this warning"
  else
    echo "[OK] Code quality checks passed"
  fi
  echo ""

  # [5/7] Show staged changes
  echo "[5/7] Staged changes:"
  echo "===================================="
  git diff --cached --name-status
  echo "===================================="
  echo ""

  # Count staged files
  local staged_count
  staged_count=$(git diff --cached --name-only | wc -l)
  echo "Files to commit: $staged_count"
  echo ""

  # [6/7] Get commit message
  echo "[6/7] Commit message..."

  if [[ -z "$commit_msg_param" ]]; then
    echo "" >&2
    echo "Commit message required" >&2
    echo "" >&2
    echo "Usage: commit_enhanced.sh [--non-interactive] \"Your commit message\"" >&2
    echo "" >&2
    echo "Conventional commit format recommended:" >&2
    echo "  feat:   New feature" >&2
    echo "  fix:    Bug fix" >&2
    echo "  docs:   Documentation changes" >&2
    echo "  chore:  Maintenance tasks" >&2
    echo "  test:   Test changes" >&2
    echo "" >&2
    echo "Example: ./commit_enhanced.sh \"feat: Add semantic search caching\"" >&2
    echo "Example: ./commit_enhanced.sh --non-interactive \"feat: Add semantic search caching\"" >&2
    exit 1
  fi

  local commit_msg="$commit_msg_param"

  # Basic commit message validation
  if ! echo "$commit_msg" | grep -qE "^(feat|fix|docs|chore|test|refactor|style|perf):"; then
    echo "[!] WARNING: Commit message doesn't follow conventional format"
    echo "  Recommended prefixes: feat:, fix:, docs:, chore:, test:"
    echo ""

    if [[ ${non_interactive} -eq 1 ]]; then
      echo "[Non-interactive mode] Continuing with non-conventional format..."
    else
      read -p "Continue anyway? (yes/no): " continue_commit
      if [[ "$continue_commit" != "yes" ]]; then
        echo "Commit cancelled"
        exit 0
      fi
    fi
  fi

  echo ""
  echo "Commit message: $commit_msg"
  echo ""

  # [7/7] Create commit
  echo "[7/7] Creating commit..."
  echo ""

  if [[ ${non_interactive} -eq 1 ]]; then
    echo "[Non-interactive mode] Branch: $current_branch"
    echo "[Non-interactive mode] Proceeding with commit..."
  else
    echo "[!] BRANCH VERIFICATION"
    echo "You are about to commit to: $current_branch"
    echo ""
    read -p "Is this the correct branch? (yes/no): " correct_branch
    if [[ "$correct_branch" != "yes" ]]; then
      echo ""
      echo "Available branches:"
      git branch
      echo ""
      echo "Switch to the correct branch first, then run this script again"
      echo "Command: git checkout <branch-name>"
      exit 0
    fi
    echo ""
    read -p "Proceed with commit? (yes/no): " confirm_commit
    if [[ "$confirm_commit" != "yes" ]]; then
      echo "Commit cancelled"
      exit 0
    fi
  fi

  git commit -m "$commit_msg"

  if [[ $? -eq 0 ]]; then
    echo ""
    echo "===================================="
    echo "[OK] COMMIT SUCCESSFUL"
    echo "===================================="
    echo ""
    echo "Commit: $(git log -1 --oneline)"
    echo "Branch: $current_branch"
    echo "Files: $staged_count"
    echo ""
    echo "[OK] Local files remained private"
    echo "[OK] Branch-specific validations passed"
    echo ""
    echo "Next steps:"
    if [[ "$current_branch" == "development" ]]; then
      echo "  - Continue development"
      echo "  - When ready: scripts/git/merge_with_validation.bat"
    elif [[ "$current_branch" == "main" ]]; then
      echo "  - Test changes thoroughly"
      echo "  - Push to remote: git push origin main"
    else
      echo "  - Push to remote: git push origin $current_branch"
    fi

    # Generate analysis report
    generate_analysis_report
  else
    echo "" >&2
    echo "[X] Commit failed - check output above" >&2
    exit 1
  fi

  exit 0
}

# Call main function
main "$@"
