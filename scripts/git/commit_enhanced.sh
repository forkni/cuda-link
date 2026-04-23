#!/usr/bin/env bash
# commit_enhanced.sh - Enhanced commit workflow with validation and logging
# Purpose: Safe commit with lint check, local-file protection, and conventional format
# Usage: ./scripts/git/commit_enhanced.sh [OPTIONS] "commit message"
#
# Globals:
#   SCRIPT_DIR       - Directory containing this script
#   PROJECT_ROOT     - Auto-detected git repo root (set by _config.sh)
#   logfile          - Set by init_logging
#   CGW_LOCAL_FILES  - Space-separated list of files never to commit
#   CGW_ALL_PREFIXES - Allowed commit message type prefixes
# Arguments:
#   --non-interactive  Skip all prompts
#   --interactive      Force interactive mode even without TTY
#   --staged-only      Use pre-staged files only, skip auto-staging
#   --no-venv          Use system ruff instead of .venv ruff
#   --skip-md-lint     (no-op, preserved for backward compat)
#   -h, --help         Show help
# Returns:
#   0 on successful commit, 1 on failure

set -uo pipefail
# Note: set -e intentionally omitted -- git diff/diff-index use exit codes for signaling

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

generate_analysis_report() {
  cat >"$reportfile" <<EOF
# Enhanced Commit Workflow Analysis Report

**Date**: $(date)
**Branch**: $current_branch
**Status**: SUCCESS

## Files Committed

$(git diff HEAD~1 --name-status 2>/dev/null)

## Commit Details

$(git log -1 --pretty=format:"- **Hash**: %H%n- **Message**: %s%n- **Author**: %an%n- **Date**: %ad%n" 2>/dev/null)

## Validations Passed

- [OK] No local-only files committed
- [OK] Branch-specific validations passed
- [OK] Code quality checks passed (or auto-fixed)
- [OK] Conventional commit format validated

## Logs

- Execution log: \`$logfile\`
- Analysis report: \`$reportfile\`

EOF

  echo "End Time: $(date)" >>"$logfile"
}

unstage_local_only_files() {
  # Unstage files listed in CGW_LOCAL_FILES (space-separated).
  # Entries ending with / are treated as directory prefixes.
  local file
  for file in ${CGW_LOCAL_FILES}; do
    if [[ "${file}" == */ ]]; then
      # Directory prefix: unstage all matching staged files
      while read -r f; do
        git reset HEAD "$f" 2>/dev/null || true
      done < <(git diff --cached --name-only | grep "^${file}" || true)
    else
      git reset HEAD "${file}" 2>/dev/null || true
    fi
  done
}

main() {
  local non_interactive=0
  local skip_lint=0
  local skip_md_lint=0
  local staged_only=0
  local commit_msg_param=""

  # Auto-detect non-interactive mode when no TTY
  if [[ ! -t 0 ]]; then
    non_interactive=1
  fi

  # CGW_* environment variable overrides
  [[ "${CGW_NON_INTERACTIVE:-0}" == "1" ]] && non_interactive=1
  [[ "${CGW_STAGED_ONLY:-0}" == "1" ]] && staged_only=1
  [[ "${CGW_NO_VENV:-0}" == "1" ]] && SKIP_VENV=1
  [[ "${CGW_SKIP_LINT:-0}" == "1" ]] && skip_lint=1 && skip_md_lint=1
  [[ "${CGW_SKIP_MD_LINT:-0}" == "1" ]] && skip_md_lint=1

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help | -h)
        echo "Usage: ./scripts/git/commit_enhanced.sh [OPTIONS] \"commit message\""
        echo ""
        echo "Enhanced commit workflow with lint validation and local-only file protection."
        echo ""
        echo "Options:"
        echo "  --non-interactive   Skip all prompts (auto-stage, auto-fix lint)"
        echo "  --interactive       Force interactive mode even without TTY"
        echo "  --staged-only       Use pre-staged files only, skip auto-staging"
        echo "  --no-venv           Use system ruff instead of .venv ruff"
        echo "  --skip-lint         Skip all lint checks (code + markdown)"
        echo "  --skip-md-lint      Skip markdown lint only (CGW_MARKDOWNLINT_CMD step)"
        echo "  -h, --help          Show this help"
        echo ""
        echo "Commit message format: <type>: <message>"
        echo "  Standard types: feat fix docs chore test refactor style perf"
        echo "  Configure extras via CGW_EXTRA_PREFIXES in .cgw.conf"
        echo ""
        echo "Environment:"
        echo "  CGW_NON_INTERACTIVE=1   Same as --non-interactive"
        echo "  CGW_STAGED_ONLY=1       Same as --staged-only"
        echo "  CGW_NO_VENV=1           Same as --no-venv"
        echo "  CGW_SKIP_LINT=1         Same as --skip-lint"
        echo "  CGW_SKIP_MD_LINT=1      Same as --skip-md-lint"
        echo "  (Also: CLAUDE_GIT_NON_INTERACTIVE, CLAUDE_GIT_STAGED_ONLY, CLAUDE_GIT_NO_VENV)"
        echo ""
        echo "Protected files (never committed): configured via CGW_LOCAL_FILES in .cgw.conf"
        echo "  Default: CLAUDE.md MEMORY.md .claude/ logs/"
        exit 0
        ;;
      --non-interactive)
        non_interactive=1
        shift
        ;;
      --skip-lint)
        skip_lint=1
        skip_md_lint=1
        shift
        ;;
      --skip-md-lint)
        skip_md_lint=1
        shift
        ;;
      --interactive)
        non_interactive=0
        shift
        ;;
      --staged-only)
        staged_only=1
        shift
        ;;
      --no-venv)
        SKIP_VENV=1
        CGW_NO_VENV=1
        shift
        ;;
      --*)
        echo "[ERROR] Unknown flag: $1" >&2
        exit 1
        ;;
      *)
        commit_msg_param="$1"
        shift
        ;;
    esac
  done

  init_logging "commit_enhanced"

  if [[ -z "$logfile" ]] || [[ -z "$reportfile" ]]; then
    err "Failed to initialize logging"
    exit 1
  fi

  {
    echo "========================================="
    echo "Enhanced Commit Workflow Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Branch: $(git branch --show-current)"
    echo ""
  } >"$logfile"

  log_message "=== Enhanced Commit Workflow ===" "$logfile"
  log_message "" "$logfile"

  cd "${PROJECT_ROOT}" || {
    err "Cannot find project root"
    exit 1
  }

  # Get Python path (best-effort, ruff may be in PATH)
  get_python_path 2>/dev/null || true

  local current_branch
  current_branch=$(git branch --show-current)
  echo "Current branch: $current_branch"
  echo ""

  # [1] Check for uncommitted changes
  echo "[1/6] Checking for changes..."

  git diff --quiet
  local has_unstaged=$?
  git diff --cached --quiet
  local has_staged=$?

  if [[ ${has_unstaged} -eq 0 ]] && [[ ${has_staged} -eq 0 ]]; then
    echo "[!] No changes to commit"
    exit 0
  fi

  if [[ ${has_unstaged} -ne 0 ]]; then
    echo "Unstaged changes detected:"
    git diff --name-status
    echo ""

    if [[ ${staged_only} -eq 1 ]]; then
      echo "[--staged-only] Using pre-staged files only"
    elif [[ ${non_interactive} -eq 1 ]]; then
      echo "[Non-interactive] Auto-staging tracked changes..."
      git add -u
      unstage_local_only_files
      echo "[OK] Changes staged"
    else
      read -rp "Stage all tracked changes? (yes/no): " stage_all
      if [[ "$stage_all" == "yes" ]]; then
        git add -u
        unstage_local_only_files
        echo "[OK] Changes staged"
      else
        echo "Please stage changes manually: git add <files>"
        exit 1
      fi
    fi
  fi
  echo ""

  # [2] Validate staged files -- unstage and verify local-only files
  echo "[2/6] Validating staged files..."
  unstage_local_only_files

  # Post-unstage check: verify nothing slipped through
  local found_local_files=0
  local staged_files
  staged_files=$(git diff --cached --name-only)

  local file
  for file in ${CGW_LOCAL_FILES}; do
    local check_file="${file%/}" # strip trailing slash
    if echo "${staged_files}" | grep -q "^${check_file}"; then
      echo "[X] ERROR: '${check_file}' is staged (local-only file -- should not be committed)" >&2
      found_local_files=1
    fi
  done

  if [[ ${found_local_files} -eq 1 ]]; then
    echo "Remove these files from staging: git reset HEAD <file>" >&2
    exit 1
  fi

  echo "[OK] Staged files validated"
  echo ""

  # [2.5] Whitespace check (non-blocking -- warns but does not abort)
  if git diff --cached --check >/dev/null 2>&1; then
    : # no whitespace issues
  else
    echo "[WARN] Whitespace issues detected in staged files:" | tee -a "$logfile"
    git diff --cached --check 2>&1 | head -20 | tee -a "$logfile"
    echo "  (continuing -- fix with: git diff --cached --check)" | tee -a "$logfile"
    echo ""
  fi

  # [3] Code quality check
  echo "[3/6] Checking code quality..."

  if [[ ${skip_lint} -eq 1 ]]; then
    echo "  (all lint checks skipped -- --skip-lint)"
  else
    get_lint_exclusions

    # Resolve lint and format binaries independently (each uses venv ruff if available)
    local lint_cmd="${CGW_LINT_CMD}"
    local format_cmd="${CGW_FORMAT_CMD}"
    if [[ -n "${CGW_LINT_CMD}" ]] || [[ -n "${CGW_FORMAT_CMD}" ]]; then
      get_python_path 2>/dev/null || true
    fi
    if [[ -n "${CGW_LINT_CMD}" && "${CGW_LINT_CMD}" == "ruff" ]]; then
      if [[ -n "${PYTHON_BIN:-}" ]] && [[ -f "${PYTHON_BIN}/ruff${PYTHON_EXT:-}" ]]; then
        lint_cmd="${PYTHON_BIN}/ruff${PYTHON_EXT:-}"
      fi
    fi
    if [[ -n "${CGW_FORMAT_CMD}" && "${CGW_FORMAT_CMD}" == "ruff" ]]; then
      if [[ -n "${PYTHON_BIN:-}" ]] && [[ -f "${PYTHON_BIN}/ruff${PYTHON_EXT:-}" ]]; then
        format_cmd="${PYTHON_BIN}/ruff${PYTHON_EXT:-}"
      fi
    fi

    local lint_error=0 format_error=0 lint_output format_output

    # -- Code lint (skipped when CGW_LINT_CMD not set) -------------------------
    if [[ -n "${CGW_LINT_CMD}" ]]; then
      log_section_start "LINT CHECK" "$logfile"
      # shellcheck disable=SC2086  # Word splitting intentional: CGW_LINT_CHECK_ARGS/CGW_LINT_EXCLUDES contain multiple flags
      lint_output=$("${lint_cmd}" ${CGW_LINT_CHECK_ARGS} ${CGW_LINT_EXCLUDES} 2>&1) || lint_error=1
      if [[ -n "$lint_output" ]] && [[ "$lint_output" != *"All checks passed"* ]]; then
        echo "[LINT ERRORS]" | tee -a "$logfile"
        echo "$lint_output" | tee -a "$logfile"
      fi
      log_section_end "LINT CHECK" "$logfile" "$lint_error"
    else
      echo "  (lint check skipped -- CGW_LINT_CMD not set)"
    fi

    # -- Format check (skipped when CGW_FORMAT_CMD not set) --------------------
    if [[ -n "${CGW_FORMAT_CMD}" ]]; then
      log_section_start "FORMAT CHECK" "$logfile"
      # shellcheck disable=SC2086  # Word splitting intentional: CGW_FORMAT_CHECK_ARGS/CGW_FORMAT_EXCLUDES contain multiple flags
      format_output=$("${format_cmd}" ${CGW_FORMAT_CHECK_ARGS} ${CGW_FORMAT_EXCLUDES} 2>&1) || format_error=1
      if [[ -n "$format_output" ]] && [[ "$format_output" == *"would reformat"* ]]; then
        echo "[FORMAT ERRORS]" | tee -a "$logfile"
        echo "$format_output" | tee -a "$logfile"
      fi
      log_section_end "FORMAT CHECK" "$logfile" "$format_error"
    fi

    # -- Combined error handling -----------------------------------------------
    local python_lint_error=$((lint_error | format_error))

    if [[ ${python_lint_error} -eq 1 ]]; then
      echo "[!] Code quality errors detected"
      if [[ ${non_interactive} -eq 1 ]]; then
        echo "[Non-interactive] Auto-fixing code quality issues..."
        if [[ -n "${CGW_LINT_CMD}" ]]; then
          # shellcheck disable=SC2086  # Word splitting intentional: CGW_LINT_FIX_ARGS/CGW_LINT_EXCLUDES contain multiple flags
          "${lint_cmd}" ${CGW_LINT_FIX_ARGS} ${CGW_LINT_EXCLUDES} 2>&1 | tee -a "$logfile"
        fi
        if [[ -n "${CGW_FORMAT_CMD}" ]]; then
          # shellcheck disable=SC2086  # Word splitting intentional: CGW_FORMAT_FIX_ARGS/CGW_FORMAT_EXCLUDES contain multiple flags
          "${format_cmd}" ${CGW_FORMAT_FIX_ARGS} ${CGW_FORMAT_EXCLUDES} 2>&1 | tee -a "$logfile"
        fi

        if [[ ${staged_only} -eq 0 ]]; then
          git add -u
          unstage_local_only_files
        fi

        # Re-check
        python_lint_error=0
        if [[ -n "${CGW_LINT_CMD}" ]]; then
          # shellcheck disable=SC2086  # Word splitting intentional: CGW_LINT_CHECK_ARGS/CGW_LINT_EXCLUDES contain multiple flags
          "${lint_cmd}" ${CGW_LINT_CHECK_ARGS} ${CGW_LINT_EXCLUDES} 2>&1 | tee -a "$logfile" || python_lint_error=1
        fi
        if [[ -n "${CGW_FORMAT_CMD}" ]]; then
          # shellcheck disable=SC2086  # Word splitting intentional: CGW_FORMAT_CHECK_ARGS/CGW_FORMAT_EXCLUDES contain multiple flags
          "${format_cmd}" ${CGW_FORMAT_CHECK_ARGS} ${CGW_FORMAT_EXCLUDES} 2>&1 | tee -a "$logfile" || python_lint_error=1
        fi

        if [[ ${python_lint_error} -eq 1 ]]; then
          err "Code quality errors remain after auto-fix"
          exit 1
        fi
      else
        read -rp "Auto-fix code quality issues? (yes/no/skip): " fix_lint
        case "$fix_lint" in
          yes | y)
            if [[ -n "${CGW_LINT_CMD}" ]]; then
              # shellcheck disable=SC2086  # Word splitting intentional: CGW_LINT_FIX_ARGS/CGW_LINT_EXCLUDES contain multiple flags
              "${lint_cmd}" ${CGW_LINT_FIX_ARGS} ${CGW_LINT_EXCLUDES}
            fi
            if [[ -n "${CGW_FORMAT_CMD}" ]]; then
              # shellcheck disable=SC2086  # Word splitting intentional: CGW_FORMAT_FIX_ARGS/CGW_FORMAT_EXCLUDES contain multiple flags
              "${format_cmd}" ${CGW_FORMAT_FIX_ARGS} ${CGW_FORMAT_EXCLUDES}
            fi
            git add -u
            unstage_local_only_files
            ;;
          skip | s)
            echo "[!] Proceeding with code quality warnings (CI may flag these)"
            ;;
          *)
            echo "Commit cancelled -- fix code quality errors first"
            exit 1
            ;;
        esac
      fi
    else
      echo "[OK] Code quality checks passed"
    fi

    # Markdown lint step (skipped if --skip-md-lint or CGW_MARKDOWNLINT_CMD not set)
    if [[ ${skip_md_lint} -eq 0 ]] && [[ -n "${CGW_MARKDOWNLINT_CMD}" ]]; then
      log_section_start "MARKDOWN LINT" "$logfile"
      local md_lint_error=0
      # shellcheck disable=SC2086  # Word splitting intentional: CGW_MARKDOWNLINT_ARGS contains multiple flags/patterns
      if ! "${CGW_MARKDOWNLINT_CMD}" ${CGW_MARKDOWNLINT_ARGS} 2>&1 | tee -a "$logfile"; then
        md_lint_error=1
      fi
      log_section_end "MARKDOWN LINT" "$logfile" "$md_lint_error"
      if [[ ${md_lint_error} -eq 1 ]]; then
        echo "[!] Markdown lint errors detected"
        if [[ ${non_interactive} -eq 1 ]]; then
          err "Markdown lint failed -- fix errors or use --skip-md-lint to bypass"
          exit 1
        fi
        read -rp "Proceed despite markdown lint errors? (yes/no): " md_choice
        [[ "${md_choice}" == "yes" ]] || exit 1
      fi
    elif [[ ${skip_md_lint} -eq 1 ]]; then
      echo "  (markdown lint skipped -- --skip-md-lint)"
    fi
  fi
  echo ""

  # [4] Show staged changes
  echo "[4/6] Staged changes:"
  echo "===================================="
  git diff --cached --name-status
  echo "===================================="
  echo ""

  local staged_count
  staged_count=$(git diff --cached --name-only | wc -l)
  echo "Files to commit: $staged_count"
  echo ""

  # [5] Get commit message
  echo "[5/6] Commit message..."

  if [[ -z "$commit_msg_param" ]]; then
    err "Commit message required"
    echo "Usage: ./scripts/git/commit_enhanced.sh \"feat: Your message\"" >&2
    echo "Types: feat fix docs chore test refactor style perf (+ extras in .cgw.conf)" >&2
    exit 1
  fi

  local commit_msg="$commit_msg_param"

  if ! echo "$commit_msg" | grep -qE "^(${CGW_ALL_PREFIXES}):"; then
    echo "[!] WARNING: Message doesn't follow conventional format"
    echo "  Configured types: ${CGW_ALL_PREFIXES/|/, }"
    if [[ ${non_interactive} -eq 1 ]]; then
      err "Commit message must follow conventional format in non-interactive mode"
      err "Use --skip-lint or set CGW_EXTRA_PREFIXES if you need a custom prefix"
      exit 1
    else
      read -rp "Continue anyway? (yes/no): " continue_commit
      if [[ "$continue_commit" != "yes" ]]; then
        echo "Commit cancelled"
        exit 0
      fi
    fi
  fi

  echo "Commit message: $commit_msg"
  echo ""

  # [6] Create commit
  echo "[6/6] Creating commit..."

  if [[ ${non_interactive} -eq 1 ]]; then
    echo "[Non-interactive] Branch: $current_branch -- Proceeding..."
  else
    echo "[!] Branch verification: you are committing to: $current_branch"
    read -rp "Is this the correct branch? (yes/no): " correct_branch
    if [[ "$correct_branch" != "yes" ]]; then
      echo "Switch to correct branch first: git checkout <branch-name>"
      exit 0
    fi
    read -rp "Proceed with commit? (yes/no): " confirm_commit
    if [[ "$confirm_commit" != "yes" ]]; then
      echo "Commit cancelled"
      exit 0
    fi
  fi

  if git commit -m "$commit_msg"; then
    echo ""
    echo "===================================="
    echo "[OK] COMMIT SUCCESSFUL"
    echo "===================================="
    echo ""
    echo "Commit: $(git log -1 --oneline)"
    echo "Branch: $current_branch"
    echo "Files:  $staged_count"
    echo ""
    echo "Next steps:"
    if [[ "$current_branch" == "${CGW_SOURCE_BRANCH}" ]]; then
      echo "  - Continue development"
      echo "  - When ready: ./scripts/git/merge_with_validation.sh --dry-run"
    else
      echo "  - Push: ./scripts/git/push_validated.sh"
    fi

    generate_analysis_report
  else
    err "Commit failed -- check output above"
    exit 1
  fi

  exit 0
}

main "$@"
