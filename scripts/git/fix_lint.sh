#!/usr/bin/env bash
# fix_lint.sh - Auto-fix lint issues
# Purpose: Run lint auto-fix and formatting
# Usage: ./scripts/git/fix_lint.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR     - Directory containing this script
#   PROJECT_ROOT   - Auto-detected git repo root (set by _config.sh)
#   logfile        - Set by init_logging
#   CGW_LINT_CMD   - Lint tool to use (default: ruff; empty = skip)
# Returns:
#   0 on success, 1 if issues remain after fix

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

main() {
  local non_interactive=0
  local modified_only=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help | -h)
        echo "Usage: ./scripts/git/fix_lint.sh [OPTIONS]"
        echo ""
        echo "Auto-fix lint issues using configured lint tool."
        echo ""
        echo "Options:"
        echo "  --modified-only     Only fix files modified vs HEAD"
        echo "  --non-interactive   Skip prompts"
        echo "  --no-venv           Use system lint tool instead of .venv"
        echo "  -h, --help          Show this help"
        echo ""
        echo "Environment:"
        echo "  CGW_NON_INTERACTIVE=1   Same as --non-interactive"
        echo "  CGW_NO_VENV=1           Same as --no-venv"
        echo "  (Also: CLAUDE_GIT_NON_INTERACTIVE, CLAUDE_GIT_NO_VENV)"
        exit 0
        ;;
      --non-interactive)
        non_interactive=1
        shift
        ;;
      --no-venv)
        CGW_NO_VENV=1
        SKIP_VENV=1
        shift
        ;;
      --modified-only)
        modified_only=1
        shift
        ;;
      *)
        echo "[ERROR] Unknown flag: $1" >&2
        exit 1
        ;;
    esac
  done

  [[ "${CGW_NON_INTERACTIVE:-0}" == "1" ]] && non_interactive=1

  if [[ -z "${CGW_LINT_CMD}" ]]; then
    echo "[OK] Lint fix skipped (CGW_LINT_CMD not set -- configure in .cgw.conf)"
    exit 0
  fi

  cd "${PROJECT_ROOT}" || {
    err "Cannot find project root"
    exit 1
  }

  get_lint_exclusions

  # Determine lint binary (venv or PATH)
  local lint_cmd="${CGW_LINT_CMD}"
  if [[ "${CGW_LINT_CMD}" == "ruff" ]]; then
    get_python_path 2>/dev/null || true
    if [[ -n "${PYTHON_BIN:-}" ]] && [[ -f "${PYTHON_BIN}/ruff${PYTHON_EXT:-}" ]]; then
      lint_cmd="${PYTHON_BIN}/ruff${PYTHON_EXT:-}"
    fi
  fi

  # Handle --modified-only mode
  if [[ "${modified_only}" -eq 1 ]]; then
    local modified_files
    # CGW_LINT_EXTENSIONS controls which files are considered (default: *.py)
    local -a lint_exts
    read -r -a lint_exts <<<"${CGW_LINT_EXTENSIONS:-*.py}"
    modified_files=$(git diff --name-only --diff-filter=ACMR HEAD -- "${lint_exts[@]}")
    if [[ -z "$modified_files" ]]; then
      echo "[OK] No modified files to fix"
      exit 0
    fi

    echo "=== Modified-Only Lint Fix ==="
    echo "Files: $modified_files"
    echo ""

    local EXIT_CODE=0

    echo "[LINT FIX]"
    # Build fix args: strip trailing path token (.) and append specific files
    local lint_fix_cmd_args="${CGW_LINT_FIX_ARGS% *}"
    # shellcheck disable=SC2086
    "${lint_cmd}" ${lint_fix_cmd_args} $modified_files || EXIT_CODE=1

    if [[ -n "${CGW_FORMAT_CMD}" ]]; then
      echo ""
      echo "[FORMAT FIX]"
      # Build format fix args: strip trailing path token (.) and append specific files
      local fmt_fix_cmd_args="${CGW_FORMAT_FIX_ARGS% *}"
      # shellcheck disable=SC2086
      "${CGW_FORMAT_CMD}" ${fmt_fix_cmd_args} $modified_files || EXIT_CODE=1
    fi

    exit $EXIT_CODE
  fi

  # Full fix with logging
  init_logging "fix_lint"

  local script_start
  script_start=$(date +%s)

  {
    echo "========================================="
    echo "Lint Auto-Fix Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
    echo "Lint tool: ${CGW_LINT_CMD}"
    echo "Mode: $([ $non_interactive -eq 1 ] && echo 'Non-interactive' || echo 'Interactive')"
  } >"$logfile"

  local fix_failed=0

  # LINT FIX
  # shellcheck disable=SC2086
  if ! run_tool_with_logging "LINT AUTO-FIX" "$logfile" \
    "${lint_cmd}" ${CGW_LINT_FIX_ARGS} ${CGW_LINT_EXCLUDES}; then
    echo "[!] Lint tool: some issues may not be auto-fixable" | tee -a "$logfile"
    fix_failed=1
  fi

  # FORMAT FIX
  if [[ -n "${CGW_FORMAT_CMD}" ]]; then
    # shellcheck disable=SC2086
    if ! run_tool_with_logging "FORMAT FIX" "$logfile" \
      "${CGW_FORMAT_CMD}" ${CGW_FORMAT_FIX_ARGS} ${CGW_FORMAT_EXCLUDES}; then
      err "Formatting failed"
      fix_failed=1
    fi
  fi

  {
    echo ""
    echo "========================================"
    echo "[FIX SUMMARY]"
    echo "========================================"
  } | tee -a "$logfile"

  if ((fix_failed == 0)); then
    echo "[OK] All lint fixes applied successfully!" | tee -a "$logfile"
  else
    echo "[!] Some issues remain -- check output above" | tee -a "$logfile"
  fi

  # Run final verification
  echo "" | tee -a "$logfile"
  echo "Running final verification..." | tee -a "$logfile"

  if "${SCRIPT_DIR}/check_lint.sh" 2>&1 | tee -a "$logfile"; then
    echo "[OK] All lint checks pass!" | tee -a "$logfile"
  else
    echo "[!] Some issues remain -- manual fixes may be required" | tee -a "$logfile"
  fi

  local script_end total_duration
  script_end=$(date +%s)
  total_duration=$((script_end - script_start))

  {
    echo ""
    echo "End Time: $(date)"
    echo "Total Duration: ${total_duration}s"
  } | tee -a "$logfile"

  echo ""
  echo "Full log: $logfile"
}

main "$@"
