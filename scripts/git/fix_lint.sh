#!/usr/bin/env bash
# Auto-fix lint issues for Git Bash / Linux / macOS
# Windows cmd.exe users: use fix_lint.bat instead

# Get directory of this script
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source common functions
source "${SCRIPT_DIR}/_common.sh"

main() {
  # Parse command-line arguments
  local non_interactive=0
  if [[ "$1" == "--non-interactive" ]]; then
    non_interactive=1
  fi

  # Change to project root
  cd "${PROJECT_ROOT}" || {
    echo "[ERROR] Cannot find project root" >&2
    exit 1
  }

  # Handle --modified-only mode
  if [[ "$1" == "--modified-only" ]]; then
    # Get cross-platform Python paths
    if ! get_python_path; then
      exit 1
    fi

    modified_py=$(git diff --name-only --diff-filter=ACMR HEAD -- '*.py')
    if [[ -z "$modified_py" ]]; then
      echo "[OK] No modified Python files to fix"
      exit 0
    fi

    echo "=== Modified-Only Lint Fix ==="
    echo "Files: $modified_py"
    echo ""

    EXIT_CODE=0

    echo "[RUFF FIX]"
    ${PYTHON_BIN}/ruff${PYTHON_EXT} check --fix $modified_py || EXIT_CODE=1

    echo ""
    echo "[RUFF FORMAT]"
    ${PYTHON_BIN}/ruff${PYTHON_EXT} format $modified_py || EXIT_CODE=1

    exit $EXIT_CODE
  fi

  # Get cross-platform Python paths
  if ! get_python_path; then
    exit 1
  fi

  # Get lint exclusion patterns
  get_lint_exclusions

  # Initialize logging
  init_logging "fix_lint"

  # Track script start time
  local script_start
  script_start=$(date +%s)

  # Write log header
  {
    echo "========================================="
    echo "Lint Auto-Fix Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
    echo "Mode: $([ $non_interactive -eq 1 ] && echo 'Non-interactive' || echo 'Interactive')"
  } > "$logfile"

  # Track any failures
  local fix_failed=0

  # RUFF FIX (linting + import sorting)
  if ! run_tool_with_logging "RUFF AUTO-FIX" "$logfile" \
      "${PYTHON_BIN}/ruff${PYTHON_EXT}" check . --fix ${RUFF_CHECK_EXCLUDE}; then
    echo "[!] Ruff encountered issues (some may not be auto-fixable)" | tee -a "$logfile"
    fix_failed=1
  fi

  # RUFF FORMAT (replaces black)
  if ! run_tool_with_logging "RUFF FORMAT" "$logfile" \
      "${PYTHON_BIN}/ruff${PYTHON_EXT}" format . ${RUFF_FORMAT_EXCLUDE}; then
    echo "[X] Ruff formatting failed" | tee -a "$logfile" >&2
    fix_failed=1
  fi

  # MARKDOWNLINT FIX - with timestamps and output capture
  if ! run_tool_with_logging "MARKDOWNLINT FIX" "$logfile" \
      markdownlint-cli2 --fix ${MD_PATTERNS}; then
    echo "[!] Markdownlint encountered issues (some may not be auto-fixable)" | tee -a "$logfile"
  fi

  # Final status message
  {
    echo ""
    echo "========================================"
    echo "[FIX SUMMARY]"
    echo "========================================"
  } | tee -a "$logfile"

  if (( fix_failed == 0 )); then
    echo "[OK] All lint fixes applied successfully!" | tee -a "$logfile"
  else
    echo "[!] Some lint tools encountered errors - check output above" | tee -a "$logfile" >&2
  fi

  # Run final verification
  {
    echo ""
    echo "========================================"
    echo "[FINAL VERIFICATION]"
    echo "========================================"
  } | tee -a "$logfile"

  echo "Running final lint check..." | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  # Run check_lint.sh and capture its output
  if "${SCRIPT_DIR}/check_lint.sh" 2>&1 | tee -a "$logfile"; then
    echo "" | tee -a "$logfile"
    echo "[OK] All lint checks now pass!" | tee -a "$logfile"
  else
    echo "" | tee -a "$logfile"
    echo "[!] Some lint issues remain - manual fixes may be required" | tee -a "$logfile"
  fi

  # Calculate total duration
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
