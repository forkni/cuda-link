#!/usr/bin/env bash
# Lint validation script for Git Bash / Linux / macOS
# Windows cmd.exe users: use check_lint.bat instead

# Get directory of this script
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source common functions
# shellcheck source=scripts/git/_common.sh
source "${SCRIPT_DIR}/_common.sh"

main() {
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
      echo "[OK] No modified Python files to check"
      exit 0
    fi

    echo "=== Modified-Only Lint Check ==="
    echo "Files: $modified_py"
    echo ""

    EXIT_CODE=0

    echo "[RUFF CHECK]"
    ${PYTHON_BIN}/ruff${PYTHON_EXT} check $modified_py || EXIT_CODE=1

    echo ""
    echo "[RUFF FORMAT]"
    ${PYTHON_BIN}/ruff${PYTHON_EXT} format --check $modified_py || EXIT_CODE=1

    exit $EXIT_CODE
  fi

  # Get cross-platform Python paths
  if ! get_python_path; then
    exit 1
  fi

  # Get lint exclusion patterns
  get_lint_exclusions

  # Initialize logging
  init_logging "check_lint"

  # Track script start time
  local script_start
  script_start=$(date +%s)

  # Write log header
  {
    echo "========================================="
    echo "Lint Validation Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } > "$logfile"

  # Track results for summary: "name:status:errors:duration"
  local -a results=()
  local ruff_status=0 format_status=0 md_status=0

  # RUFF CHECK (linting + import sorting)
  local ruff_start ruff_end ruff_duration
  ruff_start=$(date +%s)
  if ! run_tool_with_logging "RUFF CHECK" "$logfile" \
      "${PYTHON_BIN}/ruff${PYTHON_EXT}" check . ${RUFF_CHECK_EXCLUDE}; then
    ruff_status=1
  fi
  ruff_end=$(date +%s)
  ruff_duration=$((ruff_end - ruff_start))
  results+=("Ruff:$([ $ruff_status -eq 0 ] && echo PASSED || echo FAILED):${TOOL_ERROR_COUNT}:${ruff_duration}")

  # RUFF FORMAT CHECK (replaces black)
  local format_start format_end format_duration
  format_start=$(date +%s)
  if ! run_tool_with_logging "RUFF FORMAT CHECK" "$logfile" \
      "${PYTHON_BIN}/ruff${PYTHON_EXT}" format --check . ${RUFF_FORMAT_EXCLUDE}; then
    format_status=1
  fi
  format_end=$(date +%s)
  format_duration=$((format_end - format_start))
  results+=("Format:$([ $format_status -eq 0 ] && echo PASSED || echo FAILED):${TOOL_ERROR_COUNT}:${format_duration}")

  # MARKDOWNLINT - with timestamps and output capture
  local md_start md_end md_duration
  md_start=$(date +%s)
  if ! run_tool_with_logging "MARKDOWNLINT CHECK" "$logfile" \
      markdownlint-cli2 ${MD_PATTERNS}; then
    md_status=1
  fi
  md_end=$(date +%s)
  md_duration=$((md_end - md_start))
  results+=("Markdownlint:$([ $md_status -eq 0 ] && echo PASSED || echo FAILED):${TOOL_ERROR_COUNT}:${md_duration}")

  # Write summary table
  log_summary_table "$logfile" "${results[@]}"

  # Final status
  local script_end total_duration overall_status
  script_end=$(date +%s)
  total_duration=$((script_end - script_start))

  if [[ $ruff_status -eq 0 ]] && [[ $format_status -eq 0 ]] && [[ $md_status -eq 0 ]]; then
    overall_status="PASSED"
  else
    overall_status="FAILED"
  fi

  {
    echo ""
    echo "End Time: $(date)"
    echo "Total Duration: ${total_duration}s"
    echo "STATUS: $overall_status"
  } | tee -a "$logfile"

  echo ""
  echo "Full log: $logfile"

  [[ "$overall_status" == "PASSED" ]] && exit 0 || exit 1
}

main "$@"
