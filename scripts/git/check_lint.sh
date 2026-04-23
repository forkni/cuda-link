#!/usr/bin/env bash
# check_lint.sh - Lint validation (read-only, no modifications)
# Purpose: Check code quality without making changes
# Usage: ./scripts/git/check_lint.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR     - Directory containing this script
#   PROJECT_ROOT   - Auto-detected git repo root (set by _config.sh)
#   logfile        - Set by init_logging
#   CGW_LINT_CMD   - Lint tool to use (default: ruff; empty = skip)
# Returns:
#   0 on lint pass, 1 on lint errors

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/git/_common.sh
source "${SCRIPT_DIR}/_common.sh"

main() {
  local modified_only=0
  local skip_lint=0
  local skip_md_lint=0

  [[ "${CGW_SKIP_LINT:-0}" == "1" ]] && skip_lint=1 && skip_md_lint=1
  [[ "${CGW_SKIP_MD_LINT:-0}" == "1" ]] && skip_md_lint=1

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help | -h)
        echo "Usage: ./scripts/git/check_lint.sh [OPTIONS]"
        echo ""
        echo "Run lint and format checks (read-only, no modifications)."
        echo ""
        echo "Options:"
        echo "  --modified-only   Only check files modified vs HEAD"
        echo "  --no-venv         Use system lint tool instead of .venv"
        echo "  --skip-lint       Skip all lint checks"
        echo "  --skip-md-lint    Skip markdown lint only (CGW_MARKDOWNLINT_CMD step)"
        echo "  -h, --help        Show this help"
        echo ""
        echo "Environment:"
        echo "  CGW_NO_VENV=1          Same as --no-venv"
        echo "  CGW_SKIP_LINT=1        Same as --skip-lint"
        echo "  CGW_SKIP_MD_LINT=1     Same as --skip-md-lint"
        echo "  CGW_LINT_CMD=<tool>    Override lint tool (default: ruff)"
        echo "  (Also: CLAUDE_GIT_NO_VENV)"
        exit 0
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
      --skip-lint)
        skip_lint=1
        skip_md_lint=1
        shift
        ;;
      --skip-md-lint)
        skip_md_lint=1
        shift
        ;;
      *)
        echo "[ERROR] Unknown flag: $1" >&2
        exit 1
        ;;
    esac
  done

  if [[ ${skip_lint} -eq 1 ]]; then
    echo "[OK] All lint checks skipped (--skip-lint)"
    exit 0
  fi

  if [[ -z "${CGW_LINT_CMD}" ]] && [[ -z "${CGW_FORMAT_CMD}" ]] && [[ -z "${CGW_MARKDOWNLINT_CMD}" ]]; then
    echo "[OK] All lint checks skipped (CGW_LINT_CMD, CGW_FORMAT_CMD, and CGW_MARKDOWNLINT_CMD not set)"
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

  # Handle --modified-only mode (code lint only, requires CGW_LINT_CMD)
  if [[ "${modified_only}" -eq 1 ]]; then
    if [[ -z "${CGW_LINT_CMD}" ]]; then
      echo "[OK] No code lint tool configured for --modified-only (CGW_LINT_CMD not set)"
      exit 0
    fi
    local modified_files
    # CGW_LINT_EXTENSIONS controls which files are considered (default: *.py)
    local -a lint_exts
    read -r -a lint_exts <<<"${CGW_LINT_EXTENSIONS:-*.py}"
    modified_files=$(git diff --name-only --diff-filter=ACMR HEAD -- "${lint_exts[@]}")
    if [[ -z "$modified_files" ]]; then
      echo "[OK] No modified files to check"
      exit 0
    fi

    echo "=== Modified-Only Lint Check ==="
    echo "Files: $modified_files"
    echo ""

    local EXIT_CODE=0

    echo "[LINT CHECK]"
    # Build check args: strip trailing path token (.) and append specific files
    local lint_check_cmd_args="${CGW_LINT_CHECK_ARGS% *}"
    # shellcheck disable=SC2086
    "${lint_cmd}" ${lint_check_cmd_args} $modified_files || EXIT_CODE=1

    if [[ -n "${CGW_FORMAT_CMD}" ]]; then
      echo ""
      echo "[FORMAT CHECK]"
      # Build format check args: strip trailing path token (.) and append specific files
      local fmt_check_cmd_args="${CGW_FORMAT_CHECK_ARGS% *}"
      # shellcheck disable=SC2086
      "${CGW_FORMAT_CMD}" ${fmt_check_cmd_args} $modified_files || EXIT_CODE=1
    fi

    exit $EXIT_CODE
  fi

  # Full lint check with logging
  init_logging "check_lint"

  local script_start
  script_start=$(date +%s)

  {
    echo "========================================="
    echo "Lint Validation Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
    echo "Lint tool: ${CGW_LINT_CMD}"
  } >"$logfile"

  local -a results=()
  local lint_status=0 format_status=0 md_lint_status=0

  # LINT CHECK (skipped when CGW_LINT_CMD is not set)
  if [[ -n "${CGW_LINT_CMD}" ]]; then
    local lint_start lint_end lint_duration lint_status_str
    lint_start=$(date +%s)
    # shellcheck disable=SC2086  # Word splitting intentional: CGW_LINT_CHECK_ARGS contains multiple flags
    if ! run_tool_with_logging "LINT CHECK" "$logfile" \
      "${lint_cmd}" ${CGW_LINT_CHECK_ARGS} ${CGW_LINT_EXCLUDES}; then
      lint_status=1
    fi
    lint_end=$(date +%s)
    lint_duration=$((lint_end - lint_start))
    lint_status_str="PASSED"
    [[ ${lint_status} -ne 0 ]] && lint_status_str="FAILED"
    results+=("Lint:${lint_status_str}:${TOOL_ERROR_COUNT}:${lint_duration}")
  else
    echo "  (code lint skipped -- CGW_LINT_CMD not set)" | tee -a "$logfile"
  fi

  # FORMAT CHECK (independent of lint check -- runs even when CGW_LINT_CMD is unset)
  if [[ -n "${CGW_FORMAT_CMD}" ]]; then
    local format_start format_end format_duration format_status_str
    format_start=$(date +%s)
    # shellcheck disable=SC2086  # Word splitting intentional: CGW_FORMAT_CHECK_ARGS contains multiple flags
    if ! run_tool_with_logging "FORMAT CHECK" "$logfile" \
      "${CGW_FORMAT_CMD}" ${CGW_FORMAT_CHECK_ARGS} ${CGW_FORMAT_EXCLUDES}; then
      format_status=1
    fi
    format_end=$(date +%s)
    format_duration=$((format_end - format_start))
    format_status_str="PASSED"
    [[ ${format_status} -ne 0 ]] && format_status_str="FAILED"
    results+=("Format:${format_status_str}:${TOOL_ERROR_COUNT}:${format_duration}")
  fi

  # MARKDOWN LINT
  if [[ ${skip_md_lint} -eq 0 ]] && [[ -n "${CGW_MARKDOWNLINT_CMD}" ]]; then
    local md_start md_end md_duration md_status_str
    md_start=$(date +%s)
    # shellcheck disable=SC2086  # Word splitting intentional: CGW_MARKDOWNLINT_ARGS contains multiple flags/patterns
    if ! run_tool_with_logging "MARKDOWN LINT" "$logfile" \
      "${CGW_MARKDOWNLINT_CMD}" ${CGW_MARKDOWNLINT_ARGS}; then
      md_lint_status=1
    fi
    md_end=$(date +%s)
    md_duration=$((md_end - md_start))
    md_status_str="PASSED"
    [[ ${md_lint_status} -ne 0 ]] && md_status_str="FAILED"
    results+=("Markdown:${md_status_str}:${TOOL_ERROR_COUNT}:${md_duration}")
  elif [[ ${skip_md_lint} -eq 1 ]]; then
    echo "  (markdown lint skipped -- --skip-md-lint)" | tee -a "$logfile"
  fi

  log_summary_table "$logfile" "${results[@]}"

  local script_end total_duration overall_status
  script_end=$(date +%s)
  total_duration=$((script_end - script_start))

  if [[ $lint_status -eq 0 ]] && [[ $format_status -eq 0 ]] && [[ $md_lint_status -eq 0 ]]; then
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
