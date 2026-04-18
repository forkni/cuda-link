#!/usr/bin/env bash
# _common.sh - Shared utility functions for claude-git-workflow scripts
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# Sourcing this file also sources _config.sh, which:
#   - Auto-detects PROJECT_ROOT (walks up to find .git/)
#   - Loads .cgw.conf if present
#   - Applies CGW_* variable defaults
#
# Available Functions:
#   err()                   - Print error message to STDERR
#   get_timestamp()         - Sets $timestamp variable (yyyyMMdd_HHmmss)
#   init_logging()          - Sets $logfile, $reportfile; creates logs/ dir
#   get_lint_exclusions()   - Sets RUFF_CHECK_EXCLUDE / RUFF_FORMAT_EXCLUDE from CGW config
#   get_python_path()       - Sets PYTHON_BIN and PYTHON_EXT (cross-platform venv detection)
#   log_message()           - Logs message to console and file
#   log_section_start/end() - Section headers with timing (safe for nested calls)
#   run_tool_with_logging() - Run a tool and capture output to log
#   run_git_with_logging()  - Run git command with section logging
#   validate_branch_pair()  - Validate src/tgt branch names and local existence; exit 1 on error

# SCRIPT_DIR must be set by the caller before sourcing _common.sh:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/_common.sh"
if [[ -z "${SCRIPT_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Source config (sets PROJECT_ROOT + all CGW_* variables)
# shellcheck source=scripts/git/_config.sh
source "${SCRIPT_DIR}/_config.sh"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Error output helper -- always goes to STDERR per style guide
err() {
  echo "[ERROR] $*" >&2
}

# Section timer storage -- associative array avoids global clobbering when
# sections are nested (e.g. run_tool_with_logging called inside another section)
declare -A _SECTION_START_TIMES=() 2>/dev/null || true

get_timestamp() {
  timestamp=$(date +%Y%m%d_%H%M%S)
}

init_logging() {
  local script_name="$1"
  # Use PROJECT_ROOT for an absolute path so logs land in the right place
  # even when the script is invoked from a subdirectory. PROJECT_ROOT is set
  # by _config.sh before any script calls init_logging.
  local log_dir="${PROJECT_ROOT:+${PROJECT_ROOT}/}logs"

  if [[ ! -d "${log_dir}" ]]; then
    mkdir -p "${log_dir}"
  fi

  get_timestamp

  # shellcheck disable=SC2034
  logfile="${log_dir}/${script_name}_${timestamp}.log"
  # shellcheck disable=SC2034
  reportfile="${log_dir}/${script_name}_analysis_${timestamp}.log"
}

get_lint_exclusions() {
  # Build ruff exclusion flags from CGW config variables.
  # Used by check_lint.sh, fix_lint.sh, commit_enhanced.sh.
  # shellcheck disable=SC2034
  RUFF_CHECK_EXCLUDE="${CGW_LINT_EXCLUDES}"
  # shellcheck disable=SC2034
  RUFF_FORMAT_EXCLUDE="${CGW_FORMAT_EXCLUDES}"
}

get_python_path() {
  # CGW_NO_VENV=1 or SKIP_VENV=1: skip venv detection, use system ruff directly
  if [[ "${CGW_NO_VENV:-0}" == "1" ]] || [[ "${SKIP_VENV:-0}" == "1" ]]; then
    # shellcheck disable=SC2034
    PYTHON_BIN=""
    # shellcheck disable=SC2034
    PYTHON_EXT=""
    return 0
  fi

  if [[ -d ".venv/Scripts" ]]; then
    # Windows (Git Bash, MSYS)
    # shellcheck disable=SC2034
    PYTHON_BIN=".venv/Scripts"
    # shellcheck disable=SC2034
    PYTHON_EXT=".exe"
  elif [[ -d ".venv/bin" ]]; then
    # Linux, macOS
    # shellcheck disable=SC2034
    PYTHON_BIN=".venv/bin"
    # shellcheck disable=SC2034
    PYTHON_EXT=""
  else
    # Fallback to system ruff
    if command -v ruff &>/dev/null; then
      # shellcheck disable=SC2034
      PYTHON_BIN=""
      # shellcheck disable=SC2034
      PYTHON_EXT=""
      return 0
    fi
    echo "[ERROR] Virtual environment not found (.venv/Scripts or .venv/bin) and ruff not in PATH" >&2
    return 1
  fi
  return 0
}

log_message() {
  local msg="$1"
  local log_path="$2"

  echo "$msg"
  echo "$msg" >>"$log_path"
}

log_section_start() {
  # Globals: _SECTION_START_TIMES (associative array, keyed by section name)
  # Arguments: section_name, log_path
  local section_name="$1"
  local log_path="$2"
  local time_str
  time_str=$(date +%H:%M:%S)
  _SECTION_START_TIMES["${section_name}"]=$(date +%s)

  {
    echo ""
    echo "========================================"
    echo "[${section_name}] Started: ${time_str}"
    echo "========================================"
  } | tee -a "${log_path}"
}

log_section_end() {
  # Globals: _SECTION_START_TIMES (associative array, keyed by section name)
  # Arguments: section_name, log_path, exit_code, [error_count]
  local section_name="$1"
  local log_path="$2"
  local exit_code="$3"
  # shellcheck disable=SC2034  # Reserved parameter for future error-count display; not yet used in output
  local error_count="${4:-0}"

  local time_str duration status
  time_str=$(date +%H:%M:%S)
  local end_time start_time
  end_time=$(date +%s)
  start_time="${_SECTION_START_TIMES[${section_name}]:-${end_time}}"
  duration=$((end_time - start_time))

  if [[ ${exit_code} -eq 0 ]]; then
    status="PASSED"
  else
    status="FAILED"
  fi

  echo "[${section_name}] Ended: ${time_str} (${duration}s) - ${status}" | tee -a "${log_path}"
}

run_tool_with_logging() {
  local tool_name="$1"
  local log_path="$2"
  shift 2

  log_section_start "$tool_name" "$log_path"

  TOOL_OUTPUT=$("$@" 2>&1)
  local exit_code=$?

  TOOL_ERROR_COUNT=$(echo "$TOOL_OUTPUT" | grep -cE "^[^:]+:[0-9]+:[0-9]+:" || true)

  if [[ -n "$TOOL_OUTPUT" ]]; then
    echo "$TOOL_OUTPUT" | tee -a "$log_path"
  fi

  log_section_end "$tool_name" "$log_path" "$exit_code" "$TOOL_ERROR_COUNT"

  return $exit_code
}

log_summary_table() {
  local log_path="$1"
  shift

  {
    echo ""
    echo "========================================"
    echo "[ERROR SUMMARY]"
    echo "========================================"
    printf "%-14s %-8s %-8s %s\n" "Tool" "Status" "Errors" "Duration"
    printf "%-14s %-8s %-8s %s\n" "----" "------" "------" "--------"

    local total_errors=0
    for result in "$@"; do
      IFS=':' read -r name status errors duration <<<"$result"
      printf "%-14s %-8s %-8s %s\n" "$name" "$status" "$errors" "${duration}s"
      ((total_errors += errors))
    done

    echo ""
    echo "Total: $total_errors errors"
  } | tee -a "$log_path"
}

run_git_with_logging() {
  local section_name="$1"
  local log_path="$2"
  shift 2

  log_section_start "$section_name" "$log_path"

  echo "Command: git $*" | tee -a "$log_path"

  GIT_OUTPUT=$(git "$@" 2>&1)
  GIT_EXIT_CODE=$?

  if [[ -n "$GIT_OUTPUT" ]]; then
    echo "$GIT_OUTPUT" | tee -a "$log_path"
  fi

  log_section_end "$section_name" "$log_path" "$GIT_EXIT_CODE"

  return $GIT_EXIT_CODE
}

validate_branch_pair() {
  local src="${1}" tgt="${2}"
  if ! git check-ref-format --branch "${src}" 2>/dev/null; then
    err "Invalid source branch name: '${src}'"
    exit 1
  fi
  if ! git check-ref-format --branch "${tgt}" 2>/dev/null; then
    err "Invalid target branch name: '${tgt}'"
    exit 1
  fi
  if [[ "${src}" == "${tgt}" ]]; then
    err "Source and target branch are the same: '${src}'"
    exit 1
  fi
  if ! git rev-parse --verify --quiet "refs/heads/${src}" >/dev/null 2>&1; then
    err "Source branch '${src}' does not exist locally"
    exit 1
  fi
  if ! git rev-parse --verify --quiet "refs/heads/${tgt}" >/dev/null 2>&1; then
    err "Target branch '${tgt}' does not exist locally"
    exit 1
  fi
}
