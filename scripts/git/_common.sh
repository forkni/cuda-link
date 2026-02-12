#!/usr/bin/env bash
# _common.sh - Shared utility functions for git automation scripts
# Purpose: Centralized functions for timestamp generation, logging, and lint configuration
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# Available Functions:
#   get_timestamp()         - Sets timestamp variable (yyyyMMdd_HHmmss format)
#   init_logging()          - Sets logfile, reportfile variables and creates log directory
#   get_lint_exclusions()   - Sets lint exclusion variables for all tools
#   log_message()           - Logs message to console and file

# ========================================
# Function: get_timestamp
# ========================================
# Purpose: Generate locale-independent timestamp
# Globals:
#   timestamp - Set to yyyyMMdd_HHmmss format
# Arguments:
#   None
# Outputs:
#   None (sets global timestamp variable)
# Returns:
#   0
get_timestamp() {
  timestamp=$(date +%Y%m%d_%H%M%S)
}

# ========================================
# Function: init_logging
# ========================================
# Purpose: Initialize logging infrastructure for script
# Globals:
#   logfile - Set to logs/script_name_timestamp.log
#   reportfile - Set to logs/script_name_analysis_timestamp.log
# Arguments:
#   $1 - script_name (e.g., "commit_enhanced", "check_lint")
# Outputs:
#   None (creates logs/ directory, sets global variables)
# Returns:
#   0
init_logging() {
  local script_name="$1"

  # Create logs directory if it doesn't exist
  if [[ ! -d "logs" ]]; then
    mkdir -p "logs"
  fi

  # Get timestamp
  get_timestamp

  # Set log file paths
  # shellcheck disable=SC2034  # Used by sourcing scripts
  logfile="logs/${script_name}_${timestamp}.log"
  # shellcheck disable=SC2034  # Used by sourcing scripts
  reportfile="logs/${script_name}_analysis_${timestamp}.log"
}

# ========================================
# Function: get_lint_exclusions
# ========================================
# Purpose: Set consistent exclusion patterns for all lint tools
# Globals:
#   RUFF_CHECK_EXCLUDE - Set to ruff check exclusion pattern (--extend-exclude)
#   RUFF_FORMAT_EXCLUDE - Set to ruff format exclusion pattern (--exclude)
#   MD_PATTERNS - Set to markdownlint patterns
# Arguments:
#   None
# Outputs:
#   None (sets global constant variables)
# Returns:
#   0
# Notes:
#   - tests/test_data: Contains intentional lint errors for testing
#   - _archive: Historical code not subject to current standards
#   - ruff check supports --extend-exclude, ruff format only supports --exclude
get_lint_exclusions() {
  # shellcheck disable=SC2034  # Used by sourcing scripts
  RUFF_CHECK_EXCLUDE="--extend-exclude tests/test_data --extend-exclude _archive"
  # shellcheck disable=SC2034  # Used by sourcing scripts
  RUFF_FORMAT_EXCLUDE="--exclude tests/test_data --exclude _archive"
  # shellcheck disable=SC2034  # Used by sourcing scripts
  MD_PATTERNS="**/*.md !**/tests/test_data/** !**/_archive/** !**/node_modules/** !**/.venv/** !**/benchmark_results/** !**/logs/**"
}

# ========================================
# Function: get_python_path
# ========================================
# Purpose: Get cross-platform Python virtual environment path
# Globals:
#   PYTHON_BIN - Set to platform-specific Python bin directory
#   PYTHON_EXT - Set to platform-specific executable extension
# Arguments:
#   None
# Outputs:
#   None (sets global variables)
# Returns:
#   0 on success, 1 if virtual environment not found
# Notes:
#   - Windows: .venv/Scripts/, .exe extension
#   - Linux/macOS: .venv/bin/, no extension
get_python_path() {
  if [[ -d ".venv/Scripts" ]]; then
    # Windows (Git Bash, MSYS)
    # shellcheck disable=SC2034  # Used by sourcing scripts
    PYTHON_BIN=".venv/Scripts"
    # shellcheck disable=SC2034  # Used by sourcing scripts
    PYTHON_EXT=".exe"
  elif [[ -d ".venv/bin" ]]; then
    # Linux, macOS
    # shellcheck disable=SC2034  # Used by sourcing scripts
    PYTHON_BIN=".venv/bin"
    # shellcheck disable=SC2034  # Used by sourcing scripts
    PYTHON_EXT=""
  else
    echo "[ERROR] Virtual environment not found (.venv/Scripts or .venv/bin)" >&2
    return 1
  fi
  return 0
}

# ========================================
# Function: log_message
# ========================================
# Purpose: Log message to both console and file
# Globals:
#   None
# Arguments:
#   $1 - message text
#   $2 - logfile path
# Outputs:
#   Writes to stdout and specified logfile
# Returns:
#   0
log_message() {
  local msg="$1"
  local log_path="$2"

  echo "$msg"
  echo "$msg" >> "$log_path"
}

# ========================================
# Function: log_section_start
# ========================================
# Purpose: Write section header with timestamp
# Globals:
#   SECTION_START_TIME - Set to epoch seconds for duration calculation
# Arguments:
#   $1 - section_name (e.g., "RUFF CHECK", "GIT COMMIT")
#   $2 - log_path (path to log file)
# Outputs:
#   Writes formatted section header to console and log file
# Returns:
#   0
log_section_start() {
  local section_name="$1"
  local log_path="$2"
  local time_str
  time_str=$(date +%H:%M:%S)
  SECTION_START_TIME=$(date +%s)

  {
    echo ""
    echo "========================================"
    echo "[$section_name] Started: $time_str"
    echo "========================================"
  } | tee -a "$log_path"
}

# ========================================
# Function: log_section_end
# ========================================
# Purpose: Write section footer with duration and status
# Globals:
#   SECTION_START_TIME - Used to calculate duration
# Arguments:
#   $1 - section_name (e.g., "RUFF CHECK", "GIT COMMIT")
#   $2 - log_path (path to log file)
#   $3 - exit_code (0 = passed, non-zero = failed)
#   $4 - error_count (optional, defaults to 0)
# Outputs:
#   Writes formatted section footer to console and log file
# Returns:
#   0
log_section_end() {
  local section_name="$1"
  local log_path="$2"
  local exit_code="$3"
  local error_count="${4:-0}"

  local time_str duration status
  time_str=$(date +%H:%M:%S)
  local end_time
  end_time=$(date +%s)
  duration=$((end_time - SECTION_START_TIME))

  if [[ $exit_code -eq 0 ]]; then
    status="PASSED"
  else
    status="FAILED"
  fi

  echo "[$section_name] Ended: $time_str (${duration}s) - $status" | tee -a "$log_path"
}

# ========================================
# Function: run_tool_with_logging
# ========================================
# Purpose: Run a tool, capture output, log with timestamps
# Globals:
#   SECTION_START_TIME - Set by log_section_start
#   TOOL_OUTPUT - Set to captured stdout+stderr from tool
#   TOOL_ERROR_COUNT - Set to parsed error count from output
# Arguments:
#   $1 - tool_name (e.g., "RUFF CHECK", "BLACK CHECK")
#   $2 - log_path (path to log file)
#   $3... - command and arguments to execute
# Outputs:
#   Writes tool output to console and log file with section headers
# Returns:
#   Exit code of the tool
run_tool_with_logging() {
  local tool_name="$1"
  local log_path="$2"
  shift 2

  log_section_start "$tool_name" "$log_path"

  # Run command, capture stdout+stderr
  TOOL_OUTPUT=$("$@" 2>&1)
  local exit_code=$?

  # Parse error count from output (matches pattern: file:line:col: error)
  TOOL_ERROR_COUNT=$(echo "$TOOL_OUTPUT" | grep -cE "^[^:]+:[0-9]+:[0-9]+:" || echo "0")

  # Log output if any
  if [[ -n "$TOOL_OUTPUT" ]]; then
    echo "$TOOL_OUTPUT" | tee -a "$log_path"
  fi

  log_section_end "$tool_name" "$log_path" "$exit_code" "$TOOL_ERROR_COUNT"

  return $exit_code
}

# ========================================
# Function: log_summary_table
# ========================================
# Purpose: Write formatted summary table at end of log
# Globals:
#   None
# Arguments:
#   $1 - log_path (path to log file)
#   $2... - tool results in format "name:status:errors:duration"
#           Example: "Ruff:FAILED:2:1.2"
# Outputs:
#   Writes formatted table to console and log file
# Returns:
#   0
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
      IFS=':' read -r name status errors duration <<< "$result"
      printf "%-14s %-8s %-8s %s\n" "$name" "$status" "$errors" "${duration}s"
      (( total_errors += errors ))
    done

    echo ""
    echo "Total: $total_errors errors"
  } | tee -a "$log_path"
}

# ========================================
# Function: run_git_with_logging
# ========================================
# Purpose: Run git command with full output capture to log
# Globals:
#   GIT_OUTPUT - Set to captured stdout+stderr from git command
#   GIT_EXIT_CODE - Set to git command exit code
# Arguments:
#   $1 - section_name (e.g., "GIT CHECKOUT", "GIT MERGE")
#   $2 - log_path (path to log file)
#   $3... - git command arguments (e.g., "checkout", "main")
# Outputs:
#   Writes git command and output to console and log file with section headers
# Returns:
#   Exit code of the git command
run_git_with_logging() {
  local section_name="$1"
  local log_path="$2"
  shift 2

  log_section_start "$section_name" "$log_path"

  echo "Command: git $*" | tee -a "$log_path"

  # Run command, capture output
  GIT_OUTPUT=$(git "$@" 2>&1)
  GIT_EXIT_CODE=$?

  if [[ -n "$GIT_OUTPUT" ]]; then
    echo "$GIT_OUTPUT" | tee -a "$log_path"
  fi

  log_section_end "$section_name" "$log_path" "$GIT_EXIT_CODE"

  return $GIT_EXIT_CODE
}
