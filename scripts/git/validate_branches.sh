#!/usr/bin/env bash
# Branch validation script for Git Bash / Linux / macOS
# Windows cmd.exe users: use validate_branches.bat instead

set -u          # Treat unset variables as errors
set -o pipefail # Catch errors in pipelines
# Note: NOT using set -e to allow validation to continue and summarize all failures

# Get directory of this script
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source common functions
source "${SCRIPT_DIR}/_common.sh"

main() {
  # Initialize logging
  init_logging "validate_branches"

  # Track script start time
  local script_start
  script_start=$(date +%s)

  # Write log header
  {
    echo "========================================="
    echo "Branch Validation Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } > "$logfile"

  echo "=== Branch Validation ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  # Change to project root
  cd "${PROJECT_ROOT}" || {
    echo "[ERROR] Cannot find project root" >&2 | tee -a "$logfile"
    exit 1
  }

  local validation_failed=0

  # [CHECK 1] Current branch validation
  log_section_start "BRANCH CHECK" "$logfile"

  local current_branch
  current_branch=$(git rev-parse --abbrev-ref HEAD 2>&1)
  local branch_check_exit=$?

  echo "Current branch: $current_branch" | tee -a "$logfile"

  if [[ $branch_check_exit -ne 0 ]]; then
    echo "❌ Failed to get current branch" | tee -a "$logfile"
    validation_failed=1
  elif [[ "$current_branch" != "development" && "$current_branch" != "main" ]]; then
    echo "❌ Must be on 'development' or 'main' branch" | tee -a "$logfile"
    echo "   Current: $current_branch" | tee -a "$logfile"
    validation_failed=1
  else
    echo "✓ On valid branch: $current_branch" | tee -a "$logfile"
  fi

  log_section_end "BRANCH CHECK" "$logfile" "$validation_failed"

  # [CHECK 2] Uncommitted changes check
  echo "" | tee -a "$logfile"
  log_section_start "UNCOMMITTED CHANGES CHECK" "$logfile"

  local diff_output uncommitted_check=0
  diff_output=$(git diff-index --quiet HEAD -- 2>&1 || echo "changes_detected")

  if [[ "$diff_output" == "changes_detected" ]]; then
    echo "❌ Uncommitted changes detected:" | tee -a "$logfile"
    git status --short | tee -a "$logfile"
    uncommitted_check=1
    validation_failed=1
  else
    echo "✓ No uncommitted changes" | tee -a "$logfile"
  fi

  log_section_end "UNCOMMITTED CHANGES CHECK" "$logfile" "$uncommitted_check"

  # [CHECK 3] Branch relationship check
  echo "" | tee -a "$logfile"
  log_section_start "BRANCH RELATIONSHIP CHECK" "$logfile"

  local dev_ahead main_ahead
  dev_ahead=$(git rev-list --count main..development 2>/dev/null || echo "0")
  main_ahead=$(git rev-list --count development..main 2>/dev/null || echo "0")

  echo "Development ahead of main: $dev_ahead commits" | tee -a "$logfile"
  echo "Main ahead of development: $main_ahead commits" | tee -a "$logfile"

  if (( dev_ahead == 0 )) && [[ "${current_branch}" == "development" ]]; then
    echo "⚠️  Warning: Development branch has no new commits vs main" | tee -a "$logfile"
  fi

  if (( main_ahead > 0 )) && [[ "${current_branch}" == "development" ]]; then
    echo "⚠️  Warning: Main branch is ahead - consider merging main into development" | tee -a "$logfile"
  fi

  log_section_end "BRANCH RELATIONSHIP CHECK" "$logfile" "0"

  # Final status message
  echo "" | tee -a "$logfile"
  {
    echo "========================================"
    echo "[VALIDATION SUMMARY]"
    echo "========================================"
  } | tee -a "$logfile"

  # Calculate total duration
  local script_end total_duration
  script_end=$(date +%s)
  total_duration=$((script_end - script_start))

  if (( validation_failed == 0 )); then
    echo "✅ Branch validation passed" | tee -a "$logfile"
  else
    echo "❌ Branch validation failed" | tee -a "$logfile"
  fi

  {
    echo ""
    echo "End Time: $(date)"
    echo "Total Duration: ${total_duration}s"
  } | tee -a "$logfile"

  echo "" | tee -a "$logfile"
  echo "Full log: $logfile"

  exit $validation_failed
}

main "$@"
