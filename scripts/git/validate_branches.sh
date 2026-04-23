#!/usr/bin/env bash
# validate_branches.sh - Pre-merge branch validation
# Purpose: Validate branch state before a merge operation
# Usage: ./scripts/git/validate_branches.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR       - Directory containing this script
#   PROJECT_ROOT     - Auto-detected git repo root (set by _config.sh)
#   logfile          - Set by init_logging
#   CGW_SOURCE_BRANCH - Source branch name (default: development)
#   CGW_TARGET_BRANCH - Target branch name (default: main)
# Returns:
#   0 on validation pass, 1 on failure

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

main() {
  if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    echo "Usage: ./scripts/git/validate_branches.sh"
    echo ""
    echo "Validate branch state before a merge:"
    echo "  - Checks current branch (warns if not source/target branch)"
    echo "  - Fails if uncommitted changes exist"
    echo "  - Warns about untracked files"
    echo "  - Reports commits ahead/behind between source and target branches"
    echo ""
    echo "Options:"
    echo "  -h, --help   Show this help"
    echo ""
    echo "Configuration (via .cgw.conf or env vars):"
    echo "  CGW_SOURCE_BRANCH   Source branch name (default: development)"
    echo "  CGW_TARGET_BRANCH   Target branch name (default: main)"
    exit 0
  fi

  init_logging "validate_branches"

  local script_start
  script_start=$(date +%s)

  {
    echo "========================================="
    echo "Branch Validation Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } >"$logfile"

  echo "=== Branch Validation ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  cd "${PROJECT_ROOT}" || {
    err "Cannot find project root"
    exit 1
  }

  local validation_failed=0

  # [CHECK 1] Current branch
  log_section_start "BRANCH CHECK" "$logfile"

  local current_branch
  current_branch=$(git rev-parse --abbrev-ref HEAD 2>&1)
  local branch_check_exit=$?

  echo "Current branch: $current_branch" | tee -a "$logfile"

  # Verify source branch exists in this repository
  if ! git rev-parse --verify "${CGW_SOURCE_BRANCH}" >/dev/null 2>&1; then
    echo "  ERROR: Source branch '${CGW_SOURCE_BRANCH}' does not exist" | tee -a "$logfile"
    validation_failed=1
  fi

  if [[ $branch_check_exit -ne 0 ]]; then
    echo "  Failed to get current branch" | tee -a "$logfile"
    validation_failed=1
  elif [[ "$current_branch" != "${CGW_SOURCE_BRANCH}" && "$current_branch" != "${CGW_TARGET_BRANCH}" ]]; then
    echo "  WARNING: Not on '${CGW_SOURCE_BRANCH}' or '${CGW_TARGET_BRANCH}' branch" | tee -a "$logfile"
    echo "  Current: $current_branch" | tee -a "$logfile"
  else
    echo "  On valid branch: $current_branch" | tee -a "$logfile"
  fi

  log_section_end "BRANCH CHECK" "$logfile" "$validation_failed"

  # [CHECK 2] Uncommitted changes + untracked files
  echo "" | tee -a "$logfile"
  log_section_start "UNCOMMITTED CHANGES CHECK" "$logfile"

  local uncommitted_check=0
  if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo "  Uncommitted changes detected:" | tee -a "$logfile"
    git status --short | tee -a "$logfile"
    uncommitted_check=1
    validation_failed=1
  else
    echo "  No uncommitted changes" | tee -a "$logfile"
  fi

  # Check for untracked files (git diff-index only checks tracked files)
  local untracked_files
  untracked_files=$(git ls-files --others --exclude-standard)
  if [[ -n "${untracked_files}" ]]; then
    echo "  Untracked files detected (may be missing from commit):" | tee -a "$logfile"
    echo "${untracked_files}" | tee -a "$logfile"
    echo "  Use 'git add <file>' to stage, or add to .gitignore" | tee -a "$logfile"
    # Warning only -- untracked files don't affect merge safety
  fi

  log_section_end "UNCOMMITTED CHANGES CHECK" "$logfile" "$uncommitted_check"

  # [CHECK 3] Branch relationship
  echo "" | tee -a "$logfile"
  log_section_start "BRANCH RELATIONSHIP CHECK" "$logfile"

  local source_ahead target_ahead
  source_ahead=$(git rev-list --count "${CGW_TARGET_BRANCH}..${CGW_SOURCE_BRANCH}" 2>/dev/null || echo "unknown")
  target_ahead=$(git rev-list --count "${CGW_SOURCE_BRANCH}..${CGW_TARGET_BRANCH}" 2>/dev/null || echo "unknown")

  echo "${CGW_SOURCE_BRANCH} ahead of ${CGW_TARGET_BRANCH}: $source_ahead commits" | tee -a "$logfile"
  echo "${CGW_TARGET_BRANCH} ahead of ${CGW_SOURCE_BRANCH}: $target_ahead commits" | tee -a "$logfile"

  if [[ "$source_ahead" != "unknown" ]] && ((source_ahead == 0)) && [[ "${current_branch}" == "${CGW_SOURCE_BRANCH}" ]]; then
    echo "  Warning: ${CGW_SOURCE_BRANCH} has no new commits vs ${CGW_TARGET_BRANCH}" | tee -a "$logfile"
  fi

  if [[ "$target_ahead" != "unknown" ]] && ((target_ahead > 0)) && [[ "${current_branch}" == "${CGW_SOURCE_BRANCH}" ]]; then
    echo "  Warning: ${CGW_TARGET_BRANCH} is ahead -- consider merging ${CGW_TARGET_BRANCH} into ${CGW_SOURCE_BRANCH}" | tee -a "$logfile"
  fi

  log_section_end "BRANCH RELATIONSHIP CHECK" "$logfile" "0"

  # Summary
  echo "" | tee -a "$logfile"
  {
    echo "========================================"
    echo "[VALIDATION SUMMARY]"
    echo "========================================"
  } | tee -a "$logfile"

  local script_end total_duration
  script_end=$(date +%s)
  total_duration=$((script_end - script_start))

  if ((validation_failed == 0)); then
    echo "Branch validation passed" | tee -a "$logfile"
  else
    echo "Branch validation failed" | tee -a "$logfile"
  fi

  {
    echo ""
    echo "End Time: $(date)"
    echo "Total Duration: ${total_duration}s"
  } | tee -a "$logfile"

  echo ""
  echo "Full log: $logfile"

  exit $validation_failed
}

main "$@"
