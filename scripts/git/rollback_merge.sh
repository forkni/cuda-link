#!/usr/bin/env bash
# rollback_merge.sh - Emergency rollback for merge operations
# Purpose: Revert main branch to pre-merge state safely
# Usage: ./scripts/git/rollback_merge.sh

set -u

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/_common.sh"

init_logging "rollback_merge"

main() {
  # Write log header
  {
    echo "========================================="
    echo "Rollback Merge Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } > "$logfile"

  echo "=== Emergency Merge Rollback ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  # Ensure execution from project root
  cd "${PROJECT_ROOT}" || {
    echo "[ERROR] Cannot find project root" | tee -a "$logfile"
    exit 1
  }

  # [1/5] Verify current branch
  log_section_start "BRANCH VERIFICATION" "$logfile"

  local current_branch
  current_branch=$(git branch --show-current 2>&1)
  echo "Current branch: ${current_branch}" | tee -a "$logfile"

  if [[ "${current_branch}" != "main" ]]; then
    echo "" | tee -a "$logfile"
    echo "✗ ERROR: Not on main branch" | tee -a "$logfile"
    echo "This script should only be run from main branch" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "Current branch: ${current_branch}" | tee -a "$logfile"
    echo "Expected: main" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "Please checkout main first: git checkout main"
    log_section_end "BRANCH VERIFICATION" "$logfile" "1"
    exit 1
  fi
  echo "✓ On main branch" | tee -a "$logfile"
  log_section_end "BRANCH VERIFICATION" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [2/5] Check for uncommitted changes
  log_section_start "UNCOMMITTED CHANGES CHECK" "$logfile"

  if ! git diff-index --quiet HEAD --; then
    echo "⚠ WARNING: Uncommitted changes detected" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    git status --short | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "These changes will be LOST during rollback!" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    read -r -p "Continue anyway? (yes/no): " continue_choice
    if [[ "${continue_choice}" != "yes" ]]; then
      echo "" | tee -a "$logfile"
      echo "Rollback cancelled" | tee -a "$logfile"
      echo "Please commit or stash changes first"
      log_section_end "UNCOMMITTED CHANGES CHECK" "$logfile" "1"
      exit 1
    fi
  else
    echo "✓ No uncommitted changes" | tee -a "$logfile"
  fi
  log_section_end "UNCOMMITTED CHANGES CHECK" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [3/5] Find rollback target
  log_section_start "FIND ROLLBACK TARGET" "$logfile"

  # Look for pre-merge backup tags
  if git tag -l "pre-merge-backup-*" >/dev/null 2>&1; then
    echo "Available backup tags:" | tee -a "$logfile"
    git tag -l "pre-merge-backup-*" | sort -r | head -5 | tee -a "$logfile"
    echo "" | tee -a "$logfile"
  fi

  # Show recent commits
  echo "Recent commits:" | tee -a "$logfile"
  git log --oneline -5 | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  # Get latest merge commit
  local latest_merge
  latest_merge=$(git log --merges --oneline -1)
  if [[ -n "${latest_merge}" ]]; then
    echo "Latest merge commit: ${latest_merge}" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
  fi

  log_section_end "FIND ROLLBACK TARGET" "$logfile" "0"

  # [4/5] Choose rollback method
  echo "[4/5] Choose rollback method:"
  echo ""
  echo "Available options:"
  echo "  1. Rollback to latest pre-merge backup tag (recommended)"
  echo "  2. Rollback to commit before latest merge (HEAD~1)"
  echo "  3. Rollback to specific commit hash"
  echo "  4. Cancel rollback"
  echo ""

  read -r -p "Select option (1-4): " rollback_choice

  local rollback_target=""

  case "${rollback_choice}" in
    1)
      # Find latest backup tag
      rollback_target=$(git tag -l "pre-merge-backup-*" | sort -r | head -1)
      if [[ -z "${rollback_target}" ]]; then
        echo "" | tee -a "$logfile"
        echo "✗ ERROR: No backup tags found" | tee -a "$logfile"
        echo "Please use option 2 or 3"
        exit 1
      fi
      echo ""
      echo "Rollback target: ${rollback_target}"
      ;;
    2)
      rollback_target="HEAD~1"
      echo ""
      echo "Rollback target: HEAD~1 (previous commit)"
      ;;
    3)
      echo ""
      read -r -p "Enter commit hash: " rollback_target
      echo ""
      # Verify commit exists
      if ! git rev-parse "${rollback_target}" >/dev/null 2>&1; then
        echo "✗ ERROR: Invalid commit hash: ${rollback_target}" | tee -a "$logfile"
        exit 1
      fi
      echo "Rollback target: ${rollback_target}"
      ;;
    4)
      echo "" | tee -a "$logfile"
      echo "Rollback cancelled" | tee -a "$logfile"
      exit 0
      ;;
    *)
      echo "" | tee -a "$logfile"
      echo "✗ ERROR: Invalid choice: ${rollback_choice}" | tee -a "$logfile"
      exit 1
      ;;
  esac

  # [5/5] Execute rollback
  echo "" | tee -a "$logfile"
  echo "⚠ WARNING: This will permanently reset main branch to:" | tee -a "$logfile"
  git log "${rollback_target}" --oneline -1 | tee -a "$logfile"
  echo "" | tee -a "$logfile"
  echo "All commits after this point will be lost!" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  read -r -p "Type 'ROLLBACK' to confirm: " confirm
  if [[ "${confirm}" != "ROLLBACK" ]]; then
    echo "" | tee -a "$logfile"
    echo "Rollback cancelled" | tee -a "$logfile"
    exit 0
  fi

  log_section_start "GIT RESET" "$logfile"

  if run_git_with_logging "GIT RESET HARD" "$logfile" reset --hard "${rollback_target}"; then
     log_section_end "GIT RESET" "$logfile" "0"
     echo "" | tee -a "$logfile"
     {
       echo "========================================"
       echo "[ROLLBACK SUMMARY]"
       echo "========================================"
     } | tee -a "$logfile"
     echo "✓ ROLLBACK SUCCESSFUL" | tee -a "$logfile"
     echo "" | tee -a "$logfile"
     echo "Summary:" | tee -a "$logfile"
     git log --oneline -1 | while read -r line; do echo "  Current HEAD: $line" | tee -a "$logfile"; done
     echo "" | tee -a "$logfile"
     echo "Next steps:" | tee -a "$logfile"
     echo "  1. Verify rollback: git log --oneline -5" | tee -a "$logfile"
     echo "  2. If correct, force push: git push origin main --force-with-lease" | tee -a "$logfile"
     echo "  3. If issues, contact maintainer" | tee -a "$logfile"
     echo "" | tee -a "$logfile"
     echo "  ⚠ WARNING: Force push will rewrite remote history!" | tee -a "$logfile"
     {
       echo ""
       echo "End Time: $(date)"
     } | tee -a "$logfile"
     echo "" | tee -a "$logfile"
     echo "Full log: $logfile"
  else
     log_section_end "GIT RESET" "$logfile" "1"
     echo "" | tee -a "$logfile"
     echo "✗ Rollback failed" | tee -a "$logfile"
     echo "Please manually reset: git reset --hard ${rollback_target}"
     exit 1
  fi
}

main "$@"
