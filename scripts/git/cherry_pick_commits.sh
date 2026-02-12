#!/usr/bin/env bash
# cherry_pick_commits.sh - Cherry-pick specific commits from development to main
# Purpose: Cherry-pick commits with validation and automatic backup
# Usage: ./scripts/git/cherry_pick_commits.sh

set -u

# Get directory
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source common functions
source "${SCRIPT_DIR}/_common.sh"

init_logging "cherry_pick_commits"

main() {
  # Write log header
  {
    echo "========================================="
    echo "Cherry-Pick Commits Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } > "$logfile"

  echo "=== Cherry-Pick Commits: development → main ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  # Ensure execution from project root
  cd "${PROJECT_ROOT}" || {
    echo "[ERROR] Cannot find project root" | tee -a "$logfile"
    exit 1
  }

  # [1/6] Run validation
  log_section_start "PRE-CHERRY-PICK VALIDATION" "$logfile"

  if [[ -f "scripts/git/validate_branches.sh" ]]; then
    if ! bash "scripts/git/validate_branches.sh" >> "$logfile" 2>&1; then
      echo "✗ Validation failed - aborting cherry-pick" | tee -a "$logfile"
      log_section_end "PRE-CHERRY-PICK VALIDATION" "$logfile" "1"
      echo "Please fix validation errors before retrying"
      exit 1
    fi
  else
    echo "⚠ scripts/git/validate_branches.sh not found, skipping validation" | tee -a "$logfile"
  fi

  echo "✓ Pre-cherry-pick validation passed" | tee -a "$logfile"
  log_section_end "PRE-CHERRY-PICK VALIDATION" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [2/6] Store current branch and checkout main
  log_section_start "GIT CHECKOUT MAIN" "$logfile"

  local original_branch
  original_branch=$(git branch --show-current)

  if [[ -z "${original_branch}" ]]; then
    echo "✗ Failed to determine current branch" | tee -a "$logfile"
    log_section_end "GIT CHECKOUT MAIN" "$logfile" "1"
    exit 1
  fi

  echo "Current branch: ${original_branch}" | tee -a "$logfile"

  if ! run_git_with_logging "GIT CHECKOUT" "$logfile" checkout main; then
    echo "✗ Failed to checkout main branch" | tee -a "$logfile"
    exit 1
  fi

  log_section_end "GIT CHECKOUT MAIN" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [3/6] Show recent development commits
  echo "[3/6] Recent commits on development branch:"
  echo "===================================="
  git log development --oneline -20 --no-merges
  echo "===================================="
  echo ""

  # [4/6] Get commit hash to cherry-pick
  echo "[4/6] Select commit to cherry-pick..."
  echo ""
  read -r -p "Enter commit hash (or 'cancel' to abort): " commit_hash

  if [[ "${commit_hash}" == "cancel" ]]; then
    echo ""
    log_message "Cherry-pick cancelled" "${logfile}"
    git checkout "${original_branch}"
    exit 0
  fi

  # Verify commit exists
  if ! git rev-parse "${commit_hash}" >/dev/null 2>&1; then
    log_message "✗ ERROR: Invalid commit hash: ${commit_hash}" "${logfile}"
    git checkout "${original_branch}"
    exit 1
  fi

  echo ""
  echo "Selected commit:"
  git log "${commit_hash}" --oneline -1
  echo ""
  echo "Commit details:"
  git show "${commit_hash}" --stat
  echo ""

  # Check if commit modifies excluded files
  local has_excluded_files=0
  
  if git show "${commit_hash}" --name-only --format="" | grep -q "^tests/"; then has_excluded_files=1; fi
  if git show "${commit_hash}" --name-only --format="" | grep -q "^docs/TESTING_GUIDE.md"; then has_excluded_files=1; fi
  if git show "${commit_hash}" --name-only --format="" | grep -q "pytest.ini"; then has_excluded_files=1; fi

  if [[ ${has_excluded_files} -eq 1 ]]; then
    echo "⚠ WARNING: This commit modifies development-only files"
    echo "These files should not be on main branch:"
    git show "${commit_hash}" --name-only --format="" | grep -E "^tests/|^docs/TESTING_GUIDE.md|pytest.ini"
    echo ""
    read -r -p "Continue anyway? (yes/no): " continue_choice
    if [[ "${continue_choice}" != "yes" ]]; then
      echo ""
      log_message "Cherry-pick cancelled" "${logfile}"
      git checkout "${original_branch}"
      exit 0
    fi
  fi

  # [5/6] Create backup tag
  log_section_start "CREATE BACKUP TAG" "$logfile"

  # Re-using timestamp from init_logging if available, else generating new one
  if [[ -z "${timestamp:-}" ]]; then
    get_timestamp
  fi
  local backup_tag="pre-cherry-pick-${timestamp}-$$"

  if git tag "${backup_tag}" >> "$logfile" 2>&1; then
    echo "✓ Created backup tag: ${backup_tag}" | tee -a "$logfile"
    echo "  Note: Backup tags are local. Use 'git push --tags' to push to remote." | tee -a "$logfile"
    log_section_end "CREATE BACKUP TAG" "$logfile" "0"
  else
    echo "⚠ Warning: Could not create backup tag" | tee -a "$logfile"
    log_section_end "CREATE BACKUP TAG" "$logfile" "1"
  fi
  echo "" | tee -a "$logfile"

  # [6/6] Cherry-pick the commit
  log_section_start "GIT CHERRY-PICK" "$logfile"

  if run_git_with_logging "GIT CHERRY-PICK COMMIT" "$logfile" cherry-pick "${commit_hash}"; then
    log_section_end "GIT CHERRY-PICK" "$logfile" "0"
    echo "" | tee -a "$logfile"
    {
      echo "========================================"
      echo "[CHERRY-PICK SUMMARY]"
      echo "========================================"
    } | tee -a "$logfile"
    echo "✓ CHERRY-PICK SUCCESSFUL" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "Summary:" | tee -a "$logfile"
    git log -1 --oneline | while read -r line; do echo "  Cherry-picked: $line" | tee -a "$logfile"; done
    echo "  Original commit: ${commit_hash}" | tee -a "$logfile"
    echo "  Backup tag: ${backup_tag}" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "Next steps:" | tee -a "$logfile"
    echo "  1. Review changes: git show HEAD" | tee -a "$logfile"
    echo "  2. Test changes: [run your tests]" | tee -a "$logfile"
    echo "  3. Push to remote: git push origin main" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "  If issues found:" | tee -a "$logfile"
    echo "  - Rollback: git reset --hard ${backup_tag}" | tee -a "$logfile"
    echo "  - Or use: scripts/git/rollback_merge.sh" | tee -a "$logfile"
    {
      echo ""
      echo "End Time: $(date)"
    } | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "Full log: $logfile"
  else
    log_section_end "GIT CHERRY-PICK" "$logfile" "1"
    echo "" | tee -a "$logfile"
    echo "⚠ Cherry-pick conflicts detected" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    git status | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "Please resolve conflicts manually:"
    echo "  1. Edit conflicted files"
    echo "  2. git add <resolved files>"
    echo "  3. git cherry-pick --continue"
    echo ""
    echo "Or abort the cherry-pick:"
    echo "  git cherry-pick --abort"
    echo "  git checkout ${original_branch}"
    echo ""
    echo "Backup available: git reset --hard ${backup_tag}"
    exit 1
  fi
}

main "$@"
