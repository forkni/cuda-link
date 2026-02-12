#!/usr/bin/env bash
# merge_docs.sh - Documentation-only merge from development to main
# Purpose: Merge only documentation changes while ignoring code changes
# Usage: ./scripts/git/merge_docs.sh

set -u

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/_common.sh"

init_logging "merge_docs"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Cleanup function for interruption safety
original_branch=""
did_mutate_worktree=0

cleanup() {
    # Restore worktree if we mutated it
    if [[ ${did_mutate_worktree} -eq 1 ]]; then
        echo "" | tee -a "$logfile"
        echo "⚠ Interrupted - restoring original state..." | tee -a "$logfile"
        # Restore staged and working tree changes (prefer restore over reset --hard)
        git restore --staged --worktree -- . 2>/dev/null || git reset --hard HEAD 2>/dev/null || true
    fi
    # Return to original branch if we left it
    local current_branch
    current_branch=$(git branch --show-current 2>/dev/null)
    if [[ -n "$original_branch" ]] && [[ "$original_branch" != "$current_branch" ]]; then
        git checkout "$original_branch" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

main() {
  # Write log header
  {
    echo "========================================="
    echo "Documentation Merge Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } > "$logfile"

  echo "=== Documentation Merge: development → main ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  # Ensure execution from project root
  cd "${PROJECT_ROOT}" || {
    echo "[ERROR] Cannot find project root" | tee -a "$logfile"
    exit 1
  }

  # [1/7] Run validation
  log_section_start "PRE-MERGE VALIDATION" "$logfile"

  if [[ -f "scripts/git/validate_branches.sh" ]]; then
    if ! bash "scripts/git/validate_branches.sh" >> "$logfile" 2>&1; then
      echo "✗ Validation failed - aborting documentation merge" | tee -a "$logfile"
      log_section_end "PRE-MERGE VALIDATION" "$logfile" "1"
      echo "Please fix validation errors before retrying"
      exit 1
    fi
  fi

  echo "✓ Pre-merge validation passed" | tee -a "$logfile"
  log_section_end "PRE-MERGE VALIDATION" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [2/7] Store current branch and checkout main
  log_section_start "GIT CHECKOUT MAIN" "$logfile"

  local original_branch
  original_branch=$(git branch --show-current)
  echo "Current branch: ${original_branch}" | tee -a "$logfile"

  if ! run_git_with_logging "GIT CHECKOUT" "$logfile" checkout main; then
    echo "✗ Failed to checkout main branch" | tee -a "$logfile"
    exit 1
  fi

  log_section_end "GIT CHECKOUT MAIN" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [3/7] Check for documentation changes
  echo "[3/7] Checking for documentation changes..."

  if [[ -z "$(git diff --name-only main development -- docs/)" ]]; then
      log_message "⚠ No documentation changes found between main and development" "${logfile}"
      echo ""
      read -r -p "Continue anyway? (yes/no): " continue_choice
      if [[ "${continue_choice}" != "yes" ]]; then
        echo ""
        log_message "Documentation merge cancelled" "${logfile}"
        git checkout "${original_branch}"
        exit 0
      fi
  fi

  echo "Documentation files to be merged:"
  git diff --name-only main development -- docs/
  echo ""

  # [4/7] Check for non-documentation changes
  echo "[4/7] Checking for code changes..."
  # shellcheck disable=SC2034  # Reserved for future use
  local has_code_changes=0
  local non_docs_count
  non_docs_count=$(git diff --name-only main development | grep -v "^docs/" | wc -l)

  if [[ "${non_docs_count}" -gt 0 ]]; then
    echo "⚠ WARNING: Non-documentation changes detected"
    echo ""
    echo "This merge will ONLY include docs/ changes."
    echo "Other changes will remain on development branch."
    echo ""
    git diff --name-only main development | grep -v "^docs/"
    echo ""
    read -r -p "Proceed with docs-only merge? (yes/no): " confirm
    if [[ "${confirm}" != "yes" ]]; then
      echo ""
      log_message "Documentation merge cancelled" "${logfile}"
      git checkout "${original_branch}"
      exit 0
    fi
  else
    log_message "✓ No code changes detected (docs-only merge)" "${logfile}"
  fi
  echo ""

  # [5/7] Create pre-merge backup tag
  log_section_start "CREATE BACKUP TAG" "$logfile"

  if [[ -z "${timestamp:-}" ]]; then get_timestamp; fi
  local backup_tag="pre-docs-merge-${timestamp}"

  if git tag "${backup_tag}" >> "$logfile" 2>&1; then
    echo "✓ Created backup tag: ${backup_tag}" | tee -a "$logfile"
    log_section_end "CREATE BACKUP TAG" "$logfile" "0"
  else
    echo "⚠ Warning: Could not create backup tag" | tee -a "$logfile"
    log_section_end "CREATE BACKUP TAG" "$logfile" "1"
  fi
  echo "" | tee -a "$logfile"

  # [6/7] Merge documentation changes
  log_section_start "MERGE DOCUMENTATION" "$logfile"

  # Mark that we're about to mutate the worktree (for cleanup trap)
  did_mutate_worktree=1

  # Strategy: Checkout docs/ from development branch
  if ! run_git_with_logging "GIT CHECKOUT DOCS" "$logfile" checkout development -- docs/; then
    echo "✗ Failed to checkout documentation from development" | tee -a "$logfile"
    git checkout "${original_branch}" >> "$logfile" 2>&1
    exit 1
  fi

  # Check if there are changes staged
  if git diff --cached --quiet; then
    echo "" | tee -a "$logfile"
    echo "⚠ No documentation changes to merge (docs already in sync)" | tee -a "$logfile"
    log_section_end "MERGE DOCUMENTATION" "$logfile" "0"
    git checkout "${original_branch}" >> "$logfile" 2>&1
    exit 0
  fi

  echo "✓ Documentation changes staged" | tee -a "$logfile"
  echo "Staged files:" | tee -a "$logfile"
  git diff --cached --name-only | tee -a "$logfile"
  log_section_end "MERGE DOCUMENTATION" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [7/7] Commit the merge
  log_section_start "GIT COMMIT" "$logfile"

  if run_git_with_logging "GIT COMMIT DOCS" "$logfile" commit -m "docs: Sync documentation from development" -m "- Updated docs/ directory from development branch" -m "- Docs-only update (no code changes)" -m "- Backup tag: ${backup_tag}"; then
     log_section_end "GIT COMMIT" "$logfile" "0"
     echo "" | tee -a "$logfile"
     {
       echo "========================================"
       echo "[DOCS MERGE SUMMARY]"
       echo "========================================"
     } | tee -a "$logfile"
     echo "✓ DOCUMENTATION MERGE SUCCESSFUL" | tee -a "$logfile"
     echo "" | tee -a "$logfile"
     echo "Summary:" | tee -a "$logfile"
     git log -1 --oneline | while read -r line; do echo "  Latest commit: $line" | tee -a "$logfile"; done
     echo "  Backup tag: ${backup_tag}" | tee -a "$logfile"
     echo "" | tee -a "$logfile"
     echo "Merged files:" | tee -a "$logfile"
     git diff --name-only HEAD~1 HEAD | tee -a "$logfile"
     echo "" | tee -a "$logfile"
     echo "Next steps:" | tee -a "$logfile"
     echo "  1. Review changes: git show HEAD" | tee -a "$logfile"
     echo "  2. Test documentation: [review rendered docs]" | tee -a "$logfile"
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
     log_section_end "GIT COMMIT" "$logfile" "1"
     echo "" | tee -a "$logfile"
     echo "✗ Failed to commit documentation merge" | tee -a "$logfile"
     echo ""
     echo "To abort: git reset --hard HEAD"
     exit 1
  fi
}

main "$@"
