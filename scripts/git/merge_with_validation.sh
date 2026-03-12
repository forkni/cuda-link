#!/usr/bin/env bash
# merge_with_validation.sh - Safe merge from development to master with automatic conflict resolution
# Purpose: Merge development to master with validation and auto-resolution of known conflict patterns
# Usage: ./scripts/git/merge_with_validation.sh [--non-interactive]

set -u

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/_common.sh"

init_logging "merge_with_validation"

# ============================================================================
# CONFIGURABLE: Documentation file patterns (bash extended regex)
# Modify this pattern to match your project's documentation naming conventions
# ============================================================================
readonly ALLOWED_DOCS_PATTERN='^(ADVANCED_FEATURES_GUIDE\.md|BENCHMARKS\.md|CLAUDE_MD_TEMPLATE\.md|claude_code_config\.md|DOCUMENTATION_INDEX\.md|GIT_WORKFLOW\.md|HYBRID_SEARCH_CONFIGURATION_GUIDE\.md|INSTALLATION_GUIDE\.md|MCP_TOOLS_REFERENCE\.md|MODEL_MIGRATION_GUIDE\.md|PYTORCH_COMPATIBILITY\.md|VERSION_HISTORY\.md|.*_GUIDE\.md|.*_REFERENCE\.md|README\.md)$'

# Environment variable override for CI/automation
DOCS_PATTERN="${CLAUDE_DOCS_PATTERN:-$ALLOWED_DOCS_PATTERN}"

master() {
  # Write log header
  {
    echo "========================================="
    echo "Merge With Validation Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } > "$logfile"

  echo "=== Safe Merge: development → master ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"
  echo "📋 Workflow Log: ${logfile}" | tee -a "$logfile"
  echo "" | tee -a "$logfile"
  
  # Ensure execution from project root
  cd "${PROJECT_ROOT}" || {
    echo "[ERROR] Cannot find project root" | tee -a "$logfile"
    exit 1
  }

  local non_interactive=0
  if [[ "${1:-}" == "--non-interactive" ]]; then
    non_interactive=1
  fi

  # [1/7] Run validation
  log_section_start "PRE-MERGE VALIDATION" "$logfile"

  if [[ -f "scripts/git/validate_branches.sh" ]]; then
    if ! bash "scripts/git/validate_branches.sh" >> "$logfile" 2>&1; then
      echo "✗ Validation failed - aborting merge" | tee -a "$logfile"
      log_section_end "PRE-MERGE VALIDATION" "$logfile" "1"
      echo "Please fix validation errors before retrying"
      exit 1
    fi
  fi

  echo "✓ Pre-merge validation passed" | tee -a "$logfile"
  log_section_end "PRE-MERGE VALIDATION" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [2/7] Store current branch and checkout master
  log_section_start "GIT CHECKOUT MAIN" "$logfile"

  local original_branch
  original_branch=$(git branch --show-current)
  echo "Current branch: ${original_branch}" | tee -a "$logfile"

  # CRITICAL SAFEGUARD: Prevent accidental wrong-direction merge
  if [[ "${original_branch}" == "master" ]]; then
    echo "✗ ERROR: Already on master branch" | tee -a "$logfile"
    echo "  This script merges development → master" | tee -a "$logfile"
    echo "  Current branch: ${original_branch}" | tee -a "$logfile"
    echo "  You should run this from development branch" | tee -a "$logfile"
    exit 1
  elif [[ "${original_branch}" != "development" ]]; then
    echo "⚠ WARNING: Not on development branch" | tee -a "$logfile"
    echo "  Current branch: ${original_branch}" | tee -a "$logfile"
    echo "  Expected: development" | tee -a "$logfile"

    if [[ ${non_interactive} -eq 0 ]]; then
      read -p "  Continue anyway? [y/N] " -n 1 -r
      echo
      if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "  Aborted by user" | tee -a "$logfile"
        exit 1
      fi
    else
      echo "  Non-interactive mode: aborting for safety" | tee -a "$logfile"
      exit 1
    fi
  fi

  if ! run_git_with_logging "GIT CHECKOUT" "$logfile" checkout master; then
    echo "✗ Failed to checkout master branch" | tee -a "$logfile"
    exit 1
  fi

  log_section_end "GIT CHECKOUT MAIN" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [3/7] Create pre-merge backup tag
  log_section_start "CREATE BACKUP TAG" "$logfile"

  if [[ -z "${timestamp:-}" ]]; then get_timestamp; fi
  local backup_tag="pre-merge-backup-${timestamp}"

  if git tag "${backup_tag}" >> "$logfile" 2>&1; then
    echo "✓ Created backup tag: ${backup_tag}" | tee -a "$logfile"
    log_section_end "CREATE BACKUP TAG" "$logfile" "0"
  else
    echo "⚠ Warning: Could not create backup tag" | tee -a "$logfile"
    log_section_end "CREATE BACKUP TAG" "$logfile" "1"
  fi
  echo "" | tee -a "$logfile"

  # [4/7] Perform merge
  log_section_start "GIT MERGE" "$logfile"

  if run_git_with_logging "GIT MERGE DEVELOPMENT" "$logfile" merge development --no-ff -m "Merge development into master"; then
    echo "✓ Merge completed without conflicts" | tee -a "$logfile"
    log_section_end "GIT MERGE" "$logfile" "0"
    
    # [6/7] Validate docs/ against CI policy (no-conflict case)
    echo ""
    echo "[6/7] Validating documentation files against CI policy..."

    local docs_validation_failed=0

    # Check docs being added/modified in the last commit (HEAD) compared to previous (HEAD~1)
    # Using git diff --name-only HEAD~1 HEAD
    # Warning: If merge is fast-forward, HEAD~1 might not be what we want. But we used --no-ff.

    # Filter for docs/ files only (not scripts/docs/)
    local doc_files
    doc_files=$(git diff --name-only HEAD~1 HEAD | grep "^docs/" || true)

    for doc_file in ${doc_files}; do
        local doc_name
        doc_name=$(basename "${doc_file}")
        if ! [[ "$doc_name" =~ $DOCS_PATTERN ]]; then
            # Check if it was Added (A)
            if git diff --diff-filter=A HEAD~1 HEAD -- "${doc_file}" >/dev/null 2>&1; then
                echo "✗ ERROR: Unauthorized doc file: ${doc_file}"
                echo "   This file is not in the CI allowed docs list"
                docs_validation_failed=1
            fi
        fi
    done

    if [[ ${docs_validation_failed} -eq 1 ]]; then
        echo "" | tee -a "$logfile"
        echo "✗ CI POLICY VIOLATION: Unauthorized documentation detected" | tee -a "$logfile"
        echo "" | tee -a "$logfile"
        echo "Rolling back merge..." | tee -a "$logfile"
        git reset --hard HEAD~1 >> "$logfile" 2>&1
        git checkout "${original_branch}" >> "$logfile" 2>&1
        exit 1
    fi
    echo "✓ Documentation validation passed" | tee -a "$logfile"

    # [6.5/7] Cleanup tests/ directory (only if gitignored)
    echo ""
    echo "[6.5/7] Checking tests/ directory policy..."

    # Check if tests/ is in .gitignore
    if grep -q "^tests/\$" .gitignore 2>/dev/null; then
      # tests/ is gitignored, should be removed from master
      if [[ -d "tests" ]]; then
        echo "⚠ Removing tests/ directory from master branch (per .gitignore policy)" | tee -a "$logfile"
        if git rm -r tests >> "$logfile" 2>&1; then
          echo "✓ Removed tests/ directory" | tee -a "$logfile"
          # Amend the merge commit to include the removal
          git commit --amend --no-edit >> "$logfile" 2>&1
        else
          echo "✗ ERROR: Failed to remove tests/ directory" | tee -a "$logfile"
          echo "  This may cause CI validation failure" | tee -a "$logfile"
        fi
      else
        echo "✓ No tests/ directory found (correct for master branch)" | tee -a "$logfile"
      fi
    else
      # tests/ is NOT gitignored, should be kept
      echo "✓ tests/ is tracked in git - keeping it in master branch" | tee -a "$logfile"
    fi

  else
    local merge_exit_code=$?
    echo "" | tee -a "$logfile"
    echo "⚠ Merge conflicts detected - analyzing..." | tee -a "$logfile"
    log_section_end "GIT MERGE" "$logfile" "$merge_exit_code"
    echo "  Analyzing conflict types and preparing auto-resolution..."
    echo ""
    
    # Check for modify/delete conflicts ("deleted by us" in status)
    # Status format: 'DU '
    if git status --short | grep -q "^DU "; then
        echo "  Found modify/delete conflicts for excluded files"
        echo "  These are expected and will be auto-resolved..."
        echo ""
        echo "  Files to be removed from master branch:"
        git status --short | grep "^DU "
        echo ""
        
        local resolution_failed=0

        # Process conflicts using process substitution to avoid subshell
        while read -r conflict_file; do
            echo "  Resolving: ${conflict_file}"
            if git rm "${conflict_file}" >/dev/null 2>&1; then
                echo "  ✓ Removed: ${conflict_file}"
            else
                echo "  ✗ ERROR: Failed to remove ${conflict_file}"
                resolution_failed=1
            fi
        done < <(git status --short | grep "^DU " | cut -c 4-)

        if [[ ${resolution_failed} -eq 1 ]]; then
            echo "" | tee -a "$logfile"
            echo "✗ Auto-resolution failed for some files" | tee -a "$logfile"
            echo "Current status:" | tee -a "$logfile"
            git status --short | tee -a "$logfile"
            echo "" | tee -a "$logfile"
            echo "Please resolve manually or abort merge"
            exit 1
        fi

        echo "" | tee -a "$logfile"
        echo "✓ Auto-resolved modify/delete conflicts" | tee -a "$logfile"

        # Check if all conflicts are resolved
        unresolved=$(git diff --name-only --diff-filter=U 2>/dev/null)
        if [[ -n "$unresolved" ]]; then
            echo "" | tee -a "$logfile"
            echo "⚠ Some conflicts remaster unresolved:" | tee -a "$logfile"
            echo "$unresolved" | tee -a "$logfile"
            echo "  Continuing to validation and manual commit..." | tee -a "$logfile"
        else
            # Check if changes staged?
            if git diff --cached --quiet; then
                 echo "" | tee -a "$logfile"
                 echo "⚠ No changes staged after auto-resolution" | tee -a "$logfile"
                 echo "  Continuing to validation..." | tee -a "$logfile"
            else
                 # Check if merge commit is needed or already happened?
                 # If we resolved logic, we are still potentially in merging state.
                 # Check MERGE_HEAD
                 if ! git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
                    echo "" | tee -a "$logfile"
                    echo "✓ Merge commit automatically completed during auto-resolution" | tee -a "$logfile"
                    # logic flow jump? In bash we continue.
                 fi
            fi
        fi

    fi

    # Check for actual content conflicts (Both Modified: 'UU ')
    if git status --short | grep -q "^UU "; then
        echo "" | tee -a "$logfile"
        echo "✗ Content conflicts require manual resolution:" | tee -a "$logfile"
        echo ""
        git status --short | grep "^UU "
        echo ""
        echo "Please resolve these conflicts manually:"
        echo "  1. Edit conflicted files"
        echo "  2. git add <resolved files>"
        echo "  3. git commit"
        echo ""
        echo "Or abort the merge:"
        echo "  git merge --abort"
        echo "  git checkout ${original_branch}"
        exit 1
    fi

    # [6/7] Validate docs/ against CI policy
    echo ""
    echo "[6/7] Validating documentation files against CI policy..."

    local docs_validation_failed=0

    # Check added docs in staged changes (only docs/ directory, not scripts/docs/)
    local staged_docs
    staged_docs=$(git diff --cached --name-only --diff-filter=A | grep "^docs/" || true)

    for doc_file in ${staged_docs}; do
        local doc_name
        doc_name=$(basename "${doc_file}")
        if ! [[ "$doc_name" =~ $DOCS_PATTERN ]]; then
            echo "✗ ERROR: Unauthorized doc file: ${doc_file}"
            echo "   This file is not in the CI allowed docs list"
            docs_validation_failed=1
        fi
    done

    if [[ ${docs_validation_failed} -eq 1 ]]; then
        echo "" | tee -a "$logfile"
        echo "✗ CI POLICY VIOLATION: Unauthorized documentation detected" | tee -a "$logfile"
        echo "" | tee -a "$logfile"
        echo "Aborting merge to prevent CI failure..." | tee -a "$logfile"
        git merge --abort >> "$logfile" 2>&1
        git checkout "${original_branch}" >> "$logfile" 2>&1
        exit 1
    fi
    echo "✓ Documentation validation passed" | tee -a "$logfile"

    # [6.5/7] Cleanup tests/ directory (only if gitignored)
    echo ""
    echo "[6.5/7] Checking tests/ directory policy..."

    # Check if tests/ is in .gitignore
    if grep -q "^tests/\$" .gitignore 2>/dev/null; then
      # tests/ is gitignored, should be removed from master
      if [[ -d "tests" ]]; then
        echo "⚠ Removing tests/ directory from master branch (per .gitignore policy)" | tee -a "$logfile"
        if git rm -r tests >> "$logfile" 2>&1; then
          echo "✓ Removed tests/ directory" | tee -a "$logfile"
          # Stage the removal for the upcoming commit
          git add -u >> "$logfile" 2>&1
        else
          echo "✗ ERROR: Failed to remove tests/ directory" | tee -a "$logfile"
          echo "  This may cause CI validation failure" | tee -a "$logfile"
        fi
      else
        echo "✓ No tests/ directory found (correct for master branch)" | tee -a "$logfile"
      fi
    else
      # tests/ is NOT gitignored, should be kept
      echo "✓ tests/ is tracked in git - keeping it in master branch" | tee -a "$logfile"
    fi
    echo "" | tee -a "$logfile"

    # [7/7] Complete the merge
    log_section_start "GIT COMMIT" "$logfile"
    # Only if MERGE_HEAD exists (merging state)
    if git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
        if run_git_with_logging "GIT COMMIT MERGE" "$logfile" commit --no-edit; then
             echo "✓ Merge commit completed" | tee -a "$logfile"
        else
            echo "✗ Failed to complete merge commit" | tee -a "$logfile"
            log_section_end "GIT COMMIT" "$logfile" "1"
            echo "" | tee -a "$logfile"
            echo "To abort: git merge --abort"
            exit 1
        fi
    fi
    log_section_end "GIT COMMIT" "$logfile" "0"
  
  fi

  # Success logic
  echo "" | tee -a "$logfile"
  {
    echo "========================================"
    echo "[MERGE SUMMARY]"
    echo "========================================"
  } | tee -a "$logfile"

  echo "✓ MERGE SUCCESSFUL" | tee -a "$logfile"
  echo "" | tee -a "$logfile"
  echo "Summary:" | tee -a "$logfile"
  git log -1 --oneline | while read -r line; do echo "  Latest commit: $line" | tee -a "$logfile"; done
  echo "  Backup tag: ${backup_tag}" | tee -a "$logfile"
  echo "" | tee -a "$logfile"
  echo "Next steps:" | tee -a "$logfile"
  echo "  1. Review changes: git log --oneline -5" | tee -a "$logfile"
  echo "  2. Verify build: [run your build/test commands]" | tee -a "$logfile"
  echo "  3. Push to remote: git push origin master" | tee -a "$logfile"
  echo "" | tee -a "$logfile"
  echo "  If issues found:" | tee -a "$logfile"
  echo "  - scripts/git/rollback_merge.sh" | tee -a "$logfile"
  echo "  - Or: git reset --hard ${backup_tag}" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  # Generate analysis report (basic)
  echo "# Merge Validation Workflow Analysis Report" > "${reportfile}"
  echo "" >> "${reportfile}"
  echo "**Status**: ✅ SUCCESS" >> "${reportfile}"
  echo "**Backup Tag**: \`${backup_tag}\`" >> "${reportfile}"
  echo "**Files Changed**:" >> "${reportfile}"
  git diff HEAD~1 --name-status >> "${reportfile}" 2>/dev/null

  echo "📊 Analysis Report: ${reportfile}" | tee -a "$logfile"
  echo "📋 Backup Tag: ${backup_tag}" | tee -a "$logfile"

  {
    echo ""
    echo "End Time: $(date)"
  } | tee -a "$logfile"

  echo "" | tee -a "$logfile"
  echo "Full log: $logfile"
}

master "$@"
