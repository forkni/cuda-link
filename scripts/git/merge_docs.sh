#!/usr/bin/env bash
# merge_docs.sh - Documentation-only merge from source to target branch
# Purpose: Merge only docs/ directory changes while ignoring code changes
# Usage: ./scripts/git/merge_docs.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR          - Directory containing this script
#   PROJECT_ROOT        - Auto-detected git repo root (set by _config.sh)
#   logfile             - Set by init_logging
#   CGW_SOURCE_BRANCH   - Source branch (default: development)
#   CGW_TARGET_BRANCH   - Target branch (default: main)
# Arguments:
#   --non-interactive   Skip prompts
#   --source <branch>   Override source branch for this invocation
#   --target <branch>   Override target branch for this invocation
#   -h, --help          Show help
# Returns:
#   0 on success, 1 on failure

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

init_logging "merge_docs"

original_branch=""
did_mutate_worktree=0

cleanup() {
  if [[ ${did_mutate_worktree} -eq 1 ]]; then
    echo "" | tee -a "$logfile"
    echo "[!] Interrupted - restoring original state..." | tee -a "$logfile"
    git restore --staged --worktree -- . 2>/dev/null || git reset --hard HEAD 2>/dev/null || true
  fi
  local current_branch
  current_branch=$(git branch --show-current 2>/dev/null)
  if [[ -n "$original_branch" ]] && [[ "$original_branch" != "$current_branch" ]]; then
    git checkout "$original_branch" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

main() {
  local non_interactive=0
  local src_branch="${CGW_SOURCE_BRANCH}"
  local tgt_branch="${CGW_TARGET_BRANCH}"

  while [[ $# -gt 0 ]]; do
    case "${1}" in
      --help | -h)
        echo "Usage: ./scripts/git/merge_docs.sh [OPTIONS]"
        echo ""
        echo "Merge only docs/ directory from source branch into target branch."
        echo "Code changes are NOT included -- only files under docs/."
        echo ""
        echo "Options:"
        echo "  --non-interactive   Skip prompts"
        echo "  --source <branch>   Override source branch for this invocation"
        echo "  --target <branch>   Override target branch for this invocation"
        echo "  -h, --help          Show this help"
        echo ""
        echo "Configuration:"
        echo "  CGW_SOURCE_BRANCH   Source branch (default: development)"
        echo "  CGW_TARGET_BRANCH   Target branch (default: main)"
        echo ""
        echo "Environment:"
        echo "  CGW_NON_INTERACTIVE=1   Same as --non-interactive"
        echo ""
        echo "WARNING: Replaces entire docs/ on target with docs/ from source."
        echo "         Any docs-only changes on target not in source will be lost."
        exit 0
        ;;
      --non-interactive) non_interactive=1 ;;
      --source)
        src_branch="${2:-}"
        if [[ -z "${src_branch}" ]]; then
          err "--source requires a branch name"
          exit 1
        fi
        shift
        ;;
      --target)
        tgt_branch="${2:-}"
        if [[ -z "${tgt_branch}" ]]; then
          err "--target requires a branch name"
          exit 1
        fi
        shift
        ;;
      *)
        err "Unknown flag: $1"
        exit 1
        ;;
    esac
    shift
  done

  [[ "${CGW_NON_INTERACTIVE:-0}" == "1" ]] && non_interactive=1

  validate_branch_pair "${src_branch}" "${tgt_branch}"

  {
    echo "========================================="
    echo "Documentation Merge Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } >"$logfile"

  echo "=== Documentation Merge: ${src_branch} -> ${tgt_branch} ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  cd "${PROJECT_ROOT}" || {
    err "Cannot find project root"
    exit 1
  }

  # [1/7] Run validation
  log_section_start "PRE-MERGE VALIDATION" "$logfile"

  if [[ -f "${SCRIPT_DIR}/validate_branches.sh" ]]; then
    if ! CGW_SOURCE_BRANCH="${src_branch}" CGW_TARGET_BRANCH="${tgt_branch}" \
      bash "${SCRIPT_DIR}/validate_branches.sh" >>"$logfile" 2>&1; then
      echo "[FAIL] Validation failed - aborting documentation merge" | tee -a "$logfile"
      log_section_end "PRE-MERGE VALIDATION" "$logfile" "1"
      echo "Please fix validation errors before retrying"
      exit 1
    fi
  fi

  echo "[OK] Pre-merge validation passed" | tee -a "$logfile"
  log_section_end "PRE-MERGE VALIDATION" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [2/7] Store current branch and checkout target
  log_section_start "GIT CHECKOUT TARGET" "$logfile"

  original_branch=$(git branch --show-current)
  echo "Current branch: ${original_branch}" | tee -a "$logfile"

  if ! run_git_with_logging "GIT CHECKOUT" "$logfile" checkout "${tgt_branch}"; then
    echo "[FAIL] Failed to checkout ${tgt_branch} branch" | tee -a "$logfile"
    exit 1
  fi

  log_section_end "GIT CHECKOUT TARGET" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [3/7] Check for documentation changes
  echo "[3/7] Checking for documentation changes..."

  if [[ -z "$(git diff --name-only "${tgt_branch}" "${src_branch}" -- docs/)" ]]; then
    log_message "[!] No documentation changes found between ${tgt_branch} and ${src_branch}" "${logfile}"
    if [[ ${non_interactive} -eq 1 ]]; then
      log_message "Documentation merge cancelled (nothing to do)" "${logfile}"
      git checkout "${original_branch}"
      exit 0
    fi
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
  git diff --name-only "${tgt_branch}" "${src_branch}" -- docs/
  echo ""

  # [4/7] Check for non-documentation changes
  echo "[4/7] Checking for code changes..."
  local non_docs_count
  non_docs_count=$(git diff --name-only "${tgt_branch}" "${src_branch}" | grep -vc "^docs/")

  if [[ "${non_docs_count}" -gt 0 ]]; then
    echo "[!] WARNING: Non-documentation changes detected"
    echo ""
    echo "This merge will ONLY include docs/ changes."
    echo "Other changes will remain on ${src_branch} branch."
    echo ""
    git diff --name-only "${tgt_branch}" "${src_branch}" | grep -v "^docs/"
    echo ""
    if [[ ${non_interactive} -eq 1 ]]; then
      echo "[Non-interactive] Proceeding with docs-only merge despite non-docs changes" | tee -a "$logfile"
    else
      read -r -p "Proceed with docs-only merge? (yes/no): " confirm
      if [[ "${confirm}" != "yes" ]]; then
        echo ""
        log_message "Documentation merge cancelled" "${logfile}"
        git checkout "${original_branch}"
        exit 0
      fi
    fi
  else
    log_message "[OK] No code changes detected (docs-only merge)" "${logfile}"
  fi
  echo ""

  # [5/7] Create pre-merge backup tag
  log_section_start "CREATE BACKUP TAG" "$logfile"

  if [[ -z "${timestamp:-}" ]]; then get_timestamp; fi
  local backup_tag="pre-docs-merge-${timestamp}-$$"

  if git tag "${backup_tag}" >>"$logfile" 2>&1; then
    echo "[OK] Created backup tag: ${backup_tag}" | tee -a "$logfile"
    log_section_end "CREATE BACKUP TAG" "$logfile" "0"
  else
    echo "[!] Warning: Could not create backup tag" | tee -a "$logfile"
    log_section_end "CREATE BACKUP TAG" "$logfile" "1"
  fi
  echo "" | tee -a "$logfile"

  # [6/7] Merge documentation changes
  log_section_start "MERGE DOCUMENTATION" "$logfile"

  did_mutate_worktree=1

  if ! run_git_with_logging "GIT CHECKOUT DOCS" "$logfile" checkout "${src_branch}" -- docs/; then
    echo "[FAIL] Failed to checkout documentation from ${src_branch}" | tee -a "$logfile"
    git checkout "${original_branch}" >>"$logfile" 2>&1
    exit 1
  fi

  if git diff --cached --quiet; then
    echo "" | tee -a "$logfile"
    echo "[!] No documentation changes to merge (docs already in sync)" | tee -a "$logfile"
    log_section_end "MERGE DOCUMENTATION" "$logfile" "0"
    git checkout "${original_branch}" >>"$logfile" 2>&1
    exit 0
  fi

  echo "[OK] Documentation changes staged" | tee -a "$logfile"
  echo "Staged files:" | tee -a "$logfile"
  git diff --cached --name-only | tee -a "$logfile"
  log_section_end "MERGE DOCUMENTATION" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [7/7] Commit the merge
  log_section_start "GIT COMMIT" "$logfile"

  if run_git_with_logging "GIT COMMIT DOCS" "$logfile" commit \
    -m "docs: Sync documentation from ${src_branch}" \
    -m "- Updated docs/ directory from ${src_branch} branch" \
    -m "- Docs-only update (no code changes)" \
    -m "- Backup tag: ${backup_tag}"; then
    log_section_end "GIT COMMIT" "$logfile" "0"
    echo "" | tee -a "$logfile"
    {
      echo "========================================"
      echo "[DOCS MERGE SUMMARY]"
      echo "========================================"
    } | tee -a "$logfile"
    echo "[OK] DOCUMENTATION MERGE SUCCESSFUL" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    git log -1 --oneline | while read -r line; do echo "  Latest commit: $line" | tee -a "$logfile"; done
    echo "  Backup tag: ${backup_tag}" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "Merged files:" | tee -a "$logfile"
    git diff --name-only HEAD~1 HEAD | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "Next steps:" | tee -a "$logfile"
    echo "  1. Review: git show HEAD" | tee -a "$logfile"
    echo "  2. Push: ./scripts/git/push_validated.sh" | tee -a "$logfile"
    echo "  Rollback: git reset --hard ${backup_tag}" | tee -a "$logfile"
    {
      echo ""
      echo "End Time: $(date)"
    } | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "Full log: $logfile"
  else
    log_section_end "GIT COMMIT" "$logfile" "1"
    echo "" | tee -a "$logfile"
    echo "[FAIL] Failed to commit documentation merge" | tee -a "$logfile"
    echo ""
    echo "To abort: git reset --hard HEAD"
    exit 1
  fi
}

main "$@"
