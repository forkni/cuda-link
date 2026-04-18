#!/usr/bin/env bash
# rollback_merge.sh - Emergency rollback for merge operations
# Purpose: Revert target branch to pre-merge state safely
# Usage: ./scripts/git/rollback_merge.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR          - Directory containing this script
#   PROJECT_ROOT        - Auto-detected git repo root (set by _config.sh)
#   logfile             - Set by init_logging
#   CGW_TARGET_BRANCH   - Branch to roll back (default: main)
# Arguments:
#   --non-interactive   Skip prompts; auto-selects latest backup tag if --target omitted
#   --target <ref>      Commit hash, tag name, or HEAD~1 to roll back to
#   --dry-run           Show rollback target without resetting
#   -h, --help          Show help
# Returns:
#   0 on successful rollback, 1 on failure

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/git/_common.sh
source "${SCRIPT_DIR}/_common.sh"

init_logging "rollback_merge"

_rollback_done=0

_cleanup_rollback() {
  [[ ${_rollback_done} -eq 1 ]] && return 0
  echo "" >&2
  echo "[!] Rollback interrupted. Verify repository state before proceeding:" >&2
  echo "  git log --oneline -5" >&2
  echo "  git status" >&2
}
trap _cleanup_rollback EXIT INT TERM

main() {
  local non_interactive=0
  local dry_run=0
  local rollback_target_flag=""
  local use_revert=0

  while [[ $# -gt 0 ]]; do
    case "${1}" in
      --help | -h)
        echo "Usage: ./scripts/git/rollback_merge.sh [OPTIONS]"
        echo ""
        echo "Emergency rollback: resets target branch to a pre-merge state."
        echo "Must be run from the target branch (default: ${CGW_TARGET_BRANCH})."
        echo ""
        echo "Options:"
        echo "  --non-interactive   Skip prompts; auto-selects latest backup tag if --target omitted"
        echo "  --target <ref>      Commit hash, tag name, or HEAD~1 to roll back to"
        echo "  --dry-run           Show rollback target without resetting"
        echo "  --revert            Safe mode: use 'git revert -m 1' instead of 'git reset --hard'"
        echo "                      Preserves history -- safe for shared repos where commits are pushed"
        echo "  -h, --help          Show this help"
        echo ""
        echo "Environment:"
        echo "  CGW_NON_INTERACTIVE=1   Same as --non-interactive"
        echo "  CGW_REMOTE              Remote name (default: origin)"
        echo ""
        echo "CAUTION: Without --revert, this rewrites branch history. Force-push required after."
        echo "         With --revert, history is preserved -- no force-push needed."
        exit 0
        ;;
      --non-interactive) non_interactive=1 ;;
      --dry-run) dry_run=1 ;;
      --revert) use_revert=1 ;;
      --target)
        rollback_target_flag="${2:-}"
        shift
        ;;
      *)
        echo "[ERROR] Unknown flag: $1" >&2
        exit 1
        ;;
    esac
    shift
  done

  [[ "${CGW_NON_INTERACTIVE:-0}" == "1" ]] && non_interactive=1

  {
    echo "========================================="
    echo "Rollback Merge Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } >"$logfile"

  echo "=== Emergency Merge Rollback ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  cd "${PROJECT_ROOT}" || {
    err "Cannot find project root"
    exit 1
  }

  # [1/5] Verify current branch
  log_section_start "BRANCH VERIFICATION" "$logfile"

  local current_branch
  current_branch=$(git branch --show-current 2>&1)
  echo "Current branch: ${current_branch}" | tee -a "$logfile"

  if [[ "${current_branch}" != "${CGW_TARGET_BRANCH}" ]]; then
    echo "" | tee -a "$logfile"
    echo "[FAIL] ERROR: Not on target branch (${CGW_TARGET_BRANCH})" | tee -a "$logfile"
    echo "This script should only be run from the target branch" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "Current branch: ${current_branch}" | tee -a "$logfile"
    echo "Expected: ${CGW_TARGET_BRANCH}" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "Please checkout target branch first: git checkout ${CGW_TARGET_BRANCH}"
    log_section_end "BRANCH VERIFICATION" "$logfile" "1"
    exit 1
  fi
  echo "[OK] On target branch (${CGW_TARGET_BRANCH})" | tee -a "$logfile"
  log_section_end "BRANCH VERIFICATION" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [2/5] Check for uncommitted changes
  log_section_start "UNCOMMITTED CHANGES CHECK" "$logfile"

  if ! git diff-index --quiet HEAD --; then
    echo "[!] WARNING: Uncommitted changes detected" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    git status --short | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "These changes will be LOST during rollback!" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    if [[ ${non_interactive} -eq 1 ]]; then
      echo "[Non-interactive] Aborting -- commit or stash changes first" | tee -a "$logfile"
      log_section_end "UNCOMMITTED CHANGES CHECK" "$logfile" "1"
      exit 1
    fi
    read -r -p "Continue anyway? (yes/no): " continue_choice
    if [[ "${continue_choice}" != "yes" ]]; then
      echo "" | tee -a "$logfile"
      echo "Rollback cancelled" | tee -a "$logfile"
      echo "Please commit or stash changes first"
      log_section_end "UNCOMMITTED CHANGES CHECK" "$logfile" "1"
      exit 1
    fi
  else
    echo "[OK] No uncommitted changes" | tee -a "$logfile"
  fi
  log_section_end "UNCOMMITTED CHANGES CHECK" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [3/5] Find rollback target
  log_section_start "FIND ROLLBACK TARGET" "$logfile"

  local backup_tags
  backup_tags=$(git tag -l "pre-merge-backup-*" | sort -r | head -5)
  if [[ -n "${backup_tags}" ]]; then
    echo "Available backup tags:" | tee -a "$logfile"
    echo "${backup_tags}" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
  else
    echo "No backup tags found (pre-merge-backup-*)" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
  fi

  echo "Recent commits:" | tee -a "$logfile"
  git log --oneline -5 | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  local latest_merge
  latest_merge=$(git log --merges --oneline -1)
  if [[ -n "${latest_merge}" ]]; then
    echo "Latest merge commit: ${latest_merge}" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
  fi

  log_section_end "FIND ROLLBACK TARGET" "$logfile" "0"

  # [4/5] Choose rollback target
  local rollback_target=""

  if [[ -n "${rollback_target_flag}" ]]; then
    if ! git rev-parse "${rollback_target_flag}" >/dev/null 2>&1; then
      err "Invalid --target ref: ${rollback_target_flag}"
      exit 1
    fi
    rollback_target="${rollback_target_flag}"
    echo "Rollback target (from --target): ${rollback_target}" | tee -a "$logfile"
  elif [[ ${non_interactive} -eq 1 ]]; then
    local latest_tag
    latest_tag=$(git tag -l "pre-merge-backup-*" | sort -r | head -1)
    if [[ -n "${latest_tag}" ]]; then
      rollback_target="${latest_tag}"
      echo "[Non-interactive] Using latest backup tag: ${rollback_target}" | tee -a "$logfile"
    else
      rollback_target="HEAD~1"
      echo "[Non-interactive] No backup tag found -- using HEAD~1" | tee -a "$logfile"
    fi
  else
    echo "[4/5] Choose rollback method:"
    echo ""
    echo "Available options:"
    echo "  1. Rollback to latest pre-merge backup tag (recommended)"
    echo "  2. Rollback to commit before latest merge (HEAD~1)"
    echo "  3. Rollback to specific commit hash"
    echo "  4. Cancel rollback"
    echo ""

    read -r -p "Select option (1-4): " rollback_choice

    case "${rollback_choice}" in
      1)
        rollback_target=$(git tag -l "pre-merge-backup-*" | sort -r | head -1)
        if [[ -z "${rollback_target}" ]]; then
          err "No backup tags found"
          echo "Please use option 2 or 3"
          exit 1
        fi
        echo "Rollback target: ${rollback_target}"
        ;;
      2)
        rollback_target="HEAD~1"
        echo "Rollback target: HEAD~1 (previous commit)"
        ;;
      3)
        echo ""
        read -r -p "Enter commit hash: " rollback_target
        echo ""
        if ! git rev-parse "${rollback_target}" >/dev/null 2>&1; then
          err "Invalid commit hash: ${rollback_target}"
          exit 1
        fi
        echo "Rollback target: ${rollback_target}"
        ;;
      4)
        echo "" | tee -a "$logfile"
        echo "Rollback cancelled" | tee -a "$logfile"
        _rollback_done=1
        exit 0
        ;;
      *)
        err "Invalid choice: ${rollback_choice}"
        exit 1
        ;;
    esac
  fi

  # [5/5] Execute rollback
  echo "" | tee -a "$logfile"
  echo "[!] WARNING: This will permanently reset ${CGW_TARGET_BRANCH} branch to:" | tee -a "$logfile"
  git log "${rollback_target}" --oneline -1 | tee -a "$logfile"
  echo "" | tee -a "$logfile"
  echo "All commits after this point will be lost!" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  if [[ ${dry_run} -eq 1 ]]; then
    echo "=== DRY RUN -- no changes made ===" | tee -a "$logfile"
    echo "Would reset ${CGW_TARGET_BRANCH} to: ${rollback_target}" | tee -a "$logfile"
    _rollback_done=1
    exit 0
  fi

  local confirm
  if [[ ${non_interactive} -eq 0 ]]; then
    read -r -p "Type 'ROLLBACK' to confirm: " confirm
  else
    confirm="ROLLBACK"
  fi

  if [[ "${confirm}" != "ROLLBACK" ]]; then
    echo "" | tee -a "$logfile"
    echo "Rollback cancelled" | tee -a "$logfile"
    _rollback_done=1
    exit 0
  fi

  if [[ ${use_revert} -eq 1 ]]; then
    # Safe revert mode: creates a new commit that undoes the merge.
    # Preserves history -- no force-push needed (Pro Git p.288-289).
    # git revert -m 1 requires a merge commit (2+ parents); validate before attempting.
    local parent_count
    parent_count=$(git cat-file -p "${rollback_target}" 2>/dev/null | grep -c "^parent " || echo "0")
    if [[ "${parent_count}" -lt 2 ]]; then
      err "--revert requires a merge commit (2+ parents), but ${rollback_target} has ${parent_count} parent(s)"
      err "Use plain rollback (omit --revert) or provide a merge commit hash with --target"
      exit 1
    fi
    log_section_start "GIT REVERT" "$logfile"
    if run_git_with_logging "GIT REVERT MERGE" "$logfile" revert -m 1 --no-edit "${rollback_target}"; then
      log_section_end "GIT REVERT" "$logfile" "0"
      echo "" | tee -a "$logfile"
      {
        echo "========================================"
        echo "[ROLLBACK SUMMARY -- REVERT MODE]"
        echo "========================================"
      } | tee -a "$logfile"
      echo "[OK] REVERT SUCCESSFUL" | tee -a "$logfile"
      echo "" | tee -a "$logfile"
      echo "Summary:" | tee -a "$logfile"
      git log --oneline -1 | while read -r line; do echo "  Current HEAD: $line" | tee -a "$logfile"; done
      echo "" | tee -a "$logfile"
      echo "Next steps:" | tee -a "$logfile"
      echo "  1. Verify revert: git log --oneline -5" | tee -a "$logfile"
      echo "  2. Push normally: git push ${CGW_REMOTE} ${CGW_TARGET_BRANCH}" | tee -a "$logfile"
      echo "     (no force-push needed -- history is preserved)" | tee -a "$logfile"
      {
        echo ""
        echo "End Time: $(date)"
      } | tee -a "$logfile"
      echo "" | tee -a "$logfile"
      _rollback_done=1
      echo "Full log: $logfile"
    else
      log_section_end "GIT REVERT" "$logfile" "1"
      echo "" | tee -a "$logfile"
      echo "[FAIL] Revert failed" | tee -a "$logfile"
      echo "Please manually revert: git revert -m 1 ${rollback_target}"
      exit 1
    fi
  else
    log_section_start "GIT RESET" "$logfile"

    if run_git_with_logging "GIT RESET HARD" "$logfile" reset --hard "${rollback_target}"; then
      log_section_end "GIT RESET" "$logfile" "0"
      echo "" | tee -a "$logfile"
      {
        echo "========================================"
        echo "[ROLLBACK SUMMARY]"
        echo "========================================"
      } | tee -a "$logfile"
      echo "[OK] ROLLBACK SUCCESSFUL" | tee -a "$logfile"
      echo "" | tee -a "$logfile"
      echo "Summary:" | tee -a "$logfile"
      git log --oneline -1 | while read -r line; do echo "  Current HEAD: $line" | tee -a "$logfile"; done
      echo "" | tee -a "$logfile"
      echo "Next steps:" | tee -a "$logfile"
      echo "  1. Verify rollback: git log --oneline -5" | tee -a "$logfile"
      echo "  2. If correct, force push: git push ${CGW_REMOTE} ${CGW_TARGET_BRANCH} --force-with-lease" | tee -a "$logfile"
      echo "  3. If issues, contact maintainer" | tee -a "$logfile"
      echo "" | tee -a "$logfile"
      echo "  [!] WARNING: Force push will rewrite remote history!" | tee -a "$logfile"
      {
        echo ""
        echo "End Time: $(date)"
      } | tee -a "$logfile"
      echo "" | tee -a "$logfile"
      _rollback_done=1
      echo "Full log: $logfile"
    else
      log_section_end "GIT RESET" "$logfile" "1"
      echo "" | tee -a "$logfile"
      echo "[FAIL] Rollback failed" | tee -a "$logfile"
      echo "Please manually reset: git reset --hard ${rollback_target}"
      exit 1
    fi
  fi
}

main "$@"
