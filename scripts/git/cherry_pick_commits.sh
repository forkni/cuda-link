#!/usr/bin/env bash
# cherry_pick_commits.sh - Cherry-pick specific commits from source to target branch
# Purpose: Cherry-pick commits with validation and automatic backup
# Usage: ./scripts/git/cherry_pick_commits.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR            - Directory containing this script
#   PROJECT_ROOT          - Auto-detected git repo root (set by _config.sh)
#   logfile               - Set by init_logging
#   CGW_SOURCE_BRANCH     - Source branch (commits come from here; default: development)
#   CGW_TARGET_BRANCH     - Target branch (commits go here; default: main)
#   CGW_DEV_ONLY_FILES    - Space-separated dev-only file patterns (warns if commit touches these)
# Arguments:
#   --non-interactive    Skip prompts; requires --commit
#   --commit <hash>      Commit hash to cherry-pick (skips interactive selection)
#   --dry-run            Show commit details without cherry-picking
#   --source <branch>    Override source branch for this invocation
#   --target <branch>    Override target branch for this invocation
#   -h, --help           Show help
# Returns:
#   0 on success, 1 on failure or conflict

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/git/_common.sh
source "${SCRIPT_DIR}/_common.sh"

init_logging "cherry_pick_commits"

_cp_original_branch=""
_cp_did_checkout_target=0

_cleanup_cherry_pick() {
  local current
  current=$(git branch --show-current 2>/dev/null || true)
  if [[ ${_cp_did_checkout_target} -eq 1 ]] && [[ -n "${_cp_original_branch}" ]] &&
    [[ "${current}" != "${_cp_original_branch}" ]]; then
    echo "" >&2
    echo "[!] Interrupted -- you are on branch: ${current}" >&2
    echo "  Returning to: ${_cp_original_branch}" >&2
    if git rev-parse -q --verify CHERRY_PICK_HEAD >/dev/null 2>&1; then
      git cherry-pick --abort 2>/dev/null || true
    fi
    git checkout "${_cp_original_branch}" 2>/dev/null || true
  fi
}
trap _cleanup_cherry_pick EXIT INT TERM

main() {
  local non_interactive=0
  local dry_run=0
  local commit_hash_flag=""
  local src_branch="${CGW_SOURCE_BRANCH}"
  local tgt_branch="${CGW_TARGET_BRANCH}"

  while [[ $# -gt 0 ]]; do
    case "${1}" in
      --help | -h)
        echo "Usage: ./scripts/git/cherry_pick_commits.sh [OPTIONS]"
        echo ""
        echo "Cherry-pick a commit from source branch to target branch with validation."
        echo ""
        echo "Options:"
        echo "  --non-interactive    Skip prompts; requires --commit"
        echo "  --commit <hash>      Commit hash to cherry-pick (skips interactive selection)"
        echo "  --dry-run            Show commit details without cherry-picking"
        echo "  --source <branch>    Override source branch for this invocation"
        echo "  --target <branch>    Override target branch for this invocation"
        echo "  -h, --help           Show this help"
        echo ""
        echo "Configuration:"
        echo "  CGW_SOURCE_BRANCH     Branch commits come from (default: development)"
        echo "  CGW_TARGET_BRANCH     Branch commits go to (default: main)"
        echo "  CGW_DEV_ONLY_FILES    Dev-only paths to warn about (default: empty)"
        echo ""
        echo "Environment:"
        echo "  CGW_NON_INTERACTIVE=1   Same as --non-interactive"
        exit 0
        ;;
      --non-interactive) non_interactive=1 ;;
      --dry-run) dry_run=1 ;;
      --commit)
        commit_hash_flag="${2:-}"
        shift
        ;;
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
    echo "Cherry-Pick Commits Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } >"$logfile"

  echo "=== Cherry-Pick Commits: ${src_branch} -> ${tgt_branch} ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  cd "${PROJECT_ROOT}" || {
    err "Cannot find project root"
    exit 1
  }

  # [1/6] Run validation
  log_section_start "PRE-CHERRY-PICK VALIDATION" "$logfile"

  if [[ -f "${SCRIPT_DIR}/validate_branches.sh" ]]; then
    if ! CGW_SOURCE_BRANCH="${src_branch}" CGW_TARGET_BRANCH="${tgt_branch}" \
      bash "${SCRIPT_DIR}/validate_branches.sh" >>"$logfile" 2>&1; then
      echo "[FAIL] Validation failed - aborting cherry-pick" | tee -a "$logfile"
      log_section_end "PRE-CHERRY-PICK VALIDATION" "$logfile" "1"
      echo "Please fix validation errors before retrying"
      exit 1
    fi
  fi

  echo "[OK] Pre-cherry-pick validation passed" | tee -a "$logfile"
  log_section_end "PRE-CHERRY-PICK VALIDATION" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [2/6] Store current branch and checkout target
  log_section_start "GIT CHECKOUT TARGET" "$logfile"

  local original_branch
  original_branch=$(git branch --show-current)
  _cp_original_branch="${original_branch}"

  if [[ -z "${original_branch}" ]]; then
    echo "[FAIL] Failed to determine current branch" | tee -a "$logfile"
    log_section_end "GIT CHECKOUT TARGET" "$logfile" "1"
    exit 1
  fi

  echo "Current branch: ${original_branch}" | tee -a "$logfile"

  if ! run_git_with_logging "GIT CHECKOUT" "$logfile" checkout "${tgt_branch}"; then
    echo "[FAIL] Failed to checkout ${tgt_branch} branch" | tee -a "$logfile"
    exit 1
  fi
  _cp_did_checkout_target=1

  log_section_end "GIT CHECKOUT TARGET" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [3/6] Show recent source branch commits
  if [[ -z "${commit_hash_flag}" ]]; then
    echo "[3/6] Recent commits on ${src_branch} branch:"
    echo "===================================="
    git log "${src_branch}" --oneline -20 --no-merges
    echo "===================================="
    echo ""
  fi

  # [4/6] Get commit hash
  local commit_hash
  if [[ -n "${commit_hash_flag}" ]]; then
    commit_hash="${commit_hash_flag}"
    echo "[4/6] Using --commit: ${commit_hash}" | tee -a "$logfile"
  elif [[ ${non_interactive} -eq 1 ]]; then
    echo "[FAIL] [Non-interactive] --commit <hash> is required" >&2
    git checkout "${original_branch}"
    exit 1
  else
    echo "[4/6] Select commit to cherry-pick..."
    echo ""
    read -e -r -p "Enter commit hash (or 'cancel' to abort): " commit_hash

    if [[ "${commit_hash}" == "cancel" ]]; then
      echo ""
      log_message "Cherry-pick cancelled" "${logfile}"
      git checkout "${original_branch}"
      exit 0
    fi
  fi

  if ! git rev-parse "${commit_hash}" >/dev/null 2>&1; then
    log_message "[FAIL] ERROR: Invalid commit hash: ${commit_hash}" "${logfile}"
    git checkout "${original_branch}"
    exit 1
  fi

  # Validate commit is on source branch
  if ! git merge-base --is-ancestor "${commit_hash}" "${src_branch}" 2>/dev/null; then
    echo "[!] WARNING: ${commit_hash} is not an ancestor of ${src_branch}" | tee -a "$logfile"
    if [[ ${non_interactive} -eq 1 ]]; then
      echo "[FAIL] [Non-interactive] Aborting -- commit not on ${src_branch} branch" | tee -a "$logfile"
      git checkout "${original_branch}"
      exit 1
    fi
    read -r -p "Continue anyway? (yes/no): " branch_check_choice
    if [[ "${branch_check_choice}" != "yes" ]]; then
      log_message "Cherry-pick cancelled" "${logfile}"
      git checkout "${original_branch}"
      exit 0
    fi
  fi

  echo ""
  echo "Selected commit:"
  git log "${commit_hash}" --oneline -1
  echo ""
  echo "Commit details:"
  git show "${commit_hash}" --stat
  echo ""

  if [[ ${dry_run} -eq 1 ]]; then
    echo "=== DRY RUN -- no changes made ===" | tee -a "$logfile"
    echo "Would cherry-pick: ${commit_hash}" | tee -a "$logfile"
    git checkout "${original_branch}"
    exit 0
  fi

  # Check if commit modifies dev-only files (configurable warning)
  if [[ -n "${CGW_DEV_ONLY_FILES}" ]]; then
    local has_excluded_files=0
    for dev_file in ${CGW_DEV_ONLY_FILES}; do
      if git show "${commit_hash}" --name-only --format="" | grep -q "^${dev_file}"; then
        has_excluded_files=1
        break
      fi
    done

    if [[ ${has_excluded_files} -eq 1 ]]; then
      echo "[!] WARNING: This commit modifies configured dev-only files"
      echo "Dev-only files (CGW_DEV_ONLY_FILES):"
      for dev_file in ${CGW_DEV_ONLY_FILES}; do
        git show "${commit_hash}" --name-only --format="" | grep "^${dev_file}" || true
      done
      echo ""
      if [[ ${non_interactive} -eq 1 ]]; then
        echo "[FAIL] [Non-interactive] Aborting -- commit touches dev-only files" | tee -a "$logfile"
        git checkout "${original_branch}"
        exit 1
      fi
      read -r -p "Continue anyway? (yes/no): " continue_choice
      if [[ "${continue_choice}" != "yes" ]]; then
        echo ""
        log_message "Cherry-pick cancelled" "${logfile}"
        git checkout "${original_branch}"
        exit 0
      fi
    fi
  fi

  # [5/6] Create backup tag
  log_section_start "CREATE BACKUP TAG" "$logfile"

  if [[ -z "${timestamp:-}" ]]; then get_timestamp; fi
  local backup_tag="pre-cherry-pick-${timestamp}-$$"

  if git tag "${backup_tag}" >>"$logfile" 2>&1; then
    echo "[OK] Created backup tag: ${backup_tag}" | tee -a "$logfile"
    log_section_end "CREATE BACKUP TAG" "$logfile" "0"
  else
    echo "[!] Warning: Could not create backup tag" | tee -a "$logfile"
    log_section_end "CREATE BACKUP TAG" "$logfile" "1"
  fi
  echo "" | tee -a "$logfile"

  # [6/6] Cherry-pick
  log_section_start "GIT CHERRY-PICK" "$logfile"

  if run_git_with_logging "GIT CHERRY-PICK COMMIT" "$logfile" cherry-pick "${commit_hash}"; then
    trap - EXIT INT TERM
    log_section_end "GIT CHERRY-PICK" "$logfile" "0"
    echo "" | tee -a "$logfile"
    {
      echo "========================================"
      echo "[CHERRY-PICK SUMMARY]"
      echo "========================================"
    } | tee -a "$logfile"
    echo "[OK] CHERRY-PICK SUCCESSFUL" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    git log -1 --oneline | while read -r line; do echo "  Cherry-picked: $line" | tee -a "$logfile"; done
    echo "  Original commit: ${commit_hash}" | tee -a "$logfile"
    echo "  Backup tag: ${backup_tag}" | tee -a "$logfile"
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
    log_section_end "GIT CHERRY-PICK" "$logfile" "1"
    echo "" | tee -a "$logfile"
    echo "[!] Cherry-pick conflicts detected - analyzing..." | tee -a "$logfile"

    local conflict_status
    conflict_status=$(git status --short)

    # Auto-resolve DU (modify/delete) conflicts
    if printf '%s\n' "${conflict_status}" | grep -q "^DU "; then
      echo "  Found modify/delete conflicts -- auto-resolving..."
      local resolution_failed=0
      while read -r conflict_file; do
        if git rm "${conflict_file}" >/dev/null 2>&1; then
          echo "  [OK] Removed: ${conflict_file}"
        else
          echo "  [FAIL] Failed to remove ${conflict_file}"
          resolution_failed=1
        fi
      done < <(printf '%s\n' "${conflict_status}" | grep "^DU " | cut -c 4-)
      if [[ ${resolution_failed} -eq 0 ]]; then
        echo "[OK] Auto-resolved modify/delete conflicts" | tee -a "$logfile"
        conflict_status=$(git status --short)
      else
        echo "[FAIL] Auto-resolution failed for some files" | tee -a "$logfile"
        exit 1
      fi
    fi

    # DD (both deleted): auto-resolve by accepting deletion
    if printf '%s\n' "${conflict_status}" | grep -q "^DD "; then
      echo "  Found both-deleted conflicts -- auto-resolving..." | tee -a "$logfile"
      while read -r conflict_file; do
        git rm "${conflict_file}" >/dev/null 2>&1 || true
        echo "  [OK] Removed (both deleted): ${conflict_file}" | tee -a "$logfile"
      done < <(printf '%s\n' "${conflict_status}" | grep "^DD " | cut -c 4-)
      conflict_status=$(git status --short)
    fi

    # Remaining conflicts require manual resolution
    if printf '%s\n' "${conflict_status}" | grep -qE "^(UU|AU|AA|UD|AD|DA) "; then
      echo "" | tee -a "$logfile"
      printf '%s\n' "${conflict_status}" | grep -E "^(UU|AU|AA|UD|AD|DA) " | tee -a "$logfile"
      echo ""
      echo "Please resolve conflicts manually:"
      echo "  1. Edit conflicted files"
      echo "  2. git add <resolved files>"
      echo "  3. git cherry-pick --continue"
      echo ""
      echo "Or abort: git cherry-pick --abort && git checkout ${original_branch}"
      echo "Backup available: git reset --hard ${backup_tag}"
      exit 1
    fi

    # If only DU/DD were present and auto-resolved, prompt user to continue
    echo "" | tee -a "$logfile"
    echo "[OK] All conflicts auto-resolved. To complete the cherry-pick:" | tee -a "$logfile"
    echo "  git cherry-pick --continue" | tee -a "$logfile"
    echo "Backup available: git reset --hard ${backup_tag}" | tee -a "$logfile"
    exit 1
  fi
}

main "$@"
