#!/usr/bin/env bash
# sync_branches.sh - Sync local branches with remote via fetch + rebase
# Purpose: Keep branches up-to-date with origin
# Usage: ./scripts/git/sync_branches.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR          - Directory containing this script
#   PROJECT_ROOT        - Auto-detected git repo root (set by _config.sh)
#   logfile             - Set by init_logging
#   CGW_SOURCE_BRANCH   - Source branch name (default: development)
#   CGW_TARGET_BRANCH   - Target branch name (default: main)
# Arguments:
#   --all               Sync both source and target branches (default: current only)
#   --branch <name>     Sync a specific named branch (overrides --all)
#   --dry-run           Show what would be synced without making changes
#   --prune             Pass --prune to git fetch (remove stale remote-tracking refs)
#   --non-interactive   Skip prompts
#   -h, --help          Show help
# Returns:
#   0 on successful sync, 1 on failure

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/git/_common.sh
source "${SCRIPT_DIR}/_common.sh"

init_logging "sync_branches"

# ============================================================================
# CLEANUP TRAP
# ============================================================================

_sync_original_branch=""
_sync_did_checkout=0
_SYNC_AUTOSTASH=0
_sync_dry_run=0

_cleanup_sync() {
  local current
  current=$(git branch --show-current 2>/dev/null || true)
  if [[ ${_sync_did_checkout} -eq 1 ]] && [[ -n "${_sync_original_branch}" ]] &&
    [[ "${current}" != "${_sync_original_branch}" ]]; then
    echo "" >&2
    echo "[!] Interrupted -- returning to: ${_sync_original_branch}" >&2
    git rebase --abort 2>/dev/null || true
    git checkout "${_sync_original_branch}" 2>/dev/null || true
  fi
}
trap _cleanup_sync EXIT INT TERM

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# sync_one_branch - Fetch and rebase a single branch against origin.
# Arguments:
#   $1 - branch name
# Returns: 0 on success, 1 on failure
sync_one_branch() {
  local branch="$1"
  local current_branch
  current_branch=$(git branch --show-current)

  echo "" | tee -a "$logfile"
  echo "--- Syncing ${branch} ---" | tee -a "$logfile"

  if ! git show-ref --verify --quiet "refs/heads/${branch}"; then
    echo "  [!] Branch '${branch}' does not exist locally -- skipping" | tee -a "$logfile"
    return 0
  fi

  if ! git show-ref --verify --quiet "refs/remotes/${CGW_REMOTE}/${branch}"; then
    echo "  [!] No remote tracking branch '${CGW_REMOTE}/${branch}' -- skipping" | tee -a "$logfile"
    return 0
  fi

  local behind ahead
  behind=$(git rev-list --count "HEAD..${CGW_REMOTE}/${branch}" 2>/dev/null || echo "0")
  ahead=$(git rev-list --count "${CGW_REMOTE}/${branch}..HEAD" 2>/dev/null || echo "0")

  # In dry-run mode, report status and skip the actual sync
  if [[ ${_sync_dry_run} -eq 1 ]]; then
    if [[ "${current_branch}" != "${branch}" ]]; then
      # Use remote ref directly for accurate counts when not on this branch
      local remote_behind remote_ahead
      remote_behind=$(git rev-list --count "refs/heads/${branch}..${CGW_REMOTE}/${branch}" 2>/dev/null || echo "0")
      remote_ahead=$(git rev-list --count "${CGW_REMOTE}/${branch}..refs/heads/${branch}" 2>/dev/null || echo "0")
      behind="${remote_behind}"
      ahead="${remote_ahead}"
    fi
    echo "  Local: ${ahead} ahead, ${behind} behind ${CGW_REMOTE}/${branch}" | tee -a "$logfile"
    if [[ "${behind}" -eq 0 ]]; then
      echo "  [OK] Already up-to-date with ${CGW_REMOTE}/${branch}" | tee -a "$logfile"
    else
      echo "  Would pull --rebase to sync ${behind} commit(s)" | tee -a "$logfile"
    fi
    return 0
  fi

  if [[ "${current_branch}" != "${branch}" ]]; then
    if ! git checkout "${branch}" >>"$logfile" 2>&1; then
      echo "  [FAIL] Failed to checkout ${branch}" | tee -a "$logfile"
      return 1
    fi
    echo "  Switched to ${branch}" | tee -a "$logfile"
    _sync_did_checkout=1
    # Recompute ahead/behind from this branch's perspective
    behind=$(git rev-list --count "HEAD..${CGW_REMOTE}/${branch}" 2>/dev/null || echo "0")
    ahead=$(git rev-list --count "${CGW_REMOTE}/${branch}..HEAD" 2>/dev/null || echo "0")
  fi

  echo "  Local: ${ahead} ahead, ${behind} behind ${CGW_REMOTE}/${branch}" | tee -a "$logfile"

  if [[ "${behind}" -eq 0 ]]; then
    echo "  [OK] Already up-to-date with ${CGW_REMOTE}/${branch}" | tee -a "$logfile"
    return 0
  fi

  if [[ "${ahead}" -gt 0 ]]; then
    echo "  [!] Diverged: ${ahead} local commits will be rebased on top of ${behind} remote commits" | tee -a "$logfile"
  fi

  local rebase_args=(pull --rebase "${CGW_REMOTE}" "${branch}")
  [[ "${_SYNC_AUTOSTASH}" == "1" ]] && rebase_args=(pull --rebase --autostash "${CGW_REMOTE}" "${branch}")
  if run_git_with_logging "GIT REBASE ${branch}" "$logfile" "${rebase_args[@]}"; then
    echo "  [OK] ${branch} synced successfully" | tee -a "$logfile"
    return 0
  else
    echo "  [FAIL] Rebase failed for ${branch}" | tee -a "$logfile"
    echo "  Aborting rebase..." | tee -a "$logfile"
    git rebase --abort 2>/dev/null || true
    echo "  Manual action needed: git pull --rebase ${CGW_REMOTE} ${branch}" | tee -a "$logfile"
    return 1
  fi
}

# ============================================================================
# MAIN
# ============================================================================

main() {
  local sync_all=0
  local sync_branch=""
  local dry_run=0
  local prune=0
  local non_interactive=0

  while [[ $# -gt 0 ]]; do
    case "${1}" in
      --help | -h)
        echo "Usage: ./scripts/git/sync_branches.sh [OPTIONS]"
        echo ""
        echo "Sync local branches with remote (CGW_REMOTE) via fetch + rebase."
        echo ""
        echo "Options:"
        echo "  --all               Sync both source and target branches (default: current only)"
        echo "  --branch <name>     Sync a specific named branch (overrides --all)"
        echo "  --dry-run           Show what would be synced without making changes"
        echo "  --prune             Remove stale remote-tracking refs during fetch"
        echo "  --non-interactive   Abort (instead of prompt) if uncommitted changes found"
        echo "  -h, --help          Show this help"
        echo ""
        echo "Behavior:"
        echo "  - Runs git fetch ${CGW_REMOTE} first to update remote refs"
        echo "  - Uses git pull --rebase (preserves clean linear history)"
        echo "  - With --all: switches between branches, returns to starting branch"
        echo "  - With --branch: syncs only the named branch, returns to starting branch"
        echo "  - Warns if local diverges from remote before rebasing"
        echo ""
        echo "Configuration:"
        echo "  CGW_SOURCE_BRANCH   Source branch (default: development)"
        echo "  CGW_TARGET_BRANCH   Target branch (default: main)"
        echo ""
        echo "Environment:"
        echo "  CGW_NON_INTERACTIVE=1   Same as --non-interactive"
        echo "  CGW_REMOTE              Remote name (default: origin)"
        exit 0
        ;;
      --all) sync_all=1 ;;
      --branch)
        sync_branch="${2:-}"
        shift
        ;;
      --dry-run) dry_run=1 ;;
      --prune) prune=1 ;;
      --non-interactive) non_interactive=1 ;;
      *)
        echo "[ERROR] Unknown flag: $1" >&2
        exit 1
        ;;
    esac
    shift
  done

  [[ "${CGW_NON_INTERACTIVE:-0}" == "1" ]] && non_interactive=1
  _sync_dry_run=${dry_run}

  {
    echo "========================================="
    echo "Sync Branches Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } >"$logfile"

  echo "=== Branch Sync ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"
  echo "Workflow Log: ${logfile}" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  if [[ ${dry_run} -eq 1 ]]; then
    echo "=== DRY RUN -- no changes will be made ===" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
  fi

  cd "${PROJECT_ROOT}" || {
    err "Cannot find project root"
    exit 1
  }

  _sync_original_branch=$(git branch --show-current)

  # [1/4] Check working tree
  echo "[1/4] Checking working tree..." | tee -a "$logfile"
  if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo "[!] Uncommitted changes detected -- will auto-stash during rebase" | tee -a "$logfile"
    git status --short | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    if [[ ${non_interactive} -eq 1 ]]; then
      echo "[Non-interactive] Auto-stash enabled (--autostash)" | tee -a "$logfile"
    else
      read -r -p "Auto-stash changes and sync? (yes/no): " uncommitted_choice
      if [[ "${uncommitted_choice}" != "yes" ]]; then
        echo "Aborted -- commit or stash manually before syncing" | tee -a "$logfile"
        exit 0
      fi
    fi
    _SYNC_AUTOSTASH=1
  else
    echo "[OK] Working tree clean" | tee -a "$logfile"
  fi
  echo "" | tee -a "$logfile"

  # [2/4] Fetch from origin
  log_section_start "[2/4] GIT FETCH" "$logfile"
  local fetch_args=(fetch "${CGW_REMOTE}")
  [[ ${prune} -eq 1 ]] && fetch_args=(fetch "${CGW_REMOTE}" --prune)
  echo "Fetching from ${CGW_REMOTE}${prune:+ (--prune)}..." | tee -a "$logfile"
  if git "${fetch_args[@]}" >>"$logfile" 2>&1; then
    echo "[OK] Fetch complete" | tee -a "$logfile"
    log_section_end "[2/4] GIT FETCH" "$logfile" "0"
  else
    echo "[FAIL] Fetch failed -- check network/auth" | tee -a "$logfile"
    log_section_end "[2/4] GIT FETCH" "$logfile" "1"
    exit 1
  fi
  echo "" | tee -a "$logfile"

  # [3/4] Sync branches
  log_section_start "[3/4] SYNC BRANCHES" "$logfile"

  local sync_failed=0

  if [[ -n "${sync_branch}" ]]; then
    sync_one_branch "${sync_branch}" || sync_failed=1
  elif [[ ${sync_all} -eq 1 ]]; then
    sync_one_branch "${CGW_SOURCE_BRANCH}" || sync_failed=1
    sync_one_branch "${CGW_TARGET_BRANCH}" || sync_failed=1
  else
    sync_one_branch "${_sync_original_branch}" || sync_failed=1
  fi

  log_section_end "[3/4] SYNC BRANCHES" "$logfile" "${sync_failed}"
  echo "" | tee -a "$logfile"

  # Return to original branch if we moved
  local current_after
  current_after=$(git branch --show-current)
  if [[ "${current_after}" != "${_sync_original_branch}" ]]; then
    git checkout "${_sync_original_branch}" >>"$logfile" 2>&1
    echo "Returned to: ${_sync_original_branch}" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
  fi

  # [4/4] Summary
  {
    echo "========================================"
    echo "[4/4] SYNC SUMMARY"
    echo "========================================"
  } | tee -a "$logfile"

  if [[ ${dry_run} -eq 1 ]]; then
    echo "[OK] DRY RUN COMPLETE -- no changes made" | tee -a "$logfile"
  elif [[ ${sync_failed} -eq 0 ]]; then
    echo "[OK] SYNC SUCCESSFUL" | tee -a "$logfile"
  else
    echo "[!] SYNC COMPLETED WITH ERRORS" | tee -a "$logfile"
    echo "  Check log for details: ${logfile}" | tee -a "$logfile"
  fi

  {
    echo ""
    echo "End Time: $(date)"
  } | tee -a "$logfile"

  echo "Full log: $logfile"

  return ${sync_failed}
}

main "$@"
