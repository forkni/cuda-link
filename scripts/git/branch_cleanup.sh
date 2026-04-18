#!/usr/bin/env bash
# branch_cleanup.sh - Prune stale local and remote branches
# Purpose: Delete local branches already merged into the target branch,
#          prune stale remote-tracking refs, and optionally clean up old
#          backup tags. Safe by default (--dry-run). See Pro Git Ch3 p.85-87.
# Usage: ./scripts/git/branch_cleanup.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR             - Directory containing this script
#   PROJECT_ROOT           - Auto-detected git repo root (set by _config.sh)
#   CGW_TARGET_BRANCH      - Branch used as merge base (default: main)
#   CGW_PROTECTED_BRANCHES - Branches never deleted (default: $CGW_TARGET_BRANCH)
#   CGW_SOURCE_BRANCH      - Source branch (also protected by default)
# Arguments:
#   --dry-run          Preview changes without deleting anything (default)
#   --execute          Actually delete branches and prune refs
#   --remote           Also prune stale remote-tracking refs (git remote prune)
#   --tags             Also clean up old CGW backup tags (pre-merge-backup-*, pre-cherry-pick-*, pre-docs-merge-*, pre-bisect-*, pre-rebase-*, pre-undo-commit-*)
#   --older-than <N>   Only delete backup tags older than N days (default: 30)
#   --non-interactive  Skip confirmation prompts
#   -h, --help         Show help
# Returns:
#   0 on success, 1 on failure

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/git/_common.sh
source "${SCRIPT_DIR}/_common.sh"

main() {
  local execute=0
  local prune_remote=0
  local clean_tags=0
  local older_than_days=30
  local non_interactive=0

  [[ "${CGW_NON_INTERACTIVE:-0}" == "1" ]] && non_interactive=1

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help | -h)
        echo "Usage: ./scripts/git/branch_cleanup.sh [OPTIONS]"
        echo ""
        echo "Clean up merged local branches, stale remote-tracking refs, and backup tags."
        echo "Defaults to dry-run (preview only). Pass --execute to actually delete."
        echo ""
        echo "Options:"
        echo "  --execute          Delete merged branches and prune (without this, only previews)"
        echo "  --remote           Prune stale remote-tracking refs (git remote prune \${CGW_REMOTE})"
        echo "  --tags             Clean up old CGW backup tags (pre-merge-backup-*, pre-cherry-pick-*, pre-docs-merge-*, pre-bisect-*, pre-rebase-*, pre-undo-commit-*)"
        echo "  --older-than <N>   Only delete backup tags older than N days (default: 30)"
        echo "  --non-interactive  Skip confirmation prompts"
        echo "  -h, --help         Show this help"
        echo ""
        echo "Protected branches (never deleted):"
        echo "  ${CGW_TARGET_BRANCH}, ${CGW_SOURCE_BRANCH}"
        echo "  Plus: CGW_PROTECTED_BRANCHES setting"
        echo ""
        echo "Environment:"
        echo "  CGW_REMOTE   Remote name (default: origin)"
        echo ""
        echo "Examples:"
        echo "  ./scripts/git/branch_cleanup.sh                    # dry-run preview"
        echo "  ./scripts/git/branch_cleanup.sh --execute          # delete merged branches"
        echo "  ./scripts/git/branch_cleanup.sh --execute --remote # also prune remote refs"
        echo "  ./scripts/git/branch_cleanup.sh --execute --tags --older-than 14"
        exit 0
        ;;
      --execute) execute=1 ;;
      --remote) prune_remote=1 ;;
      --tags) clean_tags=1 ;;
      --older-than)
        older_than_days="${2:-30}"
        shift
        ;;
      --non-interactive) non_interactive=1 ;;
      *)
        echo "[ERROR] Unknown flag: $1" >&2
        exit 1
        ;;
    esac
    shift
  done

  cd "${PROJECT_ROOT}" || {
    err "Cannot find project root"
    exit 1
  }

  local mode_label="DRY RUN"
  [[ ${execute} -eq 1 ]] && mode_label="EXECUTE"

  echo "=== Branch Cleanup [${mode_label}] ==="
  echo ""
  [[ ${execute} -eq 0 ]] && echo "  (dry run -- pass --execute to actually delete)" && echo ""

  # Build set of protected branches
  local -a protected=("${CGW_TARGET_BRANCH}" "${CGW_SOURCE_BRANCH}")
  for pb in ${CGW_PROTECTED_BRANCHES:-}; do
    protected+=("${pb}")
  done

  local current_branch
  current_branch=$(git branch --show-current 2>/dev/null || echo "")

  # -- [1] Merged local branches ---------------------------------------------
  echo "--- [1] Merged Local Branches (merged into ${CGW_TARGET_BRANCH}) ---"

  local -a merged_branches=()
  while IFS= read -r branch; do
    # Skip empty
    [[ -z "${branch}" ]] && continue

    # Skip protected branches
    local is_protected=0
    for pb in "${protected[@]}"; do
      [[ "${branch}" == "${pb}" ]] && is_protected=1 && break
    done
    [[ ${is_protected} -eq 1 ]] && continue

    # Skip current branch
    [[ "${branch}" == "${current_branch}" ]] && continue

    merged_branches+=("${branch}")
  done < <(git for-each-ref --format='%(refname:short)' refs/heads --merged="${CGW_TARGET_BRANCH}" 2>/dev/null)

  if [[ ${#merged_branches[@]} -eq 0 ]]; then
    echo "  [OK] No merged local branches to clean up"
  else
    echo "  Branches merged into ${CGW_TARGET_BRANCH}:"
    for branch in "${merged_branches[@]}"; do
      local last_commit
      last_commit=$(git log -1 --format="%h %s (%ar)" "${branch}" 2>/dev/null || echo "(unknown)")
      echo "    ${branch}  -- ${last_commit}"
    done
    echo ""

    if [[ ${execute} -eq 1 ]]; then
      if [[ ${non_interactive} -eq 0 ]]; then
        read -r -p "  Delete ${#merged_branches[@]} merged branch(es)? (yes/no): " confirm
        if [[ "${confirm}" != "yes" ]]; then
          echo "  Skipped local branch deletion"
        else
          _delete_local_branches "${merged_branches[@]}"
        fi
      else
        _delete_local_branches "${merged_branches[@]}"
      fi
    else
      echo "  Would delete: ${#merged_branches[@]} branch(es)"
    fi
  fi
  echo ""

  # -- [2] Remote-tracking refs ----------------------------------------------
  if [[ ${prune_remote} -eq 1 ]]; then
    echo "--- [2] Stale Remote-Tracking Refs ---"

    if [[ ${execute} -eq 1 ]]; then
      echo "  Pruning stale refs from ${CGW_REMOTE}..."
      if git remote prune "${CGW_REMOTE}" 2>&1 | grep -E "pruned|\\[pruned\\]" | sed 's/^/  /'; then
        : # output shown inline
      else
        echo "  [OK] No stale remote-tracking refs"
      fi
    else
      # Dry-run: show what would be pruned
      local stale_count=0
      while IFS= read -r ref; do
        echo "  Would prune: ${ref}"
        ((stale_count++)) || true
      done < <(git remote prune --dry-run "${CGW_REMOTE}" 2>&1 | grep "\\[would prune\\]" | awk '{print $NF}')
      [[ ${stale_count} -eq 0 ]] && echo "  [OK] No stale remote-tracking refs"
    fi
    echo ""
  fi

  # -- [3] Backup tags -------------------------------------------------------
  if [[ ${clean_tags} -eq 1 ]]; then
    echo "--- [3] Old Backup Tags (older than ${older_than_days} days) ---"

    local cutoff_epoch
    # POSIX-compatible date arithmetic: go back N days in seconds
    cutoff_epoch=$(date +%s)
    cutoff_epoch=$((cutoff_epoch - older_than_days * 86400))

    local -a old_tags=()
    while IFS= read -r tag; do
      local tag_epoch
      tag_epoch=$(git log -1 --format="%ct" "${tag}" 2>/dev/null || echo "0")
      if [[ ${tag_epoch} -le ${cutoff_epoch} ]]; then
        old_tags+=("${tag}")
      fi
    done < <(git tag -l "pre-merge-backup-*" "pre-cherry-pick-*" "pre-docs-merge-*" "pre-bisect-*" "pre-rebase-*" "pre-undo-commit-*" 2>/dev/null | sort)

    if [[ ${#old_tags[@]} -eq 0 ]]; then
      echo "  [OK] No backup tags older than ${older_than_days} days"
    else
      echo "  Old backup tags:"
      local total_all
      total_all=$(git tag -l "pre-merge-backup-*" "pre-cherry-pick-*" "pre-docs-merge-*" "pre-bisect-*" "pre-rebase-*" "pre-undo-commit-*" 2>/dev/null | wc -l | tr -d ' ')
      for tag in "${old_tags[@]}"; do
        local tag_date
        tag_date=$(git log -1 --format="%ar" "${tag}" 2>/dev/null || echo "unknown")
        echo "    ${tag}  (${tag_date})"
      done
      echo ""
      echo "  (${#old_tags[@]} old / ${total_all} total backup tags)"

      if [[ ${execute} -eq 1 ]]; then
        if [[ ${non_interactive} -eq 0 ]]; then
          read -r -p "  Delete ${#old_tags[@]} old backup tag(s)? (yes/no): " confirm_tags
          if [[ "${confirm_tags}" != "yes" ]]; then
            echo "  Skipped backup tag deletion"
          else
            _delete_tags "${old_tags[@]}"
          fi
        else
          _delete_tags "${old_tags[@]}"
        fi
      else
        echo "  Would delete: ${#old_tags[@]} backup tag(s)"
      fi
    fi
    echo ""
  fi

  echo "=== Done ==="
  if [[ ${execute} -eq 0 ]]; then
    echo "Run with --execute to apply changes."
  fi
}

_delete_local_branches() {
  local deleted=0 failed=0
  for branch in "$@"; do
    if git branch -d "${branch}" 2>/dev/null; then
      echo "  [OK] Deleted: ${branch}"
      ((deleted++)) || true
    else
      echo "  [FAIL] Failed: ${branch} (may not be fully merged -- use git branch -D to force)"
      ((failed++)) || true
    fi
  done
  echo "  Deleted: ${deleted}, Failed: ${failed}"
}

_delete_tags() {
  local deleted=0
  for tag in "$@"; do
    if git tag -d "${tag}" 2>/dev/null; then
      echo "  [OK] Deleted: ${tag}"
      ((deleted++)) || true
    else
      echo "  [FAIL] Failed to delete: ${tag}" >&2
    fi
  done
  echo "  Deleted: ${deleted} tag(s)"
}

main "$@"
