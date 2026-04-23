#!/usr/bin/env bash
# repo_health.sh - Repository health check and maintenance
# Purpose: Run integrity checks (git fsck), trigger garbage collection (git gc),
#          report repository size, detect large files, and check ref consistency.
#          Especially useful for TouchDesigner projects with large binary files.
# Usage: ./scripts/git/repo_health.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR   - Directory containing this script
#   PROJECT_ROOT - Auto-detected git repo root (set by _config.sh)
# Arguments:
#   --gc         Run git gc (garbage collection) in addition to checks
#   --full       Run git fsck --full (slower but more thorough)
#   --large <N>  Report files larger than N MB (default: 10)
#   -h, --help   Show help
# Returns:
#   0 on healthy repo, 1 if issues found

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

# human_size - Convert bytes to human-readable string (POSIX: uses awk, not bc)
human_size() {
  local bytes="$1"
  if ((bytes >= 1073741824)); then
    awk "BEGIN{printf \"%.1f GB\", ${bytes}/1073741824}"
  elif ((bytes >= 1048576)); then
    awk "BEGIN{printf \"%.1f MB\", ${bytes}/1048576}"
  elif ((bytes >= 1024)); then
    awk "BEGIN{printf \"%.1f KB\", ${bytes}/1024}"
  else
    printf "%d B" "${bytes}"
  fi
}

main() {
  local run_gc=0
  local full_fsck=0
  local large_threshold_mb=10

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help | -h)
        echo "Usage: ./scripts/git/repo_health.sh [OPTIONS]"
        echo ""
        echo "Check repository health, find large files, and optionally run maintenance."
        echo ""
        echo "Options:"
        echo "  --gc          Run git gc (garbage collection) -- removes unreachable objects"
        echo "  --full        Run git fsck --full (slower, more thorough)"
        echo "  --large <N>   Report files >N MB in git history (default: 10)"
        echo "  -h, --help    Show this help"
        echo ""
        echo "Checks performed:"
        echo "  1. Repository integrity (git fsck)"
        echo "  2. Object store size"
        echo "  3. Large files in git history"
        echo "  4. Stale backup tags count"
        echo "  5. Branch divergence summary"
        echo ""
        echo "Environment:"
        echo "  CGW_REMOTE   Remote name for divergence check (default: origin)"
        exit 0
        ;;
      --gc) run_gc=1 ;;
      --full) full_fsck=1 ;;
      --large)
        large_threshold_mb="${2:-10}"
        shift
        ;;
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

  local overall_ok=0

  echo "=== Repository Health Check ==="
  echo "Repository: ${PROJECT_ROOT}"
  echo "Date:       $(date)"
  echo ""

  # -- [1] Integrity check (git fsck) ---------------------------------------
  echo "--- [1/5] Integrity Check ---"
  local fsck_args=("--no-reflogs")
  [[ ${full_fsck} -eq 1 ]] && fsck_args=("--full")

  local fsck_output
  if fsck_output=$(git fsck "${fsck_args[@]}" 2>&1); then
    echo "  [OK] No integrity issues found"
  else
    echo "  [!] Integrity issues detected:"
    echo "${fsck_output}" | grep -v "^Checking" | head -20
    overall_ok=1
  fi
  echo ""

  # -- [2] Object store size -------------------------------------------------
  echo "--- [2/5] Object Store Size ---"
  # Use git count-objects (cross-platform, no GNU du needed)
  local obj_size_kb=0 pack_count=0
  while IFS=': ' read -r key val; do
    case "${key}" in
      size) obj_size_kb=$((obj_size_kb + val)) ;;
      size-pack) obj_size_kb=$((obj_size_kb + val)) ;;
      packs) pack_count="${val}" ;;
    esac
  done < <(git count-objects -v 2>/dev/null)
  local obj_size_bytes=$((obj_size_kb * 1024))
  echo "  Objects:  $(human_size "${obj_size_bytes}")"
  echo "  Packs:    ${pack_count}"

  if [[ ${pack_count} -gt 3 ]]; then
    echo "  [!] Many pack files (${pack_count}) -- consider running: git gc"
  fi

  # Worktree size -- use POSIX du -sk (1024-byte blocks), available everywhere
  local wt_size_kb
  wt_size_kb=$(du -sk . 2>/dev/null | cut -f1 || echo "0")
  local wt_size_bytes=$((wt_size_kb * 1024))
  echo "  Worktree: $(human_size "${wt_size_bytes}") (includes .git)"
  echo ""

  # -- [3] Large files in history --------------------------------------------
  echo "--- [3/5] Large Files in History (>${large_threshold_mb}MB) ---"
  local threshold_bytes=$((large_threshold_mb * 1048576))
  local large_count=0

  # Use git cat-file to find large blobs
  while IFS=' ' read -r size hash; do
    if [[ ${size} -gt ${threshold_bytes} ]]; then
      # Find the path for this blob
      local path
      path=$(git log --all --find-object="${hash}" --oneline --name-only 2>/dev/null | grep -v "^[0-9a-f]" | head -1 || echo "(unknown path)")
      printf "  %s  %s\n" "$(human_size "${size}")" "${path:-${hash}}"
      ((large_count++)) || true
    fi
  done < <(git cat-file --batch-check='%(objectsize) %(objectname)' --batch-all-objects 2>/dev/null | awk '$1 ~ /^[0-9]+$/')

  if [[ ${large_count} -eq 0 ]]; then
    echo "  [OK] No files exceed ${large_threshold_mb}MB threshold"
  else
    echo ""
    echo "  [!] ${large_count} large file(s) found in git history"
    echo "  Consider: git lfs track for future additions"
    overall_ok=1
  fi
  echo ""

  # -- [4] Backup tag count --------------------------------------------------
  echo "--- [4/5] Backup Tags ---"
  local backup_count
  backup_count=$(git tag -l "pre-merge-backup-*" "pre-cherry-pick-*" "pre-docs-merge-*" "pre-bisect-*" "pre-rebase-*" "pre-undo-commit-*" 2>/dev/null | wc -l | tr -d ' ')
  echo "  Backup tags: ${backup_count}"

  if [[ ${backup_count} -gt 20 ]]; then
    echo "  [!] Many backup tags -- consider running:"
    echo "    ./scripts/git/branch_cleanup.sh --tags --execute"
  elif [[ ${backup_count} -gt 0 ]]; then
    echo "  Most recent: $(git tag -l 'pre-merge-backup-*' 'pre-cherry-pick-*' 'pre-docs-merge-*' 'pre-bisect-*' 'pre-rebase-*' 'pre-undo-commit-*' | sort -r | head -1)"
  fi
  echo ""

  # -- [5] Branch summary ----------------------------------------------------
  echo "--- [5/5] Branch Status ---"
  local current_branch
  current_branch=$(git branch --show-current 2>/dev/null || echo "(detached)")
  echo "  Current:  ${current_branch}"

  for branch in "${CGW_SOURCE_BRANCH}" "${CGW_TARGET_BRANCH}"; do
    if git show-ref --verify --quiet "refs/heads/${branch}" 2>/dev/null; then
      local ahead behind remote_ref="refs/remotes/${CGW_REMOTE}/${branch}"
      if git show-ref --verify --quiet "${remote_ref}" 2>/dev/null; then
        ahead=$(git rev-list --count "${CGW_REMOTE}/${branch}..${branch}" 2>/dev/null || echo "?")
        behind=$(git rev-list --count "${branch}..${CGW_REMOTE}/${branch}" 2>/dev/null || echo "?")
        echo "  ${branch}: ${ahead} ahead, ${behind} behind ${CGW_REMOTE}"
      else
        echo "  ${branch}: (no remote tracking branch)"
      fi
    fi
  done
  echo ""

  # -- Garbage collection (optional) ----------------------------------------
  if [[ ${run_gc} -eq 1 ]]; then
    echo "--- Garbage Collection ---"
    echo "Running git gc --auto..."
    if git gc --auto 2>&1 | grep -v "^Auto packing\|^Counting\|^Delta\|^Compressing\|^Writing\|^Total" | head -10; then
      echo "  [OK] Garbage collection complete"
    else
      echo "  [OK] Nothing to collect"
    fi
    echo ""
  fi

  # -- Summary ---------------------------------------------------------------
  echo "=== Summary ==="
  if [[ ${overall_ok} -eq 0 ]]; then
    echo "[OK] Repository is healthy"
  else
    echo "[!] Issues found -- review output above"
    echo ""
    echo "Common fixes:"
    echo "  Integrity issues:  git fsck --full"
    echo "  Pack files:        git gc"
    echo "  Large files:       git lfs track '*.ext'"
  fi

  return ${overall_ok}
}

main "$@"
