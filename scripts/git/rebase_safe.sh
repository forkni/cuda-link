#!/usr/bin/env bash
# rebase_safe.sh - Safe rebase wrapper with backup tag and validation
# Purpose: Wrap common rebase workflows (rebase onto branch, squash last N commits,
#          abort/continue in-progress rebase) with a backup tag before any
#          destructive operation, pre-rebase pushed-commit detection, and
#          autostash for dirty working trees.
#          See Pro Git Ch3 Rebasing p.101-110, Ch7 Rewriting History p.249-256.
# Usage: ./scripts/git/rebase_safe.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR          - Directory containing this script
#   PROJECT_ROOT        - Auto-detected git repo root (set by _config.sh)
#   logfile             - Set by init_logging
#   CGW_TARGET_BRANCH   - Default upstream ref for --onto if not specified
# Arguments:
#   --onto <branch>      Rebase current branch onto this branch (default: CGW_TARGET_BRANCH)
#   --squash-last <N>    Interactive squash of last N commits (opens editor or autosquash)
#   --autosquash         Apply fixup!/squash! commit prefixes automatically
#   --autostash          Auto-stash dirty working tree before rebase (restore after)
#   --abort              Abort an in-progress rebase
#   --continue           Continue after resolving conflicts
#   --skip               Skip the current conflicting commit
#   --non-interactive    Skip confirmation prompts
#   --dry-run            Show what would happen without rebasing
#   -h, --help           Show help
# Returns:
#   0 on success, 1 on failure or conflict

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/git/_common.sh
source "${SCRIPT_DIR}/_common.sh"

init_logging "rebase_safe"

_rebase_original_branch=""
_rebase_stash_created=0

_cleanup_rebase() {
  # Only restore stash if rebase was aborted mid-way and we created one
  if [[ ${_rebase_stash_created} -eq 1 ]]; then
    if git rebase --show-current-patch >/dev/null 2>&1 || [[ -d "${PROJECT_ROOT}/.git/rebase-merge" ]] || [[ -d "${PROJECT_ROOT}/.git/rebase-apply" ]]; then
      # Rebase still in progress -- don't auto-pop stash, user needs to resolve
      echo "" >&2
      echo "[!] Rebase was interrupted with uncommitted changes stashed." >&2
      echo "  Resolve conflicts, then: git rebase --continue" >&2
      echo "  To restore your stash: git stash pop" >&2
    fi
  fi
}
trap _cleanup_rebase EXIT INT TERM

_show_help() {
  echo "Usage: ./scripts/git/rebase_safe.sh [OPTIONS]"
  echo ""
  echo "Safe rebase wrapper. Creates a backup tag before any destructive operation."
  echo "Refuses to rebase commits that have already been pushed (history-safe by default)."
  echo ""
  echo "Options:"
  echo "  --onto <branch>      Rebase current branch onto this branch"
  echo "                       (default: ${CGW_TARGET_BRANCH})"
  echo "  --squash-last <N>    Interactively squash the last N commits"
  echo "  --autosquash         Apply fixup!/squash! commit prefixes automatically"
  echo "                       (used with --squash-last)"
  echo "  --autostash          Auto-stash dirty working tree before rebase"
  echo "  --abort              Abort the current in-progress rebase"
  echo "  --continue           Continue after manually resolving conflicts"
  echo "  --skip               Skip the current conflicting commit"
  echo "  --non-interactive    Skip confirmation prompts"
  echo "  --dry-run            Show what would happen without rebasing"
  echo "  -h, --help           Show this help"
  echo ""
  echo "Examples:"
  echo "  # Rebase feature branch onto main"
  echo "  ./scripts/git/rebase_safe.sh --onto main"
  echo ""
  echo "  # Squash last 3 commits into one (opens editor)"
  echo "  ./scripts/git/rebase_safe.sh --squash-last 3"
  echo ""
  echo "  # Squash with auto-applied fixup!/squash! markers"
  echo "  ./scripts/git/rebase_safe.sh --squash-last 5 --autosquash"
  echo ""
  echo "  # Rebase with dirty working tree (auto-stash)"
  echo "  ./scripts/git/rebase_safe.sh --onto main --autostash"
  echo ""
  echo "  # Abort an in-progress rebase"
  echo "  ./scripts/git/rebase_safe.sh --abort"
  echo ""
  echo "  # Continue after resolving conflicts"
  echo "  ./scripts/git/rebase_safe.sh --continue"
  echo ""
  echo "[!] WARNING: Rebasing rewrites history. Never rebase commits already pushed"
  echo "   to a shared branch. This script will warn you if that is the case."
  echo ""
  echo "Environment:"
  echo "  CGW_REMOTE   Remote name (default: origin)"
}

main() {
  if [[ $# -eq 0 ]] || [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    _show_help
    exit 0
  fi

  local onto_ref=""
  local squash_last=0
  local autosquash=0
  local autostash=0
  local do_abort=0
  local do_continue=0
  local do_skip=0
  local non_interactive=0
  local dry_run=0

  [[ "${CGW_NON_INTERACTIVE:-0}" == "1" ]] && non_interactive=1

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help | -h)
        _show_help
        exit 0
        ;;
      --onto)
        onto_ref="${2:-}"
        shift
        ;;
      --squash-last)
        squash_last="${2:-0}"
        shift
        ;;
      --autosquash) autosquash=1 ;;
      --autostash) autostash=1 ;;
      --abort) do_abort=1 ;;
      --continue) do_continue=1 ;;
      --skip) do_skip=1 ;;
      --non-interactive) non_interactive=1 ;;
      --dry-run) dry_run=1 ;;
      *)
        err "Unknown flag: $1"
        echo "Run with --help to see available options" >&2
        exit 1
        ;;
    esac
    shift
  done

  cd "${PROJECT_ROOT}" || {
    err "Cannot find project root"
    exit 1
  }

  {
    echo "========================================="
    echo "Rebase Safe Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Branch: $(git branch --show-current 2>/dev/null || echo 'detached')"
  } >"$logfile"

  # -- Handle in-progress rebase operations ----------------------------------
  if [[ ${do_abort} -eq 1 ]]; then
    _cmd_abort
    return $?
  fi
  if [[ ${do_continue} -eq 1 ]]; then
    _cmd_continue
    return $?
  fi
  if [[ ${do_skip} -eq 1 ]]; then
    _cmd_skip
    return $?
  fi

  # -- Validate: mutually exclusive main operations ---------------------------
  local has_onto=0
  local has_squash=0
  [[ -n "${onto_ref}" ]] && has_onto=1
  [[ "${squash_last}" -gt 0 ]] && has_squash=1

  if [[ ${has_onto} -eq 0 ]] && [[ ${has_squash} -eq 0 ]]; then
    err "Specify an operation: --onto <branch> or --squash-last <N>"
    echo "Run with --help to see available options" >&2
    exit 1
  fi
  if [[ ${has_onto} -eq 1 ]] && [[ ${has_squash} -eq 1 ]]; then
    err "Use either --onto or --squash-last, not both"
    exit 1
  fi

  # -- Check for already-active rebase ---------------------------------------
  if [[ -d "${PROJECT_ROOT}/.git/rebase-merge" ]] || [[ -d "${PROJECT_ROOT}/.git/rebase-apply" ]]; then
    echo "[!] A rebase is already in progress." >&2
    echo "  Resolve conflicts then:" >&2
    echo "    ./scripts/git/rebase_safe.sh --continue" >&2
    echo "    ./scripts/git/rebase_safe.sh --abort" >&2
    exit 1
  fi

  # -- Set default onto_ref ---------------------------------------------------
  if [[ ${has_onto} -eq 1 ]] && [[ -z "${onto_ref}" ]]; then
    onto_ref="${CGW_TARGET_BRANCH}"
    echo "Using default onto ref: ${onto_ref}" | tee -a "$logfile"
  fi

  if [[ ${has_onto} -eq 1 ]]; then
    _cmd_rebase_onto "${onto_ref}" "${autostash}" "${non_interactive}" "${dry_run}"
  else
    _cmd_squash_last "${squash_last}" "${autosquash}" "${autostash}" "${non_interactive}" "${dry_run}"
  fi
}

# ---------------------------------------------------------------------------
# Shared: create backup tag before any destructive operation
# ---------------------------------------------------------------------------
_create_backup_tag() {
  get_timestamp
  local backup_tag="pre-rebase-${timestamp}-$$"
  if git tag "${backup_tag}" 2>/dev/null; then
    echo "[OK] Backup tag: ${backup_tag}" | tee -a "$logfile"
    echo "  To restore: git checkout ${backup_tag}" | tee -a "$logfile"
  else
    echo "[!] Could not create backup tag (continuing)" | tee -a "$logfile"
  fi
  echo "" | tee -a "$logfile"
  echo "${backup_tag}"
}

# ---------------------------------------------------------------------------
# Shared: warn if current branch has commits already pushed to origin
# ---------------------------------------------------------------------------
_check_pushed_commits() {
  local upstream_count="${1}"
  local non_interactive="${2}"

  if [[ "${upstream_count}" -gt 0 ]]; then
    echo "[!] WARNING: ${upstream_count} commit(s) on this branch have already been pushed." | tee -a "$logfile"
    echo "  Rebasing will rewrite history -- you will need to force-push after rebase." | tee -a "$logfile"
    echo "  This is SAFE only on personal/feature branches, NEVER on shared branches." | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    if [[ "${non_interactive}" -eq 1 ]]; then
      err "Refusing to rebase pushed commits in non-interactive mode (history-safety)"
      err "Use interactive mode or acknowledge the risk with --non-interactive after force-push consent"
      exit 1
    fi
    read -r -p "  Rebase anyway? (yes/no): " pushed_confirm
    if [[ "${pushed_confirm}" != "yes" ]]; then
      echo "Cancelled"
      exit 0
    fi
  fi
}

# ---------------------------------------------------------------------------
# Shared: handle dirty working tree
# ---------------------------------------------------------------------------
_handle_dirty_tree() {
  local autostash="${1}"

  if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    if [[ "${autostash}" -eq 1 ]]; then
      echo "  Stashing uncommitted changes..." | tee -a "$logfile"
      if git stash push -m "rebase_safe auto-stash $(date +%Y%m%d_%H%M%S)" 2>&1 | tee -a "$logfile"; then
        _rebase_stash_created=1
        echo "  [OK] Changes stashed" | tee -a "$logfile"
      else
        err "Failed to stash changes -- resolve conflicts first"
        exit 1
      fi
    else
      err "Working tree has uncommitted changes. Use --autostash or commit/stash first."
      git diff --stat | head -10 | sed 's/^/  /'
      exit 1
    fi
  fi
}

# ---------------------------------------------------------------------------
# Shared: restore stash after successful rebase
# ---------------------------------------------------------------------------
_restore_stash_if_needed() {
  if [[ ${_rebase_stash_created} -eq 1 ]]; then
    echo "" | tee -a "$logfile"
    echo "  Restoring stashed changes..." | tee -a "$logfile"
    if git stash pop 2>&1 | tee -a "$logfile"; then
      _rebase_stash_created=0
      echo "  [OK] Stash restored" | tee -a "$logfile"
    else
      echo "  [!] Stash pop had conflicts -- resolve manually with: git stash show / git stash pop" | tee -a "$logfile"
    fi
  fi
}

# ---------------------------------------------------------------------------
# Operation: --onto <branch>
# ---------------------------------------------------------------------------
_cmd_rebase_onto() {
  local onto_ref="$1" autostash="$2" non_interactive="$3" dry_run="$4"

  echo "=== Rebase onto ${onto_ref} ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  local current_branch
  current_branch=$(git branch --show-current 2>/dev/null || true)

  if [[ -z "${current_branch}" ]]; then
    err "Cannot rebase in detached HEAD state. Check out a branch first."
    exit 1
  fi

  # Validate onto_ref
  if ! git rev-parse "${onto_ref}" >/dev/null 2>&1; then
    err "Invalid --onto ref: ${onto_ref}"
    exit 1
  fi

  # Check if onto_ref is a local or remote branch and fetch latest
  if git rev-parse "${CGW_REMOTE}/${onto_ref}" >/dev/null 2>&1; then
    echo "  Fetching latest ${onto_ref} from ${CGW_REMOTE}..." | tee -a "$logfile"
    git fetch "${CGW_REMOTE}" "${onto_ref}" 2>&1 | tee -a "$logfile" || true
  fi

  # Count pushed commits (commits on current branch not on origin/current_branch)
  local pushed_count=0
  if git rev-parse "${CGW_REMOTE}/${current_branch}" >/dev/null 2>&1; then
    pushed_count=$(git rev-list --count "${CGW_REMOTE}/${current_branch}..HEAD" 2>/dev/null || echo "0")
  fi

  # Count commits that would be rebased
  local rebase_commit_count
  rebase_commit_count=$(git rev-list --count "${onto_ref}..HEAD" 2>/dev/null || echo "?")

  # Show plan
  echo "  Current branch: ${current_branch}" | tee -a "$logfile"
  echo "  Onto:           ${onto_ref} ($(git log -1 --format='%h %s' "${onto_ref}" 2>/dev/null || echo 'unknown'))" | tee -a "$logfile"
  echo "  Commits to rebase: ${rebase_commit_count}" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  if [[ "${rebase_commit_count}" == "0" ]]; then
    echo "  Already up to date with ${onto_ref} -- nothing to rebase."
    exit 0
  fi

  if [[ "${dry_run}" -eq 1 ]]; then
    echo "--- Dry run: no changes made ---"
    echo "Would run:"
    echo "  git rebase ${onto_ref}"
    if [[ "${pushed_count}" -gt 0 ]]; then
      echo "  (then: git push --force-with-lease -- ${pushed_count} commits already pushed)"
    fi
    exit 0
  fi

  # Warn about pushed commits
  _check_pushed_commits "${pushed_count}" "${non_interactive}"

  # Handle dirty tree
  _handle_dirty_tree "${autostash}"

  # Confirmation
  if [[ "${non_interactive}" -eq 0 ]]; then
    read -r -p "  Rebase ${current_branch} onto ${onto_ref}? (yes/no): " rebase_confirm
    if [[ "${rebase_confirm}" != "yes" ]]; then
      echo "Cancelled"
      _restore_stash_if_needed
      exit 0
    fi
  fi

  # Create backup
  _create_backup_tag

  log_section_start "GIT REBASE ONTO" "$logfile"

  local rebase_exit=0
  if ! git rebase "${onto_ref}" 2>&1 | tee -a "$logfile"; then
    rebase_exit=1
  fi

  log_section_end "GIT REBASE ONTO" "$logfile" "${rebase_exit}"

  if [[ ${rebase_exit} -ne 0 ]]; then
    echo "" | tee -a "$logfile"
    echo "[FAIL] REBASE HIT CONFLICTS" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "  Conflicting files:"
    git diff --name-only --diff-filter=U 2>/dev/null | sed 's/^/    /' || true
    echo ""
    echo "  Resolve conflicts, then:"
    echo "    git add <resolved-files>"
    echo "    ./scripts/git/rebase_safe.sh --continue"
    echo ""
    echo "  To abort and restore:"
    echo "    ./scripts/git/rebase_safe.sh --abort"
    echo ""
    echo "Full log: $logfile"
    # Don't restore stash -- user needs to resolve rebase first
    _rebase_stash_created=0
    exit 1
  fi

  _restore_stash_if_needed

  echo "" | tee -a "$logfile"
  echo "[OK] REBASE COMPLETE" | tee -a "$logfile"
  echo "  $(git log -1 --oneline)" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  if [[ "${pushed_count}" -gt 0 ]]; then
    echo "  [!] Your branch was previously pushed -- force-push required:"
    echo "    ./scripts/git/push_validated.sh --force-with-lease"
    echo ""
  fi

  echo "Full log: $logfile"
}

# ---------------------------------------------------------------------------
# Operation: --squash-last <N>
# ---------------------------------------------------------------------------
_cmd_squash_last() {
  local squash_n="$1" autosquash="$2" autostash="$3" non_interactive="$4" dry_run="$5"

  echo "=== Squash Last ${squash_n} Commits ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  local current_branch
  current_branch=$(git branch --show-current 2>/dev/null || true)

  if [[ -z "${current_branch}" ]]; then
    err "Cannot rebase in detached HEAD state. Check out a branch first."
    exit 1
  fi

  # Validate N
  if ! [[ "${squash_n}" =~ ^[0-9]+$ ]] || [[ "${squash_n}" -lt 2 ]]; then
    err "--squash-last requires a number >= 2 (got: ${squash_n})"
    exit 1
  fi

  local commit_count
  commit_count=$(git rev-list --count HEAD 2>/dev/null || echo "0")
  if [[ "${commit_count}" -lt "${squash_n}" ]]; then
    err "Cannot squash ${squash_n} commits -- branch only has ${commit_count} commit(s)"
    exit 1
  fi

  # Count pushed commits in the squash range
  local pushed_count=0
  if git rev-parse "${CGW_REMOTE}/${current_branch}" >/dev/null 2>&1; then
    # Count how many of the last N commits exist on origin
    pushed_count=$(git rev-list --count "${CGW_REMOTE}/${current_branch}..HEAD" 2>/dev/null || echo "0")
    # Clamp to squash range
    if [[ "${pushed_count}" -gt "${squash_n}" ]]; then
      pushed_count="${squash_n}"
    fi
  fi

  # Show commits to be squashed
  echo "  Commits to squash:" | tee -a "$logfile"
  git log --oneline -"${squash_n}" 2>/dev/null | sed 's/^/    /' | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  if [[ "${dry_run}" -eq 1 ]]; then
    echo "--- Dry run: no changes made ---"
    local squash_flag=""
    [[ "${autosquash}" -eq 1 ]] && squash_flag=" --autosquash"
    echo "Would run:"
    echo "  git rebase -i${squash_flag} HEAD~${squash_n}"
    if [[ "${pushed_count}" -gt 0 ]]; then
      echo "  (then: git push --force-with-lease -- ${pushed_count} commits already pushed)"
    fi
    exit 0
  fi

  # Warn about pushed commits
  _check_pushed_commits "${pushed_count}" "${non_interactive}"

  # Handle dirty tree
  _handle_dirty_tree "${autostash}"

  if [[ "${non_interactive}" -eq 0 ]] && [[ "${autosquash}" -eq 0 ]]; then
    echo "  An editor will open for you to mark commits (squash, fixup, reword, etc.)"
    echo "  Change 'pick' to 'squash' (or 's') to fold a commit into the one above it."
    echo ""
    read -r -p "  Open interactive rebase for last ${squash_n} commits? (yes/no): " squash_confirm
    if [[ "${squash_confirm}" != "yes" ]]; then
      echo "Cancelled"
      _restore_stash_if_needed
      exit 0
    fi
  elif [[ "${non_interactive}" -eq 1 ]] && [[ "${autosquash}" -eq 0 ]]; then
    err "Interactive squash requires an editor -- use --autosquash for non-interactive squash"
    err "(commits must be prefixed with 'squash!' or 'fixup!' for --autosquash to work)"
    exit 1
  fi

  # Create backup
  _create_backup_tag

  log_section_start "GIT REBASE INTERACTIVE" "$logfile"

  local rebase_exit=0
  local rebase_args=(-i "HEAD~${squash_n}")
  [[ "${autosquash}" -eq 1 ]] && rebase_args=(-i --autosquash "HEAD~${squash_n}")

  # shellcheck disable=SC2068  # Intentional: rebase_args expands correctly
  if ! git rebase "${rebase_args[@]}" 2>&1 | tee -a "$logfile"; then
    rebase_exit=1
  fi

  log_section_end "GIT REBASE INTERACTIVE" "$logfile" "${rebase_exit}"

  if [[ ${rebase_exit} -ne 0 ]]; then
    echo "" | tee -a "$logfile"
    echo "[FAIL] INTERACTIVE REBASE HIT CONFLICTS" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "  Resolve conflicts, then:"
    echo "    git add <resolved-files>"
    echo "    ./scripts/git/rebase_safe.sh --continue"
    echo ""
    echo "  To abort and restore:"
    echo "    ./scripts/git/rebase_safe.sh --abort"
    echo ""
    echo "Full log: $logfile"
    _rebase_stash_created=0
    exit 1
  fi

  _restore_stash_if_needed

  echo "" | tee -a "$logfile"
  echo "[OK] SQUASH COMPLETE" | tee -a "$logfile"
  echo "  $(git log -1 --oneline)" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  if [[ "${pushed_count}" -gt 0 ]]; then
    echo "  [!] Your branch was previously pushed -- force-push required:"
    echo "    ./scripts/git/push_validated.sh --force-with-lease"
    echo ""
  fi

  echo "Full log: $logfile"
}

# ---------------------------------------------------------------------------
# Operation: --abort
# ---------------------------------------------------------------------------
_cmd_abort() {
  echo "=== Abort Rebase ===" | tee -a "$logfile"
  echo ""

  if [[ ! -d "${PROJECT_ROOT}/.git/rebase-merge" ]] && [[ ! -d "${PROJECT_ROOT}/.git/rebase-apply" ]]; then
    echo "  No rebase in progress."
    exit 0
  fi

  echo "  Aborting rebase..." | tee -a "$logfile"
  if git rebase --abort 2>&1 | tee -a "$logfile"; then
    echo ""
    echo "[OK] Rebase aborted -- returned to: $(git branch --show-current 2>/dev/null || echo 'previous state')"
    if [[ ${_rebase_stash_created} -eq 1 ]]; then
      echo ""
      echo "  Your auto-stash is still saved. To restore:"
      echo "    git stash pop"
    fi
  else
    err "rebase --abort failed -- run 'git rebase --abort' manually"
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Operation: --continue
# ---------------------------------------------------------------------------
_cmd_continue() {
  echo "=== Continue Rebase ===" | tee -a "$logfile"
  echo ""

  if [[ ! -d "${PROJECT_ROOT}/.git/rebase-merge" ]] && [[ ! -d "${PROJECT_ROOT}/.git/rebase-apply" ]]; then
    echo "  No rebase in progress."
    exit 0
  fi

  # Check for unresolved conflicts
  local unresolved
  unresolved=$(git diff --name-only --diff-filter=U 2>/dev/null || true)
  if [[ -n "${unresolved}" ]]; then
    err "Unresolved conflicts still present -- resolve and 'git add' them first:"
    # shellcheck disable=SC2001  # sed needed for per-line prefix on multi-line string
    echo "${unresolved}" | sed 's/^/  /'
    exit 1
  fi

  echo "  Continuing rebase..." | tee -a "$logfile"
  if GIT_EDITOR=true git rebase --continue 2>&1 | tee -a "$logfile"; then
    echo ""
    echo "[OK] Rebase continued"
    # Check if rebase is now complete
    if [[ ! -d "${PROJECT_ROOT}/.git/rebase-merge" ]] && [[ ! -d "${PROJECT_ROOT}/.git/rebase-apply" ]]; then
      echo "  Rebase complete!"
      _restore_stash_if_needed
    else
      echo "  More conflicts to resolve -- fix them, then run --continue again."
    fi
  else
    err "rebase --continue failed -- check for remaining conflicts"
    echo ""
    echo "  Conflicting files:"
    git diff --name-only --diff-filter=U 2>/dev/null | sed 's/^/    /' || true
    echo ""
    echo "  To abort: ./scripts/git/rebase_safe.sh --abort"
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Operation: --skip
# ---------------------------------------------------------------------------
_cmd_skip() {
  echo "=== Skip Rebase Commit ===" | tee -a "$logfile"
  echo ""

  if [[ ! -d "${PROJECT_ROOT}/.git/rebase-merge" ]] && [[ ! -d "${PROJECT_ROOT}/.git/rebase-apply" ]]; then
    echo "  No rebase in progress."
    exit 0
  fi

  echo "  [!] Skipping current commit -- its changes will be dropped."
  echo "  Current patch:"
  git log ORIG_HEAD -1 --oneline 2>/dev/null | sed 's/^/    /' || true
  echo ""

  if git rebase --skip 2>&1 | tee -a "$logfile"; then
    echo ""
    echo "[OK] Commit skipped"
    if [[ ! -d "${PROJECT_ROOT}/.git/rebase-merge" ]] && [[ ! -d "${PROJECT_ROOT}/.git/rebase-apply" ]]; then
      echo "  Rebase complete!"
      _restore_stash_if_needed
    fi
  else
    err "rebase --skip failed"
    exit 1
  fi
}

main "$@"
