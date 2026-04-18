#!/usr/bin/env bash
# bisect_helper.sh - Guided git bisect workflow for automated bug hunting
# Purpose: Wrap git bisect with a backup tag, auto-detect good/bad refs,
#          support automated test commands (git bisect run), and clean up
#          safely on interruption. See Pro Git Ch7 Debugging p.301-303.
# Usage: ./scripts/git/bisect_helper.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR          - Directory containing this script
#   PROJECT_ROOT        - Auto-detected git repo root (set by _config.sh)
#   logfile             - Set by init_logging
# Arguments:
#   --good <ref>         Known-good ref (default: latest semver tag, or HEAD~10)
#   --bad <ref>          Known-bad ref (default: HEAD)
#   --run <cmd>          Shell command to run per commit (exit 0 = good, non-0 = bad)
#   --abort              Abort an in-progress bisect session and clean up
#   --continue           Show current bisect state and continue guidance
#   --non-interactive    Skip confirmation prompts (requires --run)
#   --dry-run            Show what would happen without starting bisect
#   -h, --help           Show help
# Returns:
#   0 on success, 1 on failure

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/git/_common.sh
source "${SCRIPT_DIR}/_common.sh"

init_logging "bisect_helper"

_bisect_original_branch=""

_cleanup_bisect() {
  # If bisect is active and we were interrupted, abort it and return to original branch
  if git rev-parse --git-dir >/dev/null 2>&1; then
    if git bisect log >/dev/null 2>&1; then
      echo "" >&2
      echo "[!] Interrupted -- aborting bisect session" >&2
      git bisect reset 2>/dev/null || true
    fi
  fi
  if [[ -n "${_bisect_original_branch}" ]]; then
    local current
    current=$(git branch --show-current 2>/dev/null || true)
    if [[ -n "${current}" ]] && [[ "${current}" != "${_bisect_original_branch}" ]]; then
      git checkout "${_bisect_original_branch}" 2>/dev/null || true
    fi
  fi
}
trap _cleanup_bisect EXIT INT TERM

_show_help() {
  echo "Usage: ./scripts/git/bisect_helper.sh [OPTIONS]"
  echo ""
  echo "Guided git bisect for finding the commit that introduced a bug."
  echo "Creates a backup tag before starting and cleans up on interruption."
  echo ""
  echo "Options:"
  echo "  --good <ref>         Known-good commit/tag (default: latest semver tag or HEAD~10)"
  echo "  --bad <ref>          Known-bad commit/tag (default: HEAD)"
  echo "  --run <cmd>          Test command: exit 0 = good commit, non-0 = bad commit"
  echo "                       Enables automated bisect (git bisect run)"
  echo "  --abort              Abort an in-progress bisect session"
  echo "  --continue           Show current bisect status"
  echo "  --non-interactive    Skip prompts (requires --run)"
  echo "  --dry-run            Show plan without starting bisect"
  echo "  -h, --help           Show this help"
  echo ""
  echo "Examples:"
  echo "  # Automated: find first bad commit using a test script"
  echo "  ./scripts/git/bisect_helper.sh --good v1.0.0 --run 'bash tests/smoke_test.sh'"
  echo ""
  echo "  # Manual: interactive guided bisect"
  echo "  ./scripts/git/bisect_helper.sh --good v1.0.0 --bad HEAD"
  echo "  # Then: git bisect good / git bisect bad after each checkout"
  echo ""
  echo "  # Abort an in-progress session"
  echo "  ./scripts/git/bisect_helper.sh --abort"
  echo ""
  echo "Notes:"
  echo "  - A backup tag is created before bisect starts (pre-bisect-TIMESTAMP)"
  echo "  - git bisect run requires the test command to be repeatable and exit cleanly"
  echo "  - Exit code 125 in --run command = skip this commit (git bisect skip)"
  echo "  - On completion, bisect is reset and you are returned to your original branch"
}

main() {
  if [[ $# -eq 0 ]] || [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    _show_help
    exit 0
  fi

  local good_ref=""
  local bad_ref="HEAD"
  local run_cmd=""
  local non_interactive=0
  local dry_run=0
  local do_abort=0
  local do_continue=0

  [[ "${CGW_NON_INTERACTIVE:-0}" == "1" ]] && non_interactive=1

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help | -h)
        _show_help
        exit 0
        ;;
      --good)
        good_ref="${2:-}"
        shift
        ;;
      --bad)
        bad_ref="${2:-HEAD}"
        shift
        ;;
      --run)
        run_cmd="${2:-}"
        shift
        ;;
      --abort) do_abort=1 ;;
      --continue) do_continue=1 ;;
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
    echo "Bisect Helper Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Branch: $(git branch --show-current 2>/dev/null || echo 'detached')"
  } >"$logfile"

  # -- Handle --abort ---------------------------------------------------------
  if [[ ${do_abort} -eq 1 ]]; then
    _cmd_abort
    return $?
  fi

  # -- Handle --continue ------------------------------------------------------
  if [[ ${do_continue} -eq 1 ]]; then
    _cmd_status
    return $?
  fi

  # -- Validate: --non-interactive requires --run -----------------------------
  if [[ ${non_interactive} -eq 1 ]] && [[ -z "${run_cmd}" ]]; then
    err "--non-interactive requires --run <cmd> (automated bisect)"
    err "Without a test command, bisect requires interactive good/bad marking"
    exit 1
  fi

  # -- Check for already-active bisect ---------------------------------------
  if git bisect log >/dev/null 2>&1; then
    echo "[!] An active bisect session is already in progress." >&2
    echo "  Run './scripts/git/bisect_helper.sh --abort' to stop it first." >&2
    if [[ ${non_interactive} -eq 0 ]]; then
      read -r -p "  Abort existing session and start fresh? (yes/no): " fresh_confirm
      if [[ "${fresh_confirm}" != "yes" ]]; then
        echo "Cancelled"
        exit 0
      fi
      git bisect reset 2>/dev/null || true
    else
      err "Cannot start bisect -- session already active (abort it first)"
      exit 1
    fi
  fi

  # -- Auto-detect good_ref ---------------------------------------------------
  if [[ -z "${good_ref}" ]]; then
    good_ref=$(git tag -l "v[0-9]*" | sort -V | tail -1 2>/dev/null || true)
    if [[ -z "${good_ref}" ]]; then
      # Fall back to HEAD~10 (or root if fewer than 10 commits)
      local commit_count
      commit_count=$(git rev-list --count HEAD 2>/dev/null || echo "0")
      if [[ "${commit_count}" -gt 10 ]]; then
        good_ref="HEAD~10"
      else
        good_ref=$(git rev-list --max-parents=0 HEAD 2>/dev/null | head -1 || true)
      fi
    fi
    echo "Auto-detected good ref: ${good_ref}" | tee -a "$logfile"
  fi

  # -- Validate refs ---------------------------------------------------------
  if ! git rev-parse "${bad_ref}" >/dev/null 2>&1; then
    err "Invalid --bad ref: ${bad_ref}"
    exit 1
  fi
  if [[ -z "${good_ref}" ]] || ! git rev-parse "${good_ref}" >/dev/null 2>&1; then
    err "Invalid --good ref: ${good_ref:-<empty>}"
    err "Specify with --good <ref> (tag, commit hash, or branch name)"
    exit 1
  fi

  # -- Compute commit range --------------------------------------------------
  local commit_count_range
  commit_count_range=$(git rev-list --count "${good_ref}..${bad_ref}" 2>/dev/null || echo "?")
  local steps_estimate="?"
  if [[ "${commit_count_range}" != "?" ]] && [[ "${commit_count_range}" -gt 0 ]]; then
    # log2(N) steps estimate using awk
    steps_estimate=$(awk "BEGIN{printf \"%d\", log(${commit_count_range})/log(2) + 1}")
  fi

  # -- Show plan -------------------------------------------------------------
  echo "=== Git Bisect Helper ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"
  echo "  Good ref:   ${good_ref} ($(git log -1 --format='%h %s' "${good_ref}" 2>/dev/null || echo 'unknown'))" | tee -a "$logfile"
  echo "  Bad ref:    ${bad_ref} ($(git log -1 --format='%h %s' "${bad_ref}" 2>/dev/null || echo 'unknown'))" | tee -a "$logfile"
  echo "  Range:      ${commit_count_range} commits to search (~${steps_estimate} bisect steps)" | tee -a "$logfile"
  if [[ -n "${run_cmd}" ]]; then
    echo "  Test cmd:   ${run_cmd}" | tee -a "$logfile"
  else
    echo "  Mode:       Manual (mark commits good/bad interactively)" | tee -a "$logfile"
  fi
  echo "" | tee -a "$logfile"

  if [[ "${dry_run}" -eq 1 ]]; then
    echo "--- Dry run: no changes made ---"
    echo "Would run:"
    echo "  git bisect start"
    echo "  git bisect bad ${bad_ref}"
    echo "  git bisect good ${good_ref}"
    if [[ -n "${run_cmd}" ]]; then
      echo "  git bisect run ${run_cmd}"
    else
      echo "  (manual: git bisect good | git bisect bad for each checkout)"
    fi
    exit 0
  fi

  if [[ ${non_interactive} -eq 0 ]] && [[ -z "${run_cmd}" ]]; then
    echo "Manual bisect mode -- after each checkout, run your test then:"
    echo "  git bisect good   (if the bug is NOT present in this commit)"
    echo "  git bisect bad    (if the bug IS present in this commit)"
    echo "  git bisect skip   (if you cannot test this commit)"
    echo ""
    read -r -p "Start bisect session? (yes/no): " start_confirm
    if [[ "${start_confirm}" != "yes" ]]; then
      echo "Cancelled"
      exit 0
    fi
  fi

  # -- Save original branch --------------------------------------------------
  _bisect_original_branch=$(git branch --show-current 2>/dev/null || true)

  # -- Create backup tag -----------------------------------------------------
  get_timestamp
  local backup_tag="pre-bisect-${timestamp}-$$"
  if git tag "${backup_tag}" 2>/dev/null; then
    echo "[OK] Backup tag: ${backup_tag}" | tee -a "$logfile"
  else
    echo "[!] Could not create backup tag (continuing)" | tee -a "$logfile"
  fi
  echo "" | tee -a "$logfile"

  log_section_start "GIT BISECT" "$logfile"

  # -- Start bisect ---------------------------------------------------------
  if ! git bisect start 2>&1 | tee -a "$logfile"; then
    err "Failed to start bisect session"
    log_section_end "GIT BISECT" "$logfile" "1"
    exit 1
  fi

  if ! git bisect bad "${bad_ref}" 2>&1 | tee -a "$logfile"; then
    err "Failed to mark bad ref: ${bad_ref}"
    git bisect reset 2>/dev/null || true
    log_section_end "GIT BISECT" "$logfile" "1"
    exit 1
  fi

  if ! git bisect good "${good_ref}" 2>&1 | tee -a "$logfile"; then
    err "Failed to mark good ref: ${good_ref}"
    git bisect reset 2>/dev/null || true
    log_section_end "GIT BISECT" "$logfile" "1"
    exit 1
  fi

  # -- Automated or manual ---------------------------------------------------
  local bisect_result=0

  if [[ -n "${run_cmd}" ]]; then
    echo "Running automated bisect: git bisect run ${run_cmd}" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    # git bisect run exits 0 when first-bad-commit is found, non-0 on error
    # shellcheck disable=SC2086  # run_cmd intentionally word-splits (it's a shell command)
    if git bisect run ${run_cmd} 2>&1 | tee -a "$logfile"; then
      bisect_result=0
    else
      bisect_result=1
    fi
    log_section_end "GIT BISECT" "$logfile" "${bisect_result}"

    # Capture the identified commit before reset
    local first_bad
    first_bad=$(git bisect log 2>/dev/null | grep "^# first bad commit:" | tail -1 | sed 's/^# first bad commit: \[//' | sed 's/\].*//' || true)

    # Reset bisect (returns to original branch)
    git bisect reset 2>/dev/null || true
    _bisect_original_branch="" # Already reset, don't cleanup again

    echo "" | tee -a "$logfile"
    if [[ ${bisect_result} -eq 0 ]]; then
      echo "[OK] BISECT COMPLETE" | tee -a "$logfile"
      if [[ -n "${first_bad}" ]]; then
        echo "  First bad commit: ${first_bad}" | tee -a "$logfile"
        echo "  $(git log -1 --oneline "${first_bad}" 2>/dev/null || true)" | tee -a "$logfile"
      fi
    else
      echo "[FAIL] Bisect run encountered errors -- check log: $logfile" | tee -a "$logfile"
    fi
  else
    # Manual mode -- bisect is active, user marks good/bad interactively
    log_section_end "GIT BISECT" "$logfile" "0"
    echo "" | tee -a "$logfile"
    echo "[OK] BISECT STARTED" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "  git has checked out a commit for you to test."
    echo "  Current commit: $(git log -1 --oneline 2>/dev/null || true)"
    echo ""
    echo "  After testing, run:"
    echo "    git bisect good   -- bug not present"
    echo "    git bisect bad    -- bug is present"
    echo "    git bisect skip   -- cannot test this commit"
    echo ""
    echo "  To abort at any time:"
    echo "    ./scripts/git/bisect_helper.sh --abort"
    echo ""
    echo "  Restore point (if needed):"
    echo "    git checkout ${backup_tag}  # or: git bisect reset"
    echo ""
    # Don't run cleanup trap -- bisect is intentionally left active
    _bisect_original_branch=""
  fi

  {
    echo ""
    echo "End Time: $(date)"
  } >>"$logfile"

  echo "Full log: $logfile"
  return ${bisect_result}
}

# ---------------------------------------------------------------------------
# Abort an active bisect session
# ---------------------------------------------------------------------------
_cmd_abort() {
  echo "=== Abort Bisect Session ===" | tee -a "$logfile"
  echo ""

  if ! git bisect log >/dev/null 2>&1; then
    echo "  No active bisect session found."
    exit 0
  fi

  echo "  Active bisect log:"
  git bisect log 2>/dev/null | head -10 | sed 's/^/    /'
  echo ""

  if git bisect reset 2>&1 | tee -a "$logfile"; then
    echo ""
    echo "[OK] Bisect aborted -- returned to original branch"
    echo "  Current branch: $(git branch --show-current 2>/dev/null || echo 'detached')"
  else
    err "bisect reset failed -- run 'git bisect reset' manually"
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Show status of active bisect session
# ---------------------------------------------------------------------------
_cmd_status() {
  echo "=== Bisect Session Status ===" | tee -a "$logfile"
  echo ""

  if ! git bisect log >/dev/null 2>&1; then
    echo "  No active bisect session."
    echo ""
    echo "  Start one with:"
    echo "    ./scripts/git/bisect_helper.sh --good <ref> [--run <cmd>]"
    exit 0
  fi

  echo "  Current commit: $(git log -1 --oneline 2>/dev/null || true)"
  echo ""
  echo "  Bisect log:"
  git bisect log 2>/dev/null | sed 's/^/    /'
  echo ""
  echo "  To mark current commit:"
  echo "    git bisect good   -- bug not present here"
  echo "    git bisect bad    -- bug is present here"
  echo "    git bisect skip   -- cannot test this commit"
  echo ""
  echo "  To abort:"
  echo "    ./scripts/git/bisect_helper.sh --abort"
}

main "$@"
