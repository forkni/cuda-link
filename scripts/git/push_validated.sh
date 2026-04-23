#!/usr/bin/env bash
# push_validated.sh - Validated git push with safety checks
# Purpose: Push with remote reachability check, behind-remote warning, and force-push protection
# Usage: ./scripts/git/push_validated.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR              - Directory containing this script
#   PROJECT_ROOT            - Auto-detected git repo root (set by _config.sh)
#   logfile                 - Set by init_logging
#   CGW_PROTECTED_BRANCHES  - Branches requiring --force confirmation (default: target branch)
# Arguments:
#   --non-interactive   Skip prompts
#   --dry-run           Show what would be pushed without pushing
#   --skip-lint         Skip pre-push lint check
#   --force             Allow force-push (uses --force-with-lease)
#   --branch <name>     Override push target branch (default: current branch)
#   -h, --help          Show help
# Returns:
#   0 on successful push, 1 on failure or safety abort

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/git/_common.sh
source "${SCRIPT_DIR}/_common.sh"

init_logging "push_validated"

main() {
  local non_interactive=0
  local dry_run=0
  local skip_lint=0
  local skip_md_lint=0
  local force_push=0
  local target_branch=""

  while [[ $# -gt 0 ]]; do
    case "${1}" in
      --help | -h)
        echo "Usage: ./scripts/git/push_validated.sh [OPTIONS]"
        echo ""
        echo "Push the current branch to origin with safety checks."
        echo ""
        echo "Options:"
        echo "  --non-interactive   Skip all prompts"
        echo "  --dry-run           Show what would be pushed without pushing"
        echo "  --skip-lint         Skip pre-push lint check (all lint)"
        echo "  --skip-md-lint      Skip markdown lint only in pre-push check"
        echo "  --force             Allow force-push (uses --force-with-lease)"
        echo "  --branch <name>     Override push target branch (default: current branch)"
        echo "  -h, --help          Show this help"
        echo ""
        echo "Safety checks performed:"
        echo "  - Verifies the configured remote (CGW_REMOTE) is reachable"
        echo "  - Blocks force-push to protected branches without explicit --force"
        echo "  - Warns if local branch is behind remote (may overwrite remote work)"
        echo "  - Optional pre-push lint check"
        echo ""
        echo "Environment:"
        echo "  CGW_NON_INTERACTIVE=1         Same as --non-interactive"
        echo "  CGW_REMOTE                    Remote name (default: origin)"
        echo "  CGW_PROTECTED_BRANCHES=<list> Space-separated protected branch names"
        echo "  (Also: CLAUDE_GIT_NON_INTERACTIVE, CLAUDE_GIT_NO_VENV)"
        exit 0
        ;;
      --non-interactive) non_interactive=1 ;;
      --dry-run) dry_run=1 ;;
      --skip-lint) skip_lint=1 ;;
      --skip-md-lint) skip_md_lint=1 ;;
      --force) force_push=1 ;;
      --branch)
        target_branch="${2:-}"
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
  [[ "${CGW_SKIP_LINT:-0}" == "1" ]] && skip_lint=1
  [[ "${CGW_SKIP_MD_LINT:-0}" == "1" ]] && skip_md_lint=1

  {
    echo "========================================="
    echo "Push Validated Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } >"$logfile"

  echo "=== Validated Push ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"
  echo "Workflow Log: ${logfile}" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  cd "${PROJECT_ROOT}" || {
    err "Cannot find project root"
    exit 1
  }

  # [1/5] Determine push branch
  log_section_start "BRANCH CHECK" "$logfile"

  local current_branch
  current_branch=$(git branch --show-current)

  if [[ -z "${target_branch}" ]]; then
    target_branch="${current_branch}"
  fi

  if [[ -z "${target_branch}" ]]; then
    err "Cannot determine current branch (detached HEAD?)"
    log_section_end "BRANCH CHECK" "$logfile" "1"
    exit 1
  fi

  echo "Branch to push: ${target_branch}" | tee -a "$logfile"
  echo "Remote: ${CGW_REMOTE}" | tee -a "$logfile"

  # Check force-push protection against configured protected branches
  local is_protected=0
  for protected in ${CGW_PROTECTED_BRANCHES}; do
    if [[ "${target_branch}" == "${protected}" ]]; then
      is_protected=1
      break
    fi
  done

  if [[ ${is_protected} -eq 1 ]] && [[ ${force_push} -eq 1 ]]; then
    echo "[!] WARNING: Force-push to protected branch '${target_branch}' requested!" | tee -a "$logfile"
    echo "  This rewrites remote history and affects all collaborators." | tee -a "$logfile"
    if [[ ${non_interactive} -eq 0 ]]; then
      read -r -p "  Type 'FORCE' to confirm force-push to ${target_branch}: " force_confirm
      if [[ "${force_confirm}" != "FORCE" ]]; then
        echo "  Aborted" | tee -a "$logfile"
        log_section_end "BRANCH CHECK" "$logfile" "1"
        exit 1
      fi
    else
      echo "  [Non-interactive] Aborting -- force-push to protected branch requires manual confirmation" | tee -a "$logfile"
      log_section_end "BRANCH CHECK" "$logfile" "1"
      exit 1
    fi
  elif [[ ${is_protected} -eq 1 ]] && [[ ${force_push} -eq 0 ]]; then
    echo "[OK] Pushing to ${target_branch} (normal push)" | tee -a "$logfile"
  fi

  log_section_end "BRANCH CHECK" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [2/5] Check remote reachability
  log_section_start "REMOTE CHECK" "$logfile"

  echo "Checking remote ${CGW_REMOTE}..." | tee -a "$logfile"
  if ! git ls-remote --exit-code "${CGW_REMOTE}" HEAD >/dev/null 2>&1; then
    err "Remote '${CGW_REMOTE}' is not reachable. Check network/auth."
    log_section_end "REMOTE CHECK" "$logfile" "1"
    exit 1
  fi
  echo "[OK] Remote '${CGW_REMOTE}' is reachable" | tee -a "$logfile"

  # Check if local is behind remote
  git fetch "${CGW_REMOTE}" "${target_branch}" >>"$logfile" 2>&1 || true
  local behind
  behind=$(git rev-list --count "HEAD..${CGW_REMOTE}/${target_branch}" 2>/dev/null || echo "0")
  if [[ "${behind}" -gt 0 ]]; then
    echo "[!] WARNING: Local branch is ${behind} commit(s) behind ${CGW_REMOTE}/${target_branch}" | tee -a "$logfile"
    echo "  A normal push may fail or overwrite remote changes." | tee -a "$logfile"
    echo "  Consider: ./scripts/git/sync_branches.sh" | tee -a "$logfile"
    if [[ ${non_interactive} -eq 0 ]] && [[ ${force_push} -eq 0 ]]; then
      read -r -p "  Continue push anyway? (yes/no): " behind_choice
      if [[ "${behind_choice}" != "yes" ]]; then
        echo "  Aborted" | tee -a "$logfile"
        log_section_end "REMOTE CHECK" "$logfile" "1"
        exit 1
      fi
    fi
  fi

  log_section_end "REMOTE CHECK" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [3/5] Optional pre-push lint check
  if [[ ${skip_lint} -eq 0 ]] && [[ -n "${CGW_LINT_CMD}${CGW_FORMAT_CMD}${CGW_MARKDOWNLINT_CMD}" ]]; then
    log_section_start "PRE-PUSH LINT CHECK" "$logfile"
    echo "Running pre-push lint check..." | tee -a "$logfile"
    local lint_args=()
    [[ ${skip_md_lint} -eq 1 ]] && lint_args+=("--skip-md-lint")
    if "${SCRIPT_DIR}/check_lint.sh" "${lint_args[@]}" >>"$logfile" 2>&1; then
      echo "[OK] Lint check passed" | tee -a "$logfile"
      log_section_end "PRE-PUSH LINT CHECK" "$logfile" "0"
    else
      echo "[!] Lint check failed" | tee -a "$logfile"
      log_section_end "PRE-PUSH LINT CHECK" "$logfile" "1"
      echo "  Run ./scripts/git/fix_lint.sh to fix issues, or use --skip-lint to bypass" | tee -a "$logfile"
      if [[ ${non_interactive} -eq 0 ]]; then
        read -r -p "  Push anyway despite lint errors? (yes/no): " lint_choice
        if [[ "${lint_choice}" != "yes" ]]; then
          exit 1
        fi
      else
        echo "  [Non-interactive] Aborting due to lint errors" | tee -a "$logfile"
        exit 1
      fi
    fi
    echo "" | tee -a "$logfile"
  fi

  # [4/5] Show what will be pushed
  echo "[4/5] Commits to be pushed:" | tee -a "$logfile"
  local ahead
  ahead=$(git rev-list --count "${CGW_REMOTE}/${target_branch}..HEAD" 2>/dev/null || echo "unknown")
  echo "  Local ahead of ${CGW_REMOTE}/${target_branch}: ${ahead} commit(s)" | tee -a "$logfile"
  if [[ "${ahead}" != "0" ]] && [[ "${ahead}" != "unknown" ]]; then
    git log "${CGW_REMOTE}/${target_branch}..HEAD" --oneline 2>/dev/null | tee -a "$logfile" || true
  fi
  echo "" | tee -a "$logfile"

  if [[ ${dry_run} -eq 1 ]]; then
    echo "=== DRY RUN -- no push performed ===" | tee -a "$logfile"
    echo "Would push: ${target_branch} -> ${CGW_REMOTE}/${target_branch}" | tee -a "$logfile"
    if [[ ${force_push} -eq 1 ]]; then
      echo "Would use: --force-with-lease" | tee -a "$logfile"
    fi
    exit 0
  fi

  # [5/5] Execute push
  log_section_start "GIT PUSH" "$logfile"

  local push_flags=()
  push_flags+=("${CGW_REMOTE}" "${target_branch}")
  if [[ ${force_push} -eq 1 ]]; then
    push_flags+=("--force-with-lease")
    echo "Using --force-with-lease (safer than --force)" | tee -a "$logfile"
  fi

  if run_git_with_logging "GIT PUSH" "$logfile" push "${push_flags[@]}"; then
    log_section_end "GIT PUSH" "$logfile" "0"
    echo "" | tee -a "$logfile"
    {
      echo "========================================"
      echo "[PUSH SUMMARY]"
      echo "========================================"
    } | tee -a "$logfile"
    echo "[OK] PUSH SUCCESSFUL" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "  Branch: ${target_branch} -> ${CGW_REMOTE}/${target_branch}" | tee -a "$logfile"
    echo "  Commits pushed: ${ahead}" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    {
      echo ""
      echo "End Time: $(date)"
    } | tee -a "$logfile"
    echo "Full log: $logfile"
  else
    log_section_end "GIT PUSH" "$logfile" "1"
    echo "" | tee -a "$logfile"
    echo "[FAIL] Push failed" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "Common causes:" | tee -a "$logfile"
    echo "  - Remote has new commits: ./scripts/git/sync_branches.sh" | tee -a "$logfile"
    echo "  - Auth error: check SSH key or token" | tee -a "$logfile"
    echo "  - Branch protection: push may require a PR" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
    echo "Full log: $logfile"
    exit 1
  fi
}

main "$@"
