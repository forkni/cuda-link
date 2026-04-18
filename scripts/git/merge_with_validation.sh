#!/usr/bin/env bash
# merge_with_validation.sh - Safe merge from source to target branch
# Purpose: Merge with validation, conflict auto-resolution, and CI policy check
# Usage: ./scripts/git/merge_with_validation.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR            - Directory containing this script
#   PROJECT_ROOT          - Auto-detected git repo root (set by _config.sh)
#   logfile               - Set by init_logging
#   CGW_SOURCE_BRANCH     - Source branch (default: development)
#   CGW_TARGET_BRANCH     - Target branch (default: main)
#   CGW_DOCS_PATTERN      - Regex for allowed doc filenames (empty = skip validation)
#   CGW_CLEANUP_TESTS     - Remove tests/ from target if gitignored (0=disabled)
# Arguments:
#   --non-interactive   Skip all prompts (aborts on unexpected state)
#   --dry-run           Show what would happen without making changes
#   --source <branch>   Override source branch for this invocation
#   --target <branch>   Override target branch for this invocation
#   -h, --help          Show help
# Returns:
#   0 on successful merge, 1 on any failure

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

init_logging "merge_with_validation"

# ============================================================================
# CLEANUP TRAP
# ============================================================================

_merge_original_branch=""
_merge_did_checkout_target=0

_cleanup_merge() {
  local current
  current=$(git branch --show-current 2>/dev/null || true)
  if [[ ${_merge_did_checkout_target} -eq 1 ]] && [[ -n "${_merge_original_branch}" ]] &&
    [[ "${current}" != "${_merge_original_branch}" ]]; then
    echo "" >&2
    echo "[!] Interrupted -- you are on branch: ${current}" >&2
    echo "  Returning to: ${_merge_original_branch}" >&2
    git checkout "${_merge_original_branch}" 2>/dev/null || true
  fi
}
trap _cleanup_merge EXIT INT TERM

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# validate_docs_ci_policy - Check docs/ files against CGW_DOCS_PATTERN.
# Skipped entirely when CGW_DOCS_PATTERN is empty.
# Arguments:
#   $1 - "committed" or "staged"
#   $2 - original_branch (for rollback on failure)
validate_docs_ci_policy() {
  local check_mode="$1"
  local original_branch="$2"

  # Skip if no pattern configured
  if [[ -z "${CGW_DOCS_PATTERN}" ]]; then
    echo "  (docs CI validation skipped -- CGW_DOCS_PATTERN not set in .cgw.conf)" | tee -a "$logfile"
    return 0
  fi

  echo ""
  echo "[6/7] Validating documentation files against CI policy..."

  local docs_validation_failed=0
  local doc_files

  if [[ "${check_mode}" == "committed" ]]; then
    doc_files=$(git diff --name-only HEAD~1 HEAD | grep "^docs/" || true)
    for doc_file in ${doc_files}; do
      local doc_name
      doc_name=$(basename "${doc_file}")
      if ! [[ "${doc_name}" =~ ${CGW_DOCS_PATTERN} ]]; then
        if git diff --diff-filter=A --name-only HEAD~1 HEAD -- "${doc_file}" | grep -q .; then
          echo "[FAIL] ERROR: Unauthorized doc file: ${doc_file}" | tee -a "$logfile"
          echo "   Not in CGW_DOCS_PATTERN allowlist" | tee -a "$logfile"
          docs_validation_failed=1
        fi
      fi
    done
  else
    doc_files=$(git diff --cached --name-only --diff-filter=A | grep "^docs/" || true)
    for doc_file in ${doc_files}; do
      local doc_name
      doc_name=$(basename "${doc_file}")
      if ! [[ "${doc_name}" =~ ${CGW_DOCS_PATTERN} ]]; then
        echo "[FAIL] ERROR: Unauthorized doc file: ${doc_file}" | tee -a "$logfile"
        echo "   Not in CGW_DOCS_PATTERN allowlist" | tee -a "$logfile"
        docs_validation_failed=1
      fi
    done
  fi

  if [[ ${docs_validation_failed} -eq 1 ]]; then
    echo "" | tee -a "$logfile"
    echo "[FAIL] CI POLICY VIOLATION: Unauthorized documentation detected" | tee -a "$logfile"
    if [[ "${check_mode}" == "committed" ]]; then
      echo "Rolling back merge..." | tee -a "$logfile"
      git reset --hard HEAD~1 >>"$logfile" 2>&1
    else
      echo "Aborting merge..." | tee -a "$logfile"
      git merge --abort >>"$logfile" 2>&1
    fi
    git checkout "${original_branch}" >>"$logfile" 2>&1
    exit 1
  fi
  echo "[OK] Documentation validation passed" | tee -a "$logfile"
}

# cleanup_tests_dir - Remove tests/ from target branch if gitignored.
# Controlled by CGW_CLEANUP_TESTS (default: 0 = disabled).
# Arguments:
#   $1 - "amend" or "stage"
cleanup_tests_dir() {
  local commit_mode="$1"
  local _tgt="${2:-${CGW_TARGET_BRANCH}}"

  # Skip unless explicitly enabled
  if [[ "${CGW_CLEANUP_TESTS}" != "1" ]]; then
    return 0
  fi

  echo ""
  echo "[6.5/7] Checking tests/ directory policy..."

  if grep -q "^tests/\$" .gitignore 2>/dev/null; then
    if [[ -d "tests" ]]; then
      echo "[!] Removing tests/ directory from ${_tgt} branch (per .gitignore policy)" | tee -a "$logfile"
      if git rm -r tests >>"$logfile" 2>&1; then
        echo "[OK] Removed tests/ directory" | tee -a "$logfile"
        if [[ "${commit_mode}" == "amend" ]]; then
          git commit --amend --no-edit >>"$logfile" 2>&1
        else
          git add -u >>"$logfile" 2>&1
        fi
      else
        echo "[FAIL] ERROR: Failed to remove tests/ directory" | tee -a "$logfile"
      fi
    else
      echo "[OK] No tests/ directory found" | tee -a "$logfile"
    fi
  else
    echo "[OK] tests/ is tracked in git -- keeping on ${_tgt}" | tee -a "$logfile"
  fi
}

main() {
  local non_interactive=0
  local dry_run=0
  local src_branch="${CGW_SOURCE_BRANCH}"
  local tgt_branch="${CGW_TARGET_BRANCH}"

  while [[ $# -gt 0 ]]; do
    case "${1}" in
      --help | -h)
        echo "Usage: ./scripts/git/merge_with_validation.sh [OPTIONS]"
        echo ""
        echo "Safely merge source branch into target with conflict resolution."
        echo ""
        echo "Options:"
        echo "  --non-interactive      Skip all prompts (aborts on unexpected state)"
        echo "  --dry-run              Show commits/files that would be merged"
        echo "  --source <branch>      Override source branch for this invocation"
        echo "  --target <branch>      Override target branch for this invocation"
        echo "  -h, --help             Show this help"
        echo ""
        echo "Configuration:"
        echo "  CGW_SOURCE_BRANCH    Source branch (default: development)"
        echo "  CGW_TARGET_BRANCH    Target branch (default: main)"
        echo "  CGW_DOCS_PATTERN     Regex for allowed doc filenames (default: empty = skip)"
        echo "  CGW_CLEANUP_TESTS    Remove tests/ from target if gitignored (default: 0)"
        echo ""
        echo "Examples:"
        echo "  # Default: merge development -> main"
        echo "  ./scripts/git/merge_with_validation.sh"
        echo ""
        echo "  # Override: merge feature/x -> release/y"
        echo "  ./scripts/git/merge_with_validation.sh --source feature/x --target release/y"
        echo ""
        echo "Environment:"
        echo "  CGW_NON_INTERACTIVE=1        Same as --non-interactive"
        echo "  CGW_DOCS_PATTERN=<regex>     Override docs allowlist pattern"
        exit 0
        ;;
      --non-interactive) non_interactive=1 ;;
      --dry-run) dry_run=1 ;;
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

  local branch_label="${src_branch} -> ${tgt_branch}"
  if [[ "${src_branch}" != "${CGW_SOURCE_BRANCH}" ]] || [[ "${tgt_branch}" != "${CGW_TARGET_BRANCH}" ]]; then
    branch_label="${src_branch} -> ${tgt_branch} (overridden)"
  fi

  {
    echo "========================================="
    echo "Merge With Validation Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } >"$logfile"

  echo "=== Safe Merge: ${branch_label} ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"
  echo "Workflow Log: ${logfile}" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  cd "${PROJECT_ROOT}" || {
    err "Cannot find project root"
    exit 1
  }

  if [[ ${dry_run} -eq 1 ]]; then
    echo "=== DRY RUN MODE -- no changes will be made ===" | tee -a "$logfile"
    echo "Would merge: ${src_branch} -> ${tgt_branch}" | tee -a "$logfile"
    echo "Commits to merge:" | tee -a "$logfile"
    git log "${tgt_branch}..${src_branch}" --oneline | tee -a "$logfile"
    echo ""
    echo "Files that would change:" | tee -a "$logfile"
    git diff --name-status "${tgt_branch}..${src_branch}" | tee -a "$logfile"
    exit 0
  fi

  # [1/7] Run validation
  log_section_start "PRE-MERGE VALIDATION" "$logfile"

  if [[ -f "${SCRIPT_DIR}/validate_branches.sh" ]]; then
    if ! CGW_SOURCE_BRANCH="${src_branch}" CGW_TARGET_BRANCH="${tgt_branch}" \
      bash "${SCRIPT_DIR}/validate_branches.sh" >>"$logfile" 2>&1; then
      echo "[FAIL] Validation failed - aborting merge" | tee -a "$logfile"
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

  local original_branch
  original_branch=$(git branch --show-current)
  _merge_original_branch="${original_branch}"
  echo "Current branch: ${original_branch}" | tee -a "$logfile"

  if [[ "${original_branch}" == "${tgt_branch}" ]]; then
    echo "[FAIL] ERROR: Already on ${tgt_branch} branch" | tee -a "$logfile"
    echo "  Run this script from ${src_branch} branch" | tee -a "$logfile"
    exit 1
  elif [[ "${original_branch}" != "${src_branch}" ]]; then
    echo "[!] WARNING: Not on ${src_branch} branch" | tee -a "$logfile"
    echo "  Current: ${original_branch}" | tee -a "$logfile"

    if [[ ${non_interactive} -eq 0 ]]; then
      read -p "  Continue anyway? [y/N] " -n 1 -r
      echo
      if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "  Aborted" | tee -a "$logfile"
        exit 1
      fi
    else
      echo "  Non-interactive: aborting for safety" | tee -a "$logfile"
      exit 1
    fi
  fi

  if ! run_git_with_logging "GIT CHECKOUT" "$logfile" checkout "${tgt_branch}"; then
    echo "[FAIL] Failed to checkout ${tgt_branch} branch" | tee -a "$logfile"
    exit 1
  fi
  _merge_did_checkout_target=1

  log_section_end "GIT CHECKOUT TARGET" "$logfile" "0"
  echo "" | tee -a "$logfile"

  # [3/7] Create pre-merge backup tag
  log_section_start "CREATE BACKUP TAG" "$logfile"

  if [[ -z "${timestamp:-}" ]]; then get_timestamp; fi
  local backup_tag="pre-merge-backup-${timestamp}-$$"

  if git tag "${backup_tag}" >>"$logfile" 2>&1; then
    echo "[OK] Created backup tag: ${backup_tag}" | tee -a "$logfile"
    log_section_end "CREATE BACKUP TAG" "$logfile" "0"
  else
    echo "[!] Warning: Could not create backup tag" | tee -a "$logfile"
    log_section_end "CREATE BACKUP TAG" "$logfile" "1"
  fi
  echo "" | tee -a "$logfile"

  # [4/7] Perform merge
  log_section_start "GIT MERGE" "$logfile"

  # Build optional merge strategy flags from config
  local merge_extra_args=()
  if [[ -n "${CGW_MERGE_CONFLICT_STYLE:-}" ]]; then
    merge_extra_args+=("--conflict=${CGW_MERGE_CONFLICT_STYLE}")
  fi
  if [[ "${CGW_MERGE_IGNORE_WHITESPACE:-0}" == "1" ]]; then
    merge_extra_args+=("-Xignore-space-change")
  fi

  # shellcheck disable=SC2068  # Intentional: empty array expands to zero words (${arr[@]+...} is Bash 3.x portable)
  if run_git_with_logging "GIT MERGE SOURCE" "$logfile" merge "${src_branch}" --no-ff -m "Merge ${src_branch} into ${tgt_branch}" ${merge_extra_args[@]+"${merge_extra_args[@]}"}; then
    echo "[OK] Merge completed without conflicts" | tee -a "$logfile"
    log_section_end "GIT MERGE" "$logfile" "0"

    validate_docs_ci_policy "committed" "${original_branch}"
    cleanup_tests_dir "amend" "${tgt_branch}"

  else
    local merge_exit_code="${GIT_EXIT_CODE:-1}"
    echo "" | tee -a "$logfile"
    echo "[!] Merge conflicts detected - analyzing..." | tee -a "$logfile"
    log_section_end "GIT MERGE" "$logfile" "${merge_exit_code}"

    local conflict_status
    conflict_status=$(git status --short)

    # Auto-resolve DU (modify/delete) conflicts
    if printf '%s\n' "${conflict_status}" | grep -q "^DU "; then
      echo "  Found modify/delete conflicts -- auto-resolving..."
      echo ""

      local resolution_failed=0

      while read -r conflict_file; do
        echo "  Resolving: ${conflict_file}"
        if git rm "${conflict_file}" >/dev/null 2>&1; then
          echo "  [OK] Removed: ${conflict_file}"
        else
          echo "  [FAIL] ERROR: Failed to remove ${conflict_file}"
          resolution_failed=1
        fi
      done < <(printf '%s\n' "${conflict_status}" | grep "^DU " | cut -c 4-)

      if [[ ${resolution_failed} -eq 1 ]]; then
        echo "" | tee -a "$logfile"
        echo "[FAIL] Auto-resolution failed for some files" | tee -a "$logfile"
        printf '%s\n' "${conflict_status}" | tee -a "$logfile"
        exit 1
      fi

      echo "" | tee -a "$logfile"
      echo "[OK] Auto-resolved modify/delete conflicts" | tee -a "$logfile"
    fi

    # AU/AA conflicts require manual resolution
    if printf '%s\n' "${conflict_status}" | grep -qE "^(AU|AA) "; then
      echo "" | tee -a "$logfile"
      echo "[FAIL] Add/add or add/unmerged conflicts require manual resolution:" | tee -a "$logfile"
      printf '%s\n' "${conflict_status}" | grep -E "^(AU|AA) " | tee -a "$logfile"
      echo ""
      echo "Please resolve manually:"
      echo "  1. Edit conflicted files"
      echo "  2. git add <resolved files>"
      echo "  3. git commit"
      echo ""
      echo "Or abort: git merge --abort && git checkout ${original_branch}"
      exit 1
    fi

    # DD (both deleted): auto-resolve by accepting deletion
    if printf '%s\n' "${conflict_status}" | grep -q "^DD "; then
      echo "  Found both-deleted conflicts -- auto-resolving..." | tee -a "$logfile"
      while read -r conflict_file; do
        git rm "${conflict_file}" >/dev/null 2>&1 || true
        echo "  [OK] Removed (both deleted): ${conflict_file}" | tee -a "$logfile"
      done < <(printf '%s\n' "${conflict_status}" | grep "^DD " | cut -c 4-)
    fi

    # UU (both modified): requires manual resolution
    if printf '%s\n' "${conflict_status}" | grep -q "^UU "; then
      echo "" | tee -a "$logfile"
      echo "[FAIL] Content conflicts require manual resolution:" | tee -a "$logfile"
      printf '%s\n' "${conflict_status}" | grep "^UU "
      echo ""
      echo "Please resolve manually:"
      echo "  1. Edit conflicted files"
      echo "  2. git add <resolved files>"
      echo "  3. git commit"
      echo ""
      echo "Or abort: git merge --abort && git checkout ${original_branch}"
      exit 1
    fi

    # UD (deleted by us, modified by theirs): requires manual resolution
    if printf '%s\n' "${conflict_status}" | grep -q "^UD "; then
      echo "" | tee -a "$logfile"
      echo "[FAIL] Deleted-by-us conflicts require manual resolution:" | tee -a "$logfile"
      printf '%s\n' "${conflict_status}" | grep "^UD " | tee -a "$logfile"
      echo ""
      echo "Please resolve manually (for each file):"
      echo "  Keep deletion: git rm <file>"
      echo "  Keep theirs:   git checkout --theirs <file> && git add <file>"
      echo ""
      echo "Or abort: git merge --abort && git checkout ${original_branch}"
      exit 1
    fi

    # AD/DA (added differently on each side): requires manual resolution
    if printf '%s\n' "${conflict_status}" | grep -qE "^(AD|DA) "; then
      echo "" | tee -a "$logfile"
      echo "[FAIL] Add/delete conflicts require manual resolution:" | tee -a "$logfile"
      printf '%s\n' "${conflict_status}" | grep -E "^(AD|DA) " | tee -a "$logfile"
      echo ""
      echo "Please resolve manually (for each file):"
      echo "  Keep ours:   git checkout --ours <file> && git add <file>"
      echo "  Keep theirs: git checkout --theirs <file> && git add <file>"
      echo ""
      echo "Or abort: git merge --abort && git checkout ${original_branch}"
      exit 1
    fi

    validate_docs_ci_policy "staged" "${original_branch}"
    cleanup_tests_dir "stage" "${tgt_branch}"
    echo "" | tee -a "$logfile"

    # [7/7] Complete the merge
    log_section_start "GIT COMMIT" "$logfile"
    if git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
      if run_git_with_logging "GIT COMMIT MERGE" "$logfile" commit --no-edit; then
        echo "[OK] Merge commit completed" | tee -a "$logfile"
      else
        echo "[FAIL] Failed to complete merge commit" | tee -a "$logfile"
        log_section_end "GIT COMMIT" "$logfile" "1"
        echo "To abort: git merge --abort"
        exit 1
      fi
    fi
    log_section_end "GIT COMMIT" "$logfile" "0"
  fi

  # Success summary
  echo "" | tee -a "$logfile"
  {
    echo "========================================"
    echo "[MERGE SUMMARY]"
    echo "========================================"
  } | tee -a "$logfile"

  echo "[OK] MERGE SUCCESSFUL" | tee -a "$logfile"
  echo "" | tee -a "$logfile"
  git log -1 --oneline | while read -r line; do echo "  Latest commit: $line" | tee -a "$logfile"; done
  echo "  Backup tag: ${backup_tag}" | tee -a "$logfile"
  echo "" | tee -a "$logfile"
  echo "Next steps:" | tee -a "$logfile"
  echo "  1. Review: git log --oneline -5" | tee -a "$logfile"
  echo "  2. Test your build" | tee -a "$logfile"
  echo "  3. Push: ./scripts/git/push_validated.sh" | tee -a "$logfile"
  echo "  Rollback: ./scripts/git/rollback_merge.sh" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  {
    echo ""
    echo "End Time: $(date)"
  } | tee -a "$logfile"

  echo "" | tee -a "$logfile"
  echo "Full log: $logfile"
}

main "$@"
