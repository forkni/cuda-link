#!/usr/bin/env bash
# stash_work.sh - Safe stash wrapper with logging and untracked file support
# Purpose: Wrapper around git stash that always includes untracked files (-u),
#          supports named stashes, and logs operations for traceability.
# Usage: ./scripts/git/stash_work.sh <command> [OPTIONS]
#
# Commands:
#   push [message]   Stash current work (default command if omitted)
#   pop              Apply most recent stash and remove it
#   apply [ref]      Apply stash without removing it
#   list             Show all stashes with metadata
#   drop [ref]       Remove a specific stash (interactive if omitted)
#   show [ref]       Show contents of a stash
#   clear            Remove ALL stashes (requires confirmation)
#
# Globals:
#   SCRIPT_DIR   - Directory containing this script
#   PROJECT_ROOT - Auto-detected git repo root (set by _config.sh)
# Returns:
#   0 on success, 1 on failure

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/git/_common.sh
source "${SCRIPT_DIR}/_common.sh"

usage() {
  echo "Usage: ./scripts/git/stash_work.sh <command> [OPTIONS]"
  echo ""
  echo "Safe stash wrapper -- always includes untracked files (-u)."
  echo ""
  echo "Commands:"
  echo "  push [message]   Stash current work with optional description"
  echo "  pop              Apply and remove most recent stash"
  echo "  apply [ref]      Apply stash without removing it (default: stash@{0})"
  echo "  list             Show all stashes with branch and date"
  echo "  drop [ref]       Remove a specific stash"
  echo "  show [ref]       Show diff for a stash (default: stash@{0})"
  echo "  clear            Remove ALL stashes (requires confirmation)"
  echo ""
  echo "Options (for push):"
  echo "  --include-index  Also stash staged changes (git stash push -S)"
  echo "  --no-untracked   Omit untracked files (not recommended)"
  echo ""
  echo "Examples:"
  echo "  ./scripts/git/stash_work.sh push 'wip: half-done refactor'"
  echo "  ./scripts/git/stash_work.sh pop"
  echo "  ./scripts/git/stash_work.sh list"
  echo "  ./scripts/git/stash_work.sh apply stash@{2}"
}

main() {
  if [[ $# -eq 0 ]]; then
    usage
    exit 0
  fi

  local command="$1"
  shift

  cd "${PROJECT_ROOT}" || {
    err "Cannot find project root"
    exit 1
  }

  case "${command}" in
    push | save)
      local message=""
      local include_index=0
      local untracked=1

      while [[ $# -gt 0 ]]; do
        case "$1" in
          --include-index) include_index=1 ;;
          --no-untracked) untracked=0 ;;
          --help | -h)
            usage
            exit 0
            ;;
          -*)
            echo "[ERROR] Unknown flag: $1" >&2
            exit 1
            ;;
          *) message="$1" ;;
        esac
        shift
      done

      # Check for changes to stash
      if git diff --quiet && git diff --cached --quiet && [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
        echo "[OK] Nothing to stash -- working tree is clean"
        exit 0
      fi

      local stash_args=()
      [[ ${untracked} -eq 1 ]] && stash_args+=("-u")
      [[ ${include_index} -eq 1 ]] && stash_args+=("-S")
      if [[ -n "${message}" ]]; then
        stash_args+=("--message" "${message}")
      fi

      echo "=== Stash Work ==="
      echo ""
      echo "Changes being stashed:"
      git status --short
      echo ""

      if git stash push "${stash_args[@]}"; then
        echo ""
        echo "[OK] Stash created: $(git stash list | head -1)"
        echo ""
        echo "Working tree is now clean."
        echo "Restore with: ./scripts/git/stash_work.sh pop"
      else
        echo "[ERROR] Stash failed" >&2
        exit 1
      fi
      ;;

    pop)
      local ref="${1:-}"
      echo "=== Pop Stash ==="
      echo ""

      if [[ -z "$(git stash list 2>/dev/null)" ]]; then
        echo "[OK] No stashes to pop"
        exit 0
      fi

      local target="${ref:-stash@{0}}"
      echo "Applying: $(git stash list | grep "^${target}" || echo "${target}")"
      echo ""

      if git stash pop "${target}"; then
        echo ""
        echo "[OK] Stash applied and removed"
      else
        echo "[ERROR] Stash pop failed -- conflicts may need manual resolution" >&2
        echo "  Resolve conflicts, then: git stash drop ${target}"
        exit 1
      fi
      ;;

    apply)
      local ref="${1:-stash@{0}}"
      echo "=== Apply Stash (keeping stash) ==="
      echo ""

      if [[ -z "$(git stash list 2>/dev/null)" ]]; then
        echo "[OK] No stashes to apply"
        exit 0
      fi

      echo "Applying: $(git stash list | grep "^${ref}" || echo "${ref}")"
      echo ""

      if git stash apply "${ref}"; then
        echo ""
        echo "[OK] Stash applied (stash retained -- use 'drop' to remove)"
      else
        echo "[ERROR] Stash apply failed" >&2
        exit 1
      fi
      ;;

    list)
      echo "=== Stash List ==="
      echo ""
      if [[ -z "$(git stash list 2>/dev/null)" ]]; then
        echo "  (no stashes)"
      else
        git stash list --format="%C(yellow)%gd%C(reset) %C(green)%cr%C(reset) on %C(cyan)%gs%C(reset)"
      fi
      ;;

    drop)
      local ref="${1:-}"
      echo "=== Drop Stash ==="
      echo ""

      if [[ -z "$(git stash list 2>/dev/null)" ]]; then
        echo "[OK] No stashes to drop"
        exit 0
      fi

      if [[ -z "${ref}" ]]; then
        echo "Stashes:"
        git stash list
        echo ""
        read -e -r -p "Which stash to drop? (e.g. stash@{0}): " ref
        [[ -z "${ref}" ]] && echo "Cancelled" && exit 0
      fi

      echo "Dropping: $(git stash list | grep "^${ref}" || echo "${ref}")"
      read -r -p "Confirm drop? (yes/no): " answer
      case "$(echo "${answer}" | tr '[:upper:]' '[:lower:]')" in
        y | yes)
          git stash drop "${ref}" && echo "[OK] Stash dropped"
          ;;
        *)
          echo "Cancelled"
          ;;
      esac
      ;;

    show)
      local ref="${1:-stash@{0}}"
      echo "=== Stash Contents: ${ref} ==="
      echo ""
      git stash show -p "${ref}"
      ;;

    clear)
      echo "=== Clear All Stashes ==="
      echo ""

      if [[ -z "$(git stash list 2>/dev/null)" ]]; then
        echo "[OK] No stashes to clear"
        exit 0
      fi

      echo "All stashes:"
      git stash list
      echo ""
      echo "[!] WARNING: This permanently removes ALL stashes listed above."
      read -r -p "Type 'CLEAR' to confirm: " confirm

      if [[ "${confirm}" == "CLEAR" ]]; then
        git stash clear && echo "[OK] All stashes cleared"
      else
        echo "Cancelled"
      fi
      ;;

    --help | -h | help)
      usage
      exit 0
      ;;

    *)
      echo "[ERROR] Unknown command: ${command}" >&2
      echo ""
      usage
      exit 1
      ;;
  esac
}

main "$@"
