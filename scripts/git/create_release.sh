#!/usr/bin/env bash
# create_release.sh - Create annotated version tags to trigger GitHub Releases
# Purpose: Create a properly annotated semver tag on the target branch.
#          Annotated tags store author, date, and message -- required for the
#          release.yml GitHub Actions workflow and distinguishable from backup tags.
# Usage: ./scripts/git/create_release.sh [OPTIONS] <version>
#
# Globals:
#   SCRIPT_DIR          - Directory containing this script
#   PROJECT_ROOT        - Auto-detected git repo root (set by _config.sh)
#   CGW_TARGET_BRANCH   - Branch that receives releases (default: main)
# Arguments:
#   <version>           Version string: v1.2.3 or 1.2.3 (v prefix auto-added)
#   --message <msg>     Tag annotation message (default: "Release <version>")
#   --non-interactive   Skip prompts
#   --dry-run           Show what would happen without tagging
#   --push              Push the tag to origin after creation
#   -h, --help          Show help
# Returns:
#   0 on success, 1 on failure

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/git/_common.sh
source "${SCRIPT_DIR}/_common.sh"

# validate_semver - Check that version matches vX.Y.Z or vX.Y.Z-suffix format.
validate_semver() {
  local ver="$1"
  if [[ ! "${ver}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([._-][a-zA-Z0-9._-]+)?$ ]]; then
    echo "[ERROR] Version '${ver}' does not match semver format (e.g. v1.2.3, v1.2.3-rc1)" >&2
    return 1
  fi
}

main() {
  local version=""
  local tag_message=""
  local non_interactive=0
  local dry_run=0
  local push_tag=0

  [[ "${CGW_NON_INTERACTIVE:-0}" == "1" ]] && non_interactive=1

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help | -h)
        echo "Usage: ./scripts/git/create_release.sh [OPTIONS] <version>"
        echo ""
        echo "Create an annotated version tag to trigger the GitHub Release workflow."
        echo "Must be run from the target branch (default: ${CGW_TARGET_BRANCH})."
        echo ""
        echo "Arguments:"
        echo "  <version>           Semver version: v1.2.3 or 1.2.3 (v prefix auto-added)"
        echo ""
        echo "Options:"
        echo "  --message <msg>     Annotation message (default: 'Release <version>')"
        echo "  --push              Push tag to origin after creation"
        echo "  --non-interactive   Skip confirmation prompt"
        echo "  --dry-run           Preview without creating tag"
        echo "  -h, --help          Show this help"
        echo ""
        echo "The annotated tag triggers release.yml (GitHub Actions) which creates"
        echo "a GitHub Release with auto-generated notes and source archives."
        echo ""
        echo "Environment:"
        echo "  CGW_REMOTE   Remote to push tag to (default: origin)"
        echo ""
        echo "Examples:"
        echo "  ./scripts/git/create_release.sh v1.0.0"
        echo "  ./scripts/git/create_release.sh v1.0.0 --message 'First stable release'"
        echo "  ./scripts/git/create_release.sh v1.0.0 --push"
        exit 0
        ;;
      --message)
        tag_message="${2:-}"
        shift
        ;;
      --push) push_tag=1 ;;
      --non-interactive) non_interactive=1 ;;
      --dry-run) dry_run=1 ;;
      -*)
        echo "[ERROR] Unknown flag: $1" >&2
        exit 1
        ;;
      *)
        version="$1"
        ;;
    esac
    shift
  done

  cd "${PROJECT_ROOT}" || {
    err "Cannot find project root"
    exit 1
  }

  # Require version argument
  if [[ -z "${version}" ]]; then
    echo "[ERROR] Version argument required (e.g. v1.2.3)" >&2
    echo "Usage: ./scripts/git/create_release.sh v1.2.3" >&2
    exit 1
  fi

  # Normalize: add v prefix if missing
  if [[ "${version}" != v* ]]; then
    version="v${version}"
  fi

  # Validate semver
  if ! validate_semver "${version}"; then
    exit 1
  fi

  # Check current branch
  local current_branch
  current_branch=$(git branch --show-current 2>/dev/null || true)
  if [[ -z "${current_branch}" ]]; then
    echo "[ERROR] Detached HEAD state -- checkout ${CGW_TARGET_BRANCH} first" >&2
    exit 1
  fi

  if [[ "${current_branch}" != "${CGW_TARGET_BRANCH}" ]]; then
    echo "[ERROR] Must be on target branch (${CGW_TARGET_BRANCH}), currently on: ${current_branch}" >&2
    echo "  git checkout ${CGW_TARGET_BRANCH}" >&2
    exit 1
  fi

  # Check for uncommitted changes
  if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo "[ERROR] Uncommitted changes present -- commit before releasing" >&2
    git status --short >&2
    exit 1
  fi

  # Check tag does not already exist
  if git tag -l "${version}" | grep -q "^${version}$"; then
    echo "[ERROR] Tag '${version}' already exists" >&2
    echo "  List tags: git tag -l 'v*'" >&2
    exit 1
  fi

  # Default annotation message
  if [[ -z "${tag_message}" ]]; then
    tag_message="Release ${version}"
  fi

  # Show preview
  echo "=== Create Release Tag ==="
  echo ""
  echo "  Version:  ${version}"
  echo "  Branch:   ${current_branch}"
  echo "  Commit:   $(git log -1 --format='%h %s')"
  echo "  Message:  ${tag_message}"
  local push_label="no (manual push required)"
  [[ ${push_tag} -eq 1 ]] && push_label="yes (after creation)"
  echo "  Push:     ${push_label}"
  echo ""

  if [[ ${dry_run} -eq 1 ]]; then
    echo "--- Dry run: no tag created ---"
    echo "Command would be: git tag -a '${version}' -m '${tag_message}'"
    [[ ${push_tag} -eq 1 ]] && echo "Followed by:       git push ${CGW_REMOTE} '${version}'"
    exit 0
  fi

  # Confirm
  if [[ ${non_interactive} -eq 0 ]]; then
    read -r -p "Create annotated tag '${version}'? (yes/no): " answer
    case "$(echo "${answer}" | tr '[:upper:]' '[:lower:]')" in
      y | yes) ;;
      *)
        echo "Cancelled"
        exit 0
        ;;
    esac
  fi

  # Create annotated tag
  if git tag -a "${version}" -m "${tag_message}"; then
    echo "[OK] Created annotated tag: ${version}"
  else
    echo "[ERROR] Failed to create tag" >&2
    exit 1
  fi

  # Push tag if requested
  if [[ ${push_tag} -eq 1 ]]; then
    echo "Pushing tag to ${CGW_REMOTE}..."
    if git push "${CGW_REMOTE}" "${version}"; then
      echo "[OK] Tag pushed: ${version}"
      echo ""
      echo "GitHub Release workflow triggered."
      echo "Check: https://github.com/$(git remote get-url "${CGW_REMOTE}" | sed 's|.*github.com[:/]||;s|\.git$||')/actions"
    else
      echo "[ERROR] Failed to push tag. Push manually:" >&2
      echo "  git push ${CGW_REMOTE} ${version}" >&2
      exit 1
    fi
  else
    echo ""
    echo "Next step -- push to trigger GitHub Release:"
    echo "  git push ${CGW_REMOTE} ${version}"
  fi
}

main "$@"
