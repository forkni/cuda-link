#!/usr/bin/env bash
# install_hooks.sh - Install git hooks from .githooks/ to .git/hooks/
# Purpose: Install pre-commit hook that blocks local-only files
# Usage: ./scripts/git/install_hooks.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR     - Directory containing this script
#   PROJECT_ROOT   - Auto-detected git repo root (set by _config.sh)
#   logfile        - Set by init_logging
# Returns:
#   0 on success, 1 on failure

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

init_logging "install_hooks"

main() {
  if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    echo "Usage: ./scripts/git/install_hooks.sh"
    echo ""
    echo "Install git hooks from .githooks/ to .git/hooks/."
    echo "Installs: pre-commit (blocks local-only files, optional lint check)"
    echo ""
    echo "The hook file must exist at: \$PROJECT_ROOT/.githooks/pre-commit"
    echo "Run configure.sh first to generate this file from the template."
    echo ""
    echo "Options:"
    echo "  -h, --help   Show this help"
    echo ""
    echo "To uninstall: rm .git/hooks/pre-commit"
    echo "To bypass temporarily (not recommended): git commit --no-verify"
    exit 0
  fi

  {
    echo "========================================="
    echo "Install Hooks Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } >"$logfile"

  echo "=== Git Hooks Installer ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  cd "${PROJECT_ROOT}" || {
    err "Cannot find project root"
    exit 1
  }

  if [[ ! -d ".git" ]]; then
    err ".git directory not found. Run from repository root."
    exit 1
  fi

  if [[ ! -d ".githooks" ]]; then
    err ".githooks directory not found."
    echo "Run configure.sh first to generate hook files, or create .githooks/ manually." >&2
    exit 1
  fi

  log_section_start "INSTALL HOOKS" "$logfile"

  local hooks_ok=0

  if [[ -f ".githooks/pre-commit" ]]; then
    echo "Installing pre-commit hook..." | tee -a "$logfile"
    if cp ".githooks/pre-commit" ".git/hooks/pre-commit" >>"$logfile" 2>&1; then
      chmod +x ".git/hooks/pre-commit" >>"$logfile" 2>&1
      echo "  [OK] pre-commit installed" | tee -a "$logfile"
    else
      echo "  [FAIL] Failed to install pre-commit hook" | tee -a "$logfile"
      hooks_ok=1
    fi
  else
    err "pre-commit template not found at .githooks/pre-commit"
    echo "Run configure.sh to generate the hook from your CGW_LOCAL_FILES config." >&2
    log_section_end "INSTALL HOOKS" "$logfile" "1"
    exit 1
  fi

  if [[ -f ".githooks/pre-push" ]]; then
    echo "Installing pre-push hook..." | tee -a "$logfile"
    if cp ".githooks/pre-push" ".git/hooks/pre-push" >>"$logfile" 2>&1; then
      chmod +x ".git/hooks/pre-push" >>"$logfile" 2>&1
      echo "  [OK] pre-push installed" | tee -a "$logfile"
    else
      echo "  [!] Failed to install pre-push hook (non-fatal)" | tee -a "$logfile"
    fi
  fi

  log_section_end "INSTALL HOOKS" "$logfile" "${hooks_ok}"
  [[ ${hooks_ok} -ne 0 ]] && exit 1

  echo "" | tee -a "$logfile"
  {
    echo "========================================"
    echo "[INSTALL SUMMARY]"
    echo "========================================"
  } | tee -a "$logfile"
  echo "HOOKS INSTALLED SUCCESSFULLY" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  echo "Active hooks:"
  echo "  - pre-commit: Blocks local-only files, optional lint check"
  echo "  - pre-push:   Validates conventional commit format on unpushed commits"
  echo ""
  echo "To bypass temporarily (not recommended):"
  echo "  git commit --no-verify / git push --no-verify"
  echo ""
  echo "To uninstall:"
  echo "  rm .git/hooks/pre-commit .git/hooks/pre-push"
  echo ""

  {
    echo ""
    echo "End Time: $(date)"
  } | tee -a "$logfile"

  echo "Full log: $logfile"
}

main "$@"
