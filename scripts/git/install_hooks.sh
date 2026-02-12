#!/usr/bin/env bash
# install_hooks.sh - Install Git hooks from .githooks/ to .git/hooks/
# Purpose: automation of git hook installation
# Usage: ./scripts/git/install_hooks.sh

set -u

# Get directory of this script to ensure we can source relative files if needed,
# though here we mainly need project root.
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source common functions
source "${SCRIPT_DIR}/_common.sh"

init_logging "install_hooks"

main() {
  # Write log header
  {
    echo "========================================="
    echo "Install Hooks Log"
    echo "========================================="
    echo "Start Time: $(date)"
    echo "Working Directory: ${PROJECT_ROOT}"
  } > "$logfile"

  echo "=== Git Hooks Installer ===" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  # Ensure execution from project root
  cd "${PROJECT_ROOT}" || {
    echo "[ERROR] Cannot find project root" | tee -a "$logfile"
    exit 1
  }

  # Check if .git directory exists
  if [[ ! -d ".git" ]]; then
    echo "[ERROR] .git directory not found. This script must be run from the repository root." | tee -a "$logfile"
    exit 1
  fi

  # Check if .githooks directory exists
  if [[ ! -d ".githooks" ]]; then
    echo "[ERROR] .githooks directory not found. This repository doesn't have hook templates." | tee -a "$logfile"
    exit 1
  fi

  log_section_start "INSTALL HOOKS" "$logfile"

  # Install pre-commit hook
  if [[ -f ".githooks/pre-commit" ]]; then
    echo "Installing pre-commit hook..." | tee -a "$logfile"

    if cp ".githooks/pre-commit" ".git/hooks/pre-commit" >> "$logfile" 2>&1; then
      chmod +x ".git/hooks/pre-commit" >> "$logfile" 2>&1
      echo "✓ pre-commit hook installed" | tee -a "$logfile"
      log_section_end "INSTALL HOOKS" "$logfile" "0"
    else
      echo "✗ Failed to install pre-commit hook" | tee -a "$logfile"
      log_section_end "INSTALL HOOKS" "$logfile" "1"
      exit 1
    fi
  else
    echo "⚠ pre-commit template not found, skipping" | tee -a "$logfile"
    log_section_end "INSTALL HOOKS" "$logfile" "1"
  fi

  echo "" | tee -a "$logfile"
  {
    echo "========================================"
    echo "[INSTALL SUMMARY]"
    echo "========================================"
  } | tee -a "$logfile"
  echo "✓ HOOKS INSTALLED SUCCESSFULLY" | tee -a "$logfile"
  echo "" | tee -a "$logfile"

  echo "The following hooks are now active:"
  echo "  - pre-commit: File validation + code quality checks"
  echo ""

  echo "What this hook does:"
  echo "  1. Prevents committing local-only files"
  echo "  2. Validates documentation files"
  echo "  3. Checks code quality (Python files)"
  echo "  4. Offers auto-fix for lint errors"
  echo ""

  echo "To bypass hooks temporarily (not recommended):"
  echo "  git commit --no-verify"
  echo ""

  echo "To uninstall hooks:"
  echo "  rm .git/hooks/pre-commit"
  echo ""

  {
    echo ""
    echo "End Time: $(date)"
  } | tee -a "$logfile"

  echo "" | tee -a "$logfile"
  echo "Full log: $logfile"
}

main "$@"
