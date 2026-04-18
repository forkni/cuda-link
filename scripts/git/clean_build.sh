#!/usr/bin/env bash
# clean_build.sh - Safe cleanup of build artifacts and temporary files
# Purpose: Remove generated files that should not be committed. Uses --dry-run
#          by default to prevent accidental deletion.
# Usage: ./scripts/git/clean_build.sh [OPTIONS]
#
# Globals:
#   SCRIPT_DIR   - Directory containing this script
#   PROJECT_ROOT - Auto-detected git repo root (set by _config.sh)
# Arguments:
#   --execute    Actually remove files (default is dry-run)
#   --python     Clean Python artifacts (auto-detected if omitted)
#   --td         Clean TouchDesigner artifacts (auto-detected if omitted)
#   --glsl       Clean GLSL compiled shaders (auto-detected if omitted)
#   --all        Clean all known artifact types regardless of detection
#   -h, --help   Show help
# Returns:
#   0 on success, 1 on failure

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

main() {
  local execute=0
  local force_python=0
  local force_td=0
  local force_glsl=0
  local force_all=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help | -h)
        echo "Usage: ./scripts/git/clean_build.sh [OPTIONS]"
        echo ""
        echo "Clean build artifacts. Defaults to --dry-run (safe preview)."
        echo "Pass --execute to actually delete files."
        echo ""
        echo "Options:"
        echo "  --execute    Remove files (without this flag, only previews)"
        echo "  --python     Include Python artifacts (__pycache__, *.pyc, dist/, etc.)"
        echo "  --td         Include TouchDesigner artifacts (*.toe.bak, Backup/)"
        echo "  --glsl       Include compiled GLSL shaders (*.spv)"
        echo "  --all        Include all artifact types regardless of detection"
        echo "  -h, --help   Show this help"
        echo ""
        echo "Auto-detects project type from pyproject.toml, *.toe files, *.glsl files."
        exit 0
        ;;
      --execute) execute=1 ;;
      --python) force_python=1 ;;
      --td) force_td=1 ;;
      --glsl) force_glsl=1 ;;
      --all) force_all=1 ;;
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

  # Auto-detect project types (unless --all)
  local has_python=0 has_td=0 has_glsl=0
  if [[ ${force_all} -eq 0 ]]; then
    [[ -f "pyproject.toml" ]] || [[ -f "setup.py" ]] || [[ -f "requirements.txt" ]] && has_python=1
    [[ -n "$(find . -maxdepth 3 -name "*.py" -print -quit 2>/dev/null)" ]] && has_python=1
    [[ -n "$(find . -maxdepth 3 \( -name "*.toe" -o -name "*.tox" \) -print -quit 2>/dev/null)" ]] && has_td=1
    [[ -n "$(find . -maxdepth 5 \( -name "*.glsl" -o -name "*.spv" \) -print -quit 2>/dev/null)" ]] && has_glsl=1
  fi

  [[ ${force_python} -eq 1 ]] && has_python=1
  [[ ${force_td} -eq 1 ]] && has_td=1
  [[ ${force_glsl} -eq 1 ]] && has_glsl=1
  [[ ${force_all} -eq 1 ]] && has_python=1 && has_td=1 && has_glsl=1

  local mode_label="DRY RUN"
  [[ ${execute} -eq 1 ]] && mode_label="EXECUTE"

  echo "=== Clean Build Artifacts [${mode_label}] ==="
  echo ""
  [[ ${execute} -eq 0 ]] && echo "[!] Dry run -- pass --execute to actually delete files." && echo ""

  local total_cleaned=0

  # -- Common patterns (always) ----------------------------------------------
  echo "--- Common ---"
  local common_patterns=(
    ".DS_Store"
    "Thumbs.db"
    "desktop.ini"
    "*.tmp"
    "*.bak"
    "ehthumbs.db"
  )
  for pattern in "${common_patterns[@]}"; do
    while IFS= read -r f; do
      echo "  ${f}"
      [[ ${execute} -eq 1 ]] && rm -f "${f}"
      ((total_cleaned++)) || true
    done < <(find . -name "${pattern}" -not -path "./.git/*" 2>/dev/null)
  done

  # -- Python ----------------------------------------------------------------
  if [[ ${has_python} -eq 1 ]]; then
    echo ""
    echo "--- Python ---"
    # __pycache__ directories
    while IFS= read -r d; do
      echo "  ${d}/"
      [[ ${execute} -eq 1 ]] && rm -rf "${d}"
      ((total_cleaned++)) || true
    done < <(find . -type d -name "__pycache__" -not -path "./.git/*" 2>/dev/null)

    # .pyc / .pyo files
    while IFS= read -r f; do
      echo "  ${f}"
      [[ ${execute} -eq 1 ]] && rm -f "${f}"
      ((total_cleaned++)) || true
    done < <(find . \( -name "*.pyc" -o -name "*.pyo" \) -not -path "./.git/*" 2>/dev/null)

    # Build directories at project root
    for d in dist build .eggs .pytest_cache .mypy_cache .ruff_cache; do
      if [[ -d "${d}" ]]; then
        echo "  ${d}/"
        [[ ${execute} -eq 1 ]] && rm -rf "${d}"
        ((total_cleaned++)) || true
      fi
    done

    # .egg-info directories
    while IFS= read -r d; do
      echo "  ${d}/"
      [[ ${execute} -eq 1 ]] && rm -rf "${d}"
      ((total_cleaned++)) || true
    done < <(find . -maxdepth 3 -type d -name "*.egg-info" -not -path "./.git/*" 2>/dev/null)
  fi

  # -- TouchDesigner ---------------------------------------------------------
  if [[ ${has_td} -eq 1 ]]; then
    echo ""
    echo "--- TouchDesigner ---"
    # Auto-backup .toe.bak files
    while IFS= read -r f; do
      echo "  ${f}"
      [[ ${execute} -eq 1 ]] && rm -f "${f}"
      ((total_cleaned++)) || true
    done < <(find . -name "*.toe.bak" -not -path "./.git/*" 2>/dev/null)

    # Backup/ directories created by TD
    while IFS= read -r d; do
      echo "  ${d}/"
      [[ ${execute} -eq 1 ]] && rm -rf "${d}"
      ((total_cleaned++)) || true
    done < <(find . -maxdepth 3 -type d -name "Backup" -not -path "./.git/*" 2>/dev/null)

    # TD crash logs
    while IFS= read -r f; do
      echo "  ${f}"
      [[ ${execute} -eq 1 ]] && rm -f "${f}"
      ((total_cleaned++)) || true
    done < <(find . -maxdepth 2 -name "crash.*" -not -path "./.git/*" 2>/dev/null)
  fi

  # -- GLSL compiled shaders -------------------------------------------------
  if [[ ${has_glsl} -eq 1 ]]; then
    echo ""
    echo "--- GLSL / Compiled Shaders ---"
    while IFS= read -r f; do
      echo "  ${f}"
      [[ ${execute} -eq 1 ]] && rm -f "${f}"
      ((total_cleaned++)) || true
    done < <(find . -name "*.spv" -not -path "./.git/*" 2>/dev/null)
  fi

  # -- Summary ---------------------------------------------------------------
  echo ""
  if [[ ${total_cleaned} -eq 0 ]]; then
    echo "[OK] Nothing to clean"
  elif [[ ${execute} -eq 1 ]]; then
    echo "[OK] Cleaned ${total_cleaned} item(s)"
  else
    echo "Found ${total_cleaned} item(s) -- run with --execute to delete"
  fi
}

main "$@"
