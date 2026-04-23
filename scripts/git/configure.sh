#!/usr/bin/env bash
# configure.sh - Auto-configure claude-git-workflow for a project
# Purpose: Scan project, generate .cgw.conf, install hooks and optional Claude skill
# Usage: ./scripts/git/configure.sh [OPTIONS]
#
# Run this once after copying scripts/git/ into your project.
# It auto-detects branch names, lint tools, and local-only files,
# then generates .cgw.conf so all scripts work without manual editing.
#
# Arguments:
#   --non-interactive   Accept all auto-detected defaults without prompting
#   --reconfigure       Overwrite existing .cgw.conf
#   --skip-hooks        Don't install git pre-commit hook
#   --skip-skill        Don't install Claude Code skill
#   -h, --help          Show help
# Returns:
#   0 on success, 1 on failure

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# For configure.sh, we detect PROJECT_ROOT ourselves (can't source _common.sh yet
# because _config.sh requires PROJECT_ROOT to already exist for .cgw.conf loading)
_find_project_root() {
  local dir
  dir="$(cd "${SCRIPT_DIR}" && pwd)"
  while [[ "${dir}" != "/" ]] && [[ -n "${dir}" ]]; do
    if [[ -d "${dir}/.git" ]]; then
      echo "${dir}"
      return 0
    fi
    dir="$(dirname "${dir}")"
  done
  git rev-parse --show-toplevel 2>/dev/null && return 0
  return 1
}

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(_find_project_root)" || {
    echo "[ERROR] Cannot find git repository root." >&2
    echo "  Are you inside a git repository? Run 'git init' first, or cd into one." >&2
    exit 1
  }
fi

# ============================================================================
# AUTO-DETECTION FUNCTIONS
# ============================================================================

_detect_target_branch() {
  # Check git remote HEAD pointer
  local remote_head
  remote_head=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's|refs/remotes/origin/||')
  if [[ -n "${remote_head}" ]]; then
    echo "${remote_head}"
    return 0
  fi
  # Check common names
  if git show-ref --verify --quiet refs/heads/main 2>/dev/null; then
    echo "main"
    return 0
  fi
  if git show-ref --verify --quiet refs/heads/master 2>/dev/null; then
    echo "master"
    return 0
  fi
  echo "main"
}

_detect_source_branch() {
  local target="$1"
  # Check common source branch names (local first, then remote tracking)
  for name in development develop dev staging; do
    if git show-ref --verify --quiet "refs/heads/${name}" 2>/dev/null; then
      echo "${name}"
      return 0
    fi
    if git show-ref --verify --quiet "refs/remotes/origin/${name}" 2>/dev/null; then
      # Remote-only: create local tracking branch so downstream scripts can
      # check out by name without relying on git's DWIM --guess behaviour.
      git branch --track "${name}" "origin/${name}" >/dev/null 2>&1 || true
      echo "${name}"
      return 0
    fi
  done
  # Most recently committed branch that isn't target
  local recent
  recent=$(git for-each-ref --sort=-committerdate --format='%(refname:short)' refs/heads/ 2>/dev/null |
    grep -v "^${target}$" | head -1)
  if [[ -n "${recent}" ]]; then
    echo "${recent}"
    return 0
  fi
  echo "${target}" # fallback: same branch (will warn later)
}

_detect_lint_tool() {
  # Python project detection
  if [[ -f "pyproject.toml" ]] || [[ -f "setup.py" ]] || [[ -f "setup.cfg" ]] || [[ -f "requirements.txt" ]]; then
    if command -v ruff &>/dev/null; then
      echo "ruff"
      return 0
    fi
    if command -v flake8 &>/dev/null; then
      echo "flake8"
      return 0
    fi
    if command -v pylint &>/dev/null; then
      echo "pylint"
      return 0
    fi
  fi
  # JavaScript/TypeScript project detection
  if [[ -f "package.json" ]]; then
    if command -v eslint &>/dev/null; then
      echo "eslint"
      return 0
    fi
  fi
  # Go project detection
  if [[ -f "go.mod" ]]; then
    if command -v golangci-lint &>/dev/null; then
      echo "golangci-lint"
      return 0
    fi
  fi
  # Rust project detection
  if [[ -f "Cargo.toml" ]]; then
    if command -v cargo &>/dev/null; then
      echo "cargo"
      return 0
    fi
  fi
  # C/C++ project detection
  if [[ -f "CMakeLists.txt" ]] || [[ -f "Makefile" ]] || [[ -f "meson.build" ]]; then
    if command -v clang-tidy &>/dev/null; then
      echo "clang-tidy"
      return 0
    fi
    if command -v cppcheck &>/dev/null; then
      echo "cppcheck"
      return 0
    fi
  fi
  echo "" # no lint tool detected
}

_detect_format_tool() {
  local lint_tool="$1"
  case "${lint_tool}" in
    ruff) echo "ruff" ;;
    eslint)
      if command -v prettier &>/dev/null; then echo "prettier"; else echo ""; fi
      ;;
    clang-tidy | cppcheck)
      if command -v clang-format &>/dev/null; then echo "clang-format"; else echo ""; fi
      ;;
    *) echo "" ;;
  esac
}

_detect_local_files() {
  # Scan for files that exist on disk but are not tracked by git
  local files=()
  local check_files=(CLAUDE.md MEMORY.md SESSION_LOG.md GEMINI.md .env .env.local .env.development .env.production)
  local check_dirs=(.claude/ logs/)

  for f in "${check_files[@]}"; do
    if [[ -f "${PROJECT_ROOT}/${f}" ]] && ! git -C "${PROJECT_ROOT}" ls-files --error-unmatch "${f}" &>/dev/null 2>&1; then
      files+=("${f}")
    fi
  done

  for d in "${check_dirs[@]}"; do
    local dir_path="${PROJECT_ROOT}/${d%/}"
    if [[ -d "${dir_path}" ]] && ! git -C "${PROJECT_ROOT}" ls-files --error-unmatch "${d}" &>/dev/null 2>&1; then
      files+=("${d}")
    fi
  done

  echo "${files[*]:-}"
}

_detect_venv() {
  local venv_dirs=(".venv" "venv" "env" ".env")
  for d in "${venv_dirs[@]}"; do
    if [[ -d "${PROJECT_ROOT}/${d}" ]]; then
      echo "${d}"
      return 0
    fi
  done
  echo ""
}

_build_lint_config() {
  local lint_tool="$1"
  local venv_dir="$2"

  case "${lint_tool}" in
    ruff)
      local excludes="--extend-exclude logs"
      if [[ -n "${venv_dir}" ]]; then
        excludes="${excludes} --extend-exclude ${venv_dir}"
      fi
      echo "CGW_LINT_CMD=\"ruff\""
      echo "CGW_LINT_CHECK_ARGS=\"check .\""
      echo "CGW_LINT_FIX_ARGS=\"check --fix .\""
      echo "CGW_LINT_EXCLUDES=\"${excludes}\""
      echo "CGW_FORMAT_CMD=\"ruff\""
      echo "CGW_FORMAT_CHECK_ARGS=\"format --check .\""
      echo "CGW_FORMAT_FIX_ARGS=\"format .\""
      local fmt_excludes="--exclude logs"
      if [[ -n "${venv_dir}" ]]; then fmt_excludes="${fmt_excludes} --exclude ${venv_dir}"; fi
      echo "CGW_FORMAT_EXCLUDES=\"${fmt_excludes}\""
      ;;
    flake8)
      echo "CGW_LINT_CMD=\"flake8\""
      echo "CGW_LINT_CHECK_ARGS=\".\""
      echo "CGW_LINT_FIX_ARGS=\".\"  # flake8 has no auto-fix; use autopep8 manually"
      echo "CGW_LINT_EXCLUDES=\"--exclude logs,.venv\""
      echo "CGW_FORMAT_CMD=\"\"  # set to 'black' or 'autopep8' if available"
      echo "CGW_FORMAT_CHECK_ARGS=\"\""
      echo "CGW_FORMAT_FIX_ARGS=\"\""
      echo "CGW_FORMAT_EXCLUDES=\"\""
      ;;
    eslint)
      echo "CGW_LINT_CMD=\"eslint\""
      echo "CGW_LINT_CHECK_ARGS=\".\""
      echo "CGW_LINT_FIX_ARGS=\". --fix\""
      echo "CGW_LINT_EXCLUDES=\"\""
      echo "CGW_FORMAT_CMD=\"prettier\""
      echo "CGW_FORMAT_CHECK_ARGS=\"--check .\""
      echo "CGW_FORMAT_FIX_ARGS=\"--write .\""
      echo "CGW_FORMAT_EXCLUDES=\"\""
      ;;
    golangci-lint)
      echo "CGW_LINT_CMD=\"golangci-lint\""
      echo "CGW_LINT_CHECK_ARGS=\"run\""
      echo "CGW_LINT_FIX_ARGS=\"run --fix\""
      echo "CGW_LINT_EXCLUDES=\"\""
      echo "CGW_FORMAT_CMD=\"gofmt\""
      echo "CGW_FORMAT_CHECK_ARGS=\"-l .\""
      echo "CGW_FORMAT_FIX_ARGS=\"-w .\""
      echo "CGW_FORMAT_EXCLUDES=\"\""
      ;;
    clang-tidy)
      echo "CGW_LINT_CMD=\"clang-tidy\""
      echo "CGW_LINT_CHECK_ARGS=\"-p build\"  # adjust: path to compile_commands.json dir"
      echo "CGW_LINT_FIX_ARGS=\"-p build --fix\""
      echo "CGW_LINT_EXCLUDES=\"\""
      echo "CGW_FORMAT_CMD=\"clang-format\""
      echo "CGW_FORMAT_CHECK_ARGS=\"--dry-run --Werror -r .\""
      echo "CGW_FORMAT_FIX_ARGS=\"-i -r .\""
      echo "CGW_FORMAT_EXCLUDES=\"\""
      ;;
    cppcheck)
      echo "CGW_LINT_CMD=\"cppcheck\""
      echo "CGW_LINT_CHECK_ARGS=\"--enable=all --error-exitcode=1 .\""
      echo "CGW_LINT_FIX_ARGS=\"--enable=all --error-exitcode=1 .\"  # cppcheck has no auto-fix"
      echo "CGW_LINT_EXCLUDES=\"\""
      echo "CGW_FORMAT_CMD=\"clang-format\""
      echo "CGW_FORMAT_CHECK_ARGS=\"--dry-run --Werror -r .\""
      echo "CGW_FORMAT_FIX_ARGS=\"-i -r .\""
      echo "CGW_FORMAT_EXCLUDES=\"\""
      ;;
    "")
      echo "CGW_LINT_CMD=\"\"  # no lint tool detected; set to enable"
      echo "CGW_LINT_CHECK_ARGS=\"\""
      echo "CGW_LINT_FIX_ARGS=\"\""
      echo "CGW_LINT_EXCLUDES=\"\""
      echo "CGW_FORMAT_CMD=\"\""
      echo "CGW_FORMAT_CHECK_ARGS=\"\""
      echo "CGW_FORMAT_FIX_ARGS=\"\""
      echo "CGW_FORMAT_EXCLUDES=\"\""
      ;;
    *)
      echo "CGW_LINT_CMD=\"${lint_tool}\""
      echo "CGW_LINT_CHECK_ARGS=\".\"  # adjust for your tool"
      echo "CGW_LINT_FIX_ARGS=\".\"    # adjust for your tool"
      echo "CGW_LINT_EXCLUDES=\"\""
      echo "CGW_FORMAT_CMD=\"\""
      echo "CGW_FORMAT_CHECK_ARGS=\"\""
      echo "CGW_FORMAT_FIX_ARGS=\"\""
      echo "CGW_FORMAT_EXCLUDES=\"\""
      ;;
  esac
}

_install_hook() {
  local local_files="$1"
  local hooks_template_dir="${SCRIPT_DIR}/../../hooks"

  # Try staging area first (present during install.cmd), then fall back to already-installed hook
  hooks_template_dir="$(cd "${SCRIPT_DIR}" && cd "../../hooks" 2>/dev/null && pwd)" || {
    hooks_template_dir="${PROJECT_ROOT}/.cgw-hooks-template"
  }

  local hook_template="${hooks_template_dir}/pre-commit"

  if [[ ! -f "${hook_template}" ]]; then
    # If hook is already installed, nothing to do
    if [[ -f "${PROJECT_ROOT}/.githooks/pre-commit" ]]; then
      echo "  [OK] Pre-commit hook already installed"
      return 0
    fi
    echo "  [!] Hook template not found at: ${hook_template}"
    echo "      Fix: copy the hooks/ directory from the CGW source repo into your project root,"
    echo "      then re-run: ./scripts/git/configure.sh"
    return 1
  fi

  # Build regex pattern from local files list
  local files_pattern=""
  for f in ${local_files}; do
    local escaped="${f%/}"      # strip trailing slash
    escaped="${escaped//./\\.}" # escape dots
    [[ -n "${files_pattern}" ]] && files_pattern="${files_pattern}|"
    files_pattern="${files_pattern}${escaped}"
  done

  # Create .githooks/ and write patched pre-commit hook
  # Escape backslashes first, then & (sed replacement special char), then | (sed delimiter)
  local sed_files_pattern="${files_pattern//\\/\\\\}"
  sed_files_pattern="${sed_files_pattern//&/\\&}"
  sed_files_pattern="${sed_files_pattern//|/\\|}"
  mkdir -p "${PROJECT_ROOT}/.githooks"
  sed "s|__CGW_LOCAL_FILES_PATTERN__|${sed_files_pattern}|g" \
    "${hook_template}" >"${PROJECT_ROOT}/.githooks/pre-commit"
  chmod +x "${PROJECT_ROOT}/.githooks/pre-commit"

  # Also install pre-push hook if template exists alongside pre-commit
  local pre_push_template="${hooks_template_dir}/pre-push"
  if [[ -f "${pre_push_template}" ]]; then
    # Build CGW_ALL_PREFIXES for substitution into pre-push template.
    # Can't source _config.sh here (see top-of-file comment), so compute locally
    # by reading CGW_EXTRA_PREFIXES from the just-written .cgw.conf.
    local _base_prefixes="feat|fix|docs|chore|test|refactor|style|perf"
    local _extra_prefixes
    _extra_prefixes=$(grep -m1 '^CGW_EXTRA_PREFIXES=' "${PROJECT_ROOT}/.cgw.conf" |
      sed 's/CGW_EXTRA_PREFIXES=//;s/"//g' || true)
    local _all_prefixes
    if [[ -n "${_extra_prefixes}" ]]; then
      _all_prefixes="${_base_prefixes}|${_extra_prefixes}"
    else
      _all_prefixes="${_base_prefixes}"
    fi
    local all_prefixes_escaped="${_all_prefixes//\\/\\\\}"
    all_prefixes_escaped="${all_prefixes_escaped//&/\\&}"
    all_prefixes_escaped="${all_prefixes_escaped//|/\\|}"
    sed -e "s|__CGW_LOCAL_FILES_PATTERN__|${sed_files_pattern}|g" \
      -e "s|__CGW_ALL_PREFIXES__|${all_prefixes_escaped}|g" \
      "${pre_push_template}" >"${PROJECT_ROOT}/.githooks/pre-push"
    chmod +x "${PROJECT_ROOT}/.githooks/pre-push"
  fi

  # Run install_hooks.sh to copy to .git/hooks/
  if bash "${SCRIPT_DIR}/install_hooks.sh" >/dev/null 2>&1; then
    echo "  [OK] Git hooks installed (pre-commit + pre-push)"
  else
    echo "  [!] Hooks written to .githooks/ but failed to copy to .git/hooks/"
    echo "      Fix: run manually: ./scripts/git/install_hooks.sh"
    echo "      If that also fails, check that .git/hooks/ is writable."
  fi
}

_install_skill() {
  local install_mode="${1:-local}" # "local" or "global"
  local skill_src
  local cmd_src
  local skill_dst cmd_dst

  # Determine destination based on install mode
  if [[ "${install_mode}" == "global" ]]; then
    skill_dst="${HOME}/.claude/skills/auto-git-workflow"
    cmd_dst="${HOME}/.claude/commands"
    echo "  Installing skill globally to ${HOME}/.claude/"
  else
    skill_dst="${PROJECT_ROOT}/.claude/skills/auto-git-workflow"
    cmd_dst="${PROJECT_ROOT}/.claude/commands"
  fi

  # Try staging area first (present during install.cmd), then CGW source repo
  if skill_src="$(cd "${SCRIPT_DIR}" && cd "../../skill" 2>/dev/null && pwd)"; then
    cmd_src="${skill_src}/../command/auto-git-workflow.md"
  elif [[ -f "${skill_dst}/SKILL.md" ]]; then
    echo "  [OK] Claude Code skill already installed (${install_mode})"
    return 0
  else
    echo "  [!] Skill template not found."
    echo "      Fix: copy skill/ and command/ from the CGW source repo into your"
    echo "      project root, then re-run: ./scripts/git/configure.sh"
    return 1
  fi

  mkdir -p "${skill_dst}/references"

  cp "${skill_src}/SKILL.md" "${skill_dst}/SKILL.md" 2>/dev/null || true
  cp "${skill_src}/references/"*.md "${skill_dst}/references/" 2>/dev/null || true

  if [[ -f "${cmd_src}" ]]; then
    mkdir -p "${cmd_dst}"
    cp "${cmd_src}" "${cmd_dst}/auto-git-workflow.md" 2>/dev/null || true
    echo "  [OK] Claude Code skill + slash command installed (${install_mode})"
  else
    echo "  [OK] Claude Code skill installed (${install_mode}, command template not found)"
  fi
}

_update_gitignore() {
  local gitignore="${PROJECT_ROOT}/.gitignore"
  local entries=("logs/" ".cgw.conf")
  local added=()

  for entry in "${entries[@]}"; do
    if [[ ! -f "${gitignore}" ]] || ! grep -qxF "${entry}" "${gitignore}" 2>/dev/null; then
      echo "${entry}" >>"${gitignore}"
      added+=("${entry}")
    fi
  done

  if [[ ${#added[@]} -gt 0 ]]; then
    echo "  [OK] Added to .gitignore: ${added[*]}"
  else
    echo "  [OK] .gitignore already up to date"
  fi
}

# ============================================================================
# MAIN
# ============================================================================

main() {
  local non_interactive=0
  local reconfigure=0
  local skip_hooks=0
  local skip_skill=0
  local global_skill=0

  while [[ $# -gt 0 ]]; do
    case "${1}" in
      --help | -h)
        echo "Usage: ./scripts/git/configure.sh [OPTIONS]"
        echo ""
        echo "Auto-configure claude-git-workflow for this project."
        echo "Scans the project and generates .cgw.conf, installs hooks,"
        echo "and optionally installs the Claude Code skill."
        echo ""
        echo "Options:"
        echo "  --non-interactive   Accept all auto-detected defaults"
        echo "  --reconfigure       Overwrite existing .cgw.conf"
        echo "  --skip-hooks        Don't install git pre-commit hook"
        echo "  --skip-skill        Don't install Claude Code skill"
        echo "  --global            Install Claude Code skill to ~/.claude/ (available in all projects)"
        echo "  -h, --help          Show this help"
        echo ""
        echo "After running, edit .cgw.conf to customize any detected values."
        exit 0
        ;;
      --non-interactive) non_interactive=1 ;;
      --reconfigure) reconfigure=1 ;;
      --skip-hooks) skip_hooks=1 ;;
      --skip-skill) skip_skill=1 ;;
      --global) global_skill=1 ;;
      *)
        echo "[ERROR] Unknown flag: $1" >&2
        exit 1
        ;;
    esac
    shift
  done

  cd "${PROJECT_ROOT}" || {
    echo "[ERROR] Cannot change to project root: ${PROJECT_ROOT}" >&2
    exit 1
  }

  echo ""
  echo "=== claude-git-workflow: Auto-Configuration ==="
  echo ""
  echo "Project root: ${PROJECT_ROOT}"
  echo ""

  # Track whether this is a fresh install (no existing .cgw.conf)
  local fresh_install=0
  [[ ! -f ".cgw.conf" ]] && fresh_install=1

  # Check if .cgw.conf already exists
  if [[ -f ".cgw.conf" ]] && [[ ${reconfigure} -eq 0 ]]; then
    echo "[OK] .cgw.conf already exists."
    if [[ ${non_interactive} -eq 0 ]]; then
      read -r -p "  Reconfigure? (yes/no) [no]: " answer
      if [[ "$(echo "${answer}" | tr '[:upper:]' '[:lower:]')" =~ ^y(es)?$ ]]; then
        reconfigure=1
      else
        echo ""
        echo "Using existing configuration. Use --reconfigure to overwrite."
        echo ""
        # Still run hook + skill install
      fi
    else
      echo "  Use --reconfigure to overwrite."
    fi
  fi

  # -- Detection phase ------------------------------------------------------

  echo "Scanning project..."
  echo "  Detecting branch names, lint tools, virtual environment, and local-only files..."
  echo ""

  local detected_target
  detected_target="$(_detect_target_branch)"

  local detected_source
  detected_source="$(_detect_source_branch "${detected_target}")"

  local detected_lint
  detected_lint="$(_detect_lint_tool)"

  local detected_venv
  detected_venv="$(_detect_venv)"

  local detected_local_files
  detected_local_files="$(_detect_local_files)"

  echo "  Target branch (stable):  ${detected_target}"
  echo "  Source branch (dev):     ${detected_source}"
  echo "  Lint tool:               ${detected_lint:-none detected}"
  echo "  Venv directory:          ${detected_venv:-none found}"
  echo "  Local-only files:        ${detected_local_files:-none found}"
  echo ""

  # -- Interactive confirmation (only when generating/updating config) ----------

  local target_branch="${detected_target}"
  local source_branch="${detected_source}"

  local local_files="${detected_local_files:-CLAUDE.md MEMORY.md .claude/ logs/}"

  if [[ ${non_interactive} -eq 0 ]] && { [[ ! -f ".cgw.conf" ]] || [[ ${reconfigure} -eq 1 ]]; }; then
    echo "Press Enter to accept [default], or type a different value."
    echo ""
    read -e -r -p "Target branch [${target_branch}]: " answer
    [[ -n "${answer}" && ! "${answer}" =~ ^[Yy]([Ee][Ss])?$ ]] && target_branch="${answer}"

    read -e -r -p "Source branch [${source_branch}]: " answer
    [[ -n "${answer}" && ! "${answer}" =~ ^[Yy]([Ee][Ss])?$ ]] && source_branch="${answer}"

    echo ""
    echo "Local-only files (never committed): ${local_files}"
    read -e -r -p "Add/change local files? (press Enter to keep, or type new list): " answer
    [[ -n "${answer}" && ! "${answer}" =~ ^[Yy]([Ee][Ss])?$ ]] && local_files="${answer}"
  fi

  # -- Generate .cgw.conf ----------------------------------------------------

  if [[ ! -f ".cgw.conf" ]] || [[ ${reconfigure} -eq 1 ]]; then
    echo "Generating .cgw.conf..."
    echo "  This config file controls branch names, lint settings, and local-only"
    echo "  file protection. It is git-ignored so each developer can have their own."

    {
      echo "# .cgw.conf -- Auto-generated by configure.sh on $(date)"
      echo "# Edit as needed. See cgw.conf.example for all options."
      echo "# This file is git-ignored (.cgw.conf in .gitignore)."
      echo ""
      echo "# Branch configuration"
      echo "CGW_SOURCE_BRANCH=\"${source_branch}\""
      echo "CGW_TARGET_BRANCH=\"${target_branch}\""
      echo ""
      echo "# Local-only files (space-separated; never committed)"
      echo "CGW_LOCAL_FILES=\"${local_files}\""
      echo ""
      echo "# Lint configuration (auto-detected)"
      _build_lint_config "${detected_lint}" "${detected_venv}"
      echo ""
      echo "# Commit message prefix extras (pipe-separated, e.g. \"cuda|tensorrt\")"
      echo "CGW_EXTRA_PREFIXES=\"\""
      echo ""
      echo "# Docs CI pattern (empty = skip; set to enable doc filename validation)"
      echo "# Example: CGW_DOCS_PATTERN=\"^(README\\.md|.*_GUIDE\\.md|.*_REFERENCE\\.md)$\""
      echo "CGW_DOCS_PATTERN=\"\""
      echo ""
      echo "# Dev-only files warning for cherry-pick (space-separated; empty = skip)"
      echo "# Example: CGW_DEV_ONLY_FILES=\"tests/ pytest.ini\""
      echo "CGW_DEV_ONLY_FILES=\"\""
      echo ""
      echo "# Remove tests/ from target branch if gitignored (0=disabled, 1=enabled)"
      echo "CGW_CLEANUP_TESTS=\"0\""
    } >".cgw.conf"

    echo "  [OK] .cgw.conf generated"
  fi

  # -- Update .gitignore (first install only) --------------------------------
  # Only on fresh installs -- not on --reconfigure, so existing .gitignore
  # entries the user has customised are not modified.
  if [[ ${fresh_install} -eq 1 ]] && [[ ${reconfigure} -eq 0 ]]; then
    echo "Updating .gitignore..."
    _update_gitignore
  fi

  # -- Install pre-commit hook -----------------------------------------------

  if [[ ${skip_hooks} -eq 0 ]]; then
    echo ""
    echo "Git hooks enforce lint checks and local-file protection on every commit"
    echo "and push, catching issues before they reach the remote."
    local install_hook="yes"
    if [[ ${non_interactive} -eq 0 ]]; then
      read -r -p "Install pre-commit hook? (yes/no) [yes]: " answer
      case "$(echo "${answer}" | tr '[:upper:]' '[:lower:]')" in
        y | yes) install_hook="yes" ;;
        n | no) install_hook="no" ;;
      esac
    fi

    if [[ "${install_hook}" == "yes" ]]; then
      echo "Installing pre-commit hook..."
      _install_hook "${local_files}"
    fi
  fi

  # -- Enable git rerere -----------------------------------------------------
  # rerere (reuse recorded resolution) auto-replays known conflict resolutions.
  # Recommended for two-branch models where the same conflicts recur across merges.

  echo ""
  echo "git rerere remembers how you resolved conflicts so it can auto-replay"
  echo "the same resolution next time the same conflict reappears."
  local enable_rerere="yes"
  if [[ ${non_interactive} -eq 0 ]]; then
    read -r -p "Enable git rerere (auto-replay conflict resolutions)? (yes/no) [yes]: " answer
    case "$(echo "${answer}" | tr '[:upper:]' '[:lower:]')" in
      n | no) enable_rerere="no" ;;
    esac
  fi

  if [[ "${enable_rerere}" == "yes" ]]; then
    if git config rerere.enabled true 2>/dev/null; then
      echo "  [OK] rerere.enabled = true (conflict resolutions will be remembered)"
    else
      echo "    Note: Could not enable rerere -- run: git config rerere.enabled true"
    fi
  fi

  # -- Install Claude Code skill ---------------------------------------------

  if [[ ${skip_skill} -eq 0 ]]; then
    echo ""
    echo "The Claude Code skill teaches Claude to use CGW scripts instead of raw"
    echo "git commands, ensuring lint checks and local-file protection are never bypassed."
    if [[ ${global_skill} -eq 1 ]]; then
      echo "  (--global: skill will be installed to ~/.claude/ for all projects)"
    fi
    local install_skill="no"
    # Default to yes if .claude/ directory already exists (local mode)
    # or if --global was specified
    if [[ -d ".claude" ]] || [[ ${global_skill} -eq 1 ]]; then
      install_skill="yes"
    fi

    if [[ ${non_interactive} -eq 0 ]]; then
      local skill_dest_hint="project .claude/"
      [[ ${global_skill} -eq 1 ]] && skill_dest_hint="global ~/.claude/"
      read -r -p "Install Claude Code skill to ${skill_dest_hint}? (yes/no) [${install_skill}]: " answer
      case "$(echo "${answer}" | tr '[:upper:]' '[:lower:]')" in
        y | yes) install_skill="yes" ;;
        n | no) install_skill="no" ;;
      esac
    fi

    if [[ "${install_skill}" == "yes" ]]; then
      echo "Installing Claude Code skill..."
      if [[ ${global_skill} -eq 1 ]]; then
        _install_skill "global"
      else
        _install_skill "local"
      fi
    fi
  fi

  # -- Summary --------------------------------------------------------------

  echo ""
  echo "=== Configuration Complete ==="
  echo ""
  echo "  Config file:    ${PROJECT_ROOT}/.cgw.conf"
  # When not reconfiguring, show values from existing .cgw.conf rather than detected values
  if [[ -f ".cgw.conf" ]] && [[ ${reconfigure} -eq 0 ]]; then
    local conf_source conf_target
    conf_source=$(grep -m1 '^CGW_SOURCE_BRANCH=' .cgw.conf | sed 's/CGW_SOURCE_BRANCH=//;s/"//g')
    conf_target=$(grep -m1 '^CGW_TARGET_BRANCH=' .cgw.conf | sed 's/CGW_TARGET_BRANCH=//;s/"//g')
    echo "  Source branch:  ${conf_source:-${source_branch}}"
    echo "  Target branch:  ${conf_target:-${target_branch}}"
  else
    echo "  Source branch:  ${source_branch}"
    echo "  Target branch:  ${target_branch}"
  fi
  if [[ -n "${detected_lint}" ]]; then
    echo "  Lint tool:      ${detected_lint}"
  fi
  echo ""
  echo "Quick start:"
  echo "  ./scripts/git/commit_enhanced.sh \"feat: your feature\""
  echo "  ./scripts/git/merge_with_validation.sh --dry-run"
  echo "  ./scripts/git/push_validated.sh"
  echo ""
  echo "Edit .cgw.conf to customize any settings."
  echo ""
}

main "$@"
