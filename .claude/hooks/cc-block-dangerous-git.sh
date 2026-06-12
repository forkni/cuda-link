#!/usr/bin/env bash
# cc-block-dangerous-git.sh — Claude Code PreToolUse guardrail (installed by CGW configure.sh)
#
# Blocks dangerous git/shell commands before they reach the shell.
# Claude Code invokes this hook for every Bash tool call (PreToolUse: Bash matcher).
#
# Protocol (Claude Code hook contract):
#   - Tool input arrives as JSON on stdin
#   - Exit 2 + stderr content → Claude Code blocks the call; stderr is shown to the model
#   - Exit 0 → command is allowed through
#
# Fail-open policy: if jq is absent or stdin is unparseable, the guardrail
# degrades gracefully (logs a warning, allows the command through) rather than
# breaking the user's shell.
#
# To uninstall: remove the PreToolUse entry from .claude/settings.json and
#   delete this file.
# To temporarily disable: set SKIP_CGW_GUARDRAIL=1 in your environment.
#
# shellcheck disable=SC2034

set -uo pipefail

[[ "${SKIP_CGW_GUARDRAIL:-}" == "1" ]] && exit 0

INPUT=$(cat)

# Fail open: if jq is absent, warn and allow through
if ! command -v jq &>/dev/null; then
  printf '[CGW guardrail] WARNING: jq not found — guardrail is degraded; commands are not being inspected\n' >&2
  exit 0
fi

COMMAND=$(jq -r '.tool_input.command // empty' <<< "${INPUT}" 2>/dev/null)
[[ -z "${COMMAND}" ]] && exit 0

# Strip quoted-string contents before pattern matching so that blocked keywords
# appearing inside commit messages or other string arguments do not cause false
# positives.  For example, commit_enhanced.sh "docs: explain git commit workflow"
# should not match the 'git commit' block.  Heuristic: removes "..." and '...'
# (does not handle nested/escaped quotes, but covers all practical CGW cases).
COMMAND_UNQUOTED=$(sed 's/"[^"]*"//g; s/'"'"'[^'"'"']*'"'"'//g' <<< "${COMMAND}")

# ── Block helper ──────────────────────────────────────────────────────────────

_block() {
  local pattern="$1"
  local redirect="$2"
  printf 'BLOCKED: Command matched dangerous pattern "%s".\n%s\nThe user has prevented you from doing this.\n' \
    "${pattern}" "${redirect}" >&2
  exit 2
}

# ── Pattern checks ────────────────────────────────────────────────────────────
# Each check: if the command matches the pattern, block with a redirect message.
# CGW wrapper scripts in scripts/git/ are trusted; their subprocesses
# are not intercepted by this hook (PreToolUse only fires for direct Bash calls).

# Raw git commit — bypasses lint, local-file protection, conventional commit enforcement
grep -qE 'git commit( |$)' <<< "${COMMAND_UNQUOTED}" \
  && _block 'git commit' \
     'Use ./scripts/git/commit_enhanced.sh "<type>: <msg>" instead — it runs lint, protects local-only files, and enforces conventional commit format.'

# --no-verify — bypasses pre-commit and pre-push hooks entirely
grep -qE -- '--no-verify' <<< "${COMMAND_UNQUOTED}" \
  && _block '--no-verify' \
     'CGW pre-commit/pre-push hooks cannot be bypassed with --no-verify. Fix the underlying issue (run ./scripts/git/fix_lint.sh for lint errors, or inspect the hook output).'

# Force-push without lease — overwrites others work and bypasses protection
# Note: --force-with-lease is explicitly allowed (it is what push_validated.sh uses)
grep -qE 'git push --force( |$)' <<< "${COMMAND_UNQUOTED}" \
  && _block 'git push --force' \
     'Use ./scripts/git/push_validated.sh instead — it uses --force-with-lease and requires confirmation on protected branches. Note: --force-with-lease is allowed.'

# Hard reset — irreversibly discards uncommitted work and index changes
grep -qE 'git reset --hard' <<< "${COMMAND_UNQUOTED}" \
  && _block 'git reset --hard' \
     'Confirm with the user before running git reset --hard. This irreversibly discards uncommitted work and index changes.'

# git clean -f — permanently deletes untracked files (covers -f, -fd, -fdx)
grep -qE 'git clean -f' <<< "${COMMAND_UNQUOTED}" \
  && _block 'git clean -f' \
     'Confirm with the user before running git clean. This permanently deletes untracked files from the working tree.'

# Force-delete branch — may lose commits on an unmerged branch
grep -qE 'git branch -D( |$)' <<< "${COMMAND_UNQUOTED}" \
  && _block 'git branch -D' \
     'Use ./scripts/git/branch_cleanup.sh --execute to prune merged branches, or confirm with the user before force-deleting an unmerged branch.'

# Discard all working-tree changes (. = current directory = everything)
grep -qE 'git checkout \.( |$)' <<< "${COMMAND_UNQUOTED}" \
  && _block 'git checkout .' \
     'Confirm with the user — git checkout . irreversibly discards all working-tree changes.'

grep -qE 'git restore \.( |$)' <<< "${COMMAND_UNQUOTED}" \
  && _block 'git restore .' \
     'Confirm with the user — git restore . irreversibly discards all working-tree changes.'

# History rewrites — permanently alter commit history; extremely destructive
grep -qE 'git filter-branch' <<< "${COMMAND_UNQUOTED}" \
  && _block 'git filter-branch' \
     'Destructive history rewrite — confirm with the user before proceeding.'

grep -qE 'git filter-repo' <<< "${COMMAND_UNQUOTED}" \
  && _block 'git filter-repo' \
     'Destructive history rewrite — confirm with the user before proceeding.'

# Recovery ref destruction — eliminates the ability to recover lost commits
grep -qE 'git reflog expire' <<< "${COMMAND_UNQUOTED}" \
  && _block 'git reflog expire' \
     'This destroys reflog recovery references — confirm with the user.'

grep -qE 'git gc --prune=now' <<< "${COMMAND_UNQUOTED}" \
  && _block 'git gc --prune=now' \
     'This destroys unreachable objects needed for recovery — confirm with the user.'

grep -qE 'git update-ref -d' <<< "${COMMAND_UNQUOTED}" \
  && _block 'git update-ref -d' \
     'Destructive ref operation — confirm with the user.'

# .git directory destruction
# Catches: rm -rf .git  rm -rf .git/  rm -rf /path/to/.git
# Allows:  rm -rf .gitignore  rm -rf .github  (character after .git is alphanumeric)
if grep -qE 'rm -rf' <<< "${COMMAND_UNQUOTED}" \
   && grep -qE '[.]git(/| |$)' <<< "${COMMAND_UNQUOTED}"; then
  _block 'rm -rf .git' \
    'This would destroy the git repository — confirm with the user first.'
fi

exit 0
