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
# Heuristic limits (defense-in-depth, not a sandbox): this hook does not evaluate
# `eval`, shell aliases/functions, `git -C <path> <subcmd>` (the subcommand isn't
# adjacent to `git`), or paths hidden inside quotes — e.g. `rm -rf "$HOME/.git"`
# is stripped by the quote-stripping heuristic below before pattern matching runs.
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

# Split into individual shell invocations before pattern matching, so a flag or
# exemption belonging to one command (e.g. `--cached` after a `;`) cannot satisfy
# a check for a different, earlier command on the same line. Join backslash-
# newline continuations first so a wrapped command stays one invocation, then
# split on shell separators (; | & newline). `&&` / `||` produce an empty
# segment, which matches no pattern and is harmlessly skipped.
COMMAND_JOINED="${COMMAND_UNQUOTED//\\$'\n'/ }"
COMMAND_SEGMENTED="${COMMAND_JOINED//[;|&$'\n']/$'\n'}"

# ── Block helper ──────────────────────────────────────────────────────────────

_block() {
  local pattern="$1"
  local redirect="$2"
  printf 'BLOCKED: Command matched dangerous pattern "%s".\n%s\nThe user has prevented you from doing this.\n' \
    "${pattern}" "${redirect}" >&2
  exit 2
}

# ── Pattern checks ────────────────────────────────────────────────────────────
# Each invocation (one shell command — see segmentation above) is checked in
# isolation with bash's built-in [[ =~ ]] regex operator against a padded copy
# (" ${cmd} "), so every token is whitespace-delimited on both sides. That
# absorbs tabs/repeated spaces without spawning a subprocess per check — with
# segmentation multiplying the check count by segment count, per-check grep
# subprocesses would be visibly slow under Git Bash on Windows.
# CGW wrapper scripts in scripts/git/ are trusted; their subprocesses
# are not intercepted by this hook (PreToolUse only fires for direct Bash calls).

_check_invocation() {
  local padded=" $1 "

  # Reusable flag fragments (unquoted so [[ =~ ]] treats them as regex, not literals)
  local _force='[[:space:]](-[A-Za-z]*f[A-Za-z]*|--force)[[:space:]]'
  local _cached='[[:space:]]--cached([[:space:]]|=)'
  local _dryrun='[[:space:]](-[A-Za-z]*n[A-Za-z]*|--dry-run)[[:space:]]'
  local _dot='[[:space:]][.][[:space:]]'
  local _staged='[[:space:]]--staged([[:space:]]|=)'
  local _worktree='[[:space:]]--worktree([[:space:]]|=)'

  # Raw git commit — bypasses lint, local-file protection, conventional commit enforcement
  if [[ ${padded} =~ git[[:space:]]+commit[[:space:]] ]]; then
    _block 'git commit' \
      'Use ./scripts/git/commit_enhanced.sh "<type>: <msg>" instead — it runs lint, protects local-only files, and enforces conventional commit format.'
  fi

  # --no-verify — bypasses pre-commit and pre-push hooks entirely
  if [[ ${padded} =~ [[:space:]]--no-verify([[:space:]]|=) ]]; then
    _block '--no-verify' \
      'CGW pre-commit/pre-push hooks cannot be bypassed with --no-verify. Fix the underlying issue (run ./scripts/git/fix_lint.sh for lint errors, or inspect the hook output).'
  fi

  # Force-push without lease — overwrites others' work and bypasses protection.
  # --force-with-lease is explicitly allowed (it is what push_validated.sh uses):
  # a '-' follows "--force" in that flag, so the trailing [[:space:]] never matches.
  if [[ ${padded} =~ git[[:space:]]+push[[:space:]] ]] && [[ ${padded} =~ ${_force} ]]; then
    _block 'git push --force' \
      'Use ./scripts/git/push_validated.sh instead — it uses --force-with-lease and requires confirmation on protected branches. Note: --force-with-lease is allowed.'
  fi

  # Hard reset — irreversibly discards uncommitted work and index changes
  if [[ ${padded} =~ git[[:space:]]+reset[[:space:]] ]] && [[ ${padded} =~ [[:space:]]--hard([[:space:]]|=) ]]; then
    _block 'git reset --hard' \
      'Confirm with the user before running git reset --hard. This irreversibly discards uncommitted work and index changes.'
  fi

  # git clean -f — permanently deletes untracked files (covers -f, -fd, -df, -fdx, ...).
  # A dry-run flag (-n / --dry-run) only previews the deletion, so it is exempt.
  if [[ ${padded} =~ git[[:space:]]+clean[[:space:]] ]] \
     && [[ ${padded} =~ ${_force} ]] \
     && ! [[ ${padded} =~ ${_dryrun} ]]; then
    _block 'git clean -f' \
      'Confirm with the user before running git clean. This permanently deletes untracked files from the working tree.'
  fi

  # git rm with a force flag deletes files from the WORKING TREE. git itself refuses
  # `git rm` on modified/added files ("use -f to force"); -f overrides that and can
  # UNRECOVERABLY delete git-ignored / untracked-turned-added files (e.g. local-only
  # artifacts staged by `git cherry-pick -n`). --cached is index-only (keeps the file
  # on disk — the documented untrack workflow), so a --cached in the SAME invocation
  # is always allowed.
  if [[ ${padded} =~ git[[:space:]]+rm[[:space:]] ]] \
     && ! [[ ${padded} =~ ${_cached} ]] \
     && [[ ${padded} =~ ${_force} ]]; then
    _block 'git rm -f' \
      'git rm -f deletes files from the working tree — for git-ignored or untracked files this is UNRECOVERABLE. To untrack a file while keeping it on disk, use git rm --cached <path>. To force-delete a tracked file, confirm with the user first.'
  fi

  # Force-delete branch — may lose commits on an unmerged branch (-D, not -d)
  if [[ ${padded} =~ git[[:space:]]+branch[[:space:]] ]] \
     && [[ ${padded} =~ [[:space:]]-[A-Za-z]*D[A-Za-z]*[[:space:]] ]]; then
    _block 'git branch -D' \
      'Use ./scripts/git/branch_cleanup.sh --execute to prune merged branches, or confirm with the user before force-deleting an unmerged branch.'
  fi

  # Discard all working-tree changes (. = current directory = everything)
  if [[ ${padded} =~ git[[:space:]]+checkout[[:space:]] ]] && [[ ${padded} =~ ${_dot} ]]; then
    _block 'git checkout .' \
      'Confirm with the user — git checkout . irreversibly discards all working-tree changes.'
  fi

  if [[ ${padded} =~ git[[:space:]]+restore[[:space:]] ]] && [[ ${padded} =~ ${_dot} ]]; then
    # --staged alone only touches the index (reversible with `git reset`) — exempt.
    # --staged combined with --worktree (or --worktree/default alone) discards
    # working-tree changes and stays blocked.
    if ! { [[ ${padded} =~ ${_staged} ]] && ! [[ ${padded} =~ ${_worktree} ]]; }; then
      _block 'git restore .' \
        'Confirm with the user — git restore . irreversibly discards all working-tree changes.'
    fi
  fi

  # History rewrites — permanently alter commit history; extremely destructive
  if [[ ${padded} =~ git[[:space:]]+filter-branch[[:space:]] ]]; then
    _block 'git filter-branch' \
      'Destructive history rewrite — confirm with the user before proceeding.'
  fi

  if [[ ${padded} =~ git[[:space:]]+filter-repo[[:space:]] ]]; then
    _block 'git filter-repo' \
      'Destructive history rewrite — confirm with the user before proceeding.'
  fi

  # Recovery ref destruction — eliminates the ability to recover lost commits
  if [[ ${padded} =~ git[[:space:]]+reflog[[:space:]]+expire[[:space:]] ]]; then
    _block 'git reflog expire' \
      'This destroys reflog recovery references — confirm with the user.'
  fi

  if [[ ${padded} =~ git[[:space:]]+gc[[:space:]] ]] && [[ ${padded} =~ [[:space:]]--prune=now[[:space:]] ]]; then
    _block 'git gc --prune=now' \
      'This destroys unreachable objects needed for recovery — confirm with the user.'
  fi

  if [[ ${padded} =~ git[[:space:]]+update-ref[[:space:]] ]] \
     && [[ ${padded} =~ [[:space:]]-[A-Za-z]*d[A-Za-z]*[[:space:]] ]]; then
    _block 'git update-ref -d' \
      'Destructive ref operation — confirm with the user.'
  fi

  # .git directory destruction
  # Catches: rm -rf .git  rm -r -f .git  rm -rf .git/  rm -rf /path/to/.git
  # Allows:  rm -rf .gitignore  rm -rf .github  (character after .git is alphanumeric)
  if [[ ${padded} =~ [[:space:]]rm[[:space:]] ]] \
     && [[ ${padded} =~ [[:space:]]-[A-Za-z]*r[A-Za-z]*[[:space:]] ]] \
     && [[ ${padded} =~ ${_force} ]] \
     && [[ ${padded} =~ [.]git(/|[[:space:]]) ]]; then
    _block 'rm -rf .git' \
      'This would destroy the git repository — confirm with the user first.'
  fi
}

while IFS= read -r _invocation; do
  _check_invocation "${_invocation}"
done <<< "${COMMAND_SEGMENTED}"

exit 0
