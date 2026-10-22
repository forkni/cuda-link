# Claude Code hooks

This project registers one hook, in `.claude/settings.json`:

| Hook | Event | Script | Purpose |
| --- | --- | --- | --- |
| Commit enforcer | PreToolUse (Bash) | `git-commit-enforcer.py` | Redirects raw `git commit` to `scripts/git/commit_enhanced.sh`, so lint validation, local-file exclusion and message checks are never bypassed. |

Bash-safety, skill-activation and session hooks are not kept here. They run from the global
`~/.claude/` configuration and apply to every project. The project copies that used to sit in
this directory were never referenced by `settings.json` and were removed.
