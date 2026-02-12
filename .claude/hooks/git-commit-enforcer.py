#!/usr/bin/env python3
"""
PreToolUse hook to enforce safe git commits via commit_enhanced.sh
instead of raw 'git commit' commands.

This hook intercepts any 'git commit' bash command and redirects it
to scripts/git/commit_enhanced.sh for validation and safety checks.

Benefits:
- Automatic lint validation (ruff, black, isort)
- Local file exclusion (CLAUDE.md, MEMORY.md, _archive/)
- Commit message format validation
- Branch-specific protection
"""

import json
import os
import re
import sys


def main():
    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})
    command = tool_input.get("command", "")

    # Only intercept Bash tool
    if tool_name != "Bash":
        sys.exit(0)

    # Detect git commit patterns
    # Match: git commit, git commit -m, git commit --message, etc.
    git_commit_pattern = r"\bgit\s+commit\b"
    if not re.search(git_commit_pattern, command):
        sys.exit(0)

    # Respect --no-verify flag - user explicitly wants to bypass hooks
    no_verify_pattern = r"\b(--no-verify|-n)\b"
    if re.search(no_verify_pattern, command):
        # Allow raw command to pass through
        sys.exit(0)

    # SMART DETECTION: Allow legitimate git commit cases to pass through
    # Only intercept standard new commits (git commit -m "message")
    ALLOWED_PATTERNS = [
        r"--amend",  # Amending previous commit
        r"--no-edit",  # Merge/rebase completion
        r"--allow-empty",  # Empty commits (rare but valid)
        r"--fixup",  # Fixup commits for rebase
        r"--squash",  # Squash commits for rebase
    ]

    # Check if any allowed pattern is present
    for pattern in ALLOWED_PATTERNS:
        if re.search(pattern, command):
            # Allow special commit types to pass through
            sys.exit(0)

    # ONLY intercept: git commit -m "message" (standard new commits)
    # This is the pattern that should go through commit_enhanced.bat
    if not re.search(r'-m\s+["\']', command):
        # No -m flag = likely interactive or special case
        sys.exit(0)

    # Extract commit message if present
    # Patterns: -m "message", -m 'message', --message "message"
    message = ""

    # Try -m with quotes
    msg_match = re.search(r'-m\s+["\']([^\'"]+)["\']', command)
    if msg_match:
        message = msg_match.group(1)
    else:
        # Try --message with quotes
        msg_match = re.search(r'--message\s+["\']([^\'"]+)["\']', command)
        if msg_match:
            message = msg_match.group(1)
        else:
            # Try -m without quotes (single word)
            msg_match = re.search(r'-m\s+([^\s"\']+)', command)
            if msg_match:
                message = msg_match.group(1)

    # Build safe commit command
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "F:/RD_PROJECTS/COMPONENTS/claude-context-local")
    safe_script = f"{project_dir}/scripts/git/commit_enhanced.sh"

    # Construct updated command - shell script runs natively in Git Bash
    if message:
        updated_command = f'./scripts/git/commit_enhanced.sh "{message}"'
    else:
        # No message provided - script will prompt or use default
        updated_command = f"./scripts/git/commit_enhanced.sh"

    # Return hook decision with updated command
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": (
                "Routing through safe commit handler (commit_enhanced.sh) "
                "for validation, lint checks, and local file protection"
            ),
            "updatedInput": {"command": updated_command},
        }
    }

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
