#!/bin/bash
# bash-pre-validator.sh
# PreToolUse hook - Validates Bash commands BEFORE execution
# Prevents Windows CMD syntax and unsafe patterns from reaching Git Bash

# ============================================================================
# CONFIGURATION
# ============================================================================

# STRICT_MODE: Set to true to BLOCK execution on critical errors
# Set to false to warn only (allows execution with warning)
STRICT_MODE=true

# Critical errors (block execution in strict mode):
CRITICAL_ERRORS=(
    "windows_cmd"      # Windows CMD commands (copy, dir, cd /d, etc.)
    "cd_no_exit"       # cd without error handling
    "eval_usage"       # eval command (code injection risk)
    "unquoted_path"    # Windows paths with spaces, unquoted
    "backslash_path"   # Backslash separators in paths
)

# ============================================================================
# INPUT PARSING
# ============================================================================

# Read stdin (JSON from Claude Code PreToolUse hook)
input=$(cat)

# Extract bash command using Python
bash_cmd=$(echo "$input" | python -c "
import sys
import json
try:
    data = json.load(sys.stdin)
    tool_input = data.get('tool_input', {})

    # PreToolUse provides tool_input with command field
    command = tool_input.get('command', '')
    print(command)
except:
    print('')
")

# If no command extracted, exit silently
if [ -z "$bash_cmd" ]; then
    exit 0
fi

# Track detected issues
found_issues=0
critical_issue=0
suggestions=()

# ============================================================================
# PATTERN DETECTION FUNCTIONS
# ============================================================================

mark_critical() {
    local error_type="$1"

    # Check if this error type is in CRITICAL_ERRORS array
    for critical_type in "${CRITICAL_ERRORS[@]}"; do
        if [ "$error_type" = "$critical_type" ]; then
            critical_issue=1
            return 0
        fi
    done
    return 1
}

# ============================================================================
# WINDOWS CMD PATTERN DETECTION
# ============================================================================

# Pattern 1: "copy" command (should be "cp")
if echo "$bash_cmd" | grep -qE "^copy "; then
    found_issues=1
    mark_critical "windows_cmd"
    suggested_cmd="${bash_cmd/copy /cp }"
    suggestions+=("CRITICAL|Windows 'copy' → Git Bash 'cp'|$suggested_cmd")
fi

# Pattern 2: "cd /d" (Git Bash doesn't use /d flag)
if echo "$bash_cmd" | grep -qE "cd /d "; then
    found_issues=1
    mark_critical "windows_cmd"
    suggested_cmd="${bash_cmd/cd \/d /cd }"
    suggestions+=("CRITICAL|Windows 'cd /d' → Git Bash 'cd' (no /d flag)|$suggested_cmd")
fi

# Pattern 3: "dir" command (should be "ls" or "find")
if echo "$bash_cmd" | grep -qE "(^dir | dir /|^dir$)"; then
    found_issues=1
    mark_critical "windows_cmd"
    if echo "$bash_cmd" | grep -qE "dir /s"; then
        suggestions+=("CRITICAL|Windows 'dir /s' → Git Bash 'find' or 'ls -R'|find . -name \"*.py\"  OR  ls -R")
    else
        suggestions+=("CRITICAL|Windows 'dir' → Git Bash 'ls'|${bash_cmd/dir/ls}")
    fi
fi

# Pattern 4: "findstr" command (should be "grep")
if echo "$bash_cmd" | grep -qE "findstr"; then
    found_issues=1
    mark_critical "windows_cmd"
    suggested_cmd="${bash_cmd/findstr/grep}"
    suggestions+=("CRITICAL|Windows 'findstr' → Git Bash 'grep'|$suggested_cmd")
fi

# Pattern 5: "xcopy" command (should be "cp -r")
if echo "$bash_cmd" | grep -qE "xcopy"; then
    found_issues=1
    mark_critical "windows_cmd"
    suggestions+=("CRITICAL|Windows 'xcopy' → Git Bash 'cp -r'|cp -r source dest")
fi

# Pattern 6: "move" command (should be "mv")
if echo "$bash_cmd" | grep -qE "^move "; then
    found_issues=1
    mark_critical "windows_cmd"
    suggested_cmd="${bash_cmd/move /mv }"
    suggestions+=("CRITICAL|Windows 'move' → Git Bash 'mv'|$suggested_cmd")
fi

# Pattern 7: "del" command (should be "rm")
if echo "$bash_cmd" | grep -qE "^del "; then
    found_issues=1
    mark_critical "windows_cmd"
    suggested_cmd="${bash_cmd/del /rm }"
    suggestions+=("CRITICAL|Windows 'del' → Git Bash 'rm'|$suggested_cmd")
fi

# Pattern 8: "type" command (should be "cat")
if echo "$bash_cmd" | grep -qE "^type "; then
    found_issues=1
    mark_critical "windows_cmd"
    suggested_cmd="${bash_cmd/type /cat }"
    suggestions+=("CRITICAL|Windows 'type' → Git Bash 'cat'|$suggested_cmd")
fi

# Pattern 9: "cls" command (should be "clear")
if echo "$bash_cmd" | grep -qE "^cls$"; then
    found_issues=1
    mark_critical "windows_cmd"
    suggestions+=("CRITICAL|Windows 'cls' → Git Bash 'clear'|clear")
fi

# ============================================================================
# WINDOWS PATH PATTERN DETECTION (NEW)
# ============================================================================

# Pattern 10: Unquoted Windows paths with spaces
if echo "$bash_cmd" | grep -qE 'C:\\(Program Files|Users\\[^"]*\\[^"]*\\)' && ! echo "$bash_cmd" | grep -qE '"C:\\'; then
    found_issues=1
    mark_critical "unquoted_path"
    suggestions+=("CRITICAL|Unquoted Windows path with spaces detected|Wrap path in quotes: \"C:\\Program Files\\...\"|⚠️ Paths with spaces MUST be quoted in Git Bash!")
fi

# Pattern 11: Backslash path separators (detects in BOTH quoted AND unquoted paths)
# CRITICAL: Git Bash treats backslashes as escape chars even inside double quotes!
if echo "$bash_cmd" | grep -qE '[A-Z]:'; then
    # Command contains Windows drive letter - check for backslashes
    # Match: \followed-by-letter (path), \at-end, or \"...\...\" (quoted path with backslashes)
    if echo "$bash_cmd" | grep -qE '\\[a-zA-Z/]|\\[[:space:]]|\"[^"]*\\[^"]*\"'; then
        found_issues=1
        mark_critical "backslash_path"
        # Show conversion example (convert backslashes to forward slashes)
        # Use tr to handle all backslash variants
        example_fixed=$(echo "$bash_cmd" | tr '\\' '/')
        suggestions+=("CRITICAL|Windows backslash path separators detected (even in quotes!)|FIXED: $example_fixed|⚠️ CRITICAL: Backslashes are ESCAPE CHARS in bash, even inside quotes! Use forward slashes!")
    fi
fi

# Pattern 12: Mixed path separators (inconsistent style - detects in quoted paths too)
if echo "$bash_cmd" | grep -qE '[A-Z]:'; then
    # Check if both backslash AND forward slash appear (inconsistent separators)
    if echo "$bash_cmd" | grep -qE '\\' && echo "$bash_cmd" | grep -qE '/'; then
        found_issues=1
        suggestions+=("WARNING|Mixed path separators (backslash + forward slash)|Use consistent forward slashes: C:/foo/bar/baz|💡 Stick to one separator style for clarity")
    fi
fi

# Pattern 13: Drive letter without proper format
if echo "$bash_cmd" | grep -qE 'cd [A-Z]:([^/\\]|$)'; then
    found_issues=1
    suggestions+=("WARNING|Drive letter without path separator|Use: cd /d/path  OR  cd D:/path|💡 Git Bash expects Unix-style paths")
fi

# ============================================================================
# BASH SAFETY PATTERNS (from Ysap Style Guide)
# ============================================================================

# Pattern 14: cd without error handling (|| exit or || return)
if echo "$bash_cmd" | grep -qE "cd " && ! echo "$bash_cmd" | grep -qE "cd .*(&&|\|\|)"; then
    found_issues=1
    mark_critical "cd_no_exit"
    suggestions+=("CRITICAL|'cd' without error handling|cd path || exit  OR  cd path || return|⚠️ cd can FAIL - always check return code!")
fi

# Pattern 15: Using 'eval' (dangerous - code injection risk)
if echo "$bash_cmd" | grep -qE "eval "; then
    found_issues=1
    mark_critical "eval_usage"
    suggestions+=("CRITICAL|'eval' detected - CODE INJECTION RISK|Use arrays or proper quoting instead|⚠️ eval opens code to injection attacks!")
fi

# Pattern 16: Unquoted variables (word-splitting hazard)
if echo "$bash_cmd" | grep -qE '\$[a-zA-Z_][a-zA-Z0-9_]*[^"]' && ! echo "$bash_cmd" | grep -qE '"\$'; then
    found_issues=1
    suggestions+=("WARNING|Unquoted variable expansion detected|Use \"\$var\" instead of \$var|⚠️ Unquoted variables cause word-splitting with spaces!")
fi

# Pattern 17: Using [ instead of [[ for conditionals
if echo "$bash_cmd" | grep -qE " \[ " && ! echo "$bash_cmd" | grep -qE "\[\["; then
    found_issues=1
    suggestions+=("WARNING|Use [[ ]] instead of [ ] for conditionals|[[ condition ]] prevents word-splitting|💡 [[ ]] has more features and is safer")
fi

# Pattern 18: Useless use of cat
if echo "$bash_cmd" | grep -qE "cat .* \| "; then
    found_issues=1
    suggestions+=("WARNING|Useless cat - use redirection instead|command < file  instead of  cat file | command|💡 Redirection is more efficient than piping cat")
fi

# Pattern 19: Using seq for loops
if echo "$bash_cmd" | grep -qE "seq "; then
    found_issues=1
    suggestions+=("WARNING|Use bash brace expansion instead of seq|for i in {1..10}  OR  ((i=1; i<=10; i++))|💡 Bash builtins are faster than external commands")
fi

# Pattern 20: Parsing ls output
if echo "$bash_cmd" | grep -qE "ls.*\|"; then
    found_issues=1
    suggestions+=("WARNING|Unsafe: parsing ls output|Use glob patterns: for f in *; do ... done|⚠️ ls output breaks with spaces/special characters!")
fi

# ============================================================================
# OUTPUT WARNING/ERROR IF ISSUES DETECTED
# ============================================================================

if [ $found_issues -eq 1 ]; then
    echo ""

    if [ $critical_issue -eq 1 ] && [ "$STRICT_MODE" = "true" ]; then
        echo "╔═══════════════════════════════════════════════════╗"
        echo "║  🚫 BASH COMMAND BLOCKED - CRITICAL ERRORS        ║"
        echo "║  Command will NOT be executed (strict mode)       ║"
        echo "╚═══════════════════════════════════════════════════╝"
    else
        echo "╔═══════════════════════════════════════════════════╗"
        echo "║  ⚠️  BASH SAFETY ISSUES DETECTED                  ║"
        echo "║  Command will execute with warnings               ║"
        echo "╚═══════════════════════════════════════════════════╝"
    fi

    echo ""
    echo "COMMAND: $bash_cmd"
    echo ""
    echo "ENVIRONMENT: Git Bash (POSIX-compliant, not Windows CMD)"
    echo ""

    # Separate critical and warnings
    critical_count=0
    warning_count=0

    for suggestion in "${suggestions[@]}"; do
        severity=$(echo "$suggestion" | cut -d'|' -f1)
        if [ "$severity" = "CRITICAL" ]; then
            ((critical_count++))
        else
            ((warning_count++))
        fi
    done

    if [ $critical_count -gt 0 ]; then
        echo "🚫 CRITICAL ERRORS ($critical_count):"
        for suggestion in "${suggestions[@]}"; do
            severity=$(echo "$suggestion" | cut -d'|' -f1)
            if [ "$severity" = "CRITICAL" ]; then
                pattern=$(echo "$suggestion" | cut -d'|' -f2)
                fix=$(echo "$suggestion" | cut -d'|' -f3)
                echo "   • $pattern"
                echo "     → $fix"
                echo ""
            fi
        done
    fi

    if [ $warning_count -gt 0 ]; then
        echo "⚠️  WARNINGS ($warning_count):"
        for suggestion in "${suggestions[@]}"; do
            severity=$(echo "$suggestion" | cut -d'|' -f1)
            if [ "$severity" = "WARNING" ]; then
                pattern=$(echo "$suggestion" | cut -d'|' -f2)
                fix=$(echo "$suggestion" | cut -d'|' -f3)
                echo "   • $pattern"
                echo "     → $fix"
                echo ""
            fi
        done
    fi

    echo "📖 REFERENCES:"
    echo "   • BASH_STYLE_GUIDE.md - Project-specific bash patterns"
    echo "   • https://style.ysap.sh/ - Official Bash Style Guide"
    echo ""

    if [ "$STRICT_MODE" = "true" ]; then
        echo "⚙️  STRICT MODE: Enabled"
        echo "   Critical errors BLOCK execution"
        echo "   To disable: Edit bash-pre-validator.sh, set STRICT_MODE=false"
    else
        echo "⚙️  STRICT MODE: Disabled"
        echo "   All commands execute with warnings only"
        echo "   To enable: Edit bash-pre-validator.sh, set STRICT_MODE=true"
    fi
    echo ""

    # Decide whether to block execution
    if [ $critical_issue -eq 1 ] && [ "$STRICT_MODE" = "true" ]; then
        # Output to stderr for blocking (exit 2 sends stderr to Claude)
        {
            echo "╔═══════════════════════════════════════════════════╗"
            echo "║  ❌ EXECUTION BLOCKED                             ║"
            echo "║  Fix critical errors before proceeding            ║"
            echo "╚═══════════════════════════════════════════════════╝"
            echo ""
        } >&2

        # Exit with code 2 to block execution (per official Claude Code docs)
        exit 2
    else
        echo "╔═══════════════════════════════════════════════════╗"
        echo "║  ⚠️  PROCEEDING WITH WARNINGS                     ║"
        echo "║  Command will execute despite issues              ║"
        echo "╚═══════════════════════════════════════════════════╝"
        echo ""

        # Exit with success to allow execution
        exit 0
    fi
fi

# No issues detected, allow execution silently
exit 0
