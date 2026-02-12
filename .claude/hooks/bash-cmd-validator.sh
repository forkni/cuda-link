#!/bin/bash
# bash-cmd-validator.sh
# Proactively detects Windows CMD syntax BEFORE bash execution
# Prevents errors by catching Windows commands in Git Bash environment

# Read stdin (JSON from Claude Code)
input=$(cat)

# Extract prompt using Python (UserPromptSubmit provides 'prompt', not 'tool_input')
prompt=$(echo "$input" | python -c "
import sys
import json
try:
    data = json.load(sys.stdin)
    print(data.get('prompt', ''))
except:
    print('')
")

# If we couldn't extract prompt, exit silently
if [ -z "$prompt" ]; then
    exit 0
fi

# Check if prompt contains bash command request
# Look for patterns like "run bash:", "execute:", "bash:", "cmd:"
bash_cmd=""

# Extract bash command from common patterns
if echo "$prompt" | grep -qiE "run bash:"; then
    bash_cmd=$(echo "$prompt" | sed -n 's/.*[Rr]un bash: *\(.*\)/\1/p' | head -1)
fi

if [ -z "$bash_cmd" ] && echo "$prompt" | grep -qiE "execute bash:"; then
    bash_cmd=$(echo "$prompt" | sed -n 's/.*[Ee]xecute bash: *\(.*\)/\1/p' | head -1)
fi

if [ -z "$bash_cmd" ] && echo "$prompt" | grep -qiE "bash:"; then
    bash_cmd=$(echo "$prompt" | sed -n 's/.*[Bb]ash: *\(.*\)/\1/p' | head -1)
fi

# Clean up quotes and backticks
bash_cmd=$(echo "$bash_cmd" | sed 's/["\`]//g')

# If no bash command found in prompt, exit silently
if [ -z "$bash_cmd" ]; then
    exit 0
fi

# Track if we found Windows CMD syntax
found_windows_cmd=0
suggestions=()

# ============================================
# WINDOWS CMD PATTERN DETECTION
# ============================================

# Pattern 1: "copy" command (should be "cp")
if echo "$bash_cmd" | grep -qE "^copy "; then
    found_windows_cmd=1
    suggested_cmd="${bash_cmd/copy /cp }"
    suggestions+=("Windows 'copy' → Git Bash 'cp'|Suggestion: $suggested_cmd")
fi

# Pattern 2: "cd /d" (Git Bash doesn't use /d flag)
if echo "$bash_cmd" | grep -qE "cd /d "; then
    found_windows_cmd=1
    suggested_cmd="${bash_cmd/cd \/d /cd }"
    suggestions+=("Windows 'cd /d' → Git Bash 'cd' (no /d flag)|Suggestion: $suggested_cmd")
fi

# Pattern 3: "dir" command (should be "ls" or "find")
if echo "$bash_cmd" | grep -qE "(^dir | dir /|^dir$)"; then
    found_windows_cmd=1
    # Suggest ls for simple dir, find for recursive
    if echo "$bash_cmd" | grep -qE "dir /s"; then
        suggestions+=("Windows 'dir /s' → Git Bash 'find' or 'ls -R'|Suggestion: find . -name \"*.py\" or ls -R")
    else
        suggestions+=("Windows 'dir' → Git Bash 'ls'|Suggestion: ${bash_cmd/dir/ls}")
    fi
fi

# Pattern 4: "findstr" command (should be "grep")
if echo "$bash_cmd" | grep -qE "findstr"; then
    found_windows_cmd=1
    suggested_cmd="${bash_cmd/findstr/grep}"
    suggestions+=("Windows 'findstr' → Git Bash 'grep'|Suggestion: $suggested_cmd")
fi

# Pattern 5: "xcopy" command (should be "cp -r")
if echo "$bash_cmd" | grep -qE "xcopy"; then
    found_windows_cmd=1
    suggestions+=("Windows 'xcopy' → Git Bash 'cp -r' (recursive copy)|Suggestion: Use 'cp -r source dest'")
fi

# Pattern 6: "move" command (should be "mv")
if echo "$bash_cmd" | grep -qE "^move "; then
    found_windows_cmd=1
    suggested_cmd="${bash_cmd/move /mv }"
    suggestions+=("Windows 'move' → Git Bash 'mv'|Suggestion: $suggested_cmd")
fi

# Pattern 7: "del" command (should be "rm")
if echo "$bash_cmd" | grep -qE "^del "; then
    found_windows_cmd=1
    suggested_cmd="${bash_cmd/del /rm }"
    suggestions+=("Windows 'del' → Git Bash 'rm'|Suggestion: $suggested_cmd")
fi

# Pattern 8: "type" command (should be "cat")
if echo "$bash_cmd" | grep -qE "^type "; then
    found_windows_cmd=1
    suggested_cmd="${bash_cmd/type /cat }"
    suggestions+=("Windows 'type' → Git Bash 'cat'|Suggestion: $suggested_cmd")
fi

# Pattern 9: "cls" command (should be "clear")
if echo "$bash_cmd" | grep -qE "^cls$"; then
    found_windows_cmd=1
    suggestions+=("Windows 'cls' → Git Bash 'clear'|Suggestion: clear")
fi

# ============================================
# BASH SAFETY PATTERNS (from Ysap Style Guide)
# ============================================

# Pattern 10: cd without error handling (|| exit or || return)
if echo "$bash_cmd" | grep -qE "cd " && ! echo "$bash_cmd" | grep -qE "cd .*(&&|\|\|)"; then
    found_windows_cmd=1
    suggestions+=("UNSAFE: 'cd' without error handling|Suggestion: cd path || exit  (or: cd path || return)
|⚠️ cd can fail - always check return code!")
fi

# Pattern 11: Using 'eval' (dangerous - code injection risk)
if echo "$bash_cmd" | grep -qE "eval "; then
    found_windows_cmd=1
    suggestions+=("DANGEROUS: 'eval' detected - code injection risk|Suggestion: Use arrays or proper quoting instead
|⚠️ eval opens code to injection attacks!")
fi

# Pattern 12: Unquoted variables (word-splitting hazard)
if echo "$bash_cmd" | grep -qE '\$[a-zA-Z_][a-zA-Z0-9_]*[^"]' && ! echo "$bash_cmd" | grep -qE '"\$'; then
    found_windows_cmd=1
    suggestions+=("UNSAFE: Unquoted variable expansion detected|Suggestion: Use \"\$var\" instead of \$var
|⚠️ Unquoted variables cause word-splitting with spaces!")
fi

# Pattern 13: Using [ instead of [[ for conditionals
if echo "$bash_cmd" | grep -qE " \[ " && ! echo "$bash_cmd" | grep -qE "\[\["; then
    found_windows_cmd=1
    suggestions+=("Use [[ ]] instead of [ ] for conditionals|Suggestion: [[ condition ]] instead of [ condition ]
|💡 [[ ]] prevents word-splitting and has more features")
fi

# Pattern 14: Useless use of cat
if echo "$bash_cmd" | grep -qE "cat .* \| "; then
    found_windows_cmd=1
    suggestions+=("USELESS CAT: Use redirection instead of cat|Suggestion: command < file  (instead of: cat file | command)
|💡 Redirection is more efficient than piping cat")
fi

# Pattern 15: Using seq for loops
if echo "$bash_cmd" | grep -qE "seq "; then
    found_windows_cmd=1
    suggestions+=("Use bash brace expansion instead of seq|Suggestion: for i in {1..10}  (or: ((i=1; i<=10; i++)))
|💡 Bash builtins are faster than external commands")
fi

# Pattern 16: Parsing ls output
if echo "$bash_cmd" | grep -qE "ls.*\|"; then
    found_windows_cmd=1
    suggestions+=("UNSAFE: Parsing ls output|Suggestion: Use glob patterns: for f in *; do ... done
|⚠️ ls output breaks with spaces/special characters!")
fi

# ============================================
# OUTPUT WARNING IF ISSUES DETECTED
# ============================================

if [ $found_windows_cmd -eq 1 ]; then
    echo ""
    echo "╔═══════════════════════════════════════════╗"
    echo "  ⚠️  BASH SAFETY ISSUES DETECTED           "
    echo "  Command may FAIL or have unsafe patterns   "
    echo "╚═══════════════════════════════════════════╝"
    echo ""
    echo "COMMAND ATTEMPTED: $bash_cmd"
    echo ""
    echo "ENVIRONMENT: Git Bash (POSIX-compliant, not Windows CMD)"
    echo ""
    echo "❌ ISSUES DETECTED:"
    for suggestion in "${suggestions[@]}"; do
        pattern=$(echo "$suggestion" | cut -d'|' -f1)
        fix=$(echo "$suggestion" | cut -d'|' -f2)
        echo "   • $pattern"
        echo "     $fix"
        echo ""
    done
    echo "📖 REFERENCES:"
    echo "   • BASH_STYLE_GUIDE.md - Project-specific bash patterns"
    echo "   • https://style.ysap.sh/ - Official Bash Style Guide"
    echo ""
    echo "💡 TIPS:"
    echo "   • Git Bash uses Unix/Linux commands, not Windows CMD"
    echo "   • Always quote variables: \"\$var\" not \$var"
    echo "   • Check cd return codes: cd path || exit"
    echo "   • Use [[ ]] for conditionals, not [ ]"
    echo ""
fi

# Exit with success (don't block execution, just warn)
# To block execution, use: exit 1
exit 0
