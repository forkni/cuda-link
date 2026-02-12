#!/bin/bash
# bash-safety-reminder.sh
# Reminds about BASH_STYLE_GUIDE.md if bash errors occurred during session
# Runs on Stop hook (when user pauses/stops work)

# Read stdin (JSON from Claude Code)
input=$(cat)

# Extract session_id using Python
session_id=$(echo "$input" | python -c "
import sys
import json
try:
    data = json.load(sys.stdin)
    print(data.get('session_id', ''))
except:
    print('')
")

# If we couldn't get session_id, exit silently
if [ -z "$session_id" ]; then
    exit 0
fi

# Create cache directory for tracking bash errors
cache_dir="$CLAUDE_PROJECT_DIR/.claude/.cache"
mkdir -p "$cache_dir"

error_count_file="$cache_dir/${session_id}_bash_errors.txt"

# Check if error count file exists and has content
if [ -f "$error_count_file" ]; then
    error_count=$(cat "$error_count_file" 2>/dev/null || echo "0")

    # If error count is 2 or more, show reminder
    if [ "$error_count" -ge 2 ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "⚠️  BASH SAFETY REMINDER"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "   Detected: $error_count bash command errors in this session"
        echo ""
        echo "   📖 RECOMMENDATION: Consult \`BASH_STYLE_GUIDE.md\`"
        echo ""
        echo "   Common issues:"
        echo "   • Unquoted paths with spaces: C:\\Program Files → \"C:\\Program Files\""
        echo "   • Missing error handling: cd /path → cd /path || exit"
        echo "   • Windows path issues: backslashes, drive letters"
        echo "   • Unquoted variables: \$var → \"\$var\""
        echo ""
        echo "   Quick fixes:"
        echo "   • Section 4: Word-splitting diagnostics"
        echo "   • Section 5: Error message lookup tables"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        # Reset error count after showing reminder
        echo "0" > "$error_count_file"
    fi
fi

exit 0
