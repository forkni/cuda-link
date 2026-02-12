#!/bin/bash
# bash-error-tracker.sh
# Tracks bash command failures (PostToolUse hook)
# Increments error count when Bash tool returns errors

# Read stdin (JSON from Claude Code)
input=$(cat)

# Extract tool info using Python
tool_info=$(echo "$input" | python -c "
import sys
import json
try:
    data = json.load(sys.stdin)
    tool_name = data.get('tool_name', '')
    tool_result = data.get('tool_result', {})
    session_id = data.get('session_id', '')

    # Check if it's a Bash tool with error
    if tool_name == 'Bash':
        result_type = tool_result.get('type', '')
        if result_type == 'error':
            print(f'{session_id}|error')
        elif 'error' in str(tool_result).lower():
            print(f'{session_id}|error')
        else:
            print(f'{session_id}|success')
    else:
        print('')
except:
    print('')
")

# If we couldn't parse tool info, exit silently
if [ -z "$tool_info" ]; then
    exit 0
fi

# Parse the result
session_id=$(echo "$tool_info" | cut -d'|' -f1)
result=$(echo "$tool_info" | cut -d'|' -f2)

if [ -z "$session_id" ]; then
    exit 0
fi

# Create cache directory
cache_dir="$CLAUDE_PROJECT_DIR/.claude/.cache"
mkdir -p "$cache_dir"

error_count_file="$cache_dir/${session_id}_bash_errors.txt"

# Initialize error count if file doesn't exist
if [ ! -f "$error_count_file" ]; then
    echo "0" > "$error_count_file"
fi

# Update error count based on result
if [ "$result" = "error" ]; then
    # Increment error count
    current_count=$(cat "$error_count_file" 2>/dev/null || echo "0")
    new_count=$((current_count + 1))
    echo "$new_count" > "$error_count_file"
elif [ "$result" = "success" ]; then
    # Reset error count on successful bash command
    echo "0" > "$error_count_file"
fi

exit 0
