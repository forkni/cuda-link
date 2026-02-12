#!/bin/bash
# skill-activation-prompt.sh - REVISION 3
# Detects MCP and code exploration keywords in user prompts
# Outputs JSON to inject skill recommendations via hookSpecificOutput
# Target execution: <2s (handles multiline JSON correctly)

# Read ALL stdin lines with timeout (prevents hanging on multiline JSON)
input=""
while IFS= read -t 1 -r line; do
    input+="$line"
done

# Quick exit if no input
[[ -z "$input" ]] && exit 0

# Remove newlines (Claude Code sends multiline JSON)
input="${input//$'\n'/}"
input="${input//$'\r'/}"

# Extract prompt - handle optional space after colon
# Matches both: "prompt":"value" and "prompt": "value"
prompt="${input#*\"prompt\"*:*\"}"
prompt="${prompt%%\"*}"

# Quick exit if no prompt extracted
[[ -z "$prompt" || "$prompt" == "$input" ]] && exit 0

# Convert to lowercase using bash (no subprocess)
prompt_lower="${prompt,,}"

# Check for explicit MCP request (TIER 2 - MANDATORY)
if [[ "$prompt_lower" == *"mcp"* ]]; then
    cat <<'EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "🔴 USER EXPLICITLY REQUESTED MCP SEARCH - MANDATORY. REQUIRED: 1) Invoke Skill tool with 'mcp-search-tool' 2) Wait for results 3) DO NOT read files without MCP search first. Benefits: 40-45% token savings, 5-10x faster discovery."
  }
}
EOF
    exit 0
fi

# Check for code exploration keywords (TIER 1 - RECOMMENDED)
case "$prompt_lower" in
    *find*code*|*find*function*|*find*class*|*find*file*|*search*code*|*search*function*|*search*class*|*locate*|*where*is*|*show*me*|*look*for*|*research*code*|*explore*|*analyze*code*|*investigate*)
        cat <<'EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "🎯 SKILL RECOMMENDATION: Code exploration detected. Consider using `mcp-search-tool` skill BEFORE reading files. Benefits: 40-45% token savings, 5-10x faster discovery."
  }
}
EOF
        exit 0
        ;;
esac

# No match - silent exit
exit 0
