#!/bin/bash
# session-start-skill-reminder.sh
# Auto-loads mcp-search-tool skill context at SessionStart
# Target execution: <100ms (simple JSON output)

cat <<'EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "🔴 MANDATORY: Before ANY semantic code search, you MUST invoke the `mcp-search-tool` skill using `Skill(\"mcp-search-tool\")`. The skill contains the required 5-step workflow: (1) list_projects, (2) project validation, (3) switch_project if needed, (4) get_index_status, (5) optimized query construction with filters. Do NOT use MCP search tools directly without invoking the skill first."
  }
}
EOF
