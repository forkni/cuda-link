#!/bin/bash
# file-edit-guard.sh
# PreToolUse hook for Edit|Write|MultiEdit - Detects potential file modification conflicts
# Warns when editing files that might be auto-saved by external programs

# ============================================================================
# INPUT PARSING
# ============================================================================

# Read stdin (JSON from Claude Code PreToolUse hook)
input=$(cat)

# Extract file path and tool name using Python
file_info=$(echo "$input" | python -c "
import sys
import json
try:
    data = json.load(sys.stdin)
    tool_input = data.get('tool_input', {})

    file_path = tool_input.get('file_path', '')
    if file_path:
        # Normalize path separators
        file_path = file_path.replace('\\\\\\\\', '/').replace('\\\\', '/')
        print(file_path)
except:
    pass
")

# Exit silently if no file path extracted
if [ -z "$file_info" ]; then
    exit 0
fi

# ============================================================================
# SECURITY VALIDATION
# ============================================================================

file_path="$file_info"

# Check for path traversal attempts (../ or ..\)
if echo "$file_path" | grep -qE '(\.\./|\.\.\\)'; then
    {
        echo ""
        echo "╔═══════════════════════════════════════════════════════╗"
        echo "║  🚫 SECURITY: PATH TRAVERSAL DETECTED                ║"
        echo "║  File edit operation BLOCKED                         ║"
        echo "╚═══════════════════════════════════════════════════════╝"
        echo ""
        echo "ATTEMPTED PATH: $file_path"
        echo ""
        echo "⚠️  Path contains traversal sequence (../ or ..\\)"
        echo "   This could allow access to files outside project directory"
        echo ""
        echo "🛡️  SECURITY POLICY: Path traversal blocked for safety"
        echo ""
    } >&2
    exit 2
fi

# ============================================================================
# RISK DETECTION
# ============================================================================

risks_detected=0
warnings=()

# Check 1: TouchDesigner DAT export files
if echo "$file_path" | grep -qE '(streamdiffusionTD__Text__|__td\.py)'; then
    risks_detected=1
    warnings+=("⚠️  TouchDesigner DAT export detected - May be auto-saved by TD while editing")
fi

# Check 2: Files in Scripts directory (likely TD exports)
if echo "$file_path" | grep -qiE 'Scripts/StreamDiffusionTD|Scripts\\\\StreamDiffusionTD'; then
    risks_detected=1
    warnings+=("📁 File in TouchDesigner Scripts directory - TD may overwrite on DAT export")
fi

# Check 3: Cloud-synced directories (OneDrive, Dropbox, Google Drive)
if echo "$file_path" | grep -qiE 'OneDrive|Dropbox|Google Drive|iCloud'; then
    risks_detected=1
    warnings+=("☁️  Cloud-synced directory detected - File may be modified during sync")
fi

# Check 4: Check if file was recently modified (within last 5 seconds)
if [[ -f "$file_path" ]]; then
    # Get file modification time
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows (Git Bash)
        file_mtime=$(stat -c %Y "$file_path" 2>/dev/null || echo "0")
    else
        # Unix/Mac
        file_mtime=$(stat -f %m "$file_path" 2>/dev/null || stat -c %Y "$file_path" 2>/dev/null || echo "0")
    fi

    current_time=$(date +%s)
    time_diff=$((current_time - file_mtime))

    if [ "$time_diff" -lt 5 ]; then
        risks_detected=1
        warnings+=("🕐 File modified ${time_diff} seconds ago - External program may have updated it")
    fi
fi

# ============================================================================
# OUTPUT WARNINGS IF RISKS DETECTED
# ============================================================================

if [ $risks_detected -eq 1 ]; then
    echo ""
    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║  ⚠️  FILE EDIT CONFLICT RISK DETECTED                 ║"
    echo "║  Proceeding with caution (external modification risk) ║"
    echo "╚═══════════════════════════════════════════════════════╝"
    echo ""
    echo "FILE: $file_path"
    echo ""
    echo "POTENTIAL RISKS:"
    for warning in "${warnings[@]}"; do
        echo "   $warning"
    done
    echo ""
    echo "🔄 COMMON ERROR: \"File has been unexpectedly modified\""
    echo ""
    echo "💡 PREVENTION STRATEGIES:"
    echo "   1. Close TouchDesigner before editing DAT source files"
    echo "   2. Pause cloud sync temporarily during editing session"
    echo "   3. Close VS Code if file is open there"
    echo "   4. Use bash heredoc for large edits: cat >> file << 'EOF'"
    echo ""
    echo "🔧 IF EDIT FAILS:"
    echo "   1. Wait 500ms and retry once"
    echo "   2. If still fails → Use bash heredoc workaround:"
    echo "      cat >> \"$file_path\" << 'EOF'"
    echo "      [content]"
    echo "      EOF"
    echo ""
    echo "📖 REFERENCE: See CLAUDE.md for detailed retry logic"
    echo ""
    echo "╚═══════════════════════════════════════════════════════╝"
    echo ""
fi

# Always exit 0 (warnings only, don't block)
exit 0
