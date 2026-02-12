#!/bin/bash
# enable-strict-mode.sh
# Toggle strict mode for bash-pre-validator.sh
# Strict mode: BLOCKS execution on critical bash errors
# Warning mode: Allows execution with warnings only

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get project root
if [ -n "$CLAUDE_PROJECT_DIR" ]; then
    project_root="$CLAUDE_PROJECT_DIR"
else
    # Derive from current script location
    script_dir=$(dirname "$0")
    project_root=$(cd "$script_dir/../.." && pwd)
fi

hooks_dir="$project_root/.claude/hooks"
validator_script="$hooks_dir/bash-pre-validator.sh"
backup_script="$hooks_dir/bash-pre-validator.sh.backup"

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

mode="$1"

if [ -z "$mode" ]; then
    # No argument provided, show current mode and usage
    if [ ! -f "$validator_script" ]; then
        echo "❌ Error: bash-pre-validator.sh not found"
        exit 1
    fi

    current_mode=$(grep "^STRICT_MODE=" "$validator_script" | head -n 1)

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔧 BASH PRE-VALIDATOR - STRICT MODE CONFIGURATION"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Current setting: $current_mode"
    echo ""

    if [[ "$current_mode" == *"true"* ]]; then
        echo "✅ STRICT MODE: ENABLED"
        echo ""
        echo "   Behavior:"
        echo "   • Critical bash errors BLOCK execution"
        echo "   • Windows CMD commands are rejected"
        echo "   • Unsafe patterns (cd without exit, eval, etc.) are blocked"
        echo "   • Warnings still allow execution"
        echo ""
        echo "   To disable:"
        echo "   bash .claude/hooks/enable-strict-mode.sh off"
    else
        echo "⚠️  STRICT MODE: DISABLED"
        echo ""
        echo "   Behavior:"
        echo "   • All commands execute with warnings only"
        echo "   • Critical errors are flagged but NOT blocked"
        echo "   • Less safe, but more permissive"
        echo ""
        echo "   To enable:"
        echo "   bash .claude/hooks/enable-strict-mode.sh on"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Usage:"
    echo "  bash enable-strict-mode.sh           # Show current mode"
    echo "  bash enable-strict-mode.sh on        # Enable strict mode (block errors)"
    echo "  bash enable-strict-mode.sh off       # Disable strict mode (warn only)"
    echo ""

    exit 0
fi

# ============================================================================
# MODE VALIDATION
# ============================================================================

if [ "$mode" != "on" ] && [ "$mode" != "off" ]; then
    echo "❌ Error: Invalid argument '$mode'"
    echo ""
    echo "Usage:"
    echo "  bash enable-strict-mode.sh on        # Enable strict mode"
    echo "  bash enable-strict-mode.sh off       # Disable strict mode"
    echo ""
    exit 1
fi

# ============================================================================
# FILE VALIDATION
# ============================================================================

if [ ! -f "$validator_script" ]; then
    echo "❌ Error: bash-pre-validator.sh not found at:"
    echo "   $validator_script"
    echo ""
    echo "   Make sure you're running this from the project root."
    echo ""
    exit 1
fi

# ============================================================================
# CHECK CURRENT MODE
# ============================================================================

current_mode=$(grep "^STRICT_MODE=" "$validator_script" | head -n 1)

if [ "$mode" = "on" ] && [[ "$current_mode" == *"true"* ]]; then
    echo ""
    echo "ℹ️  Strict mode is already ENABLED."
    echo ""
    echo "   Current setting: STRICT_MODE=true"
    echo ""
    echo "   Critical bash errors are already blocked."
    echo ""
    exit 0
fi

if [ "$mode" = "off" ] && [[ "$current_mode" == *"false"* ]]; then
    echo ""
    echo "ℹ️  Strict mode is already DISABLED."
    echo ""
    echo "   Current setting: STRICT_MODE=false"
    echo ""
    echo "   Commands execute with warnings only."
    echo ""
    exit 0
fi

# ============================================================================
# CONFIRMATION PROMPT
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 BASH PRE-VALIDATOR - MODE CHANGE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$mode" = "on" ]; then
    echo "⚠️  Enabling STRICT MODE"
    echo ""
    echo "   This will BLOCK bash commands with critical errors:"
    echo "   • Windows CMD commands (copy, dir, cd /d, etc.)"
    echo "   • cd without error handling (|| exit)"
    echo "   • eval usage (code injection risk)"
    echo "   • Unquoted Windows paths with spaces"
    echo "   • Backslash path separators"
    echo ""
    echo "   Warnings (non-critical) will still allow execution."
    echo ""
else
    echo "⚠️  Disabling STRICT MODE"
    echo ""
    echo "   This will allow ALL bash commands to execute:"
    echo "   • Critical errors will show warnings but NOT block"
    echo "   • Windows CMD commands will be flagged but executed"
    echo "   • Unsafe patterns will be warned but not prevented"
    echo ""
    echo "   Less safe, but more permissive."
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$mode" = "on" ]; then
    read -p "Enable strict mode (block critical errors)? (yes/no): " confirm
else
    read -p "Disable strict mode (warn only, allow all)? (yes/no): " confirm
fi

if [ "$confirm" != "yes" ]; then
    echo ""
    echo "❌ Aborted. Mode unchanged."
    echo ""
    exit 0
fi

# ============================================================================
# BACKUP & MODIFY
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 UPDATING BASH PRE-VALIDATOR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create backup
echo "Creating backup..."
cp "$validator_script" "$backup_script"

if [ $? -ne 0 ]; then
    echo "❌ Failed to create backup. Aborting."
    echo ""
    exit 1
fi

echo "✅ Backup created: bash-pre-validator.sh.backup"
echo ""

# Modify the script
if [ "$mode" = "on" ]; then
    echo "Enabling strict mode (STRICT_MODE=true)..."
    sed -i 's/^STRICT_MODE=false/STRICT_MODE=true/' "$validator_script"
    sed -i 's/^STRICT_MODE=.*/STRICT_MODE=true/' "$validator_script"
else
    echo "Disabling strict mode (STRICT_MODE=false)..."
    sed -i 's/^STRICT_MODE=true/STRICT_MODE=false/' "$validator_script"
    sed -i 's/^STRICT_MODE=.*/STRICT_MODE=false/' "$validator_script"
fi

if [ $? -ne 0 ]; then
    echo "❌ Failed to update script. Restoring backup..."
    cp "$backup_script" "$validator_script"
    echo ""
    exit 1
fi

echo "✅ Script updated successfully"
echo ""

# ============================================================================
# VERIFICATION
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$mode" = "on" ]; then
    echo "✅ STRICT MODE ENABLED"
else
    echo "✅ STRICT MODE DISABLED"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🔍 Verification:"
new_mode=$(grep "^STRICT_MODE=" "$validator_script" | head -n 1)
echo "   $new_mode"
echo ""

if [ "$mode" = "on" ]; then
    echo "📊 Behavior:"
    echo "   • Critical bash errors will BLOCK execution"
    echo "   • Windows CMD commands will be rejected"
    echo "   • Unsafe patterns will be prevented"
    echo "   • Warnings will still allow execution"
    echo ""
    echo "🔧 To disable:"
    echo "   bash .claude/hooks/enable-strict-mode.sh off"
else
    echo "📊 Behavior:"
    echo "   • All commands execute with warnings only"
    echo "   • Critical errors flagged but NOT blocked"
    echo "   • Less safe, more permissive"
    echo ""
    echo "🔧 To enable:"
    echo "   bash .claude/hooks/enable-strict-mode.sh on"
fi

echo ""

echo "📖 To restore backup:"
echo "   cp .claude/hooks/bash-pre-validator.sh.backup \\"
echo "      .claude/hooks/bash-pre-validator.sh"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exit 0
