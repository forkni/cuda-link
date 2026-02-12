#!/bin/bash
# Shell Script Lint Check using ShellCheck

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Shell Script Lint Check ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Find all .sh files in scripts/
mapfile -t SHELL_FILES < <(find "$PROJECT_ROOT/scripts" -name "*.sh" -type f 2>/dev/null)

if [ ${#SHELL_FILES[@]} -eq 0 ]; then
    echo "No shell scripts found in scripts/"
    exit 0
fi

echo "Found ${#SHELL_FILES[@]} shell scripts to check"
echo ""

# Detect shellcheck location (project-local or system)
SHELLCHECK=""
if [ -x "$PROJECT_ROOT/tools/bin/shellcheck.exe" ]; then
    SHELLCHECK="$PROJECT_ROOT/tools/bin/shellcheck.exe"
    echo "Using project-local shellcheck: $SHELLCHECK"
elif [ -x "$PROJECT_ROOT/tools/bin/shellcheck" ]; then
    SHELLCHECK="$PROJECT_ROOT/tools/bin/shellcheck"
    echo "Using project-local shellcheck: $SHELLCHECK"
elif command -v shellcheck >/dev/null 2>&1; then
    SHELLCHECK="shellcheck"
    echo "Using system shellcheck: $(command -v shellcheck)"
else
    echo "WARNING: ShellCheck not installed"
    echo ""
    echo "Install instructions:"
    echo "  Project:  Already in tools/bin/ - check permissions"
    echo "  Windows:  scoop install shellcheck"
    echo "  Linux:    apt-get install shellcheck"
    echo "  macOS:    brew install shellcheck"
    echo ""
    echo "Skipping shell check..."
    exit 0
fi

echo ""

# Run ShellCheck
echo "Running ShellCheck (severity: warning)..."
echo ""

EXIT_CODE=0

for file in "${SHELL_FILES[@]}"; do
    echo "Checking: $file"
    # SC2154: Variable referenced but not assigned (false positive for sourced variables from _common.sh)
    if ! "$SHELLCHECK" --severity=warning --exclude=SC2154 --format=gcc "$file"; then
        EXIT_CODE=1
    fi
done

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "[OK] All shell scripts passed ShellCheck"
else
    echo "[FAIL] ShellCheck found issues"
    echo ""
    echo "To disable specific warnings, add inline comments:"
    echo '  # shellcheck disable=SC2034'
fi

exit $EXIT_CODE
