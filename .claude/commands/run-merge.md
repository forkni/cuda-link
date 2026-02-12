# Run Merge Workflow

Guided workflow for merging development branch into main using the
project's automated merge scripts with .gitattributes support.

## Overview

This command guides you through the safe merge process:

1. Validates both branches are ready
2. Runs pre-merge checks
3. Executes merge with .gitattributes exclusions
4. Verifies merge success
5. Provides rollback options if needed

## Prerequisites

- Clean working directory (no uncommitted changes)
- Both main and development branches exist
- .gitattributes file configured
- Git merge drivers configured

## Merge Types

**Arguments:**

- `full` - Full merge from development to main (default)
- `docs` - Documentation-only merge
- `status` - Check branch synchronization status

## Command

```bash
MERGE_TYPE="${ARGUMENTS:-full}"

echo "=== Git Workflow Merge Assistant ==="
echo ""

# Step 1: Check sync status
echo "[1/5] Checking branch synchronization status..."
./scripts/git/sync_status.bat

echo ""
read -p "Continue with merge? (yes/no): " CONTINUE
if [ "$CONTINUE" != "yes" ]; then
  echo "Merge cancelled"
  exit 0
fi

# Step 2: Run merge based on type
case "$MERGE_TYPE" in
  "full")
    echo ""
    echo "[2/5] Running full merge: development → main"
    echo "This will merge all changes except tests/ and development-only docs"
    echo ""
    ./scripts/git/merge_with_validation.bat
    ;;

  "docs")
    echo ""
    echo "[2/5] Running documentation-only merge"
    echo "This will merge ONLY changes in docs/ directory"
    echo ""
    ./scripts/git/merge_docs.bat
    ;;

  "status")
    echo ""
    echo "Status check complete. No merge performed."
    exit 0
    ;;

  *)
    echo "❌ Invalid merge type: $MERGE_TYPE"
    echo "Valid types: full, docs, status"
    exit 1
    ;;
esac

# Step 3: Verify merge result
if [ $? -eq 0 ]; then
  echo ""
  echo "[3/5] Merge completed successfully!"
  echo ""
  echo "Latest commit:"
  git log -1 --oneline
  echo ""

  # Step 4: Check for excluded files
  echo "[4/5] Verifying exclusions..."
  if [ -d "tests" ]; then
    echo "⚠️  WARNING: tests/ directory found on main branch"
    echo "This should not happen - .gitattributes may not be working"
  else
    echo "✅ tests/ correctly excluded from main branch"
  fi

  if [ -f "pytest.ini" ]; then
    echo "⚠️  WARNING: pytest.ini found on main branch"
  else
    echo "✅ pytest.ini correctly excluded from main branch"
  fi

  # Step 5: Push instructions
  echo ""
  echo "[5/5] Next steps:"
  echo ""
  echo "✅ Merge successful - review changes above"
  echo ""
  echo "To push to remote:"
  echo "  git push origin main"
  echo ""
  echo "To rollback if needed:"
  echo "  ./scripts/git/rollback_merge.bat"
  echo ""
else
  echo ""
  echo "❌ Merge failed or was aborted"
  echo ""
  echo "Check output above for details"
  echo "To abort merge:"
  echo "  git merge --abort"
  echo ""
fi
```

## Usage Examples

**Full Merge:**

```bash
# Run full merge workflow
/run-merge full

# Or just use default
/run-merge
```

**Documentation Only:**

```bash
# Merge only docs/ changes
/run-merge docs
```

**Status Check Only:**

```bash
# Just check synchronization status
/run-merge status
```

## Safety Features

- ✅ Pre-merge validation via `validate_branches.bat`
- ✅ Automatic backup tags created
- ✅ .gitattributes enforcement (tests/ excluded)
- ✅ Post-merge verification
- ✅ Rollback script available
- ✅ Clear next-step instructions

## Rollback

If merge has issues:

```bash
./scripts/git/rollback_merge.bat
```

This will guide you through rolling back to the pre-merge state using backup tags.
