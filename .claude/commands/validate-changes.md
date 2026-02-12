# Validate Changes

Pre-commit validation checklist to ensure changes follow project standards and won't break the build.

## Overview

This command runs comprehensive checks before committing:

1. Verifies no local-only files are staged
2. Checks branch-specific requirements
3. Validates commit message format
4. Runs quick tests if applicable
5. Checks .gitattributes compliance

## Quick Validation

```bash
echo "=== Pre-Commit Validation Checklist ==="
echo ""

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 Current branch: $CURRENT_BRANCH"
echo ""

# Check 1: Uncommitted changes
echo "[1/7] Checking for staged changes..."
if [ -z "$(git diff --cached)" ]; then
  echo "⚠️  No changes staged"
  echo "   Run: git add <files>"
  echo ""
  exit 0
else
  echo "✅ Changes staged:"
  git diff --cached --name-status
  echo ""
fi

# Check 2: Local-only files
echo "[2/7] Checking for local-only files..."
BLOCKED_FILES=""

for file in CLAUDE.md MEMORY.md; do
  if git diff --cached --name-only | grep -q "^$file$"; then
    echo "❌ ERROR: $file is staged (should be local-only)"
    BLOCKED_FILES="yes"
  fi
done

if git diff --cached --name-only | grep -q "^_archive/"; then
  echo "❌ ERROR: _archive/ files are staged (should be local-only)"
  BLOCKED_FILES="yes"
fi

if git diff --cached --name-only | grep -q "^benchmark_results/"; then
  echo "❌ ERROR: benchmark_results/ files are staged (should be local-only)"
  BLOCKED_FILES="yes"
fi

if [ -n "$BLOCKED_FILES" ]; then
  echo ""
  echo "Please unstage these files:"
  echo "  git reset HEAD <file>"
  exit 1
fi

echo "✅ No local-only files staged"
echo ""

# Check 3: Branch-specific validations
echo "[3/7] Running branch-specific checks..."

if [ "$CURRENT_BRANCH" = "main" ]; then
  echo "Checking main branch requirements..."

  # Check for test files
  if git diff --cached --name-only | grep -q "^tests/"; then
    echo "❌ ERROR: Test files staged on main branch"
    echo "   Tests should only be on development branch"
    exit 1
  fi

  # Check for pytest.ini
  if git diff --cached --name-only | grep -q "pytest.ini"; then
    echo "❌ ERROR: pytest.ini staged on main branch"
    exit 1
  fi

  # Check for development-only docs
  DEV_DOCS="GIT_WORKFLOW.md|TESTING_GUIDE.md|GPU_MEMORY_LEAK_FIX.md|PER_MODEL_INDICES_IMPLEMENTATION.md|GIT_WORKFLOW_CRITICAL_REVIEW.md|GIT_WORKFLOW_ENHANCEMENT_PLAN.md"
  if git diff --cached --name-only | grep -E "docs/($DEV_DOCS)"; then
    echo "❌ ERROR: Development-only docs staged on main branch"
    echo "   These docs should remain on development branch only"
    exit 1
  fi

  echo "✅ No development-only files on main"
else
  echo "✅ Development branch - no restrictions"
fi
echo ""

# Check 4: File count
FILE_COUNT=$(git diff --cached --name-only | wc -l)
echo "[4/7] Files to commit: $FILE_COUNT"
if [ $FILE_COUNT -gt 50 ]; then
  echo "⚠️  Warning: Large changeset ($FILE_COUNT files)"
  echo "   Consider breaking into smaller commits"
fi
echo ""

# Check 5: Commit message preview
echo "[5/7] Commit message format check..."
read -p "Enter proposed commit message: " PROPOSED_MSG

# Check conventional commit format
if echo "$PROPOSED_MSG" | grep -qE "^(feat|fix|docs|chore|test|refactor|style|perf):"; then
  echo "✅ Follows conventional commit format"
else
  echo "⚠️  Does not follow conventional commit format"
  echo "   Recommended prefixes: feat:, fix:, docs:, chore:, test:"
  echo ""
  read -p "Continue anyway? (yes/no): " CONTINUE
  if [ "$CONTINUE" != "yes" ]; then
    echo "Validation cancelled"
    exit 0
  fi
fi
echo ""

# Check 6: Clean commit message
if echo "$PROPOSED_MSG" | grep -qi "claude\|generated\|co-authored"; then
  echo "❌ ERROR: Commit message contains AI attribution"
  echo "   Per project guidelines, commit messages should be clean"
  echo "   Remove references to: Claude, Generated, Co-Authored"
  exit 1
fi
echo "✅ Clean commit message (no AI attribution)"
echo ""

# Check 7: .gitattributes consistency
echo "[7/7] Checking .gitattributes..."
if [ -f ".gitattributes" ]; then
  echo "✅ .gitattributes exists"
else
  echo "⚠️  .gitattributes not found"
fi
echo ""

# Summary
echo "======================================"
echo "✅ PRE-COMMIT VALIDATION PASSED"
echo "======================================"
echo ""
echo "Ready to commit with:"
echo "  Message: $PROPOSED_MSG"
echo "  Files: $FILE_COUNT"
echo "  Branch: $CURRENT_BRANCH"
echo ""
echo "Next steps:"
echo "  git commit -m \"$PROPOSED_MSG\""
echo ""
echo "Or use enhanced commit:"
echo "  ./scripts/git/commit_enhanced.bat \"$PROPOSED_MSG\""
echo ""
```

## Usage

**Basic validation:**

```bash
/validate-changes
```

## Common Issues Caught

This validation prevents:

- ❌ Committing CLAUDE.md or MEMORY.md (local-only)
- ❌ Committing _archive/ or benchmark_results/ (local-only)
- ❌ Adding tests/ to main branch (development-only)
- ❌ Adding pytest.ini to main branch (development-only)
- ❌ Using AI attribution in commit messages
- ❌ Non-conventional commit message format
- ⚠️  Large changesets (warns if >50 files)

## Integration with Scripts

This command complements:

- `scripts/git/commit_enhanced.bat` - Enhanced commit workflow
- `scripts/git/validate_branches.bat` - Branch state validation
- `.git/hooks/pre-commit` - Automatic pre-commit hook

## Quick Fixes

**Unstage local files:**

```bash
git reset HEAD CLAUDE.md MEMORY.md
git reset HEAD _archive/
git reset HEAD benchmark_results/
```

**Fix commit message format:**

```bash
# Good examples:
feat: Add hybrid search with BM25 + semantic fusion
fix: Escape parentheses in batch file echo statements
docs: Update installation guide with PyTorch requirements
chore: Add GitHub Actions workflows for CI/CD
test: Add integration tests for incremental indexing
```

## Exit Codes

- `0` - Validation passed or no changes
- `1` - Validation failed (blocked files, format issues, etc.)
