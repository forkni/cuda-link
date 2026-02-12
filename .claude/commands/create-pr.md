# Create Pull Request

Create a pull request with a concise description of changes. This command will:

1. Create a new branch from current changes
2. Commit the staged changes
3. Push to remote
4. Create a PR using GitHub CLI

## Important Guidelines

- **DO NOT** mention Claude Code or AI assistance in the PR description
- **DO NOT** include testing plans in the PR description (keep it concise)
- Use clean, professional commit messages per project guidelines
- Follow conventional commit format (feat:, fix:, docs:, chore:, etc.)

## Usage

Run this command after making and staging changes to create a PR automatically.

## Arguments

`$ARGUMENTS` - Optional: Specify branch name suffix. If not provided, uses timestamp.

## Command

```bash
# Create descriptive branch name
BRANCH_SUFFIX="${ARGUMENTS:-$(date +%Y%m%d-%H%M%S)}"
git checkout -b "feature-$BRANCH_SUFFIX"

# Verify changes are staged
if [ -z "$(git diff --cached)" ]; then
  echo "❌ No changes staged. Please stage your changes first:"
  echo "   git add <files>"
  exit 1
fi

# Show staged changes
echo "Staged changes:"
git diff --cached --name-status
echo ""

# Commit changes
read -p "Enter commit message: " COMMIT_MSG
git commit -m "$COMMIT_MSG"

# Push to remote
git push -u origin HEAD

# Create PR with gh CLI
read -p "Enter PR title: " PR_TITLE
read -p "Enter PR description (one-line summary): " PR_DESC

gh pr create \
  --title "$PR_TITLE" \
  --body "## Summary

$PR_DESC

## Changes
$(git diff --name-status origin/development..HEAD)
"
```

## Examples

**Good PR Title Examples:**

- `feat: Add semantic search caching for 93% token reduction`
- `fix: Escape parentheses in sync_status.bat echo statements`
- `docs: Update installation guide with PyTorch 2.6.0 requirements`
- `chore: Add GitHub Actions workflows for CI/CD`

**Good PR Description Examples:**

- `Implements hybrid search with BM25 + semantic fusion for improved accuracy`
- `Fixes batch file variable scoping issue in sync_status.bat`
- `Updates documentation to reflect new BGE-M3 model support`
