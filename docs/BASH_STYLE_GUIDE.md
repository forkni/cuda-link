# Bash Scripting Safety Guide for Claude Code

**Purpose**: Defensive bash patterns for Windows/Git Bash environments, optimized for error prevention in Python development projects.

**Target Environment**: Git Bash on Windows 10/11, Python virtual environments, CUDA/GPU development (when applicable).

**Philosophy**: Safety over elegance. Prevent errors before they occur, provide clear recovery paths when they do.

**Source**: Adapted from [bash-style-guide](https://github.com/bahamas10/bash-style-guide) by Dave Eddy (MIT License)

---

## Table of Contents

1. [Critical Safety Patterns](#1-critical-safety-patterns)
2. [Windows/Git Bash Adaptations](#2-windowsgit-bash-adaptations)
3. [Error Recovery Reference](#3-error-recovery-reference)
4. [Quick Reference Tables](#4-quick-reference-tables)
5. [Integration with Claude Code](#5-integration-with-claude-code)
6. [Summary: Golden Rules](#6-summary-golden-rules)

---

## 1. Critical Safety Patterns

### 1.1 Directory Changes - ALWAYS Check Success

**❌ DANGEROUS**:

```bash
cd /some/path
rm -rf *  # If cd failed, this deletes wrong directory!
```

**✅ SAFE**:

```bash
cd /some/path || exit
rm -rf *  # Only executes if cd succeeded
```

**Rule**: Every `cd` command MUST be followed by `|| exit` or `|| return` (in functions).

**Why**: If `cd` fails silently, subsequent commands execute in the wrong directory, causing data loss or corruption.

**Windows Example**:

```bash
cd "D:\Users\alexk\projects\my-project" || exit
# Now safe to operate on files
```

### 1.2 Variable Quoting - Prevent Word-Splitting

**❌ DANGEROUS**:

```bash
file="my document.txt"
rm $file  # Tries to delete "my" and "document.txt" separately!
```

**✅ SAFE**:

```bash
file="my document.txt"
rm "$file"  # Deletes "my document.txt" as single argument
```

**Rule**: Quote variables containing paths, filenames, or user input: `"$var"` not `$var`.

**Exceptions** (when quoting is optional):

- Controlled boolean values: `enabled=true` then `if $enabled; then`
- Special variables: `$$`, `$?`, `$#` (process ID, exit code, argument count)
- Inside `[[ ]]` conditionals: `[[ $var == "value" ]]` (word-splitting disabled)

**Windows Paths** (MANDATORY quoting):

```bash
# ALWAYS quote Windows paths
path="C:\Users\Inter\AppData\Local\Temp"
cd "$path" || exit

# ALWAYS quote when spaces/special chars possible
venv_python=".\venv\Scripts\python.exe"
"$venv_python" script.py
```

### 1.3 Error Checking - Verify Command Success

**❌ FRAGILE**:

```bash
pip install torch
python test_gpu.py  # Runs even if install failed!
```

**✅ ROBUST**:

```bash
pip install torch || { echo "pip install failed"; exit 1; }
python test_gpu.py
```

**Alternative (sequential checking)**:

```bash
pip install torch && python test_gpu.py
# test_gpu.py only runs if pip succeeds
```

**Rule**: Check success of commands that can fail (network operations, file I/O, package installs).

### 1.4 Conditional Testing - Use `[[ ]]` Not `[ ]`

**❌ OLD STYLE**:

```bash
if [ "$var" = "value" ]; then  # POSIX test command
    echo "match"
fi
```

**✅ MODERN BASH**:

```bash
if [[ "$var" == "value" ]]; then  # Bash builtin
    echo "match"
fi
```

**Why `[[ ]]` is better**:

- No word-splitting inside `[[ ]]` (safer with unquoted variables)
- Pattern matching: `[[ $file == *.txt ]]`
- Logical operators: `[[ $a == "x" && $b == "y" ]]`
- Regex matching: `[[ $str =~ ^[0-9]+$ ]]`

**Windows Path Example**:

```bash
if [[ -f "$python_path" ]]; then
    echo "Python executable exists"
fi
```

### 1.5 Never Use `eval` - Prevents Code Injection

**❌ DANGEROUS**:

```bash
cmd="ls -la"
eval $cmd  # If cmd="ls; rm -rf /", disaster!
```

**✅ SAFE ALTERNATIVES**:

**Use arrays**:

```bash
cmd=(ls -la)
"${cmd[@]}"
```

**Use functions**:

```bash
run_command() {
    ls -la "$@"
}
run_command /some/path
```

**Use parameter expansion**:

```bash
# Instead of: eval echo \$var_$suffix
# Use: (with nameref or indirect expansion)
declare -n varname="var_$suffix"
echo "$varname"
```

**Rule**: `eval` breaks static analysis and enables arbitrary code execution. NEVER use it.

### 1.6 Avoid `set -e` - Unreliable Error Handling

**❌ PROBLEMATIC**:

```bash
set -e  # Exit on any error
cd /tmp || echo "Warning: using current dir"  # Script exits here!
# Your error handler never runs
```

**✅ EXPLICIT CHECKING**:

```bash
cd /tmp || { echo "Warning: using current dir"; }
# Error handled, script continues
```

**Why avoid `set -e`**:

- Exits before your error handlers run
- Unpredictable with conditionals and functions
- Makes anticipated errors fatal

**Rule**: Use explicit error checking (`||`, `&&`) instead of `set -e`.

### 1.7 Command Substitution - Use `$(...)` Not Backticks

**❌ OLD STYLE**:

```bash
result=`ls -la`  # Hard to read, can't nest
```

**✅ MODERN**:

```bash
result=$(ls -la)  # Clear, nestable
result=$(echo "$(date): $(whoami)")  # Nesting works
```

**Rule**: Always use `$(...)` for command substitution.

### 1.8 Arithmetic - Use `$((...))` Not `let` or `-gt`

**❌ OLD STYLE**:

```bash
let count=$count+1
if [ $count -gt 10 ]; then
```

**✅ BASH ARITHMETIC**:

```bash
((count++))
if ((count > 10)); then
```

**Math operations**:

```bash
result=$((5 + 3))
((x = y * 2 + 1))
if ((fps >= 60)); then
    echo "High performance"
fi
```

### 1.9 String Manipulation - Use Bash Builtins

**❌ SLOW (external commands)**:

```bash
basename=$(basename "$filepath")
dirname=$(dirname "$filepath")
trimmed=$(echo "$str" | sed 's/^[[:space:]]*//')
```

**✅ FAST (bash built-in)**:

```bash
basename="${filepath##*/}"
dirname="${filepath%/*}"
trimmed="${str#"${str%%[![:space:]]*}"}"
```

**Common patterns**:

```bash
# Remove extension
filename="${path%.txt}"

# Remove directory
filename="${path##*/}"

# Replace all occurrences
clean_path="${path//\\//}"  # Backslash to forward slash

# Lowercase
lower="${str,,}"

# Uppercase
upper="${str^^}"
```

### 1.10 Avoid Useless Cat - Use Redirection

**❌ INEFFICIENT**:

```bash
cat file.txt | grep "pattern"
```

**✅ EFFICIENT**:

```bash
grep "pattern" file.txt
# OR
grep "pattern" < file.txt
```

**Rule**: Don't pipe files through `cat` unnecessarily. Most commands accept file arguments.

### 1.11 Loop Over Files - Use Globs Not `ls`

**❌ DANGEROUS (breaks on spaces)**:

```bash
for file in $(ls *.txt); do
    echo "$file"
done
```

**✅ SAFE (proper globbing)**:

```bash
for file in *.txt; do
    echo "$file"
done
```

**Recursive search**:

```bash
# Instead of: for file in $(find . -name "*.py")
while IFS= read -r -d '' file; do
    echo "$file"
done < <(find . -name "*.py" -print0)
```

**Rule**: Never parse `ls` output. Use globs or `find` with proper null-termination.

### 1.12 Reading Lines - Use `while read` Not `for`

**❌ MEMORY INEFFICIENT**:

```bash
# Loads entire file into memory
for line in $(cat large_file.txt); do
    process "$line"
done
```

**✅ MEMORY EFFICIENT**:

```bash
while IFS= read -r line; do
    process "$line"
done < large_file.txt
```

**Why**:

- `while read` streams line-by-line (constant memory)
- `for` loads entire file into memory (scales with file size)
- `read -r` prevents backslash interpretation

---

## 2. Windows/Git Bash Adaptations

### 2.1 Path Quoting Rules (MANDATORY)

**Critical Rule**: Windows paths with spaces, parentheses, or special characters MUST be quoted.

**Common Windows path patterns requiring quotes**:

```bash
# Spaces
"C:\Program Files\Python311\python.exe"
"C:\Users\Inter\AppData\Local\Temp"

# Parentheses
"C:\Program Files (x86)\Microsoft\file.exe"

# User directories (often contain spaces)
"C:\Users\Inter\My Documents\project"

# Network paths
"\\server\share\directory with spaces"
```

**Project paths** (ALWAYS quote):

```bash
project_root="D:\Users\username\projects\my-project"
cd "$project_root" || exit
```

### 2.2 Forward Slash vs Backslash

**Git Bash prefers forward slashes**:

```bash
# ✅ Works in Git Bash
cd /c/Users/Inter/project

# ✅ Also works (with quotes)
cd "C:\Users\Inter\project"

# ❌ Can be problematic (escaping needed)
cd C:\Users\Inter\project
```

**Best practice**: Use forward slashes in Git Bash, quote when using backslashes.

**Path conversion**:

```bash
# Convert backslashes to forward slashes
windows_path="C:\Users\Inter\file.txt"
bash_path="${windows_path//\\//}"
echo "$bash_path"  # C:/Users/Inter/file.txt
```

### 2.3 Drive Letter Handling

**Git Bash drive letter syntax**:

```bash
# Windows: D:\Users\alexk\
# Git Bash: /d/Users/alexk/

# Both work:
cd /d/Users/alexk/project
cd D:/Users/alexk/project
cd "D:\Users\alexk\project"
```

**Check if path exists across drives**:

```bash
if [[ -d "/d/Users/alexk/projects" ]]; then
    echo "D: drive accessible"
fi
```

### 2.4 Case Sensitivity

**Windows is case-insensitive, bash is case-sensitive**:

```bash
# These are DIFFERENT in bash comparison:
[[ "File.txt" == "file.txt" ]]  # false

# But refer to SAME file on Windows:
[[ -f "File.txt" && -f "file.txt" ]]  # both true (same file)
```

**Use case-insensitive comparison when needed**:

```bash
# Convert to lowercase for comparison
file_lower="${filename,,}"
if [[ "$file_lower" == "readme.md" ]]; then
```

### 2.5 Git Bash Feature Availability

**Available in Git Bash**:

- `[[ ]]` conditionals ✅
- `$((...))` arithmetic ✅
- Bash arrays ✅
- Parameter expansion ✅
- Process substitution `<(...)` ✅

**Limited or unavailable**:

- Some GNU-specific flags (check `--help`)
- Full POSIX signals (partial support)
- Advanced terminal control (limited)

**Test feature availability**:

```bash
# Check if command exists
if command -v python &> /dev/null; then
    echo "Python available"
fi
```

### 2.6 Windows Environment Variables

**Access Windows env vars**:

```bash
# Direct access
echo "$USERPROFILE"  # C:\Users\Inter
echo "$APPDATA"      # C:\Users\Inter\AppData\Roaming
echo "$TEMP"         # C:\Users\Inter\AppData\Local\Temp

# Convert to bash path
user_home="${USERPROFILE//\\//}"
echo "$user_home"  # C:/Users/Inter
```

**Common variables**:

- `$USERPROFILE` - User home directory
- `$APPDATA` - Application data folder
- `$LOCALAPPDATA` - Local application data
- `$TEMP` / `$TMP` - Temporary files
- `$PROGRAMFILES` - C:\Program Files
- `$PROGRAMFILES(X86)` - C:\Program Files (x86) (quote this!)

### 2.7 Line Endings (CRLF vs LF)

**Windows uses CRLF** (`\r\n`), **Unix uses LF** (`\n`).

**Git Bash handles this automatically**, but be aware when reading files:

```bash
# Strip Windows line endings
while IFS= read -r line; do
    line="${line%$'\r'}"  # Remove trailing \r
    process "$line"
done < windows_file.txt
```

**Git configuration**:

```bash
# Check current setting
git config core.autocrlf

# Recommended for Windows
git config --global core.autocrlf true
```

### 2.8 Backslashes in Quoted Strings (CRITICAL)

**⚠️ CRITICAL MISCONCEPTION**: Quoting does NOT protect backslashes from interpretation!

**❌ DANGEROUS (WILL FAIL)**:

```bash
# Many developers assume quotes protect backslashes - WRONG!
cp "D:\Users\alexk\file.sh" "F:\RD_PROJECTS\file.sh"
# ERROR: bash interprets \U, \R as escape sequences
# Result: bash sees D:Usersalexkfile.sh (backslashes consumed)
# Error: "unexpected EOF while looking for matching `"`
```

**Why this fails**:

1. Git Bash treats backslashes as escape characters EVEN INSIDE DOUBLE QUOTES
2. `\U` tries to escape `U` → becomes just `U`
3. `\R` tries to escape `R` → becomes just `R`
4. `\"` escapes the closing quote → bash looks for another quote (never finds it)

**Real-world failure example** (from production debugging):

```bash
# This command was blocked by bash-pre-validator.sh:
cp "D:\Users\alexk\FORKNI\STREAM_DIFFUSION\file.sh" "F:\RD_PROJECTS\file.sh"

# Error message:
/usr/bin/bash: eval: line 1: unexpected EOF while looking for matching `"`
```

**✅ SAFE (CORRECT)**:

```bash
# Use forward slashes (works on Windows!)
cp "D:/Users/alexk/file.sh" "F:/RD_PROJECTS/file.sh"

# OR use Git Bash drive notation
cp "/d/Users/alexk/file.sh" "/f/RD_PROJECTS/file.sh"
```

**Rule**: ALWAYS use forward slashes in bash commands, even on Windows. Never use backslashes, even inside quotes.

**Automatic Detection**: The `bash-pre-validator.sh` hook (Pattern 11) catches this error and blocks execution in strict mode, showing the corrected command.

---

### 2.9 Claude Code Edit Tool Bug (CRITICAL)

**⚠️ KNOWN BUG**: Claude Code's Edit/MultiEdit tools have a regression bug causing false "File has been unexpectedly modified" errors.

**Bug Timeline**:

- **Introduced**: Version 1.0.111 (July-August 2024)
- **Still Present**: Version 2.0.36+ (current)
- **GitHub Issues**: #3513, #7443, #7918, #7920, #7883, #8191, #8680, #8971
- **Status**: Open, no official fix as of 2025-11-07

**Root Cause**: Broken absolute path handling in Edit tool's file state validation.

#### The Bug Has TWO Distinct Causes

**PRIMARY CAUSE: Claude Code Bug** (issue #7443)

- Edit tool fails with absolute paths (both forward and back slashes)
- Affects Windows and Linux
- Occurs even when NO external modification happened
- Session state tracking bug: pre-existing files fail, newly created files work

**SECONDARY CAUSE: External Program Interference**

- VS Code file watcher
- Cloud sync (OneDrive, Dropbox, Google Drive)
- Git GUI clients
- Antivirus/Windows Defender

Most "unexpectedly modified" errors are the **PRIMARY** cause (Claude Code bug), NOT external programs.

#### Symptoms

```
Error: File has been unexpectedly modified. Read it again before attempting to write it.
```

**Bug Behavior**:

- ❌ Absolute paths: `F:/RD_PROJECTS/file.md` → FAILS
- ❌ Absolute paths: `F:\RD_PROJECTSile.md` → FAILS
- ✅ Relative paths: `./file.md` → WORKS
- ✅ Bash commands: `cat`, `sed`, Python scripts → WORKS

#### Workarounds (in order of reliability)

**1. Use Relative Paths** (Most Reliable)

```bash
# ❌ FAILS (absolute path triggers bug)
# Edit: F:/RD_PROJECTS/COMPONENTS/claude-context-local/README.md

# ✅ WORKS (relative path)
cd "F:/RD_PROJECTS/COMPONENTS/claude-context-local" || exit
# Edit: ./README.md
```

**2. Use Bash Commands Instead of Edit Tool**

```bash
# Instead of Edit tool, use:
cat > "path/to/file.md" << 'EOF'
# Your content here
EOF

# Or use sed for replacements
sed -i 's/old text/new text/' file.md

# Or use Python scripts
python -c "
with open('file.md', 'w') as f:
    f.write('new content')
"
```

**3. Add Project Documentation**

Create `CLAUDE.md` with:

```markdown
## File Path Rules (Claude Code Bug Workaround)
- **ALWAYS use relative paths** when editing files
- Example: `./src/components/Component.tsx` ✅
- **DO NOT use absolute paths**
- Example: `C:/Users/user/project/src/Component.tsx` ❌
- Reason: Claude Code bug #7443 (v1.0.111+)
```

**4. If External Programs Are ALSO Causing Issues**

```bash
# Close VS Code, Git GUI clients
# Pause cloud sync temporarily
# Wait 500ms between Read and Edit
# Use bash heredoc as atomic write
```

**5. Downgrade (Last Resort)**

```bash
npm install -g @anthropic-ai/claude-code@1.0.100
```

#### Why This Matters

**During our documentation updates today**, we encountered this error repeatedly because:

1. We used absolute paths: `F:/RD_PROJECTS/COMPONENTS/claude-context-local/CHANGELOG.md`
2. This triggered the Claude Code bug (PRIMARY cause)
3. Workaround: We switched to Python scripts and bash commands

**NOT caused by**: VS Code, cloud sync, or antivirus (though those can cause separate issues)

#### Official Tracking

**GitHub Issue #7443**: Most comprehensive discussion (57+ comments)

- Community workarounds documented
- Multiple users confirm relative paths work
- Bug remains unresolved

**Medium Article**: "The Elusive Claude 'File has been unexpectedly modified' Bug" (2025-09-27)

- Confirms bug is "fickle" and changes behavior
- Sometimes fixes itself (possible silent hotfixes)
- Recommends trying edit first, then workarounds

**Rule**: When you see "File has been unexpectedly modified":

1. **First**, assume it's the Claude Code bug (#7443)
2. **Try**: Relative paths or bash commands
3. **Only if that fails**: Consider external program interference

---

## 3. Error Recovery Reference

### 3.1 Common Error Patterns

#### Error: "No such file or directory"

**Symptom**: Command fails with file not found

**Likely Causes**:

1. **Unquoted path with spaces**

   ```bash
   # ❌ WRONG
   cd C:\Program Files\app
   # ✅ FIX
   cd "C:\Program Files\app"
   ```

2. **Wrong working directory**

   ```bash
   # Check current directory
   pwd
   # Navigate to correct location
   cd /d/Users/alexk/project || exit
   ```

3. **Backslash escaping issues**

   ```bash
   # Use forward slashes or quote
   cd /d/Users/alexk/project  # OR
   cd "D:\Users\alexk\project"
   ```

#### Error: "command not found"

**Symptom**: Bash can't find executable

**Likely Causes**:

1. **Command not in PATH**

   ```bash
   # Check if command exists
   command -v python
   # Use full path
   "/c/Program Files/Python311/python.exe"
   ```

2. **Virtual environment not activated**

   ```bash
   # Use direct path instead
   "./venv/Scripts/python.exe" script.py
   ```

3. **Trying to run Windows executable without extension**

   ```bash
   # ❌ WRONG
   ./venv/Scripts/python
   # ✅ FIX
   ./venv/Scripts/python.exe
   ```

#### Error: "syntax error near unexpected token"

**Symptom**: Bash parser error

**Likely Causes**:

1. **Unescaped special characters**

   ```bash
   # Parentheses in path
   cd "C:\Program Files (x86)\app"
   ```

2. **Missing quotes around expansion**

   ```bash
   # ❌ WRONG
   result=$(echo $var)
   # ✅ FIX (if var contains spaces)
   result=$(echo "$var")
   ```

### 3.2 Diagnostic Commands

**Check file existence**:

```bash
[[ -f "path/to/file" ]] && echo "File exists" || echo "File not found"
```

**Check directory existence**:

```bash
[[ -d "path/to/dir" ]] && echo "Directory exists" || echo "Not found"
```

**Check if command available**:

```bash
command -v python &> /dev/null && echo "Found" || echo "Not found"
```

**Debug variable content**:

```bash
# Print variable with delimiters
echo "Variable content: [$var]"

# Check if variable is set
[[ -z "$var" ]] && echo "Variable empty or unset"
```

**Test path with spaces**:

```bash
path="C:\Program Files\app"
echo "Unquoted: $path"
echo "Quoted: \"$path\""
ls -la "$path"  # This should work
```

### 3.3 Recovery Workflows

#### Workflow: Fix "cd" Failure

1. **Verify path exists**:

   ```bash
   ls -la "D:\Users\alexk\projects"
   ```

2. **Check working directory**:

   ```bash
   pwd
   ```

3. **Try forward slash variant**:

   ```bash
   cd /d/Users/alexk/projects
   ```

4. **Add error handler**:

   ```bash
   cd /d/Users/alexk/projects || { echo "Failed to cd"; exit 1; }
   ```

#### Workflow: Fix Pip Install Failure

1. **Verify pip exists**:

   ```bash
   "./venv/Scripts/pip.exe" --version
   ```

2. **Check Python version**:

   ```bash
   "./venv/Scripts/python.exe" --version
   ```

3. **Test with verbose output**:

   ```bash
   "./venv/Scripts/pip.exe" install package-name -v
   ```

4. **Check network/index URL**:

   ```bash
   "./venv/Scripts/pip.exe" install package-name --index-url https://pypi.org/simple
   ```

#### Workflow: Fix File Not Found in Loop

1. **Test glob pattern**:

   ```bash
   for f in *.txt; do
       echo "Found: [$f]"
   done
   ```

2. **Check if files exist**:

   ```bash
   if compgen -G "*.txt" > /dev/null; then
       echo "Files match pattern"
   fi
   ```

3. **Use find with null-termination**:

   ```bash
   while IFS= read -r -d '' file; do
       echo "Processing: $file"
   done < <(find . -name "*.txt" -print0)
   ```

---

## 4. Quick Reference Tables

### 4.1 Safe vs Dangerous Patterns

| Situation | ❌ Dangerous | ✅ Safe |
|-----------|-------------|---------|
| **Directory change** | `cd /path` | `cd /path \|\| exit` |
| **Variable with spaces** | `rm $file` | `rm "$file"` |
| **Windows path** | `cd C:\Program Files\app` | `cd "C:\Program Files\app"` |
| **Conditional test** | `if [ $var = "x" ]` | `if [[ "$var" == "x" ]]` |
| **Loop over files** | `for f in $(ls)` | `for f in *` |
| **Read lines** | `for line in $(cat file)` | `while read -r line; do ... done < file` |
| **Command substitution** | ``result=`cmd` `` | `result=$(cmd)` |
| **Arithmetic** | `let x=x+1` | `((x++))` |
| **File redirection** | `cat file \| grep x` | `grep x file` |
| **String extraction** | `basename $(cmd)` | `result=$(cmd); echo "${result##*/}"` |

### 4.2 Pre-flight Checklist

Before invoking bash command, verify:

- [ ] All paths with spaces are quoted: `"$path"`
- [ ] `cd` commands have error handlers: `cd /path || exit`
- [ ] Windows paths use quotes: `"C:\Program Files\..."`
- [ ] Virtual env paths are quoted: `"./venv/Scripts/python.exe"`
- [ ] Using `[[ ]]` not `[ ]` for conditionals
- [ ] No unquoted `$var` in file operations
- [ ] No `eval` in command execution
- [ ] Critical commands have error checking: `cmd || exit`

### 4.3 Error Message → Likely Cause

| Error Message | Likely Cause | Quick Fix |
|---------------|--------------|-----------|
| "No such file or directory" | Unquoted path with spaces | Add quotes: `"$path"` |
| "command not found" | Wrong PATH or missing .exe | Use full path: `"./venv/Scripts/python.exe"` |
| "syntax error near unexpected token" | Unescaped special char | Quote path with parentheses: `"C:\...(x86)\..."` |
| "Permission denied" | Wrong permissions or Windows path | Check file exists, use forward slashes |
| "Is a directory" | Trying to read/execute a directory | Check path, add filename |
| "Not a directory" | Trying to cd into a file | Verify path with `ls -la` |
| "Illegal option" | GNU-specific flag not in Git Bash | Check `--help`, use portable flags |

### 4.4 Windows Path Conversion Cheat Sheet

| Windows Format | Git Bash Format | Bash Variable |
|----------------|-----------------|---------------|
| `C:\Users\Inter` | `/c/Users/Inter` | `path="/c/Users/Inter"` |
| `D:\Projects` | `/d/Projects` | `path="/d/Projects"` |
| `\\server\share` | `//server/share` | `path="//server/share"` |
| `%USERPROFILE%` | `$USERPROFILE` (converted) | `path="$USERPROFILE"` |
| `C:\Program Files (x86)` | `/c/Program Files (x86)` | `path="/c/Program Files (x86)"` |

**Conversion function**:

```bash
win_to_bash() {
    local win_path="$1"
    # Convert backslashes to forward slashes
    local bash_path="${win_path//\\//}"
    echo "$bash_path"
}

# Usage
bash_path=$(win_to_bash "C:\Users\Inter\project")
cd "$bash_path" || exit
```

---

## 5. Integration with Claude Code

### 5.1 When to Consult This Guide

**Pre-flight (before invoking Bash tool)**:

- Command contains `cd`
- Command involves file operations (`rm`, `mv`, `cp`)
- Command uses paths with spaces, parentheses, or special chars
- Command chains multiple operations with `&&` or `||`
- Command uses virtual environment executables

**Error recovery (after bash failure)**:

- Check Section 3 (Error Recovery Reference)
- Match error message to Section 4.3 table
- Apply diagnostic commands from Section 3.2
- Follow recovery workflow from Section 3.3

### 5.2 Automatic Validation Triggers

Claude Code should automatically consult this guide when detecting:

- `cd` without `|| exit` or `|| return`
- Unquoted `$var` in file paths
- Windows paths without quotes: `C:\...` not in quotes
- Parsing `ls` output: `for f in $(ls ...)`
- Using `[ ]` instead of `[[ ]]`
- Using backticks instead of `$(...)`

### 5.3 Quick Lookup by Error Type

| Error Type | Section | Key Info |
|------------|---------|----------|
| Path with spaces | 1.2, 2.1 | Quote variables: `"$path"` |
| cd failure | 1.1, 3.3 | Add `\|\| exit`, verify path exists |
| Command not found | 3.1, 4.3 | Check PATH, use full path to .exe |
| Syntax error | 3.1, 2.1 | Quote special chars: `"C:\...(x86)\..."` |
| Loop breaks on spaces | 1.11, 1.12 | Use `for f in *` not `$(ls)` |
| Windows path issues | 2.1-2.4 | Use forward slashes or quote backslashes |
| Venv not working | 2.1, 3.1 | Quote exe path: `"./venv/Scripts/python.exe"` |

### 5.4 ShellCheck Integration

**Tool**: [ShellCheck](https://www.shellcheck.net/) - Static analysis for shell scripts

**Purpose**: Automatically catch common bash pitfalls before runtime

**Installation**:

```bash
# Windows (via scoop)
scoop install shellcheck

# Linux
apt-get install shellcheck

# macOS
brew install shellcheck
```

**Usage**:

```bash
# Check all project scripts
./scripts/lint/check_shell.sh

# Manual check of single script
shellcheck --severity=warning scripts/git/commit_enhanced.sh
```

**Common Error Codes and Fixes**:

| Code | Issue | Fix | Example |
|------|-------|-----|---------|
| SC2086 | Double quote to prevent globbing/splitting | `"$var"` not `$var` | `cd "$path"` |
| SC2046 | Quote to prevent word splitting | `"$(cmd)"` | `files="$(find ...)"` |
| SC2006 | Use $(...) notation | `$(cmd)` not `` `cmd` `` | `output=$(ls)` |
| SC2164 | Use cd ... \|\| exit | Always handle cd failure | `cd "$dir" \|\| exit` |
| SC2034 | Variable unused | Remove or export | `export VAR` or remove |
| SC2155 | Declare and assign separately | Split declaration | `local var; var=$(cmd)` |

**Inline Disables (When Intentional)**:

```bash
# shellcheck disable=SC2034  # Variable used by sourcing script
INTENTIONAL_UNUSED_VAR=value

# shellcheck disable=SC2155  # Safe: command doesn't use exit code
local timestamp=$(date +%Y%m%d)
```

**Integration with Lint Workflow**:

The `check_shell.sh` script:
- Automatically finds all `.sh` files in `scripts/`
- Gracefully handles missing shellcheck (warning only)
- Outputs gcc-style messages for IDE integration
- Returns exit code 1 on any issues found

**When to Disable Warnings**:
- Variables intentionally unused (e.g., sourced by other scripts)
- Commands where exit codes are safely ignored
- Platform-specific patterns that ShellCheck doesn't recognize
- Always add comments explaining why the disable is needed

---

## 6. Summary: Golden Rules

1. **ALWAYS quote paths**: `"$path"` not `$path`
2. **ALWAYS check cd**: `cd /path || exit` not just `cd /path`
3. **ALWAYS use [[ ]]**: `[[ ... ]]` not `[ ... ]`
4. **NEVER use eval**: Use arrays or functions instead
5. **NEVER parse ls**: Use globs (`for f in *`) or find
6. **Quote Windows paths**: `"C:\Program Files\..."` always
7. **Quote venv executables**: `"./venv/Scripts/python.exe"`
8. **Check command success**: `cmd || exit` for critical operations
9. **Use bash builtins**: `$(...)`, `$((...))`, `${var//old/new}`
10. **Test before deploying**: Verify paths exist, commands available

---

**Last Updated**: 2025-10-31

**Character Count**: ~24,000 characters (TouchDesigner-specific section removed)

**License**: MIT (adapted from bash-style-guide by Dave Eddy)
