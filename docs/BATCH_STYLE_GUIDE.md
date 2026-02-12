# Windows Batch Scripting Safety Guide for Claude Code

**Purpose**: Defensive Batch scripting patterns for Windows CMD environments, optimized for error prevention and maintainability.

**Target Environment**: Windows 10/11 CMD.exe, PowerShell (batch compatibility mode), Windows Terminal.

**Philosophy**: Safety over elegance. Prevent errors before they occur, provide clear recovery paths when they do.

**Sources**:

- [Windows Batch Scripting Guide](https://steve-jansen.github.io/guides/windows-batch-scripting/) by Steve Jansen
- [Batch Style Guide](https://github.com/trgwii/Batch-Style-Guide) by trgwii
- [Stack Exchange Code Review](https://codereview.stackexchange.com/questions/139188/) - Batch best practices

---

## Table of Contents

1. [Critical Safety Patterns](#1-critical-safety-patterns)
2. [Windows Batch Specifics](#2-windows-batch-specifics)
3. [Error Recovery Reference](#3-error-recovery-reference)
4. [Quick Reference Tables](#4-quick-reference-tables)
5. [Integration with Claude Code](#5-integration-with-claude-code)
6. [Summary: Golden Rules](#6-summary-golden-rules)

---

## 1. Critical Safety Patterns

### 1.1 Variable Expansion - Prevent Word-Splitting

**❌ DANGEROUS**:

```batch
set file=my document.txt
del %file%
REM Tries to delete "my" and "document.txt" separately!
```

**✅ SAFE**:

```batch
set "file=my document.txt"
del "%file%"
REM Deletes "my document.txt" as single file
```

**Rule**: Quote variables containing paths, filenames, or user input: `"%var%"` not `%var%`.

**Best Practice - Quote Assignment**:

```batch
REM Quote the entire assignment to prevent trailing spaces
set "project_path=C:\Users\Inter\My Projects"
set "python_exe=%project_path%\venv\Scripts\python.exe"

REM Now safe to use
"%python_exe%" script.py
```

**Why quote assignments**: Prevents accidental trailing spaces from becoming part of the variable value.

### 1.2 Error Checking - Verify Command Success

**❌ FRAGILE**:

```batch
pip install torch
python test_gpu.py
REM Runs even if install failed!
```

**✅ ROBUST**:

```batch
pip install torch
if %ERRORLEVEL% neq 0 (
    echo ERROR: pip install failed
    exit /b 1
)
python test_gpu.py
```

**Alternative (compact form)**:

```batch
pip install torch || (
    echo ERROR: pip install failed
    exit /b 1
)
python test_gpu.py
```

**Rule**: Check `%ERRORLEVEL%` after commands that can fail (network operations, file I/O, package installs).

**Common Error Levels**:

- `0` = Success
- `1` = General error
- `2` = Command not found or syntax error
- `9009` = Command/file not found

### 1.3 Directory Changes - Use PUSHD/POPD

**❌ DANGEROUS**:

```batch
cd C:\Some\Path
del /q *
REM If cd failed, deletes files in wrong directory!
```

**✅ SAFE**:

```batch
pushd "C:\Some\Path" || (
    echo ERROR: Failed to change to directory
    exit /b 1
)
del /q *
popd
```

**Why PUSHD/POPD is better**:

- `pushd` saves current directory automatically
- `popd` restores it reliably
- Works across drive letters: `pushd D:\OtherDrive\Path`
- Prevents directory context pollution

**Script-relative paths**:

```batch
REM Change to script's own directory
pushd "%~dp0" || exit /b 1
REM Now can use relative paths reliably
call setup.bat
popd
```

### 1.4 Delayed Expansion - Handle Variables in Loops

**❌ BROKEN (without delayed expansion)**:

```batch
set count=0
for %%f in (*.txt) do (
    set /a count=%count%+1
    echo %count%
)
REM Always prints 0 because %count% expands before loop starts
```

**✅ WORKS (with delayed expansion)**:

```batch
setlocal EnableDelayedExpansion
set count=0
for %%f in (*.txt) do (
    set /a count=!count!+1
    echo !count!
)
endlocal
REM Correctly prints 1, 2, 3...
```

**Rule**: Use `setlocal EnableDelayedExpansion` and `!var!` syntax when modifying variables inside loops or conditionals.

**When to use which syntax**:

- `%var%` - Normal expansion (expanded when line is parsed)
- `!var!` - Delayed expansion (expanded when line is executed)

### 1.5 Special Characters - Escaping Rules

**Critical Special Characters in Batch**:

- `%` - Variable expansion
- `^` - Escape character
- `&` - Command separator
- `|` - Pipe
- `<` `>` - Redirection
- `(` `)` - Command grouping
- `!` - Delayed expansion (when enabled)

**❌ DANGEROUS**:

```batch
echo 50% complete
REM Error: %c is interpreted as variable expansion
```

**✅ SAFE**:

```batch
echo 50%% complete
REM Double %% to escape literal percent sign
```

**Escaping with caret (^)**:

```batch
REM Escape special characters with ^
echo This ^& that
echo Redirection: ^< ^> ^|

REM Escape in variables
set "message=Error ^& Warning"
```

**Paths with parentheses**:

```batch
REM Always quote paths with parentheses
set "prog_x86=C:\Program Files (x86)\App"
if exist "%prog_x86%" (
    echo Found
)
```

### 1.6 Quoting Paths - Critical for Windows

**Rule**: ALWAYS quote paths that contain:

- Spaces
- Parentheses
- Ampersands (&)
- Any special characters

**Common Windows paths requiring quotes**:

```batch
set "program_files=C:\Program Files"
set "program_files_x86=C:\Program Files (x86)"
set "user_profile=%USERPROFILE%"
set "app_data=%APPDATA%"
set "temp=%TEMP%"

REM Always quote when using
cd "%program_files%\MyApp" || exit /b 1
```

**Network paths**:

```batch
set "network_share=\\server\share\folder with spaces"
pushd "%network_share%" || exit /b 1
```

### 1.7 Variable Scope - SETLOCAL/ENDLOCAL

**❌ POLLUTION (global scope)**:

```batch
set temp_var=something
call other_script.bat
REM temp_var now pollutes global environment
```

**✅ CLEAN (local scope)**:

```batch
setlocal
set temp_var=something
call other_script.bat
endlocal
REM temp_var only existed during setlocal/endlocal block
```

**Rule**: Wrap all scripts that define variables with `setlocal`/`endlocal`.

**Returning values from local scope**:

```batch
setlocal
set result=calculated_value
endlocal & set "output_var=%result%"
REM output_var is set in parent scope
```

**Critical**: Ensure `endlocal` runs even on early exits:

```batch
setlocal
if exist file.txt (
    endlocal
    exit /b 0
)
REM More processing
endlocal
```

### 1.8 Calling Other Scripts - Use CALL Command

**❌ BROKEN (script terminates)**:

```batch
other_script.bat
REM Control never returns here!
echo This never prints
```

**✅ WORKS (control returns)**:

```batch
call other_script.bat
REM Control returns here
echo This prints
```

**Rule**: Always use `call` to invoke other batch scripts. Without `call`, the calling script terminates.

**Calling with error checking**:

```batch
call setup.bat
if %ERRORLEVEL% neq 0 (
    echo ERROR: setup.bat failed
    exit /b 1
)
```

**Calling with arguments**:

```batch
call build.bat Release x64
if errorlevel 1 exit /b 1
```

### 1.9 Argument Handling - Strip Quotes with %~N

**Problem**: Arguments with paths often have quotes that need stripping.

**❌ INCLUDES QUOTES**:

```batch
set file=%1
echo %file%
REM If called with: script.bat "my file.txt"
REM Prints: "my file.txt" (quotes included)
```

**✅ STRIPS QUOTES**:

```batch
set "file=%~1"
echo %file%
REM Prints: my file.txt (quotes stripped)
```

**Argument modifiers**:

```batch
REM %~1 - Strip quotes
REM %~f1 - Full path
REM %~d1 - Drive letter only
REM %~p1 - Path only
REM %~n1 - Filename only
REM %~x1 - Extension only
REM %~dp1 - Drive + path (directory of file)

REM Example
set "input_file=%~f1"
set "input_dir=%~dp1"
set "input_name=%~n1"
```

### 1.10 Loops - FOR Command Variants

**Loop over files** (basic):

```batch
for %%f in (*.txt) do (
    echo Processing: %%f
)
```

**Loop with directory walk** (recursive):

```batch
for /r "C:\MyFolder" %%f in (*.txt) do (
    echo Found: %%f
)
```

**Loop with delimiters** (parse text):

```batch
for /f "tokens=1,2 delims=," %%a in (data.csv) do (
    echo Column 1: %%a
    echo Column 2: %%b
)
```

**Loop over command output**:

```batch
for /f "delims=" %%i in ('dir /b *.txt') do (
    echo File: %%i
)
```

**Rule**: Use `%%` for loop variables in batch files, `%` in command line.

### 1.11 Conditionals - IF Statement Patterns

**File/folder existence**:

```batch
if exist "file.txt" (
    echo File exists
) else (
    echo File not found
)

if exist "C:\Folder\" (
    echo Folder exists
)
```

**String comparison**:

```batch
if "%var%"=="value" (
    echo Match
)

REM Case-insensitive
if /i "%var%"=="VALUE" (
    echo Match (ignoring case)
)
```

**Numeric comparison**:

```batch
if %count% gtr 10 (
    echo Greater than 10
)

REM Operators: equ, neq, lss, leq, gtr, geq
```

**Check if variable defined**:

```batch
if defined MY_VAR (
    echo Variable is set
) else (
    echo Variable not set
)
```

**Check ERRORLEVEL**:

```batch
command_that_might_fail
if errorlevel 1 (
    echo Command failed
    exit /b 1
)

REM More precise
if %ERRORLEVEL% neq 0 (
    echo Error code: %ERRORLEVEL%
)
```

### 1.12 Avoid String Concatenation Errors

**❌ TRAILING SPACE PROBLEM**:

```batch
set path=C:\Users\Inter
REM Notice trailing space after Inter
set full_path=%path%\Documents
REM Results in: C:\Users\Inter \Documents (broken path)
```

**✅ PREVENT WITH QUOTED ASSIGNMENT**:

```batch
set "path=C:\Users\Inter"
set "full_path=%path%\Documents"
REM Clean path without trailing spaces
```

**Rule**: Always use quoted assignment form: `set "var=value"` to prevent trailing space bugs.

---

## 2. Windows Batch Specifics

### 2.1 Line Endings - CRLF Requirement

**Critical**: Batch files MUST use Windows-style line endings (CRLF / `\r\n`).

**Unix-style (LF) line endings can cause**:

- Commands not recognized
- Syntax errors
- Unpredictable script behavior

**Check line endings in Git**:

```batch
git config core.autocrlf
REM Should be: true (for Windows)
```

**Set globally**:

```batch
git config --global core.autocrlf true
```

**In .gitattributes**:

```
*.bat text eol=crlf
*.cmd text eol=crlf
```

### 2.2 Echo Control - @ Prefix vs @echo off

**Best Practice**: Use `@` prefix on individual commands, not global `@echo off`.

**❌ PROBLEMATIC (global echo off)**:

```batch
@echo off
command1
command2
REM Can't selectively re-enable echo for debugging
```

**✅ RECOMMENDED (selective @ prefix)**:

```batch
@command1
@command2
@echo Debug output: %var%
REM Can show/hide output per command
```

**Why avoid global `@echo off`**:

- Operates globally per process
- Makes selective re-enabling difficult
- Complicates debugging
- Not suitable for modular scripts

**Exception**: Global `@echo off` is acceptable for simple, standalone scripts:

```batch
@echo off
REM Simple installer script
echo Installing application...
copy files destination
```

### 2.3 Comment Styles - REM vs :: (Critical Choice)

**Two comment syntaxes exist with different tradeoffs:**

**REM - RECOMMENDED (safer)**:

```batch
rem This is a comment
@echo Starting process
rem Another comment
```

**Advantages of REM**:

- True comment command, not a label
- Works reliably in all contexts
- No semantic confusion with labels
- Safe inside code blocks and loops

**:: - PROBLEMATIC (label syntax)**:

```batch
:: This looks like a comment
@echo Starting process
:: But it's actually a label
```

**Problems with ::**:

- Actually a label, not a true comment
- Breaks inside parenthesized blocks
- Confuses parser in some contexts
- Has semantic meaning (jump target)

**Critical failure example**:

```batch
if exist file.txt (
    :: This breaks
    echo Processing
)
REM Error: Labels not valid inside blocks
```

**Recommendation**: Use `rem` for all comments. Only use `::` for section dividers at the top level:

```batch
@echo off
:: ====================================
:: SECTION: Setup
:: ====================================
rem This is a regular comment
```

**Inline comments with &&**:

```batch
@copy file.txt backup.txt && rem Create backup
```

### 2.4 Script Directory - %~dp0 Pattern

**Problem**: Scripts need to reference files relative to themselves, not the caller's working directory.

**❌ WRONG (uses current directory)**:

```batch
@call setup\init.bat
REM Looks for setup\ relative to caller's directory
```

**✅ RIGHT (uses script's directory)**:

```batch
@call "%~dp0setup\init.bat"
REM Looks for setup\ relative to this script
```

**The %~dp0 variable**:

- `%0` - Script's own filename
- `%~d0` - Drive letter only (e.g., `C:`)
- `%~p0` - Path only (e.g., `\Users\Inter\Scripts\`)
- `%~dp0` - Drive + path (e.g., `C:\Users\Inter\Scripts\`)
- Always ends with backslash

**Common pattern**:

```batch
@echo off
pushd "%~dp0" || exit /b 1
rem Now in script's own directory
call setup.bat
call utils\helper.bat
popd
```

**Quote %~dp0 when paths have spaces**:

```batch
set "script_dir=%~dp0"
set "config_file=%script_dir%config.ini"
```

### 2.5 Argument Passing and Quoting

**Quote arguments with spaces**:

```batch
call script.bat "C:\Program Files\App" "my file.txt"
```

**Inside script, strip quotes**:

```batch
set "arg1=%~1"
set "arg2=%~2"
echo First arg: %arg1%
echo Second arg: %arg2%
```

**Check argument count**:

```batch
if "%~1"=="" (
    echo ERROR: Missing required argument
    echo Usage: %~nx0 ^<input_file^>
    exit /b 1
)
```

**Pass arguments with special characters**:

```batch
rem Escape & with ^
call script.bat "File ^& Data"

rem Or quote the entire call
call script.bat "File & Data"
```

### 2.6 Environment Variables - Windows Built-ins

**Common Windows environment variables**:

```batch
echo %USERPROFILE%     REM C:\Users\Inter
echo %USERNAME%        REM Inter
echo %COMPUTERNAME%    REM DESKTOP-ABC123
echo %APPDATA%         REM C:\Users\Inter\AppData\Roaming
echo %LOCALAPPDATA%    REM C:\Users\Inter\AppData\Local
echo %TEMP%            REM C:\Users\Inter\AppData\Local\Temp
echo %TMP%             REM Same as TEMP
echo %PROGRAMFILES%    REM C:\Program Files
echo %PROGRAMFILES(X86)%  REM C:\Program Files (x86)
echo %WINDIR%          REM C:\Windows
echo %SYSTEMROOT%      REM C:\Windows
echo %SYSTEMDRIVE%     REM C:
echo %COMSPEC%         REM C:\Windows\System32\cmd.exe
```

**Usage**:

```batch
set "temp_file=%TEMP%\script_output.txt"
set "config=%APPDATA%\MyApp\config.ini"

if exist "%PROGRAMFILES%\MyApp\" (
    echo Found in Program Files
)
```

**Check if variable set**:

```batch
if not defined MYVAR (
    echo MYVAR is not set
    set "MYVAR=default_value"
)
```

### 2.7 String Operations - No Built-in Functions

**Batch has limited string manipulation compared to bash.**

**Substring extraction**:

```batch
set "str=Hello World"
echo %str:~0,5%
REM Output: Hello (start at 0, length 5)

echo %str:~6%
REM Output: World (start at 6, to end)

echo %str:~-5%
REM Output: World (last 5 characters)
```

**String replacement**:

```batch
set "path=C:\Users\Inter\Documents"
set "unix_path=%path:\=/%"
echo %unix_path%
REM Output: C:/Users/Inter/Documents

set "filename=my_file_name.txt"
set "no_underscores=%filename:_= %"
echo %no_underscores%
REM Output: my file name.txt
```

**String length** (no native support, need workaround):

```batch
set "str=Hello"
set "len=0"
setlocal EnableDelayedExpansion
:length_loop
if defined str (
    set "str=!str:~1!"
    set /a len+=1
    goto length_loop
)
echo Length: !len!
endlocal
```

**Case conversion** (requires helper files or PowerShell):

```batch
rem Lowercase (hacky method using label)
for %%L in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do (
    set "str=!str:%%L=%%L!"
)

rem Better: Use PowerShell for complex string ops
powershell -Command "('HELLO').ToLower()"
```

### 2.8 TIMEOUT vs PING for Delays

**❌ OLD HACK (unreliable)**:

```batch
ping 127.0.0.1 -n 6 > nul
rem Delays ~5 seconds (n-1)
```

**✅ MODERN (native timeout)**:

```batch
timeout /t 5
rem Delays 5 seconds, shows countdown

timeout /t 5 /nobreak
rem User can't interrupt with keypress

timeout /t 5 /nobreak > nul
rem Silent delay
```

**Rule**: Use `timeout` command, not `ping` hack.

### 2.9 File Organization - Recommended Structure

**Organize batch files into sections**:

```batch
@echo off
:: ============================================================
:: Script Name: backup.bat
:: Description: Backup utility for project files
:: Author: Your Name
:: Date: 2025-11-17
:: ============================================================

setlocal EnableDelayedExpansion

:: ------------------------------------------------------------
:: SECTION 1: Configuration Parameters
:: ------------------------------------------------------------
set "source_dir=C:\Projects\MyApp"
set "backup_dir=D:\Backups"
set "timestamp=%DATE:/=-%_%TIME::=-%"
set "timestamp=%timestamp: =0%"

:: ------------------------------------------------------------
:: SECTION 2: Input Validation
:: ------------------------------------------------------------
if not exist "%source_dir%\" (
    echo ERROR: Source directory not found: %source_dir%
    exit /b 1
)

if not exist "%backup_dir%\" (
    echo ERROR: Backup directory not found: %backup_dir%
    exit /b 1
)

:: ------------------------------------------------------------
:: SECTION 3: Main Script Logic
:: ------------------------------------------------------------
echo Starting backup...
call :backup_files
if errorlevel 1 exit /b 1

echo Backup completed successfully
endlocal
exit /b 0

:: ------------------------------------------------------------
:: SECTION 4: Functions
:: ------------------------------------------------------------

:backup_files
rem Function: backup_files
rem Description: Copies files to backup directory
rem Parameters: None (uses parent scope variables)
rem Returns: 0 on success, 1 on failure

    set "target=%backup_dir%\backup_%timestamp%"
    mkdir "%target%" 2>nul

    xcopy "%source_dir%\*" "%target%\" /E /I /Y
    if errorlevel 1 (
        echo ERROR: Backup failed
        exit /b 1
    )

    echo Backed up to: %target%
    exit /b 0
```

**Section order** (recommended):

1. Script header comment
2. Main options (`@echo off`, `setlocal`)
3. Configuration parameters (hard-coded variables)
4. Prompt/input parameters (user input)
5. Calculated variables (derived values)
6. Input validation
7. Main script logic
8. Functions (at end)

### 2.10 Exit Codes - Best Practices

**Use exit /b to return from script**:

```batch
exit /b 0     REM Success
exit /b 1     REM General error
exit /b 2     REM Misuse (invalid arguments)
exit /b 3     REM Configuration error
```

**Why /b flag**:

- `exit` - Closes entire CMD window
- `exit /b` - Returns from batch script only (batch return)

**Set exit code explicitly**:

```batch
if %ERRORLEVEL% neq 0 (
    echo Command failed
    exit /b %ERRORLEVEL%
)
```

**In functions, return specific codes**:

```batch
:validate_config
    if not exist "%config_file%" (
        echo ERROR: Config file missing
        exit /b 3
    )
    exit /b 0
```

---

## 3. Error Recovery Reference

### 3.1 Common Error Patterns

#### Error: "is not recognized as an internal or external command"

**Symptom**: Command not found by CMD.

**Likely Causes**:

1. **Command not in PATH**

   ```batch
   rem ❌ FAILS if python not in PATH
   python script.py

   rem ✅ FIX: Use full path
   "C:\Python311\python.exe" script.py
   ```

2. **Typo in command name**

   ```batch
   rem Check spelling
   where python
   rem Lists all python.exe in PATH
   ```

3. **Virtual environment not activated**

   ```batch
   rem ❌ WRONG: Assumes activation
   python script.py

   rem ✅ RIGHT: Use direct path
   "%~dp0venv\Scripts\python.exe" script.py
   ```

#### Error: "The system cannot find the path specified"

**Symptom**: Path doesn't exist or isn't accessible.

**Likely Causes**:

1. **Unquoted path with spaces**

   ```batch
   rem ❌ FAILS
   cd C:\Program Files\App

   rem ✅ FIX
   cd "C:\Program Files\App"
   ```

2. **Wrong working directory**

   ```batch
   rem Check current directory
   cd

   rem Use absolute paths or pushd to script directory
   pushd "%~dp0" || exit /b 1
   ```

3. **Path uses forward slashes**

   ```batch
   rem Windows CMD prefers backslashes
   rem Most commands accept both, but CD can be picky
   cd C:/Users/Inter/Projects
   rem Try with backslashes if fails:
   cd C:\Users\Inter\Projects
   ```

#### Error: "The process cannot access the file because it is being used by another process"

**Symptom**: File locked by another program.

**Likely Causes**:

1. **File open in editor/viewer**

   ```batch
   rem Close file in VS Code, Notepad, Excel, etc.
   rem Or use /Y flag to force overwrite if supported
   ```

2. **Script accessing its own file**

   ```batch
   rem Can't modify batch file while it's running
   rem Use external script or delayed modification
   ```

3. **Antivirus scanning file**

   ```batch
   rem Wait and retry
   timeout /t 2 /nobreak > nul
   copy source.txt dest.txt
   ```

#### Error: "Access is denied"

**Symptom**: Insufficient permissions.

**Likely Causes**:

1. **Need administrator privileges**

   ```batch
   rem Run CMD as Administrator
   rem Or use runas command
   runas /user:Administrator "cmd.exe"
   ```

2. **File is read-only**

   ```batch
   rem Remove read-only attribute
   attrib -r file.txt
   rem Then modify/delete
   ```

3. **System/protected directory**

   ```batch
   rem Can't write to C:\Windows\System32
   rem Use %TEMP%, %APPDATA%, or user directories instead
   set "output=%TEMP%\my_output.txt"
   ```

### 3.2 Diagnostic Commands

**Check file/directory existence**:

```batch
if exist "C:\path\to\file.txt" (echo File exists) else (echo Not found)
if exist "C:\path\to\folder\" (echo Folder exists) else (echo Not found)
```

**Find command location**:

```batch
where python
rem Shows all python.exe in PATH

where /r "C:\Program Files" python.exe
rem Search recursively for python.exe
```

**Check environment variable**:

```batch
echo %PATH%
rem Show PATH value

set MY
rem Show all variables starting with MY
```

**Test path with spaces**:

```batch
set "test_path=C:\Program Files\App"
echo Unquoted: %test_path%
echo Quoted: "%test_path%"
dir "%test_path%"
```

**Check ERRORLEVEL**:

```batch
command_that_might_fail
echo Exit code: %ERRORLEVEL%

if %ERRORLEVEL% neq 0 (
    echo Command failed with code: %ERRORLEVEL%
)
```

**List directory contents**:

```batch
dir /b
rem Basic list (filenames only)

dir /s /b *.txt
rem Recursive search for *.txt
```

### 3.3 Recovery Workflows

#### Workflow: Fix "Command Not Found"

1. **Check if command exists**:

   ```batch
   where python
   ```

2. **If not found, locate manually**:

   ```batch
   dir /s /b C:\python.exe
   rem Search C: drive for python.exe
   ```

3. **Use full path**:

   ```batch
   "C:\Python311\python.exe" --version
   ```

4. **Add to PATH** (if needed permanently):

   ```batch
   setx PATH "%PATH%;C:\Python311"
   rem Restart CMD for change to take effect
   ```

#### Workflow: Fix Path Not Found

1. **Verify path exists**:

   ```batch
   dir "C:\Program Files\MyApp"
   ```

2. **Check current directory**:

   ```batch
   cd
   ```

3. **Try with quotes**:

   ```batch
   cd "C:\Program Files\MyApp"
   ```

4. **Use pushd for reliability**:

   ```batch
   pushd "C:\Program Files\MyApp" || (
       echo ERROR: Cannot access directory
       exit /b 1
   )
   ```

#### Workflow: Fix Permission Denied

1. **Check file attributes**:

   ```batch
   attrib file.txt
   ```

2. **Remove read-only if set**:

   ```batch
   attrib -r file.txt
   ```

3. **Try running as administrator**:

   ```batch
   rem Right-click CMD icon → Run as Administrator
   ```

4. **Use accessible location**:

   ```batch
   rem Instead of C:\Windows or C:\Program Files
   set "output=%USERPROFILE%\Documents\output.txt"
   ```

---

## 4. Quick Reference Tables

### 4.1 Safe vs Dangerous Patterns

| Situation | ❌ Dangerous | ✅ Safe |
|-----------|-------------|---------|
| **Variable assignment** | `set var=value` (trailing space) | `set "var=value"` |
| **Variable use** | `echo %var%` (if spaces) | `echo "%var%"` |
| **Directory change** | `cd C:\Path` | `pushd "C:\Path" \|\| exit /b 1` |
| **Calling scripts** | `script.bat` | `call script.bat` |
| **Error checking** | `command` | `command \|\| exit /b 1` |
| **Path with spaces** | `cd C:\Program Files` | `cd "C:\Program Files"` |
| **Loop variables** | `set count=%count%+1` (in loop) | `set /a count=!count!+1` (with DelayedExpansion) |
| **Comments** | `:: comment` (in blocks) | `rem comment` |
| **Exit script** | `exit` (closes window) | `exit /b 0` |
| **String comparison** | `if %var%==value` | `if "%var%"=="value"` |
| **Arguments** | `set file=%1` | `set "file=%~1"` |

### 4.2 Pre-flight Checklist

Before executing batch script, verify:

- [ ] All paths with spaces are quoted: `"%path%"`
- [ ] Directory changes use `pushd`/`popd`: `pushd "C:\Path" || exit /b 1`
- [ ] Variable assignments use quotes: `set "var=value"`
- [ ] Script calls use `call`: `call other.bat`
- [ ] Functions return with `exit /b`: `exit /b 0`
- [ ] Delayed expansion enabled if modifying variables in loops
- [ ] Script starts with `@echo off` and `setlocal`
- [ ] Critical commands check errors: `if errorlevel 1 exit /b 1`
- [ ] Comments use `rem` not `::` (inside blocks)
- [ ] `%~dp0` used for script-relative paths

### 4.3 Error Message → Likely Cause

| Error Message | Likely Cause | Quick Fix |
|---------------|--------------|-----------|
| "not recognized as an internal or external command" | Command not in PATH | Use full path: `"C:\path\to\command.exe"` |
| "The system cannot find the path specified" | Unquoted path with spaces | Add quotes: `"%path%"` |
| "The process cannot access the file" | File in use | Close file in other programs |
| "Access is denied" | Insufficient permissions | Run as Administrator or use user directory |
| "The syntax of the command is incorrect" | Missing quotes or escape | Quote special characters |
| "A duplicate label" | Using `::` in wrong context | Use `rem` instead |
| "%ERRORLEVEL% was unexpected" | Variable not expanded | Enable delayed expansion: `!ERRORLEVEL!` |
| "The system cannot find the batch label specified" | Typo in function name | Check `:function_name` spelling |

### 4.4 Windows Environment Variables Cheat Sheet

| Variable | Example Value | Purpose |
|----------|---------------|---------|
| `%USERPROFILE%` | `C:\Users\Inter` | User home directory |
| `%USERNAME%` | `Inter` | Current username |
| `%COMPUTERNAME%` | `DESKTOP-ABC` | Computer name |
| `%APPDATA%` | `C:\Users\Inter\AppData\Roaming` | App data (roaming) |
| `%LOCALAPPDATA%` | `C:\Users\Inter\AppData\Local` | App data (local) |
| `%TEMP%` | `C:\Users\Inter\AppData\Local\Temp` | Temporary files |
| `%PROGRAMFILES%` | `C:\Program Files` | 64-bit programs |
| `%PROGRAMFILES(X86)%` | `C:\Program Files (x86)` | 32-bit programs |
| `%WINDIR%` | `C:\Windows` | Windows directory |
| `%SYSTEMDRIVE%` | `C:` | System drive |
| `%PATH%` | `C:\Windows;C:\Windows\System32;...` | Executable search paths |

### 4.5 Special Variable Syntax Cheat Sheet

| Syntax | Meaning | Example |
|--------|---------|---------|
| `%0` | Script name | `backup.bat` |
| `%1-%9` | Arguments 1-9 | `%1` = first argument |
| `%*` | All arguments | All args as single string |
| `%~1` | Argument 1, quotes stripped | `my file.txt` (no quotes) |
| `%~f1` | Argument 1, full path | `C:\Users\Inter\file.txt` |
| `%~d1` | Argument 1, drive only | `C:` |
| `%~p1` | Argument 1, path only | `\Users\Inter\` |
| `%~n1` | Argument 1, name only | `file` |
| `%~x1` | Argument 1, extension only | `.txt` |
| `%~dp0` | Script's drive + path | `C:\Scripts\` |
| `%~nx0` | Script's name + extension | `backup.bat` |
| `%CD%` | Current directory | `C:\Users\Inter\Projects` |
| `%ERRORLEVEL%` | Last exit code | `0` (success) or `1` (error) |
| `%RANDOM%` | Random number | `0` to `32767` |

---

## 5. Integration with Claude Code

### 5.1 When to Consult This Guide

**Pre-flight (before invoking Bash tool with .bat/.cmd)**:

- Script contains paths with spaces
- Script uses `cd` or changes directories
- Script calls other batch files
- Script uses loops or conditionals
- Script modifies variables inside loops
- Script involves file operations (copy, delete, move)
- Script needs error handling

**Error recovery (after batch failure)**:

- Check Section 3 (Error Recovery Reference)
- Match error message to Section 4.3 table
- Apply diagnostic commands from Section 3.2
- Follow recovery workflow from Section 3.3

### 5.2 Automatic Validation Triggers

Claude Code should automatically consult this guide when detecting:

**Critical Patterns**:

- `cd` without error checking
- Unquoted `%var%` with paths
- Using `script.bat` instead of `call script.bat`
- Using `exit` instead of `exit /b`
- Using `::` comments inside code blocks
- Modifying variables in loops without delayed expansion
- Paths with spaces not quoted

**Windows-Specific Issues**:

- `C:\Program Files` without quotes
- Mixed forward/back slashes in paths
- Missing `setlocal`/`endlocal` in scripts defining variables

### 5.3 Quick Lookup by Error Type

| Error Type | Section | Key Info |
|------------|---------|----------|
| Command not found | 3.1, 4.3 | Check PATH, use full path |
| Path not found | 3.1, 4.3 | Quote paths with spaces |
| Access denied | 3.1, 3.3 | Run as admin or use user dir |
| Syntax error | 3.1 | Quote special characters |
| Variable issues | 1.4, 1.5 | Use delayed expansion in loops |
| Script doesn't return | 1.8 | Use `call` for other scripts |
| Exit closes window | 2.10 | Use `exit /b` not `exit` |
| Comments break | 2.3 | Use `rem` not `::` in blocks |

### 5.4 Common Claude Code Batch Tasks

**Creating installation scripts**:

```batch
@echo off
setlocal

set "install_dir=%PROGRAMFILES%\MyApp"
echo Installing to: %install_dir%

if not exist "%install_dir%\" mkdir "%install_dir%"
if errorlevel 1 (
    echo ERROR: Failed to create directory
    exit /b 1
)

xcopy /E /I /Y ".\files\*" "%install_dir%\"
if errorlevel 1 (
    echo ERROR: Installation failed
    exit /b 1
)

echo Installation complete
endlocal
exit /b 0
```

**Creating build scripts**:

```batch
@echo off
setlocal EnableDelayedExpansion

set "build_config=%~1"
if "%build_config%"=="" set "build_config=Release"

echo Building configuration: %build_config%

call msbuild.exe Project.sln /p:Configuration=%build_config%
if errorlevel 1 (
    echo ERROR: Build failed
    exit /b 1
)

echo Build succeeded
endlocal
exit /b 0
```

**Creating launcher scripts**:

```batch
@echo off
rem Launcher for Python application

set "venv_python=%~dp0venv\Scripts\python.exe"

if not exist "%venv_python%" (
    echo ERROR: Virtual environment not found
    echo Please run setup.bat first
    exit /b 1
)

"%venv_python%" "%~dp0src\main.py" %*
exit /b %ERRORLEVEL%
```

---

## 6. Summary: Golden Rules

1. **ALWAYS quote variable assignments**: `set "var=value"` not `set var=value`
2. **ALWAYS quote paths with spaces**: `"%PROGRAMFILES%\App"`
3. **ALWAYS use PUSHD/POPD**: `pushd "C:\Path" || exit /b 1` not just `cd`
4. **ALWAYS use CALL for scripts**: `call script.bat` not just `script.bat`
5. **ALWAYS use EXIT /B**: `exit /b 0` not `exit` (closes window)
6. **ALWAYS check ERRORLEVEL**: `if errorlevel 1 exit /b 1` after critical commands
7. **ALWAYS use SETLOCAL/ENDLOCAL**: Wrap scripts to prevent variable pollution
8. **ALWAYS use REM for comments**: Not `::` (breaks in blocks)
9. **USE DELAYED EXPANSION in loops**: `setlocal EnableDelayedExpansion` + `!var!`
10. **USE %~dp0 for script paths**: Reference files relative to script location
11. **STRIP QUOTES from arguments**: `set "arg=%~1"` not `set "arg=%1"`
12. **QUOTE SPECIAL CHARACTERS**: `()`, `&`, `%` need escaping or quoting
13. **USE TIMEOUT not PING**: `timeout /t 5` for delays
14. **CHECK CRLF line endings**: Batch requires Windows line endings
15. **VALIDATE INPUT**: Check arguments and paths before processing

**Critical Safety Formula**:

```batch
@echo off
setlocal EnableDelayedExpansion

rem 1. Set variables with quotes
set "input=%~1"

rem 2. Validate input
if not exist "%input%" (
    echo ERROR: File not found
    exit /b 1
)

rem 3. Change directory safely
pushd "%~dp0" || exit /b 1

rem 4. Call other scripts with call
call helper.bat "%input%"
if errorlevel 1 (
    popd
    exit /b 1
)

rem 5. Clean up and exit
popd
endlocal
exit /b 0
```

---

**Last Updated**: 2025-11-17

**Character Count**: ~45,000 characters

**License**: MIT (Compiled from multiple sources, see header)

**Related Guides**: See `BASH_STYLE_GUIDE.md` for Git Bash scripting patterns

---

## Appendix: Batch vs Bash Quick Comparison

For developers familiar with Bash, here's a quick translation guide:

| Task | Bash | Batch |
|------|------|-------|
| **Variable assignment** | `var="value"` | `set "var=value"` |
| **Variable use** | `$var` or `"$var"` | `%var%` or `!var!` |
| **Command substitution** | `result=$(command)` | `for /f %%i in ('command') do set result=%%i` |
| **If statement** | `if [[ condition ]]; then` | `if condition (` |
| **For loop** | `for f in *.txt; do` | `for %%f in (*.txt) do (` |
| **Function** | `function_name() {` | `:function_name` |
| **Return value** | `return 0` | `exit /b 0` |
| **Error checking** | `command \|\| exit 1` | `command \|\| exit /b 1` |
| **Current directory** | `pwd` | `cd` or `%CD%` |
| **Script directory** | `$(dirname "$0")` | `%~dp0` |
| **Quote arguments** | `"$@"` | `%*` (no exact equivalent) |
| **Exit code** | `$?` | `%ERRORLEVEL%` |

**Philosophy difference**:

- **Bash**: Shell with scripting features (Unix philosophy)
- **Batch**: Command interpreter with scripting (DOS legacy)
- **Batch limitations**: More verbose, fewer built-in features, requires workarounds for common tasks
- **Batch advantages**: Native Windows, no dependencies, works everywhere Windows does
