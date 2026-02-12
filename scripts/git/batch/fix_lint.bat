@echo off
REM fix_lint.bat
REM Auto-fix code quality issues

setlocal enabledelayedexpansion

REM ========================================
REM Parse Command Line Arguments
REM ========================================

set "NON_INTERACTIVE=0"
if "%~1"=="--non-interactive" set "NON_INTERACTIVE=1"

REM [Guide 1.3] Ensure execution from project root
pushd "%~dp0..\.." || (
    echo ERROR: Cannot find project root
    exit /b 1
)

REM ========================================
REM Detect Git Bash/MSYS Environment
REM ========================================
REM [BUGFIX] Prevent "Commit"/"Workflow" file creation (see docs/AUTOMATED_GIT_WORKFLOW.md)
REM Git Bash cannot properly execute batch scripts - variable scoping fails,
REM causing empty %LOGFILE% redirections that create files in project root

if defined MSYSTEM (
    echo [ERROR] This script must be run from Windows CMD, not Git Bash
    echo.
    echo Detected environment: MSYSTEM=%MSYSTEM%
    echo.
    echo Solutions:
    echo   1. Use Windows CMD: cmd.exe /c scripts\git\fix_lint.bat
    echo   2. Use shell script: scripts/git/fix_lint.sh ^(in Git Bash^)
    echo.
    popd 2>nul
    exit /b 1
)

REM ========================================
REM Initialize Mandatory Logging
REM ========================================

REM Initialize logging using shared utility (locale-independent timestamp)
call "%~dp0_common.bat" :InitLogging fix_lint

REM Validate logging initialization (catches cross-environment issues)
REM [BUGFIX] Defensive check for variable scoping failures (see docs/AUTOMATED_GIT_WORKFLOW.md)
if "%LOGFILE%"=="" (
    echo [ERROR] Failed to initialize logging - LOGFILE variable not set
    echo This typically happens when running batch scripts from Git Bash
    echo.
    echo Please run from Windows CMD: cmd.exe /c %~f0 %*
    popd
    exit /b 1
)
if "%REPORTFILE%"=="" (
    echo [ERROR] Failed to initialize logging - REPORTFILE variable not set
    popd
    exit /b 1
)

REM Initialize log file
echo ========================================= > "%LOGFILE%"
echo Fix Lint Log >> "%LOGFILE%"
echo ========================================= >> "%LOGFILE%"
echo Start Time: %date% %time% >> "%LOGFILE%"
echo. >> "%LOGFILE%"

echo === Code Quality Auto-Fixer ===
echo Automatically fixing lint issues...
echo [LOG] Workflow Log: %LOGFILE%
echo.
echo === Code Quality Auto-Fixer === >> "%LOGFILE%"
echo Automatically fixing lint issues... >> "%LOGFILE%"
echo. >> "%LOGFILE%"

REM Get shared lint exclusion patterns
call "%~dp0_common.bat" :GetLintExclusions

REM [1/3] Fix issues with Ruff
echo [1/3] Fixing code issues with ruff...
echo [1/3] Fixing code issues with ruff... >> "%LOGFILE%"
call ".venv\Scripts\ruff.exe" check . --fix %RUFF_EXCLUDE%
if %ERRORLEVEL% EQU 0 (
    echo [OK] ruff completed
    echo [OK] ruff completed >> "%LOGFILE%"
) else (
    echo [X] ruff found issues that require manual fixes
    echo [X] ruff found issues that require manual fixes >> "%LOGFILE%"
)
echo.

REM [2/3] Fix formatting with ruff format
echo [2/3] Fixing code formatting with ruff format...
echo [2/3] Fixing code formatting with ruff format... >> "%LOGFILE%"
call ".venv\Scripts\ruff.exe" format . %RUFF_EXCLUDE%
if %ERRORLEVEL% EQU 0 (
    echo [OK] ruff format completed
    echo [OK] ruff format completed >> "%LOGFILE%"
) else (
    echo [X] ruff format failed
    echo [X] ruff format failed >> "%LOGFILE%"
)
echo.

REM [3/3] Fix markdown issues
echo [3/3] Fixing markdown formatting with markdownlint...
echo [3/3] Fixing markdown formatting with markdownlint... >> "%LOGFILE%"
call markdownlint-cli2 --fix %MD_PATTERNS%
if %ERRORLEVEL% EQU 0 (
    echo [OK] markdownlint completed
    echo [OK] markdownlint completed >> "%LOGFILE%"
) else (
    echo [X] markdownlint found issues that require manual fixes
    echo [X] markdownlint found issues that require manual fixes >> "%LOGFILE%"
)
echo.

REM Final verification
echo ====================================
echo Running final verification...
echo ====================================  >> "%LOGFILE%"
echo Running final verification... >> "%LOGFILE%"
echo.

REM [Guide 1.8] Call script using quoted path
call "scripts\git\check_lint.bat"

echo End Time: %date% %time% >> "%LOGFILE%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ====================================
    echo [OK] ALL ISSUES FIXED
    echo Code is ready to commit!
    echo ====================================
    echo. >> "%LOGFILE%"
    echo ==================================== >> "%LOGFILE%"
    echo STATUS: SUCCESS >> "%LOGFILE%"
    echo ==================================== >> "%LOGFILE%"
    echo [LOG] Log saved: %LOGFILE%
    echo.
    echo Next steps:
    echo   1. Review changes: git diff
    echo   2. Stage changes: git add .
    echo   3. Commit: scripts\git\commit_enhanced.bat "message"
    echo.
    popd
    exit /b 0
) else (
    echo.
    echo ====================================
    echo [!] SOME ISSUES REMAIN
    echo ====================================
    echo. >> "%LOGFILE%"
    echo ==================================== >> "%LOGFILE%"
    echo STATUS: PARTIAL - Manual fixes required >> "%LOGFILE%"
    echo ==================================== >> "%LOGFILE%"
    echo [LOG] Log saved: %LOGFILE%
    echo.
    echo Some issues could not be auto-fixed.
    echo Please review the errors above and fix manually.
    echo.
    popd
    exit /b 1
)