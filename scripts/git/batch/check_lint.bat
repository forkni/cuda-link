@echo off
REM check_lint.bat
REM Quick code quality checker - runs all linting tools without making changes

setlocal enabledelayedexpansion

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
    echo   1. Use Windows CMD: cmd.exe /c scripts\git\check_lint.bat
    echo   2. Use shell script: scripts/git/check_lint.sh ^(in Git Bash^)
    echo.
    endlocal
    popd 2>nul
    exit /b 1
)

REM ========================================
REM Initialize Mandatory Logging
REM ========================================

REM Initialize logging using shared utility (locale-independent timestamp)
call "%~dp0_common.bat" :InitLogging check_lint

REM Validate logging initialization (catches cross-environment issues)
REM [BUGFIX] Defensive check for variable scoping failures (see docs/AUTOMATED_GIT_WORKFLOW.md)
if "%LOGFILE%"=="" (
    echo [ERROR] Failed to initialize logging - LOGFILE variable not set
    echo This typically happens when running batch scripts from Git Bash
    echo.
    echo Please run from Windows CMD: cmd.exe /c %~f0 %*
    endlocal
    popd
    exit /b 1
)
if "%REPORTFILE%"=="" (
    echo [ERROR] Failed to initialize logging - REPORTFILE variable not set
    endlocal
    popd
    exit /b 1
)

REM Initialize log file
echo ========================================= > "%LOGFILE%"
echo Lint Validation Log >> "%LOGFILE%"
echo ========================================= >> "%LOGFILE%"
echo Start Time: %date% %time% >> "%LOGFILE%"
echo. >> "%LOGFILE%"

echo === Code Quality Checker ===
echo Running lint checks (read-only)...
echo [LOG] Workflow Log: %LOGFILE%
echo.
echo === Code Quality Checker === >> "%LOGFILE%"
echo Running lint checks (read-only)... >> "%LOGFILE%"
echo. >> "%LOGFILE%"

REM Get shared lint exclusion patterns
call "%~dp0_common.bat" :GetLintExclusions

set "ERRORS_FOUND=0"

REM [1/3] Check with Ruff
echo [1/3] Running ruff check...
call ".venv\Scripts\ruff.exe" check . %RUFF_EXCLUDE%
if %ERRORLEVEL% NEQ 0 (
    echo [X] Ruff found issues
    set "ERRORS_FOUND=1"
) else (
    echo [OK] Ruff passed
)
echo.

REM [2/3] Check formatting with ruff format
echo [2/3] Running ruff format check...
call ".venv\Scripts\ruff.exe" format --check . %RUFF_EXCLUDE%
if %ERRORLEVEL% NEQ 0 (
    echo [X] Ruff format found formatting issues
    set "ERRORS_FOUND=1"
) else (
    echo [OK] Ruff format passed
)
echo.

REM [3/3] Check with markdownlint
echo [3/3] Running markdownlint...
call markdownlint-cli2 %MD_PATTERNS%
if %ERRORLEVEL% NEQ 0 (
    echo [X] Markdownlint found issues
    set "ERRORS_FOUND=1"
) else (
    echo [OK] Markdownlint passed
)
echo.

REM Summary
echo ====================================
echo End Time: %date% %time% >> "%LOGFILE%"
if !ERRORS_FOUND! EQU 0 (
    echo [OK] ALL CHECKS PASSED
    echo Code is ready to commit!
    echo ====================================
    echo. >> "%LOGFILE%"
    echo ====================================  >> "%LOGFILE%"
    echo STATUS: SUCCESS >> "%LOGFILE%"
    echo ====================================  >> "%LOGFILE%"
    echo [LOG] Log saved: %LOGFILE%
    endlocal
    popd
    exit /b 0
) else (
    echo [X] ERRORS FOUND
    echo.
    echo Fix issues with one of these options:
    echo   1. Auto-fix: scripts\git\fix_lint.bat
    echo   2. Manual fix: Review errors above
    echo ====================================
    echo. >> "%LOGFILE%"
    echo ==================================== >> "%LOGFILE%"
    echo STATUS: FAILED >> "%LOGFILE%"
    echo ==================================== >> "%LOGFILE%"
    echo [LOG] Log saved: %LOGFILE%
    endlocal
    popd
    exit /b 1
)