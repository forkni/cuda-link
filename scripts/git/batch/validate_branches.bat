@echo off
REM validate_branches.bat
REM Pre-merge validation for dual-branch workflow
REM Ensures safe merge conditions before merging development → main

setlocal enabledelayedexpansion

REM [Guide 1.3] Ensure execution from project root
pushd "%~dp0..\.." || (
    echo ERROR: Cannot find project root
    exit /b 1
)

echo === Branch Validation ===
echo.

REM Initialize validation state
set "VALIDATION_PASSED=1"
set "ERROR_COUNT=0"

REM [1/9] Check if we're in a git repository
echo [1/9] Checking git repository...
git rev-parse --git-dir >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ✗ FAIL: Not in a git repository
    set "VALIDATION_PASSED=0"
    set /a ERROR_COUNT+=1
) else (
    echo ✓ PASS: Git repository detected
)
echo.

REM [2/9] Check if main branch exists
echo [2/9] Checking main branch exists...
git show-ref --verify --quiet refs/heads/main
if %ERRORLEVEL% NEQ 0 (
    echo ✗ FAIL: main branch does not exist
    set "VALIDATION_PASSED=0"
    set /a ERROR_COUNT+=1
) else (
    echo ✓ PASS: main branch exists
)
echo.

REM [3/9] Check if development branch exists
echo [3/9] Checking development branch exists...
git show-ref --verify --quiet refs/heads/development
if %ERRORLEVEL% NEQ 0 (
    echo ✗ FAIL: development branch does not exist
    set "VALIDATION_PASSED=0"
    set /a ERROR_COUNT+=1
) else (
    echo ✓ PASS: development branch exists
)
echo.

REM [4/9] Check for uncommitted changes on current branch
echo [4/9] Checking for uncommitted changes...
git diff-index --quiet HEAD --
if %ERRORLEVEL% NEQ 0 (
    echo ✗ FAIL: Uncommitted changes detected
    echo.
    git status --short
    echo.
    echo Please commit or stash changes before merging
    set "VALIDATION_PASSED=0"
    set /a ERROR_COUNT+=1
) else (
    echo ✓ PASS: No uncommitted changes
)
echo.

REM [5/9] Check if .gitattributes exists
echo [5/9] Checking .gitattributes exists...
if not exist ".gitattributes" (
    echo ✗ FAIL: .gitattributes file missing
    echo Please create .gitattributes with merge strategies
    set "VALIDATION_PASSED=0"
    set /a ERROR_COUNT+=1
) else (
    echo ✓ PASS: .gitattributes exists
)
echo.

REM [6/9] Check if merge.ours driver is configured
echo [6/9] Checking merge.ours driver configuration...
git config --get merge.ours.driver >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ✗ FAIL: merge.ours driver not configured
    echo Please run: git config --global merge.ours.driver true
    set "VALIDATION_PASSED=0"
    set /a ERROR_COUNT+=1
) else (
    for /f "tokens=*" %%i in ('git config --get merge.ours.driver') do set "MERGE_DRIVER=%%i"
    echo ✓ PASS: merge.ours driver = !MERGE_DRIVER!
)
echo.

REM [7/9] Check if branches are up to date with remote
echo [7/9] Checking remote sync status...
git fetch origin --quiet 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ⚠ WARNING: Could not fetch from remote
    echo Skipping remote sync check
) else (
    REM Check main branch
    for /f "tokens=*" %%i in ('git rev-parse main') do set "LOCAL_MAIN=%%i"
    for /f "tokens=*" %%i in ('git rev-parse origin/main') do set "REMOTE_MAIN=%%i"

    if "!LOCAL_MAIN!" NEQ "!REMOTE_MAIN!" (
        echo ⚠ WARNING: main branch not in sync with origin/main
        echo   Local:  !LOCAL_MAIN!
        echo   Remote: !REMOTE_MAIN!
        echo   Recommendation: git pull origin main
    ) else (
        echo ✓ PASS: main branch in sync with remote
    )

    REM Check development branch
    for /f "tokens=*" %%i in ('git rev-parse development') do set "LOCAL_DEV=%%i"
    for /f "tokens=*" %%i in ('git rev-parse origin/development') do set "REMOTE_DEV=%%i"

    if "!LOCAL_DEV!" NEQ "!REMOTE_DEV!" (
        echo ⚠ WARNING: development branch not in sync with origin/development
        echo   Local:  !LOCAL_DEV!
        echo   Remote: !REMOTE_DEV!
        echo   Recommendation: git pull origin development
    ) else (
        echo ✓ PASS: development branch in sync with remote
    )
)
echo.

REM [8/9] Verify merge attribute detection
echo [8/9] Verifying merge attributes...
git check-attr merge tests/conftest.py >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠ WARNING: Could not check merge attributes
    echo tests/ directory may not exist yet
) else (
    for /f "tokens=3" %%i in ('git check-attr merge tests/conftest.py') do set "MERGE_ATTR=%%i"
    if "!MERGE_ATTR!" == "ours" (
        echo ✓ PASS: Merge attributes working tests/conftest.py: merge: ours
    ) else (
        echo ✗ FAIL: Merge attributes not working correctly
        echo   Expected: tests/conftest.py: merge: ours
        echo   Got:      tests/conftest.py: merge: !MERGE_ATTR!
        set "VALIDATION_PASSED=0"
        set /a ERROR_COUNT+=1
    )
)
echo.

REM [9/9] Validate CI documentation policy compliance
echo [9/9] Checking CI documentation policy compliance...

REM Get current branch
for /f "tokens=*" %%i in ('git branch --show-current') do set "CURRENT_BRANCH=%%i"

REM Allowed docs list from .github/workflows/branch-protection.yml
set "ALLOWED_DOCS=BENCHMARKS.md claude_code_config.md GIT_WORKFLOW.md HYBRID_SEARCH_CONFIGURATION_GUIDE.md INSTALLATION_GUIDE.md MCP_TOOLS_REFERENCE.md MODEL_MIGRATION_GUIDE.md PYTORCH_COMPATIBILITY.md"

REM Only check if we're validating FROM development (about to merge TO main)
if "!CURRENT_BRANCH!" == "development" (
    REM Check for docs that would be added to main during merge
    git fetch origin main --quiet 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo ⚠ WARNING: Could not fetch origin/main
        echo Skipping CI policy check
    ) else (
        set "CI_VIOLATION_FOUND=0"

        REM Get docs that exist in development but not in main
        for /f %%f in ('git diff --name-only origin/main...HEAD ^| "%WINDIR%\System32\findstr.exe" /C:"docs/" 2^>nul') do (
            REM Check if file would be added (not just modified)
            git ls-tree origin/main %%f >nul 2>&1
            if %ERRORLEVEL% NEQ 0 (
                REM File doesn't exist on main - would be added during merge
                set "DOC_FILE=%%~nxf"

                REM Check if doc is in allowed list
                echo !ALLOWED_DOCS! | "%WINDIR%\System32\findstr.exe" /C:"!DOC_FILE!" >nul
                if %ERRORLEVEL% NEQ 0 (
                    echo ✗ FAIL: %%f would violate CI policy
                    echo    This doc is not in the allowed list for main branch
                    echo    Add to .gitattributes with merge=ours strategy
                    set "CI_VIOLATION_FOUND=1"
                    set "VALIDATION_PASSED=0"
                    set /a ERROR_COUNT+=1
                )
            )
        )

        if !CI_VIOLATION_FOUND! EQU 0 (
            echo ✓ PASS: No CI policy violations detected
        ) else (
            echo.
            echo Allowed docs on main: !ALLOWED_DOCS!
        )
    )
) else (
    echo ⚠ INFO: Running from !CURRENT_BRANCH! branch
    echo CI policy check only runs when validating from development branch
    echo ✓ PASS: CI policy check skipped
)
echo.

REM Final validation result
echo ====================================
if %VALIDATION_PASSED% EQU 1 (
    echo ✓ VALIDATION PASSED
    echo Safe to proceed with merge
    echo.
    echo Next steps:
    echo   1. Run: scripts\git\merge_with_validation.bat
    echo   2. Review merge results
    echo   3. Push to remote if successful
    popd
    exit /b 0
) else (
    echo ✗ VALIDATION FAILED
    echo Found !ERROR_COUNT! error(s)
    echo.
    echo Please fix the issues above before merging
    echo DO NOT proceed with merge until all checks pass
    popd
    exit /b 1
)