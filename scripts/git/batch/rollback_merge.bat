@echo off
REM rollback_merge.bat
REM Emergency rollback for merge operations
REM Safely reverts main branch to pre-merge state

setlocal enabledelayedexpansion

REM [Guide 1.3] Ensure execution from project root
pushd "%~dp0..\.." || (
    echo ERROR: Cannot find project root
    exit /b 1
)

echo === Emergency Merge Rollback ===
echo.

REM [1/5] Verify current branch
echo [1/5] Verifying current branch...
for /f "tokens=*" %%i in ('git branch --show-current') do set "CURRENT_BRANCH=%%i"
echo Current branch: !CURRENT_BRANCH!

if not "!CURRENT_BRANCH!" == "main" (
    echo.
    echo ✗ ERROR: Not on main branch
    echo This script should only be run from main branch
    echo.
    echo Current branch: !CURRENT_BRANCH!
    echo Expected: main
    echo.
    echo Please checkout main first: git checkout main
    popd
    exit /b 1
)
echo ✓ On main branch
echo.

REM [2/5] Check for uncommitted changes
echo [2/5] Checking for uncommitted changes...
git diff-index --quiet HEAD --
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ⚠ WARNING: Uncommitted changes detected
    echo.
    git status --short
    echo.
    echo These changes will be LOST during rollback!
    echo.
    set /p "CONTINUE=Continue anyway? (yes/no): "
    if /i not "!CONTINUE!" == "yes" (
        echo.
        echo Rollback cancelled
        echo Please commit or stash changes first
        popd
        exit /b 1
    )
) else (
    echo ✓ No uncommitted changes
)
echo.

REM [3/5] Find rollback target
echo [3/5] Finding rollback target...
echo.

REM Look for pre-merge backup tags
git tag -l "pre-merge-backup-*" | sort /r >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Available backup tags:
    git tag -l "pre-merge-backup-*" | sort /r
    echo.
)

REM Show recent commits
echo Recent commits:
git log --oneline -5
echo.

REM Get latest merge commit
for /f "tokens=*" %%i in ('git log --merges --oneline -1') do set "LATEST_MERGE=%%i"
if defined LATEST_MERGE (
    echo Latest merge commit: !LATEST_MERGE!
    echo.
)

REM [4/5] Choose rollback method
echo [4/5] Choose rollback method:
echo.
echo Available options:
echo   1. Rollback to latest pre-merge backup tag (recommended)
echo   2. Rollback to commit before latest merge (HEAD~1)
echo   3. Rollback to specific commit hash
echo   4. Cancel rollback
echo.

set /p "ROLLBACK_CHOICE=Select option (1-4): "

if "!ROLLBACK_CHOICE!" == "1" (
    REM Find latest backup tag
    REM [CRITICAL FIX] Replaced Unix 'head -1' with Batch loop
    set "ROLLBACK_TARGET="
    for /f "tokens=*" %%i in ('git tag -l "pre-merge-backup-*" ^| sort /r') do (
        if not defined ROLLBACK_TARGET set "ROLLBACK_TARGET=%%i"
    )
    
    if not defined ROLLBACK_TARGET (
        echo.
        echo ✗ ERROR: No backup tags found
        echo Please use option 2 or 3
        popd
        exit /b 1
    )
    echo.
    echo Rollback target: !ROLLBACK_TARGET!
) else if "!ROLLBACK_CHOICE!" == "2" (
    set "ROLLBACK_TARGET=HEAD~1"
    echo.
    echo Rollback target: HEAD~1 (previous commit)
) else if "!ROLLBACK_CHOICE!" == "3" (
    echo.
    set /p "ROLLBACK_TARGET=Enter commit hash: "
    echo.
    REM Verify commit exists
    git rev-parse !ROLLBACK_TARGET! >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo ✗ ERROR: Invalid commit hash: !ROLLBACK_TARGET!
        popd
        exit /b 1
    )
    echo Rollback target: !ROLLBACK_TARGET!
) else if "!ROLLBACK_CHOICE!" == "4" (
    echo.
    echo Rollback cancelled
    popd
    exit /b 0
) else (
    echo.
    echo ✗ ERROR: Invalid choice: !ROLLBACK_CHOICE!
    popd
    exit /b 1
)

REM [5/5] Execute rollback
echo.
echo [5/5] Executing rollback...
echo.
echo ⚠ WARNING: This will permanently reset main branch to:
for /f "tokens=*" %%i in ('git log !ROLLBACK_TARGET! --oneline -1') do echo   %%i
echo.
echo All commits after this point will be lost!
echo.

set /p "CONFIRM=Type 'ROLLBACK' to confirm: "
if not "!CONFIRM!" == "ROLLBACK" (
    echo.
    echo Rollback cancelled
    popd
    exit /b 0
)

echo.
echo Executing: git reset --hard !ROLLBACK_TARGET!
git reset --hard !ROLLBACK_TARGET!

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ✗ Rollback failed
    echo Please manually reset: git reset --hard !ROLLBACK_TARGET!
    popd
    exit /b 1
)

echo.
echo ====================================
echo ✓ ROLLBACK SUCCESSFUL
echo ====================================
echo.
echo Summary:
for /f "tokens=*" %%i in ('git log --oneline -1') do echo   Current HEAD: %%i
echo.
echo Next steps:
echo   1. Verify rollback: git log --oneline -5
echo   2. If correct, force push: git push origin main --force-with-lease
echo   3. If issues, contact maintainer
echo.
echo   ⚠ WARNING: Force push will rewrite remote history!
echo.

endlocal
popd
exit /b 0