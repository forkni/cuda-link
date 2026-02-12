@echo off
REM cherry_pick_commits.bat
REM Cherry-pick specific commits from development to main

setlocal enabledelayedexpansion

REM [Guide 1.3] Ensure execution from project root
pushd "%~dp0..\.." || (
    echo ERROR: Cannot find project root
    exit /b 1
)

echo === Cherry-Pick Commits: development ^→ main ===
echo.

REM [1/6] Run validation
echo [1/6] Running pre-merge validation...
call "scripts\git\validate_branches.bat"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ✗ Validation failed - aborting cherry-pick
    echo Please fix validation errors before retrying
    popd
    exit /b 1
)
echo.

REM [2/6] Store current branch and checkout main
echo [2/6] Switching to main branch...
for /f "tokens=*" %%i in ('git branch --show-current') do set "ORIGINAL_BRANCH=%%i"
echo Current branch: !ORIGINAL_BRANCH!

git checkout main
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to checkout main branch
    popd
    exit /b 1
)
echo ✓ Switched to main branch
echo.

REM [3/6] Show recent development commits
echo [3/6] Recent commits on development branch:
echo ====================================
git log development --oneline -20 --no-merges
echo ====================================
echo.

REM [4/6] Get commit hash to cherry-pick
echo [4/6] Select commit to cherry-pick...
echo.
set /p "COMMIT_HASH=Enter commit hash (or 'cancel' to abort): "

if /i "!COMMIT_HASH!" == "cancel" (
    echo.
    echo Cherry-pick cancelled
    git checkout !ORIGINAL_BRANCH!
    popd
    exit /b 0
)

REM Verify commit exists
git rev-parse !COMMIT_HASH! >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ✗ ERROR: Invalid commit hash: !COMMIT_HASH!
    git checkout !ORIGINAL_BRANCH!
    popd
    exit /b 1
)

echo.
echo Selected commit:
git log !COMMIT_HASH! --oneline -1
echo.
echo Commit details:
git show !COMMIT_HASH! --stat
echo.

REM Check if commit modifies excluded files
set "HAS_EXCLUDED_FILES=0"

git show !COMMIT_HASH! --name-only --format="" | "%WINDIR%\System32\findstr.exe" "^tests/" >nul
if %ERRORLEVEL% EQU 0 set "HAS_EXCLUDED_FILES=1"

git show !COMMIT_HASH! --name-only --format="" | "%WINDIR%\System32\findstr.exe" "^docs/TESTING_GUIDE.md" >nul
if %ERRORLEVEL% EQU 0 set "HAS_EXCLUDED_FILES=1"

git show !COMMIT_HASH! --name-only --format="" | "%WINDIR%\System32\findstr.exe" "pytest.ini" >nul
if %ERRORLEVEL% EQU 0 set "HAS_EXCLUDED_FILES=1"

if !HAS_EXCLUDED_FILES! EQU 1 (
    echo ⚠ WARNING: This commit modifies development-only files
    echo These files should not be on main branch:
    git show !COMMIT_HASH! --name-only --format="" | "%WINDIR%\System32\findstr.exe" "^tests/ ^docs/TESTING_GUIDE.md pytest.ini"
    echo.
    set /p "CONTINUE=Continue anyway? (yes/no): "
    if /i not "!CONTINUE!" == "yes" (
        echo.
        echo Cherry-pick cancelled
        git checkout !ORIGINAL_BRANCH!
        popd
        exit /b 0
    )
)

REM [5/6] Create backup tag
echo.
echo [5/6] Creating pre-cherry-pick backup...
for /f "usebackq" %%i in (`powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"`) do set "datetime=%%i"
set "BACKUP_TAG=pre-cherry-pick-%datetime%"
REM [Guide 1.4] Use delayed expansion for variables set in loops
git tag "!BACKUP_TAG!"
if %ERRORLEVEL% EQU 0 (
    echo ✓ Created backup tag: !BACKUP_TAG!
) else (
    echo ⚠ Warning: Could not create backup tag
)
echo.

REM [6/6] Cherry-pick the commit
echo [6/6] Cherry-picking commit...
echo Running: git cherry-pick !COMMIT_HASH!
echo.

git cherry-pick !COMMIT_HASH!

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ⚠ Cherry-pick conflicts detected
    echo.
    git status
    echo.
    echo Please resolve conflicts manually:
    echo   1. Edit conflicted files
    echo   2. git add ^<resolved files^>
    echo   3. git cherry-pick --continue
    echo.
    echo Or abort the cherry-pick:
    echo   git cherry-pick --abort
    echo   git checkout !ORIGINAL_BRANCH!
    echo.
    echo Backup available: git reset --hard %BACKUP_TAG%
    popd
    exit /b 1
)

echo.
echo ====================================
echo ✓ CHERRY-PICK SUCCESSFUL
echo ====================================
echo.
echo Summary:
for /f "tokens=*" %%i in ('git log -1 --oneline') do echo   Cherry-picked: %%i
echo   Original commit: !COMMIT_HASH!
echo   Backup tag: %BACKUP_TAG%
echo.
echo Next steps:
echo   1. Review changes: git show HEAD
echo   2. Test changes: [run your tests]
echo   3. Push to remote: git push origin main
echo.
echo   If issues found:
echo   - Rollback: git reset --hard %BACKUP_TAG%
echo   - Or use: scripts\git\rollback_merge.bat
echo.

endlocal
popd
exit /b 0