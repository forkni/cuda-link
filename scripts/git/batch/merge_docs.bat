@echo off
REM merge_docs.bat
REM Documentation-only merge from development to main

setlocal enabledelayedexpansion

REM [Guide 1.3] Ensure execution from project root
pushd "%~dp0..\.." || (
    echo ERROR: Cannot find project root
    exit /b 1
)

echo === Documentation Merge: development ^→ main ===
echo.

REM [1/7] Run validation
echo [1/7] Running pre-merge validation...
call "scripts\git\validate_branches.bat"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ✗ Validation failed - aborting documentation merge
    echo Please fix validation errors before retrying
    popd
    exit /b 1
)
echo.

REM [2/7] Store current branch and checkout main
echo [2/7] Switching to main branch...
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

REM [3/7] Check for documentation changes
echo [3/7] Checking for documentation changes...
git diff --name-only main development -- docs/ > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠ No documentation changes found between main and development
    echo.
    set /p "CONTINUE=Continue anyway? (yes/no): "
    if /i not "!CONTINUE!" == "yes" (
        echo.
        echo Documentation merge cancelled
        git checkout !ORIGINAL_BRANCH!
        popd
        exit /b 0
    )
)

echo Documentation files to be merged:
git diff --name-only main development -- docs/
echo.

REM [4/7] Check for non-documentation changes
echo [4/7] Checking for code changes...
set "HAS_CODE_CHANGES=0"

REM Count non-docs changes
REM [CRITICAL FIX] Use explicit path to Windows find.exe
for /f %%i in ('git diff --name-only main development ^| "%WINDIR%\System32\findstr.exe" /v "^docs/" ^| "%WINDIR%\System32\find.exe" /c /v ""') do set "NON_DOCS_COUNT=%%i"

if !NON_DOCS_COUNT! GTR 0 (
    echo ⚠ WARNING: Non-documentation changes detected
    echo.
    echo This merge will ONLY include docs/ changes.
    echo Other changes will remain on development branch.
    echo.
    git diff --name-only main development | "%WINDIR%\System32\findstr.exe" /v "^docs/"
    echo.
    set /p "CONFIRM=Proceed with docs-only merge? (yes/no): "
    if /i not "!CONFIRM!" == "yes" (
        echo.
        echo Documentation merge cancelled
        git checkout !ORIGINAL_BRANCH!
        popd
        exit /b 0
    )
) else (
    echo ✓ No code changes detected (docs-only merge)
)
echo.

REM [5/7] Create pre-merge backup tag
echo [5/7] Creating pre-merge backup tag...
for /f "usebackq" %%i in (`powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"`) do set "datetime=%%i"
set "BACKUP_TAG=pre-docs-merge-%datetime%"
git tag %BACKUP_TAG%
if %ERRORLEVEL% EQU 0 (
    echo ✓ Created backup tag: %BACKUP_TAG%
) else (
    echo ⚠ Warning: Could not create backup tag
)
echo.

REM [6/7] Merge documentation changes
echo [6/7] Merging documentation from development...
echo.

REM Strategy: Checkout docs/ from development branch
git checkout development -- docs/

if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to checkout documentation from development
    git checkout !ORIGINAL_BRANCH!
    popd
    exit /b 1
)

REM Check if there are changes
git diff --cached --quiet
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ⚠ No documentation changes to merge (docs already in sync)
    git checkout !ORIGINAL_BRANCH!
    popd
    exit /b 0
)

echo ✓ Documentation changes staged
echo Staged files:
git diff --cached --name-only
echo.

REM [7/7] Commit the merge
echo [7/7] Committing documentation merge...

git commit -m "docs: Sync documentation from development" -m "- Updated docs/ directory from development branch" -m "- Docs-only update (no code changes)" -m "- Backup tag: %BACKUP_TAG%"

if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to commit documentation merge
    echo.
    echo To abort: git merge --abort
    popd
    exit /b 1
)

echo.
echo ====================================
echo ✓ DOCUMENTATION MERGE SUCCESSFUL
echo ====================================
echo.
echo Summary:
for /f "tokens=*" %%i in ('git log -1 --oneline') do echo   Latest commit: %%i
echo   Backup tag: %BACKUP_TAG%
echo.
echo Merged files:
git diff --name-only HEAD~1 HEAD
echo.
echo Next steps:
echo   1. Review changes: git show HEAD
echo   2. Test documentation: [review rendered docs]
echo   3. Push to remote: git push origin main
echo.
echo   If issues found:
echo   - Rollback: git reset --hard %BACKUP_TAG%
echo   - Or use: scripts\git\rollback_merge.bat
echo.

endlocal
popd
exit /b 0