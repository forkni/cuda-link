@echo off
REM install_hooks.bat
REM Install Git hooks from .githooks/ to .git/hooks/

echo === Git Hooks Installer ===
echo.

REM Check if .git directory exists
if not exist ".git\" (
    echo ERROR: .git directory not found
    echo This script must be run from the repository root
    exit /b 1
)

REM Check if .githooks directory exists
if not exist ".githooks\" (
    echo ERROR: .githooks directory not found
    echo This repository doesn't have hook templates
    exit /b 1
)

echo Installing Git hooks...
echo.

REM Install pre-commit hook
if exist ".githooks\pre-commit" (
    echo Installing pre-commit hook...
    copy /Y ".githooks\pre-commit" ".git\hooks\pre-commit" >nul
    if %ERRORLEVEL% EQU 0 (
        echo ✓ pre-commit hook installed
    ) else (
        echo ✗ Failed to install pre-commit hook
        exit /b 1
    )
) else (
    echo ⚠ pre-commit template not found, skipping
)

echo.
echo ====================================
echo ✓ HOOKS INSTALLED SUCCESSFULLY
echo ====================================
echo.

echo The following hooks are now active:
echo   - pre-commit: File validation + code quality checks
echo.

echo What this hook does:
echo   1. Prevents committing local-only files
echo   2. Validates documentation files
echo   3. Checks code quality (Python files)
echo   4. Offers auto-fix for lint errors
echo.

echo To bypass hooks temporarily (not recommended):
echo   git commit --no-verify
echo.

echo To uninstall hooks:
echo   del .git\hooks\pre-commit
echo.

pause
