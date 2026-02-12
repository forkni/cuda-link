@echo off
setlocal enabledelayedexpansion
pushd "%~dp0" || exit /b 1

REM Batch script to recursively remove __pycache__ folders and Claude Code temp files
REM Run this script from the project root to clean Python cache and temporary files

echo ========================================
echo  Python Cache & Temp Files Cleanup
echo  claude-context-local
echo ========================================
echo.
echo Searching for __pycache__ folders...
echo Current directory: %CD%
echo.

set "count=0"

REM Recursively find and delete all __pycache__ folders
for /d /r %%d in (__pycache__) do (
    if exist "%%d" (
        echo Removing: %%d
        rmdir /s /q "%%d"
        if !errorlevel! equ 0 (
            set /a count+=1
        ) else (
            echo WARNING: Could not remove %%d
        )
    )
)

REM Clean up Claude Code temporary files (tmpclaude-*-cwd)
echo.
echo Searching for Claude Code temporary files...

set "tempcount=0"

for /r %%f in (tmpclaude-*-cwd) do (
    if exist "%%f" (
        echo Removing temp file: %%f
        del /f /q "%%f"
        if !errorlevel! equ 0 (
            set /a tempcount+=1
        ) else (
            echo WARNING: Could not remove %%f
        )
    )
)

echo.
echo ========================================
echo Cleanup complete!
echo Total __pycache__ folders removed: !count!
echo Total temp files removed: !tempcount!
echo ========================================
echo.
pause
popd
endlocal
exit /b 0
