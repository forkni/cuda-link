@echo off
setlocal enabledelayedexpansion
pushd "%~dp0" || exit /b 1

REM build_wheel.cmd - Build cuda-link Python wheel for distribution
REM
REM Uses: python -m build --wheel  (PyPA PEP 517 standard, isolated build env)
REM Output: dist\cuda_link-0.7.0-py3-none-any.whl
REM
REM Usage:
REM   Double-click or run from any terminal:
REM     build_wheel.cmd
REM
REM   Then install into any Python environment:
REM     pip install "dist\cuda_link-0.7.0-py3-none-any.whl"
REM     pip install "dist\cuda_link-0.7.0-py3-none-any.whl[torch]"
REM     pip install "dist\cuda_link-0.7.0-py3-none-any.whl[all]"

echo ========================================
echo  cuda-link Wheel Builder  v0.7.0
echo ========================================
echo.

REM ----------------------------------------
REM [1/4] Validate Python
REM ----------------------------------------
echo [1/4] Checking Python...

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python not found on PATH.
    echo         Ensure Python 3.9+ is installed and added to PATH.
    echo         Download from: https://www.python.org/downloads/
    goto :error
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo   !PYVER!
echo.

REM ----------------------------------------
REM [2/4] Ensure PyPA build frontend
REM ----------------------------------------
echo [2/4] Ensuring build tools...

python -m pip install --upgrade build --quiet
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install/upgrade the 'build' package.
    echo         Try manually: python -m pip install --upgrade build
    goto :error
)

echo   build package ready
echo.

REM ----------------------------------------
REM [3/4] Clean old build artifacts
REM ----------------------------------------
echo [3/4] Cleaning old artifacts...

set "cleaned=0"

if exist "dist\" (
    rmdir /s /q "dist"
    if !errorlevel! equ 0 (
        echo   Removed dist\
        set /a cleaned+=1
    ) else (
        echo   [WARN] Could not remove dist\ - continuing anyway
    )
)

if exist "build\" (
    rmdir /s /q "build"
    if !errorlevel! equ 0 (
        echo   Removed build\
        set /a cleaned+=1
    ) else (
        echo   [WARN] Could not remove build\ - continuing anyway
    )
)

if exist "src\cuda_link.egg-info\" (
    rmdir /s /q "src\cuda_link.egg-info"
    if !errorlevel! equ 0 (
        echo   Removed src\cuda_link.egg-info\
        set /a cleaned+=1
    ) else (
        echo   [WARN] Could not remove egg-info - continuing anyway
    )
)

if !cleaned! equ 0 (
    echo   Nothing to clean
)
echo.

REM ----------------------------------------
REM [4/4] Build the wheel
REM ----------------------------------------
echo [4/4] Building wheel...
echo.

python -m build --wheel
if errorlevel 1 (
    echo.
    echo [ERROR] Wheel build failed.
    echo         Check the output above for details.
    echo         Common fixes:
    echo           - Ensure pyproject.toml is valid
    echo           - Run: python -m pip install --upgrade setuptools build
    goto :error
)

REM Find the built wheel file
set "WHEEL_FILE="
for /f "tokens=*" %%f in ('dir /b /o-d "dist\*.whl" 2^>nul') do (
    if not defined WHEEL_FILE set "WHEEL_FILE=%%f"
)

if not defined WHEEL_FILE (
    echo.
    echo [ERROR] Build reported success but no .whl file found in dist\
    goto :error
)

REM Get wheel file size (in KB)
set "WHEEL_SIZE=unknown"
for /f "tokens=3" %%s in ('dir "dist\!WHEEL_FILE!" 2^>nul ^| findstr /r "[0-9]"') do (
    set "RAW_SIZE=%%s"
)
REM Convert bytes to KB via PowerShell for clean output
for /f %%k in ('powershell -Command "[math]::Ceiling((Get-Item 'dist\!WHEEL_FILE!').Length / 1KB)"') do set "WHEEL_KB=%%k"

echo.
echo ========================================
echo  BUILD COMPLETE
echo ========================================
echo.
echo   Wheel: dist\!WHEEL_FILE!
echo   Size:  !WHEEL_KB! KB
echo.
echo ----------------------------------------
echo  Install into any Python environment:
echo ----------------------------------------
echo.
echo   pip install "dist\!WHEEL_FILE!"
echo.
echo   With optional dependencies:
echo   pip install "dist\!WHEEL_FILE![torch]"
echo   pip install "dist\!WHEEL_FILE![numpy]"
echo   pip install "dist\!WHEEL_FILE![cupy]"
echo   pip install "dist\!WHEEL_FILE![all]"
echo.
echo   Force reinstall (update existing):
echo   pip install --force-reinstall "dist\!WHEEL_FILE!"
echo.
echo ========================================
echo.
goto :done

:error
echo.
echo [FAILED] Build did not complete successfully.
echo.
popd
endlocal
pause
exit /b 1

:done
popd
endlocal
pause
exit /b 0
