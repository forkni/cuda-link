@echo off
setlocal enabledelayedexpansion
pushd "%~dp0" || exit /b 1

REM build_wheel.cmd - Build cuda-link Python wheel for distribution
REM
REM Uses the resolved Python interpreter (preferring 'py -3' Windows launcher)
REM with PyPA's PEP 517 isolated build env: <python> -m build --wheel
REM Output: dist\cuda_link-<version>-py3-none-any.whl  (version from pyproject.toml)
REM
REM Usage:
REM   Double-click or run from any terminal:
REM     build_wheel.cmd
REM
REM   Then install into any Python environment:
REM     pip install "dist\cuda_link-<version>-py3-none-any.whl"
REM     pip install "dist\cuda_link-<version>-py3-none-any.whl[torch]"
REM     pip install "dist\cuda_link-<version>-py3-none-any.whl[all]"

echo ========================================
echo  cuda-link Wheel Builder
echo ========================================
echo.

REM ----------------------------------------
REM [1/4] Resolve Python interpreter
REM ----------------------------------------
echo [1/4] Resolving Python interpreter...

REM Prefer 'py -3' (Windows Python Launcher) -- bypasses Microsoft Store stub.
REM Fall back to 'python' on PATH if launcher isn't installed.
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)

if not defined PY (
    echo.
    echo [ERROR] No Python interpreter found.
    echo         Install Python 3.9 or newer from https://www.python.org/downloads/
    echo         Make sure either 'py' or 'python' resolves on PATH.
    goto :error
)

REM Reject Microsoft Store stub (sys.executable resolves under WindowsApps)
for /f "delims=" %%e in ('!PY! -c "import sys; print(sys.executable)" 2^>nul') do set "PY_EXE=%%e"
echo !PY_EXE! | findstr /i "\\WindowsApps\\" >nul
if not errorlevel 1 (
    echo.
    echo [ERROR] Detected Microsoft Store Python stub:
    echo           !PY_EXE!
    echo         This is a placeholder, not a usable Python install.
    echo         Install Python from https://www.python.org/downloads/, then
    echo         disable the App Execution Alias in Windows Settings:
    echo           Settings -^> Apps -^> Advanced app settings -^> App execution aliases
    goto :error
)

REM Enforce pyproject.toml's requires-python = ">=3.9"
!PY! -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)"
if errorlevel 1 (
    echo.
    echo [ERROR] cuda-link requires Python 3.9 or newer. Detected:
    !PY! --version
    goto :error
)

for /f "tokens=*" %%v in ('!PY! --version 2^>^&1') do set "PYVER=%%v"
echo   !PYVER!
echo   !PY_EXE!
echo.

REM ----------------------------------------
REM [1.5/4] Sync td_exporter/CUDAIPCWrapper.py from canonical source
REM ----------------------------------------
echo [1.5/4] Syncing CUDAIPCWrapper.py...

!PY! scripts\sync_td_wrapper.py
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to sync td_exporter/CUDAIPCWrapper.py from src/cuda_link/cuda_ipc_wrapper.py.
    goto :error
)
echo.

REM ----------------------------------------
REM [2/4] Ensure PyPA build frontend
REM ----------------------------------------
echo [2/4] Ensuring build tools...

!PY! -m pip install --upgrade build --quiet
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install/upgrade the 'build' package.
    echo         Try manually: !PY! -m pip install --upgrade build
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

!PY! -m build --wheel
if errorlevel 1 (
    echo.
    echo [ERROR] Wheel build failed.
    echo         Check the output above for details.
    echo         Common fixes:
    echo           - Ensure pyproject.toml is valid
    echo           - Run: !PY! -m pip install --upgrade setuptools build
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

REM Get wheel file size in KB via PowerShell
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
