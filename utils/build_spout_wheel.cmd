@echo off
setlocal enabledelayedexpansion
pushd "%~dp0.." || exit /b 1

REM build_spout_wheel.cmd - Build cuda-link-spout native wheel for distribution
REM
REM Compiles the _spout_bridge pybind11 module (CUDA + D3D11 + Spout2 SDK) into a
REM platform wheel and places it in dist\ alongside the core wheel:
REM   dist\cuda_link_spout-<version>-cp311-cp311-win_amd64.whl
REM
REM Requirements:
REM   - CUDA Toolkit 12.x or 13.x (CUDA_PATH or nvcc on PATH)
REM   - Windows SDK (D3D11 / DXGI headers)
REM   - MSVC C++17 compiler (Visual Studio 2019 or newer)
REM   - Spout2 SDK: git clone https://github.com/leadedge/Spout2 C:\src\Spout2
REM
REM Usage:
REM   build_spout_wheel.cmd                          -- default Spout2 path (C:\src\Spout2)
REM   set SPOUT2_ROOT=D:\libs\Spout2 && build_spout_wheel.cmd  -- custom SDK path
REM
REM After building, install the bridge alongside the core wheel:
REM   install_td_library.cmd --mode 5 --td-python "..." --spout

echo ========================================
echo  cuda-link-spout Wheel Builder
echo ========================================
echo.

REM ----------------------------------------
REM [1/4] Resolve Python interpreter
REM ----------------------------------------
echo [1/4] Resolving Python interpreter...

set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)

if not defined PY (
    echo.
    echo [ERROR] No Python interpreter found.
    echo         Install Python 3.9 or newer from https://www.python.org/downloads/
    goto :error
)

REM Reject Microsoft Store stub
for /f "delims=" %%e in ('!PY! -c "import sys; print(sys.executable)" 2^>nul') do set "PY_EXE=%%e"
echo !PY_EXE! | findstr /i "\\WindowsApps\\" >nul
if not errorlevel 1 (
    echo.
    echo [ERROR] Detected Microsoft Store Python stub:
    echo           !PY_EXE!
    echo         Install Python from https://www.python.org/downloads/, then
    echo         disable the App Execution Alias in Windows Settings.
    goto :error
)

!PY! -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)"
if errorlevel 1 (
    echo.
    echo [ERROR] Requires Python 3.9 or newer. Detected:
    !PY! --version
    goto :error
)

for /f "tokens=*" %%v in ('!PY! --version 2^>^&1') do set "PYVER=%%v"
echo   !PYVER!
echo   !PY_EXE!
echo.

REM ----------------------------------------
REM [2/4] Resolve Spout2 SDK root
REM ----------------------------------------
echo [2/4] Resolving Spout2 SDK...

REM Use SPOUT2_ROOT env var if set; else default to C:\src\Spout2
if defined SPOUT2_ROOT (
    echo   Using SPOUT2_ROOT env var: !SPOUT2_ROOT!
    set "SPOUT2_ROOT_RESOLVED=!SPOUT2_ROOT!"
) else (
    if exist "C:\src\Spout2\SpoutDX\SpoutDX.h" (
        set "SPOUT2_ROOT_RESOLVED=C:\src\Spout2"
        echo   Found default Spout2 SDK: C:\src\Spout2
    ) else (
        echo.
        echo [ERROR] Spout2 SDK not found.
        echo         Either:
        echo           1. Set SPOUT2_ROOT=^<path-to-Spout2^>
        echo           2. Clone the SDK to the default location:
        echo                git clone --depth 1 https://github.com/leadedge/Spout2 C:\src\Spout2
        goto :error
    )
)

REM Verify the header exists at the resolved path
if not exist "!SPOUT2_ROOT_RESOLVED!\SpoutDX\SpoutDX.h" (
    echo.
    echo [ERROR] SpoutDX.h not found under: !SPOUT2_ROOT_RESOLVED!\SpoutDX\
    echo         Make sure SPOUT2_ROOT points to the root of the Spout2 repository.
    goto :error
)

REM Convert backslashes to forward slashes for CMake config-settings
set "SPOUT2_ROOT_FWD=!SPOUT2_ROOT_RESOLVED:\=/!"
echo   SDK root: !SPOUT2_ROOT_RESOLVED!
echo.

REM ----------------------------------------
REM [3/4] Remove stale spout wheel(s) from dist\
REM ----------------------------------------
echo [3/4] Cleaning stale spout wheel(s)...

set "stale_removed=0"
if exist "dist\" (
    for %%f in ("dist\cuda_link_spout-*.whl") do (
        if exist "%%f" (
            del /q "%%f"
            echo   Removed %%~nxf
            set /a stale_removed+=1
        )
    )
)

if !stale_removed! equ 0 (
    echo   Nothing to clean
)
echo.

REM ----------------------------------------
REM [4/4] Build the native spout wheel
REM ----------------------------------------
echo [4/4] Building native spout wheel...
echo   (This downloads build tools and compiles C++; first run may take several minutes.)
echo.

REM Use pip wheel on the local spout/ subdirectory.
REM  ./spout  — required prefix so pip treats it as a local path, not a PyPI package name.
REM --no-deps: the wheel file itself has no runtime deps to bundle (cuda_link_spout
REM            only imports _spout_bridge and standard library).
REM --config-settings: passes SPOUT2_ROOT into CMakeLists.txt via scikit-build-core.
!PY! -m pip wheel ./spout --no-deps -w dist "--config-settings=cmake.define.SPOUT2_ROOT=!SPOUT2_ROOT_FWD!"
if errorlevel 1 (
    echo.
    echo [ERROR] Spout wheel build failed.
    echo         Common fixes:
    echo           - Run from a Visual Studio Developer Command Prompt if CMake can't find MSVC
    echo           - Verify CUDA Toolkit is installed and nvcc is accessible
    echo           - Check that SPOUT2_ROOT is set correctly
    echo           - Run: !PY! -m pip install --upgrade pip setuptools
    goto :error
)

REM Find the produced wheel
set "SPOUT_WHEEL="
for /f "tokens=*" %%f in ('dir /b /o-d "dist\cuda_link_spout-*.whl" 2^>nul') do (
    if not defined SPOUT_WHEEL set "SPOUT_WHEEL=%%f"
)

if not defined SPOUT_WHEEL (
    echo.
    echo [ERROR] Build reported success but no cuda_link_spout-*.whl found in dist\
    goto :error
)

for /f %%k in ('powershell -Command "[math]::Ceiling((Get-Item 'dist\!SPOUT_WHEEL!').Length / 1KB)"') do set "SPOUT_KB=%%k"

echo.
echo ========================================
echo  BUILD COMPLETE
echo ========================================
echo.
echo   Spout wheel: dist\!SPOUT_WHEEL!
echo   Size:        !SPOUT_KB! KB
echo.

REM Also show the core wheel if present
for /f "tokens=*" %%f in ('dir /b /o-d "dist\cuda_link-*.whl" 2^>nul') do (
    if not defined CORE_WHEEL set "CORE_WHEEL=%%f"
)
if defined CORE_WHEEL (
    echo   Core wheel:  dist\!CORE_WHEEL!
    echo.
)

echo ----------------------------------------
echo  Install both packages into TouchDesigner:
echo ----------------------------------------
echo.
echo   install_td_library.cmd --spout
echo.
echo   Or non-interactively (example for mode 5):
echo   install_td_library.cmd --mode 5 --td-python "^<tdpy^>" --spout
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
