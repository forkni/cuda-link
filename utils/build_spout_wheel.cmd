@echo off
setlocal enabledelayedexpansion
pushd "%~dp0.." || exit /b 1

REM build_spout_wheel.cmd - Build cuda-link-spout native wheel for distribution
REM
REM Compiles the _spout_bridge pybind11 module (CUDA + D3D11 + Spout2 SDK) into a
REM platform wheel and places it in dist\ alongside the core wheel:
REM   dist\cuda_link_spout-<version>-cp311-cp311-win_amd64.whl
REM
REM Prerequisites (auto-provisioned where noted):
REM   - CUDA Toolkit 12.x or 13.x (CUDA_PATH or nvcc on PATH)
REM       [manual -- see https://developer.nvidia.com/cuda-downloads]
REM   - Windows SDK (D3D11 / DXGI) + MSVC C++17 compiler
REM       [auto-installed via winget if absent]
REM   - Spout2 SDK (github.com/leadedge/Spout2)
REM       [auto-cloned to C:\src\Spout2 if absent]
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
REM [1/6] Resolve Python interpreter
REM ----------------------------------------
echo [1/6] Resolving Python interpreter...

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
REM [2/6] Resolve Spout2 SDK root
REM ----------------------------------------
echo [2/6] Resolving Spout2 SDK...

REM The correct Spout2 SDK layout (from spout/CMakeLists.txt:53) is:
REM   SPOUTSDK\SpoutDirectX\SpoutDX\SpoutDX.h
set "SDK_HEADER_REL=SPOUTSDK\SpoutDirectX\SpoutDX\SpoutDX.h"
set "SPOUT2_DEFAULT=C:\src\Spout2"

if defined SPOUT2_ROOT (
    echo   Using SPOUT2_ROOT env var: !SPOUT2_ROOT!
    set "SPOUT2_ROOT_RESOLVED=!SPOUT2_ROOT!"
) else (
    set "SPOUT2_ROOT_RESOLVED=!SPOUT2_DEFAULT!"
)

if exist "!SPOUT2_ROOT_RESOLVED!\!SDK_HEADER_REL!" (
    echo   Found Spout2 SDK: !SPOUT2_ROOT_RESOLVED!
) else if defined SPOUT2_ROOT (
    echo.
    echo [ERROR] SpoutDX.h not found under SPOUT2_ROOT:
    echo           !SPOUT2_ROOT_RESOLVED!\!SDK_HEADER_REL!
    echo         Make sure SPOUT2_ROOT points to the root of the Spout2 repository.
    goto :error
) else if exist "!SPOUT2_ROOT_RESOLVED!\." (
    echo.
    echo [ERROR] Partial Spout2 clone detected at: !SPOUT2_ROOT_RESOLVED!
    echo         Header not found: !SDK_HEADER_REL!
    echo         Delete the directory and re-run to auto-clone a fresh copy:
    echo           rmdir /s /q "!SPOUT2_ROOT_RESOLVED!"
    goto :error
) else (
    echo   Spout2 SDK not found. Cloning from github.com/leadedge/Spout2 ...
    where git >nul 2>&1
    if errorlevel 1 (
        echo.
        echo [ERROR] git not found on PATH. Install git or clone the SDK manually:
        echo           git clone --depth 1 https://github.com/leadedge/Spout2 "!SPOUT2_ROOT_RESOLVED!"
        echo         Then set SPOUT2_ROOT or place the clone at the default path.
        goto :error
    )
    git clone --depth 1 https://github.com/leadedge/Spout2 "!SPOUT2_ROOT_RESOLVED!"
    if errorlevel 1 (
        echo.
        echo [ERROR] git clone failed. Check your internet connection and retry.
        goto :error
    )
    if not exist "!SPOUT2_ROOT_RESOLVED!\!SDK_HEADER_REL!" (
        echo.
        echo [ERROR] Clone succeeded but header still missing: !SPOUT2_ROOT_RESOLVED!\!SDK_HEADER_REL!
        echo         The Spout2 repository layout may have changed. Check manually.
        goto :error
    )
    echo   Cloned Spout2 SDK to: !SPOUT2_ROOT_RESOLVED!
)

REM Convert backslashes to forward slashes for CMake config-settings
set "SPOUT2_ROOT_FWD=!SPOUT2_ROOT_RESOLVED:\=/!"
echo   SDK root: !SPOUT2_ROOT_RESOLVED!
echo.

REM ----------------------------------------
REM [3/6] Detect CUDA Toolkit
REM ----------------------------------------
echo [3/6] Detecting CUDA Toolkit...

set "NVCC_FOUND=0"
if defined CUDA_PATH (
    if exist "!CUDA_PATH!\bin\nvcc.exe" (
        set "NVCC_FOUND=1"
        echo   Found via CUDA_PATH: !CUDA_PATH!
    )
)
if !NVCC_FOUND! equ 0 (
    where nvcc >nul 2>&1
    if not errorlevel 1 (
        set "NVCC_FOUND=1"
        for /f "tokens=*" %%v in ('nvcc --version 2^>^&1 ^| findstr /i "release"') do echo   Found: %%v
    )
)

if !NVCC_FOUND! equ 0 (
    echo.
    echo [ERROR] CUDA Toolkit not found. nvcc is required to compile _spout_bridge.
    echo         Install CUDA 12.x or 13.x from:
    echo           https://developer.nvidia.com/cuda-downloads
    echo         Then re-run this script.
    goto :error
)
echo.

REM ----------------------------------------
REM [4/6] Detect C++ build toolchain (MSVC)
REM ----------------------------------------
echo [4/6] Detecting C++ build toolchain (MSVC)...

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VC_FOUND=0"

if exist "!VSWHERE!" (
    for /f "tokens=*" %%p in ('"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul') do (
        if not "%%p"=="" (
            set "VC_FOUND=1"
            set "VS_PATH=%%p"
        )
    )
)

if !VC_FOUND! neq 1 goto :vc_install
echo   Found: !VS_PATH!
echo.
goto :vc_done

:vc_install
echo   MSVC C++ tools not found. Installing Visual Studio 2022 Build Tools via winget...
echo   (Downloads several GB and may take a few minutes.)
where winget >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] winget not found. Install Visual Studio Build Tools manually:
    echo           https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo         Select "Desktop development with C++" workload, then re-run.
    goto :error
)
winget install --id Microsoft.VisualStudio.2022.BuildTools --accept-source-agreements --accept-package-agreements --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
if errorlevel 1 (
    echo.
    echo [ERROR] winget install failed. Install Visual Studio Build Tools manually:
    echo           https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo         Select "Desktop development with C++" workload, then re-run.
    goto :error
)
REM Re-verify: vswhere may not see the install in the same process tree
set "VC_FOUND=0"
if exist "!VSWHERE!" (
    for /f "tokens=*" %%p in ('"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul') do (
        if not "%%p"=="" (
            set "VC_FOUND=1"
            set "VS_PATH=%%p"
        )
    )
)
if !VC_FOUND! equ 0 (
    echo.
    echo [ERROR] Build Tools installed but not yet visible to this session.
    echo         Open a new terminal window and re-run this script.
    goto :error
)
echo   Installed: !VS_PATH!
echo.

:vc_done

REM ----------------------------------------
REM [5/6] Remove stale spout wheel(s) from dist\
REM ----------------------------------------
echo [5/6] Cleaning stale spout wheel(s)...

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
REM [6/6] Build the native spout wheel
REM ----------------------------------------
echo [6/6] Building native spout wheel...
echo   (Compiles C++ with CUDA; first run may take several minutes.)
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
