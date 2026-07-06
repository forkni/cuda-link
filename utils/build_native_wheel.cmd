@echo off
setlocal enabledelayedexpansion
pushd "%~dp0.." || exit /b 1

REM build_native_wheel.cmd - Build cuda-link-native native wheel for distribution
REM
REM Compiles the _native_waiter pybind11 module into a platform wheel and places
REM it in dist\ alongside the core wheel:
REM   dist\cuda_link_native-<version>-cp311-cp311-win_amd64.whl
REM
REM Unlike cuda-link-spout, this module needs NO CUDA Toolkit and NO SDK at build
REM time -- it resolves cudaEventQuery from a function-pointer address the caller
REM already has (CUDARuntimeAPI.cudart_event_query_fn_ptr()), never linking or
REM loading cudart itself. Only a C++17 compiler + Windows SDK are required.
REM
REM Prerequisites (auto-provisioned where noted):
REM   - Windows SDK + MSVC C++17 compiler
REM       [auto-installed via winget if absent]
REM
REM Usage:
REM   build_native_wheel.cmd
REM
REM After building, install it alongside the core wheel:
REM   install_td_library.cmd --mode 5 --td-python "..." --native
REM (native installs BY DEFAULT unless --no-native is passed -- see install_td_library.py)

echo ========================================
echo  cuda-link-native Wheel Builder
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
REM [2/4] Detect C++ build toolchain (MSVC)
REM ----------------------------------------
echo [2/4] Detecting C++ build toolchain (MSVC)...

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
REM [3/4] Remove stale native wheel(s) from dist\
REM ----------------------------------------
echo [3/4] Cleaning stale native wheel(s)...

set "stale_removed=0"
if exist "dist\" (
    for %%f in ("dist\cuda_link_native-*.whl") do (
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
REM [4/4] Build the native wheel
REM ----------------------------------------
echo [4/4] Building native wheel...
echo   (Compiles C++; no CUDA Toolkit needed. First run may take a minute.)
echo.

REM Use pip wheel on the local native/ subdirectory.
REM  ./native — required prefix so pip treats it as a local path, not a PyPI package name.
REM --no-deps: the wheel file itself has no runtime deps to bundle (cuda_link_native
REM            only imports _native_waiter and standard library).
!PY! -m pip wheel ./native --no-deps -w dist
if errorlevel 1 (
    echo.
    echo [ERROR] Native wheel build failed.
    echo         Common fixes:
    echo           - Run from a Visual Studio Developer Command Prompt if CMake can't find MSVC
    echo           - Run: !PY! -m pip install --upgrade pip setuptools
    goto :error
)

REM Find the produced wheel
set "NATIVE_WHEEL="
for /f "tokens=*" %%f in ('dir /b /o-d "dist\cuda_link_native-*.whl" 2^>nul') do (
    if not defined NATIVE_WHEEL set "NATIVE_WHEEL=%%f"
)

if not defined NATIVE_WHEEL (
    echo.
    echo [ERROR] Build reported success but no cuda_link_native-*.whl found in dist\
    goto :error
)

for /f %%k in ('powershell -Command "[math]::Ceiling((Get-Item 'dist\!NATIVE_WHEEL!').Length / 1KB)"') do set "NATIVE_KB=%%k"

echo.
echo ========================================
echo  BUILD COMPLETE
echo ========================================
echo.
echo   Native wheel: dist\!NATIVE_WHEEL!
echo   Size:         !NATIVE_KB! KB
echo.

REM Also show the core wheel if present
for /f "tokens=*" %%f in ('dir /b /o-d "dist\cuda_link-*.whl" 2^>nul') do (
    if not defined CORE_WHEEL set "CORE_WHEEL=%%f"
)
if defined CORE_WHEEL (
    echo   Core wheel:   dist\!CORE_WHEEL!
    echo.
)

echo ----------------------------------------
echo  Install alongside the core wheel:
echo ----------------------------------------
echo.
echo   install_td_library.cmd
echo   (native installs by default -- pass --no-native to skip it)
echo.
echo   Or non-interactively (example for mode 5):
echo   install_td_library.cmd --mode 5 --td-python "^<tdpy^>"
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
