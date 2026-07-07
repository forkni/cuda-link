@echo off
setlocal enabledelayedexpansion
pushd "%~dp0.." || exit /b 1

REM build_wheel.cmd - Build cuda-link Python wheel for distribution
REM
REM Uses the resolved Python interpreter (preferring 'py -3' Windows launcher)
REM with PyPA's PEP 517 isolated build env: <python> -m build --wheel
REM (scikit-build-core + pybind11, declared in pyproject.toml [build-system]).
REM
REM This is a DEV/CI tool. End users install a PREBUILT wheel (see
REM https://github.com/forkni/cuda-link/releases or scripts/install_td_library.py,
REM which downloads the matching release asset automatically) -- they never run
REM this script or need a compiler. See docs/adr/0013-prebuilt-wheel-distribution.md.
REM
REM Two build modes:
REM   build_wheel.cmd            Native build (default). Compiles the
REM                               _native_waiter wait-backend accelerator into
REM                               the wheel, producing a PLATFORM wheel
REM                               (dist\cuda_link-<version>-cp3XX-cp3XX-win_amd64.whl).
REM                               Requires an MSVC C++17 toolchain (checked
REM                               below via vswhere -- fails fast with an
REM                               actionable message if none is found).
REM   build_wheel.cmd nowaiter    Fallback build. Passes
REM                               -C cmake.define.BUILD_NATIVE_WAITER=OFF to
REM                               skip the native extension entirely, producing
REM                               a compiler-free py3-none-any wheel. Importer
REM                               falls back to its Python wait path at runtime
REM                               either way.
REM
REM Usage:
REM   Double-click or run from any terminal:
REM     build_wheel.cmd
REM     build_wheel.cmd nowaiter
REM
REM   Then install into any Python environment (see dist\ for the exact filename):
REM     pip install "dist\cuda_link-<version>-*.whl"
REM     pip install "dist\cuda_link-<version>-*.whl[torch]"
REM     pip install "dist\cuda_link-<version>-*.whl[all]"

echo ========================================
echo  cuda-link Wheel Builder
echo ========================================
echo.

REM ----------------------------------------
REM [0/5] Parse build mode
REM ----------------------------------------
set "FALLBACK=0"
set "BUILD_ARGS="
if /i "%~1"=="nowaiter" set "FALLBACK=1"
if /i "%~1"=="--fallback" set "FALLBACK=1"

if "!FALLBACK!"=="1" (
    set "BUILD_ARGS=-C cmake.define.BUILD_NATIVE_WAITER=OFF -C wheel.py-api=py3 -C wheel.platlib=false"
    echo   Mode: fallback ^(py3-none-any, no native extension^)
) else (
    echo   Mode: native ^(cp3XX-win_amd64, compiles _native_waiter^)
)
echo.

REM ----------------------------------------
REM [1/5] Resolve Python interpreter
REM ----------------------------------------
echo [1/5] Resolving Python interpreter...

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
REM [2/5] MSVC preflight (native mode only)
REM ----------------------------------------
if "!FALLBACK!"=="1" (
    echo [2/5] Skipping MSVC preflight ^(fallback mode^).
    echo.
) else (
    echo [2/5] Checking for MSVC C++ toolchain...

    REM vswhere is authoritative; `where cl` is a false negative -- cl.exe is
    REM only on PATH inside a VS Developer Prompt, not a plain shell.
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if not exist "!VSWHERE!" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"

    set "MSVC_PATH="
    if exist "!VSWHERE!" (
        for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "MSVC_PATH=%%i"
    )

    if not defined MSVC_PATH (
        echo.
        echo [ERROR] No MSVC C++ toolchain detected ^(checked via vswhere^).
        echo         Building the native _native_waiter extension requires the
        echo         "Desktop development with C++" workload ^(Visual Studio or
        echo         VS Build Tools^): https://visualstudio.microsoft.com/downloads/
        echo.
        echo         Options:
        echo           1. Install VS Build Tools with the C++ workload, then re-run.
        echo           2. Build the compiler-free fallback wheel instead:
        echo                build_wheel.cmd nowaiter
        echo           3. Skip building entirely and use a prebuilt wheel from
        echo              https://github.com/forkni/cuda-link/releases
        goto :error
    )

    echo   MSVC found: !MSVC_PATH!
    echo.
)

REM ----------------------------------------
REM [2.5/5] Sync td_exporter/CUDAIPCWrapper.py from canonical source
REM ----------------------------------------
echo [2.5/5] Syncing CUDAIPCWrapper.py...

!PY! scripts\sync_td_wrapper.py
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to sync td_exporter/CUDAIPCWrapper.py from src/cuda_link/cuda_ipc_wrapper.py.
    goto :error
)
echo.

REM ----------------------------------------
REM [3/5] Ensure PyPA build frontend
REM ----------------------------------------
echo [3/5] Ensuring build tools...

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
REM [4/5] Clean old build artifacts
REM ----------------------------------------
echo [4/5] Cleaning old artifacts...

set "cleaned=0"

REM Remove only stale cuda_link wheels (dist\cuda_link-*.whl) — this glob would
REM exclude any future sibling wheel package (e.g. a hypothetical cuda_link_spout-*),
REM since that requires an underscore directly after "cuda_link", not a dash.
if exist "dist\" (
    for %%f in ("dist\cuda_link-*.whl") do (
        if exist "%%f" (
            del /q "%%f"
            echo   Removed dist\%%~nxf
            set /a cleaned+=1
        )
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
REM [5/5] Build the wheel
REM ----------------------------------------
echo [5/5] Building wheel...
echo.

!PY! -m build --wheel !BUILD_ARGS!
if errorlevel 1 (
    echo.
    echo [ERROR] Wheel build failed.
    echo         Check the output above for details.
    if "!FALLBACK!"=="1" (
        echo         Common fixes:
        echo           - Ensure pyproject.toml is valid
        echo           - Run: !PY! -m pip install --upgrade setuptools build
    ) else (
        echo         The MSVC preflight above passed, so this is likely a genuine
        echo         compile error in src/cuda_link/_cpp/native_waiter.cpp or a
        echo         pyproject.toml/CMakeLists.txt misconfiguration -- not a missing
        echo         toolchain. To rule out the toolchain, try the fallback build:
        echo           build_wheel.cmd nowaiter
    )
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
