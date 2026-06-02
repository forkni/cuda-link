@echo off
REM install_td_library.cmd — TouchDesigner library-mode installer for cuda-link.
REM
REM Delegates to scripts\install_td_library.py for interactive target selection:
REM
REM   Mode 1  External folder  (pip --target)   — set CUDALINK_LIB_PATH=<folder>
REM   Mode 2  Existing venv                      — set CUDALINK_LIB_PATH=<venv/Lib/site-packages>
REM   Mode 3  Conda environment                  — set CUDALINK_LIB_PATH=<conda-env/.../site-packages>
REM   Mode 4  System / parallel Python 3.11      — add site-packages to TD Preferences
REM   Mode 5  TouchDesigner's own Python          — no path change needed (modifies TD's Python)
REM
REM Non-interactive examples:
REM   install_td_library.cmd --mode 1 --target "D:\cuda_link_lib"
REM   install_td_library.cmd --mode 2 --venv "D:\myproject\.venv"
REM   install_td_library.cmd --mode 3 --conda base
REM   install_td_library.cmd --mode 4 --python "C:\Python311\python.exe"
REM   install_td_library.cmd --mode 5 --td-python "C:\Program Files\Derivative\TouchDesigner\bin\python.exe"
REM   install_td_library.cmd --dry-run
REM
REM Get TD's Python path from the Textport: print(app.pythonExecutable)

setlocal enabledelayedexpansion
pushd "%~dp0" || exit /b 1

REM Resolve Python interpreter (py -3 preferred; falls back to python).
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [ERROR] No Python interpreter found. Install Python 3.9+ from https://www.python.org/
    popd
    endlocal
    pause
    exit /b 1
)

REM Reject Microsoft Store stub.
for /f "delims=" %%e in ('!PY! -c "import sys; print(sys.executable)" 2^>nul') do set "PY_EXE=%%e"
echo !PY_EXE! | findstr /i "\\WindowsApps\\" >nul
if not errorlevel 1 (
    echo [ERROR] Microsoft Store Python stub detected. Install from https://www.python.org/
    popd
    endlocal
    pause
    exit /b 1
)

REM Delegate all logic to the Python helper, forwarding all arguments.
!PY! scripts\install_td_library.py %*
set "EXIT_CODE=%ERRORLEVEL%"

popd
endlocal
pause
exit /b %EXIT_CODE%
