@echo off
REM install_cpp_top.cmd — install the C++ CUDA Link TOPs into TouchDesigner's system Plugins folder.
REM
REM Delegates to scripts\install_cpp_top.py. Close TouchDesigner first (TD locks the DLLs).
REM
REM   install_cpp_top.cmd                        install from cpp_top\build into
REM                                              <Documents>\Derivative\Plugins\CudaLink\
REM   install_cpp_top.cmd install --dll-dir DIR  install the two DLLs from DIR
REM   install_cpp_top.cmd status                 show the installed copy and VERSION.txt
REM   install_cpp_top.cmd uninstall              move the folder out of Plugins
REM   install_cpp_top.cmd install --dry-run      show what would happen

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
!PY! scripts\install_cpp_top.py %*
set "EXIT_CODE=%ERRORLEVEL%"

popd
endlocal
pause
exit /b %EXIT_CODE%
