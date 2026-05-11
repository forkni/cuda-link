@ECHO OFF
:: v4_capture_producer.cmd -- nsys capture for the Python producer side (v4 regression)
::
:: Regression env: CUDALINK_TD_PERSIST_STREAM=0 (F8 dropped) to reproduce stream
:: destroy/recreate serialisation on each TD cook.
::
:: Output: benchmarks\results\nsys\td_pipeline_v4_producer\producer.nsys-rep
::
:: Usage (launched automatically by run_v4_regression_capture.cmd, or standalone):
::   scripts\probes\v4_capture_producer.cmd
::
:: Stop: Ctrl+C  (or press Enter if the sender has a "press Enter to stop" prompt)
:: nsys finalises the .nsys-rep on process exit.

SETLOCAL ENABLEDELAYEDEXPANSION

:: --- resolve repo root --------------------------------------------------------
PUSHD "%~dp0..\.."
SET "REPO_ROOT=%CD%"
POPD

SET "SENDER=%REPO_ROOT%\td_exporter\example_sender_python.py"
SET "OUT_DIR=%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_v4_producer"

IF NOT EXIST "%OUT_DIR%" MKDIR "%OUT_DIR%"

:: --- sanity check -------------------------------------------------------------
IF NOT EXIST "%SENDER%" (
    ECHO [FAIL] example_sender_python.py not found: %SENDER%
    EXIT /B 2
)

:: --- regression env -----------------------------------------------------------
SET CUDALINK_TD_PERSIST_STREAM=0
SET CUDALINK_TD_STREAM_PRIO=normal
SET CUDALINK_EXPORT_SYNC=1
SET CUDALINK_LIB_STREAM_PRIO=high
SET CUDALINK_NVTX=1
SET CUDALINK_NVTX_VERBOSE=1

ECHO ============================================================
ECHO  V4 Producer capture  (CUDALINK_TD_PERSIST_STREAM=0)
ECHO  Output: %OUT_DIR%\producer.nsys-rep
ECHO ============================================================
ECHO.
ECHO  Start the consumer (TD) before running this script.
ECHO  Stop with Ctrl+C or Enter when captures are done.
ECHO.

nsys profile --force-overwrite=true ^
  --trace=cuda,nvtx,wddm ^
  --wddm-memory-trace=false ^
  --wddm-additional-events=true ^
  --wddm-backtraces=true ^
  --output "%OUT_DIR%\producer" ^
  python "%SENDER%"

ECHO.
ECHO [OK] Producer capture complete: %OUT_DIR%\producer.nsys-rep

ENDLOCAL
