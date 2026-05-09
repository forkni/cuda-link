@ECHO OFF
:: v5_capture_producer.cmd -- nsys capture for the Python producer side (v5 async-flush-probe)
::
:: Async-path env: CUDALINK_EXPORT_SYNC=0 + CUDALINK_EXPORT_FLUSH_PROBE=1
:: Purpose: validate that switching from blocking cudaStreamSynchronize (~630 us/slot)
:: to the non-blocking cudaStreamQuery flush probe (~12 us/slot) recovers the
:: dominant producer slot time without consumer regression.
::
:: Compare to v4_capture_producer.cmd which set EXPORT_SYNC=1 (blocking path).
::
:: Output: benchmarks\results\nsys\td_pipeline_v5_producer\producer.nsys-rep
::
:: Usage (launched automatically by run_v5_regression_capture.cmd, or standalone):
::   scripts\probes\v5_capture_producer.cmd
::
:: Stop: Ctrl+C  (or press Enter if the sender has a "press Enter to stop" prompt)
:: nsys finalises the .nsys-rep on process exit.

SETLOCAL ENABLEDELAYEDEXPANSION

:: --- resolve repo root --------------------------------------------------------
PUSHD "%~dp0..\.."
SET "REPO_ROOT=%CD%"
POPD

SET "SENDER=%REPO_ROOT%\td_exporter\example_sender_python.py"
SET "OUT_DIR=%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_v5_producer"

IF NOT EXIST "%OUT_DIR%" MKDIR "%OUT_DIR%"

:: --- sanity check -------------------------------------------------------------
IF NOT EXIST "%SENDER%" (
    ECHO [FAIL] example_sender_python.py not found: %SENDER%
    EXIT /B 2
)

:: --- v5 async-flush-probe env -------------------------------------------------
:: Key change vs v4: EXPORT_SYNC=0 activates cudaStreamQuery flush probe path.
:: All other flags at their CORRECT (non-regression) values.
SET CUDALINK_TD_PERSIST_STREAM=1
SET CUDALINK_TD_STREAM_PRIO=normal
SET CUDALINK_EXPORT_SYNC=0
SET CUDALINK_EXPORT_FLUSH_PROBE=1
SET CUDALINK_LIB_STREAM_PRIO=high
SET CUDALINK_NVTX=1
SET CUDALINK_NVTX_VERBOSE=1

ECHO ============================================================
ECHO  V5 Producer capture  (EXPORT_SYNC=0 + FLUSH_PROBE=1)
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
