@ECHO OFF
:: sync_capture_producer.cmd -- nsys capture for the Python producer side (production-sync path)
::
:: Sync-path env: CUDALINK_EXPORT_SYNC=1 (production default) + CUDALINK_EXPORT_FLUSH_PROBE=1
:: All other flags identical to v5_capture_producer.cmd so only EXPORT_SYNC differs.
::
:: Purpose: capture a clean production-sync baseline for A/B comparison against v5
:: (EXPORT_SYNC=0 low-latency variant).  Expected: cudaStreamSynchronize visible in
:: top CUDA API calls; slot p50 ~190-390 us vs v5's 91 us.
::
:: Output: benchmarks\results\nsys\td_pipeline_sync_producer\producer.nsys-rep
::
:: Usage (launched automatically by run_sync_capture.cmd, or standalone):
::   scripts\probes\sync_capture_producer.cmd
::
:: Stop: Ctrl+C  (or press Enter if the sender has a "press Enter to stop" prompt)
:: nsys finalises the .nsys-rep on process exit.

SETLOCAL ENABLEDELAYEDEXPANSION

:: --- resolve repo root --------------------------------------------------------
PUSHD "%~dp0..\.."
SET "REPO_ROOT=%CD%"
POPD

SET "SENDER=%REPO_ROOT%\td_exporter\example_sender_python.py"
SET "OUT_DIR=%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_sync_producer"

IF NOT EXIST "%OUT_DIR%" MKDIR "%OUT_DIR%"

:: --- sanity check -------------------------------------------------------------
IF NOT EXIST "%SENDER%" (
    ECHO [FAIL] example_sender_python.py not found: %SENDER%
    EXIT /B 2
)

:: --- production-sync env (only EXPORT_SYNC differs from v5) -------------------
SET CUDALINK_TD_PERSIST_STREAM=1
SET CUDALINK_TD_STREAM_PRIO=normal
SET CUDALINK_EXPORT_SYNC=1
SET CUDALINK_EXPORT_FLUSH_PROBE=1
SET CUDALINK_LIB_STREAM_PRIO=high
SET CUDALINK_NVTX=1
SET CUDALINK_NVTX_VERBOSE=1

ECHO ============================================================
ECHO  Sync Producer capture  (EXPORT_SYNC=1 + FLUSH_PROBE=1)
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
