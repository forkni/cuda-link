@ECHO OFF
:: sync_capture_consumer.cmd -- nsys capture for the TouchDesigner consumer side (production-sync path)
::
:: Sync-path env: CUDALINK_EXPORT_SYNC=1 (production default) + CUDALINK_EXPORT_FLUSH_PROBE=1
:: All other flags identical to v5_capture_consumer.cmd so only EXPORT_SYNC differs.
::
:: --duration 150: bounds the consumer nsys session to 2.5 min (matches v5 stagger protocol).
::
:: Output: benchmarks\results\nsys\td_pipeline_sync_consumer\td_consumer.nsys-rep
::
:: Usage (launched automatically by run_sync_capture.cmd, or standalone):
::   scripts\probes\sync_capture_consumer.cmd  [<toe>]  [<td_exe>]
::
:: Defaults:
::   toe    = <repo_root>\CUDA_Link_Example.toe
::   td_exe = C:\Program Files\Derivative\TouchDesigner.2025.32820\bin\TouchDesigner.exe
::
:: Stop: nsys auto-stops after 150 s; close TouchDesigner immediately after.
:: nsys finalises the .nsys-rep on process exit.

SETLOCAL ENABLEDELAYEDEXPANSION

:: --- resolve repo root --------------------------------------------------------
PUSHD "%~dp0..\.."
SET "REPO_ROOT=%CD%"
POPD

SET "DEFAULT_TOE=%REPO_ROOT%\CUDA_Link_Example.toe"
SET "DEFAULT_TD=C:\Program Files\Derivative\TouchDesigner.2025.32820\bin\TouchDesigner.exe"

IF NOT "%~1"=="" (SET "TOE=%~1")    ELSE (SET "TOE=%DEFAULT_TOE%")
IF NOT "%~2"=="" (SET "TD_EXE=%~2") ELSE (SET "TD_EXE=%DEFAULT_TD%")

FOR %%F IN ("%TOE%")    DO SET "TOE=%%~fF"
FOR %%F IN ("%TD_EXE%") DO SET "TD_EXE=%%~fF"

SET "OUT_DIR=%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_sync_consumer"

IF NOT EXIST "%OUT_DIR%" MKDIR "%OUT_DIR%"

:: --- sanity checks ------------------------------------------------------------
IF NOT EXIST "%TOE%" (
    ECHO [FAIL] .toe not found: %TOE%
    EXIT /B 2
)
IF NOT EXIST "%TD_EXE%" (
    ECHO [FAIL] TouchDesigner.exe not found: %TD_EXE%
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
ECHO  Sync Consumer capture  (EXPORT_SYNC=1 + FLUSH_PROBE=1)
ECHO  Output: %OUT_DIR%\td_consumer.nsys-rep
ECHO  .toe  : %TOE%
ECHO  Note  : nsys auto-stops after 150 s
ECHO ============================================================
ECHO.
ECHO  Start the producer AFTER TouchDesigner has fully loaded.
ECHO  Close TouchDesigner when prompted after capture completes.
ECHO.

nsys profile --force-overwrite=true ^
  --trace=cuda,nvtx,wddm ^
  --wddm-memory-trace=false ^
  --wddm-additional-events=true ^
  --wddm-backtraces=true ^
  --duration 150 ^
  --output "%OUT_DIR%\td_consumer" ^
  "%TD_EXE%" "%TOE%"

ECHO.
ECHO [OK] Consumer capture complete: %OUT_DIR%\td_consumer.nsys-rep
ECHO      Close TouchDesigner if it is still open.

ENDLOCAL
