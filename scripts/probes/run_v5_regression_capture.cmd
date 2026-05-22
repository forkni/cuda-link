@ECHO OFF
:: run_v5_regression_capture.cmd -- async-flush-probe baseline (lever 1a)
::
:: Coordinator for the two-terminal nsys capture of the TD pipeline with
:: CUDALINK_EXPORT_SYNC=0 + CUDALINK_EXPORT_FLUSH_PROBE=1.
:: All other flags are at their CORRECT (non-regression) values.
::
:: Purpose: validate that the non-blocking cudaStreamQuery flush probe path
:: (cudaStreamQuery ~12 us vs cudaStreamSynchronize ~630 us) recovers producer
:: slot time without consumer regression.  Compare findings against v4 baseline:
::   benchmarks/results/nsys/td_pipeline_v4_findings.md
::
:: Topology:
::   Python producer (example_sender_python.py) -> Receiver-A inside CUDA_Link_Example.toe
::
:: Captures:
::   benchmarks/results/nsys/td_pipeline_v5_producer/producer.nsys-rep
::   benchmarks/results/nsys/td_pipeline_v5_consumer/td_consumer.nsys-rep  (auto-stops at 150 s)
::
:: Usage (from repo root):
::   scripts\probes\run_v5_regression_capture.cmd  [<toe>]  [<td_exe>]
::
:: Defaults:
::   toe    = <repo_root>\CUDA_Link_Example.toe
::   td_exe = C:\Program Files\Derivative\TouchDesigner.2025.32820\bin\TouchDesigner.exe

SETLOCAL ENABLEDELAYEDEXPANSION

:: --- resolve repo root --------------------------------------------------------
PUSHD "%~dp0..\.."
SET "REPO_ROOT=%CD%"
POPD

:: --- defaults (for env.txt traceability only -- helpers use their own defaults)
SET "DEFAULT_TOE=%REPO_ROOT%\CUDA_Link_Example.toe"
SET "DEFAULT_TD=C:\Program Files\Derivative\TouchDesigner.2025.32820\bin\TouchDesigner.exe"

IF NOT "%~1"=="" (SET "TOE=%~1")     ELSE (SET "TOE=%DEFAULT_TOE%")
IF NOT "%~2"=="" (SET "TD_EXE=%~2")  ELSE (SET "TD_EXE=%DEFAULT_TD%")

FOR %%F IN ("%TOE%")    DO SET "TOE=%%~fF"
FOR %%F IN ("%TD_EXE%") DO SET "TD_EXE=%%~fF"

:: --- output directories -------------------------------------------------------
SET "PROD_OUT=%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_v5_producer"
SET "CONS_OUT=%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_v5_consumer"
IF NOT EXIST "%PROD_OUT%" MKDIR "%PROD_OUT%"
IF NOT EXIST "%CONS_OUT%" MKDIR "%CONS_OUT%"

:: --- sanity checks ------------------------------------------------------------
IF NOT EXIST "%TOE%" (
    ECHO [FAIL] .toe not found: %TOE%
    EXIT /B 2
)
IF NOT EXIST "%TD_EXE%" (
    ECHO [FAIL] TouchDesigner.exe not found: %TD_EXE%
    EXIT /B 2
)

:: --- v5 async-flush-probe env (for env.txt snapshot) -------------------------
SET CUDALINK_TD_PERSIST_STREAM=1
SET CUDALINK_TD_STREAM_PRIO=normal
SET CUDALINK_EXPORT_SYNC=0
SET CUDALINK_EXPORT_FLUSH_PROBE=1
SET CUDALINK_LIB_STREAM_PRIO=high
SET CUDALINK_NVTX=1
SET CUDALINK_NVTX_VERBOSE=1

:: env snapshot
SET > "%PROD_OUT%\env.txt"
ECHO [OK] Env snapshot: %PROD_OUT%\env.txt

ECHO.
ECHO ============================================================
ECHO  V5 capture  (EXPORT_SYNC=0 + FLUSH_PROBE=1)
ECHO ============================================================
ECHO.
ECHO  Output dirs:
ECHO    Producer : %PROD_OUT%
ECHO    Consumer : %CONS_OUT%
ECHO.

:: --- launch consumer (TD) first, then producer after a 5-second stagger -------
ECHO [INFO] Launching consumer (TouchDesigner) in its own window...
ECHO        nsys will auto-stop the consumer session after 150 s.
START "V5-Consumer (nsys)" cmd /k "%REPO_ROOT%\scripts\probes\v5_capture_consumer.cmd"

TIMEOUT /T 5 /NOBREAK >NUL

ECHO [INFO] Launching producer (Python sender) in its own window...
START "V5-Producer (nsys)" cmd /k "%REPO_ROOT%\scripts\probes\v5_capture_producer.cmd"

ECHO.
ECHO ============================================================
ECHO  Capture protocol (steady-state, no reactivation)
ECHO ============================================================
ECHO.
ECHO  1. Wait until CUDA_Link_Example.toe shows Receiver-A streaming at 60 fps.
ECHO  2. Let it run for ~90 s of steady-state data.
ECHO  3. Stop the producer: press Ctrl+C in the V5-Producer window.
ECHO     nsys finalises producer.nsys-rep on exit.
ECHO  4. The consumer nsys session auto-stops at 150 s; close TouchDesigner.
ECHO     nsys finalises td_consumer.nsys-rep on exit.
ECHO.
ECHO  Then press any key HERE to export SQLite and run analysis.
ECHO ============================================================
ECHO.
PAUSE
CALL "%~dp0v5_analyze.cmd"
ENDLOCAL
