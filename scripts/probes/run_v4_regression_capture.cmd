@ECHO OFF
:: run_v4_regression_capture.cmd -- Phase 3.7 regression baseline
::
:: Coordinator for the two-terminal nsys capture of the TD pipeline with
:: CUDALINK_TD_PERSIST_STREAM=0.  Launches the producer and consumer helpers
:: (v4_capture_consumer.cmd, v4_capture_producer.cmd) in their own cmd windows,
:: then waits for you to run the protocol and press a key before running analysis.
::
:: Purpose: confirm that dropping F8 serialises ipc_stream and _rx_stream onto a
:: single WDDM Render queue lane (expected stream-stack visible in nsys-ui timeline).
::
:: Topology:
::   Python producer (example_sender_python.py) -> Receiver-A inside CUDA_Link_Example.toe
::
:: Captures:
::   benchmarks/results/nsys/td_pipeline_v4_producer/producer.nsys-rep
::   benchmarks/results/nsys/td_pipeline_v4_consumer/td_consumer.nsys-rep
::
:: Analysis:
::   python scripts/profiling/analyze_td_pipeline.py \
::     --prod-db benchmarks/results/nsys/td_pipeline_v4_producer/producer.sqlite \
::     --cons-db benchmarks/results/nsys/td_pipeline_v4_consumer/td_consumer.sqlite \
::     --e2e-csv benchmarks/results/nsys/td_pipeline_v4_e2e.csv \
::     --findings-md benchmarks/results/nsys/td_pipeline_v4_findings.md
::
:: Usage (from repo root):
::   scripts\probes\run_v4_regression_capture.cmd  [<toe>]  [<td_exe>]
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
SET "PROD_OUT=%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_v4_producer"
SET "CONS_OUT=%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_v4_consumer"
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

:: --- regression env (for env.txt snapshot) ------------------------------------
SET CUDALINK_TD_PERSIST_STREAM=0
SET CUDALINK_TD_STREAM_PRIO=normal
SET CUDALINK_EXPORT_SYNC=1
SET CUDALINK_LIB_STREAM_PRIO=high
SET CUDALINK_NVTX=1
SET CUDALINK_NVTX_VERBOSE=1

:: env snapshot
SET > "%PROD_OUT%\env.txt"
ECHO [OK] Env snapshot: %PROD_OUT%\env.txt

ECHO.
ECHO ============================================================
ECHO  V4 regression capture  (CUDALINK_TD_PERSIST_STREAM=0)
ECHO ============================================================
ECHO.
ECHO  Output dirs:
ECHO    Producer : %PROD_OUT%
ECHO    Consumer : %CONS_OUT%
ECHO.

:: --- launch consumer (TD) first, then producer after a 5-second stagger -------
ECHO [INFO] Launching consumer (TouchDesigner) in its own window...
START "V4-Consumer (nsys)" cmd /k "%REPO_ROOT%\scripts\probes\v4_capture_consumer.cmd"

TIMEOUT /T 5 /NOBREAK >NUL

ECHO [INFO] Launching producer (Python sender) in its own window...
START "V4-Producer (nsys)" cmd /k "%REPO_ROOT%\scripts\probes\v4_capture_producer.cmd"

ECHO.
ECHO ============================================================
ECHO  Capture protocol (3 reactivation cycles)
ECHO ============================================================
ECHO.
ECHO  1. Wait until CUDA_Link_Example.toe shows Receiver-A streaming at 60 fps.
ECHO  2. In TD: toggle Receiver-A Active  OFF -> wait ~30 s -> ON   (cycle 1)
ECHO  3. Repeat OFF/ON for cycles 2 and 3.
ECHO  4. Stop the producer: press Ctrl+C in the V4-Producer window.
ECHO     nsys finalises producer.nsys-rep on exit.
ECHO  5. Close TouchDesigner.
ECHO     nsys finalises td_consumer.nsys-rep on exit.
ECHO.
ECHO  Then press any key HERE to export SQLite and run analysis.
ECHO ============================================================
ECHO.
PAUSE

:: --- verify SQLite source files exist -----------------------------------------
IF NOT EXIST "%PROD_OUT%\producer.nsys-rep" (
    ECHO.
    ECHO [WARN] %PROD_OUT%\producer.nsys-rep not found.
    ECHO        Was the V4-Producer window closed cleanly (Ctrl+C)?
    EXIT /B 1
)
IF NOT EXIST "%CONS_OUT%\td_consumer.nsys-rep" (
    ECHO.
    ECHO [WARN] %CONS_OUT%\td_consumer.nsys-rep not found.
    ECHO        Was the TouchDesigner window closed after the producer stopped?
    EXIT /B 1
)

:: --- export SQLite ------------------------------------------------------------
ECHO.
ECHO [INFO] Exporting SQLite (producer)...
nsys export --sqlite "%PROD_OUT%\producer.sqlite" "%PROD_OUT%\producer.nsys-rep"

ECHO [INFO] Exporting SQLite (consumer)...
nsys export --sqlite "%CONS_OUT%\td_consumer.sqlite" "%CONS_OUT%\td_consumer.nsys-rep"

:: --- verify SQLite files exist ------------------------------------------------
IF NOT EXIST "%PROD_OUT%\producer.sqlite" (
    ECHO.
    ECHO [WARN] %PROD_OUT%\producer.sqlite not found after export.
    ECHO        Run manually: nsys export --sqlite "%PROD_OUT%\producer.sqlite" "%PROD_OUT%\producer.nsys-rep"
    EXIT /B 1
)
IF NOT EXIST "%CONS_OUT%\td_consumer.sqlite" (
    ECHO.
    ECHO [WARN] %CONS_OUT%\td_consumer.sqlite not found after export.
    ECHO        Run manually: nsys export --sqlite "%CONS_OUT%\td_consumer.sqlite" "%CONS_OUT%\td_consumer.nsys-rep"
    EXIT /B 1
)

:: --- run analysis -------------------------------------------------------------
ECHO.
ECHO [INFO] Running analysis...
python scripts\profiling\analyze_td_pipeline.py ^
    --prod-db "%PROD_OUT%\producer.sqlite" ^
    --cons-db "%CONS_OUT%\td_consumer.sqlite" ^
    --e2e-csv "benchmarks\results\nsys\td_pipeline_v4_e2e.csv" ^
    --findings-md "benchmarks\results\nsys\td_pipeline_v4_findings.md"

IF %ERRORLEVEL% NEQ 0 (
    ECHO [FAIL] Analysis script exited with error %ERRORLEVEL%.
    EXIT /B %ERRORLEVEL%
)

:: --- export nsys stats CSVs ---------------------------------------------------
ECHO.
ECHO [INFO] Exporting nsys stats CSVs...
SET "STAT_REPORTS=nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum,wddm_queue_sum"
nsys stats --format csv --report %STAT_REPORTS% --output "%PROD_OUT%\producer" "%PROD_OUT%\producer.nsys-rep"
nsys stats --format csv --report %STAT_REPORTS% --output "%CONS_OUT%\td_consumer" "%CONS_OUT%\td_consumer.nsys-rep"

ECHO.
ECHO ============================================================
ECHO  Analysis complete.
ECHO  Findings : benchmarks\results\nsys\td_pipeline_v4_findings.md
ECHO  E2E CSV  : benchmarks\results\nsys\td_pipeline_v4_e2e.csv
ECHO ============================================================
ECHO.
ECHO  Open the .nsys-rep files in nsys-ui to inspect the GPU timeline:
ECHO    nsys-ui "%PROD_OUT%\producer.nsys-rep" "%CONS_OUT%\td_consumer.nsys-rep"
ECHO.
ECHO  Expected regression signature (PROFILING.md [S4]):
ECHO    - ipc_stream and _rx_stream CUDA kernels stacking vertically (serialised)
ECHO    - Multi-second settle gap after Receiver reactivation (if tested)
ECHO    - EXPORT_PROFILE post= latency growing monotonically across reactivation cycles

ENDLOCAL
