@ECHO OFF
:: run_v4_regression_capture.cmd -- Phase 3.7 regression baseline
::
:: Two-terminal nsys capture of the TD pipeline with CUDALINK_TD_PERSIST_STREAM=0.
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
::   td_exe = C:\Program Files\Derivative\TouchDesigner.2025.32460\bin\TouchDesigner.exe

SETLOCAL ENABLEDELAYEDEXPANSION

:: --- resolve repo root --------------------------------------------------------
PUSHD "%~dp0..\.."
SET "REPO_ROOT=%CD%"
POPD

:: --- defaults -----------------------------------------------------------------
SET "DEFAULT_TOE=%REPO_ROOT%\CUDA_Link_Example.toe"
SET "DEFAULT_TD=C:\Program Files\Derivative\TouchDesigner.2025.32460\bin\TouchDesigner.exe"
SET "DEFAULT_SENDER=%REPO_ROOT%\td_exporter\example_sender_python.py"

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
IF NOT EXIST "%DEFAULT_SENDER%" (
    ECHO [FAIL] example_sender_python.py not found: %DEFAULT_SENDER%
    EXIT /B 2
)

:: --- regression env -----------------------------------------------------------
:: F8 dropped (PERSIST_STREAM=0) to expose stream destroy/recreate on each TD cook.
:: F2 kept because it is load-bearing (Phase 3.6 Step C confirmed).
:: EXPORT_SYNC kept because it is required.

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
ECHO  This is a TWO-TERMINAL capture.  Follow the steps below:
ECHO.
ECHO ---- TERMINAL 1 (Python producer) --------------------------
ECHO  Open a NEW cmd.exe window in the repo root and run:
ECHO.
ECHO    SET CUDALINK_TD_PERSIST_STREAM=0
ECHO    SET CUDALINK_TD_STREAM_PRIO=normal
ECHO    SET CUDALINK_EXPORT_SYNC=1
ECHO    SET CUDALINK_NVTX=1
ECHO    SET CUDALINK_NVTX_VERBOSE=1
ECHO    nsys profile --force-overwrite=true ^
ECHO      --trace=cuda,nvtx,wddm ^
ECHO      --wddm-memory-trace=false ^
ECHO      --wddm-additional-events=true ^
ECHO      --wddm-backtraces=true ^
ECHO      --output "%PROD_OUT%\producer" ^
ECHO      python "%DEFAULT_SENDER%"
ECHO.
ECHO ---- TERMINAL 2 (TouchDesigner consumer) -------------------
ECHO  Open ANOTHER new cmd.exe window in the repo root and run:
ECHO.
ECHO    SET CUDALINK_TD_PERSIST_STREAM=0
ECHO    SET CUDALINK_TD_STREAM_PRIO=normal
ECHO    SET CUDALINK_EXPORT_SYNC=1
ECHO    SET CUDALINK_NVTX=1
ECHO    SET CUDALINK_NVTX_VERBOSE=1
ECHO    nsys profile --force-overwrite=true ^
ECHO      --trace=cuda,nvtx,wddm ^
ECHO      --wddm-memory-trace=false ^
ECHO      --wddm-additional-events=true ^
ECHO      --wddm-backtraces=true ^
ECHO      --output "%CONS_OUT%\td_consumer" ^
ECHO      "%TD_EXE%" "%TOE%"
ECHO.
ECHO ---- CAPTURE DURATION --------------------------------------
ECHO  Capture both for ~60 seconds of steady-state streaming.
ECHO  Start Terminal 2 (TD) a few seconds BEFORE Terminal 1 so
ECHO  the clock overlap window is maximised.
ECHO.
ECHO  Stop by closing the Python sender (Ctrl+C or Enter) --
ECHO  nsys will finalise the .nsys-rep on process exit.
ECHO  Stop TD only after the producer has exited.
ECHO.
ECHO ---- AFTER CAPTURE -----------------------------------------
ECHO  Export SQLite from each .nsys-rep (nsys stats requires it):
ECHO.
ECHO    nsys export --sqlite "%PROD_OUT%\producer.sqlite" "%PROD_OUT%\producer.nsys-rep"
ECHO    nsys export --sqlite "%CONS_OUT%\td_consumer.sqlite" "%CONS_OUT%\td_consumer.nsys-rep"
ECHO.
ECHO ============================================================

ECHO.
ECHO Press any key in THIS window once both captures are complete
ECHO and both SQLite exports have been created.
PAUSE

:: --- verify SQLite files exist ------------------------------------------------
IF NOT EXIST "%PROD_OUT%\producer.sqlite" (
    ECHO.
    ECHO [WARN] %PROD_OUT%\producer.sqlite not found.
    ECHO        Run: nsys export --sqlite "%PROD_OUT%\producer.sqlite" "%PROD_OUT%\producer.nsys-rep"
    ECHO        Then re-run this script from the PAUSE line (or run analysis manually).
    EXIT /B 1
)
IF NOT EXIST "%CONS_OUT%\td_consumer.sqlite" (
    ECHO.
    ECHO [WARN] %CONS_OUT%\td_consumer.sqlite not found.
    ECHO        Run: nsys export --sqlite "%CONS_OUT%\td_consumer.sqlite" "%CONS_OUT%\td_consumer.nsys-rep"
    ECHO        Then re-run this script from the PAUSE line (or run analysis manually).
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

:: --- export nsys stats CSVs (mirrors v2/v3 directory layout) -------------------
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
ECHO  Expected regression signature (PROFILING.md ^[S4]):
ECHO    - ipc_stream and _rx_stream CUDA kernels stacking vertically (serialised)
ECHO    - Multi-second settle gap after Receiver reactivation (if tested)
ECHO    - EXPORT_PROFILE post= latency growing monotonically across reactivation cycles

ENDLOCAL
