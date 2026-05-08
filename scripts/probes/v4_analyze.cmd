@ECHO OFF
:: v4_analyze.cmd -- Post-capture analysis for v4 regression captures
::
:: Standalone script: safe to re-run against existing .nsys-rep files without
:: re-capturing.  Called automatically by run_v4_regression_capture.cmd after
:: PAUSE, or run directly to re-analyse.
::
:: Expected inputs (produced by v4_capture_producer.cmd / v4_capture_consumer.cmd):
::   benchmarks\results\nsys\td_pipeline_v4_producer\producer.nsys-rep
::   benchmarks\results\nsys\td_pipeline_v4_consumer\td_consumer.nsys-rep
::
:: Outputs:
::   benchmarks\results\nsys\td_pipeline_v4_findings.md
::   benchmarks\results\nsys\td_pipeline_v4_e2e.csv
::   benchmarks\results\nsys\td_pipeline_v4_producer\producer_<report>.csv  (x6)
::   benchmarks\results\nsys\td_pipeline_v4_consumer\td_consumer_<report>.csv  (x6)
::
:: Usage (from repo root or any cmd.exe window):
::   scripts\probes\v4_analyze.cmd

SETLOCAL ENABLEDELAYEDEXPANSION

:: --- resolve repo root --------------------------------------------------------
PUSHD "%~dp0..\.."
SET "REPO_ROOT=%CD%"
POPD

SET "PROD_OUT=%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_v4_producer"
SET "CONS_OUT=%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_v4_consumer"

:: --- validate inputs ----------------------------------------------------------
IF NOT EXIST "%PROD_OUT%\producer.nsys-rep" (
    ECHO.
    ECHO [WARN] %PROD_OUT%\producer.nsys-rep not found.
    ECHO        Was the V4-Producer window closed cleanly with Ctrl+C?
    EXIT /B 1
)
IF NOT EXIST "%CONS_OUT%\td_consumer.nsys-rep" (
    ECHO.
    ECHO [WARN] %CONS_OUT%\td_consumer.nsys-rep not found.
    ECHO        Was the TouchDesigner window closed after the producer stopped?
    EXIT /B 1
)

:: --- export SQLite (nsys 2026+ syntax) ----------------------------------------
:: Skip if SQLite already exists -- nsys export exits 1 on existing files (no skip behaviour).
ECHO.
IF EXIST "%PROD_OUT%\producer.sqlite" (
    ECHO [INFO] SQLite already exists for producer, skipping export.
) ELSE (
    ECHO [INFO] Exporting SQLite (producer^)...
    nsys export --type sqlite --output "%PROD_OUT%\producer.sqlite" "%PROD_OUT%\producer.nsys-rep"
)

IF EXIST "%CONS_OUT%\td_consumer.sqlite" (
    ECHO [INFO] SQLite already exists for consumer, skipping export.
) ELSE (
    ECHO [INFO] Exporting SQLite (consumer^)...
    nsys export --type sqlite --output "%CONS_OUT%\td_consumer.sqlite" "%CONS_OUT%\td_consumer.nsys-rep"
)

:: --- verify SQLite files exist ------------------------------------------------
IF NOT EXIST "%PROD_OUT%\producer.sqlite" (
    ECHO.
    ECHO [WARN] %PROD_OUT%\producer.sqlite not found after export.
    ECHO        Run manually: nsys export --type sqlite --output "%PROD_OUT%\producer.sqlite" "%PROD_OUT%\producer.nsys-rep"
    EXIT /B 1
)
IF NOT EXIST "%CONS_OUT%\td_consumer.sqlite" (
    ECHO.
    ECHO [WARN] %CONS_OUT%\td_consumer.sqlite not found after export.
    ECHO        Run manually: nsys export --type sqlite --output "%CONS_OUT%\td_consumer.sqlite" "%CONS_OUT%\td_consumer.nsys-rep"
    EXIT /B 1
)

:: --- run analysis -------------------------------------------------------------
ECHO.
ECHO [INFO] Running analysis...
python "%REPO_ROOT%\scripts\profiling\analyze_td_pipeline.py" ^
    --prod-db "%PROD_OUT%\producer.sqlite" ^
    --cons-db "%CONS_OUT%\td_consumer.sqlite" ^
    --e2e-csv "%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_v4_e2e.csv" ^
    --findings-md "%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_v4_findings.md"

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
