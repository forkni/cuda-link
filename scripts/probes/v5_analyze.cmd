@ECHO OFF
:: v5_analyze.cmd -- Post-capture analysis for v5 async-flush-probe captures
::
:: Standalone script: safe to re-run against existing .nsys-rep files without
:: re-capturing.  Called automatically by run_v5_regression_capture.cmd after
:: PAUSE, or run directly to re-analyse.
::
:: Expected inputs (produced by v5_capture_producer.cmd / v5_capture_consumer.cmd):
::   benchmarks\results\nsys\td_pipeline_v5_producer\producer.nsys-rep
::   benchmarks\results\nsys\td_pipeline_v5_consumer\td_consumer.nsys-rep
::
:: Outputs:
::   benchmarks\results\nsys\td_pipeline_v5_findings.md
::   benchmarks\results\nsys\td_pipeline_v5_e2e.csv
::   benchmarks\results\nsys\td_pipeline_v5_producer\producer_<report>.csv  (x6)
::   benchmarks\results\nsys\td_pipeline_v5_consumer\td_consumer_<report>.csv  (x6)
::
:: Usage (from repo root or any cmd.exe window):
::   scripts\probes\v5_analyze.cmd

SETLOCAL ENABLEDELAYEDEXPANSION

:: --- resolve repo root --------------------------------------------------------
PUSHD "%~dp0..\.."
SET "REPO_ROOT=%CD%"
POPD

SET "PROD_OUT=%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_v5_producer"
SET "CONS_OUT=%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_v5_consumer"

:: --- validate inputs ----------------------------------------------------------
IF NOT EXIST "%PROD_OUT%\producer.nsys-rep" (
    ECHO.
    ECHO [WARN] %PROD_OUT%\producer.nsys-rep not found.
    ECHO        Was the V5-Producer window closed cleanly with Ctrl+C?
    EXIT /B 1
)
IF NOT EXIST "%CONS_OUT%\td_consumer.nsys-rep" (
    ECHO.
    ECHO [WARN] %CONS_OUT%\td_consumer.nsys-rep not found.
    ECHO        Did the consumer nsys session complete with Ctrl+C or --duration timeout?
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
    --e2e-csv "%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_v5_e2e.csv" ^
    --findings-md "%REPO_ROOT%\benchmarks\results\nsys\td_pipeline_v5_findings.md"

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
ECHO  Findings : benchmarks\results\nsys\td_pipeline_v5_findings.md
ECHO  E2E CSV  : benchmarks\results\nsys\td_pipeline_v5_e2e.csv
ECHO ============================================================
ECHO.
ECHO  Open the .nsys-rep files in nsys-ui to inspect the GPU timeline:
ECHO    nsys-ui "%PROD_OUT%\producer.nsys-rep" "%CONS_OUT%\td_consumer.nsys-rep"
ECHO.
ECHO  Acceptance criteria (compare against v4 baseline):
ECHO    - cudaStreamSynchronize absent from top-3 producer CUDA API OR avg below 30 us
ECHO    - Consumer import_frame p50 within 10%% of v4 baseline (183 us)
ECHO    - Producer effective FPS >= 59.5 (v4 baseline: 58.7 FPS)
ECHO.
ECHO  V4 baseline for comparison:
ECHO    benchmarks\results\nsys\td_pipeline_v4_findings.md

ENDLOCAL
