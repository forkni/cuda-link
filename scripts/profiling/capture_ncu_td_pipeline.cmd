@ECHO OFF
:: capture_ncu_td_pipeline.cmd -- end-to-end ncu kernel capture for the TD pipeline.
::
:: Self-contained orchestrator: launches TouchDesigner, waits for warmup, attaches ncu
:: to the Python sender, streams steady-state frames, terminates gracefully, and saves
:: benchmarks/results/ncu/<ts>/sender.ncu-rep — all without any manual window-juggling.
::
:: Workload: td_exporter/example_sender_python.py (sender) + CUDA_Link_Example.toe
:: (Receiver-A as consumer).  Uses the v5 async-flush-probe env stack so the ncu
:: trace is directly comparable to benchmarks/results/nsys/td_pipeline_v5_*.
::
:: Capture protocol (automated — 45 s total window):
::   Step 1  Launch TouchDesigner in NCU-TD-Consumer window.
::   Step 2  Wait 20 s for TD + Receiver-A warmup (until 60 fps in viewer).
::   Step 3  Launch ncu + Python sender in NCU-Sender-Capture (minimised).
::   Step 4  Wait 25 s — sender streams past --launch-skip 5 into steady state.
::   Step 5  Terminate sender gracefully (WM_CLOSE -> CTRL_CLOSE_EVENT chain via
::           SetConsoleCtrlHandler; fallback force-kill after 3 s grace period).
::   Step 6  Poll for sender.ncu-rep (1 s interval, 30 s ceiling).
::   Step 7  Close TouchDesigner.
::   Step 8  Mirror <ts>/ -> _latest/.
::   Step 9  Print report path + SOL-classification cheat-sheet; EXIT /B 0.
::
:: Usage (from repo root or double-click):
::   scripts\profiling\capture_ncu_td_pipeline.cmd              HW pass (default)
::   scripts\profiling\capture_ncu_td_pipeline.cmd full         SW pass (--set full)
::   scripts\profiling\capture_ncu_td_pipeline.cmd full <toe>   Override .toe path
::   scripts\profiling\capture_ncu_td_pipeline.cmd full <toe> <td_exe>
::
:: Pass modes:
::   (empty)   --section SpeedOfLight --section MemoryWorkloadAnalysis  --launch-count 5
::   full      --set full                                      --launch-count 2 (TDR budget)
::
:: ncu flags (mandatory -- see docs/PROFILING.md for rationale):
::   --replay-mode kernel   CUDA Graphs on export path; range/app replay re-executes
::                          graphs and breaks IPC state (PROFILING.md §5).
::   --clock-control base   WDDM cannot lock clocks via nvidia-smi -pm 1.
::   --cache-control all    Clean cache state between replays.
::   --import-source yes    Source/SASS correlation + ncu_report Python API.
::   --target-processes all ncu launches python.exe directly; safe to leave on.
::   --nvtx                 Enable NVTX collection (required for --nvtx-include).
::   --nvtx-include ...     Scope collection to cudalink.exporter.slot*@* only.
::
:: Mid-capture restart recipe (if aborted mid-run):
::   taskkill /F /FI "WINDOWTITLE eq NCU-TD-Consumer"
::   taskkill /F /FI "WINDOWTITLE eq NCU-Sender-Capture" /T
::   Then re-run this script.
::
:: Manual fallback (for ad-hoc / debugging runs with Ctrl+C control):
::   scripts\profiling\run_ncu_td_pipeline.ps1
::
:: Correctness gate note:
::   compute-sanitizer is skipped (bench_sweep.py missing locally).
::   Relies on the v5 nsys capture clean state as the proxy correctness baseline.
::
:: v2 debug additions (vs v1):
::   -- where ncu pre-flight (Step 0b): fail fast if ncu.exe not in PATH instead
::      of silently launching a window that immediately closes.
::   -- launch_ncu.bat helper (Step 0c): generated into OUT_DIR before Step 1.
::      Contains the exact ncu invocation with "> ncu.log 2>&1" redirection.
::      Double-click to replay manually; open in text editor to inspect the
::      command line.  ncu.log captures ncu's startup banner, CUPTI errors,
::      python sender stdout, and the "Profiled N kernels" summary.
::   -- Step 3 launches the helper instead of an inline "cmd /c ncu ..." string.
::      The @ in --nvtx-include is now inside a .bat, not across a cmd /c
::      boundary, so quote-stripping is no longer an issue.
::   -- Step 6 poll timeout dumps the last 40 lines of ncu.log to the console
::      and prints the helper path for manual reproduction.

SETLOCAL ENABLEDELAYEDEXPANSION

:: --- resolve repo root (two levels up from scripts\profiling\) -----------------
PUSHD "%~dp0..\.."
SET "REPO_ROOT=%CD%"
POPD

:: --- positional args -----------------------------------------------------------
IF NOT "%~1"=="" (SET "PASS=%~1")    ELSE (SET "PASS=")
IF NOT "%~2"=="" (SET "TOE=%~2")     ELSE (SET "TOE=%REPO_ROOT%\CUDA_Link_Example.toe")
IF NOT "%~3"=="" (SET "TD_EXE=%~3")  ELSE (SET "TD_EXE=C:\Program Files\Derivative\TouchDesigner.2025.32820\bin\TouchDesigner.exe")

FOR %%F IN ("%TOE%")    DO SET "TOE=%%~fF"
FOR %%F IN ("%TD_EXE%") DO SET "TD_EXE=%%~fF"

:: --- section / launch-count based on pass mode --------------------------------
IF "%PASS%"=="full" (
    SET "SECTION_FLAG=--set"
    SET "SECTION_VAL=full"
    SET "LAUNCH_COUNT=2"
) ELSE (
    SET "SECTION_FLAG=--section"
    SET "SECTION_VAL=SpeedOfLight --section MemoryWorkloadAnalysis"
    SET "LAUNCH_COUNT=5"
)

:: --- locale-safe timestamp (avoids dd/mm/yyyy vs mm/dd/yyyy ambiguity) --------
FOR /F "delims=" %%i IN ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HHmmss"') DO SET "TS=%%i"

:: --- output directory ---------------------------------------------------------
SET "OUT_DIR=%REPO_ROOT%\benchmarks\results\ncu\%TS%"
IF NOT EXIST "%OUT_DIR%" MKDIR "%OUT_DIR%"

:: --- v5 async-flush-probe env (matches scripts\probes\v5_capture_producer.cmd:61-67)
SET CUDALINK_TD_PERSIST_STREAM=1
SET CUDALINK_TD_STREAM_PRIO=normal
SET CUDALINK_EXPORT_SYNC=0
SET CUDALINK_EXPORT_FLUSH_PROBE=1
SET CUDALINK_LIB_STREAM_PRIO=high
SET CUDALINK_NVTX=1
SET CUDALINK_NVTX_VERBOSE=1
:: Disable Intel Fortran Runtime's console handler so Python's CTRL_CLOSE handler
:: (example_sender_python.py:106-116) can run _do_cleanup() on WM_CLOSE.  Without
:: this, libfortran (injected via an ncu CUPTI DLL) aborts with forrtl error 200.
SET FOR_DISABLE_CONSOLE_CTRL_HANDLER=1

:: --- env snapshot for traceability -------------------------------------------
SET > "%OUT_DIR%\env.txt"
ECHO [OK] Env snapshot: %OUT_DIR%\env.txt

:: --- sanity checks ------------------------------------------------------------
IF NOT EXIST "%TOE%" (
    ECHO [FAIL] .toe not found: %TOE%
    ECHO        Provide path as second arg or copy CUDA_Link_Example.toe to repo root.
    EXIT /B 2
)
IF NOT EXIST "%TD_EXE%" (
    ECHO [FAIL] TouchDesigner.exe not found: %TD_EXE%
    ECHO        Provide path as third arg or update DEFAULT_TD in this script.
    EXIT /B 2
)

:: --- Step 0b: ncu pre-flight --------------------------------------------------
where ncu >NUL 2>&1
IF ERRORLEVEL 1 (
    ECHO [FAIL] ncu.exe not on PATH.
    ECHO        Install Nsight Compute and add its target\windows-desktop-win7-x64
    ECHO        directory to PATH, or run from the Nsight Compute developer prompt.
    EXIT /B 4
)
ECHO [OK]   ncu found: & where ncu

:: --- Step 0c: generate launch_ncu.bat helper ----------------------------------
:: Self-contained helper that replays the ncu invocation with full stdout/stderr
:: capture.  Avoids cmd /c "..." quote-stripping on the @ in --nvtx-include.
> "%OUT_DIR%\launch_ncu.bat" (
    ECHO @ECHO OFF
    ECHO :: Generated by capture_ncu_td_pipeline.cmd at %DATE% %TIME%
    ECHO :: Re-run manually for diagnostics ^(with TD already running^):
    ECHO ::   "%OUT_DIR%\launch_ncu.bat"
    ECHO cd /D "%REPO_ROOT%"
    ECHO ncu %SECTION_FLAG% %SECTION_VAL% --clock-control base --cache-control all --import-source yes --launch-skip 5 --launch-count %LAUNCH_COUNT% --replay-mode kernel --target-processes all --nvtx --nvtx-include "cudalink.exporter.slot*@*" -o "%OUT_DIR%\sender" python td_exporter\example_sender_python.py ^> "%OUT_DIR%\ncu.log" 2^>^&1
    ECHO EXIT /B %%ERRORLEVEL%%
)
ECHO [OK]   Helper written: %OUT_DIR%\launch_ncu.bat

:: --- header -------------------------------------------------------------------
ECHO.
ECHO ============================================================
ECHO  ncu TD-pipeline capture  (%SECTION_FLAG% %SECTION_VAL%)
ECHO  Pass       : %PASS% (empty = HW pass)
ECHO  Launch cnt : %LAUNCH_COUNT%
ECHO  Output     : %OUT_DIR%\sender.ncu-rep
ECHO  .toe       : %TOE%
ECHO  Env        : EXPORT_SYNC=0  FLUSH_PROBE=1  PERSIST_STREAM=1
ECHO  Timeline   : 20 s warmup + 25 s capture = 45 s total
ECHO ============================================================
ECHO.

:: ===========================================================================
:: STEP 1: Launch TouchDesigner
:: ===========================================================================
ECHO [Step 1/9] Launching TouchDesigner (NCU-TD-Consumer) ...
START "NCU-TD-Consumer" "%TD_EXE%" "%TOE%"

:: ===========================================================================
:: STEP 2: Warmup wait — TD loads and Receiver-A connects
:: ===========================================================================
ECHO [Step 2/9] Waiting 20 s for TD + Receiver-A warmup ...
TIMEOUT /T 20 /NOBREAK >NUL

:: ===========================================================================
:: STEP 3: Launch launch_ncu.bat helper in its own window.
::
:: Child cmd inherits parent env (v5 stack already set above).
:: /D sets working dir to repo root so relative paths resolve correctly.
:: cmd /c closes the window automatically when the helper exits.
:: Set CUDALINK_NCU_MIN=1 before running to minimise the window.
:: The helper redirects all ncu output to ncu.log — inspect on timeout.
:: ===========================================================================
ECHO [Step 3/9] Launching ncu + Python sender via helper (NCU-Sender-Capture) ...
IF "%CUDALINK_NCU_MIN%"=="1" (
    START "NCU-Sender-Capture" /MIN /D "%REPO_ROOT%" cmd /c "%OUT_DIR%\launch_ncu.bat"
) ELSE (
    START "NCU-Sender-Capture" /D "%REPO_ROOT%" cmd /c "%OUT_DIR%\launch_ncu.bat"
)

:: ===========================================================================
:: STEP 4: Steady-state wait — sender streams past launch-skip into capture
:: ===========================================================================
ECHO [Step 4/9] Waiting 25 s for steady-state frames past --launch-skip 5 ...
TIMEOUT /T 25 /NOBREAK >NUL

:: ===========================================================================
:: STEP 5: Graceful termination.
::
:: taskkill without /F sends WM_CLOSE to the console window, which generates
:: CTRL_CLOSE_EVENT for the entire console group (ncu + python sender).
:: example_sender_python.py:106-116 catches CTRL_CLOSE_EVENT, runs _do_cleanup,
:: returns True — OS terminates python.  ncu detects target exit, writes report.
:: Force-kill fallback after 3 s grace period in case WM_CLOSE was not enough.
:: ===========================================================================
ECHO [Step 5/9] Terminating sender gracefully (WM_CLOSE) ...
taskkill /FI "WINDOWTITLE eq NCU-Sender-Capture" >NUL 2>&1
TIMEOUT /T 8 /NOBREAK >NUL
taskkill /F /FI "WINDOWTITLE eq NCU-Sender-Capture" /T >NUL 2>&1

:: ===========================================================================
:: STEP 6: Poll for sender.ncu-rep (ncu writes on its own exit).
::
:: Cap at 30 s — protects against SW-pass replay tail where report finalisation
:: can lag the target exit by a few seconds.  Exits non-zero if not found.
:: ===========================================================================
ECHO [Step 6/9] Waiting for ncu to finalize sender.ncu-rep ...
SET "REPORT=%OUT_DIR%\sender.ncu-rep"
SET /A POLL=0
:POLL_LOOP
IF EXIST "%REPORT%" GOTO POLL_DONE
IF %POLL% GEQ 30 (
    ECHO.
    ECHO [FAIL] sender.ncu-rep not found after 30 s.
    ECHO        --- ncu.log ^(last 40 lines^) ---
    IF EXIST "%OUT_DIR%\ncu.log" (
        powershell -NoProfile -Command "Get-Content -Tail 40 '%OUT_DIR%\ncu.log'"
    ) ELSE (
        ECHO        ncu.log not created -- helper batch never ran or ncu not found.
    )
    ECHO        --- end of log tail ---
    ECHO        Replay manually ^(with TD running^):  "%OUT_DIR%\launch_ncu.bat"
    ECHO        Partial output may be in: %OUT_DIR%
    EXIT /B 3
)
TIMEOUT /T 1 /NOBREAK >NUL
SET /A POLL+=1
GOTO POLL_LOOP
:POLL_DONE
ECHO [OK]   Report found: %REPORT%

:: ===========================================================================
:: STEP 7: Close TouchDesigner
:: ===========================================================================
ECHO [Step 7/9] Closing TouchDesigner ...
taskkill /FI "WINDOWTITLE eq NCU-TD-Consumer" >NUL 2>&1
TIMEOUT /T 3 /NOBREAK >NUL
taskkill /F /FI "WINDOWTITLE eq NCU-TD-Consumer" /T >NUL 2>&1

:: ===========================================================================
:: STEP 8: Mirror timestamped dir -> _latest
:: ===========================================================================
ECHO [Step 8/9] Mirroring to benchmarks\results\ncu\_latest ...
SET "LATEST=%REPO_ROOT%\benchmarks\results\ncu\_latest"
IF EXIST "%LATEST%" RMDIR /S /Q "%LATEST%"
XCOPY /E /I /Q /Y "%OUT_DIR%" "%LATEST%" >NUL

:: ===========================================================================
:: STEP 9: Summary
:: ===========================================================================
ECHO [Step 9/9] Done.
ECHO.
ECHO ============================================================
ECHO  Capture complete
ECHO ============================================================
ECHO  Report  : %OUT_DIR%\sender.ncu-rep
ECHO  Open    : ncu-ui benchmarks\results\ncu\_latest\sender.ncu-rep
ECHO.
ECHO  SOL classification (docs\PROFILING.md §3):
ECHO    DRAM ^>= 80%%  -- memory-bandwidth bound (expected for D2D copy)
ECHO    SM   ^>= 60%%  -- compute-bound (unexpected; investigate)
ECHO    Both ^<  30%%  -- latency- or congestion-bound (investigate)
ECHO.
IF "%PASS%"=="full" (
    ECHO  SW pass complete.  Write findings to:
    ECHO    %OUT_DIR%\findings.md
    ECHO.
    ECHO  Sections to fill in  (PROFILING.md §3^):
    ECHO    Run metadata, HW pass SOL table, SW pass WarpStateStats,
    ECHO    Sectors/Request, L1/L2 hit rates, comparison vs v3 figures,
    ECHO    correctness-gate note (compute-sanitizer skipped), open items.
) ELSE (
    ECHO  HW pass complete.  Re-run with "full" for SW pass:
    ECHO    scripts\profiling\capture_ncu_td_pipeline.cmd full
)
ECHO ============================================================

ENDLOCAL
EXIT /B 0
