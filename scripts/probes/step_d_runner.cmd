@ECHO OFF
:: Step D — F9-drop probe runner (Phase 3.6)
::
:: Launches example_sender_python.py with the F9 env vars explicitly cleared
:: so env-state leaks from prior probe runs cannot contaminate this run.
::
:: Usage: double-click or run from repo root in any cmd.exe window.
:: After the sender starts, perform the TD probe protocol:
::   1. Open the production .toe (Pair A streaming at 60 fps).
::   2. Toggle Sender-B Active OFF -> ON >= 3 cycles, ~30 s each.
::   3. Save the TouchDesigner textport to <artifact_dir>\td_sender.txt.
::   4. Close the sender (Ctrl+C or X button).
:: Then run:  python scripts\probes\scan_step_d.py <artifact_dir>

SETLOCAL ENABLEDELAYEDEXPANSION

:: --- locate repo root (directory containing this script's parent) -----------
SET "SCRIPT_DIR=%~dp0"
SET "REPO_ROOT=%SCRIPT_DIR%..\.."

:: normalise to absolute path
PUSHD "%REPO_ROOT%"
SET "REPO_ROOT=%CD%"
POPD

:: --- timestamped artifact directory -----------------------------------------
FOR /F "tokens=1-6 delims=/:. " %%a IN ("%DATE% %TIME%") DO (
    SET "YY=%%a"
    SET "MM=%%b"
    SET "DD=%%c"
    SET "HH=%%d"
    SET "MIN=%%e"
    SET "SS=%%f"
)
:: Handle single-digit hours (time like " 9:05") — pad with leading zero
IF "!HH:~0,1!"==" " SET "HH=0!HH:~1!"
SET "TS=!YY!!MM!!DD!-!HH!!MIN!!SS!"
SET "ARTIFACT_DIR=%REPO_ROOT%\logs\probes\step_d_!TS!"

IF NOT EXIST "%ARTIFACT_DIR%" MKDIR "%ARTIFACT_DIR%"

ECHO ============================================================
ECHO  Step D — F9-drop probe runner
ECHO  Artifact dir: %ARTIFACT_DIR%
ECHO ============================================================

:: --- hard-clear F9 vars (immune to caller env leaks) ------------------------
SET CUDALINK_TD_ACTIVATION_BARRIER=
SET CUDALINK_ACTIVATION_BARRIER=

:: --- set the kept stack ------------------------------------------------------
SET CUDALINK_TD_STREAM_PRIO=normal
SET CUDALINK_TD_PERSIST_STREAM=1
SET CUDALINK_EXPORT_SYNC=1
SET CUDALINK_LIB_STREAM_PRIO=high

:: --- pre-flight: assert F9 vars are unset -----------------------------------
IF DEFINED CUDALINK_TD_ACTIVATION_BARRIER (
    ECHO [FAIL] CUDALINK_TD_ACTIVATION_BARRIER is still set in this process env.
    ECHO        This should be impossible — check SETLOCAL / environment inheritance.
    EXIT /B 2
)
IF DEFINED CUDALINK_ACTIVATION_BARRIER (
    ECHO [FAIL] CUDALINK_ACTIVATION_BARRIER is still set in this process env.
    ECHO        This should be impossible — check SETLOCAL / environment inheritance.
    EXIT /B 2
)

:: --- dump full env to artifact dir for post-run audit -----------------------
SET > "%ARTIFACT_DIR%\env.txt"
ECHO [OK] Env snapshot: %ARTIFACT_DIR%\env.txt
ECHO      F9 vars (should be absent from snapshot):
FINDSTR /I "CUDALINK_TD_ACTIVATION_BARRIER\|CUDALINK_ACTIVATION_BARRIER" "%ARTIFACT_DIR%\env.txt" || ECHO      (none found — correct)

:: --- launch producer with log capture ----------------------------------------
ECHO.
ECHO [INFO] Launching producer... logging to %ARTIFACT_DIR%\producer.log
ECHO [INFO] Now perform the TD probe protocol in TouchDesigner.
ECHO [INFO] Remember to Save Textport as: %ARTIFACT_DIR%\td_sender.txt
ECHO.

CD /D "%REPO_ROOT%"
python td_exporter\example_sender_python.py > "%ARTIFACT_DIR%\producer.log" 2>&1

ECHO.
ECHO [INFO] Sender exited.  Artifact dir: %ARTIFACT_DIR%
ECHO.
ECHO [NEXT] Run the scanner:
ECHO        python scripts\probes\scan_step_d.py "%ARTIFACT_DIR%"
ECHO.
PAUSE

ENDLOCAL
