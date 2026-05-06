@ECHO OFF
:: Step D — F9-drop probe runner (Phase 3.6)
::
:: Launches TouchDesigner with the probe .toe so the Python producer subprocess
:: inherits a clean env (F9 vars explicitly cleared, kept-stack vars set).
:: The producer writes its log to <artifact_dir>\producer.log via the
:: CUDALINK_PROBE_LOG_FILE FileHandler hook in example_sender_python.py.
::
:: Usage (from repo root or any cmd.exe window):
::   scripts\probes\step_d_runner.cmd  [<toe_path>]  [<td_exe_path>]
::
:: Defaults (override as positional args if your installation differs):
::   toe_path   = <repo_root>\CUDA_Link_Example.toe
::   td_exe     = C:\Program Files\Derivative\TouchDesigner.2025.32460\bin\TouchDesigner.exe
::
:: After TD opens, perform the probe protocol:
::   1. Wait for Pair A to be streaming at 60 fps.
::   2. Toggle Sender-B Active OFF -> ON, wait ~30 s.  Repeat >=3 cycles.
::   3. After the last cycle: Dialogs > Textport and DATs > Save...
::      Save the textport as:  <artifact_dir>\td_sender.txt
::   4. Close TouchDesigner.
:: Then run:
::   python scripts\probes\scan_step_d.py <artifact_dir>

SETLOCAL ENABLEDELAYEDEXPANSION

:: --- default paths ----------------------------------------------------------
SET "DEFAULT_TOE=%~dp0..\..\CUDA_Link_Example.toe"
SET "DEFAULT_TD=C:\Program Files\Derivative\TouchDesigner.2025.32460\bin\TouchDesigner.exe"

IF NOT "%~1"=="" (SET "TOE_PATH=%~1") ELSE (SET "TOE_PATH=%DEFAULT_TOE%")
IF NOT "%~2"=="" (SET "TD_EXE=%~2") ELSE (SET "TD_EXE=%DEFAULT_TD%")

:: normalise to absolute paths
FOR %%F IN ("%TOE_PATH%") DO SET "TOE_PATH=%%~fF"
FOR %%F IN ("%TD_EXE%")  DO SET "TD_EXE=%%~fF"

:: --- sanity checks ----------------------------------------------------------
IF NOT EXIST "%TOE_PATH%" (
    ECHO [FAIL] .toe not found: %TOE_PATH%
    EXIT /B 2
)
IF NOT EXIST "%TD_EXE%" (
    ECHO [FAIL] TouchDesigner.exe not found: %TD_EXE%
    EXIT /B 2
)

:: --- timestamped artifact directory -----------------------------------------
FOR /F "tokens=1-6 delims=/:. " %%a IN ("%DATE% %TIME%") DO (
    SET "YY=%%a"
    SET "MM=%%b"
    SET "DD=%%c"
    SET "HH=%%d"
    SET "MIN=%%e"
    SET "SS=%%f"
)
IF "!HH:~0,1!"==" " SET "HH=0!HH:~1!"
SET "TS=!YY!!MM!!DD!-!HH!!MIN!!SS!"

:: derive repo root from this script's location (scripts\probes\)
PUSHD "%~dp0..\.."
SET "REPO_ROOT=%CD%"
POPD

SET "ARTIFACT_DIR=%REPO_ROOT%\logs\probes\step_d_!TS!"
IF NOT EXIST "%ARTIFACT_DIR%" MKDIR "%ARTIFACT_DIR%"

ECHO ============================================================
ECHO  Step D  --  F9-drop probe runner
ECHO  Artifact: %ARTIFACT_DIR%
ECHO ============================================================

:: --- hard-clear F9 vars in THIS process env (SETLOCAL makes it isolated) ---
SET CUDALINK_TD_ACTIVATION_BARRIER=
SET CUDALINK_ACTIVATION_BARRIER=

:: --- set the kept stack ------------------------------------------------------
SET CUDALINK_TD_STREAM_PRIO=normal
SET CUDALINK_TD_PERSIST_STREAM=1
SET CUDALINK_EXPORT_SYNC=1
SET CUDALINK_LIB_STREAM_PRIO=high

:: --- wire producer logging to artifact dir ----------------------------------
SET "CUDALINK_PROBE_LOG_FILE=%ARTIFACT_DIR%\producer.log"

:: --- pre-flight: assert F9 vars are unset -----------------------------------
IF DEFINED CUDALINK_TD_ACTIVATION_BARRIER (
    ECHO [FAIL] CUDALINK_TD_ACTIVATION_BARRIER still defined after explicit clear.
    EXIT /B 2
)
IF DEFINED CUDALINK_ACTIVATION_BARRIER (
    ECHO [FAIL] CUDALINK_ACTIVATION_BARRIER still defined after explicit clear.
    EXIT /B 2
)

:: --- dump full env to artifact (audit trail) --------------------------------
SET > "%ARTIFACT_DIR%\env.txt"
ECHO [OK] Env snapshot written to: %ARTIFACT_DIR%\env.txt
ECHO      F9 vars (should be absent):
FINDSTR /I "CUDALINK_TD_ACTIVATION_BARRIER CUDALINK_ACTIVATION_BARRIER" "%ARTIFACT_DIR%\env.txt" || ECHO      (none found -- correct)
ECHO.

:: --- launch TouchDesigner with the .toe -------------------------------------
ECHO [INFO] Launching TouchDesigner:
ECHO        %TD_EXE%
ECHO        %TOE_PATH%
ECHO.
ECHO [INFO] Probe protocol (perform in TD after Pair A is streaming at 60 fps):
ECHO    1. Toggle Sender-B Active  OFF -> ON   (wait ~30 s)
ECHO    2. Toggle Sender-B Active  OFF -> ON   (wait ~30 s)
ECHO    3. Toggle Sender-B Active  OFF -> ON   (wait ~30 s)
ECHO    4. Dialogs > Textport and DATs > Save...
ECHO       Save as:  %ARTIFACT_DIR%\td_sender.txt
ECHO    5. Close TouchDesigner.
ECHO.

START "" /WAIT "%TD_EXE%" "%TOE_PATH%"

ECHO.
ECHO [INFO] TouchDesigner exited.
ECHO.
ECHO [NEXT] Run the scanner:
ECHO    python scripts\probes\scan_step_d.py "%ARTIFACT_DIR%"
ECHO.
PAUSE

ENDLOCAL
