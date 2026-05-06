@ECHO OFF
:: Step D -- F9-drop probe runner (Phase 3.6)
::
:: Launches TWO TouchDesigner instances so the full Sender-B -> Receiver-B
:: TD-to-TD pipeline is exercised, plus the Python producer -> Receiver-A
:: pipeline inside the sender .toe.
::
:: Topology:
::   Python producer -> Pair A receiver (inside CUDA_Link_Example.toe)
::   TD Sender-B     -> TD-B receiver  (inside Test_TD.toe, separate process)
::
:: Both TD instances inherit this script's clean env (F9 vars cleared,
:: kept-stack vars set, CUDALINK_PROBE_LOG_FILE set).
::
:: Usage (from repo root or any cmd.exe window):
::   scripts\probes\step_d_runner.cmd  [<sender_toe>]  [<receiver_toe>]  [<td_exe>]
::
:: Defaults (override as positional args):
::   sender_toe   = <repo_root>\CUDA_Link_Example.toe
::   receiver_toe = <repo_root>\Test_TD.toe
::   td_exe       = C:\Program Files\Derivative\TouchDesigner.2025.32460\bin\TouchDesigner.exe
::
:: After both TD windows open, perform the probe protocol:
::   1. Confirm Pair A (Python -> Receiver-A) is streaming at 60 fps inside
::      CUDA_Link_Example.toe.
::   2. Confirm Test_TD.toe shows the Receiver-B pulling from Sender-B.
::   3. Toggle Sender-B Active OFF -> ON, wait ~30 s.  Repeat >=3 cycles.
::   4. Save BOTH textports (Dialogs > Textport and DATs > Save...):
::        Sender side   ->  <artifact_dir>\td_sender.txt
::        Receiver side ->  <artifact_dir>\td_receiver.txt
::   5. Close both TouchDesigner windows.
:: Then run:
::   python scripts\probes\scan_step_d.py <artifact_dir>

SETLOCAL ENABLEDELAYEDEXPANSION

:: --- default paths ----------------------------------------------------------
SET "DEFAULT_SENDER_TOE=%~dp0..\..\CUDA_Link_Example.toe"
SET "DEFAULT_RECEIVER_TOE=%~dp0..\..\Test_TD.toe"
SET "DEFAULT_TD=C:\Program Files\Derivative\TouchDesigner.2025.32460\bin\TouchDesigner.exe"

IF NOT "%~1"=="" (SET "SENDER_TOE=%~1")    ELSE (SET "SENDER_TOE=%DEFAULT_SENDER_TOE%")
IF NOT "%~2"=="" (SET "RECEIVER_TOE=%~2")  ELSE (SET "RECEIVER_TOE=%DEFAULT_RECEIVER_TOE%")
IF NOT "%~3"=="" (SET "TD_EXE=%~3")        ELSE (SET "TD_EXE=%DEFAULT_TD%")

:: normalise to absolute paths
FOR %%F IN ("%SENDER_TOE%")   DO SET "SENDER_TOE=%%~fF"
FOR %%F IN ("%RECEIVER_TOE%") DO SET "RECEIVER_TOE=%%~fF"
FOR %%F IN ("%TD_EXE%")       DO SET "TD_EXE=%%~fF"

:: --- sanity checks ----------------------------------------------------------
IF NOT EXIST "%SENDER_TOE%" (
    ECHO [FAIL] Sender .toe not found: %SENDER_TOE%
    EXIT /B 2
)
IF NOT EXIST "%RECEIVER_TOE%" (
    ECHO [FAIL] Receiver .toe not found: %RECEIVER_TOE%
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

PUSHD "%~dp0..\.."
SET "REPO_ROOT=%CD%"
POPD

SET "ARTIFACT_DIR=%REPO_ROOT%\logs\probes\step_d_!TS!"
IF NOT EXIST "%ARTIFACT_DIR%" MKDIR "%ARTIFACT_DIR%"

ECHO ============================================================
ECHO  Step D  --  F9-drop probe runner  (TD->TD pipeline)
ECHO  Artifact: %ARTIFACT_DIR%
ECHO ============================================================

:: --- hard-clear F9 vars ------------------------------------------------------
SET CUDALINK_TD_ACTIVATION_BARRIER=
SET CUDALINK_ACTIVATION_BARRIER=

:: --- set the kept stack ------------------------------------------------------
SET CUDALINK_TD_STREAM_PRIO=normal
SET CUDALINK_TD_PERSIST_STREAM=1
SET CUDALINK_EXPORT_SYNC=1
SET CUDALINK_LIB_STREAM_PRIO=high

:: --- wire producer logging --------------------------------------------------
SET "CUDALINK_PROBE_LOG_FILE=%ARTIFACT_DIR%\producer.log"

:: --- pre-flight assertions --------------------------------------------------
IF DEFINED CUDALINK_TD_ACTIVATION_BARRIER (
    ECHO [FAIL] CUDALINK_TD_ACTIVATION_BARRIER still defined after explicit clear.
    EXIT /B 2
)
IF DEFINED CUDALINK_ACTIVATION_BARRIER (
    ECHO [FAIL] CUDALINK_ACTIVATION_BARRIER still defined after explicit clear.
    EXIT /B 2
)

:: --- env snapshot -----------------------------------------------------------
SET > "%ARTIFACT_DIR%\env.txt"
ECHO [OK] Env snapshot: %ARTIFACT_DIR%\env.txt
ECHO      F9 vars (should be absent):
FINDSTR /I "CUDALINK_TD_ACTIVATION_BARRIER CUDALINK_ACTIVATION_BARRIER" "%ARTIFACT_DIR%\env.txt" || ECHO      (none found -- correct)
ECHO.

:: --- launch both TD instances -----------------------------------------------
ECHO [INFO] TD path: %TD_EXE%
ECHO [INFO] Sender   .toe: %SENDER_TOE%
ECHO [INFO] Receiver .toe: %RECEIVER_TOE%
ECHO.
ECHO [INFO] Launching Sender TD (Python producer + Sender-B + Receiver-A)...
START "TD-Sender" "%TD_EXE%" "%SENDER_TOE%"

:: small stagger so two TD processes don't fight for the same WDDM init slot
TIMEOUT /T 3 /NOBREAK >NUL

ECHO [INFO] Launching Receiver TD (Receiver-B for Sender-B output)...
START "TD-Receiver" "%TD_EXE%" "%RECEIVER_TOE%"
ECHO.
ECHO ============================================================
ECHO  Probe protocol:
ECHO    1. Wait until Pair A (Python -> Receiver-A) is streaming at 60 fps
ECHO       inside the Sender TD.
ECHO    2. Wait until Test_TD.toe shows Receiver-B pulling from Sender-B.
ECHO    3. Toggle Sender-B Active  OFF -> ON   (wait ~30 s)  -- cycle 1
ECHO    4. Toggle Sender-B Active  OFF -> ON   (wait ~30 s)  -- cycle 2
ECHO    5. Toggle Sender-B Active  OFF -> ON   (wait ~30 s)  -- cycle 3
ECHO    6. Save BOTH textports (Dialogs ^> Textport and DATs ^> Save...):
ECHO         Sender   --^>  %ARTIFACT_DIR%\td_sender.txt
ECHO         Receiver --^>  %ARTIFACT_DIR%\td_receiver.txt
ECHO    7. Close both TouchDesigner windows.
ECHO ============================================================
ECHO.
ECHO Press any key in this window AFTER both TD windows are closed
ECHO and both textports are saved.
PAUSE

ECHO.
ECHO [NEXT] Run the scanner:
ECHO    python scripts\probes\scan_step_d.py "%ARTIFACT_DIR%"
ECHO.

ENDLOCAL
