"""
Execute DAT — CUDA-Link TouchDesigner → Python Receiver Launcher

Paste this into an Execute DAT in your example project.
Enable "Start", "Frame Start", and "On Exit" toggles.

This DAT spawns example_receiver_python.py as a separate OS process on project
start and terminates it on exit. CUDA IPC requires separate processes — sender
and receiver cannot share GPU handles within the same process.

Pipeline:
    CUDAIPCLink_to_Python  (Sender mode, Active=ON, Ipcmemname=cudalink_input_ipc)
         ↓  CUDA IPC
    onStart()  →  subprocess.Popen(example_receiver_python.py)
         ↓
    Prints frame stats in its own console window

TD Setup:
    1. Add a CUDAIPCLink_to_Python component to the network
    2. Set Mode       → Sender
    3. Set Ipcmemname → cudalink_input_ipc
    4. Set Active     → ON
    5. Paste THIS script into an Execute DAT — enable Start, Frame Start, On Exit
    6. Press Play (or reopen the project) to trigger onStart()

Python executable resolution (priority order):
    1. CUDALINK_RECEIVER_PYTHON_EXE env var — full path, highest priority.
    2. Windows Python Launcher: 'py -3' resolves the system Python 3 installation
       and returns its full path (e.g. C:\\Users\\...\\Python311\\python.exe).
       Reliable on any Windows machine with the standard Python installer.
    3. 'python' — bare fallback if the Launcher is unavailable (may be TD's
       bundled Python, which lacks third-party packages like torch/cupy).

    The resolved path is printed on each onStart() so you can verify which
    interpreter is used without opening a terminal.
"""

import os
import shutil
import signal
import subprocess


def _find_python_exe() -> str:
    """Resolve the Python executable for the receiver subprocess.

    Runs once at Execute DAT load time so the path is ready before onStart().
    """
    # 1. Explicit env-var override — highest priority.
    if env := os.environ.get("CUDALINK_RECEIVER_PYTHON_EXE", ""):
        return env

    # 2. Windows Python Launcher: 'py -3' always resolves the registered system
    #    Python 3, regardless of PATH order.  Ask it for sys.executable so we
    #    get the full absolute path rather than relying on 'py' staying available.
    if shutil.which("py"):
        try:
            result = subprocess.run(
                ["py", "-3", "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

    # 3. Bare fallback — first 'python' in PATH.
    return "python"


_RECEIVER_PYTHON_EXE = _find_python_exe()

_process = None  # Receiver subprocess handle


def onStart() -> None:
    """Launch the Python receiver as a separate subprocess."""
    global _process

    script = os.path.join(project.folder, "td_exporter", "example_receiver_python.py")

    if not os.path.isfile(script):
        print("[CUDA-Link Receiver Launcher] ERROR: receiver script not found:")
        print(f"  {script}")
        return

    _process = subprocess.Popen(
        [_RECEIVER_PYTHON_EXE, script],
        # CREATE_NEW_CONSOLE: opens a visible console window for the receiver.
        # CREATE_NEW_PROCESS_GROUP: required to send CTRL_BREAK_EVENT on shutdown
        # (CTRL_C_EVENT is blocked for new process groups on Windows).
        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    print(f"[CUDA-Link Receiver Launcher] Receiver subprocess started  (PID {_process.pid})")
    print(f"  Script:     {script}")
    print(f"  Python exe: {_RECEIVER_PYTHON_EXE}")


def onCreate() -> None:
    return


def onExit() -> None:
    """Terminate the receiver subprocess when the project closes."""
    global _process

    if _process is None:
        return

    if _process.poll() is None:
        pid = _process.pid
        try:
            # CTRL_BREAK_EVENT gives the Python receiver a chance to run its IPC cleanup.
            # CTRL_C_EVENT cannot cross CREATE_NEW_PROCESS_GROUP boundaries on Windows;
            # CTRL_BREAK_EVENT can.
            _process.send_signal(signal.CTRL_BREAK_EVENT)
            _process.wait(timeout=3)
            print(f"[CUDA-Link Receiver Launcher] Receiver subprocess exited gracefully (PID {pid}).")
        except subprocess.TimeoutExpired:
            _process.terminate()
            try:
                _process.wait(timeout=2)
                print(f"[CUDA-Link Receiver Launcher] Receiver subprocess terminated (PID {pid}).")
            except subprocess.TimeoutExpired:
                _process.kill()
                print(f"[CUDA-Link Receiver Launcher] Receiver subprocess force-killed (PID {pid}).")
        except OSError:
            _process.kill()
            print(f"[CUDA-Link Receiver Launcher] Receiver subprocess force-killed (PID {pid}).")

    _process = None


def onFrameStart(frame: int) -> None:
    """Check if the subprocess is still running; warn if it exited unexpectedly."""
    if _process is not None and _process.poll() is not None:
        code = _process.returncode
        if code != 0:
            print(f"[CUDA-Link Receiver Launcher] WARNING: receiver subprocess exited unexpectedly (code={code}).")


def onFrameEnd(frame: int) -> None:
    return


def onPlayStateChange(state: bool) -> None:
    return


def onDeviceChange() -> None:
    return


def onProjectPreSave() -> None:
    return


def onProjectPostSave() -> None:
    return
