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

Environment variables (set before launching TouchDesigner):
    CUDALINK_RECEIVER_PYTHON_EXE
        Python executable to use for the receiver subprocess.
        Default: "python" (resolves via PATH — may not be the system Python that
        has torch/cupy if TD's bundled Python appears first in PATH).
        Set to the full path of your preferred interpreter, e.g.:
            C:\\Python312\\python.exe
            C:\\Users\\you\\AppData\\Local\\Programs\\Python\\Python312\\python.exe
        Any CUDALINK_RECEIVER_* env vars are forwarded automatically to the
        receiver subprocess through the inherited environment.
"""

import os
import signal
import subprocess

_process = None  # Receiver subprocess handle


def onStart() -> None:
    """Launch the Python receiver as a separate subprocess."""
    global _process

    script = os.path.join(project.folder, "td_exporter", "example_receiver_python.py")

    if not os.path.isfile(script):
        print("[CUDA-Link Receiver Launcher] ERROR: receiver script not found:")
        print(f"  {script}")
        return

    python_exe = os.environ.get("CUDALINK_RECEIVER_PYTHON_EXE", "python")

    _process = subprocess.Popen(
        [python_exe, script],
        # CREATE_NEW_CONSOLE: opens a visible console window for the receiver.
        # CREATE_NEW_PROCESS_GROUP: required to send CTRL_BREAK_EVENT on shutdown
        # (CTRL_C_EVENT is blocked for new process groups on Windows).
        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    print(f"[CUDA-Link Receiver Launcher] Receiver subprocess started  (PID {_process.pid})")
    print(f"  Script:     {script}")
    print(f"  Python exe: {python_exe}")


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
            print(
                f"[CUDA-Link Receiver Launcher] WARNING: receiver subprocess exited unexpectedly (code={code})."
            )


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
