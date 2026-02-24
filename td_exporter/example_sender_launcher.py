"""
Execute DAT — CUDA-Link Python → TouchDesigner Launcher

Paste this into an Execute DAT in your example project.
Enable "Start", "Frame Start", and "On Exit" toggles.

This DAT spawns example_sender_python.py as a separate OS process on project
start and terminates it on exit. CUDA IPC requires separate processes — sender
and receiver cannot share GPU handles within the same process.

Pipeline:
    onStart()  →  subprocess.Popen(example_sender_python.py)
                         ↓  CUDA IPC  (cudalink_output_ipc)
               CUDAIPCLink_from_Python  (Receiver mode, same project)
                         ↓
               Script TOP output  →  cycling solid colors

TD Setup:
    1. Add CUDAIPCLink_from_Python component to the network
    2. Set Mode       → Receiver
    3. Set Ipcmemname → cudalink_output_ipc
    4. Set Active     → ON
    5. Paste THIS script into an Execute DAT — enable Start, Frame Start, On Exit
    6. Press Play (or reopen the project) to trigger onStart()
"""

import os
import subprocess

_process = None  # Sender subprocess handle


def onStart() -> None:
    """Launch the Python sender as a separate subprocess."""
    global _process

    script = os.path.join(project.folder, "td_exporter", "example_sender_python.py")

    if not os.path.isfile(script):
        print("[CUDA-Link Launcher] ERROR: sender script not found:")
        print(f"  {script}")
        return

    _process = subprocess.Popen(
        ["python", script],
        creationflags=subprocess.CREATE_NEW_CONSOLE,  # Windows: opens a visible console
    )
    print(f"[CUDA-Link Launcher] Sender subprocess started  (PID {_process.pid})")
    print(f"  Script: {script}")


def onCreate() -> None:
    return


def onExit() -> None:
    """Terminate the sender subprocess when the project closes."""
    global _process

    if _process is None:
        return

    if _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=3)
            print(f"[CUDA-Link Launcher] Sender subprocess terminated (PID {_process.pid}).")
        except subprocess.TimeoutExpired:
            _process.kill()
            print(f"[CUDA-Link Launcher] Sender subprocess force-killed (PID {_process.pid}).")

    _process = None


def onFrameStart(frame: int) -> None:
    """Check if the subprocess is still running; warn if it exited unexpectedly."""
    if _process is not None and _process.poll() is not None:
        code = _process.returncode
        if code != 0:
            print(f"[CUDA-Link Launcher] WARNING: sender subprocess exited unexpectedly (code={code}).")


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
