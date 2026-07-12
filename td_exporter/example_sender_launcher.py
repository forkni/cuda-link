"""
Execute DAT — CUDA-Link Python → TouchDesigner Launcher

Paste this into an Execute DAT in your example project.
Enable "Start", "Frame Start", and "On Exit" toggles.

This DAT spawns example_sender_python.py as a separate OS process on project
start and terminates it on exit. CUDA IPC requires separate processes — sender
and receiver cannot share GPU handles within the same process.

Pipeline:
    onStart()  →  subprocess.Popen(example_sender_python.py)
                         ↓  CUDA IPC  (cudalink_ipc_Python>>TD)
               CUDAIPCLink_from_Python  (Receiver mode, same project)
                         ↓
               Script TOP output  →  cycling solid colors

TD Setup:
    1. Add CUDAIPCLink_from_Python component to the network
    2. Set Mode       → Receiver
    3. Set Ipcmemname → cudalink_ipc_Python>>TD
    4. Set Active     → ON
    5. Paste THIS script into an Execute DAT — enable Start, Frame Start, On Exit
    6. Press Play (or reopen the project) to trigger onStart()

Python executable resolution (priority order) — mirrors example_receiver_launcher.py:
    1. CUDALINK_SENDER_PYTHON_EXE env var — full path, highest priority.
    2. Windows Python Launcher: 'py -3' resolves the system Python 3 installation
       and returns its full path (e.g. C:\\Users\\...\\Python311\\python.exe).
       Reliable on any Windows machine with the standard Python installer.
    3. 'python' — bare fallback, only used if it actually resolves on PATH
       (checked via shutil.which). If it does not resolve, onStart() prints a
       clear error and does not spawn — a bare "python" that resolves to TD's
       own bundled interpreter (or to nothing) fails with an opaque
       ImportError deep in the subprocess console instead of here.

    The resolved path is printed on each onStart() so you can verify which
    interpreter is used without opening a terminal.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess


def _find_python_exe() -> str | None:
    """Resolve the Python executable for the sender subprocess.

    Runs once at Execute DAT load time so the path is ready before onStart().
    Returns None only if no usable interpreter could be resolved at all.
    """
    # 1. Explicit env-var override — highest priority.
    if env := os.environ.get("CUDALINK_SENDER_PYTHON_EXE", ""):
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

    # 3. Bare fallback — only if 'python' actually resolves on PATH. Unlike the
    #    receiver launcher, we do NOT blindly return "python" here: inside a TD
    #    Execute DAT, sys.executable resolves to TD's own bundled Python (not
    #    the sender's env), so a PATH-less "python" would spawn something that
    #    fails with an opaque ImportError instead of a clear message here.
    return shutil.which("python")


_SENDER_PYTHON_EXE = _find_python_exe()

_process = None  # Sender subprocess handle


def onStart() -> None:
    """Launch the Python sender as a separate subprocess."""
    global _process

    if _SENDER_PYTHON_EXE is None:
        print("[CUDA-Link Launcher] ERROR: could not resolve a Python interpreter for the sender.")
        print("  Set CUDALINK_SENDER_PYTHON_EXE to a full python.exe path, or ensure 'py' or")
        print("  'python' is on PATH. Not spawning — see docstring for resolution order.")
        return

    script = os.path.join(project.folder, "td_exporter", "example_sender_python.py")

    if not os.path.isfile(script):
        print("[CUDA-Link Launcher] ERROR: sender script not found:")
        print(f"  {script}")
        return

    _process = subprocess.Popen(
        [_SENDER_PYTHON_EXE, script],
        # CREATE_NEW_CONSOLE: opens a visible console window for the sender.
        # CREATE_NEW_PROCESS_GROUP: required to send CTRL_BREAK_EVENT on shutdown
        # (CTRL_C_EVENT is blocked for new process groups on Windows).
        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    print(f"[CUDA-Link Launcher] Sender subprocess started  (PID {_process.pid})")
    print(f"  Script:     {script}")
    print(f"  Python exe: {_SENDER_PYTHON_EXE}")


def onCreate() -> None:
    return


def onExit() -> None:
    """Terminate the sender subprocess when the project closes."""
    global _process

    if _process is None:
        return

    if _process.poll() is None:
        pid = _process.pid
        try:
            # CTRL_BREAK_EVENT gives the Python sender a chance to run its IPC cleanup
            # (7-step GPU teardown). CTRL_C_EVENT cannot cross CREATE_NEW_PROCESS_GROUP
            # boundaries on Windows; CTRL_BREAK_EVENT can.
            _process.send_signal(signal.CTRL_BREAK_EVENT)
            _process.wait(timeout=3)
            print(f"[CUDA-Link Launcher] Sender subprocess exited gracefully (PID {pid}).")
        except subprocess.TimeoutExpired:
            _process.terminate()
            try:
                _process.wait(timeout=2)
                print(f"[CUDA-Link Launcher] Sender subprocess terminated (PID {pid}).")
            except subprocess.TimeoutExpired:
                _process.kill()
                print(f"[CUDA-Link Launcher] Sender subprocess force-killed (PID {pid}).")
        except OSError:
            _process.kill()
            print(f"[CUDA-Link Launcher] Sender subprocess force-killed (PID {pid}).")

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
