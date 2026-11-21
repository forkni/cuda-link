# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""
Execute DAT — CUDA-Link TouchDesigner → Python Receiver Launcher

Paste this into an Execute DAT in your example project.
Enable "Start", "Frame Start", and "On Exit" toggles.

This DAT spawns example_receiver_python.py as a separate OS process on project
start and terminates it on exit. CUDA IPC requires separate processes — sender
and receiver cannot share GPU handles within the same process.

Pipeline:
    CUDAIPCLink_to_Python  (Sender mode, Active=ON, Ipcmemname=cudalink_ipc_TD>>Python)
         ↓  CUDA IPC
    onStart()  →  subprocess.Popen(example_receiver_python.py)
         ↓
    Prints frame stats in its own console window

TD Setup:
    1. Add a CUDAIPCLink_to_Python component to the network
    2. Set Mode       → Sender
    3. Set Ipcmemname → cudalink_ipc_TD>>Python
    4. Set Active     → ON
    5. Paste THIS script into an Execute DAT — enable Start, Frame Start, On Exit
    6. Press Play (or reopen the project) to trigger onStart()

Python executable resolution (priority order) — mirrors example_sender_launcher.py:
    1. CUDALINK_RECEIVER_PYTHON_EXE env var — full path, highest priority.
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

cuda_link library of the child:
    The launcher sets CUDALINK_PEER_LIB_PATH for the child to the folder of the cuda_link
    package that this TouchDesigner process loaded (or ``src`` next to the .toe). The example
    script loads cuda_link from that folder, so both sides use one SHM protocol version, also
    when the child's Python has another cuda-link installed.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys


def _find_python_exe() -> str | None:
    """Resolve the Python executable for the receiver subprocess.

    Runs once at Execute DAT load time so the path is ready before onStart().
    Returns None only if no usable interpreter could be resolved at all.
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

    # 3. Bare fallback — only if 'python' actually resolves on PATH. We do NOT
    #    blindly return "python" here: inside a TD Execute DAT, sys.executable
    #    resolves to TD's own bundled Python (not the receiver's env), so a
    #    PATH-less "python" would spawn something that fails with an opaque
    #    ImportError instead of a clear message here.
    return shutil.which("python")


def _peer_library_root() -> str | None:
    """Folder that holds the ``cuda_link`` package this TouchDesigner process uses.

    The child must speak the same SHM protocol as the component in this project. Another
    cuda-link in the child's site-packages (an older release) would not: a 1.x peer cannot
    read a 2.x segment. Order: the package this process loaded, then ``src`` next to the .toe.
    """
    loaded = getattr(sys.modules.get("cuda_link"), "__file__", None)
    if loaded:
        return os.path.dirname(os.path.dirname(os.path.abspath(loaded)))
    src = os.path.join(project.folder, "src")
    if os.path.isfile(os.path.join(src, "cuda_link", "__init__.py")):
        return src
    return None


def _peer_env() -> dict[str, str]:
    """Environment of the child: this process's, plus CUDALINK_PEER_LIB_PATH when known."""
    env = dict(os.environ)
    root = _peer_library_root()
    if root:
        env["CUDALINK_PEER_LIB_PATH"] = root
    return env


_RECEIVER_PYTHON_EXE = _find_python_exe()

_process = None  # Receiver subprocess handle


def onStart() -> None:
    """Launch the Python receiver as a separate subprocess."""
    global _process

    if _RECEIVER_PYTHON_EXE is None:
        print("[CUDA-Link Receiver Launcher] ERROR: could not resolve a Python interpreter for the receiver.")
        print("  Set CUDALINK_RECEIVER_PYTHON_EXE to a full python.exe path, or ensure 'py' or")
        print("  'python' is on PATH. Not spawning — see docstring for resolution order.")
        return

    script = os.path.join(project.folder, "td_exporter", "example_receiver_python.py")

    if not os.path.isfile(script):
        print("[CUDA-Link Receiver Launcher] ERROR: receiver script not found:")
        print(f"  {script}")
        return

    env = _peer_env()
    _process = subprocess.Popen(
        [_RECEIVER_PYTHON_EXE, script],
        env=env,
        # CREATE_NEW_CONSOLE: opens a visible console window for the receiver.
        # CREATE_NEW_PROCESS_GROUP: required to send CTRL_BREAK_EVENT on shutdown
        # (CTRL_C_EVENT is blocked for new process groups on Windows).
        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    print(f"[CUDA-Link Receiver Launcher] Receiver subprocess started  (PID {_process.pid})")
    print(f"  Script:     {script}")
    print(f"  Python exe: {_RECEIVER_PYTHON_EXE}")
    print(f"  cuda_link:  {env.get('CUDALINK_PEER_LIB_PATH', 'the one installed in that Python')}")


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
    """Check if the subprocess is still running; warn one time if it exited unexpectedly."""
    global _process

    if _process is not None and _process.poll() is not None:
        code = _process.returncode
        _process = None  # report the exit one time, not on every frame
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
