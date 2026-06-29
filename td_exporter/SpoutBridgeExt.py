"""
SpoutBridgeExt — extension class for the CUDALinkSpoutBridge COMP.

Launched as a droppable helper COMP that supervises the existing
``python -m cuda_link_spout.bridge`` sidecar process.  All bridge inputs are
exposed as visible TD custom parameters (no env-var reads).

textDAT name: SpoutBridgeExt  (must match the importable module name inside the
COMP namespace so ``parent().ext.SpoutBridgeExt`` resolves correctly).
"""

from __future__ import annotations

import shutil
import signal
import subprocess
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # TDHost is a Protocol defined in td_exporter/TDHost.py and also available inside
    # the COMP namespace as a text DAT.  Import under TYPE_CHECKING only so this module
    # loads outside TD (unit tests) without touching the seam.
    from TDHost import TDHost  # noqa: F401


_LOG_PREFIX = "[Spout Bridge]"


class SpoutBridgeExt:
    """Manages the ``cuda_link_spout.bridge`` sidecar subprocess for one COMP instance.

    Per-instance state — multiple dropped COMPs never share a process handle.
    Built against the :class:`~TDHost.TDHost` seam so it is fully testable under
    :class:`~_td_fakes.FakeTDHost` without a live TouchDesigner environment.
    """

    def __init__(self, host: Any) -> None:
        self._host = host
        self._process: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._debug: bool = False

    # ------------------------------------------------------------------
    # Interpreter resolution (no env-var; Pythonexe param is the knob)
    # ------------------------------------------------------------------

    def _resolve_python(self) -> str:
        """Resolve the Python interpreter to use for the sidecar.

        Priority (no env-var branch — every knob is a visible TD param):

        1. ``Pythonexe`` param — if non-empty, used verbatim.
        2. Windows Python Launcher ``py -3`` — resolves the registered system Python 3
           and returns its full absolute path.
        3. Bare ``python`` fallback (may be TD's embedded interpreter if 'python' is on
           PATH — ``Pythonexe`` must be set explicitly to avoid that).
        """
        explicit = str(self._host.param_value("Pythonexe") or "").strip()
        if explicit:
            return explicit

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

        return "python"

    # ------------------------------------------------------------------
    # argv construction (no width/height/fmt — derived from IPC frame)
    # ------------------------------------------------------------------

    def _build_argv(self) -> list[str]:
        """Build the sidecar argv from a snapshot of the current params.

        Never reads ``os.environ`` — every knob is a visible TD custom parameter.
        ``--spout`` is always emitted, even when blank (``--dir in`` can bind the
        active sender by passing an empty name to ``bridge.py``).
        """
        exe = self._resolve_python()
        direction = str(self._host.param_value("Direction") or "out").lower()
        spout_name = str(self._host.param_value("Spoutname") or "")
        ipc_name = str(self._host.param_value("Ipcname") or "")

        # No --device flag: single-GPU workflow; bridge.py defaults to device 0.
        # The CLI --device flag and library LUID affinity remain intact for direct CLI use.
        argv = [
            exe,
            "-m",
            "cuda_link_spout.bridge",
            "--dir",
            direction,
            "--ipc",
            ipc_name,
            "--spout",
            spout_name,
        ]
        # Propagate the Debug param to the sidecar so it enables verbose logging in the cmd
        # window.  Read directly from the host (not self._debug) so the flag is always
        # correct even on project load / Re-Init before any set_debug() change-event fires.
        if bool(self._host.param_value("Debug")):
            argv.append("--verbose")
        return argv

    # ------------------------------------------------------------------
    # Lifecycle: start / stop / restart
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the sidecar subprocess with a snapshot of the current params.

        If a process is already running, this is a no-op (call :meth:`restart`
        to respawn with updated params).
        """
        if self._process is not None and self._process.poll() is None:
            print(f"{_LOG_PREFIX} Already running (PID {self._process.pid}) — call Restart to respawn.")
            return

        # Sync _debug from the Debug param before building argv and printing the banner.
        # Change-events set _debug on the fly, but on project load / Re-Init the extension
        # is reconstructed with _debug=False and may never receive a change-event if the
        # param value hasn't changed.  Reading it here ensures the correct state at spawn.
        self._debug = bool(self._host.param_value("Debug"))

        argv = self._build_argv()
        direction = str(self._host.param_value("Direction") or "out")
        spout_name = str(self._host.param_value("Spoutname") or "")
        ipc_name = str(self._host.param_value("Ipcname") or "")

        print(f"{_LOG_PREFIX} Starting bridge...")
        print(f"  Direction : {direction}")
        print(f"  Spout name: {spout_name!r}")
        print(f"  IPC name  : {ipc_name!r}")
        print(f"  Python    : {argv[0]}")
        if self._debug:
            print(f"  argv      : {argv}")

        try:
            popen_kwargs: dict = {}
            if sys.platform == "win32":
                # CREATE_NEW_CONSOLE opens a visible console window (the bridge log).
                # CREATE_NEW_PROCESS_GROUP is required to send CTRL_BREAK_EVENT on Windows
                # (CTRL_C_EVENT cannot cross new-process-group boundaries).
                popen_kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
                )
            self._process = subprocess.Popen(argv, **popen_kwargs)
        except FileNotFoundError:
            msg = f"Interpreter not found: {argv[0]!r}"
            print(f"{_LOG_PREFIX} ERROR: {msg}")
            self._host.set_param_value("Status", f"Error: {msg}")
            return

        self._host.set_param_value("Status", f"Running (PID {self._process.pid})")
        print(f"{_LOG_PREFIX} Sidecar started (PID {self._process.pid}).")

    def stop(self) -> None:
        """Terminate the sidecar using CTRL_BREAK → terminate → kill escalation.

        Mirrors the shutdown sequence used by ``example_receiver_launcher.py``
        (``onExit``): CTRL_BREAK_EVENT gives the Python bridge a chance to flush
        its shutdown sequence (close the Importer / Exporter), then escalates.
        """
        if self._process is None:
            return

        if self._process.poll() is None:
            pid = self._process.pid
            try:
                # CTRL_BREAK_EVENT gives the bridge a chance to flush its shutdown
                # sequence on Windows; on POSIX use SIGTERM instead.
                if sys.platform == "win32":
                    self._process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self._process.terminate()
                self._process.wait(timeout=3)
                if self._debug:
                    print(f"{_LOG_PREFIX} Sidecar exited gracefully (PID {pid}).")
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                    if self._debug:
                        print(f"{_LOG_PREFIX} Sidecar terminated (PID {pid}).")
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    if self._debug:
                        print(f"{_LOG_PREFIX} Sidecar force-killed (PID {pid}).")
            except OSError:
                self._process.kill()
                if self._debug:
                    print(f"{_LOG_PREFIX} Sidecar force-killed (PID {pid}).")

        self._process = None
        self._host.set_param_value("Status", "Stopped")

    def restart(self) -> None:
        """Kill the running sidecar (if any) and respawn with the current param snapshot."""
        self.stop()
        self.start()

    # ------------------------------------------------------------------
    # Crash detection (called from Execute DAT onFrameStart)
    # ------------------------------------------------------------------

    def poll_status(self) -> None:
        """Check if the sidecar is still alive; update ``Status`` if it has exited.

        Called once per cook from ``spout_bridge_exec.py:onFrameStart``.
        Cheap: only ``subprocess.Popen.poll()`` (non-blocking) + a param write on
        transition.
        """
        if self._process is None:
            return
        code = self._process.poll()
        if code is None:
            return  # still running — leave Status alone

        # Process has exited.
        status = f"Exited (code {code})"
        self._host.set_param_value("Status", status)
        if code != 0:
            print(f"{_LOG_PREFIX} WARNING: sidecar exited unexpectedly (code={code}).")
        elif self._debug:
            print(f"{_LOG_PREFIX} Sidecar exited (code 0).")
        self._process = None

    # ------------------------------------------------------------------
    # Debug verbosity
    # ------------------------------------------------------------------

    def set_debug(self, enabled: bool) -> None:
        """Toggle verbose lifecycle logging (maps to the ``Debug`` param)."""
        self._debug = bool(enabled)
        if self._debug:
            print(f"{_LOG_PREFIX} Debug logging ENABLED.")
        else:
            print(f"{_LOG_PREFIX} Debug logging DISABLED.")
