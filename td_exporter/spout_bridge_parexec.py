"""
Parameter Execute DAT callback for the CUDALinkSpoutBridge COMP.

Paste into a Parameter Execute DAT inside the .tox.
Enable monitoring for: Active, Direction, Spoutname, Ipcname, Pythonexe, Debug.

Dispatch pattern mirrors the existing CUDAIPCLink ``parexecute_callbacks.py``:
  ``onValueChange`` → dispatch by ``par.name`` → ``handle_*_change(ext, new, prev)``

Parameters are snapshot-at-spawn — editing a config param while the sidecar is
running sets Status to "Changed — restart Active" but does NOT respawn automatically.

Active toggle is the single lifecycle control:
  OFF → stop the sidecar subprocess immediately.
  ON  → (re)start the sidecar with the current param snapshot (same as toggling OFF
        then ON to apply config changes while running).
The former Restart pulse has been removed; fold restart into Active OFF→ON instead.
"""


def _resolve_ext() -> object:
    """Return the SpoutBridgeExt extension, or None with a clear hint if uninitialized.

    `parent().ext.SpoutBridgeExt` *raises* AttributeError when the extension hasn't been
    registered (the `if ext is None: return` guard would otherwise be dead code).  This
    wrapper converts that failure into a one-line Textport message + a Status-par update so
    the user knows exactly how to fix it — without a raw traceback.
    """
    try:
        return parent().ext.SpoutBridgeExt  # type: ignore[name-defined]
    except AttributeError:
        import contextlib

        print("[Spout Bridge] ERROR: extension not initialized — right-click the COMP and choose 'Re-Init Extensions'.")
        with contextlib.suppress(Exception):
            parent().par.Status = "Error: ext not initialized — Re-Init"  # type: ignore[name-defined]
        return None


def onValueChange(par: object, prev: object) -> None:
    """Called by TD when any monitored parameter changes."""
    ext = _resolve_ext()
    if ext is None:
        return

    param_name = par.name  # type: ignore[attr-defined]
    new_value = par.eval()  # type: ignore[attr-defined]

    if param_name == "Active":
        handle_active_change(ext, new_value, prev)
    elif param_name == "Debug":
        handle_debug_change(ext, new_value, prev)
    elif param_name in ("Direction", "Spoutname", "Ipcname", "Pythonexe"):
        handle_config_change(ext, param_name, new_value, prev)


def handle_active_change(ext: object, new_value: object, prev: object) -> None:
    """Handle the Active toggle: ON → (re)spawn sidecar, OFF → kill sidecar.

    Uses ``restart()`` on the ON path (stop-then-start) so toggling Active OFF→ON
    always applies the latest param snapshot — no separate Restart pulse needed.
    ``stop()`` is a fast no-op when the sidecar is not running.
    """
    if bool(new_value):
        ext.restart()  # type: ignore[attr-defined]
    else:
        ext.stop()  # type: ignore[attr-defined]


def handle_debug_change(ext: object, new_value: object, prev: object) -> None:
    """Handle the Debug toggle: enable/disable verbose lifecycle logging."""
    ext.set_debug(bool(new_value))  # type: ignore[attr-defined]


def handle_config_change(ext: object, name: str, new_value: object, prev: object) -> None:
    """Handle config param changes while the sidecar is running.

    Params are snapshot-at-spawn, so a change mid-run has no effect until the sidecar
    is restarted (toggle Active OFF then ON).  Update Status to make this visible.
    """
    try:
        status = str(parent().par.Status.eval())  # type: ignore[name-defined]
        if status.startswith("Running"):
            parent().par.Status = "Changed — restart Active"  # type: ignore[name-defined]
    except (AttributeError, RuntimeError):
        pass


# ---------------------------------------------------------------------------
# Pulse handler — TD calls onPulse for Pulse-type parameters.
# The Restart pulse has been removed from the COMP; Active OFF→ON replaces it.
# ---------------------------------------------------------------------------


def onPulse(par: object) -> None:
    """Called when a pulse parameter is triggered."""
    pass


# ---------------------------------------------------------------------------
# Stubs required by the Parameter Execute DAT interface
# ---------------------------------------------------------------------------


def onExpressionChange(par: object, val: object, prev: object) -> None:
    pass


def onExportChange(par: object, val: object, prev: object) -> None:
    pass


def onEnableChange(par: object, val: object, prev: object) -> None:
    pass


def onModeChange(par: object, val: object, prev: object) -> None:
    pass


def onNameChange(par: object, val: object, prev: object) -> None:
    pass
