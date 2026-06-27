"""
Parameter Execute DAT callback for the CUDALinkSpoutBridge COMP.

Paste into a Parameter Execute DAT inside the .tox.
Enable monitoring for: Active, Restart, Direction, Spoutname, Ipcname, Device,
Pythonexe, Debug.

Dispatch pattern mirrors the existing CUDAIPCLink ``parexecute_callbacks.py``:
  ``onValueChange`` → dispatch by ``par.name`` → ``handle_*_change(ext, new, prev)``

Parameters are snapshot-at-spawn — editing a config param while the sidecar is
running sets Status to "Changed — press Restart" but does NOT respawn automatically.
"""


def onValueChange(par: object, prev: object) -> None:
    """Called by TD when any monitored parameter changes."""
    ext = parent().ext.SpoutBridgeExt  # type: ignore[name-defined]
    if ext is None:
        return

    param_name = par.name  # type: ignore[attr-defined]
    new_value = par.eval()  # type: ignore[attr-defined]

    if param_name == "Active":
        handle_active_change(ext, new_value, prev)
    elif param_name == "Restart":
        # Pulse params fire onPulse, but include here for completeness.
        handle_restart_change(ext, new_value, prev)
    elif param_name == "Debug":
        handle_debug_change(ext, new_value, prev)
    elif param_name in ("Direction", "Spoutname", "Ipcname", "Device", "Pythonexe"):
        handle_config_change(ext, param_name, new_value, prev)


def handle_active_change(ext: object, new_value: object, prev: object) -> None:
    """Handle the Active toggle: ON → spawn sidecar, OFF → kill sidecar."""
    if bool(new_value):
        ext.start()  # type: ignore[attr-defined]
    else:
        ext.stop()  # type: ignore[attr-defined]


def handle_restart_change(ext: object, new_value: object, prev: object) -> None:
    """Handle the Restart pulse: kill + respawn with current params."""
    ext.restart()  # type: ignore[attr-defined]


def handle_debug_change(ext: object, new_value: object, prev: object) -> None:
    """Handle the Debug toggle: enable/disable verbose lifecycle logging."""
    ext.set_debug(bool(new_value))  # type: ignore[attr-defined]


def handle_config_change(ext: object, name: str, new_value: object, prev: object) -> None:
    """Handle config param changes while the sidecar is running.

    Params are snapshot-at-spawn, so a change mid-run has no effect until Restart.
    Update Status to make this visible to the user.
    """
    try:
        status = str(parent().par.Status.eval())  # type: ignore[name-defined]
        if status.startswith("Running"):
            parent().par.Status = "Changed — press Restart"  # type: ignore[name-defined]
    except (AttributeError, RuntimeError):
        pass


# ---------------------------------------------------------------------------
# Pulse handler — TD calls onPulse for Pulse-type parameters
# ---------------------------------------------------------------------------


def onPulse(par: object) -> None:
    """Called when a pulse parameter is triggered."""
    ext = parent().ext.SpoutBridgeExt  # type: ignore[name-defined]
    if ext is None:
        return
    if par.name == "Restart":  # type: ignore[attr-defined]
        ext.restart()  # type: ignore[attr-defined]


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
