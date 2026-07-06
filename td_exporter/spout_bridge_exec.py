"""
Execute DAT callback for the CUDALinkSpoutBridge COMP.

Paste into an Execute DAT inside the .tox.
Enable toggles: Start, Exit, Frame Start.

Handles project-level lifecycle:
  - ``onStart``     — re-arm: if Active=ON was saved, re-spawn the sidecar.
  - ``onExit``      — kill the sidecar gracefully when the TD project closes.
  - ``onFrameStart``— poll for unexpected crashes each cook; flip Status when detected.
"""


def onStart() -> None:
    """Re-arm the bridge on project (re)start.

    If ``Active`` was ON when the project was saved, respawn the sidecar.
    Params are snapshot-at-spawn, so this uses the saved param values.
    """
    try:
        ext = parent().ext.SpoutBridgeExt  # type: ignore[name-defined]
        if ext is None:
            return
        if bool(parent().par.Active.eval()):  # type: ignore[name-defined]
            ext.start()  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        pass


def onExit() -> None:
    """Kill the sidecar when the TD project closes or TD exits."""
    try:
        ext = parent().ext.SpoutBridgeExt  # type: ignore[name-defined]
        if ext is not None:
            ext.stop()  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        pass


def onFrameStart(frame: int) -> None:
    """Poll for sidecar crashes each cook; update Status if the process has exited."""
    try:
        ext = parent().ext.SpoutBridgeExt  # type: ignore[name-defined]
        if ext is not None:
            ext.poll_status()  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        pass


# ---------------------------------------------------------------------------
# Stubs for unused Execute DAT callbacks
# ---------------------------------------------------------------------------


def onCreate() -> None:
    pass


def onFrameEnd(frame: int) -> None:
    pass


def onPlayStateChange(state: bool) -> None:
    pass


def onDeviceChange() -> None:
    pass


def onProjectPreSave() -> None:
    pass


def onProjectPostSave() -> None:
    pass
