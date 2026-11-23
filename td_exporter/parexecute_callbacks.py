# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""
Parameter Execute DAT Callback for CUDAIPCExtension

Copy this into a Parameter Execute DAT inside your .tox component.
Enable the parameters you want to monitor (Active, Ipcmemname, Numslots, Debug, Hidebuiltin, Mode).

The extension decides what each change does: CUDAIPCExtension.on_par_change and its PARS table.
"""


def _extension():
    """Return the CUDAIPCExtension, or None if it is not available.

    `parent().ext.CUDAIPCExtension` raises td.tdAttributeError when the extension did
    not compile; a callback must never raise, so catch broadly.
    """
    try:
        return getattr(parent().ext, "CUDAIPCExtension", None)  # noqa: F821
    except Exception:  # noqa: BLE001 -- td.tdAttributeError, plus parent() == None
        return None


def onValueChange(par: object, prev: object) -> None:
    """Called when any monitored parameter changes.

    Args:
        par: The parameter that changed
        prev: The previous value of the parameter
    """
    ext = _extension()
    if ext is None:
        return
    ext.on_par_change(par.name, par.eval(), prev)


# Other callback stubs (not used for parameter monitoring)
def onPulse(par: object) -> None:
    """Called when a pulse parameter is triggered."""
    pass


def onExpressionChange(par: object, val: object, prev: object) -> None:
    """Called when an expression parameter changes."""
    pass


def onExportChange(par: object, val: object, prev: object) -> None:
    """Called when an export parameter changes."""
    pass


def onEnableChange(par: object, val: object, prev: object) -> None:
    """Called when a parameter's enable state changes."""
    pass


def onModeChange(par: object, val: object, prev: object) -> None:
    """Called when a parameter's mode changes."""
    pass


def onNameChange(par: object, val: object, prev: object) -> None:
    """Called when a parameter's name changes."""
    pass
