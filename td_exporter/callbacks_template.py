# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""
Execute DAT Callback for CUDAIPCExtension

Copy this into an Execute DAT inside your .tox component.
Enable "Frame Start", "Frame End", and "On Exit" toggles.

Architecture:
- Sender: onFrameStart=housekeeping, onFrameEnd=export (avoids 8.8ms GPU wait)
- Receiver: onFrameStart=force-cook ImportBuffer (triggers Script TOP onCook)
"""

import weakref

# Extension instances whose parameters were checked against their own state after init.
_RESYNCED: "weakref.WeakSet[object]" = weakref.WeakSet()

# Parameters that CUDAIPCExtension reads in __init__ and afterwards follows only through
# parexec1's onValueChange: (parameter, extension attribute, type to compare as). Mode
# comes first, so the later handlers see the engine of the new mode.
_RESYNC_PARS = (
    ("Mode", "mode", str),
    ("Ipcmemname", "shm_name", str),
    ("Numslots", "num_slots", int),
    ("Debug", "verbose_performance", bool),
)


def _extension():
    """Return the CUDAIPCExtension, or None if it is not available.

    `parent().ext.CUDAIPCExtension` raises td.tdAttributeError when the extension did
    not compile; a frame callback must never raise, so catch broadly.
    """
    try:
        return getattr(parent().ext, "CUDAIPCExtension", None)  # noqa: F821
    except Exception:  # noqa: BLE001 -- td.tdAttributeError, plus parent() == None
        return None


def _resync_params_once(ext: object) -> None:
    """Apply parameter writes that the extension missed, once per extension instance.

    The extension reads Mode, Ipcmemname, Numslots and Debug in __init__. After that, only
    parexec1 tells it about a change, and parexec1 reports nothing that happened before it
    first cooked. So a script that sets these parameters in the same call as loadTox() or
    create() leaves the extension on the old values, with no error (2026-11-21: a Receiver
    set up this way stayed a Sender on the default Ipcmemname). On the first frame, each
    parameter that differs from the extension's own value goes through parexec1's
    onValueChange, the same path as a normal change. Never raises.
    """
    try:
        if ext in _RESYNCED:
            return
        _RESYNCED.add(ext)
        comp = parent()  # noqa: F821
        handlers = comp.op("parexec1")
        on_change = getattr(getattr(handlers, "module", None), "onValueChange", None)
        if on_change is None:
            return  # a COMP without parexec1 has nothing to resync through
        profile = bool(getattr(getattr(ext, "_config", None), "export_profile", False))
        for par_name, attr, kind in _RESYNC_PARS:
            if par_name == "Debug" and profile:
                continue  # CUDALINK_EXPORT_PROFILE forces verbose on; Debug Off must not undo it
            par = getattr(comp.par, par_name, None)
            if par is None:
                continue
            seen = getattr(ext, attr)
            if kind(par.eval()) != kind(seen):
                on_change(par, seen)
    except Exception:  # noqa: BLE001 -- a frame callback must never raise
        return


def _cook_slot_meta() -> None:
    """Force-cook the slot_meta Script CHOP after an export (never raises).

    In Sender mode nothing else makes it cook: it reads plain extension state, which
    TD does not track as a dependency. A .tox without the CHOP (1.x layout) skips this.
    """
    try:
        meta_chop = parent().op("slot_meta")  # noqa: F821
        if meta_chop is not None:
            meta_chop.cook(force=True)
    except Exception:  # noqa: BLE001 -- a frame callback must never raise
        return


def onFrameStart(frame: int) -> None:
    """Called at the start of every frame.

    Sender: Lightweight housekeeping (deferred GPU cleanup).
    Receiver: Force-cook ImportBuffer (triggers Script TOP onCook).

    Args:
        frame: Current frame number
    """
    ext = _extension()
    if ext is None:
        return

    _resync_params_once(ext)

    if ext.mode == "Sender":
        # Check if deferred GPU cleanup is scheduled (lightweight, ~0ms normally)
        ext._check_deferred_cleanup()

    elif ext.mode == "Receiver":
        import_buffer = op("ImportBuffer")
        if import_buffer is None:
            return

        # P11: skip the cook entirely when the sender hasn't produced a new frame.
        # has_new_frame() reads write_idx from SHM and returns True for: init/retry,
        # version change, shutdown, or a new write_idx. Returns False only when the
        # sender's write_idx matches our cursor (no new data this frame).
        if not ext.has_new_frame():
            # Framewaitms (default 3 ms): wait for the sender's doorbell before this
            # frame is a repeat. An extension without the method (1.x) keeps poll-once.
            wait = getattr(ext, "wait_for_new_frame", None)
            if wait is None or not wait():
                return

        # Force-cook ImportBuffer: triggers Script TOP onCook, which calls import_frame()
        # and consume_pending_format(). update_receiver_format() afterwards is belt-and-suspenders:
        # a no-op when the pending format flag is already cleared in onCook, but catches any
        # residual flag if onCook couldn't apply the par write.
        import_buffer.cook(force=True)
        ext.update_receiver_format(import_buffer)


def onFrameEnd(frame: int) -> None:
    """Called at the end of every frame.

    Sender: Export frame AFTER cook phase (texture already rendered on GPU).
            cudaMemory() returns instantly instead of blocking 8.8ms waiting for GPU.
    Receiver: Nothing (import already happened via Script TOP onCook).

    Args:
        frame: Current frame number
    """
    ext = _extension()
    if ext is None:
        return

    # The CHOP values change only when a frame was published.
    if ext.mode == "Sender" and ext.export_frame():
        _cook_slot_meta()


def onExit() -> None:
    """Called when TouchDesigner exits or when this DAT is destroyed."""
    ext = _extension()
    if ext is not None:
        ext.cleanup()


# Other callback stubs (not used for CUDA IPC, but required by TD)
def onStart() -> None:
    """TD required callback - not used."""
    return


def onCreate() -> None:
    """TD required callback - not used."""
    return
