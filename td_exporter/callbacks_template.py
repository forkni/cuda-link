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

# Names of the Parameter Execute DAT: parexec1 in the shipped .tox, parexecute in
# docs/TOX_BUILD_GUIDE.md.
_PAREXEC_NAMES = ("parexec1", "parexecute")


def _numslots_enabled(ext: object, comp: object) -> bool:
    return bool(comp.par.Numslots.enable)


def _numslots_should_enable(comp: object) -> bool:
    return not comp.par.Active.eval() and str(comp.par.Mode.eval()) == "Sender"


# Parameters whose state follows a change only through parexec1's onValueChange:
# (parameter, read of the state the handler keeps, type to compare as, wanted value or
# None for the parameter's own value). Mode comes first, so the later handlers see the
# engine of the new mode. Active is live (is_active() reads the parameter); only its
# Numslots.enable side effect can be stale.
_RESYNC_PARS = (
    ("Mode", lambda ext, comp: ext.mode, str, None),
    ("Active", _numslots_enabled, bool, _numslots_should_enable),
    ("Ipcmemname", lambda ext, comp: ext.shm_name, str, None),
    ("Numslots", lambda ext, comp: ext.num_slots, int, None),
    ("Debug", lambda ext, comp: ext.verbose_performance, bool, None),
    ("Hidebuiltin", lambda ext, comp: comp.showCustomOnly, bool, None),
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

    The extension reads Mode, Ipcmemname, Numslots, Debug and Hidebuiltin in __init__.
    After that, only parexec1 tells it about a change, and parexec1 reports nothing that
    happened before it first cooked. So a script that sets these parameters in the same
    call as loadTox() or create() leaves the extension on the old values, with no error
    (2026-11-21: a Receiver set up this way stayed a Sender on the default Ipcmemname).
    On the first frame (start or end, whichever runs first), each parameter that differs
    from the state its handler keeps goes through parexec1's onValueChange, the same path
    as a normal change. One parameter that fails does not stop the others. Never raises.
    """
    try:
        if ext in _RESYNCED:
            return
        _RESYNCED.add(ext)
        comp = parent()  # noqa: F821
        on_change = None
        for name in _PAREXEC_NAMES:
            on_change = getattr(getattr(comp.op(name), "module", None), "onValueChange", None)
            if on_change is not None:
                break
        if on_change is None:
            return  # a COMP without a Parameter Execute DAT has nothing to resync through
        profile = bool(getattr(getattr(ext, "_config", None), "export_profile", False))
    except Exception:  # noqa: BLE001 -- a frame callback must never raise
        return
    for par_name, read_state, kind, wanted in _RESYNC_PARS:
        if par_name == "Debug" and profile:
            continue  # CUDALINK_EXPORT_PROFILE forces verbose on; Debug Off must not undo it
        try:
            par = getattr(comp.par, par_name, None)
            if par is None:
                continue
            seen = read_state(ext, comp)
            target = par.eval() if wanted is None else wanted(comp)
            if kind(target) == kind(seen):
                continue
            # A value parameter reports the extension's value as `prev`. Active's handler
            # does not read `prev`; it gets the opposite of the current state.
            on_change(par, seen if wanted is None else not par.eval())
        except Exception:  # noqa: BLE001,S112 -- isolate each parameter; never raise
            continue


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

    # A loadTox() in mid-frame reaches Frame End before Frame Start: resync before export.
    _resync_params_once(ext)

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
