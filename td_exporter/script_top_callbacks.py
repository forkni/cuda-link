"""
Shared Script TOP callbacks for CUDAIPCLink.

ONE Callbacks DAT serves BOTH Script TOPs inside the .tox; onCook dispatches by
operator name:
  ImportBuffer    (Receiver mode) — import latest CUDA-IPC frame + pending resolution
  warning_emitter (status badge)  — re-emit RealTDHost's 'cuda_link_status_msg' as addWarning

Wire both Script TOPs' Callbacks DAT parameter to this DAT:
    ImportBuffer.par.callbacks    = op('script_top_callbacks')
    warning_emitter.par.callbacks = op('script_top_callbacks')

The warning_emitter Script TOP must have Cook Type set to 'Off' (Pulse to Cook).
RealTDHost force-cooks it on every status transition so the badge stays in sync
without relying on continuous cooking.

The ImportBuffer Script TOP handles receiver-mode frame import.  In TD 2025+ with
modoutsidecook enabled, import_frame() is driven from the Execute DAT instead; this
onCook still handles the one-time resolution update as a safety net.
"""

_STATUS_EMITTER_NAME = "warning_emitter"


def onCook(scriptOp: object) -> None:
    """Called every time a Script TOP that references this DAT needs to cook."""
    # Status badge host: warning_emitter (force-cooked by RealTDHost on transitions)
    if scriptOp.name == _STATUS_EMITTER_NAME:
        msg = scriptOp.parent().fetch("cuda_link_status_msg", None)
        if msg:
            scriptOp.addWarning(str(msg))
        return

    # Receiver-mode frame import: ImportBuffer
    ext = parent().ext.CUDAIPCExtension
    if ext is None:
        return

    # Handle resolution update (one-time, after initialize_receiver)
    # With modoutsidecook, this may already be handled by Execute DAT
    pending = ext.consume_pending_resolution()
    if pending is not None:
        width, height = pending
        try:
            scriptOp.par.outputresolution = 9  # Custom Resolution
            scriptOp.par.resolutionw = width
            scriptOp.par.resolutionh = height
            ext._log(
                f"Set ImportBuffer resolution to {width}x{height}",
                force=True,
            )
        except (AttributeError, RuntimeError) as e:
            ext._log(f"Could not set ImportBuffer resolution: {e}", force=True)

    # TD 2023 path: Import frame from CUDA IPC into this Script TOP
    # With modoutsidecook (TD 2025+), import_frame() is called from Execute DAT instead
    # Check if modoutsidecook is active; if so, skip to avoid double-import
    try:
        if hasattr(scriptOp.par, "modoutsidecook") and scriptOp.par.modoutsidecook.eval():
            return  # Import handled by Execute DAT
    except (AttributeError, RuntimeError):
        pass  # Parameter doesn't exist or can't be read, proceed with import

    ext.import_frame(scriptOp)


def onSetupParameters(scriptOp: object, page: object) -> None:
    """Called when Setup Parameters is pressed."""
    return
