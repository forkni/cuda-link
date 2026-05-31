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

The ImportBuffer Script TOP handles receiver-mode frame import, driven from the Execute DAT
force-cook on every frame. onCook calls import_frame() directly and applies any pending
resolution or pixel-format updates.
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

    ext.import_frame(scriptOp)

    # Handle pixel-format update (mirrors the resolution block above).
    # _refresh_on_version_change sets needs_format_update = True when dtype/channels change.
    # consume_pending_format() clears the flag and returns the par.format string to apply.
    # Setting par.format from inside onCook mirrors how resolution is handled above
    # (scriptOp.par.outputresolution etc.) — it takes effect on the next cook, which is
    # the NEW_FRAME tick where copy_cuda_memory writes into the correctly-sized texture.
    fmt = ext.consume_pending_format()
    if fmt is not None:
        try:
            scriptOp.par.format = fmt
            ext._log(f"Set ImportBuffer pixel format to {fmt!r}", force=True)
        except (AttributeError, RuntimeError) as e:
            ext._log(f"Could not set ImportBuffer pixel format in onCook: {e}", force=True)


def onSetupParameters(scriptOp: object, page: object) -> None:
    """Called when Setup Parameters is pressed."""
    return
