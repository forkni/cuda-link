"""
Execute DAT Callback for CUDAIPCExtension

Copy this into an Execute DAT inside your .tox component.
Enable "Frame Start", "Frame End", and "On Exit" toggles.

Architecture:
- Sender: onFrameStart=housekeeping, onFrameEnd=export (avoids 8.8ms GPU wait)
- Receiver: onFrameStart=force-cook ImportBuffer (triggers Script TOP onCook)
"""


def onFrameStart(frame: int) -> None:
    """Called at the start of every frame.

    Sender: Lightweight housekeeping (deferred GPU cleanup).
    Receiver: Force-cook ImportBuffer (triggers Script TOP onCook).

    Args:
        frame: Current frame number
    """
    ext = parent().ext.CUDAIPCExtension
    if ext is None:
        return

    if ext.mode == "Sender":
        # Check if deferred GPU cleanup is scheduled (lightweight, ~0ms normally)
        ext._check_deferred_cleanup()

    elif ext.mode == "Receiver":
        import_buffer = op("ImportBuffer")
        if import_buffer is None:
            return

        # TD 2025+: modoutsidecook enables copyCUDAMemory from Execute DAT
        # This eliminates force-cook overhead and fixes resolution delay
        if hasattr(import_buffer.par, "modoutsidecook") and import_buffer.par.modoutsidecook.eval():
            # Import frame first: initialize_receiver() sets resolution + format flags
            ext.import_frame(import_buffer)
            # Resolution and pixel-format updates after: catch flags set during
            # initialization or mid-stream dtype/geometry changes.
            ext.update_receiver_resolution(import_buffer)
            ext.update_receiver_format(import_buffer)
        else:
            # TD 2023 fallback: force-cook triggers Script TOP onCook.
            # Resolution and pixel-format updates happen inside onCook via
            # consume_pending_resolution / consume_pending_format (1-frame delay).
            # update_receiver_format after cook() is belt-and-suspenders: it's a
            # no-op when needs_format_update is already False (cleared in onCook),
            # but catches any residual flag if onCook couldn't apply the par write.
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
    ext = parent().ext.CUDAIPCExtension
    if ext is None:
        return

    if ext.mode == "Sender":
        ext.export_frame()


def onExit() -> None:
    """Called when TouchDesigner exits or when this DAT is destroyed."""
    ext = parent().ext.CUDAIPCExtension
    if ext is not None:
        ext.cleanup()


# Other callback stubs (not used for CUDA IPC, but required by TD)
def onStart() -> None:
    """TD required callback - not used."""
    return


def onCreate() -> None:
    """TD required callback - not used."""
    return
