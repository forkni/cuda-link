"""
Template Execute DAT Callback for CUDAIPCExtension

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
            # Import frame first: initialize_receiver() sets resolution flag
            ext.import_frame(import_buffer)
            # Resolution update after: catches flag set during initialization
            ext.update_receiver_resolution(import_buffer)
        else:
            # TD 2023 fallback: force-cook triggers Script TOP onCook
            # Resolution update happens inside onCook (1-frame delay for changes)
            import_buffer.cook(force=True)


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
        export_buffer = op("ExportBuffer")
        if export_buffer:
            ext.export_frame(export_buffer)


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
