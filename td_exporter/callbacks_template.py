"""
Template Execute DAT Callback for CUDAIPCExtension

Copy this into an Execute DAT inside your .tox component.
Enable "Frame Start" and "On Exit" toggles.

Handles both Sender and Receiver modes.
"""


def onFrameStart(frame: int) -> None:
    """Called every frame.

    Args:
        frame: Current frame number
    """
    ext = parent().ext.CUDAIPCExtension

    if ext is None:
        return

    # Mode parameter changes are already handled by parexecute_callbacks.py
    # No need to re-check here (redundant eval removed for performance)

    # Sender: export frame from ExportBuffer TOP
    if ext.mode == "Sender":
        export_buffer = op("ExportBuffer")
        if export_buffer:
            ext.export_frame(export_buffer)
    # Receiver: Force ImportBuffer TOP to cook (triggers onCook -> import_frame)
    elif ext.mode == "Receiver":
        import_buffer = op("ImportBuffer")
        if import_buffer:
            # Update ImportBuffer resolution if receiver initialization completed
            if ext._rx_needs_resolution_update:
                try:
                    import_buffer.par.outputresolution = 9  # Custom Resolution
                    import_buffer.par.resolutionw = ext._rx_width
                    import_buffer.par.resolutionh = ext._rx_height
                    ext._rx_needs_resolution_update = False
                    ext._log(f"Set ImportBuffer resolution to {ext._rx_width}x{ext._rx_height}", force=True)
                except (AttributeError, RuntimeError) as e:
                    ext._log(f"Could not set ImportBuffer resolution: {e}", force=True)

            # Force cook to trigger onCook callback
            import_buffer.cook(force=True)


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
