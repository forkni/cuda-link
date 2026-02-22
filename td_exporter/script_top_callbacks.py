"""
ImportBuffer Callback for CUDAIPCExtension Receiver Mode

Copy this into the ImportBuffer's Callbacks DAT inside the .tox component.
This is ONLY used when Mode = 'Receiver'.

The ImportBuffer's callbacks parameter should point to this DAT:
    ImportBuffer.par.callbacks = op('script_top_callbacks')

This Script TOP is force-cooked from Execute DAT onFrameStart (nothing pulls it downstream).
The onCook callback handles resolution update (one-time) and frame import.
"""


def onCook(scriptTop: object) -> None:
    """Called every time the Script TOP needs to cook.

    TD 2023 path: Handles resolution update (one-time) and imports frame from CUDA IPC.
    TD 2025+ with modoutsidecook: This callback may still fire but import_frame()
    is driven from Execute DAT. The resolution update here serves as a safety net.

    Args:
        scriptTop: The Script TOP operator instance (same as 'me')
    """
    ext = parent().ext.CUDAIPCExtension
    if ext is None:
        return

    # Handle resolution update (one-time, after initialize_receiver)
    # With modoutsidecook, this may already be handled by Execute DAT
    if ext._rx_needs_resolution_update:
        try:
            scriptTop.par.outputresolution = 9  # Custom Resolution
            scriptTop.par.resolutionw = ext._rx_width
            scriptTop.par.resolutionh = ext._rx_height
            ext._rx_needs_resolution_update = False
            ext._log(
                f"Set ImportBuffer resolution to {ext._rx_width}x{ext._rx_height}",
                force=True,
            )
        except (AttributeError, RuntimeError) as e:
            ext._log(f"Could not set ImportBuffer resolution: {e}", force=True)

    # TD 2023 path: Import frame from CUDA IPC into this Script TOP
    # With modoutsidecook (TD 2025+), import_frame() is called from Execute DAT instead
    # Check if modoutsidecook is active; if so, skip to avoid double-import
    try:
        if hasattr(scriptTop.par, "modoutsidecook") and scriptTop.par.modoutsidecook.eval():
            return  # Import handled by Execute DAT
    except (AttributeError, RuntimeError):
        pass  # Parameter doesn't exist or can't be read, proceed with import

    ext.import_frame(scriptTop)


def onSetupParameters(scriptTop: object, page: object) -> None:
    """Called when Setup Parameters is pressed.

    Args:
        scriptTop: The Script TOP
        page: The custom parameter page
    """
    # No custom parameters needed for receiver
    pass
