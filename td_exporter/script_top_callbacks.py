"""
ImportBuffer Callback for CUDAIPCExtension Receiver Mode

Copy this into the ImportBuffer's Callbacks DAT inside the .tox component.
This is ONLY used when Mode = 'Receiver'.

The ImportBuffer's callbacks parameter should point to this DAT:
    ImportBuffer.par.callbacks = op('script_top_callbacks')
"""


def onCook(scriptTop: object) -> None:
    """Called every time the Script TOP needs to cook.

    Args:
        scriptTop: The Script TOP operator instance (same as 'me')
    """
    ext = parent().ext.CUDAIPCExtension

    if ext and ext.mode == "Receiver":
        ext.import_frame(scriptTop)


def onSetupParameters(scriptTop: object, page: object) -> None:
    """Called when Setup Parameters is pressed.

    Args:
        scriptTop: The Script TOP
        page: The custom parameter page
    """
    # No custom parameters needed for receiver
    pass
