"""
Parameter Execute DAT Callback for CUDAIPCExtension

Copy this into a Parameter Execute DAT inside your .tox component.
Enable the parameters you want to monitor (Active, Ipcmemname, Numslots, Debug, Hidebuiltin, Mode).

Handles parameter changes with debug logging and triggers appropriate re-initialization.
"""

import contextlib


def onValueChange(par: object, prev: object) -> None:
    """Called when any monitored parameter changes.

    Args:
        par: The parameter that changed
        prev: The previous value of the parameter
    """
    ext = parent().ext.CUDAIPCExtension

    if ext is None:
        return

    param_name = par.name
    new_value = par.eval()

    # Log the parameter change if debug is enabled
    ext._log(f"Parameter '{param_name}' changed: {prev} -> {new_value}", force=False)

    # Handle each parameter
    if param_name == "Active":
        handle_active_change(ext, new_value, prev)

    elif param_name == "Ipcmemname":
        handle_ipcmemname_change(ext, new_value, prev)

    elif param_name == "Numslots":
        handle_numslots_change(ext, new_value, prev)

    elif param_name == "Debug":
        handle_debug_change(ext, new_value, prev)

    elif param_name == "Hidebuiltin":
        handle_hidebuiltin_change(ext, new_value, prev)

    elif param_name == "Mode":
        handle_mode_change(ext, new_value, prev)


def handle_active_change(ext: object, new_value: object, prev: object) -> None:
    """Handle Active parameter toggle.

    Args:
        ext: CUDAIPCExtension instance
        new_value: New Active state (bool or int)
        prev: Previous Active state
    """
    # Convert to bool
    new_value = bool(new_value)

    if new_value:
        ext._log("Component activated", force=True)
        # Re-initialize based on current mode
        if ext.mode == "Sender":
            # Sender initialization happens on first export_frame() call
            ext._log("Sender mode ready - will initialize on first frame export", force=False)
        elif ext.mode == "Receiver":
            # Trigger receiver initialization attempt
            ext._log("Receiver mode activated - will attempt connection", force=False)
    else:
        ext._log("Component deactivated - cleaning up", force=True)
        # Clean up current mode resources
        ext.cleanup()

    # Disable Numslots while active to prevent runtime array size mismatch.
    # Receiver mode always keeps Numslots disabled (sender controls slot count).
    # Sender mode: editable only when inactive.
    with contextlib.suppress(AttributeError):
        parent().par.Numslots.enable = not new_value and ext.mode == "Sender"


def handle_ipcmemname_change(ext: object, new_value: object, prev: object) -> None:
    """Handle Ipcmemname parameter change.

    Args:
        ext: CUDAIPCExtension instance
        new_value: New IPC memory name (str)
        prev: Previous IPC memory name
    """
    # Convert to string
    new_value = str(new_value)
    prev = str(prev) if prev is not None else ""

    if new_value == prev:
        return

    ext._log("IPC memory name changed - reinitializing", force=True)

    # Clean up existing connection
    ext.cleanup()

    # Update internal state
    ext.shm_name = new_value

    # Re-initialize based on mode
    if ext.mode == "Sender":
        ext._log("Sender will reinitialize on next frame export", force=False)
    elif ext.mode == "Receiver":
        # Reset retry counter to trigger immediate reconnection
        ext._rx_frames_since_last_retry = ext._rx_retry_interval_frames
        ext._log("Receiver will attempt reconnection on next frame", force=False)


def handle_numslots_change(ext: object, new_value: object, prev: object) -> None:
    """Handle Numslots parameter change.

    Args:
        ext: CUDAIPCExtension instance
        new_value: New number of ring buffer slots (int or str)
        prev: Previous number of slots
    """
    # Convert to int if string
    new_value = int(new_value)
    prev = int(prev) if prev is not None else 0

    if new_value == prev:
        return

    # Receiver ignores manual Numslots changes — slot count comes from sender via SharedMemory.
    # The parameter is disabled in the UI when in Receiver mode, but this guard handles
    # any edge case where the callback fires anyway.
    if ext.mode == "Receiver":
        ext._log("Numslots change ignored in Receiver mode (controlled by sender)", force=True)
        return

    # Skip if component is active — Numslots should be disabled in UI, but guard
    # against script-based changes which bypass the UI parameter enable state.
    try:
        if bool(ext.ownerComp.par.Active.eval()):
            ext._log("Numslots change ignored while Active (deactivate first)", force=True)
            return
    except AttributeError:
        pass

    # Validate slot count (2-5 slots supported)
    if new_value < 2 or new_value > 5:
        ext._log(f"WARNING: Numslots={new_value} outside recommended range (2-5)", force=True)

    ext._log("Ring buffer slot count changed - reinitializing", force=True)

    # Clean up existing buffers
    ext.cleanup()

    # Update internal state
    ext.num_slots = new_value

    # Re-initialize based on mode
    if ext.mode == "Sender":
        ext._log("Sender will recreate ring buffer on next frame export", force=False)
    elif ext.mode == "Receiver":
        # Reset retry counter to trigger immediate reconnection
        ext._rx_frames_since_last_retry = ext._rx_retry_interval_frames
        ext._log("Receiver will reconnect with new slot count on next frame", force=False)


def handle_debug_change(ext: object, new_value: object, prev: object) -> None:
    """Handle Debug parameter toggle.

    Args:
        ext: CUDAIPCExtension instance
        new_value: New debug state (bool or int)
        prev: Previous debug state
    """
    # Convert to bool
    new_value = bool(new_value)

    ext.verbose_performance = new_value

    if new_value:
        ext._log("Debug logging ENABLED", force=True)
    else:
        ext._log("Debug logging DISABLED", force=True)


def handle_hidebuiltin_change(ext: object, new_value: object, prev: object) -> None:
    """Handle Hidebuiltin parameter toggle.

    Args:
        ext: CUDAIPCExtension instance
        new_value: New hide state (bool or int)
        prev: Previous hide state
    """
    new_value = bool(new_value)
    parent().showCustomOnly = new_value
    ext._log(f"Built-in parameters {'hidden' if new_value else 'visible'}", force=True)


def handle_mode_change(ext: object, new_value: object, prev: object) -> None:
    """Handle Mode parameter change ('Sender' <-> 'Receiver').

    Args:
        ext: CUDAIPCExtension instance
        new_value: New mode ('Sender' or 'Receiver')
        prev: Previous mode
    """
    # Convert to string
    new_value = str(new_value)
    prev = str(prev) if prev is not None else ""

    if new_value == prev:
        return

    ext._log(f"Mode switching: {prev} -> {new_value}", force=True)

    # Use extension's built-in switch_mode method
    try:
        ext.switch_mode(new_value)
        ext._log(f"Mode switch complete: now in {new_value} mode", force=True)

        # Update 'bg' selectTOP to display correct buffer
        try:
            bg_select = parent().op("bg")
            if bg_select:
                if new_value == "Sender":
                    bg_select.par.top = "ExportBuffer"
                    ext._log("Updated bg selectTOP -> ExportBuffer", force=False)
                elif new_value == "Receiver":
                    bg_select.par.top = "ImportBuffer"
                    ext._log("Updated bg selectTOP -> ImportBuffer", force=False)
        except (AttributeError, RuntimeError) as e:
            ext._log(f"Could not update bg selectTOP: {e}", force=False)

    except (AttributeError, RuntimeError) as e:
        ext._log(f"ERROR switching mode: {e}", force=True)


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
