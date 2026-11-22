# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""
Script CHOP callbacks for the CUDAIPCLink `slot_meta` CHOP (protocol v0.6).

The `slot_meta` Script CHOP inside the .tox shows the slot meta of the current frame, one
sample per channel, with the same channel names as the C++ In TOP Info CHOP:

  producer_frame       sender TD absolute frame, mod 2^24 (exact in float32)
  stream_id            sender Streamid
  latency_us           publish-to-import time (Receiver only)
  frame_gap_count      frames the ring overwrote before the import (Receiver only)
  producer_skip_count  sender skipped TD frames between two publishes (Receiver only)

In Sender mode it shows the last published producer_frame and stream_id; the other
channels are 0. An Out CHOP `out_slot_meta` exposes it outside the COMP.

Wire it:  slot_meta.par.callbacks = op('script_chop_callbacks')
In Receiver mode the ImportBuffer Script TOP force-cooks `slot_meta` right after each
import (script_top_callbacks.py), so CHOP readers see the frame that is on the output.
"""


def _extension():
    """Return the CUDAIPCExtension, or None if it is not available (never raises)."""
    try:
        return getattr(parent().ext, "CUDAIPCExtension", None)  # noqa: F821
    except Exception:  # noqa: BLE001 -- td.tdAttributeError, plus parent() == None
        return None


def write_channels(scriptOp: object, channels: dict) -> None:
    """Replace the CHOP's channels with one sample per (name, value) pair."""
    scriptOp.clear()
    scriptOp.numSamples = 1
    for name, value in channels.items():
        scriptOp.appendChan(name)[0] = value


def onCook(scriptOp: object) -> None:
    """Called every time the slot_meta Script CHOP cooks."""
    ext = _extension()
    channels = ext.slot_meta_channels() if ext is not None else {}
    write_channels(scriptOp, channels)


def onSetupParameters(scriptOp: object) -> None:
    """Called when Setup Parameters is pressed."""
    return


def onPulse(par: object) -> None:
    return
