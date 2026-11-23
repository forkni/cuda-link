# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""Help text (Alt+hover in TD) of the CUDAIPCLink .tox custom parameters.

TD stores ``par.help`` only inside the .tox, so this file is the source that a rebuilt or
repaired .tox gets its help from. The Semver* parameters are not here: the Semantic Version
Control tool owns them. Header parameters have no help.

Run inside TouchDesigner (Textport or twozero), with the COMP to update:

    import runpy
    m = runpy.run_path(r"<repo>/scripts/tox_par_help.py")
    m["apply"](op("/project1/CUDAIPCLink"))    # -> names whose help changed
    m["missing"](op("/project1/CUDAIPCLink"))  # -> custom pars with no help

Then save the .tox (TOXES/, at release time only).
"""

from __future__ import annotations

from typing import Any

PAR_HELP: dict[str, str] = {
    "Status": (
        "Live status of the pipeline, written by the component: Idle, the active texture format, "
        "or a WARNING / ERROR line. Read-only."
    ),
    "Mode": (
        "'Sender' exports TD textures to an external process. \n"
        "'Receiver' imports frames from an external process into TD."
    ),
    "Ipcmemname": (
        "SharedMemory channel name used to exchange GPU handles.\nMust match exactly on both sender and receiver."
    ),
    "Active": (
        "Enables / disables the CUDA IPC pipeline. \nWhen off, no GPU work is performed and all resources are released."
    ),
    "Numslots": (
        "Ring buffer depth (2–4 slots). \n"
        "More slots reduce contention risk but use more GPU memory. \n"
        "Only editable in Sender mode while inactive."
    ),
    "Streamid": (
        "Sender mode: a number written as stream_id into each frame's slot meta, so one receiver "
        "network that reads several streams can tell them apart. \n"
        "A label only: each stream still needs its own Ipcmemname. Applies from the next frame. \n"
        "Greyed out in Receiver mode."
    ),
    "Framewaitms": (
        "Receiver mode: wait up to this time (ms) for the sender's doorbell when no new frame "
        "is there. 0 turns the wait off. Applies from the next frame. \n"
        "Greyed out in Sender mode."
    ),
    "Debug": ("Prints average per-frame timing metrics to the Textport every 100 frames. \nToggle safely at any time."),
    "Nativetops": (
        "The installed CUDA-Link C++ TOPs (Plugins/CudaLink/VERSION.txt): version and CUDA or "
        "build, or 'not installed'. Read-only. A version that differs from this component "
        "tints the COMP yellow."
    ),
    "Hidebuiltin": ("Hides built-in TouchDesigner parameters (Common, Extensions), showing only CUDA IPC controls."),
}


def apply(comp: Any) -> list[str]:
    """Write PAR_HELP into *comp*'s custom pars; return the names whose help changed.

    A par the COMP does not have (an older .tox) is skipped.
    """
    pars = {par.name: par for par in comp.customPars}
    changed: list[str] = []
    for name, text in PAR_HELP.items():
        par = pars.get(name)
        if par is not None and par.help != text:
            par.help = text
            changed.append(name)
    return changed


def missing(comp: Any) -> list[str]:
    """Custom pars of *comp* with empty help, except headers and the Semver* pars."""
    return [
        par.name
        for par in comp.customPars
        if not par.help and par.style != "Header" and not par.name.startswith("Semver")
    ]
