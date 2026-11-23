# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""Write the help text (Alt+hover in TD) of the CUDAIPCLink .tox custom parameters.

TD stores ``par.help`` only inside the .tox. The text comes from the COMP's own
CUDAIPCExtension DAT: ``par_help()`` there builds it from the ``PARS`` table and the enable
rule (``td_exporter/CUDAIPCExtension.py``). So update that DAT first, then run this.

Run inside TouchDesigner (Textport or twozero), with the COMP to update:

    import runpy
    m = runpy.run_path(r"<repo>/scripts/tox_par_help.py")
    m["apply"](op("/project1/CUDAIPCLink"))    # -> names whose help changed
    m["missing"](op("/project1/CUDAIPCLink"))  # -> custom pars with no help

Then save the .tox (TOXES/, at release time only).
"""

from __future__ import annotations

from typing import Any


def par_help(comp: Any) -> dict[str, str]:
    """The help text by par name, from the CUDAIPCExtension DAT of *comp*."""
    module = getattr(comp.op("CUDAIPCExtension"), "module", None)
    build = getattr(module, "par_help", None)
    if build is None:
        raise RuntimeError("the CUDAIPCExtension DAT of this COMP has no par_help(); update the DAT text first")
    return build()


def apply(comp: Any) -> list[str]:
    """Write par_help(comp) into *comp*'s custom pars; return the names whose help changed.

    A par the COMP does not have (an older .tox) is skipped.
    """
    pars = {par.name: par for par in comp.customPars}
    changed: list[str] = []
    for name, text in par_help(comp).items():
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
