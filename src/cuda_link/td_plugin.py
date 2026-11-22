# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""Where the TouchDesigner Custom TOP DLLs of this install are.

The release platform wheel (``cp311-win_amd64``) ships ``CudaLinkOutTOP.dll`` and
``CudaLinkInTOP.dll`` in ``cuda_link/td_plugins/``, built from the same tree and version as the
Python package. A host that installs the wheel (for example a TouchDesigner project with a venv)
points a Custom TOP's Plugin Path at :func:`plugin_path`. Dev builds and the ``py3-none-any``
wheel carry no DLLs; then :func:`plugin_path` returns None.

Loading rules are in docs/CPP_TOP.md "Install: from the wheel": TouchDesigner refuses a second
copy of the same plugin, so a host must not load these while a system Plugins folder copy is
installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

PluginKind = Literal["out", "in"]

#: DLL file name per Custom TOP: "out" = CUDA Link Out (sender), "in" = CUDA Link In (receiver).
DLL_NAMES: dict[str, str] = {"out": "CudaLinkOutTOP.dll", "in": "CudaLinkInTOP.dll"}

PLUGIN_DIR_NAME = "td_plugins"


def plugin_dir() -> Path:
    """The folder that holds the shipped DLLs (it may not exist in a dev build)."""
    return Path(__file__).resolve().parent / PLUGIN_DIR_NAME


def plugin_path(kind: PluginKind) -> Path | None:
    """Absolute path of the DLL for ``kind``, or None when this install carries no DLLs.

    Raises:
        ValueError: ``kind`` is not "out" or "in".
    """
    try:
        name = DLL_NAMES[kind]
    except KeyError:
        raise ValueError(f"kind must be 'out' or 'in', not {kind!r}") from None
    path = plugin_dir() / name
    return path if path.is_file() else None


def has_plugins() -> bool:
    """True when both DLLs are present."""
    return all(plugin_path(kind) is not None for kind in ("out", "in"))
