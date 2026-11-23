# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""scripts/tox_par_help.py: the help text source for the .tox custom pars."""

from __future__ import annotations

import runpy
from dataclasses import dataclass
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "tox_par_help.py"

# The custom pars of the shipped CUDAIPCLink .tox (TD 2025.33230 read, 2026-11-23).
_TOX_PARS = [
    ("Status", "Str"),
    ("Mode", "Menu"),
    ("Ipcmemname", "Str"),
    ("Active", "Toggle"),
    ("Numslots", "Menu"),
    ("Streamid", "Int"),
    ("Framewaitms", "Int"),
    ("Debug", "Toggle"),
    ("Cudalinkheader", "Header"),
    ("Cudalinkheader2", "Header"),
    ("Semverversion", "Str"),
    ("Semvertoxsavebuild", "Str"),
    ("Nativetops", "Str"),
    ("Semverauthor", "Str"),
    ("Semvershowhelp", "Pulse"),
    ("Hidebuiltin", "Toggle"),
]


@dataclass
class _Par:
    name: str
    style: str
    help: str = ""


class _Comp:
    def __init__(self, pars: list[tuple[str, str]]) -> None:
        self.customPars = [_Par(name, style) for name, style in pars]

    def help_of(self, name: str) -> str:
        return next(p.help for p in self.customPars if p.name == name)


@pytest.fixture(scope="module")
def script() -> dict:
    return runpy.run_path(str(_SCRIPT))


def test_apply_leaves_no_par_without_help(script: dict) -> None:
    """After apply(), every non-header, non-Semver par of the .tox has help."""
    comp = _Comp(_TOX_PARS)
    assert "Streamid" in script["missing"](comp)

    script["apply"](comp)

    assert script["missing"](comp) == []


def test_mode_specific_pars_say_which_mode_greys_them_out(script: dict) -> None:
    """The help matches the extension's gating (CUDAIPCExtension._sync_mode_pars)."""
    help_text = script["PAR_HELP"]
    assert "Greyed out in Receiver mode" in help_text["Streamid"]
    assert "Greyed out in Sender mode" in help_text["Framewaitms"]


def test_apply_is_idempotent_and_skips_absent_pars(script: dict) -> None:
    """A second apply() changes nothing; a .tox without a par is not an error."""
    comp = _Comp([("Mode", "Menu")])

    assert script["apply"](comp) == ["Mode"]
    assert script["apply"](comp) == []
    assert comp.help_of("Mode") == script["PAR_HELP"]["Mode"]
