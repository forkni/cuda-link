# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""scripts/tox_par_help.py and CUDAIPCExtension.par_help: the help of the .tox custom pars."""

from __future__ import annotations

import runpy
import types
from dataclasses import dataclass
from pathlib import Path

import CUDAIPCExtension as extension_module  # noqa: N813 -- the module, not the class
import pytest
from CUDAIPCExtension import PARS, _enable_sentence, par_help

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

    def op(self, name: str) -> object | None:
        return types.SimpleNamespace(module=extension_module) if name == "CUDAIPCExtension" else None


@pytest.fixture(scope="module")
def script() -> dict:
    return runpy.run_path(str(_SCRIPT))


def test_apply_leaves_no_par_without_help(script: dict) -> None:
    """After apply(), every non-header, non-Semver par of the .tox has help."""
    comp = _Comp(_TOX_PARS)
    assert "Streamid" in script["missing"](comp)

    script["apply"](comp)

    assert script["missing"](comp) == []


def test_every_par_in_the_table_has_help() -> None:
    assert [spec.name for spec in PARS if not spec.help.strip()] == []


def test_the_greyed_out_sentence_comes_from_the_enable_rule() -> None:
    """The help cannot disagree with CUDAIPCExtension.par_enable_state: it is made from it."""
    text = par_help()
    assert text["Streamid"].endswith(" \nGreyed out in Receiver mode.")
    assert text["Framewaitms"].endswith(" \nGreyed out in Sender mode.")
    assert text["Numslots"].endswith(" \nOnly editable in Sender mode while inactive.")
    assert "reyed out" not in text["Mode"]


def test_a_changed_enable_rule_changes_the_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        extension_module, "par_enable_state", lambda mode, active: {"Streamid": mode == "Receiver" and not active}
    )

    assert _enable_sentence("Streamid") == "Only editable in Receiver mode while inactive."
    assert _enable_sentence("Framewaitms") == ""


def test_a_par_the_rule_always_enables_has_no_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extension_module, "par_enable_state", lambda mode, active: {"Debug": True})

    assert _enable_sentence("Debug") == ""
    assert par_help()["Debug"] == next(spec.help for spec in PARS if spec.name == "Debug")


def test_every_par_in_the_enable_rule_has_a_row() -> None:
    """par_help() makes the sentence for the pars in PARS only; a rule key outside it is lost."""
    assert set(extension_module.par_enable_state("Sender", False)) <= {spec.name for spec in PARS}


def test_apply_reports_a_comp_whose_extension_dat_does_not_compile(script: dict) -> None:
    class _BrokenDAT:
        @property
        def module(self) -> object:
            raise SyntaxError("the DAT text does not compile")

    comp = _Comp(_TOX_PARS)
    comp.op = lambda name: _BrokenDAT()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="par_help"):
        script["apply"](comp)


def test_apply_reports_a_comp_without_the_extension_dat(script: dict) -> None:
    comp = _Comp(_TOX_PARS)
    comp.op = lambda name: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="par_help"):
        script["apply"](comp)


def test_an_enable_rule_with_no_sentence_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new rule shape must get a sentence, not a silent gap in the help."""
    monkeypatch.setattr(extension_module, "par_enable_state", lambda mode, active: {"Streamid": active})

    with pytest.raises(ValueError, match="Streamid"):
        _enable_sentence("Streamid")


def test_apply_reports_a_comp_whose_extension_dat_has_no_par_help(script: dict) -> None:
    """A .tox with the 2.0.0 extension text: the DAT must be updated first."""
    comp = _Comp(_TOX_PARS)
    comp.op = lambda name: types.SimpleNamespace(module=types.SimpleNamespace())  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="par_help"):
        script["apply"](comp)


def test_apply_is_idempotent_and_skips_absent_pars(script: dict) -> None:
    """A second apply() changes nothing; a .tox without a par is not an error."""
    comp = _Comp([("Mode", "Menu")])

    assert script["apply"](comp) == ["Mode"]
    assert script["apply"](comp) == []
    assert comp.help_of("Mode") == par_help()["Mode"]
