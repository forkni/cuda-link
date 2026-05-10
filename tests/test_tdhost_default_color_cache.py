"""
Loop A — Bug 1 regression: RealTDHost._default_color stale-cache.

Problem: RealTDHost.__init__ caches owner_comp.color once at construction time.
If the COMP was already yellow (e.g. .tox saved while tinted, or re-init ran
after Phase D already set the warning tint), _default_color is cached AS the
warning colour. clear_status() then faithfully restores it to yellow -- stuck.

These tests exercise RealTDHost in pure Python using a FakeOwnerComp that has
a settable .color field.  No TD runtime required.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "td_exporter"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from TDHost import RealTDHost  # noqa: E402

_WARNING_COLOR = (0.9137, 1.0, 0.0)
_ERROR_COLOR = (0.7, 0.0, 0.0)


class FakeOwnerComp:
    """Minimal TD ownerComp stand-in with a settable .color property."""

    def __init__(self, initial_color: tuple[float, float, float]) -> None:
        self._color: tuple[float, float, float] = initial_color
        self._store: dict = {}

    @property
    def color(self) -> tuple[float, float, float]:
        return self._color

    @color.setter
    def color(self, value: tuple[float, float, float]) -> None:
        self._color = tuple(value)

    def store(self, k: str, v: object) -> None:
        self._store[k] = v

    def unstore(self, k: str) -> None:
        self._store.pop(k, None)

    def addScriptError(self, msg: str) -> None:
        pass

    def clearScriptErrors(self, error: str = "*") -> None:
        pass


class _NoPar:
    """Attribute-less namespace so owner_comp.par.Active raises AttributeError."""

    pass


class FakeOwnerCompNoPar(FakeOwnerComp):
    """FakeOwnerComp without .par — lets RealTDHost.__init__ take the except branch."""

    par = _NoPar()


# ---------------------------------------------------------------------------
# Baseline — comp starts at a normal grey colour (should always pass)
# ---------------------------------------------------------------------------


def test_clear_status_when_comp_starts_grey_restores_grey() -> None:
    comp = FakeOwnerCompNoPar(initial_color=(0.55, 0.55, 0.55))
    host = RealTDHost(comp)
    host.set_warning_status("bad pixel format")
    assert comp.color == _WARNING_COLOR, "set_warning_status must tint yellow"
    host.clear_status()
    assert comp.color == (0.55, 0.55, 0.55), "clear_status must restore grey when comp started grey"


def test_set_error_status_then_clear_restores_grey() -> None:
    comp = FakeOwnerCompNoPar(initial_color=(0.55, 0.55, 0.55))
    host = RealTDHost(comp)
    host.set_error_status("init failed")
    assert comp.color == _ERROR_COLOR, "set_error_status must tint red"
    host.clear_status()
    assert comp.color == (0.55, 0.55, 0.55), "clear_status must restore grey after error"


# ---------------------------------------------------------------------------
# Regression — comp starts ALREADY YELLOW (stale .tox or re-init after tint)
# ---------------------------------------------------------------------------


def test_clear_status_when_comp_starts_yellow_does_not_leave_yellow() -> None:
    """BUG DEMO: FAILS on un-fixed code; PASSES after F1 fix.

    When the COMP is constructed while already yellow (e.g. .tox was saved mid-
    warning or the extension re-initialised after Phase D set the tint),
    _default_color is cached as (1.0, 0.8, 0.2).  clear_status() then restores
    to yellow -- the tint never clears even after the format is fixed.
    """
    comp = FakeOwnerCompNoPar(initial_color=_WARNING_COLOR)  # pre-tinted COMP
    host = RealTDHost(comp)
    host.set_warning_status("bad pixel format")
    host.clear_status()
    assert comp.color != _WARNING_COLOR, (
        "clear_status must NOT leave COMP yellow when _default_color was cached "
        "as the warning colour (stale .tox scenario)"
    )
    assert comp.color not in (_WARNING_COLOR, _ERROR_COLOR), "clear_status must restore to a non-managed colour"


def test_clear_status_when_comp_starts_red_does_not_leave_red() -> None:
    """Parallel regression for the red/error tint."""
    comp = FakeOwnerCompNoPar(initial_color=_ERROR_COLOR)
    host = RealTDHost(comp)
    host.set_error_status("init failed")
    host.clear_status()
    assert comp.color not in (_WARNING_COLOR, _ERROR_COLOR), (
        "clear_status must NOT leave COMP red when _default_color was cached as the error colour (stale .tox scenario)"
    )


# ---------------------------------------------------------------------------
# Fix invariant — after fix, a warning+clear cycle is idempotent
# ---------------------------------------------------------------------------


def test_multiple_warn_clear_cycles_do_not_drift() -> None:
    """After fix: repeated warn→clear cycles always return to the original colour."""
    original = (0.3, 0.6, 0.9)  # user-customised purple-ish node
    comp = FakeOwnerCompNoPar(initial_color=original)
    host = RealTDHost(comp)
    for _ in range(3):
        host.set_warning_status("bad fmt")
        assert comp.color == _WARNING_COLOR
        host.clear_status()
        assert comp.color == original, "each clear_status must restore to original custom colour"


# ---------------------------------------------------------------------------
# Init-reset regression — COMP boots tinted; __init__ must restore grey
# ---------------------------------------------------------------------------


class FakeOwnerCompWithFetch(FakeOwnerComp):
    """FakeOwnerComp with fetch() and a _NoPar so RealTDHost __init__ runs cleanly."""

    par = _NoPar()

    def fetch(self, key: str, default: object = None) -> object:
        return self._store.get(key, default)


def test_init_resets_yellow_tint_to_default_grey() -> None:
    """COMP saved yellow → __init__ must restore to _DEFAULT_NODE_COLOR immediately."""
    from TDHost import _DEFAULT_NODE_COLOR

    comp = FakeOwnerCompWithFetch(initial_color=_WARNING_COLOR)
    RealTDHost(comp)
    assert comp.color not in (_WARNING_COLOR, _ERROR_COLOR), (
        "__init__ must reset a stale yellow tint to a non-managed colour"
    )
    assert comp.color == _DEFAULT_NODE_COLOR


def test_init_resets_red_tint_to_default_grey() -> None:
    """COMP saved red → __init__ must restore to _DEFAULT_NODE_COLOR immediately."""
    from TDHost import _DEFAULT_NODE_COLOR

    comp = FakeOwnerCompWithFetch(initial_color=_ERROR_COLOR)
    RealTDHost(comp)
    assert comp.color == _DEFAULT_NODE_COLOR


def test_init_does_not_change_neutral_colour() -> None:
    """COMP with a custom (non-managed) colour → __init__ must NOT change it."""
    custom = (0.3, 0.6, 0.9)
    comp = FakeOwnerCompWithFetch(initial_color=custom)
    RealTDHost(comp)
    assert comp.color == custom, "__init__ must not touch a non-managed colour"


def test_init_clears_stale_storage_on_yellow_boot() -> None:
    """COMP saved yellow with a stale status message → __init__ must unstore it."""
    comp = FakeOwnerCompWithFetch(initial_color=_WARNING_COLOR)
    comp.store("cuda_link_status_msg", "WARNING: stale from prior session")
    RealTDHost(comp)
    assert comp.fetch("cuda_link_status_msg") is None, (
        "__init__ must unstore cuda_link_status_msg when resetting stale tint"
    )
