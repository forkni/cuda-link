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

import math
import struct

import pytest
from TDHost import RealTDHost

_WARNING_COLOR = (0.9137, 1.0, 0.0)
_ERROR_COLOR = (0.7, 0.0, 0.0)


def _f32(value: float) -> float:
    """Round-trip a float through IEEE-754 single precision, like TD's comp.color storage."""
    return struct.unpack("f", struct.pack("f", value))[0]


class FakeOwnerComp:
    """Minimal TD ownerComp stand-in with a settable .color property."""

    def __init__(self, initial_color: tuple[float, float, float]) -> None:
        self._store: dict = {}
        # Route through the (possibly-overridden) property setter so a subclass like
        # FakeOwnerCompF32 quantizes the constructor's initial colour exactly like every
        # later assignment -- a direct self._color = ... here would skip that override
        # and silently stop the F32 fake from reproducing the real-TD rounding bug.
        self.color = initial_color

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


class FakeOwnerCompF32(FakeOwnerComp):
    """FakeOwnerComp whose .color quantizes through float32, like real TD.

    Real TD stores comp.color internally as float32, so a value assigned as a Python
    float64 literal (e.g. our 0.7 _ERROR_COLOR) reads back slightly off (0.699999988...).
    This fake reproduces that rounding so tests can catch an exact-equality /
    exact-membership regression that only manifests against a live TD instance.
    """

    @property
    def color(self) -> tuple[float, float, float]:
        return self._color

    @color.setter
    def color(self, value: tuple[float, float, float]) -> None:
        self._color = tuple(_f32(v) for v in value)


class _NoPar:
    """Attribute-less namespace so owner_comp.par.Active raises AttributeError."""

    pass


class FakeOwnerCompNoPar(FakeOwnerComp):
    """FakeOwnerComp without .par — lets RealTDHost.__init__ take the except branch."""

    par = _NoPar()


class FakeOwnerCompF32NoPar(FakeOwnerCompF32):
    """FakeOwnerCompF32 without .par — lets RealTDHost.__init__ take the except branch."""

    par = _NoPar()


# Both fakes exercise the init-reset / warn-clear invariants; the F32 variant is the one
# that reproduces the real-TD bug (exact-equality colour comparison never matching).
_COMP_CLASSES = (FakeOwnerCompNoPar, FakeOwnerCompF32NoPar)


def _close(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """Tolerance-based colour comparison, mirroring TDHost._is_managed_color's tolerance.

    Exact tuple equality is the wrong check here: a float32-quantizing fake (and real TD)
    round-trips colour values, so a value assigned from a float64 literal reads back
    slightly off. Tests must assert on the same tolerance the fix uses, not bit-exactness.
    """
    return all(math.isclose(x, y, abs_tol=1e-3) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Baseline — comp starts at a normal grey colour (should always pass)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("comp_cls", _COMP_CLASSES)
def test_clear_status_when_comp_starts_grey_restores_grey(comp_cls: type[FakeOwnerComp]) -> None:
    comp = comp_cls(initial_color=(0.55, 0.55, 0.55))
    host = RealTDHost(comp)
    host.set_warning_status("bad pixel format")
    assert _close(comp.color, _WARNING_COLOR), "set_warning_status must tint yellow"
    host.clear_status()
    assert _close(comp.color, (0.55, 0.55, 0.55)), "clear_status must restore grey when comp started grey"


@pytest.mark.parametrize("comp_cls", _COMP_CLASSES)
def test_set_error_status_then_clear_restores_grey(comp_cls: type[FakeOwnerComp]) -> None:
    comp = comp_cls(initial_color=(0.55, 0.55, 0.55))
    host = RealTDHost(comp)
    host.set_error_status("init failed")
    assert _close(comp.color, _ERROR_COLOR), "set_error_status must tint red"
    host.clear_status()
    assert _close(comp.color, (0.55, 0.55, 0.55)), "clear_status must restore grey after error"


# ---------------------------------------------------------------------------
# Regression — comp starts ALREADY YELLOW (stale .tox or re-init after tint)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("comp_cls", _COMP_CLASSES)
def test_clear_status_when_comp_starts_yellow_does_not_leave_yellow(comp_cls: type[FakeOwnerComp]) -> None:
    """BUG DEMO: FAILS on un-fixed code (F32 variant); PASSES after F1+F2 fix.

    When the COMP is constructed while already yellow (e.g. .tox was saved mid-
    warning or the extension re-initialised after Phase D set the tint),
    _default_color is cached as the warning colour.  clear_status() then restores
    to yellow -- the tint never clears even after the format is fixed.  Against real TD
    (F32 variant) the un-fixed exact-equality membership check never even recognises the
    starting colour as "managed", making this fail even harder pre-fix.
    """
    comp = comp_cls(initial_color=_WARNING_COLOR)  # pre-tinted COMP
    host = RealTDHost(comp)
    host.set_warning_status("bad pixel format")
    host.clear_status()
    assert not _close(comp.color, _WARNING_COLOR), (
        "clear_status must NOT leave COMP yellow when _default_color was cached "
        "as the warning colour (stale .tox scenario)"
    )
    assert not _close(comp.color, _WARNING_COLOR) and not _close(comp.color, _ERROR_COLOR), (
        "clear_status must restore to a non-managed colour"
    )


@pytest.mark.parametrize("comp_cls", _COMP_CLASSES)
def test_clear_status_when_comp_starts_red_does_not_leave_red(comp_cls: type[FakeOwnerComp]) -> None:
    """Parallel regression for the red/error tint."""
    comp = comp_cls(initial_color=_ERROR_COLOR)
    host = RealTDHost(comp)
    host.set_error_status("init failed")
    host.clear_status()
    assert not _close(comp.color, _WARNING_COLOR) and not _close(comp.color, _ERROR_COLOR), (
        "clear_status must NOT leave COMP red when _default_color was cached as the error colour (stale .tox scenario)"
    )


# ---------------------------------------------------------------------------
# Fix invariant — after fix, a warning+clear cycle is idempotent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("comp_cls", _COMP_CLASSES)
def test_multiple_warn_clear_cycles_do_not_drift(comp_cls: type[FakeOwnerComp]) -> None:
    """After fix: repeated warn→clear cycles always return to the original colour."""
    original = (0.3, 0.6, 0.9)  # user-customised purple-ish node
    comp = comp_cls(initial_color=original)
    host = RealTDHost(comp)
    for _ in range(3):
        host.set_warning_status("bad fmt")
        assert _close(comp.color, _WARNING_COLOR)
        host.clear_status()
        assert _close(comp.color, original), "each clear_status must restore to original custom colour"


# ---------------------------------------------------------------------------
# Init-reset regression — COMP boots tinted; __init__ must restore grey
# ---------------------------------------------------------------------------


class FakeOwnerCompWithFetch(FakeOwnerComp):
    """FakeOwnerComp with fetch() and a _NoPar so RealTDHost __init__ runs cleanly."""

    par = _NoPar()

    def fetch(self, key: str, default: object = None) -> object:
        return self._store.get(key, default)


class FakeOwnerCompF32WithFetch(FakeOwnerCompF32):
    """FakeOwnerCompF32 with fetch() and a _NoPar so RealTDHost __init__ runs cleanly."""

    par = _NoPar()

    def fetch(self, key: str, default: object = None) -> object:
        return self._store.get(key, default)


_COMP_WITH_FETCH_CLASSES = (FakeOwnerCompWithFetch, FakeOwnerCompF32WithFetch)


@pytest.mark.parametrize("comp_cls", _COMP_WITH_FETCH_CLASSES)
def test_init_resets_yellow_tint_to_default_grey(comp_cls: type[FakeOwnerCompWithFetch]) -> None:
    """COMP saved yellow → __init__ must restore to _DEFAULT_NODE_COLOR immediately."""
    from TDHost import _DEFAULT_NODE_COLOR

    comp = comp_cls(initial_color=_WARNING_COLOR)
    RealTDHost(comp)
    assert not _close(comp.color, _WARNING_COLOR) and not _close(comp.color, _ERROR_COLOR), (
        "__init__ must reset a stale yellow tint to a non-managed colour"
    )
    assert _close(comp.color, _DEFAULT_NODE_COLOR)


@pytest.mark.parametrize("comp_cls", _COMP_WITH_FETCH_CLASSES)
def test_init_resets_red_tint_to_default_grey(comp_cls: type[FakeOwnerCompWithFetch]) -> None:
    """COMP saved red → __init__ must restore to _DEFAULT_NODE_COLOR immediately."""
    from TDHost import _DEFAULT_NODE_COLOR

    comp = comp_cls(initial_color=_ERROR_COLOR)
    RealTDHost(comp)
    assert _close(comp.color, _DEFAULT_NODE_COLOR)


@pytest.mark.parametrize("comp_cls", _COMP_WITH_FETCH_CLASSES)
def test_init_does_not_change_neutral_colour(comp_cls: type[FakeOwnerCompWithFetch]) -> None:
    """COMP with a custom (non-managed) colour → __init__ must NOT change it."""
    custom = (0.3, 0.6, 0.9)
    comp = comp_cls(initial_color=custom)
    RealTDHost(comp)
    assert _close(comp.color, custom), "__init__ must not touch a non-managed colour"


@pytest.mark.parametrize("comp_cls", _COMP_WITH_FETCH_CLASSES)
def test_init_clears_stale_storage_on_yellow_boot(comp_cls: type[FakeOwnerCompWithFetch]) -> None:
    """COMP saved yellow with a stale status message → __init__ must unstore it."""
    comp = comp_cls(initial_color=_WARNING_COLOR)
    comp.store("cuda_link_status_msg", "WARNING: stale from prior session")
    RealTDHost(comp)
    assert comp.fetch("cuda_link_status_msg") is None, (
        "__init__ must unstore cuda_link_status_msg when resetting stale tint"
    )
