"""
Regression tests for RealTDHost Status parameter writes.

RealTDHost._write_status_par deduplicates writes so that repeated calls with
the same value do not call set_param_value again. set_info_status is the
normal-flow path (resolution + dtype); set_warning_status, set_error_status,
and clear_status each also push a value to the Status par.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "td_exporter"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from TDHost import RealTDHost  # noqa: E402

# ---------------------------------------------------------------------------
# Fake TD objects
# ---------------------------------------------------------------------------


class _NoPar:
    pass


class FakeOwnerCompRecordingPar:
    """ownerComp stub that records set_param_value calls via par attribute writes."""

    par = _NoPar()

    def __init__(self) -> None:
        self._color = (0.55, 0.55, 0.55)
        self._store: dict = {}
        self.par_writes: list[tuple[str, object]] = []

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value) -> None:
        self._color = tuple(value)

    def store(self, k: str, v: object) -> None:
        self._store[k] = v

    def unstore(self, k: str) -> None:
        self._store.pop(k, None)

    def addScriptError(self, msg: str) -> None:
        pass

    def clearScriptErrors(self, error: str = "*") -> None:
        pass

    def op(self, name: str):
        return None

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)


class _ParProxy:
    """Records setattr calls to simulate par.Status writes."""

    def __init__(self, owner: FakeOwnerCompRecordingPar) -> None:
        object.__setattr__(self, "_owner", owner)

    def __setattr__(self, name: str, value: object) -> None:
        object.__getattribute__(self, "_owner").par_writes.append((name, value))


def _make_host() -> tuple[RealTDHost, FakeOwnerCompRecordingPar]:
    comp = FakeOwnerCompRecordingPar()
    comp.par = _ParProxy(comp)
    host = RealTDHost(comp)
    return host, comp


# ---------------------------------------------------------------------------
# set_warning_status writes Status par
# ---------------------------------------------------------------------------


def test_set_warning_status_writes_status_par() -> None:
    host, comp = _make_host()
    host.set_warning_status("bad format")
    writes = [(n, v) for n, v in comp.par_writes if n == "Status"]
    assert writes == [("Status", "WARNING: bad format")]


def test_set_warning_status_same_msg_does_not_rewrite_par() -> None:
    host, comp = _make_host()
    host.set_warning_status("bad format")
    before = len([n for n, _ in comp.par_writes if n == "Status"])
    host.set_warning_status("bad format")
    after = len([n for n, _ in comp.par_writes if n == "Status"])
    assert after == before  # no second write


def test_set_warning_status_different_msg_rewrites_par() -> None:
    host, comp = _make_host()
    host.set_warning_status("bad format")
    host.set_warning_status("other error")
    writes = [v for n, v in comp.par_writes if n == "Status"]
    assert writes == ["WARNING: bad format", "WARNING: other error"]


# ---------------------------------------------------------------------------
# set_error_status writes Status par
# ---------------------------------------------------------------------------


def test_set_error_status_writes_status_par() -> None:
    host, comp = _make_host()
    host.set_error_status("init failed")
    writes = [v for n, v in comp.par_writes if n == "Status"]
    assert writes == ["ERROR: init failed"]


def test_set_error_status_same_msg_does_not_rewrite_par() -> None:
    host, comp = _make_host()
    host.set_error_status("init failed")
    before = len([n for n, _ in comp.par_writes if n == "Status"])
    host.set_error_status("init failed")
    after = len([n for n, _ in comp.par_writes if n == "Status"])
    assert after == before


# ---------------------------------------------------------------------------
# clear_status writes "OK" to Status par
# ---------------------------------------------------------------------------


def test_clear_status_writes_idle_to_par() -> None:
    host, comp = _make_host()
    host.set_warning_status("bad format")
    comp.par_writes.clear()
    host.clear_status()
    writes = [v for n, v in comp.par_writes if n == "Status"]
    assert writes == ["Idle"]


def test_clear_status_from_clean_state_writes_idle() -> None:
    host, comp = _make_host()
    host.clear_status()
    writes = [v for n, v in comp.par_writes if n == "Status"]
    assert writes == ["Idle"]


def test_double_clear_does_not_rewrite_idle() -> None:
    host, comp = _make_host()
    host.set_warning_status("bad format")
    host.clear_status()  # writes "Idle"
    before = len([n for n, _ in comp.par_writes if n == "Status"])
    host.clear_status()  # already "Idle" → no second write
    after = len([n for n, _ in comp.par_writes if n == "Status"])
    assert after == before


# ---------------------------------------------------------------------------
# set_info_status writes informational string with no tint side effects
# ---------------------------------------------------------------------------


def test_set_info_status_writes_par() -> None:
    host, comp = _make_host()
    host.set_info_status("1920x1080 float32 4ch")
    writes = [v for n, v in comp.par_writes if n == "Status"]
    assert writes == ["1920x1080 float32 4ch"]


def test_set_info_status_dedup() -> None:
    host, comp = _make_host()
    host.set_info_status("1920x1080 float32 4ch")
    before = len([n for n, _ in comp.par_writes if n == "Status"])
    host.set_info_status("1920x1080 float32 4ch")
    after = len([n for n, _ in comp.par_writes if n == "Status"])
    assert after == before


def test_set_info_status_after_warning_overwrites_par() -> None:
    host, comp = _make_host()
    host.set_warning_status("bad format")
    host.set_info_status("1920x1080 float32 4ch")
    writes = [v for n, v in comp.par_writes if n == "Status"]
    assert "WARNING: bad format" in writes
    assert "1920x1080 float32 4ch" in writes
    assert writes[-1] == "1920x1080 float32 4ch"


def test_set_info_status_does_not_tint_comp() -> None:
    host, comp = _make_host()
    host.set_info_status("1920x1080 float32 4ch")
    assert comp._color == (0.55, 0.55, 0.55)
