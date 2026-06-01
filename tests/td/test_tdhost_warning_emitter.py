"""
Regression tests for RealTDHost warning-emitter cook-on-transition behaviour.

RealTDHost.set_warning_status / set_error_status / clear_status must force-cook
the 'warning_emitter' Script TOP child only when the status message changes, not
on every call. This keeps the hot path (bad-format frames calling set_warning_status
every frame) from force-cooking the emitter repeatedly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "td_exporter"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from TDHost import RealTDHost  # noqa: E402

_WARNING_COLOR = (0.9137, 1.0, 0.0)
_ERROR_COLOR = (0.7, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Fake TD objects
# ---------------------------------------------------------------------------


class _NoPar:
    pass


class FakeCookable:
    """Minimal Script TOP double that counts force-cook calls."""

    def __init__(self) -> None:
        self.cook_count = 0

    def cook(self, force: bool = False) -> None:
        if force:
            self.cook_count += 1


class FakeOwnerCompWithEmitter:
    """ownerComp stand-in that exposes .color, storage, and op('warning_emitter')."""

    par = _NoPar()

    def __init__(self, emitter: FakeCookable | None = None) -> None:
        self._color = (0.55, 0.55, 0.55)
        self._store: dict = {}
        self._emitter = emitter

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
        return self._emitter if name == "warning_emitter" else None


def _make_host(emitter: FakeCookable | None = None) -> tuple[RealTDHost, FakeOwnerCompWithEmitter]:
    comp = FakeOwnerCompWithEmitter(emitter=emitter)
    host = RealTDHost(comp)
    return host, comp


# ---------------------------------------------------------------------------
# Cook-on-first-transition
# ---------------------------------------------------------------------------


def test_set_warning_status_cooks_emitter_on_first_call() -> None:
    emitter = FakeCookable()
    host, _ = _make_host(emitter)
    host.set_warning_status("bad format")
    assert emitter.cook_count == 1


def test_set_error_status_cooks_emitter_on_first_call() -> None:
    emitter = FakeCookable()
    host, _ = _make_host(emitter)
    host.set_error_status("init failed")
    assert emitter.cook_count == 1


def test_clear_status_cooks_emitter_when_message_was_set() -> None:
    emitter = FakeCookable()
    host, _ = _make_host(emitter)
    host.set_warning_status("bad format")  # cook 1
    host.clear_status()  # cook 2
    assert emitter.cook_count == 2


# ---------------------------------------------------------------------------
# State-transition gate: no redundant cooks
# ---------------------------------------------------------------------------


def test_set_warning_status_same_msg_does_not_recook() -> None:
    emitter = FakeCookable()
    host, _ = _make_host(emitter)
    host.set_warning_status("bad format")
    host.set_warning_status("bad format")  # identical — no re-cook
    assert emitter.cook_count == 1


def test_set_warning_status_different_msg_does_recook() -> None:
    emitter = FakeCookable()
    host, _ = _make_host(emitter)
    host.set_warning_status("bad format")  # cook 1
    host.set_warning_status("other error")  # cook 2 (msg changed)
    assert emitter.cook_count == 2


def test_clear_status_when_already_clear_does_not_cook() -> None:
    emitter = FakeCookable()
    host, _ = _make_host(emitter)
    host.clear_status()  # nothing was set → no cook
    assert emitter.cook_count == 0


def test_double_clear_does_not_cook_twice() -> None:
    emitter = FakeCookable()
    host, _ = _make_host(emitter)
    host.set_warning_status("bad format")  # cook 1
    host.clear_status()  # cook 2
    host.clear_status()  # already clear → no cook
    assert emitter.cook_count == 2


# ---------------------------------------------------------------------------
# Graceful degradation — no emitter op present (old .tox without the child)
# ---------------------------------------------------------------------------


def test_no_emitter_op_does_not_raise_on_warning() -> None:
    host, _ = _make_host(emitter=None)
    host.set_warning_status("bad format")  # must not raise


def test_no_emitter_op_does_not_raise_on_clear() -> None:
    host, _ = _make_host(emitter=None)
    host.set_warning_status("bad format")
    host.clear_status()  # must not raise


def test_no_emitter_op_does_not_raise_on_error() -> None:
    host, _ = _make_host(emitter=None)
    host.set_error_status("init failed")  # must not raise
