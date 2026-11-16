"""
parexecute_callbacks handlers: the decisions each parameter change makes.

test_td_callbacks_dispatch.py pins which handler onValueChange routes to; these pin what the
handlers do with the extension. The module is a TD textDAT script, so `parent()` is not a
real module attribute: tests install a stub via monkeypatch (raising=False).
"""

from __future__ import annotations

import types

import parexecute_callbacks as cb
import pytest


class _Ext:
    """Recording CUDAIPCExtension stand-in."""

    def __init__(self, *, mode: str = "Sender", active: bool = False, switch_error: Exception | None = None) -> None:
        self.mode = mode
        self._active = active
        self._switch_error = switch_error
        self.logs: list[tuple[str, bool]] = []
        self.reconfigured: list[tuple[str, object]] = []
        self.switched_to: list[str] = []
        self.cleanups = 0
        self.status_cleared = 0
        self.verbose_performance = None
        self._host = types.SimpleNamespace(clear_status=self._clear_status)

    def _clear_status(self) -> None:
        self.status_cleared += 1

    def _log(self, msg: str, force: bool = False) -> None:
        self.logs.append((msg, force))

    def is_active(self) -> bool:
        return self._active

    def reconfigure_and_reinit(self, key: str, value: object) -> None:
        self.reconfigured.append((key, value))

    def switch_mode(self, mode: str) -> None:
        if self._switch_error is not None:
            raise self._switch_error
        self.switched_to.append(mode)

    def cleanup(self) -> None:
        self.cleanups += 1

    def messages(self) -> list[str]:
        return [m for m, _ in self.logs]


class _Par:
    def __init__(self) -> None:
        self.top = None


class _Comp:
    """Stand-in for parent(): `.op('bg')`, `.par.Numslots`, `.showCustomOnly`."""

    def __init__(self, *, bg: object | None = None) -> None:
        self._bg = bg
        self.par = types.SimpleNamespace(Numslots=types.SimpleNamespace(enable=None))
        self.showCustomOnly = None

    def op(self, name: str) -> object | None:
        return self._bg if name == "bg" else None


def _install_parent(monkeypatch: pytest.MonkeyPatch, comp: _Comp) -> _Comp:
    monkeypatch.setattr(cb, "parent", lambda: comp, raising=False)
    return comp


# ---------------------------------------------------------------------------
# handle_numslots_change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("new_value", [2, 3, 5, "4"], ids=["min", "mid", "recommended-max", "string-from-par"])
def test_numslots_in_recommended_range_reconfigures_without_warning(new_value: object) -> None:
    ext = _Ext()

    cb.handle_numslots_change(ext, new_value, 3 if str(new_value) != "3" else 2)

    assert ext.reconfigured == [("num_slots", int(new_value))]
    assert not any("WARNING" in m for m in ext.messages())


@pytest.mark.parametrize("new_value", [6, 10], ids=["just-above-recommended", "supported-max"])
def test_numslots_above_recommended_but_supported_warns_and_reconfigures(new_value: int) -> None:
    ext = _Ext()

    cb.handle_numslots_change(ext, new_value, 3)

    assert ext.reconfigured == [("num_slots", new_value)]
    assert (f"WARNING: Numslots={new_value} outside recommended range (2-5)", True) in ext.logs


@pytest.mark.parametrize("new_value", [0, 1, 11, 99], ids=["zero", "below-min", "above-max", "far-above-max"])
def test_numslots_outside_supported_range_is_rejected(new_value: int) -> None:
    ext = _Ext()

    cb.handle_numslots_change(ext, new_value, 3)

    assert ext.reconfigured == []
    assert (f"WARNING: Numslots={new_value} outside supported range (2-10); change rejected", True) in ext.logs


def test_numslots_unchanged_is_a_noop() -> None:
    ext = _Ext()

    cb.handle_numslots_change(ext, "4", 4)

    assert ext.reconfigured == []
    assert ext.logs == []


def test_numslots_with_no_previous_value_treats_it_as_zero() -> None:
    ext = _Ext()

    cb.handle_numslots_change(ext, 3, None)

    assert ext.reconfigured == [("num_slots", 3)]


def test_numslots_zero_with_no_previous_value_is_unchanged() -> None:
    ext = _Ext()

    cb.handle_numslots_change(ext, 0, None)

    assert ext.reconfigured == []
    assert ext.logs == []


def test_numslots_change_is_ignored_in_receiver_mode() -> None:
    ext = _Ext(mode="Receiver")

    cb.handle_numslots_change(ext, 4, 3)

    assert ext.reconfigured == []
    assert ("Numslots change ignored in Receiver mode (controlled by sender)", True) in ext.logs


def test_numslots_change_is_ignored_while_active() -> None:
    ext = _Ext(active=True)

    cb.handle_numslots_change(ext, 4, 3)

    assert ext.reconfigured == []
    assert ("Numslots change ignored while Active (deactivate first)", True) in ext.logs


# ---------------------------------------------------------------------------
# handle_mode_change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("new_mode", "prev_mode", "expected_top"),
    [("Sender", "Receiver", "ExportBuffer"), ("Receiver", "Sender", "ImportBuffer")],
    ids=["to-sender", "to-receiver"],
)
def test_mode_change_switches_extension_and_retargets_bg_select(
    monkeypatch: pytest.MonkeyPatch, new_mode: str, prev_mode: str, expected_top: str
) -> None:
    bg_select = types.SimpleNamespace(par=_Par())
    _install_parent(monkeypatch, _Comp(bg=bg_select))
    ext = _Ext()

    cb.handle_mode_change(ext, new_mode, prev_mode)

    assert ext.switched_to == [new_mode]
    assert bg_select.par.top == expected_top
    assert (f"Mode switching: {prev_mode} -> {new_mode}", True) in ext.logs
    assert (f"Mode switch complete: now in {new_mode} mode", True) in ext.logs
    assert (f"Updated bg selectTOP -> {expected_top}", False) in ext.logs


def test_mode_change_unchanged_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_parent(monkeypatch, _Comp())
    ext = _Ext()

    cb.handle_mode_change(ext, "Sender", "Sender")

    assert ext.switched_to == []
    assert ext.logs == []


def test_mode_change_with_no_previous_value_still_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_parent(monkeypatch, _Comp())
    ext = _Ext()

    cb.handle_mode_change(ext, "Receiver", None)

    assert ext.switched_to == ["Receiver"]
    assert ("Mode switching:  -> Receiver", True) in ext.logs


def test_mode_change_without_bg_select_still_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_parent(monkeypatch, _Comp(bg=None))
    ext = _Ext()

    cb.handle_mode_change(ext, "Receiver", "Sender")

    assert ext.switched_to == ["Receiver"]
    assert ("Mode switch complete: now in Receiver mode", True) in ext.logs
    assert not any("bg selectTOP" in m for m in ext.messages())


def test_mode_change_unknown_mode_leaves_bg_select_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    bg_select = types.SimpleNamespace(par=_Par())
    _install_parent(monkeypatch, _Comp(bg=bg_select))
    ext = _Ext()

    cb.handle_mode_change(ext, "Other", "Sender")

    assert ext.switched_to == ["Other"]
    assert bg_select.par.top is None


def test_mode_change_bg_update_failure_is_logged_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Broken:
        @property
        def par(self):
            raise AttributeError("no par")

    _install_parent(monkeypatch, _Comp(bg=_Broken()))
    ext = _Ext()

    cb.handle_mode_change(ext, "Sender", "Receiver")

    assert ext.switched_to == ["Sender"]
    assert ("Could not update bg selectTOP: no par", False) in ext.logs
    assert ("Mode switch complete: now in Sender mode", True) in ext.logs


@pytest.mark.parametrize(
    "error", [AttributeError("no such mode"), RuntimeError("switch failed")], ids=["attribute-error", "runtime-error"]
)
def test_mode_change_switch_failure_is_logged_as_error_without_completion(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    _install_parent(monkeypatch, _Comp())
    ext = _Ext(switch_error=error)

    cb.handle_mode_change(ext, "Receiver", "Sender")

    assert (f"ERROR switching mode: {error}", True) in ext.logs
    assert not any("Mode switch complete" in m for m in ext.messages())


# ---------------------------------------------------------------------------
# remaining handlers
# ---------------------------------------------------------------------------


def test_active_on_logs_for_current_mode_and_locks_numslots(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = _install_parent(monkeypatch, _Comp())
    ext = _Ext(mode="Sender")

    cb.handle_active_change(ext, 1, 0)

    assert ("Component activated", True) in ext.logs
    assert ("Sender mode ready - will initialize on first frame export", False) in ext.logs
    assert ext.cleanups == 0
    assert comp.par.Numslots.enable is False


def test_active_on_in_receiver_mode_keeps_numslots_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    comp = _install_parent(monkeypatch, _Comp())
    ext = _Ext(mode="Receiver")

    cb.handle_active_change(ext, True, False)

    assert ("Receiver mode activated - will attempt connection", False) in ext.logs
    assert comp.par.Numslots.enable is False


@pytest.mark.parametrize(
    ("mode", "numslots_enabled"), [("Sender", True), ("Receiver", False)], ids=["sender", "receiver"]
)
def test_active_off_cleans_up_and_enables_numslots_only_for_sender(
    monkeypatch: pytest.MonkeyPatch, mode: str, numslots_enabled: bool
) -> None:
    comp = _install_parent(monkeypatch, _Comp())
    ext = _Ext(mode=mode)

    cb.handle_active_change(ext, 0, 1)

    assert ("Component deactivated - cleaning up", True) in ext.logs
    assert ext.cleanups == 1
    assert ext.status_cleared == 1
    assert comp.par.Numslots.enable is numslots_enabled


def test_active_change_tolerates_missing_numslots_par(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_parent(monkeypatch, types.SimpleNamespace(par=types.SimpleNamespace()))
    ext = _Ext()

    cb.handle_active_change(ext, 0, 1)

    assert ext.cleanups == 1


def test_ipcmemname_change_reconfigures_shm_name() -> None:
    ext = _Ext()

    cb.handle_ipcmemname_change(ext, "cuda_b", "cuda_a")

    assert ext.reconfigured == [("shm_name", "cuda_b")]


@pytest.mark.parametrize(
    ("new_value", "prev"), [("cuda_a", "cuda_a"), ("", None)], ids=["same-name", "empty-from-unset"]
)
def test_ipcmemname_unchanged_is_a_noop(new_value: str, prev: object) -> None:
    ext = _Ext()

    cb.handle_ipcmemname_change(ext, new_value, prev)

    assert ext.reconfigured == []


def test_ipcmemname_from_unset_reconfigures() -> None:
    ext = _Ext()

    cb.handle_ipcmemname_change(ext, "cuda_a", None)

    assert ext.reconfigured == [("shm_name", "cuda_a")]


@pytest.mark.parametrize(
    ("value", "message"), [(1, "Debug logging ENABLED"), (0, "Debug logging DISABLED")], ids=["enabled", "disabled"]
)
def test_debug_change_sets_verbosity_flag(value: int, message: str) -> None:
    ext = _Ext()

    cb.handle_debug_change(ext, value, 1 - value)

    assert ext.verbose_performance is bool(value)
    assert (message, True) in ext.logs


@pytest.mark.parametrize(("value", "message"), [(1, "hidden"), (0, "visible")], ids=["hidden", "visible"])
def test_hidebuiltin_change_toggles_custom_only_view(monkeypatch: pytest.MonkeyPatch, value: int, message: str) -> None:
    comp = _install_parent(monkeypatch, _Comp())
    ext = _Ext()

    cb.handle_hidebuiltin_change(ext, value, 1 - value)

    assert comp.showCustomOnly is bool(value)
    assert (f"Built-in parameters {message}", True) in ext.logs
