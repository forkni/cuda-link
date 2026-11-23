"""
CUDAIPCExtension.on_par_change and resync_pars: what each parameter change does.

test_td_callbacks_dispatch.py pins that the Parameter Execute DAT forwards a change here, and
test_callbacks_template_dispatch.py that the Execute DAT runs the resync once. These pin the
decisions, through the extension's own interface with a FakeTDHost: no `parent()` stub.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import sys

import pytest
from CUDAIPCExtension import PARS, CUDAIPCExtension, ParSpec
from fakes import FakeTDHost

_SHM = "cuda_a"


def _host(mode: str = "Sender", active: bool = False, **params: object) -> FakeTDHost:
    """A saved .tox: built-in parameters hidden, 3 slots."""
    base: dict[str, object] = {
        "Mode": mode,
        "Ipcmemname": _SHM,
        "Numslots": 3,
        "Active": active,
        "Debug": False,
        "Hidebuiltin": True,
    }
    return FakeTDHost(params={**base, **params})


class _Ext(CUDAIPCExtension):
    """The real change handlers, with the heavy effects recorded and not run."""

    def __init__(self, *, mode: str = "Sender", active: bool = False, switch_error: Exception | None = None) -> None:
        self.logs: list[tuple[str, bool]] = []
        self.reconfigured: list[tuple[str, object]] = []
        self.switched_to: list[str] = []
        self.cleanups = 0
        self.status_cleared = 0
        self._switch_error = switch_error
        self.host = _host(mode, active)
        super().__init__(None, host=self.host)
        self.logs.clear()
        self.host.enable_writes.clear()
        self.host.custom_only_calls.clear()

    def _log(self, msg: str, force: bool = False) -> None:
        self.logs.append((msg, force))

    def reconfigure_and_reinit(self, field_name: str, new_value: object) -> None:
        self.reconfigured.append((field_name, new_value))

    def switch_mode(self, new_mode: str) -> None:
        if self._switch_error is not None:
            raise self._switch_error
        self.switched_to.append(new_mode)

    def cleanup(self) -> None:
        self.cleanups += 1

    def clear_status(self) -> None:
        self.status_cleared += 1

    def messages(self) -> list[str]:
        return [m for m, _ in self.logs]


def _change(ext: _Ext, name: str, value: object, prev: object) -> None:
    """on_par_change, with its own 'Parameter ... changed' line taken out of ext.logs."""
    at = len(ext.logs)
    ext.on_par_change(name, value, prev)
    assert ext.logs[at] == (f"Parameter '{name}' changed: {prev} -> {value}", False)
    del ext.logs[at]


def _last_enable(host: FakeTDHost, name: str) -> bool | None:
    states = [enabled for par, enabled in host.enable_writes if par == name]
    return states[-1] if states else None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_pars_table_names_the_monitored_pars_mode_first() -> None:
    """Mode first: resync_pars walks this order, and the later pars need the new engine."""
    assert [spec.name for spec in PARS] == [
        "Mode",
        "Active",
        "Ipcmemname",
        "Numslots",
        "Cudadevice",
        "Streamid",
        "Framewaitms",
        "Debug",
        "Hidebuiltin",
    ]


def test_live_pars_keep_a_copy_only_for_an_action_to_replay() -> None:
    """Active: resync_pars must run the cleanup of a deactivation it missed."""
    live = [spec for spec in PARS if spec.policy == "live"]
    assert [spec.name for spec in live] == ["Active", "Streamid", "Framewaitms"]
    assert [spec.name for spec in live if spec.state is not None] == ["Active"]


def test_reinit_pars_have_state_and_apply() -> None:
    reinit = [spec for spec in PARS if spec.policy == "reinit"]
    assert all(spec.state is not None and spec.apply is not None for spec in reinit)


@pytest.mark.parametrize(
    ("policy", "state", "apply"),
    [("reinit", None, None), ("reinit", lambda ext: 0, None), ("live", lambda ext: 0, None)],
    ids=["reinit-without-state", "reinit-without-apply", "live-with-state"],
)
def test_parspec_rejects_a_row_that_does_not_fit_its_policy(policy: str, state: object, apply: object) -> None:
    with pytest.raises(ValueError, match="Somepar"):
        ParSpec("Somepar", policy, int, state, apply)  # type: ignore[arg-type]


def test_every_par_the_package_reads_has_a_row() -> None:
    """A par read through param_value() with no row has no declared policy (the old Cudadevice)."""
    source_dir = pathlib.Path(sys.modules[CUDAIPCExtension.__module__].__file__).parent
    read = set()
    for name in ("CUDAIPCExtension.py", "TDSender.py", "TDReceiver.py"):
        read |= set(re.findall(r'param_value\("(\w+)"\)', (source_dir / name).read_text(encoding="utf-8")))

    assert read == {spec.name for spec in PARS} - {"Active"}  # Active has its own reader, is_active()


@pytest.mark.parametrize("name", ["Streamid", "Framewaitms"], ids=["streamid", "framewaitms"])
def test_a_live_par_without_apply_is_logged_and_ignored(name: str) -> None:
    ext = _Ext()

    ext.on_par_change(name, 5, 0)

    assert (ext.reconfigured, ext.switched_to, ext.cleanups) == ([], [], 0)


def test_cudadevice_change_makes_a_new_engine_on_the_new_device() -> None:
    ext = _Ext()
    engine = ext._engine

    _change(ext, "Cudadevice", 1, 0)

    assert (ext._device, ext.cleanups) == (1, 1)
    assert ext._engine is not engine
    assert ("device changed - reinitializing", True) in ext.logs


def test_cudadevice_unchanged_is_a_noop() -> None:
    ext = _Ext()

    _change(ext, "Cudadevice", 0, 0)

    assert (ext.cleanups, ext.logs) == (0, [])


def test_cudadevice_below_zero_is_rejected() -> None:
    ext = _Ext()

    _change(ext, "Cudadevice", -1, 0)

    assert (ext._device, ext.cleanups) == (0, 0)
    assert ("WARNING: Cudadevice=-1 is not a device index; change rejected", True) in ext.logs


def test_unknown_par_is_logged_and_ignored() -> None:
    ext = _Ext()

    ext.on_par_change("SomeOtherParam", "value", "prev")

    assert ext.logs == [("Parameter 'SomeOtherParam' changed: prev -> value", False)]
    assert (ext.reconfigured, ext.switched_to, ext.cleanups) == ([], [], 0)
    assert ext.host.enable_writes == []


# ---------------------------------------------------------------------------
# Numslots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("new_value", [2, 3, 5, "4"], ids=["min", "mid", "recommended-max", "string-from-par"])
def test_numslots_in_recommended_range_reconfigures_without_warning(new_value: object) -> None:
    ext = _Ext()

    _change(ext, "Numslots", new_value, 3 if str(new_value) != "3" else 2)

    assert ext.reconfigured == [("num_slots", int(new_value))]  # type: ignore[call-overload]
    assert not any("WARNING" in m for m in ext.messages())


@pytest.mark.parametrize("new_value", [6, 10], ids=["just-above-recommended", "supported-max"])
def test_numslots_above_recommended_but_supported_warns_and_reconfigures(new_value: int) -> None:
    ext = _Ext()

    _change(ext, "Numslots", new_value, 3)

    assert ext.reconfigured == [("num_slots", new_value)]
    assert (f"WARNING: Numslots={new_value} outside recommended range (2-5)", True) in ext.logs


@pytest.mark.parametrize("new_value", [0, 1, 11, 99], ids=["zero", "below-min", "above-max", "far-above-max"])
def test_numslots_outside_supported_range_is_rejected(new_value: int) -> None:
    ext = _Ext()

    _change(ext, "Numslots", new_value, 3)

    assert ext.reconfigured == []
    assert (f"WARNING: Numslots={new_value} outside supported range (2-10); change rejected", True) in ext.logs


def test_numslots_unchanged_is_a_noop() -> None:
    ext = _Ext()

    _change(ext, "Numslots", "4", 4)

    assert ext.reconfigured == []
    assert ext.logs == []


def test_numslots_with_no_previous_value_treats_it_as_zero() -> None:
    ext = _Ext()

    _change(ext, "Numslots", 3, None)

    assert ext.reconfigured == [("num_slots", 3)]


def test_numslots_zero_with_no_previous_value_is_unchanged() -> None:
    ext = _Ext()

    _change(ext, "Numslots", 0, None)

    assert ext.reconfigured == []
    assert ext.logs == []


def test_numslots_change_is_ignored_in_receiver_mode() -> None:
    ext = _Ext(mode="Receiver")

    _change(ext, "Numslots", 4, 3)

    assert ext.reconfigured == []
    assert ("Numslots change ignored in Receiver mode (controlled by sender)", True) in ext.logs


def test_numslots_change_is_ignored_while_active() -> None:
    ext = _Ext(active=True)

    _change(ext, "Numslots", 4, 3)

    assert ext.reconfigured == []
    assert ("Numslots change ignored while Active (deactivate first)", True) in ext.logs


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("new_mode", "prev_mode", "expected_top"),
    [("Sender", "Receiver", "ExportBuffer"), ("Receiver", "Sender", "ImportBuffer")],
    ids=["to-sender", "to-receiver"],
)
def test_mode_change_switches_extension_and_retargets_bg_select(
    new_mode: str, prev_mode: str, expected_top: str
) -> None:
    ext = _Ext()

    _change(ext, "Mode", new_mode, prev_mode)

    assert ext.switched_to == [new_mode]
    assert ext.host.display_buffers == [expected_top]
    assert (f"Mode switching: {prev_mode} -> {new_mode}", True) in ext.logs
    assert (f"Mode switch complete: now in {new_mode} mode", True) in ext.logs
    assert (f"Updated bg selectTOP -> {expected_top}", False) in ext.logs


def test_mode_change_unchanged_is_a_noop() -> None:
    ext = _Ext()

    _change(ext, "Mode", "Sender", "Sender")

    assert ext.switched_to == []
    assert ext.logs == []


def test_mode_change_with_no_previous_value_still_switches() -> None:
    ext = _Ext()

    _change(ext, "Mode", "Receiver", None)

    assert ext.switched_to == ["Receiver"]
    assert ("Mode switching:  -> Receiver", True) in ext.logs


def test_mode_change_without_bg_select_still_completes() -> None:
    ext = _Ext()
    ext.host.has_display_buffer = False

    _change(ext, "Mode", "Receiver", "Sender")

    assert ext.switched_to == ["Receiver"]
    assert ("Mode switch complete: now in Receiver mode", True) in ext.logs
    assert not any("bg selectTOP" in m for m in ext.messages())


def test_mode_change_unknown_mode_leaves_bg_select_alone() -> None:
    ext = _Ext()

    _change(ext, "Mode", "Other", "Sender")

    assert ext.switched_to == ["Other"]
    assert ext.host.display_buffers == []


def test_mode_change_bg_update_failure_is_logged_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _Ext()

    def _broken(top_name: str) -> bool:
        raise AttributeError("no par")

    monkeypatch.setattr(ext.host, "set_display_buffer", _broken)

    _change(ext, "Mode", "Sender", "Receiver")

    assert ext.switched_to == ["Sender"]
    assert ("Could not update bg selectTOP: no par", False) in ext.logs
    assert ("Mode switch complete: now in Sender mode", True) in ext.logs


@pytest.mark.parametrize(
    "error", [AttributeError("no such mode"), RuntimeError("switch failed")], ids=["attribute-error", "runtime-error"]
)
def test_mode_change_switch_failure_is_logged_as_error_without_completion(error: Exception) -> None:
    ext = _Ext(switch_error=error)

    _change(ext, "Mode", "Receiver", "Sender")

    assert (f"ERROR switching mode: {error}", True) in ext.logs
    assert not any("Mode switch complete" in m for m in ext.messages())
    assert ext.host.display_buffers == []


# ---------------------------------------------------------------------------
# Active
# ---------------------------------------------------------------------------


def test_active_on_logs_for_sender_and_locks_numslots() -> None:
    ext = _Ext(mode="Sender", active=True)

    _change(ext, "Active", 1, 0)

    assert ("Component activated", True) in ext.logs
    assert ("Sender mode ready - will initialize on first frame export", False) in ext.logs
    assert ext.cleanups == 0
    assert _last_enable(ext.host, "Numslots") is False


def test_active_on_in_receiver_mode_keeps_numslots_locked() -> None:
    ext = _Ext(mode="Receiver", active=True)

    _change(ext, "Active", True, False)

    assert ("Receiver mode activated - will attempt connection", False) in ext.logs
    assert _last_enable(ext.host, "Numslots") is False


@pytest.mark.parametrize(
    ("mode", "numslots_enabled"), [("Sender", True), ("Receiver", False)], ids=["sender", "receiver"]
)
def test_active_off_cleans_up_and_unlocks_numslots_only_for_sender(mode: str, numslots_enabled: bool) -> None:
    ext = _Ext(mode=mode, active=False)

    _change(ext, "Active", 0, 1)

    assert ("Component deactivated - cleaning up", True) in ext.logs
    assert ext.cleanups == 1
    assert ext.status_cleared == 1
    assert _last_enable(ext.host, "Numslots") is numslots_enabled


# ---------------------------------------------------------------------------
# Ipcmemname, Debug, Hidebuiltin
# ---------------------------------------------------------------------------


def test_ipcmemname_change_reconfigures_shm_name() -> None:
    ext = _Ext()

    _change(ext, "Ipcmemname", "cuda_b", "cuda_a")

    assert ext.reconfigured == [("shm_name", "cuda_b")]


@pytest.mark.parametrize(
    ("new_value", "prev"), [("cuda_a", "cuda_a"), ("", None)], ids=["same-name", "empty-from-unset"]
)
def test_ipcmemname_unchanged_is_a_noop(new_value: str, prev: object) -> None:
    ext = _Ext()

    _change(ext, "Ipcmemname", new_value, prev)

    assert ext.reconfigured == []


def test_ipcmemname_from_unset_reconfigures() -> None:
    ext = _Ext()

    _change(ext, "Ipcmemname", "cuda_a", None)

    assert ext.reconfigured == [("shm_name", "cuda_a")]


@pytest.mark.parametrize(
    ("value", "message"), [(1, "Debug logging ENABLED"), (0, "Debug logging DISABLED")], ids=["enabled", "disabled"]
)
def test_debug_change_sets_verbosity_flag(value: int, message: str) -> None:
    ext = _Ext()

    _change(ext, "Debug", value, 1 - value)

    assert ext.verbose_performance is bool(value)
    assert (message, True) in ext.logs


@pytest.mark.parametrize(("value", "message"), [(1, "hidden"), (0, "visible")], ids=["hidden", "visible"])
def test_hidebuiltin_change_toggles_custom_only_view(value: int, message: str) -> None:
    ext = _Ext()

    _change(ext, "Hidebuiltin", value, 1 - value)

    assert ext.host.custom_only_calls == [bool(value)]
    assert (f"Built-in parameters {message}", True) in ext.logs


# ---------------------------------------------------------------------------
# resync_pars: writes made before the Parameter Execute DAT first cooked
# ---------------------------------------------------------------------------


class _Live(CUDAIPCExtension):
    """The real extension end to end; records each change that resync_pars routes."""

    def __init__(self, host: FakeTDHost) -> None:
        self.changes: list[tuple[str, object]] = []
        self.logs: list[str] = []
        self.host = host
        super().__init__(None, host=host)

    def _log(self, msg: str, force: bool = False) -> None:
        self.logs.append(msg)

    def on_par_change(self, name: str, value: object, prev: object = None) -> None:
        self.changes.append((name, prev))
        super().on_par_change(name, value, prev)


def _loaded(active: bool = True, **written_after_load: object) -> _Live:
    """A Sender .tox loaded with its saved values, then pars written in the same call."""
    host = _host(active=active)
    ext = _Live(host)
    host._params.update(written_after_load)
    return ext


def test_resync_routes_drift_through_on_par_change_mode_first() -> None:
    """The 2026-11-21 case: Mode and Ipcmemname set in the same call as loadTox()."""
    ext = _loaded(Mode="Receiver", Ipcmemname="probe")

    ext.resync_pars()

    assert ext.changes == [("Mode", "Sender"), ("Ipcmemname", _SHM)]
    assert (ext.mode, ext.shm_name) == ("Receiver", "probe")
    assert ext.host.display_buffers == ["ImportBuffer"]
    assert _last_enable(ext.host, "Numslots") is False
    assert _last_enable(ext.host, "Framewaitms") is True


def test_resync_asks_for_more_calls_until_the_frame_after_the_first() -> None:
    """TD runs Frame Start inside loadTox(), before the script writes its pars."""
    host = _host(active=True)
    ext = _Live(host)
    host.abs_frame_value = 100
    assert ext.resync_pars() is True  # inside loadTox(): nothing written yet
    assert ext.changes == []

    host._params.update(Mode="Receiver", Ipcmemname="probe")
    assert ext.resync_pars() is True  # Frame End of the same frame
    assert (ext.mode, ext.shm_name) == ("Receiver", "probe")

    host.abs_frame_value = 101
    assert ext.resync_pars() is False  # the first callback of the next frame is the last


def test_resync_applies_writes_first_seen_on_the_next_frame() -> None:
    host = _host(active=True)
    ext = _Live(host)
    host.abs_frame_value = 100
    ext.resync_pars()
    host._params.update(Ipcmemname="probe")
    host.abs_frame_value = 101

    assert ext.resync_pars() is False
    assert ext.shm_name == "probe"


def test_resync_with_an_unknown_frame_asks_for_one_call_only() -> None:
    ext = _loaded(Ipcmemname="probe")  # FakeTDHost.abs_frame() gives None by default

    assert ext.resync_pars() is False
    assert ext.shm_name == "probe"


def test_resync_applies_a_cudadevice_written_after_load() -> None:
    """Before, Cudadevice was read once in __init__ and a later write was lost."""
    ext = _loaded(Cudadevice=1)

    ext.resync_pars()

    assert ext.changes == [("Cudadevice", 0)]
    assert ext._device == 1


def test_resync_without_drift_changes_nothing() -> None:
    ext = _loaded()

    ext.resync_pars()

    assert ext.changes == []


def test_resync_numslots_drift_on_an_inactive_sender_applies() -> None:
    """Active off before the first frame: Numslots applies and the parameter is editable."""
    ext = _loaded(active=False, Numslots=5)

    ext.resync_pars()

    assert ext.changes == [("Numslots", 3)]
    assert ext.num_slots == 5
    assert _last_enable(ext.host, "Numslots") is True


def test_resync_numslots_drift_on_an_active_sender_follows_the_active_guard() -> None:
    """Same result as a normal write: Numslots does not change while Active is on."""
    ext = _loaded(Numslots=5)

    ext.resync_pars()

    assert ext.num_slots == 3
    assert any("ignored while Active" in m for m in ext.logs)


def test_resync_hidebuiltin_drift() -> None:
    ext = _loaded(Hidebuiltin=False)

    ext.resync_pars()

    assert ext.changes == [("Hidebuiltin", True)]
    assert ext.host.custom_only_calls[-1] is False


def test_resync_debug_drift() -> None:
    ext = _loaded(Debug=True)

    ext.resync_pars()

    assert ext.changes == [("Debug", False)]
    assert ext.verbose_performance is True


def test_resync_skips_debug_under_export_profile() -> None:
    """CUDALINK_EXPORT_PROFILE forces verbose on; a Debug par that is Off must not undo it."""
    ext = _loaded()
    ext._config = dataclasses.replace(ext._config, export_profile=True)
    ext.verbose_performance = True

    ext.resync_pars()

    assert ext.changes == []
    assert ext.verbose_performance is True


def test_resync_runs_the_cleanup_of_a_missed_deactivation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A .tox saved Active, set inactive in the same call as loadTox(): the engine it
    opened inside loadTox() must be released, and the Status cleared."""
    ext = _loaded(Active=False)
    done: list[str] = []
    monkeypatch.setattr(ext, "cleanup", lambda: done.append("cleanup"))
    monkeypatch.setattr(ext, "clear_status", lambda: done.append("clear_status"))
    ext.host.enable_writes.clear()

    ext.resync_pars()

    assert ext.changes == [("Active", True)]
    assert done == ["cleanup", "clear_status"]
    assert _last_enable(ext.host, "Numslots") is True


def test_resync_routes_a_missed_activation_once() -> None:
    ext = _loaded(active=False, Active=True)

    ext.resync_pars()
    ext.resync_pars()

    assert ext.changes == [("Active", False)]
    assert _last_enable(ext.host, "Numslots") is False


def test_resync_sends_a_rejected_value_once() -> None:
    """Numslots written on an Active Sender is rejected; the next resync calls stay quiet."""
    ext = _loaded(Numslots=4)

    ext.resync_pars()
    ext.resync_pars()
    ext.resync_pars()

    assert ext.changes == [("Numslots", 3)]
    assert ext.logs.count("Numslots change ignored while Active (deactivate first)") == 1


def test_resync_does_not_repeat_a_change_the_parexec_dat_reported() -> None:
    """The Receiver writes the sender's slot count into Numslots when it connects, and the
    Parameter Execute DAT reports that write. The par then differs from num_slots for good."""
    host = _host(mode="Receiver", active=True)
    ext = _Live(host)
    host._params["Numslots"] = 4
    ext.on_par_change("Numslots", 4, 3)
    ext.changes.clear()

    ext.resync_pars()

    assert ext.changes == []


def test_resync_applies_a_par_that_was_absent_at_load() -> None:
    """No kept value to compare with: the value the par has now is the change, also False."""
    host = _host()
    del host._params["Hidebuiltin"]
    ext = _Live(host)
    host._params["Hidebuiltin"] = False

    ext.resync_pars()

    assert ext.changes == [("Hidebuiltin", None)]
    assert ext._custom_only is False


def test_resync_skips_a_par_the_comp_does_not_have() -> None:
    host = _host()
    del host._params["Hidebuiltin"]
    ext = _Live(host)

    ext.resync_pars()

    assert ext.changes == []
    assert host.custom_only_calls == []


def test_resync_one_failing_par_does_not_stop_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _loaded(Mode="Receiver", Debug=True)

    def _fails(new_mode: str) -> None:
        raise ValueError("switch failed")  # not one of the errors the Mode handler logs

    monkeypatch.setattr(ext, "switch_mode", _fails)
    ext.host.enable_writes.clear()

    ext.resync_pars()  # must not raise

    assert ext.changes == [("Mode", "Sender"), ("Debug", False)]
    assert ext.verbose_performance is True
    assert _last_enable(ext.host, "Numslots") is False  # the enable sync still runs
