"""
Characterization tests for callbacks_template.py (Execute DAT: onFrameStart / onFrameEnd
/ onExit) -- previously had zero coverage of any kind.

Like script_top_callbacks.py and parexecute_callbacks.py, this module references the
TD-injected globals `parent()` and `op()`, which don't exist when imported standalone;
tests monkeypatch the module's `parent`/`op` names (raising=False, since they aren't
normal module attributes pre-patch).

Two failure modes are exercised for every callback, matching the reported bug:
  - `parent().ext.CUDAIPCExtension` evaluates to None (extension absent) -- the case the
    old `if ext is None` guard already handled.
  - `parent().ext.CUDAIPCExtension` RAISES td.tdAttributeError (extension failed to
    compile) -- the actual reported per-frame crash, which the dead `is None` guard could
    never catch. _extension()'s broad `except Exception` is what fixes this.
"""

from __future__ import annotations

import types
from collections.abc import Callable

import callbacks_template
import pytest


class _TDAttributeError(RuntimeError):
    """Stand-in for td.tdAttributeError (NOT an AttributeError subclass in real TD)."""


class _RaisingExt:
    """`.CUDAIPCExtension` raises instead of evaluating to None -- see module docstring."""

    @property
    def CUDAIPCExtension(self) -> object:
        raise _TDAttributeError("'td.Ext' object has no attribute 'CUDAIPCExtension'")


def _patch_parent(monkeypatch: pytest.MonkeyPatch, ext: object | None) -> None:
    monkeypatch.setattr(
        callbacks_template,
        "parent",
        lambda: types.SimpleNamespace(ext=types.SimpleNamespace(CUDAIPCExtension=ext)),
        raising=False,
    )


def _patch_parent_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        callbacks_template,
        "parent",
        lambda: types.SimpleNamespace(ext=_RaisingExt()),
        raising=False,
    )


# The two ways `parent().ext.CUDAIPCExtension` can be unavailable -- see module docstring.
_MISSING_EXTENSION_PATCHES = [
    pytest.param(lambda mp: _patch_parent(mp, None), id="ext-absent"),
    pytest.param(_patch_parent_raising, id="ext-raises"),
]


class _StubExt:
    def __init__(self, mode: str = "Sender", has_new_frame: bool = True) -> None:
        self.mode = mode
        self._has_new_frame = has_new_frame
        self.deferred_cleanup_calls = 0
        self.export_frame_calls = 0
        self.cleanup_calls = 0
        self.update_receiver_format_calls: list[object] = []

    def _check_deferred_cleanup(self) -> None:
        self.deferred_cleanup_calls += 1

    def has_new_frame(self) -> bool:
        return self._has_new_frame

    def update_receiver_format(self, import_buffer: object) -> None:
        self.update_receiver_format_calls.append(import_buffer)

    def export_frame(self) -> None:
        self.export_frame_calls += 1

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class _StubImportBuffer:
    def __init__(self) -> None:
        self.cook_calls: list[bool] = []

    def cook(self, force: bool = False) -> None:
        self.cook_calls.append(force)


# ---------------------------------------------------------------------------
# onFrameStart
# ---------------------------------------------------------------------------


def _patch_op_recorder(monkeypatch: pytest.MonkeyPatch, result: object = None) -> list[str]:
    """Patch the TD global `op()` with a recorder; returns the list of looked-up names."""
    lookups: list[str] = []

    def _op(name: str) -> object:
        lookups.append(name)
        return result

    monkeypatch.setattr(callbacks_template, "op", _op, raising=False)
    return lookups


def test_extension_returns_none_when_ext_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_parent(monkeypatch, None)
    assert callbacks_template._extension() is None


def test_extension_swallows_td_attribute_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reported bug: `.CUDAIPCExtension` raised td.tdAttributeError every frame."""
    _patch_parent_raising(monkeypatch)
    assert callbacks_template._extension() is None


def test_extension_returns_present_ext(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt()
    _patch_parent(monkeypatch, ext)
    assert callbacks_template._extension() is ext


def test_onframestart_ext_none_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_parent(monkeypatch, None)
    lookups = _patch_op_recorder(monkeypatch)
    callbacks_template.onFrameStart(1)
    assert lookups == [], "must return before resolving ImportBuffer"


def test_onframestart_ext_raises_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduces the reported bug: this fired every frame in production."""
    _patch_parent_raising(monkeypatch)
    lookups = _patch_op_recorder(monkeypatch)
    callbacks_template.onFrameStart(1)
    assert lookups == [], "must return before resolving ImportBuffer"


def test_onframestart_sender_runs_deferred_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt(mode="Sender")
    _patch_parent(monkeypatch, ext)
    callbacks_template.onFrameStart(1)
    assert ext.deferred_cleanup_calls == 1


def test_onframestart_receiver_no_import_buffer_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt(mode="Receiver")
    _patch_parent(monkeypatch, ext)
    lookups = _patch_op_recorder(monkeypatch, result=None)
    callbacks_template.onFrameStart(1)
    assert lookups == ["ImportBuffer"]
    assert ext.update_receiver_format_calls == []
    assert ext.deferred_cleanup_calls == 0


def test_onframestart_receiver_skips_cook_without_new_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt(mode="Receiver", has_new_frame=False)
    _patch_parent(monkeypatch, ext)
    import_buffer = _StubImportBuffer()
    monkeypatch.setattr(callbacks_template, "op", lambda name: import_buffer, raising=False)

    callbacks_template.onFrameStart(1)

    assert import_buffer.cook_calls == []
    assert ext.update_receiver_format_calls == []


@pytest.mark.parametrize(
    ("rescued", "expected_cooks"), [(True, [True]), (False, [])], ids=["wait_got_frame", "wait_timed_out"]
)
def test_onframestart_receiver_frame_wait_decides_the_cook(
    monkeypatch: pytest.MonkeyPatch, rescued: bool, expected_cooks: list[bool]
) -> None:
    """No new frame at the poll: ImportBuffer cooks only when the Framewaitms wait got one."""
    ext = _StubExt(mode="Receiver", has_new_frame=False)
    wait_calls: list[None] = []

    def wait_for_new_frame() -> bool:
        wait_calls.append(None)
        return rescued

    ext.wait_for_new_frame = wait_for_new_frame
    _patch_parent(monkeypatch, ext)
    import_buffer = _StubImportBuffer()
    monkeypatch.setattr(callbacks_template, "op", lambda name: import_buffer, raising=False)

    callbacks_template.onFrameStart(1)

    assert len(wait_calls) == 1
    assert import_buffer.cook_calls == expected_cooks


def test_onframestart_receiver_force_cooks_import_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt(mode="Receiver", has_new_frame=True)
    _patch_parent(monkeypatch, ext)
    import_buffer = _StubImportBuffer()
    monkeypatch.setattr(callbacks_template, "op", lambda name: import_buffer, raising=False)

    callbacks_template.onFrameStart(1)

    assert import_buffer.cook_calls == [True]
    assert ext.update_receiver_format_calls == [import_buffer]


# ---------------------------------------------------------------------------
# onFrameEnd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("patch_parent", _MISSING_EXTENSION_PATCHES)
def test_onframeend_without_extension_returns_normally(
    monkeypatch: pytest.MonkeyPatch, patch_parent: Callable[[pytest.MonkeyPatch], None]
) -> None:
    """ext-raises reproduces the reported bug for the Frame End toggle. Without the
    `ext is None` guard, `ext.mode` raises AttributeError here."""
    patch_parent(monkeypatch)
    assert callbacks_template.onFrameEnd(1) is None


def test_onframeend_sender_exports_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt(mode="Sender")
    _patch_parent(monkeypatch, ext)
    callbacks_template.onFrameEnd(1)
    assert ext.export_frame_calls == 1


def test_onframeend_receiver_does_not_export(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt(mode="Receiver")
    _patch_parent(monkeypatch, ext)
    callbacks_template.onFrameEnd(1)
    assert ext.export_frame_calls == 0


# ---------------------------------------------------------------------------
# onExit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("patch_parent", _MISSING_EXTENSION_PATCHES)
def test_onexit_without_extension_returns_normally(
    monkeypatch: pytest.MonkeyPatch, patch_parent: Callable[[pytest.MonkeyPatch], None]
) -> None:
    """ext-raises reproduces the reported bug on TD shutdown / DAT destruction.
    Without the `ext is not None` guard, `ext.cleanup()` raises AttributeError here."""
    patch_parent(monkeypatch)
    assert callbacks_template.onExit() is None


def test_onexit_calls_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt()
    _patch_parent(monkeypatch, ext)
    callbacks_template.onExit()
    assert ext.cleanup_calls == 1


# ---------------------------------------------------------------------------
# Required-but-unused TD callback stubs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["onStart", "onCreate"], ids=["onStart", "onCreate"])
def test_required_td_callback_stub_exists_and_returns_none(name: str) -> None:
    """TD looks these callbacks up by name; renaming or deleting one breaks the Execute DAT."""
    callback = getattr(callbacks_template, name)
    assert callable(callback)
    assert callback() is None


# ---------------------------------------------------------------------------
# One-time parameter resync (writes made before parexec1 first cooked)
# ---------------------------------------------------------------------------


class _StubPar:
    def __init__(self, name: str, value: object, enable: bool = True) -> None:
        self.name = name
        self.value = value
        self.enable = enable

    def eval(self) -> object:  # TD Par.eval(), not builtins.eval
        return self.value


class _ResyncExt(_StubExt):
    def __init__(self, mode: str = "Sender", shm_name: str = "cudalink_ipc_TD>>TD") -> None:
        super().__init__(mode=mode)
        self.shm_name = shm_name
        self.num_slots = 3
        self.verbose_performance = False
        self._config = types.SimpleNamespace(export_profile=False)


def _make_comp(
    ext: object,
    pars: dict[str, object],
    parexec: object = None,
    parexec_name: str = "parexec1",
    numslots_enable: bool = False,
    show_custom_only: bool = True,
) -> types.SimpleNamespace:
    par_ns = types.SimpleNamespace(**{name: _StubPar(name, v) for name, v in pars.items()})
    par_ns.Numslots.enable = numslots_enable
    ops: dict[str, object] = {parexec_name: parexec} if parexec is not None else {}
    comp = types.SimpleNamespace(
        ext=types.SimpleNamespace(CUDAIPCExtension=ext),
        par=par_ns,
        showCustomOnly=show_custom_only,
        op=ops.get,
    )
    ops["bg"] = types.SimpleNamespace(par=types.SimpleNamespace(top="ExportBuffer"))
    return comp


def _patch_comp(
    monkeypatch: pytest.MonkeyPatch,
    ext: object,
    pars: dict[str, object],
    with_parexec: bool = True,
    **comp_kwargs: object,
) -> tuple[list[tuple[str, object]], types.SimpleNamespace]:
    """Patch `parent()` with a COMP that has pars and a parexec1 recorder.

    Returns the recorded (par name, previous value) calls and the parexec1 module.
    """
    calls: list[tuple[str, object]] = []
    module = types.SimpleNamespace(onValueChange=lambda par, prev: calls.append((par.name, prev)))
    parexec = types.SimpleNamespace(module=module) if with_parexec else None
    comp = _make_comp(ext, pars, parexec, **comp_kwargs)
    monkeypatch.setattr(callbacks_template, "parent", lambda: comp, raising=False)
    return calls, module


# A saved Sender .tox, Active on: Numslots greyed out, built-in parameters hidden.
_IN_SYNC = {
    "Mode": "Sender",
    "Active": True,
    "Ipcmemname": "cudalink_ipc_TD>>TD",
    "Numslots": 3,
    "Debug": False,
    "Hidebuiltin": True,
}


def test_resync_routes_drift_through_parexec_mode_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 2026-11-21 case: Mode and Ipcmemname set in the same call as loadTox()."""
    ext = _ResyncExt()
    calls, _ = _patch_comp(monkeypatch, ext, dict(_IN_SYNC, Mode="Receiver", Ipcmemname="probe"))
    callbacks_template._resync_params_once(ext)
    assert calls == [("Mode", "Sender"), ("Ipcmemname", "cudalink_ipc_TD>>TD")]


def test_resync_without_drift_calls_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _ResyncExt()
    calls, _ = _patch_comp(monkeypatch, ext, dict(_IN_SYNC))
    callbacks_template._resync_params_once(ext)
    assert calls == []


def test_resync_runs_once_per_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _ResyncExt()
    calls, _ = _patch_comp(monkeypatch, ext, dict(_IN_SYNC, Numslots=5))
    callbacks_template._resync_params_once(ext)
    callbacks_template._resync_params_once(ext)
    assert calls == [("Numslots", 3)]


def test_resync_new_extension_instance_checks_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """An extension re-init (a new instance) gets its own check."""
    first, second = _ResyncExt(), _ResyncExt()
    calls, _ = _patch_comp(monkeypatch, first, dict(_IN_SYNC, Debug=True))
    callbacks_template._resync_params_once(first)
    callbacks_template._resync_params_once(second)
    assert calls == [("Debug", False), ("Debug", False)]


def test_resync_skips_debug_under_export_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _ResyncExt()
    ext.verbose_performance = True  # forced on by CUDALINK_EXPORT_PROFILE
    ext._config.export_profile = True
    calls, _ = _patch_comp(monkeypatch, ext, dict(_IN_SYNC))
    callbacks_template._resync_params_once(ext)
    assert calls == []


def test_resync_without_parexec_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _ResyncExt()
    calls, _ = _patch_comp(monkeypatch, ext, dict(_IN_SYNC, Mode="Receiver"), with_parexec=False)
    callbacks_template._resync_params_once(ext)
    assert calls == []


def test_resync_handler_error_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _ResyncExt()
    _, module = _patch_comp(monkeypatch, ext, dict(_IN_SYNC, Mode="Receiver"))

    def _boom(par: object, prev: object) -> None:
        raise RuntimeError("handler failed")

    module.onValueChange = _boom
    assert callbacks_template._resync_params_once(ext) is None


def test_onframestart_resyncs_before_mode_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The resync runs first, so a Mode change takes effect on that same frame."""
    ext = _ResyncExt(mode="Sender")
    calls, module = _patch_comp(monkeypatch, ext, dict(_IN_SYNC, Mode="Receiver"))

    def _switch(par: _StubPar, prev: object) -> None:
        calls.append((par.name, prev))
        ext.mode = str(par.eval())

    module.onValueChange = _switch
    lookups = _patch_op_recorder(monkeypatch, result=None)
    callbacks_template.onFrameStart(1)
    assert calls == [("Mode", "Sender")]
    assert ext.deferred_cleanup_calls == 0
    assert lookups == ["ImportBuffer"]


def test_resync_hidebuiltin_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _ResyncExt()
    calls, _ = _patch_comp(monkeypatch, ext, dict(_IN_SYNC, Hidebuiltin=False))
    callbacks_template._resync_params_once(ext)
    assert calls == [("Hidebuiltin", True)]


def test_resync_active_off_reenables_numslots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Active is read live; only the Numslots.enable side effect of its handler can be stale."""
    ext = _ResyncExt()
    calls, _ = _patch_comp(monkeypatch, ext, dict(_IN_SYNC, Active=False))
    callbacks_template._resync_params_once(ext)
    assert calls == [("Active", True)]


def test_resync_active_matching_numslots_enable_calls_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _ResyncExt()
    calls, _ = _patch_comp(monkeypatch, ext, dict(_IN_SYNC, Active=False), numslots_enable=True)
    callbacks_template._resync_params_once(ext)
    assert calls == []


def test_resync_finds_parexecute_name_from_build_guide(monkeypatch: pytest.MonkeyPatch) -> None:
    """docs/TOX_BUILD_GUIDE.md names the Parameter Execute DAT `parexecute`."""
    ext = _ResyncExt()
    calls, _ = _patch_comp(monkeypatch, ext, dict(_IN_SYNC, Numslots=4), parexec_name="parexecute")
    callbacks_template._resync_params_once(ext)
    assert calls == [("Numslots", 3)]


def test_resync_one_failing_parameter_does_not_stop_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _ResyncExt()
    calls, module = _patch_comp(monkeypatch, ext, dict(_IN_SYNC, Mode="Receiver", Debug=True))

    def _mode_fails(par: _StubPar, prev: object) -> None:
        if par.name == "Mode":
            raise RuntimeError("switch failed")
        calls.append((par.name, prev))

    module.onValueChange = _mode_fails
    callbacks_template._resync_params_once(ext)
    assert calls == [("Debug", False)]


def test_onframeend_resyncs_before_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mid-frame loadTox() reaches Frame End first: a Receiver must not export as a Sender."""
    ext = _ResyncExt(mode="Sender")
    calls, module = _patch_comp(monkeypatch, ext, dict(_IN_SYNC, Mode="Receiver"))

    def _switch(par: _StubPar, prev: object) -> None:
        calls.append((par.name, prev))
        ext.mode = str(par.eval())

    module.onValueChange = _switch
    callbacks_template.onFrameEnd(1)
    assert calls == [("Mode", "Sender")]
    assert ext.export_frame_calls == 0


# ---------------------------------------------------------------------------
# Resync through the real parexecute_callbacks handlers
# ---------------------------------------------------------------------------


class _HandlerExt(_ResyncExt):
    """Enough of CUDAIPCExtension for parexecute_callbacks' handlers."""

    def __init__(self, comp_ref: list[types.SimpleNamespace]) -> None:
        super().__init__()
        self._comp_ref = comp_ref
        self.logs: list[str] = []
        self.reinits: list[tuple[str, object]] = []
        self._host = types.SimpleNamespace(clear_status=lambda: None)

    def _log(self, msg: str, force: bool = False) -> None:
        self.logs.append(msg)

    def is_active(self) -> bool:
        return bool(self._comp_ref[0].par.Active.eval())

    def switch_mode(self, mode: str) -> None:
        self.mode = mode

    def reconfigure_and_reinit(self, field: str, value: object) -> None:
        self.reinits.append((field, value))
        setattr(self, field, value)


def _wire_real_handlers(monkeypatch: pytest.MonkeyPatch, pars: dict[str, object]) -> tuple[_HandlerExt, object]:
    import parexecute_callbacks

    comp_ref: list[types.SimpleNamespace] = []
    ext = _HandlerExt(comp_ref)
    comp = _make_comp(ext, pars, types.SimpleNamespace(module=parexecute_callbacks))
    comp_ref.append(comp)
    monkeypatch.setattr(callbacks_template, "parent", lambda: comp, raising=False)
    monkeypatch.setattr(parexecute_callbacks, "parent", lambda: comp, raising=False)
    return ext, comp


def test_real_handlers_receiver_set_in_loadtox_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 2026-11-21 case end to end: Mode and Ipcmemname reach the extension and bg."""
    ext, comp = _wire_real_handlers(monkeypatch, dict(_IN_SYNC, Mode="Receiver", Ipcmemname="probe"))
    callbacks_template._resync_params_once(ext)
    assert (ext.mode, ext.shm_name) == ("Receiver", "probe")
    assert comp.op("bg").par.top == "ImportBuffer"
    assert comp.par.Numslots.enable is False


def test_real_handlers_inactive_sender_numslots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Active off before the first frame: Numslots applies and the parameter is editable."""
    ext, comp = _wire_real_handlers(monkeypatch, dict(_IN_SYNC, Active=False, Numslots=5))
    callbacks_template._resync_params_once(ext)
    assert ext.reinits == [("num_slots", 5)]
    assert comp.par.Numslots.enable is True


def test_real_handlers_active_sender_numslots_follows_the_active_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same result as a normal write: Numslots does not change while Active is on."""
    ext, _ = _wire_real_handlers(monkeypatch, dict(_IN_SYNC, Numslots=5))
    callbacks_template._resync_params_once(ext)
    assert ext.reinits == []
    assert any("ignored while Active" in m for m in ext.logs)


def test_real_handlers_hidebuiltin(monkeypatch: pytest.MonkeyPatch) -> None:
    ext, comp = _wire_real_handlers(monkeypatch, dict(_IN_SYNC, Hidebuiltin=False))
    callbacks_template._resync_params_once(ext)
    assert comp.showCustomOnly is False
