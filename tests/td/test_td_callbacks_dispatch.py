"""
Characterization tests: callback dispatch in script_top_callbacks.onCook (TDReceiver
consumer, script_top_callbacks.py:25-68) and parexecute_callbacks.onValueChange
(parexecute_callbacks.py:13-49).

Both files are textDAT scripts that only run inside TouchDesigner: they reference the
TD-injected globals `parent()` / `op()`, which do not exist as real Python names when the
module is imported standalone. That absence is itself useful — if onCook's early-return
branch (warning_emitter) were ever removed or reordered to fall through to the
`parent()`-using code below it, calling the undefined global would raise NameError,
failing the test loudly. For every other branch this file monkeypatches the module's
`parent` name (via `monkeypatch.setattr(..., raising=False)`, since it is not a normal
module attribute) to a lightweight stub exposing exactly `.ext.CUDAIPCExtension`.

script_top_callbacks.onCook dispatches by `scriptOp.name`:
  - "warning_emitter" -> re-emit the extension's stored status message via addWarning(),
    then return immediately (never touches the CUDAIPCExtension facade).
  - anything else (the ImportBuffer Script TOP) -> fetch the extension via the global
    parent(), apply any pending resolution/format update, then call ext.import_frame().

parexecute_callbacks.onValueChange forwards each change to
CUDAIPCExtension.on_par_change, which decides what it does
(tests/td/test_extension_par_changes.py).
"""

from __future__ import annotations

import types

import parexecute_callbacks
import pytest
import script_top_callbacks


class _TDAttributeError(RuntimeError):
    """Stand-in for td.tdAttributeError (NOT an AttributeError subclass in real TD)."""


class _RaisingExt:
    """Simulates TD's actual failure mode: `ext.ext.CUDAIPCExtension` RAISES rather than
    evaluating to None, e.g. when the extension failed to compile (td.tdAttributeError).

    getattr(obj, name, default) only swallows AttributeError, so a bare
    `getattr(parent().ext, "CUDAIPCExtension", None)` would NOT save a caller here.
    These tests prove _extension()'s broad `except Exception` -- not getattr's default
    -- is what actually fixes the reported per-frame crash."""

    @property
    def CUDAIPCExtension(self) -> object:
        raise _TDAttributeError("'td.Ext' object has no attribute 'CUDAIPCExtension'")


# ---------------------------------------------------------------------------
# script_top_callbacks.onCook
# ---------------------------------------------------------------------------


class _StubOwnerComp:
    """Stand-in for scriptOp.parent() (the owning COMP), used only by the
    warning_emitter branch to fetch the stored status message."""

    def __init__(self, status_msg: str | None = None) -> None:
        self._status_msg = status_msg

    def fetch(self, key: str, default: object = None) -> object:
        return self._status_msg if key == "cuda_link_status_msg" else default


class _StubScriptOp:
    def __init__(self, name: str, owner: _StubOwnerComp | None = None, par: object | None = None) -> None:
        self.name = name
        self._owner = owner or _StubOwnerComp()
        self.warnings: list[str] = []
        self.par = par if par is not None else types.SimpleNamespace()

    def parent(self) -> _StubOwnerComp:
        return self._owner

    def addWarning(self, msg: str) -> None:
        self.warnings.append(msg)


class _StubExt:
    def __init__(
        self,
        pending_resolution: tuple[int, int] | None = None,
        pending_format: str | None = None,
    ) -> None:
        self._pending_resolution = pending_resolution
        self._pending_format = pending_format
        self.log_calls: list[str] = []
        self.import_frame_calls: list[object] = []
        self.mono_alpha_fix_calls: list[tuple[object, str]] = []

    def sync_mono_alpha_fix(self, script_op: object, fmt: str) -> None:
        self.mono_alpha_fix_calls.append((script_op, fmt))

    def consume_pending_resolution(self) -> tuple[int, int] | None:
        return self._pending_resolution

    def consume_pending_format(self) -> str | None:
        return self._pending_format

    def import_frame(self, script_op: object) -> None:
        self.import_frame_calls.append(script_op)

    def _log(self, msg: str, force: bool = False) -> None:  # noqa: ARG002
        self.log_calls.append(msg)


def _patch_global_parent(monkeypatch: pytest.MonkeyPatch, ext: object | None) -> None:
    monkeypatch.setattr(
        script_top_callbacks,
        "parent",
        lambda: types.SimpleNamespace(ext=types.SimpleNamespace(CUDAIPCExtension=ext)),
        raising=False,
    )


def test_warning_emitter_reemits_stored_message() -> None:
    """warning_emitter re-emits the stored status message and returns before ever
    touching the global parent()/CUDAIPCExtension facade."""
    scriptOp = _StubScriptOp("warning_emitter", owner=_StubOwnerComp("Unsupported pixel format"))
    script_top_callbacks.onCook(scriptOp)  # 'parent' intentionally left unpatched -- see module docstring
    assert scriptOp.warnings == ["Unsupported pixel format"]


def test_warning_emitter_no_message_emits_nothing() -> None:
    scriptOp = _StubScriptOp("warning_emitter", owner=_StubOwnerComp(None))
    script_top_callbacks.onCook(scriptOp)
    assert scriptOp.warnings == []


def test_import_buffer_ext_none_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_global_parent(monkeypatch, None)
    scriptOp = _StubScriptOp("ImportBuffer")
    script_top_callbacks.onCook(scriptOp)  # must not raise
    assert scriptOp.warnings == []


def test_import_buffer_ext_raises_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduces the reported bug directly: parent().ext.CUDAIPCExtension raises
    td.tdAttributeError (uncompiled extension) instead of returning None. Before the
    _extension() fix, this propagated out of onCook on every single cook."""
    monkeypatch.setattr(
        script_top_callbacks,
        "parent",
        lambda: types.SimpleNamespace(ext=_RaisingExt()),
        raising=False,
    )
    scriptOp = _StubScriptOp("ImportBuffer")
    script_top_callbacks.onCook(scriptOp)  # must not raise
    assert scriptOp.warnings == []


def test_import_buffer_dispatches_to_import_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt()
    _patch_global_parent(monkeypatch, ext)
    scriptOp = _StubScriptOp("ImportBuffer")

    script_top_callbacks.onCook(scriptOp)

    assert ext.import_frame_calls == [scriptOp]


def test_pending_resolution_is_applied_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt(pending_resolution=(128, 64))
    _patch_global_parent(monkeypatch, ext)
    scriptOp = _StubScriptOp("ImportBuffer")

    script_top_callbacks.onCook(scriptOp)

    assert scriptOp.par.outputresolution == 9  # Custom Resolution
    assert scriptOp.par.resolutionw == 128
    assert scriptOp.par.resolutionh == 64
    assert any("Set ImportBuffer resolution to 128x64" in m for m in ext.log_calls)
    assert ext.import_frame_calls == [scriptOp]


def test_pending_format_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt(pending_format="rgba16fixed")
    _patch_global_parent(monkeypatch, ext)
    scriptOp = _StubScriptOp("ImportBuffer")

    script_top_callbacks.onCook(scriptOp)

    assert scriptOp.par.format == "rgba16fixed"
    assert ext.mono_alpha_fix_calls == [(scriptOp, "rgba16fixed")]
    assert any("Set ImportBuffer pixel format to 'rgba16fixed'" in m for m in ext.log_calls)


def test_pending_format_with_an_extension_without_the_fix_method(monkeypatch: pytest.MonkeyPatch) -> None:
    """An older extension has no sync_mono_alpha_fix: the format is set and no failure is logged."""

    class _OldExt(_StubExt):
        sync_mono_alpha_fix = None  # type: ignore[assignment]

    ext = _OldExt(pending_format="rgba16fixed")
    _patch_global_parent(monkeypatch, ext)
    scriptOp = _StubScriptOp("ImportBuffer")

    script_top_callbacks.onCook(scriptOp)

    assert scriptOp.par.format == "rgba16fixed"
    assert not any("Could not set ImportBuffer pixel format" in m for m in ext.log_calls)


class _RaisingPar:
    """par-like object whose attribute writes raise, to exercise onCook's except branches."""

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"cannot set {name}")


def test_resolution_apply_failure_is_caught_and_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt(pending_resolution=(64, 64))
    _patch_global_parent(monkeypatch, ext)
    scriptOp = _StubScriptOp("ImportBuffer", par=_RaisingPar())

    script_top_callbacks.onCook(scriptOp)  # must not propagate the AttributeError

    assert any("Could not set ImportBuffer resolution" in m for m in ext.log_calls)
    assert ext.import_frame_calls == [scriptOp], "import_frame must still run after a resolution-apply failure."


def test_format_apply_failure_is_caught_and_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt(pending_format="rgba8fixed")
    _patch_global_parent(monkeypatch, ext)
    scriptOp = _StubScriptOp("ImportBuffer", par=_RaisingPar())

    script_top_callbacks.onCook(scriptOp)  # must not propagate the AttributeError

    assert any("Could not set ImportBuffer pixel format in onCook" in m for m in ext.log_calls)


# ---------------------------------------------------------------------------
# parexecute_callbacks.onValueChange
# ---------------------------------------------------------------------------


class _StubExecExt:
    def __init__(self) -> None:
        self.changes: list[tuple[str, object, object]] = []

    def on_par_change(self, name: str, value: object, prev: object = None) -> None:
        self.changes.append((name, value, prev))


class _StubPar:
    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self._value = value

    def eval(self) -> object:
        # Mirrors TouchDesigner's Par.eval() (returns the parameter's evaluated value) --
        # unrelated to Python's builtin eval(); no string/code execution involved.
        return self._value


def _patch_exec_parent(monkeypatch: pytest.MonkeyPatch, ext: object | None) -> None:
    monkeypatch.setattr(
        parexecute_callbacks,
        "parent",
        lambda: types.SimpleNamespace(ext=types.SimpleNamespace(CUDAIPCExtension=ext)),
        raising=False,
    )


def test_onvaluechange_ext_none_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_exec_parent(monkeypatch, None)
    assert parexecute_callbacks.onValueChange(_StubPar("Active", True), False) is None


def test_onvaluechange_ext_raises_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduces the reported bug directly: parent().ext.CUDAIPCExtension raises
    td.tdAttributeError instead of returning None."""
    monkeypatch.setattr(
        parexecute_callbacks,
        "parent",
        lambda: types.SimpleNamespace(ext=_RaisingExt()),
        raising=False,
    )
    assert parexecute_callbacks._extension() is None
    assert parexecute_callbacks.onValueChange(_StubPar("Active", True), False) is None


_FORWARDED_PARS = ["Active", "Ipcmemname", "Numslots", "Debug", "Hidebuiltin", "Mode", "Other"]


@pytest.mark.parametrize("param_name", _FORWARDED_PARS, ids=[name.lower() for name in _FORWARDED_PARS])
def test_onvaluechange_forwards_the_change_to_the_extension(monkeypatch: pytest.MonkeyPatch, param_name: str) -> None:
    """The DAT decides nothing: name, evaluated value and previous value go to on_par_change."""
    ext = _StubExecExt()
    _patch_exec_parent(monkeypatch, ext)

    parexecute_callbacks.onValueChange(_StubPar(param_name, "new_value"), "prev_value")

    assert ext.changes == [(param_name, "new_value", "prev_value")]
