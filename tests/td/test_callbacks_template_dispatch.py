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
