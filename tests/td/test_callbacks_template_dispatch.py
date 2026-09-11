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


def test_onframestart_ext_none_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_parent(monkeypatch, None)
    callbacks_template.onFrameStart(1)  # must not raise


def test_onframestart_ext_raises_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduces the reported bug: this fired every frame in production."""
    _patch_parent_raising(monkeypatch)
    callbacks_template.onFrameStart(1)  # must not raise


def test_onframestart_sender_runs_deferred_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt(mode="Sender")
    _patch_parent(monkeypatch, ext)
    callbacks_template.onFrameStart(1)
    assert ext.deferred_cleanup_calls == 1


def test_onframestart_receiver_no_import_buffer_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt(mode="Receiver")
    _patch_parent(monkeypatch, ext)
    monkeypatch.setattr(callbacks_template, "op", lambda name: None, raising=False)
    callbacks_template.onFrameStart(1)  # must not raise


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


def test_onframeend_ext_none_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_parent(monkeypatch, None)
    callbacks_template.onFrameEnd(1)  # must not raise


def test_onframeend_ext_raises_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduces the reported bug for the Frame End toggle."""
    _patch_parent_raising(monkeypatch)
    callbacks_template.onFrameEnd(1)  # must not raise


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


def test_onexit_ext_none_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_parent(monkeypatch, None)
    callbacks_template.onExit()  # must not raise


def test_onexit_ext_raises_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduces the reported bug on TD shutdown / DAT destruction."""
    _patch_parent_raising(monkeypatch)
    callbacks_template.onExit()  # must not raise


def test_onexit_calls_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = _StubExt()
    _patch_parent(monkeypatch, ext)
    callbacks_template.onExit()
    assert ext.cleanup_calls == 1


# ---------------------------------------------------------------------------
# Required-but-unused TD callback stubs
# ---------------------------------------------------------------------------


def test_onstart_and_oncreate_are_noops() -> None:
    callbacks_template.onStart()
    callbacks_template.onCreate()
