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
# One-time parameter resync: the Execute DAT only starts it. What it does is the
# extension's (CUDAIPCExtension.resync_pars, tests/td/test_extension_par_changes.py).
# ---------------------------------------------------------------------------


class _ResyncExt(_StubExt):
    def __init__(self, mode: str = "Sender", resync_to_mode: str | None = None) -> None:
        super().__init__(mode=mode)
        self.resyncs = 0
        self._resync_to_mode = resync_to_mode

    def resync_pars(self) -> None:
        self.resyncs += 1
        if self._resync_to_mode is not None:
            self.mode = self._resync_to_mode


def test_resync_runs_once_per_extension() -> None:
    ext = _ResyncExt()
    callbacks_template._resync_params_once(ext)
    callbacks_template._resync_params_once(ext)
    assert ext.resyncs == 1


def test_resync_repeats_while_the_extension_asks_for_it() -> None:
    """TD runs Frame Start inside loadTox(): the extension asks for calls until the next frame."""

    class _Again(_StubExt):
        answers = [True, True, False]
        resyncs = 0

        def resync_pars(self) -> bool:
            self.resyncs += 1
            return self.answers[self.resyncs - 1]

    ext = _Again()
    for _ in range(5):
        callbacks_template._resync_params_once(ext)
    assert ext.resyncs == 3


def test_resync_new_extension_instance_checks_again() -> None:
    """An extension re-init (a new instance) gets its own check."""
    first, second = _ResyncExt(), _ResyncExt()
    callbacks_template._resync_params_once(first)
    callbacks_template._resync_params_once(second)
    assert (first.resyncs, second.resyncs) == (1, 1)


def test_resync_error_does_not_raise() -> None:
    class _Boom(_StubExt):
        def resync_pars(self) -> None:
            raise RuntimeError("resync failed")

    assert callbacks_template._resync_params_once(_Boom()) is None


def test_resync_extension_without_resync_pars_does_not_raise() -> None:
    """A 2.0.0 extension under a newer Execute DAT has no resync_pars()."""
    assert callbacks_template._resync_params_once(_StubExt()) is None


def test_onframestart_resyncs_before_mode_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The resync runs first, so a Mode change takes effect on that same frame."""
    ext = _ResyncExt(mode="Sender", resync_to_mode="Receiver")
    _patch_parent(monkeypatch, ext)
    lookups = _patch_op_recorder(monkeypatch, result=None)
    callbacks_template.onFrameStart(1)
    assert ext.resyncs == 1
    assert ext.deferred_cleanup_calls == 0
    assert lookups == ["ImportBuffer"]


def test_onframeend_resyncs_before_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loadTox() in mid-frame reaches Frame End first: a Receiver must not export."""
    ext = _ResyncExt(mode="Sender", resync_to_mode="Receiver")
    _patch_parent(monkeypatch, ext)
    callbacks_template.onFrameEnd(1)
    assert ext.resyncs == 1
    assert ext.export_frame_calls == 0
