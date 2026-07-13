"""Coverage-focused tests for cuda_link._nvtx.

Covers the _detect_nvtx() success/failure import branches (via a fake `nvtx`
module injected into sys.modules) and the available/verbose delegation
branches of annotate/verbose_range/push_range/pop_range/is_enabled/is_verbose
(via a monkeypatched module-level _NVTX_STATE). None of this requires the
real `nvtx` PyPI package or a GPU.
"""

from __future__ import annotations

import sys
import types

import cuda_link._nvtx as nvtx_mod
from cuda_link._nvtx import _detect_nvtx, _NvtxState


def test_detect_nvtx_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CUDALINK_NVTX", raising=False)
    monkeypatch.delenv("CUDALINK_NVTX_VERBOSE", raising=False)
    state = _detect_nvtx()
    assert state.available is False
    assert state.verbose is False
    assert state.lib is None


def test_detect_nvtx_success_when_enabled_and_lib_importable(monkeypatch):
    fake_nvtx = types.ModuleType("nvtx")
    fake_nvtx.annotate = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "nvtx", fake_nvtx)
    monkeypatch.setenv("CUDALINK_NVTX", "1")
    monkeypatch.delenv("CUDALINK_NVTX_VERBOSE", raising=False)

    state = _detect_nvtx()

    assert state.available is True
    assert state.verbose is False
    assert state.lib is fake_nvtx


def test_detect_nvtx_falls_back_when_import_fails(monkeypatch):
    # Sentinel None in sys.modules forces `import nvtx` to raise ImportError.
    monkeypatch.setitem(sys.modules, "nvtx", None)
    monkeypatch.setenv("CUDALINK_NVTX", "1")
    monkeypatch.delenv("CUDALINK_NVTX_VERBOSE", raising=False)

    state = _detect_nvtx()

    assert state.available is False
    assert state.lib is None


def test_detect_nvtx_verbose_implies_enabled(monkeypatch):
    fake_nvtx = types.ModuleType("nvtx")
    fake_nvtx.annotate = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "nvtx", fake_nvtx)
    monkeypatch.delenv("CUDALINK_NVTX", raising=False)
    monkeypatch.setenv("CUDALINK_NVTX_VERBOSE", "1")

    state = _detect_nvtx()

    assert state.available is True
    assert state.verbose is True


def test_annotate_and_verbose_range_delegate_when_available(monkeypatch):
    calls = []

    class _FakeLib:
        def annotate(self, message, color="white"):
            calls.append((message, color))
            return "range_obj"

    fake_state = _NvtxState(lib=_FakeLib(), available=True, verbose=True)
    monkeypatch.setattr(nvtx_mod, "_NVTX_STATE", fake_state)

    assert nvtx_mod.annotate("phase", color="green") == "range_obj"
    assert nvtx_mod.verbose_range("sub_phase", color="blue") == "range_obj"
    assert calls == [("phase", "green"), ("sub_phase", "blue")]


def test_verbose_range_falls_back_to_noop_when_not_verbose(monkeypatch):
    fake_state = _NvtxState(lib=object(), available=True, verbose=False)
    monkeypatch.setattr(nvtx_mod, "_NVTX_STATE", fake_state)

    result = nvtx_mod.verbose_range("sub_phase")
    assert result is nvtx_mod._NOOP


def test_push_range_and_pop_range_delegate_when_available(monkeypatch):
    calls = []

    class _FakeLib:
        def push_range(self, message, color="white"):
            calls.append(("push", message, color))

        def pop_range(self):
            calls.append(("pop",))

    fake_state = _NvtxState(lib=_FakeLib(), available=True, verbose=False)
    monkeypatch.setattr(nvtx_mod, "_NVTX_STATE", fake_state)

    nvtx_mod.push_range("phase", color="red")
    nvtx_mod.pop_range()

    assert calls == [("push", "phase", "red"), ("pop",)]


def test_push_range_and_pop_range_noop_when_unavailable(monkeypatch):
    fake_state = _NvtxState(lib=None, available=False, verbose=False)
    monkeypatch.setattr(nvtx_mod, "_NVTX_STATE", fake_state)

    # Must not raise even though lib is None.
    nvtx_mod.push_range("phase")
    nvtx_mod.pop_range()


def test_is_enabled_and_is_verbose_reflect_state(monkeypatch):
    monkeypatch.setattr(nvtx_mod, "_NVTX_STATE", _NvtxState(lib=object(), available=True, verbose=True))
    assert nvtx_mod.is_enabled() is True
    assert nvtx_mod.is_verbose() is True

    monkeypatch.setattr(nvtx_mod, "_NVTX_STATE", _NvtxState(lib=None, available=False, verbose=False))
    assert nvtx_mod.is_enabled() is False
    assert nvtx_mod.is_verbose() is False

    # available but not verbose -> is_verbose False
    monkeypatch.setattr(nvtx_mod, "_NVTX_STATE", _NvtxState(lib=object(), available=True, verbose=False))
    assert nvtx_mod.is_verbose() is False
