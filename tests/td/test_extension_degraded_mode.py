"""
Characterization tests for CUDAIPCExtension's compile-always / degraded-mode contract.

Locks the core invariant of the modal-storm fix: CUDAIPCExtension MUST construct and
every public method MUST be callable without raising, even when cuda_link (SHMProtocol,
TDConfig, TDReceiver, TDSender) failed to import.  Before this fix, a missing cuda_link
made the module itself fail to import, so `ext.CUDAIPCExtension` raised
td.tdAttributeError on every Execute DAT callback (the reported per-frame crash) --
see StreamDiffusionTD modal-storm / SHMProtocol ImportError incident.

No live TD or CUDA device required: these tests exercise the LIBRARY_READY=False path
directly via monkeypatching, the same way a real machine hits it when the selected
Base Folder has no cuda_link installed yet.
"""

from __future__ import annotations

from types import SimpleNamespace

import CUDAIPCExtension as ext_module
from fakes import FakeTDHost


def _degraded_host(mode: str = "Sender") -> FakeTDHost:
    return FakeTDHost(params={"Mode": mode, "Ipcmemname": "test_ipc", "Numslots": 3, "Active": True})


def test_extension_compiles_when_library_unavailable(monkeypatch: object) -> None:
    """Simulates the reported bug: cuda_link import failed.  Construction must not raise."""
    monkeypatch.setattr(ext_module, "LIBRARY_READY", False)
    monkeypatch.setattr(ext_module, "LIBRARY_ERROR", "ModuleNotFoundError: No module named 'SHMProtocol'")
    monkeypatch.setattr(ext_module, "_banner_shown_for_comps", set())

    ext = ext_module.CUDAIPCExtension(None, host=_degraded_host())

    assert isinstance(ext._engine, ext_module._NullEngine)
    assert ext.library_ready is False


def test_degraded_sender_public_methods_never_raise(monkeypatch: object) -> None:
    monkeypatch.setattr(ext_module, "LIBRARY_READY", False)
    monkeypatch.setattr(ext_module, "LIBRARY_ERROR", "OSError: nvcuda.dll not found")
    monkeypatch.setattr(ext_module, "_banner_shown_for_comps", set())

    sender = ext_module.CUDAIPCExtension(None, host=_degraded_host("Sender"))
    assert sender.initialize(16, 16, 4) is False
    assert sender.export_frame(None) is False
    assert sender.import_frame(None) is False  # wrong-mode guard fires first, also inert
    assert sender.has_new_frame() is True       # Sender mode: unconditionally True (unused path)
    assert sender.is_ready() is False
    sender._check_deferred_cleanup()
    sender.update_receiver_resolution(None)
    sender.update_receiver_format(None)
    sender.request_immediate_reconnect()
    assert sender.consume_pending_resolution() is None
    assert sender.consume_pending_format() is None
    assert sender.get_stats() == {"ready": False, "error": "OSError: nvcuda.dll not found"}
    sender.cleanup()
    sender.__delTD__()


def test_degraded_receiver_public_methods_never_raise(monkeypatch: object) -> None:
    monkeypatch.setattr(ext_module, "LIBRARY_READY", False)
    monkeypatch.setattr(ext_module, "LIBRARY_ERROR", "OSError: nvcuda.dll not found")
    monkeypatch.setattr(ext_module, "_banner_shown_for_comps", set())

    receiver = ext_module.CUDAIPCExtension(None, host=_degraded_host("Receiver"))
    assert isinstance(receiver._engine, ext_module._NullEngine)
    assert receiver.has_new_frame() is False
    assert receiver.initialize_receiver() is False
    assert receiver.export_frame(None) is False  # wrong-mode guard
    assert receiver.import_frame(None) is False
    receiver.update_receiver_resolution(None)
    receiver.update_receiver_format(None)
    receiver.request_immediate_reconnect()
    assert receiver.consume_pending_resolution() is None
    assert receiver.consume_pending_format() is None
    receiver.cleanup()


def test_library_ready_true_when_engine_is_real(monkeypatch: object) -> None:
    """Sanity check the property's other branch: real engine -> library_ready True."""
    monkeypatch.setattr(ext_module, "_banner_shown_for_comps", set())
    ext = ext_module.CUDAIPCExtension(None, host=_degraded_host())
    assert not isinstance(ext._engine, ext_module._NullEngine)
    assert ext.library_ready is True


def test_bootstrap_helpers_survive_missing_bootstrap_module(monkeypatch: object) -> None:
    """_bootstrap_active/_bootstrap_error must not raise even when CUDALinkBootstrap is absent
    (the sibling Text DAT does not exist as a COMP-local import)."""
    monkeypatch.setattr(ext_module, "CUDALinkBootstrap", None)
    assert ext_module._bootstrap_active() is False
    assert ext_module._bootstrap_error() == ""


def test_bootstrap_helpers_read_real_bootstrap_state(monkeypatch: object) -> None:
    fake_bootstrap = SimpleNamespace(_active=True, last_error="")
    monkeypatch.setattr(ext_module, "CUDALinkBootstrap", fake_bootstrap)
    assert ext_module._bootstrap_active() is True
    assert ext_module._bootstrap_error() == ""

    fake_bootstrap._active = False
    fake_bootstrap.last_error = "CUDA-Link could not be resolved -- Base Folder parameter: not set"
    assert ext_module._bootstrap_active() is False
    assert ext_module._bootstrap_error() == "CUDA-Link could not be resolved -- Base Folder parameter: not set"
