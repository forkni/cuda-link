"""
Characterization tests for CUDAIPCExtension's compile-always / degraded-mode contract.

Locks the core invariant of the modal-storm fix: CUDAIPCExtension MUST construct and
every public method MUST be callable without raising, even when cuda_link (SHMProtocol,
TDConfig, TDReceiver, TDSender) failed to import.  Before this fix, a missing cuda_link
made the module itself fail to import, so `ext.CUDAIPCExtension` raised
td.tdAttributeError on every Execute DAT callback (the reported per-frame crash) --
see StreamDiffusionTD modal-storm / SHMProtocol ImportError incident.

No live TD or CUDA device required: these tests exercise the LIBRARY_READY=False path
directly via monkeypatching, the same way a real machine hits it before an ADR-0014
layer (an explicit folder, the project folder, ``CUDALINK_LIB_PATH``, ``sys.path``)
resolves a matching cuda_link install.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import CUDAIPCExtension as ext_module  # noqa: N813
from fakes import FakeTDHost


def _degraded_host(mode: str = "Sender") -> FakeTDHost:
    return FakeTDHost(params={"Mode": mode, "Ipcmemname": "test_ipc", "Numslots": 3, "Active": True})


class _FakeHolder:
    """Dict-backed double for the COMP storage ``_claim_notice_once`` reads/writes via
    ``fetch``/``store`` -- distinct from ``FakeTDHost``, which has neither method."""

    def __init__(self, seed: dict[str, object] | None = None) -> None:
        self._store: dict[str, object] = dict(seed or {})

    def fetch(self, key: str, default: object = None, storeDefault: bool = False) -> object:  # noqa: N803 -- mirrors TD's real COMP.fetch signature
        return self._store.get(key, default)

    def store(self, key: str, value: object) -> None:
        self._store[key] = value


class _FakeOwnerComp:
    """Minimal COMP double: ``_claim_notice_once`` only ever calls ``.parent()`` on it."""

    def __init__(self, holder: _FakeHolder) -> None:
        self._holder = holder
        self.path = "/project1/fake_comp"

    def parent(self) -> _FakeHolder:
        return self._holder


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
    assert sender.has_new_frame() is True  # Sender mode: unconditionally True (unused path)
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
    fake_bootstrap.last_error = "CUDA-Link could not be resolved -- CUDALINK_LIB_PATH: not set"
    assert ext_module._bootstrap_active() is False
    assert ext_module._bootstrap_error() == "CUDA-Link could not be resolved -- CUDALINK_LIB_PATH: not set"


def test_notify_library_unavailable_status_names_version_and_how_to_install(monkeypatch: object) -> None:
    """The short Status line must point at the version and where to install it -- there is
    no per-COMP parameter to fix anymore, and not the retired StreamDiffusionTD
    'Base Folder' wording."""
    monkeypatch.setattr(ext_module, "LIBRARY_READY", False)
    monkeypatch.setattr(ext_module, "LIBRARY_ERROR", "ModuleNotFoundError: No module named 'SHMProtocol'")
    monkeypatch.setattr(ext_module, "_banner_shown_for_comps", set())
    monkeypatch.setattr(ext_module, "_notice_printed", False)
    monkeypatch.setattr(
        ext_module,
        "CUDALinkBootstrap",
        SimpleNamespace(_active=False, last_error="", MIRROR_VERSION="1.13.0"),
    )
    host = _degraded_host()

    ext = ext_module.CUDAIPCExtension(None, host=host)
    ext._notify_library_unavailable()

    assert host.status_calls, "set_warning_status was never called"
    kind, msg = host.status_calls[-1]
    assert kind == "warning"
    assert "1.13.0" in msg
    assert "restart" in msg.lower()
    assert "Libpath" not in msg
    assert "StreamDiffusionTD" not in msg


def test_notify_library_unavailable_detail_prefers_bootstrap_last_error(monkeypatch: object) -> None:
    """The Textport detail must lead with CUDALinkBootstrap.last_error (the cause) over
    the extension's own LIBRARY_ERROR (the downstream import-time symptom)."""
    monkeypatch.setattr(ext_module, "LIBRARY_READY", False)
    monkeypatch.setattr(ext_module, "LIBRARY_ERROR", "ModuleNotFoundError: No module named 'SHMProtocol'")
    monkeypatch.setattr(ext_module, "_banner_shown_for_comps", set())
    monkeypatch.setattr(ext_module, "_notice_printed", False)
    monkeypatch.setattr(
        ext_module,
        "CUDALinkBootstrap",
        SimpleNamespace(
            _active=False,
            last_error="folder argument: not set; CUDALINK_LIB_PATH: not set",
            MIRROR_VERSION="1.13.0",
        ),
    )
    printed: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(str(x) for x in a)))

    ext = ext_module.CUDAIPCExtension(None, host=_degraded_host())
    printed.clear()  # drop the "Extension initialized" log line emitted by __init__

    ext._notify_library_unavailable()

    assert any("CUDALINK_LIB_PATH: not set" in line for line in printed)
    assert not any("SHMProtocol" in line for line in printed)


def test_claim_notice_once_self_heals_from_a_stale_persisted_true(monkeypatch: object) -> None:
    """A .toe saved by the pre-fix code has a bare ``True`` pickled into COMP storage --
    that must not suppress the notice forever.  It must fire once for this (new) process,
    stamping the holder with this process's pid, then stay suppressed for the rest of it."""
    monkeypatch.setattr(ext_module, "_banner_shown_for_comps", set())
    holder = _FakeHolder(seed={ext_module._NOTICE_STORE_KEY: True})
    owner = _FakeOwnerComp(holder)

    ext = ext_module.CUDAIPCExtension(owner, host=_degraded_host())

    assert ext._claim_notice_once() is True
    assert holder.fetch(ext_module._NOTICE_STORE_KEY) == os.getpid()
    assert ext._claim_notice_once() is False
