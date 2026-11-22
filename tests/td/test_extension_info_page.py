"""The .tox Info page: live version label, C++ TOPs row, and the standing warning.

The label and the row come from the loaded code on every init (a saved par value in a .toe goes
stale, a label written on load does not). The C++ TOPs row reads the VERSION.txt that
scripts/install_cpp_top.py writes. A version that differs from this .tox tints the COMP yellow
through a standing warning, which clear_status() does not remove.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import CUDAIPCExtension
import pytest
from fakes import FakeTDHost
from TDHost import _WARNING_COLOR, RealTDHost


def _version_txt(tmp_path: Path, version: str, cuda: str = "12.8") -> Path:
    path = tmp_path / "VERSION.txt"
    path.write_text(
        f"cuda-link {version}\nprotocol v0.6 (magic 0x0)\ntd header OP_CommonAPIVersion 10\n"
        f"cuda toolkit {cuda}\ninstalled 2026-11-20T12:00:00 from x\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# native_tops_status
# ---------------------------------------------------------------------------


def test_status_not_installed_is_no_warning(tmp_path: Path) -> None:
    assert CUDAIPCExtension.native_tops_status("2.0.0", tmp_path / "missing.txt") == ("not installed", "")


def test_status_same_version_shows_version_and_cuda(tmp_path: Path) -> None:
    assert CUDAIPCExtension.native_tops_status("2.0.0", _version_txt(tmp_path, "2.0.0")) == ("2.0.0 (CUDA 12.8)", "")


def test_status_other_version_is_a_warning(tmp_path: Path) -> None:
    text, warning = CUDAIPCExtension.native_tops_status("2.0.0", _version_txt(tmp_path, "2.1.0"))
    assert text == "2.1.0 (CUDA 12.8) - differs from 2.0.0"
    assert "2.1.0" in warning
    assert "2.0.0" in warning


def test_status_unknown_cuda_shows_version_only(tmp_path: Path) -> None:
    assert CUDAIPCExtension.native_tops_status("2.0.0", _version_txt(tmp_path, "2.0.0", cuda="unknown")) == (
        "2.0.0",
        "",
    )


def test_status_without_version_line_is_no_warning(tmp_path: Path) -> None:
    path = tmp_path / "VERSION.txt"
    path.write_text("something else\n", encoding="utf-8")
    assert CUDAIPCExtension.native_tops_status("2.0.0", path) == ("installed, version unknown", "")


def test_status_build_line_wins_over_cuda(tmp_path: Path) -> None:
    path = _version_txt(tmp_path, "2.0.0")
    path.write_text("cuda-link 2.0.0\nbuild ba5e5ab\ncuda toolkit 12.8\n", encoding="utf-8")
    assert CUDAIPCExtension.native_tops_status("2.0.0", path) == ("2.0.0 (build ba5e5ab)", "")


def test_status_no_tox_version_is_no_warning(tmp_path: Path) -> None:
    assert CUDAIPCExtension.native_tops_status("", _version_txt(tmp_path, "2.1.0")) == ("2.1.0 (CUDA 12.8)", "")


def test_version_file_is_under_documents_plugins() -> None:
    assert CUDAIPCExtension.native_tops_version_file().parts[-4:] == (
        "Derivative",
        "Plugins",
        "CudaLink",
        "VERSION.txt",
    )


# ---------------------------------------------------------------------------
# CUDAIPCExtension init
# ---------------------------------------------------------------------------


def _make_ext(monkeypatch: pytest.MonkeyPatch, version_file: Path) -> FakeTDHost:
    monkeypatch.setattr(CUDAIPCExtension, "native_tops_version_file", lambda: version_file)
    host = FakeTDHost(params={"Mode": "Receiver", "Ipcmemname": "cudalink_test_info_page", "Active": False})
    CUDAIPCExtension.CUDAIPCExtension(SimpleNamespace(path="/project1/info_test"), host=host)
    return host


def test_init_sets_live_header_label(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    host = _make_ext(monkeypatch, tmp_path / "missing.txt")
    label = host.param_labels[CUDAIPCExtension.INFO_HEADER_PAR]
    assert label.startswith(f"CUDA-Link {CUDAIPCExtension.LoadState.read().version or 'unknown'} (")


def test_init_writes_native_tops_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    host = _make_ext(monkeypatch, tmp_path / "missing.txt")
    assert (CUDAIPCExtension.NATIVE_TOPS_PAR, "not installed") in host.param_writes
    assert not [c for c in host.status_calls if c[0] == "standing"]


def test_init_other_version_sets_standing_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    host = _make_ext(monkeypatch, _version_txt(tmp_path, "0.9.9"))
    standing = [msg for kind, msg in host.status_calls if kind == "standing"]
    assert len(standing) == 1
    assert "0.9.9" in (standing[0] or "")


def test_init_same_version_sets_no_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    host = _make_ext(monkeypatch, _version_txt(tmp_path, CUDAIPCExtension.LoadState.read().version))
    assert not [c for c in host.status_calls if c[0] == "standing"]


def test_init_survives_a_failing_status_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> tuple[str, str]:
        raise RuntimeError("boom")

    monkeypatch.setattr(CUDAIPCExtension, "native_tops_status", _boom)
    host = FakeTDHost(params={"Mode": "Receiver", "Active": False})
    CUDAIPCExtension.CUDAIPCExtension(SimpleNamespace(path="/project1/info_boom"), host=host)  # must not raise
    # The label is written before the failing read; the row is not written at all.
    assert CUDAIPCExtension.INFO_HEADER_PAR in host.param_labels
    assert not [w for w in host.param_writes if w[0] == CUDAIPCExtension.NATIVE_TOPS_PAR]


# ---------------------------------------------------------------------------
# RealTDHost: set_param_label and the standing warning
# ---------------------------------------------------------------------------


class _Par:
    def __init__(self, label: str = "") -> None:
        self.label = label


class _Comp:
    def __init__(self) -> None:
        self.color = (0.55, 0.55, 0.55)
        self.par = SimpleNamespace(Cudalinkheader=_Par("old"), Status="")
        self._store: dict = {}

    def store(self, k: str, v: object) -> None:
        self._store[k] = v

    def unstore(self, k: str) -> None:
        self._store.pop(k, None)

    def addScriptError(self, msg: str) -> None:
        pass

    def clearScriptErrors(self, error: str = "*") -> None:
        pass

    def op(self, name: str) -> None:
        return None


def test_real_host_sets_label() -> None:
    comp = _Comp()
    RealTDHost(comp).set_param_label("Cudalinkheader", "CUDA-Link 2.0.0")
    assert comp.par.Cudalinkheader.label == "CUDA-Link 2.0.0"


def test_real_host_label_on_missing_par_is_noop() -> None:
    comp = _Comp()
    RealTDHost(comp).set_param_label("Nosuchpar", "x")  # must not raise
    assert not hasattr(comp.par, "Nosuchpar")
    assert comp.par.Cudalinkheader.label == "old"


def test_standing_warning_survives_clear_status() -> None:
    comp = _Comp()
    host = RealTDHost(comp)
    host.set_standing_warning("C++ TOPs differ")
    assert comp.color == _WARNING_COLOR
    host.set_warning_status("unsupported pixel format")
    host.clear_status()
    assert comp.color == _WARNING_COLOR
    assert comp.par.Status == "WARNING: C++ TOPs differ"


def test_standing_warning_removed_restores_default() -> None:
    comp = _Comp()
    host = RealTDHost(comp)
    host.set_standing_warning("C++ TOPs differ")
    host.set_standing_warning(None)
    assert comp.color == (0.55, 0.55, 0.55)
    assert comp.par.Status == "Idle"


def test_info_status_keeps_the_standing_reason() -> None:
    comp = _Comp()
    host = RealTDHost(comp)
    host.set_standing_warning("C++ TOPs differ")
    host.set_info_status("1920x1080 rgba8")
    assert comp.par.Status == "WARNING: C++ TOPs differ"


def test_info_status_still_shows_without_standing_warning() -> None:
    comp = _Comp()
    host = RealTDHost(comp)
    host.set_info_status("1920x1080 rgba8")
    assert comp.par.Status == "1920x1080 rgba8"


def test_removing_standing_warning_keeps_another_warning() -> None:
    comp = _Comp()
    host = RealTDHost(comp)
    host.set_standing_warning("C++ TOPs differ")
    host.set_warning_status("unsupported pixel format")
    host.set_standing_warning(None)
    assert comp.color == _WARNING_COLOR
    assert comp.par.Status == "WARNING: unsupported pixel format"


class _Emitter:
    def __init__(self) -> None:
        self.cooks = 0

    def cook(self, force: bool = False) -> None:
        self.cooks += 1


class _LateEmitterComp(_Comp):
    """warning_emitter appears only after the first lookup, as during a .toe load."""

    def __init__(self) -> None:
        super().__init__()
        self.emitter: _Emitter | None = None

    def op(self, name: str) -> _Emitter | None:  # type: ignore[override]
        return self.emitter if name == "warning_emitter" else None


def test_emitter_missing_at_first_lookup_is_found_later() -> None:
    comp = _LateEmitterComp()
    host = RealTDHost(comp)
    host.set_warning_status("first")  # emitter not loaded yet
    comp.emitter = _Emitter()
    host.set_warning_status("second")
    assert comp.emitter.cooks == 1


def test_standing_warning_is_deferred_in_td(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scheduled: list[tuple] = []

    def _run(script: str, *args: object, **kwargs: object) -> None:
        scheduled.append((script, args, kwargs))

    monkeypatch.setattr(CUDAIPCExtension, "run", _run, raising=False)
    monkeypatch.setattr(CUDAIPCExtension, "op", SimpleNamespace(TDResources=object()), raising=False)
    host = _make_ext(monkeypatch, _version_txt(tmp_path, "0.9.9"))
    deferred = [s for s in scheduled if s[0] == "args[0](args[1])"]
    assert len(deferred) == 1
    assert deferred[0][2]["delayFrames"] == 5
    assert not [c for c in host.status_calls if c[0] == "standing"]  # not set during init
    callback, warning = deferred[0][1]
    callback(warning)
    assert [m for k, m in host.status_calls if k == "standing"] == [warning]


def test_deferred_warning_on_a_deleted_comp_does_not_raise(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    host = _make_ext(monkeypatch, tmp_path / "missing.txt")
    ext = CUDAIPCExtension.CUDAIPCExtension(SimpleNamespace(path="/project1/info_gone"), host=host)

    class _DeadOpError(Exception):  # stands in for td.tdError, which is not a RuntimeError
        pass

    def _dead(_msg: str | None) -> None:
        raise _DeadOpError("operator was deleted")

    monkeypatch.setattr(host, "set_standing_warning", _dead)
    ext._apply_standing_warning("C++ TOPs differ")  # must not raise
    assert not [c for c in host.status_calls if c[0] == "standing"]


def test_schedule_failure_sets_the_warning_at_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _run(script: str, *_a: object, **_k: object) -> None:
        if script == "args[0](args[1])":  # only the standing warning; not the library notice
            raise RuntimeError("no TDResources")

    monkeypatch.setattr(CUDAIPCExtension, "run", _run, raising=False)
    monkeypatch.setattr(CUDAIPCExtension, "op", SimpleNamespace(TDResources=object()), raising=False)
    host = _make_ext(monkeypatch, _version_txt(tmp_path, "0.9.9"))
    assert len([c for c in host.status_calls if c[0] == "standing"]) == 1


def test_extension_tests_never_read_the_real_record(monkeypatch: pytest.MonkeyPatch) -> None:
    # The autouse conftest fixture points Documents at a folder that does not exist.
    host = FakeTDHost(params={"Mode": "Receiver", "Active": False})
    CUDAIPCExtension.CUDAIPCExtension(SimpleNamespace(path="/project1/info_iso"), host=host)
    assert (CUDAIPCExtension.NATIVE_TOPS_PAR, "not installed") in host.param_writes
