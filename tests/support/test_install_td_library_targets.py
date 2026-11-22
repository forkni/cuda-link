"""install_td_library.py: each mode builds an InstallTarget; run_install() runs it.

No pip, no setx, no network: wheel resolution, pip and setx are replaced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent

sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
import install_td_library as itl  # noqa: E402

_WHEEL = Path("dist/cuda_link-2.0.0.dev0-cp311-cp311-win_amd64.whl")
_PIP_TAIL = [str(_WHEEL), "--upgrade", "--force-reinstall", "--no-deps"]


@pytest.fixture(autouse=True)
def _fake_wheel(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Every mode resolves this wheel; the probe command each mode passes is recorded."""
    probes: list[list[str]] = []

    def resolve(probe_cmd: list[str], *_args: object) -> Path:
        probes.append(probe_cmd)
        return _WHEEL

    monkeypatch.setattr(itl, "_resolve_wheel_for", resolve)
    return probes


def _file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_mode_1_targets_the_folder_with_the_installer_python(tmp_path: Path, _fake_wheel: list[list[str]]) -> None:
    dest = tmp_path / "lib"
    t = itl.mode_1_external_folder(str(dest), None, False, True, False)
    assert dest.is_dir()
    assert t.pip_args == [sys.executable, "-m", "pip", "install", "--target", str(dest), *_PIP_TAIL]
    assert (t.site_packages, t.td_preferences_only, t.receiver_python) == (dest, False, None)
    assert _fake_wheel == [[sys.executable]]


def test_mode_2_uses_the_venv_pip_and_python(tmp_path: Path, _fake_wheel: list[list[str]]) -> None:
    pip = _file(tmp_path / "Scripts" / "pip.exe")
    python = _file(tmp_path / "Scripts" / "python.exe")
    (tmp_path / "Lib" / "site-packages").mkdir(parents=True)
    t = itl.mode_2_venv(str(tmp_path), None, False, True, False)
    assert t.pip_args == [str(pip), "install", *_PIP_TAIL]
    assert t.site_packages == tmp_path / "Lib" / "site-packages"
    assert (t.receiver_python, t.site_query, t.td_preferences_only) == (python, None, True)
    assert _fake_wheel == [[str(python)]]


def test_mode_2_without_pip_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        itl.mode_2_venv(str(tmp_path), None, False, True, False)


def test_mode_3_queries_site_packages_through_conda(_fake_wheel: list[list[str]]) -> None:
    t = itl.mode_3_conda("myenv", None, False, True, False)
    conda = ["conda", "run", "-n", "myenv"]
    assert t.pip_args == [*conda, "pip", "install", *_PIP_TAIL]
    assert t.site_query == [*conda, "python", "-c", itl._SITE_PKGS_QUERY]
    assert t.site_packages is None and t.receiver_python is None
    assert "conda" in t.missing_site_hint[0]
    assert _fake_wheel == [[*conda, "python"]]


def test_mode_4_accepts_a_folder_and_sets_the_receiver(tmp_path: Path) -> None:
    python = _file(tmp_path / ("python.exe" if sys.platform == "win32" else "python"))
    t = itl.mode_4_system_python(str(tmp_path), None, False, True, False)
    assert t.pip_args == [str(python), "-m", "pip", "install", *_PIP_TAIL]
    assert t.site_query == [str(python), "-c", itl._SITE_PKGS_QUERY]
    assert t.receiver_python == python


def test_mode_5_is_td_python(tmp_path: Path) -> None:
    python = _file(tmp_path / "python.exe")
    t = itl.mode_5_td_python(str(python), None, False, True, False)
    assert t.pip_args == [str(python), "-m", "pip", "install", *_PIP_TAIL]
    assert (t.td_python, t.receiver_python, t.site_packages) == (True, None, None)


@pytest.mark.parametrize(
    "build",
    [
        lambda: itl.mode_1_external_folder(None, None, False, True, False),
        lambda: itl.mode_2_venv(None, None, False, True, False),
        lambda: itl.mode_3_conda(None, None, False, True, False),
        lambda: itl.mode_4_system_python(None, None, False, True, False),
        lambda: itl.mode_5_td_python(None, None, False, True, False),
    ],
    ids=["mode1", "mode2", "mode3", "mode4", "mode5"],
)
def test_non_interactive_without_the_target_exits(build: object) -> None:
    with pytest.raises(SystemExit):
        build()  # type: ignore[operator]


class _Calls:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.pip: list[list[str]] = []
        self.setx: list[tuple[str, str]] = []
        self.queries: list[list[str]] = []
        monkeypatch.setattr(itl, "_run_pip", lambda args, dry_run: self.pip.append(args))
        monkeypatch.setattr(itl, "_setx_user", lambda name, value, dry_run: self.setx.append((name, value)) or True)
        monkeypatch.setattr(itl, "_query_site_packages", self._query)
        monkeypatch.setattr(itl, "_wheel_version", lambda path: "2.0.0.dev0")

    def _query(self, cmd: list[str]) -> Path | None:
        self.queries.append(cmd)
        return Path("C:/py/Lib/site-packages")


def test_run_install_runs_pip_then_the_setx_tail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _Calls(monkeypatch)
    target = itl.InstallTarget(
        "venv site-packages", _WHEEL, ["pip", "x"], site_packages=tmp_path, receiver_python=Path("C:/v/python.exe")
    )
    assert itl.run_install(target, dry_run=False, set_env=True) == tmp_path
    assert calls.pip == [["pip", "x"]]
    assert calls.setx == [
        ("CUDALINK_RECEIVER_PYTHON_EXE", str(Path("C:/v/python.exe"))),
        ("CUDALINK_WAIT_BACKEND", "auto"),
    ]
    assert calls.queries == []


def test_run_install_queries_site_packages_after_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _Calls(monkeypatch)
    target = itl.InstallTarget("Python site-packages", _WHEEL, ["pip"], site_query=["py", "-c", "q"])
    assert itl.run_install(target, dry_run=False, set_env=False) == Path("C:/py/Lib/site-packages")
    assert calls.queries == [["py", "-c", "q"]]
    assert calls.setx == []


def test_run_install_dry_run_skips_the_query(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _Calls(monkeypatch)
    target = itl.InstallTarget(
        "conda env site-packages",
        _WHEEL,
        ["pip"],
        site_query=["q"],
        missing_site_hint=("  Could not auto-detect conda site-packages path.",),
    )
    assert itl.run_install(target, dry_run=True, set_env=False) is None
    assert calls.queries == []
    assert "Could not auto-detect" in capsys.readouterr().out


def test_run_install_td_python_prints_its_own_banner(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _Calls(monkeypatch)
    itl.run_install(itl.InstallTarget("TD Python", _WHEEL, ["pip"], td_python=True), dry_run=True, set_env=False)
    assert "INSTALL COMPLETE (TD Python)" in capsys.readouterr().out
