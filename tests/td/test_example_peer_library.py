# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""
The example launchers hand the TouchDesigner side's cuda_link to the Python child.

Problem: the child's Python can have another cuda-link release installed (1.x speaks SHM
protocol v0.5, 2.x speaks v0.6), and the example scripts preferred that installed package.
The launcher now sets CUDALINK_PEER_LIB_PATH, and the scripts load cuda_link from it.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TD_EXPORTER = Path(__file__).resolve().parents[2] / "td_exporter"
_LAUNCHERS = ("example_sender_launcher", "example_receiver_launcher")
_SCRIPTS = ("example_sender_python.py", "example_receiver_python.py")


def _launcher(name: str):
    return importlib.reload(importlib.import_module(name))


@pytest.mark.parametrize("name", _LAUNCHERS)
def test_peer_env_points_at_the_cuda_link_this_process_loaded(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _launcher(name)
    fake = SimpleNamespace(__file__=os.path.join("X:" + os.sep, "lib", "cuda_link", "__init__.py"))
    monkeypatch.setitem(sys.modules, "cuda_link", fake)

    env = mod._peer_env()

    assert env["CUDALINK_PEER_LIB_PATH"] == os.path.abspath(os.path.join("X:" + os.sep, "lib"))
    assert env["PATH"] == os.environ["PATH"], "the child keeps the rest of the environment"


@pytest.mark.parametrize("name", _LAUNCHERS)
def test_peer_env_falls_back_to_src_next_to_the_project(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mirror-mode component: this process has no cuda_link module, the checkout has src/."""
    mod = _launcher(name)
    monkeypatch.delitem(sys.modules, "cuda_link", raising=False)
    (tmp_path / "src" / "cuda_link").mkdir(parents=True)
    (tmp_path / "src" / "cuda_link" / "__init__.py").write_text("")
    monkeypatch.setattr(mod, "project", SimpleNamespace(folder=str(tmp_path)), raising=False)

    assert mod._peer_env()["CUDALINK_PEER_LIB_PATH"] == str(tmp_path / "src")


@pytest.mark.parametrize("name", _LAUNCHERS)
def test_peer_env_has_no_path_when_no_library_is_known(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _launcher(name)
    monkeypatch.delitem(sys.modules, "cuda_link", raising=False)
    monkeypatch.delenv("CUDALINK_PEER_LIB_PATH", raising=False)
    monkeypatch.setattr(mod, "project", SimpleNamespace(folder=str(tmp_path)), raising=False)

    assert "CUDALINK_PEER_LIB_PATH" not in mod._peer_env()


@pytest.mark.parametrize("name", _LAUNCHERS)
def test_onstart_passes_the_peer_env_to_the_child(name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _launcher(name)
    script_dir = tmp_path / "td_exporter"
    script_dir.mkdir()
    (script_dir / name.replace("_launcher", "_python.py")).write_text("")
    monkeypatch.setattr(mod, "project", SimpleNamespace(folder=str(tmp_path)), raising=False)
    exe_attr = "_SENDER_PYTHON_EXE" if "sender" in name else "_RECEIVER_PYTHON_EXE"
    monkeypatch.setattr(mod, exe_attr, "python-for-test")
    monkeypatch.setattr(mod.subprocess, "CREATE_NEW_CONSOLE", 0, raising=False)
    monkeypatch.setattr(mod.subprocess, "CREATE_NEW_PROCESS_GROUP", 0, raising=False)
    monkeypatch.setitem(
        sys.modules, "cuda_link", SimpleNamespace(__file__=str(tmp_path / "lib" / "cuda_link" / "i.py"))
    )
    calls: list[dict] = []

    def _popen(args: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append({"args": args, **kwargs})
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(mod.subprocess, "Popen", _popen)

    mod.onStart()

    assert len(calls) == 1
    assert calls[0]["env"]["CUDALINK_PEER_LIB_PATH"] == str(tmp_path / "lib")


_FAKE_CONSOLE = """
def install_console_ctrl_handler(*args, **kwargs):
    return None

def run_with_watchdog(*args, **kwargs):
    return None
"""


@pytest.mark.parametrize("script", _SCRIPTS)
def test_example_script_loads_cuda_link_from_the_peer_path(script: str, tmp_path: Path) -> None:
    """The package at CUDALINK_PEER_LIB_PATH wins over the cuda_link this Python can import."""
    package = tmp_path / "peer" / "cuda_link"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = 'peer-build'\n")
    (package / "_console.py").write_text(_FAKE_CONSOLE)
    code = (
        "import runpy, sys\n"
        f"runpy.run_path({str(_TD_EXPORTER / script)!r}, run_name='loaded_by_test')\n"
        "import cuda_link\n"
        "print('VERSION', cuda_link.__version__)\n"
        f"print('ON_SYS_PATH', {str(tmp_path / 'peer')!r} in sys.path)\n"
    )
    env = dict(os.environ, CUDALINK_PEER_LIB_PATH=str(tmp_path / "peer"))

    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    assert "VERSION peer-build" in result.stdout
    assert "ON_SYS_PATH False" in result.stdout, "sys.path must not change: no other package may be hidden"


@pytest.mark.parametrize("script", _SCRIPTS)
def test_example_script_keeps_the_installed_cuda_link_of_the_same_release(script: str, tmp_path: Path) -> None:
    """Same version on both sides: the installed build stays (it has the native wait extension)."""
    import cuda_link

    package = tmp_path / "peer" / "cuda_link"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f'__version__ = "{cuda_link.__version__}"\nIS_PEER_COPY = True\n')
    code = (
        "import runpy\n"
        f"runpy.run_path({str(_TD_EXPORTER / script)!r}, run_name='loaded_by_test')\n"
        "import cuda_link\n"
        "print('IS_PEER_COPY', hasattr(cuda_link, 'IS_PEER_COPY'))\n"
    )
    env = dict(os.environ, CUDALINK_PEER_LIB_PATH=str(tmp_path / "peer"))

    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    assert "IS_PEER_COPY False" in result.stdout
