"""The delayed In TOP setup of the sync kit builder (examples/td_multistream_sync/build_kit.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[2] / "examples" / "td_multistream_sync" / "build_kit.py"
_spec = importlib.util.spec_from_file_location("build_kit_example", _PATH)
assert _spec is not None and _spec.loader is not None
build_kit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_kit)


class _Par:
    """A TD par stand-in: a value and an expression text."""

    def __init__(self) -> None:
        self.val: object = None
        self.expr = ""


class _Pars:
    """A ParCollection stand-in; plugin pars exist only after the plugin has loaded."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self._pars = {name: _Par() for name in names}

    def __getattr__(self, name: str) -> _Par:
        try:
            return self.__dict__["_pars"][name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_pars":
            object.__setattr__(self, name, value)
        else:
            self._pars[name].val = value


_PLUGIN_PARS = ("Ipcmemname", "Framewaitms", "Active")


class _Op:
    def __init__(self, pars: tuple[str, ...] = (), children: dict | None = None) -> None:
        self.par = _Pars(pars)
        self._children = children or {}

    def op(self, name: str) -> _Op | None:
        return self._children.get(name)


def _kit(loaded: list[bool]) -> tuple[_Op, dict]:
    ins = {f"in{s}": _Op(_PLUGIN_PARS if ok else ()) for s, ok in enumerate(loaded)}
    comp = _Op(children={"sync": _Op(("Streams",)), **ins})
    return comp, ins


@pytest.fixture
def td(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Fake TD globals: op() finds the registered COMPs, run() records its calls."""
    env = SimpleNamespace(comps={}, runs=[])

    def run(fn: object, *args: object, delayFrames: int) -> None:  # noqa: N803 (TD's name)
        env.runs.append((fn, args, delayFrames))

    monkeypatch.setattr(build_kit, "op", env.comps.get, raising=False)
    monkeypatch.setattr(build_kit, "run", run, raising=False)
    return env


def test_sets_pars_when_every_plugin_has_loaded(td: SimpleNamespace) -> None:
    comp, ins = _kit([True, True])
    td.comps["/k"] = comp
    build_kit._apply_in_pars("/k", 2, ["a", "b"])
    assert comp.op("sync").par.Streams.val == 2
    assert [ins[f"in{s}"].par.Ipcmemname.val for s in range(2)] == ["a", "b"]
    assert all(t.par.Active.val is True for t in ins.values())
    assert all(t.par.Framewaitms.expr == "parent().par.Framewaitms" for t in ins.values())
    assert td.runs == []


def test_requeues_itself_and_writes_nothing_while_a_plugin_is_loading(td: SimpleNamespace) -> None:
    comp, ins = _kit([True, False])
    td.comps["/k"] = comp
    build_kit._apply_in_pars("/k", 2, ["a", "b"], retries_left=3)
    assert td.runs == [(build_kit._apply_in_pars, ("/k", 2, ["a", "b"], 2), build_kit._PARAM_DELAY_FRAMES)]
    assert ins["in0"].par.Ipcmemname.val is None  # no partial setup
    assert comp.op("sync").par.Streams.val is None


def test_raises_with_the_waiting_tops_when_retries_run_out(td: SimpleNamespace) -> None:
    comp, _ = _kit([True, False, False])
    td.comps["/k"] = comp
    with pytest.raises(RuntimeError, match="in1, in2"):
        build_kit._apply_in_pars("/k", 3, ["a", "b", "c"], retries_left=0)
    assert td.runs == []


def test_missing_in_top_counts_as_waiting(td: SimpleNamespace) -> None:
    comp, _ = _kit([True])  # names asks for 2 streams; in1 does not exist
    td.comps["/k"] = comp
    build_kit._apply_in_pars("/k", 2, ["a", "b"], retries_left=1)
    assert len(td.runs) == 1


def test_removed_comp_is_a_no_op(td: SimpleNamespace) -> None:
    build_kit._apply_in_pars("/gone", 2, ["a", "b"])
    assert td.runs == []
