"""
RealTDHost status setters are called on every cook while a condition lasts.

An unchanged message must not touch the COMP again (color, storage, script errors),
and a message that replaces an earlier one must replace its script-error badge
instead of stacking on it or leaving it behind.
"""

from __future__ import annotations

from TDHost import RealTDHost


class _NoPar:
    pass


class _RecordingComp:
    """ownerComp stub that counts the writes the status setters make."""

    def __init__(self) -> None:
        self.par = _NoPar()
        self._color = (0.55, 0.55, 0.55)
        self.fail_color_writes = 0
        self.color_writes = 0
        self.store_writes = 0
        self.script_errors: list[str] = []

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value) -> None:
        if self.fail_color_writes:
            self.fail_color_writes -= 1
            raise RuntimeError("COMP is cooking")
        self.color_writes += 1
        self._color = tuple(value)

    def store(self, key: str, value: object) -> None:
        self.store_writes += 1

    def unstore(self, key: str) -> None:
        pass

    def addScriptError(self, msg: str) -> None:
        self.script_errors.append(msg)

    def clearScriptErrors(self, error: str = "*") -> None:
        self.script_errors.clear()

    def op(self, name: str):
        return None


def _host() -> tuple[RealTDHost, _RecordingComp]:
    comp = _RecordingComp()
    return RealTDHost(comp), comp


def test_repeated_warning_writes_the_comp_once() -> None:
    host, comp = _host()
    for _ in range(5):
        host.set_warning_status("unsupported pixel format 'rgb8'")
    assert comp.color_writes == 1
    assert comp.store_writes == 1


def test_repeated_error_writes_the_comp_once_and_adds_one_badge() -> None:
    host, comp = _host()
    for _ in range(5):
        host.set_error_status("Initialization failed: boom")
    assert comp.color_writes == 1
    assert comp.store_writes == 1
    assert comp.script_errors == ["Initialization failed: boom"]


def test_changed_message_is_written() -> None:
    host, comp = _host()
    host.set_warning_status("first")
    host.set_warning_status("second")
    assert comp.color_writes == 2
    assert comp.store_writes == 2


def test_warning_after_error_clears_the_error_badge() -> None:
    host, comp = _host()
    host.set_error_status("Initialization failed: boom")
    host.set_warning_status("unsupported pixel format 'rgb8'")
    assert comp.script_errors == []


def test_new_error_replaces_the_previous_badge() -> None:
    host, comp = _host()
    host.set_error_status("first failure")
    host.set_error_status("second failure")
    assert comp.script_errors == ["second failure"]


def test_same_warning_is_written_again_after_clear_status() -> None:
    host, comp = _host()
    host.set_warning_status("w")
    host.clear_status()
    writes_before = comp.color_writes
    host.set_warning_status("w")
    assert comp.color_writes == writes_before + 1


def test_repeated_error_restores_the_status_par_after_an_info_write() -> None:
    host, comp = _host()
    host.set_error_status("Initialization failed: boom")
    host.set_info_status("1920x1080 rgba8")
    host.set_error_status("Initialization failed: boom")
    assert comp.par.Status == "ERROR: Initialization failed: boom"
    assert comp.color_writes == 1


def test_warning_does_not_clear_script_errors_it_did_not_add() -> None:
    host, comp = _host()
    comp.script_errors.append("raised by a user callback")
    host.set_warning_status("unsupported pixel format 'rgb8'")
    assert comp.script_errors == ["raised by a user callback"]


def test_failed_comp_write_is_retried_on_the_next_call() -> None:
    host, comp = _host()
    comp.fail_color_writes = 1
    host.set_warning_status("w")
    host.set_warning_status("w")
    assert comp.color_writes == 1
    assert comp.store_writes == 1
