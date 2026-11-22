"""The facade calls its engine without a mode check (CUDAIPCExtension._Engine).

Every engine has every verb of the Protocol: TDSenderEngine has the receiver verbs as no-ops,
TDReceiverEngine has the sender verbs as no-ops, and _NullEngine has all of them. The no-ops
return what the facade returned in that mode before the Protocol.
"""

from __future__ import annotations

import inspect

import CUDAIPCExtension
import pytest
from TDReceiver import TDReceiverEngine
from TDSender import TDSenderEngine


def _verbs() -> list[str]:
    return sorted(
        name
        for name, member in vars(CUDAIPCExtension._Engine).items()
        if inspect.isfunction(member) and not name.startswith("__")
    )


@pytest.mark.parametrize(
    "engine",
    [TDSenderEngine, TDReceiverEngine, CUDAIPCExtension._NullEngine],
    ids=["sender", "receiver", "null"],
)
def test_every_engine_has_every_verb(engine: type) -> None:
    assert len(_verbs()) == 16
    missing = [name for name in _verbs() if not callable(getattr(engine, name, None))]
    assert missing == []


def test_sender_receiver_verbs_are_the_old_sender_mode_answers() -> None:
    engine = TDSenderEngine.__new__(TDSenderEngine)  # the no-ops read no state
    assert engine.import_frame(object()) is False
    assert engine.initialize_receiver() is False
    assert engine.has_new_frame() is True
    assert engine.wait_for_new_frame() is False
    assert engine.consume_pending_resolution() is None
    assert engine.consume_pending_format() is None
    assert engine.update_receiver_resolution(object()) is None
    assert engine.update_receiver_format(object()) is None
    assert engine.request_immediate_reconnect() is None


def test_receiver_sender_verbs_are_the_old_receiver_mode_answers() -> None:
    engine = TDReceiverEngine.__new__(TDReceiverEngine)
    assert engine.export_frame(object()) is False
    assert engine.initialize(64, 64) is False
    assert engine._check_deferred_cleanup() is None


@pytest.mark.parametrize(("mode", "expected"), [("Sender", True), ("Receiver", False)], ids=["sender", "receiver"])
def test_null_engine_has_new_frame_follows_the_mode(mode: str, expected: bool) -> None:
    assert CUDAIPCExtension._NullEngine(mode).has_new_frame() is expected
