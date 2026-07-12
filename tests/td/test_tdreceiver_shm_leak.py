"""
Regression test: TDReceiverEngine.initialize_receiver() must not leak the locally-opened
shm_handle when a later step raises before the connection is committed.

Bug (#5b in the td_exporter audit, TDReceiver.py): initialize_receiver() opens `shm_handle`
early via `SharedMemory(name=self.shm_name)`. Every validation-guard failure closes it
explicitly, and every slot-loop failure routes through `_cleanup_partial(...)`, which also
closes it. But `create_stream_with_priority` (called after all validation guards, before the
slot loop -- TDReceiver.py ~797) sat outside both: a raise there hit the outer `except` while
`connection_committed` was still False, and the outer except only closed the handle when
`connection_committed` was True. Result: every failed attempt -- and every backoff retry --
leaked the SHM mapping.

Fix: track `shm_handle` independently of `connection_committed` and close it in the outer
`except` whenever it was opened but the connection was never committed.
"""

from __future__ import annotations

import contextlib
import sys
import uuid
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "td_exporter"))
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _make_receiver(shm_name: str):
    from TDConfig import TDSenderConfig
    from TDHost import TDHost
    from TDReceiver import TDReceiverEngine

    class _NullHost(TDHost):
        def __init__(self):
            self.cuda_shapes = []

        def param_value(self, name):
            return {"Active": True, "Debug": False}.get(name)

        def set_param_value(self, name, value):
            pass

        def set_param_enabled(self, name, enabled):
            pass

        def show_custom_only(self, value):
            pass

        def is_active(self):
            return True

        def find_top(self, name):
            return None

        def wrap_top(self, top):
            return top

        def make_cuda_shape(self, width, height, num_comps, data_type):
            shape = type(
                "_Shape",
                (),
                {"width": width, "height": height, "numComps": num_comps, "dataType": data_type},
            )()
            self.cuda_shapes.append(shape)
            return shape

        def set_warning_status(self, msg):
            pass

        def set_error_status(self, msg):
            pass

        def clear_status(self):
            pass

        def set_info_status(self, msg):
            pass

    config = TDSenderConfig()
    return TDReceiverEngine(
        host=_NullHost(),
        config=config,
        cuda=None,
        log_fn=lambda *a, **k: None,
        num_slots=1,
        device=0,
        shm_name=shm_name,
        verbose=False,
    )


def _write_valid_shm_frame(shm: SharedMemory) -> None:
    """Write a minimal-but-fully-valid header + metadata so initialize_receiver() sails
    through every validation guard and reaches the stream-creation step."""
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT, Metadata, SHMLayout

    layout = SHMLayout(num_slots=1)
    W, H, C = 64, 64, 4
    shm.buf[: layout.total_size] = layout.build_buffer(version=1, write_idx=0)
    Metadata(
        width=W,
        height=H,
        num_comps=C,
        format_kind=FORMAT_KIND_FLOAT,
        bits_per_comp=32,
        flags=0,
        data_size=W * H * C * 4,
    ).pack_into(memoryview(shm.buf), layout)


class _FakeCudaFailsOnStreamCreate:
    """Fake CUDA adapter: succeeds through get_device(), raises on stream creation -- the
    exact call (create_stream_with_priority, TDReceiver.py ~797) that sits outside both the
    validation guards' shm_handle.close() calls and _cleanup_partial's coverage."""

    def get_device(self) -> int:
        return 0

    def create_stream_with_priority(self, flags: int):  # noqa: ARG002
        raise RuntimeError("simulated stream creation failure")


def test_initialize_receiver_closes_shm_on_stream_create_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED before the #5b fix: a raise from create_stream_with_priority must close the
    locally-opened shm_handle instead of leaking it (leaks compound on every backoff retry)."""
    import TDReceiver as TDReceiverMod

    from cuda_link.shm_protocol import SHMLayout

    shm_name = f"test_shm_leak_{uuid.uuid4().hex[:8]}"
    layout = SHMLayout(num_slots=1)
    producer_shm = SharedMemory(create=True, name=shm_name, size=layout.total_size)

    tracked: list = []

    class _TrackingSharedMemory(SharedMemory):
        """Wraps SharedMemory to record whether close() was called on this instance.

        Installed as TDReceiver's module-level SharedMemory name, so only the receiver's
        internally-opened handle (via `SharedMemory(name=self.shm_name)`) is tracked -- the
        producer handle above is a plain, unpatched SharedMemory.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.close_called = False
            tracked.append(self)

        def close(self):
            self.close_called = True
            super().close()

    try:
        _write_valid_shm_frame(producer_shm)

        monkeypatch.setattr(TDReceiverMod, "get_cuda_runtime", lambda device: _FakeCudaFailsOnStreamCreate())
        monkeypatch.setattr(TDReceiverMod, "SharedMemory", _TrackingSharedMemory)

        engine = _make_receiver(shm_name)

        result = engine.initialize_receiver()

        assert result is False, "initialize_receiver() must return False when stream creation raises"
        assert len(tracked) == 1, f"expected exactly one receiver-side SharedMemory open, got {len(tracked)}"
        assert tracked[0].close_called is True, (
            "shm_handle.close() must be called when create_stream_with_priority raises before "
            "connection_committed is set -- otherwise the SHM mapping leaks on every backoff retry"
        )
        assert engine._connection.shm_handle is None, (
            "a failed attempt must not commit its shm_handle onto self._connection"
        )
    finally:
        producer_shm.close()
        with contextlib.suppress(FileNotFoundError):
            producer_shm.unlink()
