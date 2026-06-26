"""
First on-hardware smoke tests for the _spout_bridge native module.

These tests are the first exercises of the CUDA↔D3D11 interop copy path that is
otherwise only covered by pure-Python unit tests against FakeSpoutBackend. They
require a Windows box with CUDA + D3D11 and the native module compiled via:

    pip install . --config-settings=cmake.define.SPOUT2_ROOT=C:/src/Spout2

Run:
    pytest spout/tests/test_native_smoke.py -m requires_spout -v

All tests skip automatically when the native module is absent so the file is
safe to collect in any environment.
"""

from __future__ import annotations

import time

import pytest
from cuda_link_spout._format import resolve_format, row_pitch
from cuda_link_spout._types import (
    ReceiveOutcome,
    SendOutcome,
    SpoutFrame,
    SpoutReceiverSpec,
    SpoutSenderSpec,
)
from cuda_link_spout.receiver import SpoutReceiver
from cuda_link_spout.sender import SpoutSender

# ---------------------------------------------------------------------------
# Test geometry — tiny (64×64 RGBA8) for fast copies.
# ---------------------------------------------------------------------------
_W, _H, _FMT_NAME = 64, 64, "RGBA8"
_FMT = resolve_format(_FMT_NAME)
_BPP = _FMT.bytes_per_pixel  # 4
_PITCH = row_pitch(_W, _FMT)  # 256 bytes
_SIZE = _PITCH * _H  # 16 384 bytes
_SENDER_NAME = "test_cuda_link_spout_smoke"


def _native_module_present() -> bool:
    """Return True if the compiled _spout_bridge extension is importable.

    Goes through load_native_backend() so the os.add_dll_directory() call in
    _native.py runs first — a direct import of _spout_bridge skips that setup
    and fails on Python 3.8+ which no longer searches PATH for DLL dependencies.
    """
    try:
        from cuda_link_spout._native import load_native_backend  # noqa: PLC0415

        load_native_backend()
        return True
    except (ImportError, RuntimeError):
        return False


_skip_no_native = pytest.mark.skipif(
    not _native_module_present(),
    reason="native _spout_bridge not built — compile with SPOUT2_ROOT to run these tests",
)

pytestmark = [pytest.mark.requires_spout, _skip_no_native]


# ---------------------------------------------------------------------------
# Helper: borrow gpu allocation from cuda_link's runtime wrapper.
# ---------------------------------------------------------------------------
def _alloc_gpu(size: int):  # type: ignore[type-arg]
    """Allocate ``size`` bytes of device memory; return (runtime, c_void_p)."""
    from cuda_link.cuda_ipc_wrapper import get_cuda_runtime  # noqa: PLC0415

    rt = get_cuda_runtime()
    return rt, rt.malloc(size)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sender_opens_and_closes() -> None:
    """SpoutSender.open() creates the shared D3D11 texture without raising."""
    spec = SpoutSenderSpec(name=_SENDER_NAME, width=_W, height=_H, fmt=_FMT_NAME)
    with SpoutSender.open(spec) as tx:
        assert tx.is_open
    # After the context manager, the sender must be cleaned up.
    assert not tx.is_open


def test_send_real_gpu_buffer_returns_sent() -> None:
    """send() with a valid device pointer returns SENT (native backend accepted the frame)."""
    rt, dev_ptr = _alloc_gpu(_SIZE)
    try:
        spec = SpoutSenderSpec(name=_SENDER_NAME, width=_W, height=_H, fmt=_FMT_NAME)
        with SpoutSender.open(spec) as tx:
            frame = SpoutFrame(ptr=int(dev_ptr.value or 0), width=_W, height=_H, pitch=_PITCH)
            outcome = tx.send(frame)
        assert outcome is SendOutcome.SENT
    finally:
        rt.free(dev_ptr)


def test_receiver_opens_and_closes() -> None:
    """SpoutReceiver.open() creates a receiver handle without raising."""
    spec = SpoutSenderSpec(name=_SENDER_NAME, width=_W, height=_H, fmt=_FMT_NAME)
    rx_spec = SpoutReceiverSpec(name=_SENDER_NAME, device=0)
    with SpoutSender.open(spec), SpoutReceiver.open(rx_spec) as rx:
        assert rx.is_open
    assert not rx.is_open


def test_receiver_gets_new_frame_from_sender() -> None:
    """In-process SpoutReceiver receives at least one NEW_FRAME from a live SpoutSender.

    Validates the full CUDA↔D3D11 interop copy path:
      malloc GPU buffer → SpoutSender.send() → SpoutReceiver.receive() → NEW_FRAME
    """
    rt, dev_ptr = _alloc_gpu(_SIZE)
    try:
        sender_spec = SpoutSenderSpec(name=_SENDER_NAME, width=_W, height=_H, fmt=_FMT_NAME)
        receiver_spec = SpoutReceiverSpec(name=_SENDER_NAME, device=0)
        frame = SpoutFrame(ptr=int(dev_ptr.value or 0), width=_W, height=_H, pitch=_PITCH)

        with SpoutSender.open(sender_spec) as tx, SpoutReceiver.open(receiver_spec) as rx:
            assert tx.send(frame) is SendOutcome.SENT
            # Spout shared-texture handoff may take a moment; poll for up to 500 ms.
            result = None
            for _ in range(50):
                result = rx.receive()
                if result.outcome is ReceiveOutcome.NEW_FRAME:
                    break
                time.sleep(0.01)

        assert result is not None, "receive() never returned a result"
        assert result.outcome is ReceiveOutcome.NEW_FRAME, (
            f"Expected NEW_FRAME after send, got {result.outcome}. "
            "The in-process CUDA↔D3D11 copy path did not deliver a frame."
        )
        assert result.width == _W
        assert result.height == _H
        assert result.fmt is not None
        assert result.fmt.name == _FMT_NAME
    finally:
        rt.free(dev_ptr)


def test_luid_matched_d3d11_device() -> None:
    """Opening a sender does not raise even on a multi-GPU box (LUID device selection)."""
    # device=0 in the spec drives the LUID lookup that picks the D3D11 adapter whose
    # LUID matches the CUDA device. If the wrong adapter is chosen, CUDA↔D3D11 interop
    # calls fail with cudaErrorInvalidDevice.
    spec = SpoutSenderSpec(name=_SENDER_NAME + "_luid", width=_W, height=_H, fmt=_FMT_NAME, device=0)
    with SpoutSender.open(spec) as tx:
        assert tx.is_open
