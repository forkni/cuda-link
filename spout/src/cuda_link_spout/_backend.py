"""
Spout backend port — the seam between the pure-Python API and the native
CUDA↔D3D11↔Spout module.

Mirrors cuda-link's port-adapter pattern (ADR-0001): a structural ``Protocol``
plus two adapters —

  _NativeSpoutBackend  — wraps the compiled ``_spout_bridge`` pybind11 module
                         (in _native.py; Windows + CUDA + D3D11 only)
  FakeSpoutBackend     — in-memory, no GPU / no native module; drives all unit tests

The high-level :class:`SpoutSender` / :class:`SpoutReceiver` depend only on this
Protocol, so every behaviour except the actual GPU copy is testable on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Opaque native handles are represented as ints (pointer/identity tokens).
SenderHandle = int
ReceiverHandle = int


@dataclass(frozen=True)
class NativeReceiveResult:
    """What the backend reports after a single receive attempt.

    ``new_frame`` is False when the sender has not advanced since the last call
    (Spout ``IsFrameNew()``); the dst buffer is then left untouched. ``connected``
    is False when no sender of the requested name is currently present.
    """

    connected: bool
    new_frame: bool
    width: int
    height: int
    dxgi_format: int
    dst_ptr: int = 0  # device pointer to the backend's destination buffer (valid when new_frame)


@runtime_checkable
class SpoutBackend(Protocol):
    """Structural interface the Spout sender/receiver require from the native layer.

    All methods raise RuntimeError on a native/CUDA/D3D failure.
    """

    # --- sender ------------------------------------------------------------
    def create_sender(self, name: str, width: int, height: int, dxgi_format: int, device: int) -> SenderHandle:
        """Create a Spout sender + a CUDA-registered shared D3D11 texture on *device*'s adapter."""
        ...

    def send(
        self,
        handle: SenderHandle,
        src_ptr: int,
        src_pitch: int,
        width: int,
        height: int,
        bytes_per_pixel: int,
        stream: int,
    ) -> None:
        """De-swizzle-copy linear device memory at *src_ptr* into the shared texture and publish.

        *stream* is the producer's CUDA stream handle (0 = default) for pre-copy ordering.
        """
        ...

    def close_sender(self, handle: SenderHandle) -> None:
        """Release the sender, its shared texture, and the CUDA registration."""
        ...

    # --- receiver ----------------------------------------------------------
    def create_receiver(self, name: str, device: int) -> ReceiverHandle:
        """Create a Spout receiver bound to sender *name* (empty = active sender) on *device*."""
        ...

    def receive(self, handle: ReceiverHandle, dst_ptr: int, dst_pitch: int, max_bytes: int) -> NativeReceiveResult:
        """Receive the latest frame; on a new frame, de-swizzle-copy it into *dst_ptr*."""
        ...

    def close_receiver(self, handle: ReceiverHandle) -> None:
        """Release the receiver and its CUDA registration."""
        ...

    # --- diagnostics -------------------------------------------------------
    def adapter_luid(self, device: int) -> int:
        """Return the DXGI adapter LUID the backend will use for *device* (for affinity checks)."""
        ...


# ---------------------------------------------------------------------------
# Fake backend — in-memory, deterministic; the test substrate.
# ---------------------------------------------------------------------------


class FakeSpoutBackend:
    """In-memory SpoutBackend for unit tests — records calls, fabricates frames.

    Satisfies the :class:`SpoutBackend` Protocol structurally. No GPU, no native
    module. Tune behaviour via the public attributes before exercising the API.
    """

    def __init__(self, device: int = 0) -> None:
        self.device = device
        self._next_handle = 0x5000_0000
        # call logs (assertable in tests)
        self.created_senders: dict[SenderHandle, dict] = {}
        self.created_receivers: dict[ReceiverHandle, dict] = {}
        self.sent: list[dict] = []
        self.closed_senders: list[SenderHandle] = []
        self.closed_receivers: list[ReceiverHandle] = []
        # receiver-side scripting
        self.fake_connected: bool = True
        self.fake_new_frame: bool = True
        self.fake_width: int = 1920
        self.fake_height: int = 1080
        self.fake_dxgi_format: int = 28  # DXGI_FORMAT_R8G8B8A8_UNORM
        self.fake_dst_ptr: int = 0x7000_0000  # fabricated device buffer pointer
        self.receive_calls: int = 0
        # failure injection
        self.fail_on_create_sender: bool = False
        self.fail_on_send: bool = False

    def _alloc(self) -> int:
        h = self._next_handle
        self._next_handle += 0x10
        return h

    # --- sender ---
    def create_sender(self, name: str, width: int, height: int, dxgi_format: int, device: int) -> SenderHandle:
        if self.fail_on_create_sender:
            raise RuntimeError("fake: create_sender failed")
        h = self._alloc()
        self.created_senders[h] = {
            "name": name,
            "width": width,
            "height": height,
            "dxgi_format": dxgi_format,
            "device": device,
        }
        return h

    def send(self, handle, src_ptr, src_pitch, width, height, bytes_per_pixel, stream) -> None:
        if self.fail_on_send:
            raise RuntimeError("fake: send failed")
        if handle not in self.created_senders:
            raise RuntimeError("fake: send on unknown/closed sender handle")
        self.sent.append(
            {
                "handle": handle,
                "src_ptr": src_ptr,
                "src_pitch": src_pitch,
                "width": width,
                "height": height,
                "bytes_per_pixel": bytes_per_pixel,
                "stream": stream,
            }
        )

    def close_sender(self, handle) -> None:
        self.created_senders.pop(handle, None)
        self.closed_senders.append(handle)

    # --- receiver ---
    def create_receiver(self, name: str, device: int) -> ReceiverHandle:
        h = self._alloc()
        self.created_receivers[h] = {"name": name, "device": device}
        return h

    def receive(self, handle, dst_ptr, dst_pitch, max_bytes) -> NativeReceiveResult:
        if handle not in self.created_receivers:
            raise RuntimeError("fake: receive on unknown/closed receiver handle")
        self.receive_calls += 1
        new_frame = self.fake_connected and self.fake_new_frame
        return NativeReceiveResult(
            connected=self.fake_connected,
            new_frame=new_frame,
            width=self.fake_width,
            height=self.fake_height,
            dxgi_format=self.fake_dxgi_format,
            dst_ptr=self.fake_dst_ptr if new_frame else 0,
        )

    def close_receiver(self, handle) -> None:
        self.created_receivers.pop(handle, None)
        self.closed_receivers.append(handle)

    # --- diagnostics ---
    def adapter_luid(self, device: int) -> int:
        return 0xABCD_0000 | device
