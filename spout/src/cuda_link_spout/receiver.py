"""
SpoutReceiver — receive a Spout sender's output as a cuda-link device buffer.

The native backend opens the sender's shared D3D11 texture, does one device-to-device
de-swizzle copy (texture array → linear device memory) into a backend-owned buffer,
and hands back a device pointer. From there the caller can wrap it as a torch/cupy
tensor or re-publish it through cuda-link's Exporter.
"""

from __future__ import annotations

from types import TracebackType

from ._backend import SpoutBackend
from ._format import format_from_dxgi
from ._types import ReceivedFrame, ReceiveOutcome, SpoutReceiverSpec


class SpoutReceiver:
    """A live Spout receiver.

    Construct with :meth:`open`; use as a context manager::

        with SpoutReceiver.open(SpoutReceiverSpec("resolume_out")) as rx:
            frame = rx.receive()
            if frame.outcome is ReceiveOutcome.NEW_FRAME:
                use(frame.ptr, frame.width, frame.height, frame.fmt)
    """

    def __init__(self, spec: SpoutReceiverSpec, backend: SpoutBackend, handle: int) -> None:
        self._spec = spec
        self._backend = backend
        self._handle: int | None = handle

    @classmethod
    def open(cls, spec: SpoutReceiverSpec | None = None, backend: SpoutBackend | None = None) -> SpoutReceiver:
        """Create the Spout receiver.

        Args:
            spec: receiver config; None → bind to the host's active sender on device 0.
            backend: inject a SpoutBackend (tests pass FakeSpoutBackend). None → native.
        """
        spec = spec or SpoutReceiverSpec()
        if backend is None:
            from ._native import load_native_backend

            backend = load_native_backend(spec.device)
        handle = backend.create_receiver(spec.name, spec.device)
        return cls(spec, backend, handle)

    @property
    def spec(self) -> SpoutReceiverSpec:
        return self._spec

    @property
    def is_open(self) -> bool:
        return self._handle is not None

    def receive(self) -> ReceivedFrame:
        """Receive the latest frame from the sender.

        Returns a :class:`ReceivedFrame`. On ``NEW_FRAME`` the backend has copied the
        frame into its device buffer and ``ptr``/``width``/``height``/``fmt`` are valid
        until the next ``receive()``. ``NO_FRAME`` means the sender has not advanced;
        ``NOT_CONNECTED`` means no matching sender is present.
        """
        if self._handle is None:
            return ReceivedFrame(ReceiveOutcome.FAILED)
        # dst_ptr/pitch/max_bytes are 0 here: the backend owns and sizes the destination
        # buffer (it must know the sender's geometry, which is only known after receive).
        res = self._backend.receive(self._handle, 0, 0, 0)
        if not res.connected:
            return ReceivedFrame(ReceiveOutcome.NOT_CONNECTED)
        if not res.new_frame:
            return ReceivedFrame(ReceiveOutcome.NO_FRAME)
        fmt = format_from_dxgi(res.dxgi_format)
        return ReceivedFrame(ReceiveOutcome.NEW_FRAME, ptr=res.dst_ptr, width=res.width, height=res.height, fmt=fmt)

    def close(self) -> None:
        """Release the receiver. Idempotent."""
        if self._handle is not None:
            self._backend.close_receiver(self._handle)
            self._handle = None

    def __enter__(self) -> SpoutReceiver:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
