"""
SpoutSender — publish cuda-link / torch / cupy GPU frames as a Spout sender.

The pixel data stays on the GPU: the native backend does one device-to-device
de-swizzle copy (linear → shared D3D11 texture array) and publishes via Spout.
All policy/validation lives here in pure Python; the GPU work is behind the
:class:`SpoutBackend` seam, so this class is fully unit-testable with FakeSpoutBackend.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

from ._backend import SpoutBackend
from ._format import row_pitch
from ._types import SendOutcome, SpoutFrame, SpoutSenderSpec


class SpoutSender:
    """A live Spout sender bound to one cuda-link frame geometry.

    Construct with :meth:`open`; use as a context manager for guaranteed cleanup::

        with SpoutSender.open(SpoutSenderSpec("ai_out", 1024, 1024, "RGBA8")) as tx:
            tx.send(tensor)   # torch/cupy GPU tensor, or a SpoutFrame
    """

    def __init__(self, spec: SpoutSenderSpec, backend: SpoutBackend, handle: int) -> None:
        self._spec = spec
        self._backend = backend
        self._handle: int | None = handle
        self._fmt = spec.resolved_format

    @classmethod
    def open(cls, spec: SpoutSenderSpec, backend: SpoutBackend | None = None) -> SpoutSender:
        """Create the Spout sender and its CUDA-registered shared texture.

        Args:
            spec: sender geometry + format.
            backend: inject a SpoutBackend (tests pass FakeSpoutBackend). None →
                the native backend (Windows + CUDA + D3D11).
        """
        if backend is None:
            from ._native import load_native_backend

            backend = load_native_backend(spec.device)
        fmt = spec.resolved_format
        handle = backend.create_sender(spec.name, spec.width, spec.height, fmt.dxgi_format, spec.device)
        return cls(spec, backend, handle)

    @property
    def spec(self) -> SpoutSenderSpec:
        return self._spec

    @property
    def is_open(self) -> bool:
        return self._handle is not None

    def send(self, frame: Any, *, stream: int = 0) -> SendOutcome:
        """Publish one frame.

        Args:
            frame: a :class:`SpoutFrame`, or a torch/cupy GPU tensor (duck-typed via
                ``.data_ptr()`` / ``__cuda_array_interface__``) whose geometry matches the spec.
            stream: producer CUDA stream handle for pre-copy ordering (used only when
                *frame* is a raw tensor; a SpoutFrame carries its own ``stream``).

        Returns:
            SendOutcome.SENT on success; SendOutcome.FAILED if the sender is closed.
        """
        if self._handle is None:
            return SendOutcome.FAILED
        sf = self._normalize(frame, stream)
        self._backend.send(
            self._handle,
            sf.ptr,
            sf.pitch,
            sf.width,
            sf.height,
            self._fmt.bytes_per_pixel,
            sf.stream,
        )
        return SendOutcome.SENT

    def _normalize(self, frame: Any, stream: int) -> SpoutFrame:
        """Coerce the send() argument into a SpoutFrame matching the spec geometry."""
        if isinstance(frame, SpoutFrame):
            return frame
        # Treat as a tensor: derive geometry from the spec, pitch = tightly-packed row.
        return SpoutFrame.from_tensor(
            frame,
            width=self._spec.width,
            height=self._spec.height,
            pitch=row_pitch(self._spec.width, self._fmt),
            stream=stream,
        )

    def close(self) -> None:
        """Release the sender. Idempotent."""
        if self._handle is not None:
            self._backend.close_sender(self._handle)
            self._handle = None

    def __enter__(self) -> SpoutSender:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
