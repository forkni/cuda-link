"""
Bridge mode CLI — wire an existing cuda-link IPC channel to a Spout name, both directions.

    # cuda-link IPC  ->  Spout sender   (AI output reaches Resolume/UE/OBS/...)
    python -m cuda_link_spout.bridge --dir out --ipc my_texture_ipc --spout cuda_link_out \
        --width 1024 --height 1024 --fmt RGBA8

    # Spout sender   ->  cuda-link IPC  (VJ-app output reaches a Python ML process)
    python -m cuda_link_spout.bridge --dir in  --spout resolume_out --ipc ai_input_ipc

Argument parsing and spec-building are pure (no cuda-link import) and unit-tested;
the run loop lazily imports cuda_link so the module loads without a GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from ._types import SpoutReceiverSpec, SpoutSenderSpec


@dataclass(frozen=True)
class BridgeArgs:
    """Validated bridge configuration parsed from argv."""

    direction: str  # "out" (ipc->spout) or "in" (spout->ipc)
    ipc: str
    spout: str
    width: int
    height: int
    fmt: str
    device: int


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cuda_link_spout.bridge",
        description="Bridge a cuda-link CUDA-IPC channel to/from a Spout name (same machine, on-GPU).",
    )
    p.add_argument("--dir", required=True, choices=("out", "in"), help="out = ipc->spout, in = spout->ipc")
    p.add_argument("--ipc", required=True, help="cuda-link SHM name (Ipcmemname)")
    p.add_argument("--spout", required=True, help="Spout sender name to create (out) or receive (in)")
    p.add_argument("--width", type=int, default=0, help="frame width (required for --dir out)")
    p.add_argument("--height", type=int, default=0, help="frame height (required for --dir out)")
    p.add_argument("--fmt", default="RGBA8", help="pixel format (out only); one of the supported Spout formats")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    return p


def parse_args(argv: list[str]) -> BridgeArgs:
    """Parse and validate argv into a :class:`BridgeArgs`.

    Raises:
        SystemExit: on malformed argv (argparse) — or argparse.ArgumentError-style
            ``ValueError`` for the cross-field 'out requires width/height' rule.
    """
    ns = build_parser().parse_args(argv)
    if ns.dir == "out" and (ns.width <= 0 or ns.height <= 0):
        raise ValueError("--dir out requires positive --width and --height")
    return BridgeArgs(
        direction=ns.dir,
        ipc=ns.ipc,
        spout=ns.spout,
        width=ns.width,
        height=ns.height,
        fmt=ns.fmt,
        device=ns.device,
    )


def sender_spec(args: BridgeArgs) -> SpoutSenderSpec:
    """Build the Spout sender spec for an --dir out bridge."""
    return SpoutSenderSpec(name=args.spout, width=args.width, height=args.height, fmt=args.fmt, device=args.device)


def receiver_spec(args: BridgeArgs) -> SpoutReceiverSpec:
    """Build the Spout receiver spec for an --dir in bridge."""
    return SpoutReceiverSpec(name=args.spout, device=args.device)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Lazily imports cuda_link (and torch/cupy where needed)."""
    import sys

    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.direction == "out":
        return _run_out(args)
    return _run_in(args)


def _run_out(args: BridgeArgs) -> int:  # pragma: no cover - requires cuda_link + GPU
    """ipc -> spout: import frames from cuda-link, publish each to Spout."""
    from cuda_link import Importer, ImportOutcome, ImportSpec  # lazy

    from .sender import SpoutSender

    spec = sender_spec(args)
    with Importer.open(ImportSpec(shm_name=args.ipc, device=args.device)) as imp, SpoutSender.open(spec) as tx:
        while True:
            res = imp.get_frame_cupy()
            if res.outcome is ImportOutcome.NEW_FRAME:
                tx.send(res.frame)


def _exporter_spec_key(frame) -> tuple:  # type: ignore[type-arg]
    """Identity tuple that determines whether the inbound Exporter must be reopened.

    Any change in geometry or pixel layout means the live cuda-link FrameSpec no
    longer matches incoming frames. Duck-typed on the frame's attributes (no GPU,
    no cuda_link import) so it is unit-testable.
    """
    return (frame.width, frame.height, frame.fmt.channels, frame.fmt.dtype)


def _run_in(args: BridgeArgs) -> int:  # pragma: no cover - requires cuda_link + GPU
    """spout -> ipc: receive frames from Spout, publish each through cuda-link.

    A Spout sender's geometry/format can change at runtime (the sender restarts or
    reconfigures). The cuda-link Exporter is bound to a fixed FrameSpec, so we track
    the active (width, height, channels, dtype) and reopen the Exporter whenever it
    changes — otherwise a size mismatch makes every later export() return FAILED and
    the bridge silently stops publishing. FAILED is also handled explicitly: it is
    unrecoverable for the current Exporter, so we drop it and let the next frame
    reopen one.
    """
    import logging

    from cuda_link import Exporter, FrameOutcome, FrameSpec, GpuFrame  # lazy

    from ._format import frame_nbytes
    from ._types import ReceiveOutcome
    from .receiver import SpoutReceiver

    log = logging.getLogger(__name__)
    exporter = None
    active_key = None  # (width, height, channels, dtype) the current Exporter is bound to
    try:
        with SpoutReceiver.open(receiver_spec(args)) as rx:
            while True:
                frame = rx.receive()
                if frame.outcome is not ReceiveOutcome.NEW_FRAME or frame.fmt is None:
                    continue
                key = _exporter_spec_key(frame)
                if key != active_key:
                    if exporter is not None:
                        exporter.close()
                    exporter = Exporter.open(
                        FrameSpec(
                            shm_name=args.ipc,
                            height=frame.height,
                            width=frame.width,
                            channels=frame.fmt.channels,
                            dtype=frame.fmt.dtype,
                            device=args.device,
                        )
                    )
                    active_key = key
                    log.info(
                        "cuda-link-spout: (re)opened exporter for %dx%d %s",
                        frame.width,
                        frame.height,
                        frame.fmt.name,
                    )
                outcome = exporter.export(
                    GpuFrame(ptr=frame.ptr, size=frame_nbytes(frame.width, frame.height, frame.fmt))
                )
                if outcome is FrameOutcome.FAILED:
                    # Unrecoverable for this Exporter — close and force a reopen next frame.
                    log.warning(
                        "cuda-link-spout: export() FAILED for %dx%d; reopening exporter",
                        frame.width,
                        frame.height,
                    )
                    exporter.close()
                    exporter = None
                    active_key = None
    finally:
        if exporter is not None:
            exporter.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
