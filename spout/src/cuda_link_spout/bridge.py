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

_REPORT_EVERY = 150  # frames between stats lines (mirrors example_receiver_python.py)


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
    verbose: bool = False  # --verbose / --debug: enable DEBUG-level console output


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
    p.add_argument(
        "--verbose",
        "--debug",
        dest="verbose",
        action="store_true",
        default=False,
        help="enable verbose (DEBUG-level) console output",
    )
    return p


def parse_args(argv: list[str]) -> BridgeArgs:
    """Parse and validate argv into a :class:`BridgeArgs`.

    ``--width`` / ``--height`` default to ``0`` for both directions.  For
    ``--dir out``, geometry is now derived lazily from the first IPC frame
    (mirroring ``--dir in``).  Explicit positive values are still accepted and
    take precedence (back-compat).

    Raises:
        SystemExit: on malformed argv (argparse).
    """
    ns = build_parser().parse_args(argv)
    return BridgeArgs(
        direction=ns.dir,
        ipc=ns.ipc,
        spout=ns.spout,
        width=ns.width,
        height=ns.height,
        fmt=ns.fmt,
        device=ns.device,
        verbose=ns.verbose,
    )


def sender_spec(args: BridgeArgs) -> SpoutSenderSpec:
    """Build the Spout sender spec for an --dir out bridge."""
    return SpoutSenderSpec(name=args.spout, width=args.width, height=args.height, fmt=args.fmt, device=args.device)


def receiver_spec(args: BridgeArgs) -> SpoutReceiverSpec:
    """Build the Spout receiver spec for an --dir in bridge."""
    return SpoutReceiverSpec(name=args.spout, device=args.device)


def _print_banner(args: BridgeArgs) -> None:  # pragma: no cover
    """Print the startup banner to stdout (flush=True so it appears immediately in cmd)."""
    flow = "ipc → spout" if args.direction == "out" else "spout → ipc"
    print("=" * 56, flush=True)
    print(f"  CUDA-Link Spout Bridge  --  {flow}", flush=True)
    print("=" * 56, flush=True)
    print(f"  ipc    : {args.ipc}", flush=True)
    print(f"  spout  : {args.spout}", flush=True)
    print(f"  device : {args.device}", flush=True)
    if args.direction == "out" and args.fmt != "RGBA8":
        print(f"  fmt    : {args.fmt}  (explicit override)", flush=True)
    print(flush=True)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Lazily imports cuda_link (and torch/cupy where needed)."""
    import logging
    import sys

    args = parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        stream=sys.stdout,
        format="[bridge] %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    if args.direction == "out":
        return _run_out(args)
    return _run_in(args)


def _run_out(args: BridgeArgs) -> int:  # pragma: no cover - requires cuda_link + GPU
    """ipc -> spout: import frames from cuda-link, publish each to Spout.

    Geometry (width, height, pixel format) is derived lazily from the first
    imported IPC frame and whenever the frame geometry changes — mirroring the
    :func:`_run_in` lazy-reopen pattern for the cuda-link Exporter.

    Explicit ``--width``/``--height`` (both > 0) are honoured as an override for
    back-compat and force-open the sender before the first frame arrives.  An
    explicit ``--fmt`` overrides the dtype-derived format in both cases.

    Auto-derived ``uint8`` frames use **BGRA8** (matching TD's ``cudaMemory()``
    uint8=BGRA wire convention).  Pass ``--fmt RGBA8`` explicitly when the IPC
    source is a non-TD uint8 sender that emits RGBA-ordered bytes.
    """
    import logging
    import time

    from cuda_link import Importer, ImportOutcome, ImportSpec  # lazy

    from ._format import format_from_dtype, resolve_format
    from ._types import SpoutSenderSpec
    from .sender import SpoutSender

    log = logging.getLogger(__name__)
    sender: SpoutSender | None = None
    active_key: tuple | None = None  # (width, height, fmt_name) the live sender is bound to
    explicit_override = args.width > 0 and args.height > 0

    _print_banner(args)
    print("[bridge] Opening IPC channel — waiting for first frame ...\n", flush=True)

    frame_count = 0
    start_time = time.perf_counter()
    last_report = start_time
    last_report_frame = 0

    try:
        with Importer.open(ImportSpec(shm_name=args.ipc, device=args.device)) as imp:
            while True:
                res = imp.get_frame_cupy()
                if res.outcome is not ImportOutcome.NEW_FRAME:
                    continue

                frame = res.frame
                # Derive geometry: explicit values win, else read from the cupy tensor.
                if explicit_override:
                    w, h = args.width, args.height
                    fmt = resolve_format(args.fmt)
                else:
                    h, w = int(frame.shape[0]), int(frame.shape[1])
                    fmt = format_from_dtype(str(frame.dtype))

                key = (w, h, fmt.name)
                if key != active_key:
                    if sender is not None:
                        sender.close()
                    spec = SpoutSenderSpec(name=args.spout, width=w, height=h, fmt=fmt.name, device=args.device)
                    sender = SpoutSender.open(spec)
                    active_key = key
                    log.info(
                        "cuda-link-spout: (re)opened sender for %dx%d %s",
                        w,
                        h,
                        fmt.name,
                    )

                sender.send(frame)

                frame_count += 1
                now = time.perf_counter()
                if frame_count % _REPORT_EVERY == 0 or (now - last_report) >= 5.0:
                    window_dt = now - last_report
                    window_frames = frame_count - last_report_frame
                    fps = window_frames / window_dt if window_dt > 0 else 0.0
                    latency_ms = imp.last_latency
                    print(
                        f"  Frame {frame_count:5d} | {fps:5.1f} FPS | "
                        f"shape={frame.shape} dtype={frame.dtype} | "
                        f"latency={latency_ms:.2f} ms | -> spout {args.spout!r}",
                        flush=True,
                    )
                    last_report = now
                    last_report_frame = frame_count

    except KeyboardInterrupt:
        print(f"\n[bridge] Stopped after {frame_count} frames.", flush=True)
    finally:
        if sender is not None:
            sender.close()
        if args.verbose:
            total = time.perf_counter() - start_time
            avg_fps = frame_count / total if total > 0 else 0.0
            print(
                f"[bridge] Done — {frame_count} frames in {total:.1f}s ({avg_fps:.1f} FPS avg)",
                flush=True,
            )

    return 0


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
    import time

    from cuda_link import Exporter, FrameOutcome, FrameSpec, GpuFrame  # lazy

    from ._format import frame_nbytes
    from ._types import ReceiveOutcome
    from .receiver import SpoutReceiver

    log = logging.getLogger(__name__)
    exporter = None
    active_key = None  # (width, height, channels, dtype) the current Exporter is bound to

    _print_banner(args)
    print("[bridge] Opening Spout receiver — waiting for sender ...\n", flush=True)

    frame_count = 0
    start_time = time.perf_counter()
    last_report = start_time
    last_report_frame = 0

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

                frame_count += 1
                now = time.perf_counter()
                if frame_count % _REPORT_EVERY == 0 or (now - last_report) >= 5.0:
                    window_dt = now - last_report
                    window_frames = frame_count - last_report_frame
                    fps = window_frames / window_dt if window_dt > 0 else 0.0
                    print(
                        f"  Frame {frame_count:5d} | {fps:5.1f} FPS | "
                        f"shape={frame.height}x{frame.width}x{frame.fmt.channels} "
                        f"dtype={frame.fmt.dtype} | -> ipc {args.ipc!r}",
                        flush=True,
                    )
                    last_report = now
                    last_report_frame = frame_count

    except KeyboardInterrupt:
        print(f"\n[bridge] Stopped after {frame_count} frames.", flush=True)
    finally:
        if exporter is not None:
            exporter.close()
        if args.verbose:
            total = time.perf_counter() - start_time
            avg_fps = frame_count / total if total > 0 else 0.0
            print(
                f"[bridge] Done — {frame_count} frames in {total:.1f}s ({avg_fps:.1f} FPS avg)",
                flush=True,
            )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
