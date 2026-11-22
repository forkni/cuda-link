# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""Write the golden SHM fixtures that cpp_top/tests/topcore_tests.cpp byte-compares.

Every fixture is built only through src/cuda_link/shm_protocol.py, the source of truth
for the wire layout. The C++ test runs the same operations through ring_writer and
shm_layout.h and must produce identical bytes; it also runs ring_reader::acquire_slot on
these buffers.

The scenario values below (handle byte patterns, version, sender_pid, write_idx, slot
meta, metadata cases) are duplicated in topcore_tests.cpp. Change both together.

ring_v05_n2.bin is not written here: it is a frozen v0.5 (cuda-link 1.x) segment, kept so
the C++ test can check the "sender uses protocol v0.5" path.

Usage:
    python scripts/gen_golden_shm.py           # (re)write the fixtures
    python scripts/gen_golden_shm.py --check   # exit 1 if a fixture is missing or stale
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_shm_protocol():  # noqa: ANN202
    # Load the module file directly: shm_protocol.py is stdlib-only, but importing it as
    # cuda_link.shm_protocol runs the package __init__ (CUDA/torch probing).
    path = _REPO_ROOT / "src" / "cuda_link" / "shm_protocol.py"
    spec = importlib.util.spec_from_file_location("_golden_shm_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


sp = _load_shm_protocol()

FIXTURE_DIR = _REPO_ROOT / "cpp_top" / "tests" / "fixtures"

# --- Scenario values (mirrored in topcore_tests.cpp) ---------------------------------
# All eight bytes differ, so a byte-order or width error in the version field shows.
VERSION = 0x0102030405060708
SENDER_PID = 0x0A0B0C0D
WRITE_IDX = 6  # slot (6 - 1) % N = 1, 2, 1 for N = 2, 3, 4
SLOT_COUNTS = (2, 3, 4)

# Frames 1..WRITE_IDX are published in order, so every slot holds the slot meta of the
# last frame written to it. Frame w gets slot_meta_for(w); all fields differ per frame.
HOST_TIMESTAMP_BASE = 12345.678
PRODUCER_FRAME_BASE = 0x1122334455667700
SOURCE_TIMESTAMP_US_BASE = 0x0102030405060700
STREAM_ID = 0x0000ABCD


def slot_meta_for(write_idx: int) -> sp.SlotMeta:
    return sp.SlotMeta(
        host_timestamp=HOST_TIMESTAMP_BASE + write_idx,
        producer_frame=PRODUCER_FRAME_BASE + write_idx,
        source_timestamp_us=SOURCE_TIMESTAMP_US_BASE + write_idx,
        stream_id=STREAM_ID,
    )


def _publish_frames(buf: bytearray, layout: sp.SHMLayout) -> None:
    for w in range(1, WRITE_IDX + 1):
        m = slot_meta_for(w)
        sp.publish_frame(
            memoryview(buf),
            layout,
            w,
            host_timestamp=m.host_timestamp,
            producer_frame=m.producer_frame,
            source_timestamp_us=m.source_timestamp_us,
            stream_id=m.stream_id,
        )


# (name, width, height, num_comps, dtype, extra flags). data_size = expected_size.
METADATA_CASES: tuple[tuple[str, int, int, int, str, int], ...] = (
    ("float32_rgba", 1920, 1080, 4, "float32", 0),
    ("float16_rgba", 1280, 720, 4, "float16", 0),
    ("bfloat16_mono", 640, 480, 1, "bfloat16", 0),
    ("uint8_rgba", 1920, 1080, 4, "uint8", 0),
    ("uint8_bgra", 1920, 1080, 4, "uint8", sp.FLAGS_BGRA),
    ("uint8_mono_alpha", 512, 512, 2, "uint8", sp.FLAGS_MONO_ALPHA),
    ("uint16_mono", 1024, 1024, 1, "uint16", 0),
    ("int8_mono", 256, 256, 1, "int8", 0),
    ("int16_rg", 256, 256, 2, "int16", 0),
)
RING_METADATA_CASE = "uint8_bgra"  # nonzero flags in every ring fixture


def mem_handle(slot: int) -> bytes:
    """Deterministic 64-byte stand-in for slot's cudaIpcMemHandle_t (never all zero)."""
    return bytes(((slot * 64 + k + 1) & 0xFF) for k in range(sp.IPC_HANDLE_SIZE))


def event_handle(slot: int) -> bytes:
    """Deterministic 64-byte stand-in for slot's cudaIpcEventHandle_t (differs from mem)."""
    return bytes((((slot * 64 + k) * 7 + 0x80) & 0xFF) for k in range(sp.IPC_HANDLE_SIZE))


def metadata_for(case_name: str) -> sp.Metadata:
    for name, width, height, comps, dtype, extra_flags in METADATA_CASES:
        if name == case_name:
            kind, bits, flags = sp.DtypeCodec.encode(dtype)
            draft = sp.Metadata(width, height, comps, kind, bits, flags | extra_flags, 0)
            return sp.Metadata(width, height, comps, kind, bits, flags | extra_flags, draft.expected_size)
    raise KeyError(case_name)


def _initialized(num_slots: int) -> tuple[bytearray, sp.SHMLayout]:
    """Header + every slot handle + metadata + shutdown cleared; write_idx 0."""
    layout = sp.SHMLayout(num_slots)
    buf = layout.build_buffer(version=VERSION, write_idx=0, sender_pid=SENDER_PID)
    view = memoryview(buf)
    for slot in range(num_slots):
        mem_off = layout.mem_handle_offset(slot)
        evt_off = layout.event_handle_offset(slot)
        view[mem_off : mem_off + sp.IPC_HANDLE_SIZE] = mem_handle(slot)
        view[evt_off : evt_off + sp.IPC_HANDLE_SIZE] = event_handle(slot)
    metadata_for(RING_METADATA_CASE).pack_into(view, layout)
    sp.clear_shutdown(view, layout)
    return buf, layout


def build_fixtures() -> dict[str, bytes]:
    fixtures: dict[str, bytes] = {}

    for n in SLOT_COUNTS:
        buf, layout = _initialized(n)
        _publish_frames(buf, layout)
        fixtures[f"ring_n{n}.bin"] = bytes(buf)

    fresh, _ = _initialized(2)
    fixtures["fresh_n2.bin"] = bytes(fresh)

    shut, layout = _initialized(3)
    _publish_frames(shut, layout)
    sp.set_shutdown(memoryview(shut), layout)
    fixtures["shutdown_n3.bin"] = bytes(shut)

    # One 20-byte record per METADATA_CASES entry, in order.
    scratch_layout = sp.SHMLayout(2)
    records = bytearray()
    for case in METADATA_CASES:
        scratch = scratch_layout.build_buffer()
        metadata_for(case[0]).pack_into(memoryview(scratch), scratch_layout)
        off = scratch_layout.metadata_offset
        records += scratch[off : off + sp.METADATA_SIZE]
    fixtures["meta_formats.bin"] = bytes(records)

    return fixtures


def stale_fixtures(fixture_dir: Path = FIXTURE_DIR) -> list[str]:
    """Names of fixtures that are missing or differ from build_fixtures()."""
    return [
        name
        for name, data in build_fixtures().items()
        if not (fixture_dir / name).is_file() or (fixture_dir / name).read_bytes() != data
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if a fixture is missing or stale")
    args = parser.parse_args(argv)

    if args.check:
        stale = stale_fixtures()
        for name in stale:
            print(f"stale or missing: {FIXTURE_DIR / name}")
        if stale:
            print("run: python scripts/gen_golden_shm.py")
        return 1 if stale else 0

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in build_fixtures().items():
        (FIXTURE_DIR / name).write_bytes(data)
        print(f"wrote {FIXTURE_DIR / name} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
