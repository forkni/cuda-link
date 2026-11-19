# ADR-0015: SHM protocol v0.6 — per-slot metadata, 8-aligned version, sender process ID

**Status**: Proposed (cuda-link 2.0.0.dev0 on `feat/orbbec-v0.6`; nothing is released, users
stay on 1.14.1)
**Date**: 2026-11-16
**Applies to**: the shared-memory wire layout in `src/cuda_link/shm_protocol.py` (the source
of truth) and its C++ transcription in `cpp_top/src/core/shm_layout.h`; every sender
(Python `Exporter`, TD Sender, C++ `CudaLinkOutTOP`) and every receiver (Python `Importer`,
`TDReceiver`, C++ `CudaLinkInTOP`, `protocol_dump`).

---

## Context

The target setup has two TD instances and Orbbec cameras. Instance A sends the color and
depth streams of three cameras through C++ Out TOPs, one segment per stream. Instance B
receives all six streams. Instance B must match frames across the streams, for example the
color and depth images of one camera at one capture time.

Protocol v0.5 (cuda-link 1.x) had three problems for this setup:

1. **No frame identity per slot.** v0.5 had one global `timestamp` after the metadata. It
   told the receiver when the latest publish happened. It did not tell the receiver which
   frame a given slot holds. A receiver could not pair a slot of one stream with the slot of
   another stream that holds the same producer frame.
2. **A misaligned 64-bit atomic.** The v0.5 header put `version` (uint64) at byte 4. The C++
   reader and writer access it through `std::atomic_ref<uint64_t>`. A 4-aligned address does
   not meet `atomic_ref<uint64_t>::required_alignment`, so that access was formally undefined
   behavior. It worked on x64, but the code had no guarantee.
3. **No early loop-back check.** CUDA IPC cannot open a handle in the process that exported
   it. A receiver in the same process as its sender found out only from a CUDA error
   (`cudaErrorDeviceUninitialized`, 201) at `cudaIpcOpenMemHandle`. See
   [`docs/CPP_TOP.md`](../CPP_TOP.md) "Same-process limitation" for the 2026-11-16 probe.

## Decision

**Make one wire break, from v0.5 to v0.6, that fixes all three problems together.**

- **New magic.** `PROTOCOL_MAGIC = 0x43495036` ("CIP6") at byte 0. The v0.5 magic
  `0x43495044` ("CIPD") stays in the code as `PROTOCOL_MAGIC_V05` only. Receivers use it to
  report "sender uses protocol v0.5 (cuda-link 1.x); this receiver needs v0.6".
- **24-byte header**, all fields little-endian:

  | Offset | Field | Type | Note |
  |--------|-------|------|------|
  | 0 | `magic` | u32 | "CIP6" |
  | 4 | `num_slots` | u32 | |
  | 8 | `version` | u64 | 8-aligned, so `atomic_ref<uint64_t>` is valid |
  | 16 | `write_idx` | u32 | written last on each publish |
  | 20 | `sender_pid` | u32 | 0 = unknown |

- **160-byte slots.** Slot `i` starts at `24 + i*160`: a 64 B memory handle, a 64 B event
  handle, then 32 B of slot meta. The slot-meta offsets are relative to the slot base:

  | Offset | Field | Type |
  |--------|-------|------|
  | +128 | `host_timestamp` | f64 (sender `perf_counter` / QPC seconds at publish) |
  | +136 | `producer_frame` | u64 (sender frame number; the cross-stream match key) |
  | +144 | `source_timestamp_us` | u64 (optional producer time; not interpreted) |
  | +152 | `stream_id` | u32 (user-set stream name) |
  | +156 | `meta_flags` | u32 (reserved, 0) |

- **Trailer.** The shutdown flag is at `24 + N*160`. The 20-byte metadata is at
  `25 + N*160`. Three pad bytes follow. The total is `48 + N*160` bytes: 368, 528 and 688 for
  N = 2, 3 and 4 (v0.5: 305, 433 and 561).
- **The global `timestamp` is removed.** The per-slot `host_timestamp` does its job.
  `AcquireResult.timestamp` becomes `AcquireResult.slot_meta`, a `SlotMeta` NamedTuple.
- **Slot meta is written before the fence.**
  `publish_frame(buf, layout, write_idx, *, host_timestamp, producer_frame=0,
  source_timestamp_us=0, stream_id=0)` writes the slot meta of slot `(write_idx - 1) % N`,
  then issues the release fence, then writes `write_idx` last. A receiver that reads
  `write_idx` with acquire ordering therefore sees the slot meta of that frame.
- **Sender process ID.** The sender writes `sender_pid` together with `magic` and
  `num_slots`, before the version commit (Python `os.getpid()`, C++ Out TOP
  `GetCurrentProcessId()`). The receivers (C++ In TOP, Python `Importer`, `TDReceiver`)
  refuse a segment whose `sender_pid` equals their own process ID, before they open any
  handle. The error is "sender and receiver appear to be in the same process -- CUDA IPC
  cannot loop back within one process; run the sender in a separate TD instance or use the
  Python peer". The In TOP keeps the 201 mapping as a fallback for `sender_pid` 0.
- **Scope of this phase.** Both senders write `producer_frame = 0` and `stream_id = 0`.
  A later phase (Phase 4) fills them from Out TOP parameters and `GpuFrame` fields. This ADR
  does not define those parameters.

## Rejected alternatives

- **A separate sync sidecar process.** A third process could collect timestamps from each
  sender and publish a match table. It adds a process to start, monitor and stop, and a
  second channel that can drift from the frame data. The frame identity belongs next to the
  frame, in the slot that holds it.
- **An Orbbec-only fork of the protocol.** A second wire format for one camera family doubles
  the parity tests, the golden fixtures and the reader code in Python and C++. The slot meta
  is generic: any multi-stream sender can use it.
- **One RGBA8 atlas per cook.** Instance A could pack all streams into one texture so that
  one segment carries one consistent frame. Depth is a float or 16-bit stream. Packing it into
  8-bit channels loses precision, and the atlas forces one resolution and one format on all
  streams.
- **Append fields to v0.5 and keep the old magic.** A v0.5 receiver would read the new
  segment with the old offsets and get wrong handles. A new magic makes the mismatch a named
  error instead of undefined reads.

## Consequences

**Positive:**

- A receiver can read which frame each slot holds and when the sender published it.
- Phase 6 slot selection becomes possible: a receiver can pick, in each stream, the slot
  whose `producer_frame` matches the other streams, instead of always the latest slot.
- The 64-bit `version` access is aligned and well defined.
- A same-process loop-back fails with a clear message before any CUDA call, on all three
  receivers.

**Negative / trade-offs:**

- v0.5 and v0.6 peers cannot talk. Both directions give a named error and do not crash: a
  v0.6 receiver names a v0.5 sender, and a v0.5 receiver rejects the "CIP6" magic as a bad
  magic.
- cuda-link 1.x users must upgrade the sender and the receiver together.
- Each slot is 32 bytes larger (528 instead of 433 bytes for 3 slots). The cost is
  negligible; the segment is a control packet with no pixel data.
- The golden fixtures in `cpp_top/tests/fixtures/` and the parity tests change with the
  layout.

## Reopen condition

Revisit if a receiver needs per-slot data that does not fit in 32 bytes, if `meta_flags`
gets its first defined bit, or if a sender on another OS cannot provide a process ID (then
`sender_pid` 0 becomes the normal case and the 201 fallback becomes the main check).
