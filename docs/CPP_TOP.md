# C++ Custom TOPs (cpp_top)

> Status: stub. This page holds the user-facing reference for the `CudaLinkOutTOP` and
> `CudaLinkInTOP` Custom Operator DLLs. Sections marked "to be written" are filled in when
> the work they describe lands.

## Build

The CMake project is `cpp_top/`. Its presets are in `cpp_top/CMakePresets.json`:

| Preset | Generator | Output | Use |
|---|---|---|---|
| `ninja-tooling` | Ninja, MSVC `cl` | `cpp_top/build/` | CI, clang-tidy, clangd (emits `compile_commands.json`) |
| `msvc-release` | Visual Studio 17 2022, x64 | `cpp_top/build_msvc/` | Shipping DLLs from the IDE |
| `core-only` | Ninja, default compiler | `cpp_top/build_core/` | Protocol core, `topcore_tests` and `protocol_dump` only; no CUDA, no TD headers; any OS (CI Linux leg) |

Run the Ninja preset from an MSVC developer environment (`vcvars64.bat`):

```bat
cmake --preset ninja-tooling
cmake --build --preset ninja-tooling
```

To build against a different CUDA toolkit, add `-DCUDAToolkit_ROOT=<path>` to the configure
command.

## Tests

`topcore_tests` checks the protocol core (`cpp_top/src/core/`) against golden SHM images
in `cpp_top/tests/fixtures/`. `scripts/gen_golden_shm.py` writes those images through
`src/cuda_link/shm_protocol.py`, which is the source of truth for the wire layout. The test
runs the same writes through `ring_writer` and compares the bytes. Then it runs
`acquire_slot` on the Python images (NoFrame, NewFrame, Shutdown, VersionChanged).

```bat
ctest --preset ninja-tooling
```

If you change the wire layout, change `shm_protocol.py` first. Then run
`python scripts/gen_golden_shm.py`, update the C++ side, and make both test suites pass.
`tests/core/test_golden_shm_fixtures.py` fails while the fixtures are stale.
`tests/core/test_cpp_protocol_parity.py` fails when a constant differs between the two
sides.

## Debugging: protocol_dump

`protocol_dump` prints a segment's header (with `sender_pid`), the slot handles, one
slot-meta line per slot (`host_ts`, `producer_frame`, `source_ts_us`, `stream_id`, `flags`),
the shutdown flag and the metadata. It also prints what a new receiver would get from
`acquire_slot`. It opens the
segment read-only, so it never changes the state of a running sender.

```bat
protocol_dump <Ipcmemname>
protocol_dump --full <Ipcmemname>
protocol_dump --file cpp_top\tests\fixtures\ring_n3.bin
```

- "cannot open segment ... error 2": no sender has created that name. Compare the
  `Ipcmemname` of the sender and the receiver.
- `magic ... (v0.5 sender, cuda-link 1.x: not readable by this build)`: the sender uses
  protocol v0.5. Update the sender to cuda-link 2.x.
- `magic ... (MISMATCH)`: the segment does not hold a cuda-link protocol.
- `sender_pid 0 (unknown)`: the sender did not write its process ID.
- `producer_frame 0` on every slot: the sender writes no frame number. The C++ Out TOP
  writes TD's absolute frame. The Python `Exporter` writes `write_idx` unless the caller
  sets `GpuFrame.producer_frame`.
- `stream_id 0` on every slot: the default. Set `Streamid` on the Out TOP (or
  `GpuFrame.stream_id` in Python) to name the stream.
- `write_idx 0`: the sender made the segment but has not published a frame.
- `shutdown 1`: the sender has stopped.

The tool is built with the TOPs. The `--file` form also works on Linux.

## Out TOP parameters

All parameters are on the page "CUDA Link".

| Name | Label | Type | Default | Use |
|---|---|---|---|---|
| `Ipcmemname` | IPC Mem Name | String | `cudalink_ipc_TD>>Python` | Name of the shared-memory segment. The receiver must use the same name. |
| `Active` | Active | Toggle | On | Off stops the publish and frees the ring. |
| `Debug` | Debug | Toggle | Off | Writes the debug log and fills the GPU timing channels. |
| `Numslots` | Num Slots | Menu (`2`, `3`, `4`) | `3` | Number of ring slots. |
| `Streamid` | Stream ID | Integer, min 0 | `0` | Written to the slot meta `stream_id` of each frame. Give each stream a different value. |

## Slot meta from the Out TOP

Each publish writes these slot meta fields:

- `producer_frame`: TD `OP_TimeInfo::absFrame` of that cook. All Out TOPs that cook in one
  TD frame write the same value.
- `stream_id`: the `Streamid` parameter.
- `host_timestamp`: the publish time in seconds, from `steady_clock`
  (`cpp_top/src/common/clock.h`). On Windows this clock is QPC, the same clock as Python
  `time.perf_counter()`. A receiver on the same machine subtracts it from its own clock to
  get the latency.
- `source_timestamp_us`: always 0 from the Out TOP.

## Info CHOP and Info DAT

Connect an Info CHOP or an Info DAT to a TOP to read these values.

### Out TOP Info CHOP (13 channels)

| Channel | Value |
|---|---|
| `frames` | Frames published in this session |
| `cook_us` | Time of the last cook, in microseconds |
| `copy_us` | Time of the slot copy, in microseconds |
| `begin_us` | Time of `beginCUDAOperations()`, in microseconds |
| `end_us` | Time of `endCUDAOperations()`, in microseconds |
| `write_idx` | Last published `write_idx` |
| `num_slots` | Number of ring slots |
| `gpu_ipc_us` | GPU time of the slot copy and the event record (0 while `Debug` is off) |
| `gpu_pass_us` | GPU time of the pass-through copy (0 while `Debug` is off) |
| `graph_hits` | CUDA graph cache hits (only with `Debug` on and `CUDALINK_CPP_USE_GRAPHS` set) |
| `graph_builds` | CUDA graph captures (same condition as `graph_hits`) |
| `producer_frame` | Last published `producer_frame`, mod 2^24 |
| `stream_id` | Last published `stream_id` |

### In TOP Info CHOP (20 channels)

| Channel | Value |
|---|---|
| `frames` | Frames shown in this session |
| `cook_us` | Time of the last cook, in microseconds |
| `copy_us` | Time of the copy into the output, in microseconds |
| `begin_us` | Time of `beginCUDAOperations()`, in microseconds |
| `end_us` | Time of `endCUDAOperations()`, in microseconds |
| `write_idx` | `write_idx` of the shown frame |
| `read_slot` | Ring slot of the shown frame |
| `num_slots` | Number of ring slots, from the segment header |
| `event_wait_us` | GPU time of the wait on the sender event (0 while `Debug` is off) |
| `gpu_copy_us` | GPU time of the copy into the output (0 while `Debug` is off) |
| `noframe_count` | Cooks with no new frame (only with `Debug` on) |
| `version_changed_count` | Cooks that saw a sender restart (only with `Debug` on) |
| `rescued_count` | Frames that the `Framewaitms` wait got (only with `Debug` on) |
| `graph_hits` | CUDA graph cache hits (only with `Debug` on and `CUDALINK_CPP_USE_GRAPHS` set) |
| `graph_builds` | CUDA graph captures (same condition as `graph_hits`) |
| `producer_frame` | `producer_frame` of the shown frame, mod 2^24 |
| `stream_id` | `stream_id` of the shown frame |
| `latency_us` | Time from the sender publish to the copy (now minus `host_timestamp`), in microseconds. 0 when the sender wrote no timestamp. |
| `frame_gap_count` | Shown frames whose `write_idx` step was more than 1: the ring overwrote frames before this TOP read them |
| `producer_skip_count` | Shown frames whose `write_idx` step was 1 but whose `producer_frame` step was more than 1: the sender skipped TD frames. Not counted when `producer_frame` is 0. |

The slot meta channels (`producer_frame` to `producer_skip_count`) change only after a
successful copy. `cpp_top/src/core/frame_step.h` classifies the step between two shown
frames, and `topcore_tests` tests it. Python has the same rules in
`shm_protocol.classify_frame_step()`.

### Info DAT rows

| Row | Out TOP | In TOP |
|---|---|---|
| `ipc_version` | yes | yes |
| `status` | yes | yes |
| `last_error` | yes | yes |
| `last_error_frame` | yes | yes |
| `init_note` | yes | yes |
| `producer_frame` | yes: last published, full 64-bit value | yes: shown frame, full 64-bit value |
| `source_timestamp_us` | no | yes: shown frame |

### producer_frame in a CHOP channel

A CHOP channel is float32. A float32 holds an integer exactly only below 2^24 (16,777,216).
Thus both Info CHOPs show `producer_frame` mod 2^24 (`INFO_CHOP_FRAME_MOD` in
`shm_protocol.py`, `kInfoChopFrameMod` in C++). At 60 FPS the value wraps after about
78 hours. To compare two channel values `a` and `b`, use `(a - b) mod 2^24`, not `a - b`.
For the full value, read the `producer_frame` row of the Info DAT.

## Matching streams on the receiver

Sender A has several Out TOPs (for example color and depth), each with its own
`Ipcmemname` and `Streamid`. Receiver B has one In TOP for each stream. All Out TOPs that
cook in one TD frame of A write the same `producer_frame`. Thus two frames on B with an
equal `producer_frame` come from the same cook of A.

On B, compare the `producer_frame` channels of the In TOPs:

- Equal values: the frames on the outputs come from the same cook of A.
- Different values: one In TOP shows a newer frame. The difference `(a - b) mod 2^24` is
  the number of TD frames between them.
- Use `stream_id` to make sure that each In TOP reads the stream that you expect.

`frame_gap_count` tells you that B was too slow to read every frame. `producer_skip_count`
tells you that A did not publish in every TD frame.

## Same-process limitation

CUDA IPC cannot open a handle in the process that exported it. A `CudaLinkInTOP` that
reads from a `CudaLinkOutTOP` (or a Python sender) in the same TD instance gets this error:

```text
sender and receiver appear to be in the same process -- CUDA IPC cannot loop back within
one process; run the sender in a separate TD instance or use the Python peer
(cudaIpcOpenMemHandle: cudaErrorDeviceUninitialized)
```

Run the sender and the receiver in two TD instances. To move a texture inside one TD
instance, use a normal TOP wire or a Select TOP.

How the receiver detects it: a probe on 2026-11-16 (CUDA runtime 12.8, driver 13.05,
RTX 4090) opened the process's own handles. `cudaIpcOpenMemHandle` and
`cudaIpcOpenEventHandle` both returned `cudaErrorDeviceUninitialized` (201, "invalid
device context"). The In TOP in TD gave the same result. The error is not sticky: the CUDA
context stays usable. The In TOP always has a current context when it opens handles, so it
reads status 201 at that call as the same-process case.

Since protocol v0.6 the segment header holds the sender's process ID (`sender_pid`). The
receiver (the In TOP, the Python `Importer` and `TDReceiver`) compares it with its own
process ID before it opens any handle, and gives the error above without a CUDA call. The
201 mapping stays as the fallback for a segment with `sender_pid` 0 (unknown).

## Code rules

- No C++ exception crosses the plugin ABI: every TD entry point is fenced (ADR-0009,
  mitigations). The empty `catch (...) {}` fences are intentional and carry a `NOLINT`.
- clang-format covers `cpp_top/src/`, `cpp_top/tests/` and `cpp_top/tools/`. clang-tidy
  runs only on `cpp_top/src/`. The vendored TD headers in `cpp_top/vendor/` are a
  primary source and are never linted or reformatted.

## To be written

- Topology: one sender TD instance, one receiver TD instance.
- Windows display driver model (WDDM) IPC caveat (ADR-0004).
- Teardown grace period (the 100 ms hitch on Active off or a format change).
- Trust model for named shared memory.
- Debug-log file locations.
