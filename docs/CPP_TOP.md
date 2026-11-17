# C++ Custom TOPs (cpp_top)

This page is the user reference for the `CudaLinkOutTOP` and `CudaLinkInTOP` Custom
Operator DLLs (protocol v0.6, cuda-link 2.0.0.dev0).

## Topology

One TouchDesigner (TD) instance sends, and a different TD instance receives:

```text
TD instance A                                TD instance B
Orbbec TOP -> Select TOP -> Out TOP  ==SHM==> In TOP -> Info CHOP -> sync rule -> Switch TOP
             (one Out TOP per stream)         (one In TOP per stream)
```

- Each stream has its own Out TOP in A and its own In TOP in B, with the same `Ipcmemname`
  on both sides and a different `Streamid` for each stream.
- The texture stays on the GPU. The named shared memory (SHM) segment holds only the ring
  header, the CUDA IPC handles and the slot meta.
- A and B must be different processes (see "Same-process limitation"). A Python sender or
  receiver (the `.tox`, or `cuda_link` in a script) can be either side.
- A and B must use the same GPU.

## Install: one project's Plugins folder

- Put `CudaLinkOutTOP.dll` and `CudaLinkInTOP.dll` in the `Plugins/` folder of the test
  project only. Do not put them in `Documents/Derivative/Plugins`.
- Remove any older build (protocol v0.5) from `Documents/Derivative/Plugins` first. Both
  builds register the op types `Cudalinkout` and `Cudalinkin`, and two DLLs with the same op
  type collide.
- TD locks a loaded DLL. Close TD before you replace it.
- Keep cuda-link 1.14.1 off TD's Python module path while you test 2.0.0.dev0.
- Give each stream its own `Ipcmemname`. The default names are the same in 1.x and 2.x. A
  v0.5 segment is refused with a magic error; the two protocols do not mix.

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

## Cook order on the receiver

Read `producer_frame` through an Info CHOP with Passive Off, and make it an input of the
expression that uses it. With Passive Off, TD cooks the In TOP before the Info CHOP reads
it, so the value belongs to the frame that the In TOP shows in this cook. With Passive On,
the value can be one frame late. A Python read of `op.info` or a CHOP Execute DAT does not
have this order.

The sync kit (`examples/td_multistream_sync/`) does this: each Info CHOP has Passive Off,
the Execute DAT force-cooks the `sync` CHOP once at frame start, and the Cache TOP and
Switch TOP read `op('sync')` in their expressions.

To check the order: pause A, step A one frame, and look at B in the Performance Monitor.
The Switch TOP index and the shown texture must change in the same B frame.

## Orbbec wiring

- A: one Orbbec TOP for each camera, with device timer sync on. A Select TOP takes the
  color (RGBA8) and the depth (Mono32Float, meters) to two Out TOPs. Three cameras give six
  streams, `Streamid` 0 to 5, for example `Ipcmemname` `cam0_color` to `cam2_depth`.
- Keep depth at native resolution in Mono32Float. Mono16Float has steps of 2 to 4 mm, and
  the camera gives 1 mm.
- B: build the receiver with `build_receiver()` from `examples/td_multistream_sync/`. Its
  README has the steps, the sync rule and the run logger.

## WDDM caveat (ADR-0004)

Windows runs the GPU in the Windows display driver model (WDDM) by default. NVIDIA documents
legacy CUDA IPC (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`) as Linux-only, or Windows in
TCC mode. cuda-link uses it on WDDM. It works in practice, and the tests have shown no error
801, but this is outside NVIDIA's support statement.

- Test again after each driver or CUDA update.
- The `init_note` row of each TOP's Info DAT shows the result of the IPC capability probe
  (`CudaDeviceSession`), with the same WDDM/TCC note as the Python
  `CUDARuntimeAPI.check_ipc_capability()`.
- If the probe or the open of a handle fails after an update, read `last_error` in the Info
  DAT and the debug log.

## Teardown grace period

When the Out TOP frees its GPU buffers, a receiver can still hold imported handles to them.
A use of such a handle is undefined behavior. So the Out TOP does these steps first:

1. It sets the shutdown flag and rings the doorbell event.
2. It writes zeros over the IPC handles in the segment.
3. It waits 100 ms (`kIpcCloseGracePeriodMs`) so that the receiver can close its handles.

The wait is on TD's main thread. Thus TD drops about 6 frames at 60 FPS when:

- you set `Active` Off, or you remove or close the Out TOP;
- the input changes resolution or pixel format (the ring is made again, and the old buffers
  are freed after the wait).

The first allocation does not wait: there are no old handles. Change the format before a
show, not during it.

## Trust model

- The Out TOP makes the segment with `CreateFileMappingW` and the default security. The name
  has no `Global\` prefix, so the segment is in the user's session. Other processes of the
  same user in that session can open it, read it and write to it if they know the name.
- cuda-link does not authenticate the sender. Use it on a machine whose processes you trust.
- The In TOP reads the segment as untrusted input. It refuses a wrong magic, a `num_slots` of
  0 or more than 64, a layout larger than the mapped size, and metadata with a zero size or a
  `data_size` that does not agree with the width, height and format. It shows the reason in
  `last_error`.

## Debug log

With `Debug` on, each TOP appends lines to its own file in the Windows temp folder
(`GetTempPath`, normally `%TEMP%`, that is `C:\Users\<user>\AppData\Local\Temp`):

| TOP | File |
|---|---|
| Out TOP | `%TEMP%\cudalink_out_top_debug.log` |
| In TOP | `%TEMP%\cudalink_in_top_debug.log` |

The file grows while `Debug` is on; delete it when you no longer need it. Every 97 frames the
log has a `bench:` line with the mean cook times. With `Debug` off, no file is written and
the GPU timing channels show 0.
