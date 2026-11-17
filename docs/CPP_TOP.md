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
- Start after the project opens depends on the install form:
  - System Plugins folder (Custom OP, the main form): an Out TOP starts on its own, also
    when nothing pulls it (test R6: publishing 12-14 s after TD started).
  - CPlusPlus TOP with the Plugin Path parameter (project form): an Out TOP that nothing
    pulls does not start. TD loads the DLL only on the op's first cook, and `cookOnStart` /
    `cookEveryFrame` take effect only after that load. Cook each Out TOP once when the
    project starts, for example `op('out0').cook(force=True)` in an Execute DAT `onStart`.
    The sync kit's run logger does this for every C++ TOP in its COMP. After one cook the
    Out TOP cooks every frame.
  - The In TOP is not affected when the network reads it (the sync rule does).

## Install: system Plugins folder

This is the main form. The ops then show in the OP Create dialog (Custom tab) as
"CUDA Link Out" and "CUDA Link In".

```text
<Documents>\Derivative\Plugins\
└── CudaLink\              one folder, nothing else in it
    ├── CudaLinkOutTOP.dll  opType Cudalinkout
    ├── CudaLinkInTOP.dll   opType Cudalinkin
    └── VERSION.txt         cuda-link version, protocol, TD header API, CUDA toolkit, DLL hashes
```

- Install: close TouchDesigner, then run `install_cpp_top.cmd` (or
  `python scripts\install_cpp_top.py install --dll-dir <folder with the two DLLs>`). The
  script finds the real Documents folder, also when OneDrive moves it. It refuses while
  TouchDesigner runs.
- First start: TD asks to approve each DLL. It asks again after each DLL change. TD writes the
  approvals to `Plugins.json` in the `Plugins` folder, so the folder must be writable.
- Keep one copy on a machine. With a system copy and a copy in a project's `Plugins/` folder,
  TD uses the system copy and shows a modal "Plugin Load Error" for the project copy. The start
  waits until someone clicks OK (test R4).
- Upgrade: close TD (TD locks a loaded DLL), install again, and approve the prompts.
- Uninstall: `install_cpp_top.cmd uninstall` moves the folder to
  `<Documents>\Derivative\CudaLink.removed-<time>`. Do not only rename the folder inside
  `Plugins`: TD loads every subfolder (test R1). Saved projects keep their nodes and parameter
  values; the nodes show an error until the DLLs are back.
- `install_cpp_top.cmd status` shows the installed copy and its `VERSION.txt`, and warns about
  other CUDA Link DLLs under `Plugins`.
- Versions: TD 2025.33230 does not check the plugin major or minor version (test R2). A
  breaking parameter or protocol change therefore needs a new opType. Parameter names and menu
  tokens stay as they are.

## Install: one project's Plugins folder (portable form)

- Use this form only on a machine that has no system copy (see above).
- Put `CudaLinkOutTOP.dll` and `CudaLinkInTOP.dll` in the `Plugins/` folder next to the
  `.toe`, or point a CPlusPlus TOP's Plugin Path at a DLL. An Out TOP in this form needs the
  `onStart` cook (see "Topology").
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

The Out TOP also copies its input to its own output (the pass-through copy), so that it
shows the image in the network. On the dev box (RTX 4090) this copy costs about 14 µs CPU
(`cudaMemcpy2DToArrayAsync`) and 8 µs GPU for each stream and frame. A parameter to skip
the copy was considered and rejected (2026-11-17). Reopen it only if a show stream, for
example 4K float32, makes the copy expensive.

## In TOP parameters

All parameters are on the page "CUDA Link".

| Name | Label | Type | Default | Use |
|---|---|---|---|---|
| `Ipcmemname` | IPC Mem Name | String | `cudalink_ipc_Python>>TD` | Name of the shared-memory segment. Use the name of the sender. |
| `Active` | Active | Toggle | On | Off closes the segment. |
| `Debug` | Debug | Toggle | Off | Writes the debug log and fills the GPU timing channels. |
| `Framewaitms` | Frame Wait (ms) | Integer, 0 to 8 | `3` | On a cook with no new frame, wait for the sender publish for this time. 0 turns the wait off. |

About `Framewaitms`:

- The sender and the receiver both cook at the display rate. Their phase drifts. When the
  receiver cooks just before the sender publishes, it shows the last frame again, and in
  the next cook it skips one frame.
- The wait turns these duplicate frames into new frames that come a little later. In
  2026-11-17 sweeps on the dev box (1280×720 RGBA8, one stream), the duplicate frames
  were 4 % to 6 % at 0 ms and about 1 % at 3 ms.
- The cost depends on the phase. When the receiver cooks just before the publish, each
  cook waits for it. The mean In TOP cook time was 0.2 ms to 1.3 ms at 3 ms and 4 ms,
  against 0.2 ms at 0 ms.
- The default 3 ms gives the same wait as the Phase 5 sync kit setting (0.19 %
  mismatches). That setting was 4, but the older wait code waited about 1 ms less.
- The wait is in steps of 1 ms. After each step, the TOP reads `write_idx` again. Thus two
  consumers of one segment (two In TOPs, or an In TOP and a Python receiver with
  `CUDALINK_DOORBELL`) can both use the wait. The sender signal wakes only one waiter.
- One doorbell waiter per `Ipcmemname` is best. With an In TOP and a Python receiver on one
  name, the In TOP is not affected. The Python receiver misses the signal on about 3 % of
  frames and then waits its full `wait_for_doorbell()` timeout (2 ms in the examples).
  Measured: Python p99 latency 0.09 ms alone, 2.7 ms shared; no lost frames ("Live matrix
  v0.6", P2).
- The wait can be up to about 1.5 ms longer than the value. The wait blocks the TD main
  thread, so increase the value only if the duplicates stay high.
- The TOP skips the wait when it cannot get a frame: the next publish is not expected within
  the budget (a sender slower than TD, for example a 30 fps camera into 60 fps), or the newest
  frame is older than two sender periods (a paused sender). The period comes from the slot
  meta `host_timestamp` of the shown frames (`cpp_top/src/core/frame_wait.h`,
  `cadence_wait_useful`). Until four periods are measured, the TOP waits as before. The Info
  CHOP `frame_wait_skipped` counts the skips. The `.tox` Receiver has the same wait and guard.
- Before 2026-11-17 the default was 0, and each value waited about 1 ms less than set.

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
| `frame_wait_skipped` | Cooks with no new frame that skipped the `Framewaitms` wait: the next publish was not expected within the budget, or the sender was paused |

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

## Live matrix v0.6 (2026-11-17)

Dev box: RTX 4090, TD 2025.33230 Non-Commercial, two TD instances (A sends, B receives),
60 FPS, 1280×720 rgba8 Noise source, Debug off, 60 s per cell. DLLs built from 00d7406.

Pixel check: A writes a 4×4 patch at the bottom-left with R = `absFrame % 256`. Both TD
senders stamp the same `absFrame` as `producer_frame`. B reads one pixel of the patch per cook
and compares it with `producer_frame % 256`. The Python sender has no patch. It sends solid
colors, 30 frames each, and stamps `write_idx`. B checks each color against `producer_frame`.

| Cell | Sender (A) | Receiver (B) | Pixel check | Duplicate % | Gaps | Latency p50 / p95 |
|---|---|---|---|---|---|---|
| C0 | Out TOP | In TOP, Framewaitms 3 | 3593/3593 | 0.42 | 17 | 0.09 / 3.0 ms |
| C1 | `.tox` Sender (doorbell on, see notes) | In TOP, Framewaitms 0 | 3595/3595 | 2.87 | 101 | 15.4 / 16.4 ms |
| C1 | `.tox` Sender (doorbell on, see notes) | In TOP, Framewaitms 3 | 3595/3595 | 1.03 | 29 | 0.14 / 15.7 ms |
| C2 | Out TOP | `.tox` Receiver | 3598/3598 | 5.25 | 188 | 16.3 / 16.8 ms |
| C3 | Out TOP | Python `Importer` | 3588/3589, rerun 3596/3596 | n/a | 13, rerun 6 | 0.83 / 1.45 ms |
| C4 | Python `Exporter`, doorbell off | In TOP, Framewaitms 3 | 119 color runs, 0 wrong | 3.95 | 78 | 8.0 / 15.8 ms |
| C5 | Python `Exporter`, `CUDALINK_DOORBELL=1` | In TOP, Framewaitms 3 | 118 color runs, 0 wrong | 2.67 | 15 | 4.8 / 12.5 ms |
| P1 | `.tox` Sender, `CUDALINK_DOORBELL` unset (new default) | In TOP, Framewaitms 3 | 3597/3597 | 0.89 | 34 | 0.15 / 15.7 ms |
| P1 | Python `Exporter`, `CUDALINK_DOORBELL` unset (new default) | In TOP, Framewaitms 3 | 119 color runs, 0 wrong | 2.67 | 31 | 5.3 / 12.8 ms |
| P2 | Out TOP | In TOP, Framewaitms 3, shared with a Python `Importer` (doorbell on) | 3596/3596 | 0.33 | 12 | 0.93 / 3.6 ms |
| P2 | Out TOP | Python `Importer` (doorbell on), shared with the In TOP above | 3718 frames, 0 bad | n/a | 4 | 0.019 / 0.070 ms, p99 2.7 ms |
| F16 | Out TOP, rgba16float | In TOP, Framewaitms 3 | 1199/1199 (20 s) | 0.83 | 10 | 0.10 / 15.1 ms |

Duplicate % is the number of B cooks that show the same `producer_frame` as the cook before,
divided by all B cooks. Gaps are `frame_gap_count` (TD receivers) or the steps in
`producer_frame` above 1 (Python receiver). `last_error` was empty in all cells.

Notes:

- C1: TD A had `CUDALINK_DOORBELL=1` in its environment, so the `.tox` Sender signaled the
  doorbell. The P1 rows use the new default (producer doorbell on, variable unset).
- C3: the first run had one frame whose pixels and `producer_frame` did not agree (delta 9).
  The script did not log which frame it was. A rerun with logging had no mismatch. The cause
  is not known. The 30-minute show-machine run (consolidated plan Phase 5) will show if it
  comes back.
- An 8-bit frame from the Out TOP arrives in Python in BGRA byte order: the R value is at
  channel index 2. Row 0 of the tensor is the bottom row of the TD image.
- In C4 and C5 the Python sender runs on its own 60 FPS clock, not locked to TD. This causes
  the higher duplicate and gap counts. The doorbell (C5) lowers duplicates, gaps and latency.
- `noframe_count` and `rescued_count` in the In TOP Info CHOP advance only while Debug is on.
  With Debug off they stay 0, also when the In TOP shows a repeat frame.
- P2: the same Python receiver alone on the name had p99 latency 0.088 ms and 23 waits that
  timed out while a frame was ready. Shared with the In TOP, it had 118. The Importer used the
  native wait backend in both runs.
- An Out TOP that nothing pulls cooks only once. After one `cook(force=True)` it cooks every
  frame (see "Topology"). The dev-box scenes load the DLLs through the Plugin Path parameter.

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
- Keep depth at native resolution in Mono32Float. Mono16Float has steps of 2 mm from 2 to
  4 m and 4 mm from 4 to 8 m, and the camera gives 1 mm.
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
- you change `Ipcmemname` while the Out TOP is active (the old segment is closed);
- you change `Numslots`, or the input changes resolution or pixel format (the ring is made
  again, and the old buffers are freed after the wait).

The first allocation does not wait: there are no old handles. Change these parameters and
the format before a show, not during it.

## Trust model

- The Out TOP makes the segment with `CreateFileMappingW` and the default security. The name
  has no `Global\` prefix, so the segment is in the user's session. Other processes of the
  same user in that session can open it, read it and write to it if they know the name.
  The default security also gives access to SYSTEM and to elevated administrator
  processes.
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

### Measuring cost

Measure the cost of a TOP with `Debug` off:

- Sample `copy_us`, `begin_us` and `end_us` from the Info CHOP over many frames, and
  compare medians.
- On the In TOP, do not use `cook_us` for a cost: it includes the `Framewaitms` wait for
  the producer.
- `Debug` adds GPU timer events to each cook. They make Out TOP `copy_us` about 3.5 times
  larger (67 µs to about 230 µs on the dev box), and the CUDA graph path and the legacy
  path do not carry the same number of them. Compare `bench:` lines only between runs
  on the same path, both with `Debug` on.

With `Debug` off, the CUDA graph path (`CUDALINK_CPP_USE_GRAPHS`) gives no gain on either
TOP. Keep the variable unset (ADR-0010, ADR-0011).
