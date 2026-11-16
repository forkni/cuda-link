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
- `producer_frame 0` and `stream_id 0` on every slot: expected in this release. No sender
  fills these fields yet.
- `write_idx 0`: the sender made the segment but has not published a frame.
- `shutdown 1`: the sender has stopped.

The tool is built with the TOPs. The `--file` form also works on Linux.

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
- Debug-log file locations and Info CHOP / Info DAT channel reference.
