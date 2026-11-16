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

`protocol_dump` prints a segment's header, slot handles, shutdown flag, metadata and
timestamp. It also prints what a new receiver would get from `acquire_slot`. It opens the
segment read-only, so it never changes the state of a running sender.

```bat
protocol_dump <Ipcmemname>
protocol_dump --full <Ipcmemname>
protocol_dump --file cpp_top\tests\fixtures\ring_n3.bin
```

- "cannot open segment ... error 2": no sender has created that name. Compare the
  `Ipcmemname` of the sender and the receiver.
- `magic ... (MISMATCH)`: the sender uses a different protocol version.
- `write_idx 0`: the sender made the segment but has not published a frame.
- `shutdown 1`: the sender has stopped.

The tool is built with the TOPs. The `--file` form also works on Linux.

## Code rules

- No C++ exception crosses the plugin ABI: every TD entry point is fenced (ADR-0009,
  mitigations). The empty `catch (...) {}` fences are intentional and carry a `NOLINT`.
- clang-format covers `cpp_top/src/`, `cpp_top/tests/` and `cpp_top/tools/`. clang-tidy
  runs only on `cpp_top/src/`. The vendored TD headers in `cpp_top/vendor/` are a
  primary source and are never linted or reformatted.

## To be written

- Topology: one sender TD instance, one receiver TD instance.
- Same-process limitation and its error message.
- Windows display driver model (WDDM) IPC caveat (ADR-0004).
- Teardown grace period (the 100 ms hitch on Active off or a format change).
- Trust model for named shared memory.
- Debug-log file locations and Info CHOP / Info DAT channel reference.
