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

Run the Ninja preset from an MSVC developer environment (`vcvars64.bat`):

```bat
cmake --preset ninja-tooling
cmake --build --preset ninja-tooling
```

To build against a different CUDA toolkit, add `-DCUDAToolkit_ROOT=<path>` to the configure
command.

## Code rules

- No C++ exception crosses the plugin ABI: every TD entry point is fenced (ADR-0009,
  mitigations). The empty `catch (...) {}` fences are intentional and carry a `NOLINT`.
- clang-tidy runs only on `cpp_top/src/`. The vendored TD headers in `cpp_top/vendor/` are a
  primary source and are never linted or reformatted.

## To be written

- Topology: one sender TD instance, one receiver TD instance.
- Same-process limitation and its error message.
- Windows display driver model (WDDM) IPC caveat (ADR-0004).
- Teardown grace period (the 100 ms hitch on Active off or a format change).
- Trust model for named shared memory.
- Debug-log file locations and Info CHOP / Info DAT channel reference.
