# cuda-link-native

Optional native add-on that accelerates **cuda-link's `Importer` consumer wait
path**. When a Python consumer has drained the current frame and is waiting for
the next one, the pure-Python path spins then falls into a `time.sleep(0.0001)`
loop (`Importer._wait_for_slot`) — measured 136-286 µs p50 cross-process (see
`../docs/BENCHMARKS.md`). This package moves that wait into one native call that
spins on `cudaEventQuery` then blocks on the existing Win32 doorbell event —
target p50 < 10 µs, p95 < 50 µs.

This package **relaxes cuda-link's pure-Python rule** ([ADR-0006](../docs/adr/0006-stay-pure-python-no-rust.md))
on purpose — the same escape hatch [`cuda-link-spout`](../spout/README.md)
exercises. Unlike spout, this module needs **no CUDA Toolkit or SDK at build
time**: it resolves `cudaEventQuery` at runtime from whatever cudart the host
process has already loaded (`GetModuleHandleW` + `GetProcAddress`, never
`LoadLibraryW`) — so building it only requires a C++17 compiler (MSVC) + CMake.

The core `cuda_link` wheel is unaffected if this package is absent: `Importer`
falls back to its existing Python wait automatically
(`ImportPolicy.wait_backend="auto"`, env `CUDALINK_WAIT_BACKEND`).

## Architecture

```
Importer._wait_for_slot            (pure Python, cuda_link core)
        │  depends only on …
        ▼
WaitBackend  (Protocol)            ← seam (ADR-0001 port-adapter pattern)
   ├── FakeWaitBackend             in-memory; drives all CI tests, no GPU
   └── _NativeWaitBackend          wraps the compiled _native_waiter module
                                    (cudart resolution + spin/block state
                                    machine + Win32 doorbell wait), Windows-only
```

Everything except the actual spin/block timing is behind `WaitBackend`, so the
counter-translation and timeout behavior in `Importer._wait_for_slot` is
testable on any machine via `FakeWaitBackend`.

## Build

Windows only:

```bat
pip install ./native
```

On any other platform, or if no C++ toolchain is found, `pip install` still
succeeds — it produces a pure-Python wheel and the native backend raises a
clear `RuntimeError` only if actually used without having been built.

## Activation

Nothing to configure by default: when both `cuda-link` and `cuda-link-native`
are installed on Windows and a CUDA runtime is already loaded in-process,
`ImportPolicy.wait_backend="auto"` (the default) picks the native backend
automatically the next time an `Importer` connects, and forces the Win32
doorbell on for that connection (required — without it the native backend
cannot beat the ~1 ms Windows sleep-loop floor either). Force a specific
backend via `CUDALINK_WAIT_BACKEND=python|native|auto`.

## Testing without a compiler or GPU

```bat
pip install -e .[test]
pytest native/tests -m "not requires_native"
```

Runs entirely against `FakeWaitBackend` — no CUDA, no compiled extension, no
Windows-only APIs exercised.
