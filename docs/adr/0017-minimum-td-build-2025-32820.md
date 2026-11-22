# ADR-0017: Minimum TD build 2025.32820

**Status**: Accepted
**Date**: 2026-11-21
**Applies to**: the TouchDesigner side of cuda-link 2.0.0: the `.tox` (`TOXES/CUDAIPCLink_v*.tox`),
the example `.toe`, and the C++ Custom TOP DLLs (`cpp_top/`).

---

## Context

Up to 1.14.1 the README said "TouchDesigner 2022.x or later", but nobody tested a 2022 or
2023 build with the current code. Two TD builds were installed on the dev box for 2.0.0:

- **2025.33230**, with a Non-Commercial key. It caps textures at 1280x1280, so 1080p and 4K
  cannot be tested on it.
- **2025.32820**, with a Commercial key. It has no cap.

The two builds differ in the C++ plugin API. `OP_CommonAPIVersion` is 1 in the 2025.32820
headers and 3 in the 2025.33230 headers (33230 adds `saveData`/`loadData` and other virtuals).
`TOPCPlusPlusAPIVersion` is `12 | (OP_CommonAPIVersion << 16)` in both. The vendored headers in
`cpp_top/vendor/td/2025/` have `OP_CommonAPIVersion` 1 and are a subset of the 2025.32820
headers: those add only `parms` pointers and POP fields inside the `reserved[]` space, so the
struct sizes are the same.

Both builds ship Python 3.11 and load the same CUDA 12 runtime (`cudart64_12.dll`). One known
difference: on 2025.32820 the 11-bit float pixel format comes back from `cudaMemory()` as raw
uint8x4 (`td_exporter/TDSender.py`); the Sender already rejects that format.

A `.tox` or `.toe` saved in a newer build opens in an older build only with a "newer build"
warning. So the build that saves the release files sets the oldest build that opens them
cleanly.

## Decision

TD **2025.32820** is the minimum supported build for cuda-link 2.0.0.

- The release `.tox` and the example `.toe` are saved in 2025.32820. `TOXES/RELEASES.tsv`
  records the save build in its `td_build` column.
- The C++ DLLs keep compiling against the vendored 2025.32820 headers. Code does not use the
  2025.33230 API (`OP_CommonAPIVersion` 3) while this ADR stands.
- Releases are tested on 2025.32820 (Commercial, 1080p) and on 2025.33230.

Tested on 2025.32820 for this decision (2026-11-21, `docs/BENCHMARKS.md`, section "`.tox`
Sender on TD 2025.32820 at 1080p"): two 32820 instances at 1920x1080 at the same time, the
`.tox` Sender and Receiver, the 2.0 `.tox` against 1.14.1, and a 32820 Out TOP sending to a
33230 In TOP at 1280x720.

## Rejected alternatives

- **Keep "2022.x or later".** Not tested with the 2.0 code, and the C++ DLLs need TD 2025
  (the CUDA 12 runtime and the 2025 plugin API).
- **2025.33230 as the minimum.** It is the newest build on the dev box, but only with a
  Non-Commercial key, so the release could not be tested above 1280x1280. Saving the release
  files in 33230 would also make every 32820 user see the "newer build" warning.
- **Build the DLLs against the 33230 headers.** The DLLs would gain nothing they use today, and
  an `OP_CommonAPIVersion` 3 DLL is not known to load in 2025.32820.

## Consequences

- README, `docs/TOX_BUILD_GUIDE.md` and `docs/CPP_TOP.md` name 2025.32820 as the minimum.
- Users on 2023.x or older 2025 builds are not supported by 2.0.0. They can stay on 1.14.1,
  whose files were saved in 2025.33230 (`TOXES/RELEASES.tsv` of that release).
- A plugin feature that needs `saveData`/`loadData` or another `OP_CommonAPIVersion` 3 API
  needs this ADR reopened first.

## Reopen condition

- A feature needs the 2025.33230 (or newer) plugin API or a TD feature that 2025.32820 lacks.
- Derivative ends support for 2025.32820, or a TD update changes the plugin ABI.
- A user needs a build older than 2025.32820 and someone can test it.
