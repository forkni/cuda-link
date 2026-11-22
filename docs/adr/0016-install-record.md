# ADR-0016: Install record of the C++ TOPs system install

**Status**: Accepted
**Date**: 2026-11-22
**Applies to**: `src/cuda_link/td_plugin.py`, `scripts/install_cpp_top.py`,
`scripts/install_td_library.py`, `cpp_top/CMakeLists.txt`, root `CMakeLists.txt`,
`scripts/check_wheel_plugins.py`, and the `.tox` reader `CUDAIPCExtension.native_tops_status`.

---

## Context

The C++ Custom TOPs (`CudaLinkOutTOP.dll`, `CudaLinkInTOP.dll`) are installed once per machine
in `<Documents>\Derivative\Plugins\CudaLink\`, the system install. Since 2026-11-17
`scripts/install_cpp_top.py` wrote a `VERSION.txt` there, and since #130 the `.tox` Info page
reads it to show the `Nativetops` row and to warn when the DLLs and the `.tox` differ.

Three gaps were left:

- The file had no build sha. Two builds of the same version (for example a `-dirty` dev build
  and the release build) looked the same, while the DLL Info page showed the difference.
- Only `install_cpp_top.py` wrote it, from a local `cpp_top/build`. Since #129 the cp311 release
  wheel carries both DLLs, but `install_td_library.py` did not install them, so a user of the
  wheel had to find the DLLs and copy them by hand.
- The writer lived in a script and the reader in the `.tox`, with no test that they agree.

## Decision

1. **One writer module.** `cuda_link.td_plugin` holds the system install: the Plugins folder
   location (`documents_dir`, `default_plugins_dir`), `td_running`, the `InstallRecord`
   (`text`, `parse`), `make_record`, `write_record`, `read_record`, `install`, `status`,
   `uninstall`. It imports nothing from the package, so the two installer scripts load it by
   path and need no installed `cuda_link`. It has no TD twin (`CANONICAL_ONLY` in
   `scripts/sync_td_wrapper.py`).
2. **The record format.** `VERSION.txt`, one field per line:

   ```text
   cuda-link 2.0.0
   build ba5e5ab
   protocol v0.6 (magic 0x43495036)
   td header OP_CommonAPIVersion 1
   cuda toolkit 12.6
   installed 2026-11-22T10:00:00 from <source folder>
   sha256 CudaLinkOutTOP.dll <hex>
   sha256 CudaLinkInTOP.dll <hex>
   ```

   The first line stays `cuda-link <version>` (the contract with records written before this
   ADR). A reader ignores lines it does not know, and a missing line reads as `unknown`, so
   lines can be added later.
3. **Build info sidecar.** `cpp_top/CMakeLists.txt` writes `BUILD_INFO.txt` (`cuda-link
   <version>`, `build <sha>`, `cuda toolkit <x.y>`, `td header OP_CommonAPIVersion <n>`) next
   to the DLLs, from the same values it stamps into them and the vendored TD header. The root `CMakeLists.txt` ships it in `cuda_link/td_plugins/` and refuses
   a sidecar of another version. `check_wheel_plugins.py` checks it in each platform wheel.
   `make_record` takes version, build and CUDA toolkit from it.
4. **Two installers, one step.** `install_cpp_top.py install` copies a local build (or, with
   `--from-wheel` or a `--dll-dir` that is an installed package's `td_plugins/`, the wheel's
   DLLs). `install_td_library.py` does the same after pip for every mode that has a known
   site-packages folder, unless `--no-td-plugins`. When TouchDesigner is running it does not
   copy (TD locks a loaded DLL) and prints the `install_cpp_top.py` command to run later.
5. **Two readers, one contract test.** The `.tox` keeps its own small reader in
   `CUDAIPCExtension.py`, because the `.tox` can run without the package.
   `tests/td/test_install_record_contract.py` writes a record with `td_plugin` and reads it
   with `native_tops_status`.

## Rejected alternatives

- **The `.tox` imports `cuda_link.td_plugin`.** The `.tox` must show the row in embedded mode,
  where the package is not importable.
- **Read the build sha from the DLL bytes at install time.** The sha is a string among others
  in `.rdata`; finding it needs a pattern that breaks when the Info page text changes. The
  sidecar is plain text written from the same CMake variable.
- **Keep `install_td_library.py` out of the DLLs.** Users of the wheel would keep two install
  steps, and the Python side and the TOPs could differ in version without a warning until the
  `.tox` runs.
- **JSON for the record.** The existing text form is already read by the `.tox` and by people
  (`install_cpp_top.cmd status`); one field per line keeps both simple.

## Consequences

- A system install from this version on carries the build sha, and the `.tox` row shows
  `<version> (build <sha>)`.
- A wheel built with `CUDALINK_TD_PLUGIN_DIR` from a `cpp_top` build without `BUILD_INFO.txt`
  fails at configure time: configure and build `cpp_top` again from the same tree.
- `install_td_library.py` now writes to `<Documents>\Derivative\Plugins\CudaLink\` by default.
  `--no-td-plugins` restores the old behavior; `--plugins-dir` points it elsewhere.
- A change to the record format changes `td_plugin.py` and, when the `.tox` reads the changed
  line, `CUDAIPCExtension.native_tops_status`. The contract test fails when they differ.

## Reopen condition

- TouchDesigner gains a supported way to install or version plugins (for example a package
  manager) that replaces the vendor folder.
- The `.tox` needs a field that changes the first line.
