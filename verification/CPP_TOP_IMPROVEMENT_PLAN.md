# C++ Custom TOP — Improvement Plan

Companion to `CPP_TOP_CUDA_GUIDELINES_VERIFICATION.md` (findings F1–F8 + research-doc
corrections, verified against `feat/cpp-custom-top` @ `9162a5f`). This plan turns every
finding into a scheduled, verifiable work item.

**Guiding principle**: the verification found no correctness bugs, so nothing here is an
emergency. Ordering is chosen so that (1) enforcement tooling lands first and locks in the
current high conformance before any code churn, (2) zero-risk polish rides immediately behind
it, (3) anything touching the wire protocol or the Python peer is batched into a single
coordinated protocol revision instead of dribbling out.

Target branch for all phases: `feat/cpp-custom-top` (or its successor after merge).
Effort scale: **S** ≤ ~1 h · **M** ≈ half a day · **L** ≈ 1–3 days.

---

## Phase 0 — Enforcement tooling (do first, before any code churn)

Locks in the current hand-maintained conformance so later phases can't regress it silently.
Addresses **F8**.

### 0.1 `.clang-format` derived from the existing style — S
- Derive the config from the code as written (4-space indent, ~110-column limit, attached
  braces, pointer-left `void* p`), *not* from a stock preset — the goal is a zero-diff or
  near-zero-diff first run, so formatting history stays clean and `git blame` stays useful.
- If the `cpp-lsp` plugin is adopted later (Phase 6), note that its *suggested* config is
  Google-based 4-space/100-col — the derived project `.clang-format` is authoritative and
  must not be replaced by the plugin's template (verified against the plugin README: it
  suggests configs, it does not force them).
- Place at repo root; add `cpp_top/vendor/` to `.clang-format-ignore` (vendored TD headers
  must never be reformatted — the verification report treats them as a primary source).
- **Acceptance**: `clang-format --dry-run --Werror` over `cpp_top/src/` passes with no diff
  (or a single mechanical commit ≤ a few dozen lines applies the residue).

### 0.2 `.clang-tidy` — S/M
- Checks: `cppcoreguidelines-*, modernize-*, bugprone-*, performance-*, readability-*` with
  targeted suppressions where the verification already accepted a deviation:
  - `cppcoreguidelines-pro-type-reinterpret-cast` / `pro-type-const-cast`: suppress *only* at
    the two documented `atomic_ref` sites in `ring_reader.cpp`/`ring_writer.cpp` via
    `// NOLINT(...)` with the existing justification comments — not globally.
  - `cppcoreguidelines-macro-usage`: allow the `CUDALINK_CUDA_CHECK*` family (error-check
    macros with `__FILE__`/`__LINE__` are the sanctioned CUDA idiom; NVIDIA's own
    `helper_cuda.h` does the same).
- Scope: `cpp_top/src/` only; exclude `vendor/`. CUDA-specific tidy support is best-effort
  (research doc §4) — this tree is plain C++ linking cudart, so no `--cuda-path` gymnastics
  are needed until a `.cu` file exists.
- Check-set note: this is a strict superset of the set the `cpp-lsp` plugin's template
  suggests (`clang-analyzer-*, modernize-*, performance-*, bugprone-*`) — adopting that
  plugin later (Phase 6) therefore requires no `.clang-tidy` changes.
- **Acceptance**: clean run (warnings-as-errors) on `cpp_top/src/` with the documented
  NOLINTs in place.

### 0.4 Export `compile_commands.json` — S
- Add `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)` to `cpp_top/CMakeLists.txt` (or pass
  `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` in CI/dev presets). This is the substrate every
  clang-based tool needs — clang-tidy standalone, clangd, and all four `cpp-lsp` hooks
  (`cpp-format-on-edit`, `cpp-tidy-on-edit`, `cpp-compile-check`, `cpp-cppcheck`) resolve
  includes through it, including the CUDA and vendored-TD header paths.
- Note: MSVC generators don't emit compile databases — generate with the Ninja preset for
  tooling even if the shipping build stays MSVC.
- **Acceptance**: `compile_commands.json` produced; `clang-tidy -p build cpp_top/src/...`
  resolves `cuda_runtime.h` and the vendored TD headers without manual `-I` flags.

### 0.3 C++ CI job — M
- GitHub Actions, `windows-latest`: (a) format gate, (b) configure + build both DLLs
  (`cmake -B build && cmake --build`), (c) clang-tidy pass, (d) Phase 4 tests once they exist.
- CUDA dependency: full toolkit install is slow on hosted runners — cache the CUDA installer
  step (e.g. `Jimver/cuda-toolkit` with `sub-packages: '["cudart", "nvcc"]'`) or pin a
  cached minimal cudart. Only headers + `cudart.lib` are needed to *compile*; no GPU is
  needed for any current CI step (the protocol core is deliberately CUDA-free, and the DLLs
  only need to link).
- **Acceptance**: CI green on the branch; a deliberately mis-formatted / tidy-violating test
  commit goes red.
- **Dependency**: 0.1, 0.2.

---

## Phase 1 — Zero-risk code polish (F4, F5, F6 + drift-proofing)

No behavior changes on the frame path. Each item is independently committable; land behind
Phase 0 so the new gates verify them.

### 1.1 Remove the dead plain `CUDALINK_CUDA_CHECK` macro (F5) — S
- Only `_BOOL` and `_FATAL` variants have call sites. Delete the plain variant rather than
  annotate it: `void`-returning contexts that later need a check can trivially re-add it,
  and dead macros with hidden `return` control flow are exactly what ES-series guidelines
  say not to keep around speculatively.
- **Acceptance**: grep shows no references; both DLLs build.

### 1.2 Exception fence on `getInfoPopupString` (F6) — S
- Wrap the body in `try { ... } catch (...) {}` in both TOPs, matching the other seven ABI
  callbacks. Two lines each; removes the single inconsistency in the "no exception crosses
  the ABI" invariant.
- **Acceptance**: all ABI entry points that call into TD or allocate are fenced (grep audit).

### 1.3 Debug-log CUDA failures in teardown paths (F4) — S
- In `CudaLinkOutTOP::teardown()`, `CudaLinkInTOP::closeHandles()`/`teardown()`: capture the
  `cudaError_t` from `cudaFree` / `cudaEventDestroy` / `cudaIpcCloseMemHandle` and emit one
  `debugLog()` line per failure (Debug-gated, so production cost is zero). Do **not** latch
  `myError` — teardown failures are non-actionable and must not block the state reset.
- Leave `raii_handles.h` destructors as-is (guards fire on early-return paths where the
  latched `myError` already tells the story; adding logging plumbing to the guards couples
  `common/` to `DebugLogger` for no diagnostic gain).
- **Acceptance**: forced-failure smoke test (destroy an event twice under Debug=on) produces
  a log line; Release-path behavior unchanged.

### 1.4 De-duplicate sender/receiver common code — M *(optional but recommended)*
- The two TOPs currently copy-paste: constructor device/IPC-probe/stream setup, the
  97-frame bench-accumulator block, `checkParamAppend`, `asHandle`, `widen`, and the
  sticky-error `getErrorString`/`getWarningString` bodies. That's the main drift risk the
  audit surfaced (everything else already lives in `common/`/`core/`).
- Extract into `src/common/` (header-only or a small `cudalink_topcommon` static lib beside
  `cudalink_topcore`): `CudaDeviceSession` (ctor probe + stream), `BenchAccumulator`,
  `param_util.h`, `win_util.h`.
- Keep the two `Parameters.{h,cpp}` pairs separate — their parameter surfaces genuinely
  differ (Numslots exists only on the sender) and merging them would trade clarity for LOC.
- **Acceptance**: no observable behavior change (Info CHOP/DAT output identical); diff review
  confirms moved-not-modified; both DLLs build and load in TD.
- **Risk**: low, but this is the one Phase-1 item with real diff surface — do it as its own
  commit after 1.1–1.3.

---

## Phase 2 — Product-limitation UX (F1, F3)

### 2.1 Same-process loopback: detect + document (F1) — M
CUDA IPC handles cannot be opened in the exporting process (verified against official docs),
so Out TOP → In TOP inside one TouchDesigner instance can never connect; today the receiver
just shows a generic `cudaIpcOpenMemHandle failed` badge every cook.
1. **Probe step (do first)**: capture the *actual* error code Windows/CUDA 12.8 returns for a
   same-process open — the docs don't pin it down (candidates: `cudaErrorDeviceUninitialized`,
   `cudaErrorInvalidContext`, `cudaErrorInvalidValue`). Extend the existing `verification/`
   probe-script convention with a small standalone C++ probe or a TD scene; record results in
   `verification/results/`.
2. **Targeted error**: in `openSlotHandlesIfNeeded()`, when the open fails with the probed
   code(s), replace the generic latched error with:
   `"sender and receiver appear to be in the same process -- CUDA IPC cannot loop back within one process; run the sender in a separate TD instance or use the Python peer"`.
   Keep the generic path for all other codes. Optional stronger signal: sender writes its PID
   into a reserved SHM byte range in a future protocol rev (see 3.1) so the receiver can
   detect same-PID *before* attempting the open — defer to 3.1, don't bump the protocol for
   this alone.
3. **Docs**: one paragraph in the user-facing docs (README / HELP_DOC when the C++ TOPs ship)
   stating the limitation and the two supported topologies.
- **Acceptance**: same-process scene shows the targeted message; cross-process operation
  unaffected; docs updated.

### 2.2 `Sleep(100)` teardown hitch (F3) — decision item, default **keep**
- Current behavior blocks TD's main thread ~6 frames at 60 fps on Active-off / name change /
  resolution-format switch — cold paths only, mirroring the Python exporter's identical
  grace period, protecting against the documented imported-handle use-after-free UB window.
- **Default: keep, documented** (already is). Implement the alternative only if hitchless
  live resolution switching becomes a stated requirement.
- Alternative design (recorded for that trigger): deferred-free list — on
  reallocate/teardown, move old `devPtr`/`event` sets plus a `steady_clock` deadline into a
  member queue; drain entries whose deadline passed at the top of subsequent `execute()`
  cooks; destructor drains unconditionally (dtor may block — TD is unloading the node
  anyway). Effort M; adds lifecycle state that must survive Active toggling, hence not free.
- **Acceptance (if implemented)**: no frame-time spike > 1 cook on resolution switch under a
  running receiver; receiver shows no error badge during the switch (same criterion the
  original race fix used).

---

## Phase 3 — Protocol-coordinated hardening (F2, F7 + F1's PID option)

Everything here changes the wire contract, so it ships together as **protocol v0.6**, gated
on the Python side moving in lockstep (`shm_protocol.py` is the declared single source of
truth; golden-byte tests from Phase 4 must exist first to prove parity).

### 3.1 Protocol v0.6 layout revision — L (C++ side S once specified; cost is coordination)
- **8-align `version`**: move it so `atomic_ref<uint64_t>` runs at its required alignment,
  retiring the formally-UB-but-guarded access documented in `shm_layout.h` (F2). Simplest
  layout: `magic u32 | num_slots u32 | version u64 | write_idx u32 | pad u32` — header grows
  20 → 24 bytes, version lands at offset 8.
- **Sender PID field** (from 2.1): one u32 gives receivers a pre-open same-process check.
- Bump `PROTOCOL_MAGIC` (or add an explicit header version byte) so old/new peers fail loud
  with the existing "magic mismatch" path instead of misparsing.
- Update `shm_protocol.py` first, mirror in `shm_layout.h`/`ring_*.cpp`, keep the
  `static_assert(is_always_lock_free)` guards (they're correct regardless of alignment).
- **Acceptance**: golden-byte parity suite (4.1) green on both sides for v0.6; mixed-version
  peers produce the magic-mismatch error, not garbage.
- **Trigger/scheduling**: do NOT ship v0.6 for F2 alone — the current guarded access is safe
  on the shipped x86-64 target. Batch it with the next functional protocol change (PID field,
  new formats, etc.). If no such change appears, revisit at the next major release.

### 3.2 Threat-model note for unverifiable IPC buffer sizes (F7) — S
- No CUDA API exposes the true allocation size behind an imported IPC pointer, so a malicious
  exporter can over-declare dimensions → device-side OOB in the D2D copy. No code fix exists;
  `validateNumSlots`/`validateMetadata` already stop all accidental cases.
- Add a short "Security / trust model" section to `docs/ARCHITECTURE.md`: named SHM +
  IPC handles are attachable by any local process that knows the name; the receiver validates
  structural consistency but must trust the exporter's allocation sizes; do not point
  `Ipcmemname` at untrusted producers.
- **Acceptance**: section exists and is linked from the C++ TOP docs.

---

## Phase 4 — Deferred test infrastructure (CMakeLists' declared future work)

The `core/` layer was built TD-free/CUDA-free precisely to make this cheap; this phase cashes
that in. No GPU needed anywhere.

### 4.1 `topcore_tests` golden-byte parity suite — M/L
- Cross-language fixtures: a small Python script (imports `src/cuda_link/shm_protocol.py`)
  writes golden buffers for representative cases — fresh init header, N∈{2,3,4} slot handle
  writes, metadata for each supported wire format (including `FLAGS_BGRA`/`FLAGS_MONO_ALPHA`),
  publish sequences, shutdown/clear — into checked-in binary fixtures.
- C++ test binary (plain CTest; any single-header framework or bare `assert` main) runs the
  same operations through `ring_writer`/`shm_layout` and byte-compares; runs `ring_reader::
  acquire_slot` against Python-produced buffers and checks classifications (NoFrame /
  NewFrame / Shutdown / VersionChanged, slot math, write_idx==0 edge, version-adoption edge).
- Include the two regression cases the code comments record: shutdown-offset misread when
  layout was built with num_slots=0, and version-visible-before-handles ordering (the latter
  is single-threaded byte-order verification here; the concurrency claim itself stays a
  design argument — that's fine, encode the byte ordering).
- Wire into CI (0.3) on both Windows and a Linux runner (core is OS-free; a Linux leg makes
  the suite fast and catches endianness/UB-adjacent assumptions cheaply).
- **Acceptance**: suite runs in CI; mutating any layout constant on one side only turns CI red.

### 4.2 `protocol_dump` utility — S/M
- Tiny console exe linking `cudalink_topcore`: opens a named SHM segment read-only, prints
  header/metadata/slot-handle hex/shutdown/timestamp. Primary debugging tool for field
  reports ("receiver stuck on Waiting for producer") and the natural harness for 2.1's probe.
- **Acceptance**: builds in CI; documented in the debug section of the C++ TOP docs.

### 4.3 compute-sanitizer CI — deferred, with explicit trigger
- Currently N/A: no kernels, and memcheck adds nothing over the API-return checks for a pure
  cudart-API DLL. **Trigger**: the first `.cu` file (e.g. the anticipated Y-flip kernel).
  When it lands: enable `LANGUAGES CUDA` + `CUDA_ARCHITECTURES` in CMake (per the existing
  CMakeLists comment), add a self-hosted/GPU runner job running memcheck + racecheck with
  `--error-exitcode 1` over a minimal headless harness that exercises the kernel outside TD.
- Record the trigger in CI config as a comment so it isn't forgotten.

---

## Phase 5 — Documentation corrections (research doc + user docs)

### 5.1 Correct the research document — S
1. **CUDA IPC on Windows**: replace the "Linux-only" caveat with the current official wording
   (supported on Windows for compatibility, performance-cost caveat, `cudaDevAttrIpcEventSupport`
   probe). Prevents a future contributor from "fixing" compliant code into the external-
   semaphore API for no reason.
2. **C++ standard**: note that `cpp_top` targets **C++20** (required: `std::atomic_ref`), a
   deliberate upgrade over the doc's C++17 baseline; all recommended C++17 idioms remain in use.

### 5.2 User-facing docs for the C++ TOPs — S/M
- Consolidate into the eventual C++ TOP help doc: same-process limitation (2.1), Windows IPC
  performance caveat, the teardown hitch (2.2), trust model pointer (3.2), Debug-log file
  locations (`%TEMP%\cudalink_{in,out}_top_debug.log`), Info CHOP/DAT channel reference.

---

## Phase 6 — Claude-assisted enforcement (Agent Skills layer)

Source: the July 2026 skills-ecosystem research ("Claude Code Skills for C++/CUDA
Verification, Quality & Guidelines"). Its three Tier-1 building blocks were re-verified
against the live repos for this plan; its central finding stands: **no existing skill
covers CUDA IPC handle lifecycle / zero-copy GPU sharing** — which is exactly the domain
this repo has now verified and documented. This phase (a) adopts the pieces that exist,
(b) authors the missing piece from this repo's own verified invariants.

This is a natural fit here: the repo already runs a skills+hooks workflow
(`.claude/hooks/skill-activation-prompt.sh`, `session-start-skill-reminder.sh`,
`settings.json` custom instructions, `git-commit-enforcer.py`), so the additions below
extend an existing convention rather than introducing one.

> **Security gate for everything in this phase**: skills execute arbitrary code and the
> aggregator ecosystem is demonstrably risky (Snyk "ToxicSkills": prompt injection in 36%
> of skills tested; a 2026 audit of 22,511 skills found 140,963 issues). Rule: install
> only from the three audited repos below, read every `SKILL.md` + bundled script before
> enabling, pin to a reviewed commit (clone, don't track HEAD), and never install
> marketplace/aggregator variants of the same skills.

### 6.1 Install + audit the three verified building blocks — S (~15 min + audit time)
| Piece | Kind | What it adds here | Verified facts |
|---|---|---|---|
| `technillogue/ptx-isa-markdown` → `~/.claude/skills/cuda` | true `SKILL.md` skill | Instant offline lookup of Runtime **and Driver** API semantics (405+107+128 markdown files, ~4.2 MB, SKILL.md ~13 KB always loaded) + `debugging-tools.md` (compute-sanitizer, cuda-gdb) + `nsys-guide.md`/`ncu-guide.md` | Repo confirmed; installs via `cp -r cuda_skill ~/.claude/skills/cuda` |
| `zircote/cpp-lsp` | Claude Code **plugin** (not a skill) | Auto-runs clang-format/clang-tidy/cppcheck/clangd on every C++ edit — turns Phase 0's configs into edit-time enforcement instead of CI-time-only | v0.1.3 (Jan 2026), MIT. Caveat found in verification: README claims "14 automated hooks" but documents four (`cpp-format-on-edit`, `cpp-tidy-on-edit`, `cpp-compile-check`, `cpp-cppcheck`) — audit the hook directory before trusting the larger claim. Young (~3 stars): fork-and-pin rather than depend |
| `awesome-skills/code-review-skill` → `~/.claude/skills/` | true `SKILL.md` skill | Structured review workflow + `reference/cpp.md` (~890 lines: RAII, Rule of 0/3/5, move semantics, exception safety, `noexcept`) — matches the exact rule set §3 of the verification report audited by hand | Repo confirmed: 20+ languages, 21,000+ lines, progressive disclosure (~220-line core), 1.3k stars, MIT |
- Ordering note: `cpp-lsp` is only useful after 0.1/0.2/0.4 exist (it runs *our* configs
  against *our* compile database; its bundled Google-style template is explicitly not
  adopted — see 0.1).
- **Acceptance**: all three audited and pinned; a test edit to a `cpp_top/src/` file
  triggers format+tidy hooks; "Use code-review-skill" on a sample diff loads
  `reference/cpp.md`.

### 6.2 Author the missing `cuda-ipc` skill from this repo's verified invariants — M
The ecosystem gap is this repo's home turf: the verification report already contains the
doc-verified rule set no published skill encodes. Write `.claude/skills/cuda-ipc/SKILL.md`
(progressive disclosure: slim core + `references/`) capturing, at minimum:
1. **Error-checking contract**: `CUDALINK_CUDA_CHECK_BOOL`/`_FATAL` usage, file/line
   capture, sticky-vs-recoverable classification (`myFatal` latch; why `cudaDeviceReset`
   is forbidden in-process with TD).
2. **RAII pairing rules**: `cudaMalloc`↔`cudaFree` vs `cudaIpcOpenMemHandle`↔
   `cudaIpcCloseMemHandle` are *non-interchangeable*; imported events die by
   `cudaEventDestroy`; guard `release()` exactly once on the success path
   (mirrors `raii_handles.h`).
3. **IPC lifecycle invariants**: export requires classic `cudaMalloc` (stream-ordered
   allocations can't produce `cudaIpcMemHandle_t`); `cudaEventInterprocess` requires
   `cudaEventDisableTiming`; same-process open is impossible; shutdown-flag → zero
   handles → grace period → free ordering (imported-handle use-after-free is UB);
   Windows IPC is supported-but-perf-costly, probe `cudaDevAttrIpcEventSupport` first.
4. **Hot-path rules**: non-blocking streams only, GPU-side event sync only, no per-frame
   alloc/sync, import-once-per-version.
5. **TD bracket policy**: what must sit inside `begin/endCUDAOperations()` (interop-array
   access) vs what may not (resource management), with the CudaOpScope pattern.
6. **Wire/atomics rules**: `write_idx`/`version` only via `std::atomic_ref`
   acquire/release; publish-order contract; the alignment caveat and its guards.
- Structure the skill so the *rules* cite the shipped code (`cpp_top/src/...`) as the
  canonical example — the code passed audit; the skill's job is keeping future edits at
  that bar. Wire activation via the existing `.claude/hooks/skill-activation-prompt.sh`
  keyword mechanism (add cuda/ipc/TOP keywords) so it loads when C++ TOP work starts.
- Optional (flagged by the research doc as a publishable gap): once stable, extract a
  project-agnostic variant — either standalone or as a `reference/cuda.md` PR to
  `code-review-skill`. Out of scope for this plan's acceptance.
- **Acceptance**: skill loads on demand in a session touching `cpp_top/`; a deliberately
  wrong test edit (e.g. `cudaFree` on an imported IPC pointer, or an unbracketed interop
  copy) gets flagged by Claude citing the skill rule.

### 6.3 Watch-list with explicit re-evaluation triggers — S (recurring, zero standing cost)
Recorded so the hand-rolled 6.2 layer is retired the moment something better ships:
- **NVIDIA ships a first-party correctness/sanitizer skill** (their current TensorRT-LLM
  set is Python/Triton/CuTe-oriented with performance-only Nsight skills and *no*
  compute-sanitizer skill — a negative result the research doc itself marks as
  name/description-based; re-verify against the live `.claude/skills/` tree before acting
  on it) → prefer it for the sanitizer-orchestration half of 6.2/4.3.
- **A skill appears bundling compute-sanitizer orchestration + CUDA-C++ guidelines** →
  same.
- **Work shifts kernel-heavy** (the `.cu` trigger from 4.3 fires) → add
  `tensormux/kernel-skills` (`debug-cuda-kernel-correctness`, `write-kernel-test-plan`)
  and evaluate the HF `cuda-kernels` skill; also the moment `ptx-isa-markdown`'s
  nsys/ncu guides earn their keep.
- Sources to watch: `VoltAgent/awesome-agent-skills`, `travisvn/awesome-claude-skills`,
  `hesreallyhim/awesome-claude-code`, NVIDIA repos' `.claude/skills/` trees.

---

## Sequencing, dependencies, and estimates

| Order | Item | Effort | Depends on | Risk |
|---|---|---|---|---|
| 1 | 0.1 clang-format | S | — | none |
| 2 | 0.2 clang-tidy | S/M | 0.1 | none |
| 3 | 0.4 compile_commands.json | S | — | Ninja-vs-MSVC generator wrinkle |
| 4 | 0.3 CI (format+build+tidy) | M | 0.1–0.2 | CI CUDA-install flakiness (cache it) |
| 5 | 6.1 install+audit skills trio | S | 0.1–0.4 (cpp-lsp runs our configs) | supply-chain (mitigated: audit+pin) |
| 6 | 1.1–1.3 polish trio | S each | 0.3 (gates) | none |
| 7 | 1.4 de-dup common code | M | 1.1–1.3 | low (pure refactor, TD load-test it) |
| 8 | 6.2 author cuda-ipc skill | M | verification report (done); 6.1 patterns | none (docs-only artifact) |
| 9 | 4.1 golden-byte suite | M/L | 0.3 | none (no GPU) |
| 10 | 4.2 protocol_dump | S/M | — | none |
| 11 | 2.1 same-process UX (probe → error → docs) | M | 4.2 helpful | needs a Windows+GPU probe session |
| 12 | 5.1–5.2 docs | S/M | 2.1 findings | none |
| 13 | 3.2 threat-model note | S | — | none |
| — | 6.3 skills watch-list | S | recurring | none |
| — | 2.2 hitchless teardown | M | **trigger**: hitchless switching becomes a requirement | medium (lifecycle state) |
| — | 3.1 protocol v0.6 | L | 4.1 green; **trigger**: next functional wire change | cross-repo coordination |
| — | 4.3 compute-sanitizer | M | **trigger**: first `.cu` kernel | needs GPU runner |

Items 1–13 are all unconditionally worth doing and sum to roughly **4–6 focused days**; the
trigger-gated items are consciously parked with their activation conditions written down so
they can't silently rot.

## Thresholds that change this plan

- **Golden-byte suite (4.1) finds a real C++/Python divergence** → that fix jumps to the
  front of the queue ahead of everything else; the layering claim ("verbatim transcription")
  is the foundation the C++ TOPs stand on.
- **A second target architecture (ARM64 Windows / TD on ARM) appears** → F2 stops being
  benign-in-practice (the x86-64 unaligned-atomicity argument doesn't transfer); protocol
  v0.6 (3.1) becomes unconditional and immediate.
- **Live reports of torn/gray frames** → per the existing code comment, do *not* restore a
  full-stream sync; add the narrower `cudaEventSynchronize`-on-copy-done fallback and
  re-benchmark — and add that scenario to the probe scripts.
- **TD 2023/CUDA 11.8 leg activates** (the CMake `CUDALINK_TOP_CUDA_MAJOR=11` path) → re-run
  the verification's IPC-doc checks against the 11.8 runtime docs (the Windows-IPC wording
  and `cudaDevAttrIpcEventSupport` exist there too, but confirm before shipping) and add the
  11.8 toolkit to the CI matrix.
- **A published skill supersedes the hand-rolled layer** (Phase 6.3 triggers) → retire the
  overlapping parts of the local `cuda-ipc` skill rather than maintaining both; keep only
  the project-specific invariants (TD bracket policy, wire/atomics rules) that no generic
  skill can know.

## Skills-layer sources (Phase 6)

Re-verified against the live repos for this plan (July 2026):
- [technillogue/ptx-isa-markdown](https://github.com/technillogue/ptx-isa-markdown) — CUDA
  `SKILL.md` + PTX ISA 9.1 / Runtime API 13.1 / Driver API 13.1 markdown reference (~4.2 MB),
  `debugging-tools.md` (compute-sanitizer, cuda-gdb), `nsys-guide.md`, `ncu-guide.md`.
- [zircote/cpp-lsp](https://github.com/zircote/cpp-lsp) — Claude Code plugin, v0.1.3
  (Jan 2026), MIT; hooks `cpp-format-on-edit`/`cpp-tidy-on-edit`/`cpp-compile-check`/
  `cpp-cppcheck` over clangd/clang-format/clang-tidy/cppcheck. (README's "14 hooks" claim
  vs 4 documented — noted in 6.1.)
- [awesome-skills/code-review-skill](https://github.com/awesome-skills/code-review-skill) —
  1.3k★, MIT; 20+ languages, 21,000+ lines, progressive disclosure, `reference/cpp.md`
  (RAII, Rule of 0/3/5, move semantics, exception safety, `noexcept`).
- Research document: "Claude Code Skills for C++/CUDA Verification, Quality & Guidelines"
  (July 2026) — ecosystem survey incl. tensormux/kernel-skills, HF `cuda-kernels`,
  wshobson/agents `systems-programming`, NVIDIA TensorRT-LLM `.claude/skills` inventory
  (negative result on a correctness skill), and the Snyk "ToxicSkills" / Agentman 2026
  security findings cited in the Phase 6 security gate.
