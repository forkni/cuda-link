#include "CudaLinkInTOP.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "Parameters.h"
#include "pixel_format_map.h"
#include "../common/cuda_check.h"
#include "../common/cuda_device_session.h"
#include "../common/cuda_op_scope.h"
#include "../common/raii_handles.h"
#include "../common/win_util.h"

using cudalink::common::asHandle;
using cudalink::common::widen;
using cudalink::core::AcquireResult;
using cudalink::core::Metadata;
using cudalink::core::PROTOCOL_MAGIC;
using cudalink::core::SHMLayout;
using cudalink::core::SlotState;
using cudalink::core::acquire_slot;
using cudalink::core::read_magic;
using cudalink::core::read_num_slots;
using cudalink::core::read_version;

namespace {
// Names the Info CHOP/DAT counts returned by getNumInfoCHOPChans()/getInfoDATSize() --
// must stay in sync with the number of `case` labels / index branches in
// getInfoCHOPChan()/getInfoDATEntries() below.
constexpr int32_t kNumInfoCHOPChans = 15; // frames, cook_us, copy_us, begin_us, end_us, write_idx, read_slot,
                                          // num_slots, event_wait_us, gpu_copy_us, noframe_count,
                                          // version_changed_count, rescued_count, graph_hits, graph_builds
constexpr int32_t kNumInfoDATRows = 5;    // ipc_version, status, last_error, last_error_frame, init_note
} // namespace

extern "C" {

DLLEXPORT
void FillTOPPluginInfo(TD::TOP_PluginInfo* info) {
    if (!info->setAPIVersion(TD::TOPCPlusPlusAPIVersion)) return;

    info->executeMode = TD::TOP_ExecuteMode::CUDA;

    TD::OP_CustomOPInfo& customInfo = info->customOPInfo;
    customInfo.opType->setString("Cudalinkin");
    customInfo.opLabel->setString("CUDA Link In");
    customInfo.opIcon->setString("CLI");
    customInfo.authorName->setString("forkni");
    customInfo.authorEmail->setString("forkni@gmail.com");

    customInfo.minInputs = 0;
    customInfo.maxInputs = 0;

    // Required alongside cookEveryFrame=true (getGeneralInfo) or every-frame cooking
    // never kick-starts for a node with no inputs and nothing initially viewing it.
    customInfo.cookOnStart = true;
}

DLLEXPORT
TD::TOP_CPlusPlusBase* CreateTOPInstance(const TD::OP_NodeInfo* info, TD::TOP_Context* context) {
    // A std::bad_alloc (or any other exception) out of the constructor must not cross the
    // ABI back into TD -- report failure as null rather than let it propagate.
    try {
        return new CudaLinkInTOP(info, context);
    } catch (...) {
        return nullptr;
    }
}

DLLEXPORT
void DestroyTOPInstance(TD::TOP_CPlusPlusBase* instance, TD::TOP_Context* context) {
    // static_cast, not a C-style cast -- this is a safe downcast to the concrete type
    // CreateTOPInstance actually returned.
    delete static_cast<CudaLinkInTOP*>(instance);
}

} // extern "C"

CudaLinkInTOP::CudaLinkInTOP(const TD::OP_NodeInfo*, TD::TOP_Context* context) : myContext(context) {
    // Device alignment + IPC-capability probe + stream creation is shared with
    // CudaLinkOutTOP (see cuda_device_session.h) -- the receiver has no need for a
    // priority stream (no Python-exporter-parity concern on this side), hence false here.
    const cudalink::common::CudaDeviceSession session(context, /*highPriorityStream=*/false);
    myStream = session.stream;
    myError = session.error;
    myFatal = session.fatal;
    myInitNote = session.note;

    // CUDA Graphs opt-in (PLAN-005 task #13 / ADR-0011, default OFF). Deliberately the SAME
    // env var the Out TOP reads: its parked path self-disables at frame 3 when set (harmless
    // but noisy, accepted in ADR-0010), and one var enables "the graphs feature" project-wide
    // rather than per-TOP-role -- matching the Python side's single CUDALINK_USE_GRAPHS gate.
    // Read once at construction for the same no-UI-exposure rationale as the sender
    // (CudaLinkOutTOP's ctor comment).
    const char* graphsEnv = std::getenv("CUDALINK_CPP_USE_GRAPHS");
    myGraphsRequested = graphsEnv && graphsEnv[0] != '\0' && graphsEnv[0] != '0';
}

CudaLinkInTOP::~CudaLinkInTOP() {
    // A destructor is implicitly noexcept -- anything escaping here (e.g. a std::string
    // std::bad_alloc from teardown()'s debugLog()/myError plumbing) would call
    // std::terminate() and crash the whole TD host process, not just this plugin. Same ABI
    // fence discipline as the other catch (...) sites in this file.
    try {
        teardown();
        if (myStream) {
            cudaStreamDestroy(myStream);
        }
    } catch (...) { // NOLINT(bugprone-empty-catch) -- deliberate ABI fence, see comment above
    }
}

void CudaLinkInTOP::getGeneralInfo(TD::TOP_GeneralInfo* ginfo, const TD::OP_Inputs*, void*) {
    // The producer is external to TD; nothing in TD's dependency graph (no inputs, no
    // changing parameters most cooks) would otherwise trigger a re-cook when a new
    // frame arrives. Must poll every frame while Active.
    ginfo->cookEveryFrame = true;
}

void CudaLinkInTOP::setupParameters(TD::OP_ParameterManager* manager, void*) {
    // Parameters::setup() builds parameter strings; a std::bad_alloc must not cross the ABI
    // back into TD.
    try {
        Parameters::setup(manager);
    } catch (...) { // NOLINT(bugprone-empty-catch) -- deliberate ABI fence, see comment above
    }
}

// ---------------------------------------------------------------------------
// Parameter-change detection -- the Custom TOP API has no push notification for
// parameter edits, so this diffs the values we care about against last cook's cache.
// ---------------------------------------------------------------------------

void CudaLinkInTOP::checkParameterChanges(const TD::OP_Inputs* inputs) {
    const bool active = Parameters::evalActive(inputs);
    const char* ipcmemnameRaw = Parameters::evalIpcmemname(inputs);
    const std::string ipcmemname = ipcmemnameRaw ? ipcmemnameRaw : "";

    // Refreshed once per cook so other methods pick up the current Debug toggle
    // without needing OP_Inputs threaded through (mirrors the sender side).
    myDebugLog.setEnabled(Parameters::evalDebug(inputs));

    if (myFirstCook) {
        myCachedActive = active;
        myCachedIpcmemname = ipcmemname;
        myFirstCook = false;
        return;
    }

    // Active edge (either direction): full teardown. Off->On must behave as a fresh
    // connection, never resuming stale state; On->Off must free everything immediately.
    if (active != myCachedActive) {
        teardown();
    }
    // Ipcmemname changed while running: drop the current connection so step 1 attempts
    // the new name this very cook (mirrors request_immediate_reconnect() -- there is no
    // backoff timer here to reset, since a Custom TOP's execute() already retries every
    // cook, unlike Python's exponential-backoff retry state machine).
    else if (ipcmemname != myCachedIpcmemname) {
        teardown();
    }

    myCachedActive = active;
    myCachedIpcmemname = ipcmemname;
}

// ---------------------------------------------------------------------------
// Teardown / handle lifecycle
// ---------------------------------------------------------------------------

void CudaLinkInTOP::closeHandles() {
    // Cached graph execs bake the slot device pointers AND slot events this function is
    // about to invalidate -- destroy them first, before either loop below. closeHandles()
    // is the single funnel for both VERSION_CHANGED and teardown(), so no other call site
    // needs to remember this.
    destroyGraphs();
    for (auto* devPtr : mySlotDevPtrs) {
        if (devPtr) {
            // Non-fatal: closeHandles()/teardown() must run to completion and reset state
            // regardless of a single slot's cleanup failure. Logged only (Debug-gated) so a
            // stuck/corrupted handle doesn't silently vanish, but myError is never latched
            // here -- this is not actionable by the user, and latching would block the
            // state reset that follows.
            const cudaError_t err = cudaIpcCloseMemHandle(devPtr);
            if (err != cudaSuccess) {
                debugLog(std::string("closeHandles: cudaIpcCloseMemHandle failed: ") +
                         cudaGetErrorString(err));
            }
        }
    }
    mySlotDevPtrs.clear();
    for (auto* evt : mySlotEvents) {
        if (evt) {
            const cudaError_t err = cudaEventDestroy(evt);
            if (err != cudaSuccess) {
                debugLog(std::string("closeHandles: cudaEventDestroy failed: ") + cudaGetErrorString(err));
            }
        }
    }
    mySlotEvents.clear();
    myHandlesOpen = false;
}

void CudaLinkInTOP::teardown() {
    debugLog("teardown: begin");
    closeHandles();
    if (myShmView) {
        UnmapViewOfFile(myShmView);
        myShmView = nullptr;
    }
    if (myShmHandle) {
        CloseHandle(asHandle(myShmHandle));
        myShmHandle = nullptr;
    }
    if (myDoorbellHandle) {
        CloseHandle(asHandle(myDoorbellHandle));
        myDoorbellHandle = nullptr;
    }
    myLastVersion = 0;
    myLastWriteIdx = 0;
    myLayout = SHMLayout(0);
    myShmMappedSize = 0;
    myStatus = "Waiting for producer";
    debugLog("teardown: complete");
}

void CudaLinkInTOP::destroyGraphs() {
    // cudaGraphExecDestroy (via each entry's CudaGraphExecGuard) is safe without draining
    // in-flight work on myStream -- the driver defers actual release until any queued launch
    // completes -- and it never dereferences the memory operands baked into the graph's
    // nodes, so entries whose TD-owned dst array was already freed are safe to destroy too
    // (they were only ever unsafe to LAUNCH, which the pointer-keyed lookup prevents).
    // Non-fatal by construction: the guard swallows the destroy status, matching the
    // logged-only, never-latched stance of the teardown loops in closeHandles().
    size_t total = 0;
    for (const auto& entries : myGraphCache) {
        total += entries.size();
    }
    if (total > 0) {
        debugLog("destroyGraphs: dropping " + std::to_string(total) + " cached graph exec(s)");
    }
    myGraphCache.clear();
}

bool CudaLinkInTOP::tryGraphedCopy(uint32_t slot, cudaArray_t dstArray, size_t rowBytes, uint32_t height) {
    // Sized lazily (not in openSlotHandlesIfNeeded()) so the graphs-off path never touches
    // this state at all; destroyGraphs() resets it to empty on every handle teardown.
    if (myGraphCache.size() < mySlotDevPtrs.size()) {
        myGraphCache.resize(mySlotDevPtrs.size());
    }

    auto& entries = myGraphCache[slot];
    for (const auto& entry : entries) {
        if (entry.dstArray == dstArray) {
            // Pointer-equality hit: TD handed back the same output array it gave the cook
            // this graph was captured on, so launching is exactly equivalent to re-issuing
            // the waitEvent+memcpy pair with these operands. Residual ABA risk (TD freeing
            // and reallocating a DIFFERENT array at the same address with identical
            // dims/format between cooks) is accepted and documented in ADR-0011 -- the
            // vendor API exposes no generation counter to detect it.
            const cudaError_t st = cudaGraphLaunch(entry.exec.get(), myStream);
            if (st != cudaSuccess) {
                debugLog(std::string("graphs: launch failed: ") + cudaGetErrorString(st));
                myGraphsDisabled = true;
                return false;
            }
            if (myDebugLog.enabled()) {
                ++myGraphHits;
            }
            return true;
        }
    }

    if (entries.size() >= kGraphCacheCapPerSlot) {
        // More distinct output arrays per slot than any plausible swap-chain depth: the
        // pointer-stability assumption this cache rests on has failed for this session.
        // Hard-disable rather than evict -- eviction would silently mask exactly the
        // instability signal this cap exists to surface (and keep building execs against
        // arrays TD may free at any time).
        debugLog("graphs: dst-array cache cap (" + std::to_string(kGraphCacheCapPerSlot) +
                 ") exceeded for slot " + std::to_string(slot) + ", disabling");
        myGraphsDisabled = true;
        return false;
    }

    // Cache miss: capture this cook's real wait+copy pair with BOTH operands baked in, then
    // launch the fresh exec below (capture only RECORDS the ops -- the launch is what
    // actually performs this cook's copy; same-cook capture-then-launch mirrors the Out
    // TOP's tryGraphCopy()). Relaxed mode is mandatory: myStream is handed to TD as
    // TOP_CUDAOutputInfo::stream for its Vulkan-interop ordering, and Thread-Local/Global
    // capture would abort the process if any other thread touched the stream mid-capture.
    cudaError_t st = cudaStreamBeginCapture(myStream, cudaStreamCaptureModeRelaxed);
    if (st != cudaSuccess) {
        debugLog(std::string("graphs: stream_begin_capture failed: ") + cudaGetErrorString(st));
        myGraphsDisabled = true;
        return false;
    }

    // MUST pass cudaEventWaitExternal (not flags=0) to capture a wait on an event recorded
    // OUTSIDE the graph -- the cross-process analog of the Python exporter's
    // _GRAPH_EVENT_WAIT_EXTERNAL. Waiting on an IMPORTED interprocess event inside capture
    // is the one step the Python side never proved (it only records one); this first build
    // attempt doubles as the runtime probe, and any rejection latches the fallback. The
    // captured-wait precondition (event already in recorded state) is naturally satisfied:
    // the producer records mySlotEvents[slot] before acquire_slot() ever classifies that
    // slot as a NewFrame.
    st = cudaStreamWaitEvent(myStream, mySlotEvents[slot], cudaEventWaitExternal);
    if (st == cudaSuccess) {
        st = cudaMemcpy2DToArrayAsync(dstArray, 0, 0, mySlotDevPtrs[slot], rowBytes, rowBytes, height,
                                      cudaMemcpyDeviceToDevice, myStream);
    }
    if (st != cudaSuccess) {
        debugLog(std::string("graphs: capture body failed: ") + cudaGetErrorString(st));
        cudaGraph_t abandoned = nullptr;
        cudaStreamEndCapture(myStream, &abandoned); // must still end capture even on failure
        if (abandoned) {
            cudaGraphDestroy(abandoned);
        }
        myGraphsDisabled = true;
        return false;
    }

    cudaGraph_t templateGraph = nullptr;
    st = cudaStreamEndCapture(myStream, &templateGraph);
    if (st != cudaSuccess || !templateGraph) {
        debugLog(std::string("graphs: stream_end_capture failed: ") + cudaGetErrorString(st));
        myGraphsDisabled = true;
        return false;
    }

    size_t nodeCount = 0;
    cudaGraphGetNodes(templateGraph, nullptr, &nodeCount);
    if (nodeCount != 2) {
        // Shape validation before instantiating, mirroring the Out TOP: anything but the
        // expected [EventWaitNode, MemcpyNode] pair means the capture picked up extra work
        // (e.g. a driver difference) and the graph no longer represents just this copy.
        debugLog("graphs: unexpected node count " + std::to_string(nodeCount) + " (expected 2)");
        cudaGraphDestroy(templateGraph);
        myGraphsDisabled = true;
        return false;
    }

    cudaGraphExec_t graphExec = nullptr;
    st = cudaGraphInstantiate(&graphExec, templateGraph, 0);
    cudaGraphDestroy(templateGraph); // the exec owns its own copy; the template is done
    if (st != cudaSuccess) {
        debugLog(std::string("graphs: instantiate failed: ") + cudaGetErrorString(st));
        myGraphsDisabled = true;
        return false;
    }

    GraphCacheEntry entry;
    entry.dstArray = dstArray;
    entry.exec.reset(graphExec);

    st = cudaGraphLaunch(entry.exec.get(), myStream);
    if (st != cudaSuccess) {
        debugLog(std::string("graphs: launch failed: ") + cudaGetErrorString(st));
        myGraphsDisabled = true;
        return false; // entry's guard destroys the exec on scope exit
    }

    entries.push_back(std::move(entry));
    if (myDebugLog.enabled()) {
        ++myGraphBuilds;
    }
    // Diagnostic-only pointer print (cache-key identity for live A/B logs); the integer is
    // never dereferenced or converted back to a pointer.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto dstAddr = reinterpret_cast<uintptr_t>(dstArray);
    debugLog("graphs: built for slot " + std::to_string(slot) + " dst=" + std::to_string(dstAddr) + " (" +
             std::to_string(entries.size()) + "/" + std::to_string(kGraphCacheCapPerSlot) + ")");
    return true;
}

// This function's CUDA calls (cudaIpcOpenMemHandle, cudaEventCreate, cudaIpcOpenEventHandle)
// never touch a TD-owned Vulkan-interop cudaArray, so they don't need a
// begin/endCUDAOperations() bracket -- same reasoning and same primary-source confirmation
// (TD's vendored CudaTOP sample) as documented at CudaLinkOutTOP.cpp's reallocate().
// execute()'s Step 5 (the actual interop copy) is the only place that needs -- and has --
// the bracket.
bool CudaLinkInTOP::openSHM(const char* name) {
    if (myShmView) {
        return true; // already open
    }
    // widen() is a naive byte-widening, not a general UTF-8 decoder -- correct for the
    // ASCII SHM names this protocol uses in practice (matches Python's CreateFileMapping
    // tagname verbatim-and-unprefixed contract). Shared with CudaLinkOutTOP (win_util.h).
    HANDLE h = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, widen(name).c_str());
    if (!h) {
        myStatus = "Waiting for producer";
        return false;
    }
    // Map generously; the mapping size is not stored anywhere on the wire, so Python
    // attachers observe a page-rounded size and this side does the same by mapping 0 bytes
    // (maps the whole underlying section).
    void* v = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!v) {
        CloseHandle(h);
        myError = "MapViewOfFile failed";
        return false;
    }
    // From here on, every remaining failure path below returns through this guard's
    // destructor instead of a hand-duplicated UnmapViewOfFile+CloseHandle pair (see
    // src/common/raii_handles.h).
    cudalink::common::ShmViewGuard guard(h, static_cast<uint8_t*>(v));

    // Real .tox's own Troubleshooting doc documents this exact failure mode: "another
    // process is using the same Ipcmemname for a different purpose."
    if (read_magic(guard.view()) != PROTOCOL_MAGIC) {
        myError = "Protocol magic mismatch -- Ipcmemname is in use by an unrelated process";
        return false;
    }

    // Query the mapping's actual accessible size (same technique Python attachers use --
    // the mapping size is not stored anywhere on the wire). Used to bounds-check a
    // wire-sourced num_slots before trusting it
    // (validateNumSlots): named SHM is attachable by any process that knows the name,
    // so its contents -- including num_slots -- are treated as untrusted input.
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(guard.view(), &mbi, sizeof(mbi)) == 0) {
        myError = "VirtualQuery failed";
        return false;
    }

    myShmHandle = guard.mapping();
    myShmView = guard.view();
    myShmMappedSize = mbi.RegionSize;
    guard.release(); // ownership transferred to myShmHandle/myShmView above
    return true;
}

// ---------------------------------------------------------------------------
// C4 stutter fix: bounded doorbell wait on NoFrame cooks (opt-in via Framewaitms).
//
// Root cause being addressed: producer and receiver TD processes both cook vsync-locked at
// the same rate, so the publish-vs-read phase offset inside each display frame is
// quasi-stationary. When the producer's publish lands just *after* this cook's
// acquire_slot() poll, ordinary ms-level jitter flips the classification between repeat
// and skip cook after cook -- a continuous judder that a producer restart re-rolls
// (live-confirmed 2026-07-06: noframe_ratio 9-31% on 32F dropped to 0.000 across 8
// consecutive windows after re-enabling the producer). Waiting a few ms for the publish
// doorbell converts those repeats into slightly-later fresh frames, which is phase-immune.
//
// Caveats, all deliberate:
// - The doorbell is an auto-reset event, so a stale signal from a publish this side already
//   consumed can wake the wait spuriously; the loop re-polls acquire_slot() after every
//   wake and re-waits with the remaining budget.
// - Auto-reset also wakes exactly ONE waiter: if a Python consumer waits on the same
//   doorbell, this TOP can consume signals that consumer was counting on (and vice versa).
//   That is why the wait is opt-in (Framewaitms defaults to 0) -- enable it only when this
//   TOP is the doorbell's sole waiter.
// - This blocks TD's cook thread for up to waitMs (<= kFramewaitMax, 8 ms) -- the entire
//   point of the fix: trade a bounded CPU wait for a fresh frame instead of a repeat.
// ---------------------------------------------------------------------------

cudalink::core::AcquireResult CudaLinkInTOP::waitForFreshFrame(cudalink::core::AcquireResult result,
                                                               int waitMs) {
    // Lazy open, retried on every waiting cook until the producer has created the event
    // (CudaLinkOutTOP::reallocate() creates it before its first commit_version; a producer
    // DLL predating the doorbell never will, in which case this stays null and the classic
    // instant-repeat path below is preserved). Name construction mirrors the producer's
    // verbatim (win_util.h widen() contract).
    if (!myDoorbellHandle) {
        const std::wstring wDoorbell = L"Local\\cudalink_db_" + widen(myCachedIpcmemname.c_str());
        myDoorbellHandle = OpenEventW(SYNCHRONIZE, FALSE, wDoorbell.c_str());
        if (myDoorbellHandle) {
            debugLog("doorbell opened (Framewaitms=" + std::to_string(waitMs) + ")");
        }
    }
    if (!myDoorbellHandle) {
        return result;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(waitMs);
    while (result.state == SlotState::NoFrame) {
        const auto remainingMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now())
                .count();
        if (remainingMs <= 0) {
            break; // budget exhausted -- this cook stays a repeat frame
        }
        const DWORD wake = WaitForSingleObject(asHandle(myDoorbellHandle), static_cast<DWORD>(remainingMs));
        if (wake != WAIT_OBJECT_0) {
            break; // WAIT_TIMEOUT (or FAILED/ABANDONED): give up until the next cook
        }
        // Doorbell rang -- could be the publish this cook is hoping for, or a stale signal
        // from one already consumed (auto-reset semantics). Re-classify and either return a
        // fresh state (NewFrame/VersionChanged/Shutdown, handled by execute()'s switch) or
        // loop back into the wait with whatever budget is left.
        result = acquire_slot(myShmView, myLayout, myLastWriteIdx, myLastVersion);
    }
    return result;
}

bool CudaLinkInTOP::validateNumSlots(uint32_t numSlots) {
    // SHMLayout's offset math is uint32_t; numSlots * SLOT_SIZE can silently wrap on a
    // pathological value, which would defeat a total_size()-based bounds check applied
    // after the fact. The realistic range is 2-4 (HELP_DOC.md's Numslots menu); a cap of
    // 64 leaves generous headroom while keeping numSlots * 128 (SLOT_SIZE) many orders
    // of magnitude below uint32_t's wraparound point.
    constexpr uint32_t kMaxReasonableSlots = 64;
    if (numSlots == 0 || numSlots > kMaxReasonableSlots) {
        myError = "invalid num_slots on wire (" + std::to_string(numSlots) + ")";
        return false;
    }
    const cudalink::core::SHMLayout candidate(numSlots);
    if (candidate.total_size() > myShmMappedSize) {
        // The claimed slot count would require reading past the actual mapped view --
        // reject rather than risk an out-of-bounds read into unmapped memory.
        myError =
            "num_slots (" + std::to_string(numSlots) + ") implies a layout larger than the mapped SHM region";
        return false;
    }
    return true;
}

bool CudaLinkInTOP::validateMetadata(const cudalink::core::Metadata& metadata) {
    if (metadata.width == 0 || metadata.height == 0 || metadata.num_comps == 0 ||
        metadata.bits_per_comp == 0) {
        myError = "invalid metadata on wire (width=" + std::to_string(metadata.width) +
                  " height=" + std::to_string(metadata.height) +
                  " num_comps=" + std::to_string(metadata.num_comps) +
                  " bits=" + std::to_string(metadata.bits_per_comp) + ")";
        return false;
    }
    const uint64_t expected = metadata.expected_size();
    if (expected == 0 || expected != metadata.data_size) {
        myError = "metadata data_size mismatch on wire (data_size=" + std::to_string(metadata.data_size) +
                  " expected=" + std::to_string(expected) + ")";
        return false;
    }
    return true;
}

bool CudaLinkInTOP::openSlotHandlesIfNeeded() {
    if (myHandlesOpen) {
        return true;
    }

    const uint32_t numSlots = read_num_slots(myShmView);
    if (!validateNumSlots(numSlots)) {
        return false;
    }
    myLayout = SHMLayout(numSlots);
    if (myLastVersion == 0) {
        // True first connect (never adopted a version yet): acquire_slot() deliberately
        // does not report VERSION_CHANGED when last_version==0, so this is the only place
        // the initial version gets adopted.
        myLastVersion = read_version(myShmView);
    }

    myMetadata = Metadata::read_from(myShmView, myLayout);
    if (!validateMetadata(myMetadata)) {
        return false;
    }

    // Build into local guards first and only commit to the class members
    // (mySlotDevPtrs/mySlotEvents) on full-loop success. Previously
    // this loop wrote straight into the class members via .assign(), so a mid-loop
    // CUDALINK_CUDA_CHECK_BOOL failure (e.g. slot 2 of 4) left slots 0-1's already-opened IPC
    // imports dangling in the members -- silently leaked on the next retry's .assign()
    // overwrite. Local guards make that leak structurally impossible: on early return here,
    // every guard already constructed in newDevPtrs/newEvents destructs and closes its handle.
    std::vector<cudalink::common::CudaIpcMemGuard> newDevPtrs(numSlots);
    std::vector<cudalink::common::CudaEventGuard> newEvents(numSlots);

    for (uint32_t slot = 0; slot < numSlots; ++slot) {
        cudaIpcMemHandle_t memHandle;
        std::memcpy(&memHandle, myShmView + myLayout.mem_handle_offset(slot), sizeof(memHandle));
        void* rawDevPtr = nullptr;
        CUDALINK_CUDA_CHECK_BOOL(cudaIpcOpenMemHandle(&rawDevPtr, memHandle, cudaIpcMemLazyEnablePeerAccess),
                                 myError);
        newDevPtrs[slot].reset(rawDevPtr);

        cudaIpcEventHandle_t eventHandle;
        std::memcpy(&eventHandle, myShmView + myLayout.event_handle_offset(slot), sizeof(eventHandle));
        cudaEvent_t rawEvent = nullptr;
        CUDALINK_CUDA_CHECK_BOOL(cudaIpcOpenEventHandle(&rawEvent, eventHandle), myError);
        newEvents[slot].reset(rawEvent);
    }

    mySlotDevPtrs.clear();
    mySlotDevPtrs.reserve(newDevPtrs.size());
    for (auto& guard : newDevPtrs) {
        mySlotDevPtrs.push_back(guard.release());
    }
    mySlotEvents.clear();
    mySlotEvents.reserve(newEvents.size());
    for (auto& guard : newEvents) {
        mySlotEvents.push_back(guard.release());
    }

    myHandlesOpen = true;
    return true;
}

// ---------------------------------------------------------------------------
// execute() -- per-cook: detect parameter changes, open/reopen the SHM connection as
// needed, classify the current wire state, then wait-and-copy the current slot into the
// output texture.
// ---------------------------------------------------------------------------

void CudaLinkInTOP::execute(TD::TOP_Output* output, const TD::OP_Inputs* inputs, void*) {
    const auto cookStart = std::chrono::steady_clock::now();
    try {
        myError.clear();
        myWarning.clear();

        // Once a fatal CUDA error has latched (ctor's device/IPC checks, or a
        // CUDALINK_CUDA_CHECK_FATAL failure below), stop retrying CUDA calls every cook --
        // the stream/context may now be corrupted for the rest of the process. Recreating
        // this node (or restarting TD) is the only recovery; cudaDeviceReset() is not an
        // option (it would tear down TD's own CUDA state too).
        if (myFatal) {
            myError = "fatal CUDA error latched -- recreate this node or restart TouchDesigner "
                      "to recover (see last_error in the Info DAT)";
            return;
        }

        // Step 0: cached-diff parameter-change detection (no push notification exists).
        checkParameterChanges(inputs);

        // One-time emission of the ctor's device-session diagnostic (skipped/failed IPC
        // capability probe, WDDM/TCC support-envelope note, ...). myDebugLog isn't enabled
        // yet at construction time (the Debug toggle is only readable via OP_Inputs, which
        // the ctor doesn't have), so this is deferred to the first cook, right after
        // checkParameterChanges() has refreshed myDebugLog's enabled state above. myInitNote
        // itself is left intact (not cleared) so it stays reachable via the init_note Info
        // DAT row for the rest of this instance's lifetime.
        if (!myInitNoteLogged) {
            myInitNoteLogged = true;
            if (!myInitNote.empty()) {
                debugLog("init: " + myInitNote);
            }
        }

        // Step 1: Active gate + SHM open (retried every cook -- no backoff needed for a
        // per-cook-polled Custom TOP).
        if (!Parameters::evalActive(inputs)) {
            myStatus = "Idle";
            return;
        }
        if (!myShmView) {
            const char* name = Parameters::evalIpcmemname(inputs);
            if (!name || !openSHM(name)) {
                return; // status already set to "Waiting for producer"
            }
            // BUG FIX: myLayout must be built from the real num_slots *before* the first
            // acquire_slot() call below -- acquire_slot() reads shutdown_offset(), which
            // depends on num_slots. Left at its zero-initialized default (num_slots=0),
            // shutdown_offset() resolves to byte 20 -- inside slot 0's raw IPC mem-handle
            // bytes, not the real shutdown flag -- which is essentially random binary
            // data and was being misread as a nonzero (shutdown) flag on every cook,
            // causing a rapid open/teardown/reopen churn loop that looked like a TD hang.
            const uint32_t numSlots = read_num_slots(myShmView);
            if (!validateNumSlots(numSlots)) {
                teardown();
                return;
            }
            myLayout = SHMLayout(numSlots);
        }

        // Step 2: classify SHM state.
        AcquireResult result = acquire_slot(myShmView, myLayout, myLastWriteIdx, myLastVersion);

        // C4 stutter fix (opt-in, Framewaitms > 0): an initially-frameless cook gets one
        // bounded chance to pick up a publish that is about to land, instead of instantly
        // committing to a repeat frame -- see waitForFreshFrame() for the full rationale
        // and caveats. 'rescued' only steers the cadence tally below; the rescued frame
        // itself flows through the ordinary NewFrame path.
        bool rescued = false;
        if (result.state == SlotState::NoFrame) {
            const int waitMs = Parameters::evalFramewaitms(inputs);
            if (waitMs > 0) {
                result = waitForFreshFrame(result, waitMs);
                rescued = (result.state == SlotState::NewFrame);
            }
        }

        switch (result.state) {
            case SlotState::NoFrame:
                // Previous output persists (matches the SpectrumTOP.cpp / PyTorchTOP.cpp
                // samples) -- which is exactly a repeat frame at the display, so tally it:
                // a rising noframe_ratio in the cadence log is the signature of a producer
                // running below display rate (stutter hypothesis H3/H4), invisible to every
                // timing channel because this cook does no CUDA work at all.
                if (myDebugLog.enabled()) {
                    ++myTotalNoFrame;
                    myCadence.tally(cudalink::common::CadenceCounters::Kind::NoFrame, myFrameCount,
                                    myDebugLog);
                }
                return;
            case SlotState::Shutdown:
                teardown();
                myStatus = "Producer exited";
                return;
            case SlotState::VersionChanged:
                if (myDebugLog.enabled()) {
                    ++myTotalVersionChanged;
                    myCadence.tally(cudalink::common::CadenceCounters::Kind::VersionChanged, myFrameCount,
                                    myDebugLog);
                }
                debugLog("VERSION_CHANGED: new_version=" + std::to_string(result.new_version));
                closeHandles(); // re-derive layout from a freshly-read num_slots
                myLastVersion = result.new_version;
                // A new producer session's write_idx also restarts at 0 (SHMLayout::
                // build_buffer's default); without this reset, a stale myLastWriteIdx
                // that happens to numerically match the new session's counter later
                // would cause acquire_slot() to silently report NO_FRAME and miss a
                // real frame.
                myLastWriteIdx = 0;
                break; // fall through to handle-opening below
            case SlotState::NewFrame:
                if (myDebugLog.enabled()) {
                    if (rescued) {
                        ++myTotalRescued;
                    }
                    myCadence.tally(rescued ? cudalink::common::CadenceCounters::Kind::Rescued
                                            : cudalink::common::CadenceCounters::Kind::NewFrame,
                                    myFrameCount, myDebugLog);
                }
                break;
        }

        // Step 3: open-once-per-version (covers both VERSION_CHANGED fall-through and
        // the true first-connect case uniformly).
        if (!openSlotHandlesIfNeeded()) {
            return; // myError already set
        }

        if (result.state == SlotState::VersionChanged) {
            // VERSION_CHANGED alone carries no frame to display this cook; wait for the
            // next NEW_FRAME classification (mirrors _refresh_on_version_change).
            return;
        }

        // Defensive bound check: result.slot is derived from wire-sourced write_idx and
        // num_slots (both untrusted -- see validateNumSlots()); a malformed or malicious
        // SHM segment could in principle desync this from the actual size of
        // mySlotDevPtrs/mySlotEvents. Never index with an unchecked value.
        if (result.slot >= mySlotDevPtrs.size()) {
            myError = "acquire_slot returned out-of-range slot index";
            return;
        }
        myReadSlot = result.slot;

        // Step 4: create the output texture from cached metadata.
        const TD::OP_PixelFormat pixelFormat = cudalink::in_top::mapToPixelFormat(myMetadata);
        if (pixelFormat == TD::OP_PixelFormat::Invalid) {
            myWarning = "unsupported pixel format on wire (kind=" + std::to_string(myMetadata.format_kind) +
                        " bits=" + std::to_string(myMetadata.bits_per_comp) +
                        " flags=" + std::to_string(myMetadata.flags) + ")";
            return;
        }

        TD::TOP_CUDAOutputInfo info;
        info.textureDesc.width = myMetadata.width;
        info.textureDesc.height = myMetadata.height;
        info.textureDesc.texDim = TD::OP_TexDim::e2D;
        info.textureDesc.pixelFormat = pixelFormat;
        info.stream = myStream;

        const TD::OP_CUDAArrayInfo* outputInfo = output->createCUDAArray(info, nullptr);
        if (!outputInfo) {
            myError = "createCUDAArray failed";
            return;
        }

        // Step 5: GPU-side wait + D2D copy, no CPU block -- confirmed by
        // CPlusPlus_Common.h's documented begin/end bracket purpose, not just inferred.
        // CudaOpScope ties endCUDAOperations() to the closing brace of this nested block --
        // scoped tightly to just the CUDA-ops region (matching where the original explicit
        // endCUDAOperations() call sat, right after the D2D copy and before the non-CUDA
        // copy_us/Step 6 bookkeeping) so a CUDALINK_CUDA_CHECK_FATAL early `return` -- or an
        // exception unwinding through the catch(...) below -- can never leave the bracket
        // unbalanced, and the bracket isn't held open longer than TD's contract actually
        // requires.
        // myBeginUs/myCopyUs/myEndUs isolate begin-bracket, actual-CUDA-call, and
        // end-bracket cost (see CudaLinkOutTOP.cpp's Step 5 comment for the shared
        // rationale). 'workEnd' is captured just before the closing brace and read again
        // immediately after it; every early-return path below exits execute() entirely
        // before the post-block myEndUs line, so it's never read unset.
        const auto beginStart = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point workEnd;
        {
            cudalink::common::CudaOpScope cudaOps(myContext);
            const auto beginEnd = std::chrono::steady_clock::now();
            myBeginUs = std::chrono::duration<float, std::micro>(beginEnd - beginStart).count();
            if (!cudaOps) {
                myError = "beginCUDAOperations failed";
                return;
            }

            const size_t itemsize = myMetadata.bits_per_comp / cudalink::core::BITS_PER_BYTE;
            const size_t rowBytes = static_cast<size_t>(myMetadata.width) * myMetadata.num_comps * itemsize;

            // Debug-gated GPU-side timing (see gpu_timer.h). Legacy path: a three-point
            // A->B->C bracket via two rings -- myWaitTimer spans A->B (how long the
            // cudaStreamWaitEvent below actually stalls this stream, i.e. how late the
            // producer's interprocess event fired: the receiver-visible stutter suspect),
            // myCopyTimer spans B->C (the real GPU cost of the D2D copy, which scales ~4x on
            // float32). Graph path (tryGraphedCopy()): wait+copy are fused inside one
            // cudaGraphLaunch, so myCopyTimer brackets the whole launch (gpu_copy_us = the
            // combined pair) and event_wait_us is pinned to 0.0 as an explicit "not
            // separately observable in graph mode" marker -- A/B against legacy compares
            // gpu_copy_us(graph) vs gpu_copy_us+event_wait_us(legacy). Reads are of the pair
            // recorded kRingSize cooks ago -- never this cook's own, never a sync.
            uint32_t waitTimerIdx = 0;
            uint32_t copyTimerIdx = 0;
            bool timersArmed = false;
            if (myDebugLog.enabled()) {
                if (myWaitTimer.ensureCreated() && myCopyTimer.ensureCreated()) {
                    float us = 0.0f;
                    if (myWaitTimer.tryRead(myWaitTimer.peekReadIdx(), &us)) {
                        myEventWaitUs = us;
                    }
                    if (myCopyTimer.tryRead(myCopyTimer.peekReadIdx(), &us)) {
                        myGpuCopyUs = us;
                    }
                    timersArmed = true;
                } else if (!myGpuTimingUnavailable) {
                    myGpuTimingUnavailable = true;
                    debugLog("gpu timing unavailable: cudaEventCreate failed");
                }
            }

            bool graphed = false;
            if (myGraphsRequested && !myGraphsDisabled) {
                if (timersArmed) {
                    copyTimerIdx = myCopyTimer.recordStart(myStream);
                }
                graphed = tryGraphedCopy(myReadSlot, outputInfo->cudaArray, rowBytes, myMetadata.height);
                if (timersArmed && graphed) {
                    myCopyTimer.recordStop(copyTimerIdx, myStream);
                    myEventWaitUs = 0.0f; // fused into the graph -- see the bracket comment above
                }
                // On failure the start recorded above is simply never stopped (one dropped
                // sample; failure latches myGraphsDisabled, so at most once per session) and
                // the legacy path below arms its own fresh bracket.
            }

            if (!graphed) {
                if (timersArmed) {
                    waitTimerIdx = myWaitTimer.recordStart(myStream); // event A
                }

                CUDALINK_CUDA_CHECK_FATAL(cudaStreamWaitEvent(myStream, mySlotEvents[myReadSlot], 0), myError,
                                          myFatal);

                if (timersArmed) {
                    myWaitTimer.recordStop(waitTimerIdx, myStream);   // event B
                    copyTimerIdx = myCopyTimer.recordStart(myStream); // event B' (back-to-back with B)
                }

                CUDALINK_CUDA_CHECK_FATAL(
                    cudaMemcpy2DToArrayAsync(outputInfo->cudaArray, 0, 0, mySlotDevPtrs[myReadSlot], rowBytes,
                                             rowBytes, myMetadata.height, cudaMemcpyDeviceToDevice, myStream),
                    myError, myFatal);

                if (timersArmed) {
                    myCopyTimer.recordStop(copyTimerIdx, myStream); // event C
                }
            }
            // myCopyUs below stays CPU-side enqueue cost only (the copy itself is async on
            // myStream); the true GPU-side numbers are the event_wait_us/gpu_copy_us pair
            // recorded just above.
            workEnd = std::chrono::steady_clock::now();
            myCopyUs = std::chrono::duration<float, std::micro>(workEnd - beginEnd).count();
        }
        myEndUs =
            std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - workEnd).count();

        // Step 6.
        myLastWriteIdx = result.write_idx;
        ++myFrameCount;
        myStatus = std::to_string(myMetadata.width) + "x" + std::to_string(myMetadata.height);
    } catch (...) {
        // No exception ever crosses the ABI.
        myError = "unexpected exception in execute()";
    }
    myCookUs = std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - cookStart).count();

    // Periodic bench log (Debug-gated, every 97 frames -- mirrors CudaLinkOutTOP's
    // cadence): gives an offline avg cook_us/copy_us/begin_us/end_us number without
    // needing to wire and eyeball an Info CHOP live.
    if (myDebugLog.enabled()) {
        myBench.record(myCookUs, myCopyUs, myBeginUs, myEndUs, myEventWaitUs, "event_wait_us", myGpuCopyUs,
                       "gpu_copy_us", myFrameCount, myDebugLog);
    }
}

// ---------------------------------------------------------------------------
// Info CHOP / DAT / status
// ---------------------------------------------------------------------------

int32_t CudaLinkInTOP::getNumInfoCHOPChans(void*) {
    return kNumInfoCHOPChans; // frames, cook_us, copy_us, begin_us, end_us, write_idx, read_slot, num_slots
}

void CudaLinkInTOP::getInfoCHOPChan(int32_t index, TD::OP_InfoCHOPChan* chan, void*) {
    // setString() may allocate; a std::bad_alloc must not cross the ABI.
    try {
        switch (index) {
            case 0:
                chan->name->setString("frames");
                chan->value = static_cast<float>(myFrameCount);
                break;
            case 1:
                chan->name->setString("cook_us");
                chan->value = myCookUs;
                break;
            case 2:
                chan->name->setString("copy_us");
                chan->value = myCopyUs;
                break;
            case 3:
                // beginCUDAOperations() cost (CudaOpScope construction).
                chan->name->setString("begin_us");
                chan->value = myBeginUs;
                break;
            case 4:
                // endCUDAOperations() cost (~CudaOpScope() destructor).
                chan->name->setString("end_us");
                chan->value = myEndUs;
                break;
            case 5:
                chan->name->setString("write_idx");
                chan->value = static_cast<float>(myLastWriteIdx);
                break;
            case 6:
                chan->name->setString("read_slot");
                chan->value = static_cast<float>(myReadSlot);
                break;
            case 7:
                // Read-only Numslots display, sourced from the wire header -- surfaced here
                // rather than as a parameter (see CudaLinkInTOP.h).
                chan->name->setString("num_slots");
                chan->value = static_cast<float>(myLayout.num_slots());
                break;
            case 8:
                // GPU-side (cudaEventElapsedTime) stall of the cudaStreamWaitEvent on the
                // producer's ready-event, read one ring-lap late; 0.0 until Debug is on.
                chan->name->setString("event_wait_us");
                chan->value = myEventWaitUs;
                break;
            case 9:
                // GPU-side duration of the D2D copy into the output array; 0.0 until Debug
                // is on.
                chan->name->setString("gpu_copy_us");
                chan->value = myGpuCopyUs;
                break;
            case 10:
                // Session-lifetime count of NoFrame cooks (repeat frames at the display);
                // only advances while Debug is on.
                chan->name->setString("noframe_count");
                chan->value = static_cast<float>(myTotalNoFrame);
                break;
            case 11:
                // Session-lifetime count of VersionChanged cooks (also frameless); only
                // advances while Debug is on.
                chan->name->setString("version_changed_count");
                chan->value = static_cast<float>(myTotalVersionChanged);
                break;
            case 12:
                // Session-lifetime count of doorbell-rescued frames (NoFrame cooks the
                // Framewaitms wait converted into fresh frames -- see waitForFreshFrame());
                // only advances while Debug is on.
                chan->name->setString("rescued_count");
                chan->value = static_cast<float>(myTotalRescued);
                break;
            case 13:
                // Session-lifetime count of graph-path cache hits (one cudaGraphLaunch in
                // place of the waitEvent+memcpy pair -- see tryGraphedCopy()); only advances
                // while Debug is on AND CUDALINK_CPP_USE_GRAPHS is set.
                chan->name->setString("graph_hits");
                chan->value = static_cast<float>(myGraphHits);
                break;
            case 14:
                // Session-lifetime count of graph captures (cache misses). Should plateau at
                // (num_slots x observed output arrays) within seconds; a count that keeps
                // climbing is the live signal that TD's output arrays are NOT pointer-stable
                // and the cache cap will eventually latch graphs off. Only advances while
                // Debug is on AND CUDALINK_CPP_USE_GRAPHS is set.
                chan->name->setString("graph_builds");
                chan->value = static_cast<float>(myGraphBuilds);
                break;
            default:
                break;
        }
    } catch (...) { // NOLINT(bugprone-empty-catch) -- deliberate ABI fence, see comment above
    }
}

bool CudaLinkInTOP::getInfoDATSize(TD::OP_InfoDATSize* infoSize, void*) {
    infoSize->rows = kNumInfoDATRows; // ipc_version, status, last_error, last_error_frame, init_note
    infoSize->cols = 2;
    infoSize->byColumn = false;
    return true;
}

void CudaLinkInTOP::getInfoDATEntries(int32_t index, int32_t, TD::OP_InfoDATEntries* entries, void*) {
    // std::to_string()/setString() may allocate; a std::bad_alloc must not cross the ABI.
    try {
        if (index == 0) {
            entries->values[0]->setString("ipc_version");
            entries->values[1]->setString(std::to_string(myLastVersion).c_str());
        } else if (index == 1) {
            entries->values[0]->setString("status");
            entries->values[1]->setString(myStatus.c_str());
        } else if (index == 2) {
            // Sticky (never auto-cleared, only overwritten) -- see getErrorString(). Survives
            // across cooks, unlike the transient error badge (live-test finding: a real error
            // during the sender's VERSION_CHANGED window flashed and disappeared too fast to
            // read).
            entries->values[0]->setString("last_error");
            entries->values[1]->setString(myLastError.c_str());
        } else if (index == 3) {
            entries->values[0]->setString("last_error_frame");
            entries->values[1]->setString(std::to_string(myLastErrorFrame).c_str());
        } else if (index == 4) {
            // Ctor-time device-session diagnostic (see the emission comment in execute()) --
            // surfaced here too so it's reachable without turning Debug on. Empty when the
            // IPC capability probe succeeded with nothing to report.
            entries->values[0]->setString("init_note");
            entries->values[1]->setString(myInitNote.c_str());
        }
    } catch (...) { // NOLINT(bugprone-empty-catch) -- deliberate ABI fence, see comment above
    }
}

void CudaLinkInTOP::getErrorString(TD::OP_String* error, void*) {
    // string copy/setString() may allocate; a std::bad_alloc must not cross the ABI.
    try {
        if (!myError.empty()) {
            myLastError = myError;
            myLastErrorFrame = myFrameCount;
            debugLog("ERROR: " + myError);
        }
        error->setString(myError.c_str());
        myError.clear();
    } catch (...) { // NOLINT(bugprone-empty-catch) -- deliberate ABI fence, see comment above
    }
}

void CudaLinkInTOP::getWarningString(TD::OP_String* warning, void*) {
    // string copy/setString() may allocate; a std::bad_alloc must not cross the ABI.
    try {
        if (!myWarning.empty()) {
            debugLog("WARNING: " + myWarning);
        }
        warning->setString(myWarning.c_str());
        myWarning.clear();
    } catch (...) { // NOLINT(bugprone-empty-catch) -- deliberate ABI fence, see comment above
    }
}

void CudaLinkInTOP::getInfoPopupString(TD::OP_String* info, void*) {
    // string copy/setString() may allocate; a std::bad_alloc must not cross the ABI.
    try {
        info->setString(myStatus.c_str());
    } catch (...) { // NOLINT(bugprone-empty-catch) -- deliberate ABI fence, see comment above
    }
}
