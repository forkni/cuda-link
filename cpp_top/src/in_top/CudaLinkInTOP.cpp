#include "CudaLinkInTOP.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <algorithm>
#include <windows.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "Parameters.h"
#include "pixel_format_map.h"
#include "../common/cuda_check.h"
#include "../common/common_page_td.h"
#include "../common/info_page.h"
#include "../common/resize_convert.h"
#include "../core/common_page.h"
#include "../core/doorbell_name.h"
#include "../core/pixel_codec.h"
#include "../out_top/pixel_format_map.h" // OP_PixelFormat -> wire tuple, for the Common target
#include "../common/cuda_device_session.h"
#include "../common/cuda_op_scope.h"
#include "../common/raii_handles.h"
#include "../common/clock.h"
#include "../common/win_util.h"
#include "../core/frame_step.h"
#include "../core/frame_wait.h"

using cudalink::common::asHandle;
using cudalink::common::now_seconds;
using cudalink::common::widen;
using cudalink::core::AcquireResult;
using cudalink::core::Metadata;
using cudalink::core::PROTOCOL_MAGIC;
using cudalink::core::PROTOCOL_MAGIC_V05;
using cudalink::core::SHMLayout;
using cudalink::core::SlotState;
using cudalink::core::acquire_slot;
using cudalink::core::kInfoChopFrameMod;
using cudalink::core::read_magic;
using cudalink::core::read_num_slots;
using cudalink::core::read_sender_pid;
using cudalink::core::read_version;

namespace {
// Names the Info CHOP/DAT counts returned by getNumInfoCHOPChans()/getInfoDATSize() --
// must stay in sync with the number of `case` labels / index branches in
// getInfoCHOPChan()/getInfoDATEntries() below.
constexpr int32_t kNumInfoCHOPChans = 20; // frames, cook_us, copy_us, begin_us, end_us, write_idx, read_slot,
                                          // num_slots, event_wait_us, gpu_copy_us, noframe_count,
                                          // version_changed_count, rescued_count,
                                          // producer_frame, stream_id, latency_us, frame_gap_count,
                                          // producer_skip_count, frame_wait_skipped, converted
constexpr int32_t kNumOwnInfoDATRows = 7; // ipc_version, status, last_error, last_error_frame, init_note,
                                          // producer_frame, source_timestamp_us
// ... then the shared version and build rows (info_page.h).
constexpr int32_t kNumInfoDATRows = kNumOwnInfoDATRows + cudalink::common::kNumBuildInfoDATRows;

// CUDA IPC cannot open a handle in the process that exported it. Probed 2026-11-16 (runtime
// 12.8, driver 13.05, RTX 4090): cudaIpcOpenMemHandle and cudaIpcOpenEventHandle on own-process
// handles both return cudaErrorDeviceUninitialized (201, "invalid device context"). The error is
// not sticky. The In TOP always opens with a current context (CudaDeviceSession), so 201 here
// names the same-process case; the raw status name stays in the message. Since v0.6 the
// sender_pid header field catches this case first (openSlotHandlesIfNeeded); 201 is the
// fallback only for a sender that writes sender_pid 0 (unknown). With a known, different
// sender_pid, 201 is a real context error and keeps the generic message.
constexpr const char* kSameProcessMessage =
    "sender and receiver appear to be in the same process -- CUDA IPC cannot loop back within one "
    "process; run the sender in a separate TD instance or use the Python peer";

// Like CUDALINK_CUDA_CHECK_BOOL (same "(file:line)" suffix), but turns status 201 into
// kSameProcessMessage when the sender's PID is unknown. 'line' is the call site's __LINE__.
bool checkIpcOpen(cudaError_t status, const char* call, bool senderPidUnknown, int line, std::string& err) {
    if (status == cudaSuccess) {
        return true;
    }
    (void)cudaGetLastError(); // clear the non-sticky error so later calls do not report it
    const std::string where = std::string(" (") + __FILE__ + ":" + std::to_string(line) + ")";
    if (status == cudaErrorDeviceUninitialized && senderPidUnknown) {
        err = std::string(kSameProcessMessage) + " (" + call + ": " + cudaGetErrorName(status) + ")" + where;
    } else {
        err = std::string(call) + " failed: " + cudaGetErrorString(status) + where;
    }
    return false;
}
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

    // 2: Info page added (Semverversion/Semverbuild/Semverauthor/Semvershowhelp).
    // 3: Semverversion/Semverbuild removed (a saved value hid the loaded build); version and
    //    build moved to the Info page header label and the Info DAT.
    customInfo.minorVersion = 3;
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
    if (myFatal) {
        myFatalError = session.error;
    }
    myInitNote = session.note;
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
        cudalink::common::appendInfoPage(manager, "CudaLinkInTOP");
    } catch (...) { // NOLINT(bugprone-empty-catch) -- deliberate ABI fence, see comment above
    }
}

// ---------------------------------------------------------------------------
// Parameter-change detection -- the Custom TOP API has no push notification for
// parameter edits, so this diffs the values we care about against last cook's cache.
// ---------------------------------------------------------------------------

void CudaLinkInTOP::pulsePressed(const char* name, void*) {
    cudalink::common::handleInfoPulse(name);
}

void CudaLinkInTOP::checkParameterChanges(const TD::OP_Inputs* inputs) {
    cudalink::common::greyInfoPage(inputs);
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
    myConvertScratch.release();
    myConverted = false;
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
    myPublishCadence = {};
    myLayout = SHMLayout(0);
    myShmMappedSize = 0;
    myStatus = "Waiting for producer";
    debugLog("teardown: complete");
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
    const uint32_t magic = read_magic(guard.view());
    if (magic == PROTOCOL_MAGIC_V05) {
        myError = "sender uses protocol v0.5 (cuda-link 1.x); this receiver needs v0.6";
        return false;
    }
    if (magic != PROTOCOL_MAGIC) {
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
// C4 stutter fix: bounded doorbell wait on NoFrame cooks (Framewaitms, default 3 ms).
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
// - Auto-reset also wakes exactly ONE waiter: a second consumer of the same Ipcmemname (a
//   second In TOP, or a Python consumer with CUDALINK_DOORBELL) can take the signal this TOP
//   waits for. The wait therefore runs in slices of kFrameWaitSliceMs (1 ms) and re-polls
//   acquire_slot() after each, so a lost signal costs at most one slice, not the budget.
// - Default 3 ms: the real budget of the Phase 5 sync-kit setting (old "4", which truncated
//   to ~3 ms; 0.19 % mismatches). 2026-11-17 dev-box sweeps, 1280x720: duplicates 4-6 % at 0
//   -> about 1 % at 3. The cost depends on the A/B phase: a read just before the publish
//   waits for it (mean cook 0.2-1.3 ms at 3-4 ms). 0 turns the wait off.
// - This blocks TD's cook thread for up to waitMs (<= kFramewaitMax, 8 ms) -- the entire
//   point of the fix: trade a bounded CPU wait for a fresh frame instead of a repeat.
// - Guard (cadence_wait_useful, core/frame_wait.h): execute() skips the wait when the next
//   publish is not expected within the budget (a sender slower than this TOP, for example a
//   30 fps camera into 60 fps) or the sender is paused. Counted in frame_wait_skipped.
// ---------------------------------------------------------------------------

void CudaLinkInTOP::recordSlotMeta(const cudalink::core::AcquireResult& result) {
    using cudalink::core::FrameStep;
    const cudalink::core::SlotMeta& meta = result.slot_meta;
    // myLastWriteIdx is 0 on the first frame of a connection (teardown / VersionChanged reset
    // it), which classify_frame_step() reports as First.
    const FrameStep step = cudalink::core::classify_frame_step(myLastWriteIdx, myProducerFrame,
                                                               result.write_idx, meta.producer_frame);
    if (step == FrameStep::FrameGap) {
        ++myFrameGapCount;
    } else if (step == FrameStep::ProducerSkip) {
        ++myProducerSkipCount;
    }
    // A new connection or producer session (myLastWriteIdx reset to 0) starts the cadence over.
    if (myLastWriteIdx == 0) {
        myPublishCadence = {};
    }
    cudalink::core::cadence_add(myPublishCadence, meta.host_timestamp,
                                static_cast<uint64_t>(result.write_idx - myLastWriteIdx));
    myProducerFrame = meta.producer_frame;
    mySourceTimestampUs = meta.source_timestamp_us;
    myStreamId = meta.stream_id;
    myLatencyUs =
        meta.host_timestamp > 0.0 ? static_cast<float>((now_seconds() - meta.host_timestamp) * 1e6) : 0.0f;
}

cudalink::core::AcquireResult CudaLinkInTOP::waitForFreshFrame(cudalink::core::AcquireResult result,
                                                               int waitMs) {
    // Lazy open, retried on every waiting cook until the producer has created the event
    // (CudaLinkOutTOP::reallocate() creates it before its first commit_version; a producer
    // DLL predating the doorbell never will, in which case this stays null and the classic
    // instant-repeat path below is preserved). The producer builds the name with the same
    // function (core/doorbell_name.h).
    if (!myDoorbellHandle) {
        const std::wstring wDoorbell = cudalink::core::doorbell_event_name(myCachedIpcmemname.c_str());
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
        const int64_t remainingUs =
            std::chrono::duration_cast<std::chrono::microseconds>(deadline - std::chrono::steady_clock::now())
                .count();
        const uint32_t sliceMs = cudalink::core::next_wait_slice_ms(remainingUs);
        if (sliceMs == 0) {
            break; // budget spent -- this cook stays a repeat frame
        }
        const DWORD wake = WaitForSingleObject(asHandle(myDoorbellHandle), static_cast<DWORD>(sliceMs));
        if (wake != WAIT_OBJECT_0 && wake != WAIT_TIMEOUT) {
            break; // WAIT_FAILED / WAIT_ABANDONED: give up until the next cook
        }
        // Doorbell rang (the publish this cook hopes for, or a stale signal from one already
        // consumed) or the slice ran out (another consumer may have taken the signal).
        // Re-classify and either return a fresh state (NewFrame/VersionChanged/Shutdown,
        // handled by execute()'s switch) or wait again with whatever budget is left.
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

    const uint32_t senderPid = read_sender_pid(myShmView);
    if (senderPid != 0 && senderPid == GetCurrentProcessId()) {
        myError = kSameProcessMessage; // the cudaIpcOpen* status 201 path below stays as a fallback
        return false;
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
        if (!checkIpcOpen(cudaIpcOpenMemHandle(&rawDevPtr, memHandle, cudaIpcMemLazyEnablePeerAccess),
                          "cudaIpcOpenMemHandle", senderPid == 0, __LINE__, myError)) {
            return false;
        }
        newDevPtrs[slot].reset(rawDevPtr);

        cudaIpcEventHandle_t eventHandle;
        std::memcpy(&eventHandle, myShmView + myLayout.event_handle_offset(slot), sizeof(eventHandle));
        cudaEvent_t rawEvent = nullptr;
        if (!checkIpcOpen(cudaIpcOpenEventHandle(&rawEvent, eventHandle), "cudaIpcOpenEventHandle",
                          senderPid == 0, __LINE__, myError)) {
            return false;
        }
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
            // Keep the error that latched: getErrorString() copied it into myLastError after
            // the failing cook, and the constructor stores its own reason in myFatalError.
            // Without this, the latch text replaced it within one frame and last_error could
            // only ever show the latch text.
            if (myFatalError.empty()) {
                myFatalError = myLastError;
            }
            myError = "fatal CUDA error latched: " + myFatalError +
                      " -- recreate this node or restart TouchDesigner to recover";
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

        // C4 stutter fix (Framewaitms > 0, default 3): an initially-frameless cook gets one
        // bounded chance to pick up a publish that is about to land, instead of instantly
        // committing to a repeat frame -- see waitForFreshFrame() for the full rationale
        // and caveats. 'rescued' only steers the cadence tally below; the rescued frame
        // itself flows through the ordinary NewFrame path.
        bool rescued = false;
        if (result.state == SlotState::NoFrame) {
            const int waitMs = Parameters::evalFramewaitms(inputs);
            if (waitMs > 0) {
                if (cudalink::core::cadence_wait_useful(myPublishCadence, now_seconds(), waitMs)) {
                    result = waitForFreshFrame(result, waitMs);
                    rescued = (result.state == SlotState::NewFrame);
                } else {
                    ++myFrameWaitSkipCount;
                }
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

        // Step 4b: the Common page. This TOP has no TOP input, so TD resolves "Use Input"
        // and the relative modes against a 127x127 placeholder (Phase 0 probe). Resolve
        // them against the wire frame here (core/common_page.h); take TD's suggested desc
        // only where it does not depend on an input: Custom / Parameter Panel sizes, a
        // custom aspect, an explicit Pixel Format.
        TD::OP_TextureDesc suggested;
        output->getSuggestedOutputDesc(&suggested, nullptr);
        const cudalink::core::Size2 wireSize{myMetadata.width, myMetadata.height};
        const cudalink::core::ResMode resMode =
            cudalink::core::parse_res_mode(inputs->getParString("outputresolution"));
        cudalink::core::Size2 box{};
        if (resMode == cudalink::core::ResMode::Fit || resMode == cudalink::core::ResMode::Limit) {
            // The Resolution tuplet: getParInt("resolutionw") does not exist for a plugin
            // (probe); the tuplet name with an index does.
            box.w = static_cast<uint32_t>(std::max(1, inputs->getParInt("resolution", 0)));
            box.h = static_cast<uint32_t>(std::max(1, inputs->getParInt("resolution", 1)));
        }
        const cudalink::core::Size2 outSize =
            cudalink::core::resolve_size(resMode, wireSize, box, {suggested.width, suggested.height});
        const cudalink::core::Aspect2 aspect = cudalink::core::resolve_aspect(
            cudalink::core::parse_aspect_mode(inputs->getParString("outputaspect")), outSize,
            {static_cast<float>(wireSize.w), static_cast<float>(wireSize.h)},
            {suggested.aspectX, suggested.aspectY});
        const char* formatName = inputs->getParString("format");
        TD::OP_PixelFormat targetFormat = pixelFormat;
        if (formatName != nullptr && std::strcmp(formatName, "useinput") != 0 &&
            suggested.pixelFormat != TD::OP_PixelFormat::Invalid) {
            targetFormat = suggested.pixelFormat;
        }
        const cudalink::out_top::WireFormat targetWire = cudalink::out_top::mapFromPixelFormat(targetFormat);
        if (!targetWire.supported) {
            myError = std::string("Common page Pixel Format ") +
                      cudalink::common::pixelFormatName(targetFormat) +
                      " is not supported by this TOP; pick another Pixel Format";
            return;
        }
        const bool convert = !(outSize == wireSize) || targetFormat != pixelFormat;
        myConverted = convert;
        // The Common Pixel Format defaults to rgba8fixed on a new node, so a float stream is
        // converted unless the user picks Use Input: say so (a BGRA <-> RGBA swap loses
        // nothing and gets no warning).
        if (targetFormat != pixelFormat && std::strcmp(cudalink::common::pixelFormatName(targetFormat),
                                                       cudalink::common::pixelFormatName(pixelFormat)) != 0) {
            myWarning = cudalink::common::formatConvertWarning("stream", pixelFormat, targetFormat);
        }
        // Read before beginCUDAOperations(): OP_Inputs may not be used inside the bracket.
        const bool bilinear = cudalink::common::commonBilinear(inputs);

        TD::TOP_CUDAOutputInfo info;
        info.textureDesc.width = outSize.w;
        info.textureDesc.height = outSize.h;
        info.textureDesc.texDim = TD::OP_TexDim::e2D;
        info.textureDesc.pixelFormat = targetFormat;
        info.textureDesc.aspectX = aspect.x;
        info.textureDesc.aspectY = aspect.y;
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

            // TD can give a texture smaller than the one asked for: TouchDesigner
            // Non-Commercial caps textures at 1280x1280 (2026-11-21: a 1920x1080 stream made
            // cudaMemcpy2DToArrayAsync fail with "invalid argument" every cook). cudaArray is
            // only valid inside the bracket, so the check is here and not after createCUDAArray.
            {
                cudaChannelFormatDesc arrayDesc{};
                cudaExtent arrayExtent{};
                unsigned int arrayFlags = 0;
                if (cudaArrayGetInfo(&arrayDesc, &arrayExtent, &arrayFlags, outputInfo->cudaArray) ==
                    cudaSuccess) {
                    if (arrayExtent.width < outSize.w || arrayExtent.height < outSize.h) {
                        myError = "TouchDesigner gave a " + std::to_string(arrayExtent.width) + "x" +
                                  std::to_string(arrayExtent.height) + " texture for a " +
                                  std::to_string(outSize.w) + "x" + std::to_string(outSize.h) +
                                  " output (TouchDesigner Non-Commercial is limited to 1280x1280); set the "
                                  "Common page Output Resolution to fit";
                        return;
                    }
                } else {
                    cudaGetLastError(); // diagnostic only: never leave the slot set for TD's own CUDA code
                }
            }

            const size_t itemsize = myMetadata.bits_per_comp / cudalink::core::BITS_PER_BYTE;
            const size_t rowBytes = static_cast<size_t>(myMetadata.width) * myMetadata.num_comps * itemsize;

            // Debug-gated GPU-side timing (see gpu_timer.h). Legacy path: a three-point
            // A->B->C bracket via two rings -- myWaitTimer spans A->B (how long the
            // cudaStreamWaitEvent below actually stalls this stream, i.e. how late the
            // producer's interprocess event fired: the receiver-visible stutter suspect),
            // myCopyTimer spans B->C (the real GPU cost of the D2D copy, which scales ~4x on
            // float32). Reads are of the pair
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

            if (timersArmed) {
                waitTimerIdx = myWaitTimer.recordStart(myStream); // event A
            }

            CUDALINK_CUDA_CHECK_FATAL(cudaStreamWaitEvent(myStream, mySlotEvents[myReadSlot], 0), myError,
                                      myFatal);

            if (timersArmed) {
                myWaitTimer.recordStop(waitTimerIdx, myStream);   // event B
                copyTimerIdx = myCopyTimer.recordStart(myStream); // event B' (back-to-back with B)
            }

            if (!convert) {
                CUDALINK_CUDA_CHECK_FATAL(
                    cudaMemcpy2DToArrayAsync(outputInfo->cudaArray, 0, 0, mySlotDevPtrs[myReadSlot], rowBytes,
                                             rowBytes, myMetadata.height, cudaMemcpyDeviceToDevice, myStream),
                    myError, myFatal);
            } else {
                // Common page path: the kernel reads the wire slot and writes the target
                // frame to linear scratch, then the same array copy as the fast path
                // puts it on the output. gpu_copy_us brackets both.
                const size_t outRowBytes = static_cast<size_t>(outSize.w) * targetWire.num_comps *
                                           (targetWire.bits_per_comp / cudalink::core::BITS_PER_BYTE);
                cudaError_t scratchStatus = cudaSuccess;
                void* scratch = myConvertScratch.ensure(outRowBytes * outSize.h, &scratchStatus);
                if (scratch == nullptr) {
                    myError = std::string("Common page scratch allocation failed: ") +
                              cudaGetErrorString(scratchStatus);
                    return;
                }
                const cudalink::common::LinearImage src{
                    mySlotDevPtrs[myReadSlot], rowBytes, myMetadata.width, myMetadata.height,
                    cudalink::core::make_codec(myMetadata.format_kind, myMetadata.bits_per_comp,
                                               myMetadata.flags, myMetadata.num_comps)};
                cudalink::core::PixelCodec outCodec = cudalink::core::make_codec(
                    targetWire.format_kind, targetWire.bits_per_comp, targetWire.flags, targetWire.num_comps);
                outCodec.alpha_only = cudalink::common::isAlphaOnly(targetFormat);
                const cudalink::common::LinearImageMut dst{scratch, outRowBytes, outSize.w, outSize.h,
                                                           outCodec};
                // launchResizeConvert() returns cudaGetLastError() after the launch, so an
                // error an earlier call left in the per-thread slot would be blamed on the
                // kernel. Take it out first and name it in the message instead.
                const cudaError_t priorStatus = cudaGetLastError();
                const cudaError_t convertStatus =
                    cudalink::common::launchResizeConvert(src, dst, bilinear, myStream);
                if (convertStatus != cudaSuccess) {
                    myError =
                        std::string("Common page convert failed: ") + cudaGetErrorString(convertStatus) +
                        " (wire " + std::to_string(wireSize.w) + "x" + std::to_string(wireSize.h) + " " +
                        cudalink::common::pixelFormatName(pixelFormat) + " codec " +
                        (src.codec.valid ? "ok" : "invalid") + ", out " + std::to_string(outSize.w) + "x" +
                        std::to_string(outSize.h) + " " + cudalink::common::pixelFormatName(targetFormat) +
                        " codec " + (dst.codec.valid ? "ok" : "invalid") + ", src " +
                        (src.data == nullptr ? "null" : "set") + ")";
                    if (priorStatus != cudaSuccess) {
                        myError +=
                            std::string("; an earlier CUDA call left: ") + cudaGetErrorString(priorStatus);
                    }
                    if (cudalink::common::isStickyCudaError(convertStatus)) {
                        myFatal = true;
                    } else {
                        cudaGetLastError();
                    }
                    return;
                }
                CUDALINK_CUDA_CHECK_FATAL(cudaMemcpy2DToArrayAsync(outputInfo->cudaArray, 0, 0, scratch,
                                                                   outRowBytes, outRowBytes, outSize.h,
                                                                   cudaMemcpyDeviceToDevice, myStream),
                                          myError, myFatal);
            }

            if (timersArmed) {
                myCopyTimer.recordStop(copyTimerIdx, myStream); // event C
            }
            // myCopyUs below stays CPU-side enqueue cost only (the copy itself is async on
            // myStream); the true GPU-side numbers are the event_wait_us/gpu_copy_us pair
            // recorded just above.
            workEnd = std::chrono::steady_clock::now();
            myCopyUs = std::chrono::duration<float, std::micro>(workEnd - beginEnd).count();
        }
        myEndUs =
            std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - workEnd).count();

        // Step 6. Slot meta of the frame now on the output -- updated only after a successful
        // copy, so the Info values never describe a frame that is not shown.
        recordSlotMeta(result);
        myLastWriteIdx = result.write_idx;
        ++myFrameCount;
        myStatus =
            std::to_string(outSize.w) + "x" + std::to_string(outSize.h) + (convert ? " (converted)" : "");
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
    return kNumInfoCHOPChans;
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
                // Sender's TD absolute frame of the shown frame, mod 2^24 (exact in float32; see
                // kInfoChopFrameMod). Equal across streams = the frames come from one sender cook.
                chan->name->setString("producer_frame");
                chan->value = static_cast<float>(myProducerFrame % kInfoChopFrameMod);
                break;
            case 14:
                chan->name->setString("stream_id");
                chan->value = static_cast<float>(myStreamId);
                break;
            case 15:
                // Publish-to-copy time: now minus the slot meta host_timestamp (same clock on
                // both sides, see common/clock.h). 0 when the sender wrote no timestamp.
                chan->name->setString("latency_us");
                chan->value = myLatencyUs;
                break;
            case 16:
                // Session count of shown frames whose write_idx step was > 1: frames the ring
                // overwrote before this receiver read them.
                chan->name->setString("frame_gap_count");
                chan->value = static_cast<float>(myFrameGapCount);
                break;
            case 17:
                // Session count of shown frames with write_idx step 1 but producer_frame step > 1:
                // the sender skipped TD frames (dropped frames or a sender that does not publish
                // every cook). Not counted when the sender writes no producer_frame (0).
                chan->name->setString("producer_skip_count");
                chan->value = static_cast<float>(myProducerSkipCount);
                break;
            case 18:
                // Session count of no-frame cooks that skipped the Framewaitms wait: the next
                // publish was not expected within the budget, or the sender was paused.
                chan->name->setString("frame_wait_skipped");
                chan->value = static_cast<float>(myFrameWaitSkipCount);
                break;
            case 19:
                // 1 while the Common page resizes or converts the wire frame (off the
                // fast path); gpu_copy_us then includes the kernel.
                chan->name->setString("converted");
                chan->value = myConverted ? 1.0f : 0.0f;
                break;
            default:
                break;
        }
    } catch (...) { // NOLINT(bugprone-empty-catch) -- deliberate ABI fence, see comment above
    }
}

bool CudaLinkInTOP::getInfoDATSize(TD::OP_InfoDATSize* infoSize, void*) {
    infoSize->rows = kNumInfoDATRows; // see kNumInfoDATRows
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
        } else if (index == 5) {
            // Full 64-bit value (the Info CHOP channel shows it mod 2^24).
            entries->values[0]->setString("producer_frame");
            entries->values[1]->setString(std::to_string(myProducerFrame).c_str());
        } else if (index == 6) {
            // Optional producer time in us, not interpreted by cuda-link (the C++ Out TOP writes 0).
            entries->values[0]->setString("source_timestamp_us");
            entries->values[1]->setString(std::to_string(mySourceTimestampUs).c_str());
        } else if (index >= kNumOwnInfoDATRows) {
            cudalink::common::fillBuildInfoDATRow(index - kNumOwnInfoDATRows, entries);
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
