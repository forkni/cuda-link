// CudaLinkOutTOP -- sender TOP: publishes a texture into shared memory for zero-copy
// CUDA IPC consumption by other processes/TOPs.
// opType "Cudalinkout", 1 input, TOP_ExecuteMode::CUDA.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "CPlusPlus_Common.h"
#include "TOP_CPlusPlusBase.h"

#include "../common/bench_accumulator.h"
#include "../common/debug_log.h"
#include "../common/gpu_timer.h"
#include "../core/ring_writer.h"
#include "../core/shm_layout.h"
#include "pixel_format_map.h"

class CudaLinkOutTOP : public TD::TOP_CPlusPlusBase {
public:
    CudaLinkOutTOP(const TD::OP_NodeInfo* info, TD::TOP_Context* context);
    ~CudaLinkOutTOP() override;

    // This class owns raw HANDLE/cudaStream_t/IPC resources with a user-declared
    // destructor -- an implicitly-defaulted copy would shallow-copy those and double-free
    // them in ~CudaLinkOutTOP(). TD constructs this class exactly once and never copies or
    // moves it, so deleting all four is correct, not just defensive.
    CudaLinkOutTOP(const CudaLinkOutTOP&) = delete;
    CudaLinkOutTOP& operator=(const CudaLinkOutTOP&) = delete;
    CudaLinkOutTOP(CudaLinkOutTOP&&) = delete;
    CudaLinkOutTOP& operator=(CudaLinkOutTOP&&) = delete;

    void getGeneralInfo(TD::TOP_GeneralInfo* ginfo, const TD::OP_Inputs* inputs, void* reserved1) override;
    void execute(TD::TOP_Output* output, const TD::OP_Inputs* inputs, void* reserved1) override;

    int32_t getNumInfoCHOPChans(void* reserved1) override;
    void getInfoCHOPChan(int32_t index, TD::OP_InfoCHOPChan* chan, void* reserved1) override;
    bool getInfoDATSize(TD::OP_InfoDATSize* infoSize, void* reserved1) override;
    void getInfoDATEntries(int32_t index, int32_t nEntries, TD::OP_InfoDATEntries* entries,
                           void* reserved1) override;

    void getErrorString(TD::OP_String* error, void* reserved1) override;
    void getWarningString(TD::OP_String* warning, void* reserved1) override;
    void getInfoPopupString(TD::OP_String* info, void* reserved1) override;

    void setupParameters(TD::OP_ParameterManager* manager, void* reserved1) override;

private:
    // Diffs Active/Ipcmemname/Numslots against their cached values at the top of every
    // execute() -- no push-based parameter-change notification exists in the Custom TOP
    // API, so this is the only way to detect an edit (same approach as
    // CudaLinkInTOP::checkParameterChanges).
    void checkParameterChanges(const TD::OP_Inputs* inputs);

    // Publishes shutdown_flag=1 (if a segment is open) and frees every IPC/device/SHM
    // resource. Called on Active On->Off, on parameter changes that require a fresh
    // session, and from the destructor.
    void teardown();

    // (Re)allocates device buffers/events, exports their IPC handles, creates/re-creates
    // the named SHM mapping + doorbell, and writes the initial header + slot handles +
    // metadata. Called on first cook and whenever resolution/format/numslots changes.
    // Returns false (myError set) on any allocation/export failure.
    bool reallocate(uint32_t width, uint32_t height, const cudalink::out_top::WireFormat& fmt, int numSlots,
                    const char* name);

    // Frees all per-slot graph execs and resets myGraphBuilt -- called before rebuilding
    // device buffers/events in reallocate() (the graphs reference them) and from teardown().
    void destroyGraphs();

    // Attempts the CUDA-Graphs launch path for 'slot' using this cook's srcArray as the
    // IPC-copy source: captures (first use after (re)allocation) or updates-and-relaunches
    // (subsequent cooks) a 2-node graph (memcpy + interprocess event-record) that replaces
    // those two ops' separate WDDM submissions with one cudaGraphLaunch. See the capture-site
    // comment in CudaLinkOutTOP.cpp for the full design/rationale (PLAN-005 §2.1). Returns
    // false and latches myGraphsDisabled on ANY failure -- caller must run the legacy per-op
    // path for this cook instead (a failed capture performs none of the enclosed work).
    bool tryGraphCopy(uint32_t slot, cudaArray_t srcArray, size_t rowBytes, uint32_t height);

    // Debug-gated diagnostics: neither the error/warning badges nor GPU% are enough to
    // diagnose a transient failure (see getErrorString/getWarningString for the sticky
    // myLastError capture, and this for the deeper opt-in trace). Forwards to myDebugLog,
    // which handles the lazy file-open/gating itself; this wrapper just supplies the
    // current frame number so call sites don't have to.
    void debugLog(const std::string& msg) { myDebugLog.log(msg, myFrameCount); }

    TD::TOP_Context* myContext;
    cudaStream_t myStream = nullptr;

    // Latched fatal-error flag: set once by the ctor's device/IPC-capability checks or by
    // a CUDALINK_CUDA_CHECK_FATAL failure on the per-frame hot path. Once set,
    // execute() short-circuits every subsequent cook instead of retrying CUDA calls against a
    // stream/context that documented behavior says may now be corrupted for the rest of the
    // process (cudaDeviceReset() is NOT an option here -- it destroys all CUDA resources in
    // this process, which would also kill TD's own CUDA state).
    bool myFatal = false;

    // Cached parameter values for the change-detection diff above.
    bool myFirstCook = true;
    bool myCachedActive = false;
    std::string myCachedIpcmemname;
    int myCachedNumslots = 0;

    // SHM connection state.
    void* myShmHandle = nullptr;      // HANDLE, void* to avoid <windows.h> in the header
    void* myDoorbellHandle = nullptr; // HANDLE
    uint8_t* myShmView = nullptr;
    cudalink::core::SHMLayout myLayout{0};
    uint64_t myVersion = 0;
    uint32_t myWriteIdx = 0;
    bool myAllocated = false;

    // Cached geometry/format this allocation was built for -- a change in any of these
    // (or Numslots) triggers reallocate().
    uint32_t myWidth = 0;
    uint32_t myHeight = 0;
    cudalink::core::Metadata myMetadata;

    // Per-slot device resources, sized to myLayout.num_slots() once allocated.
    std::vector<void*> mySlotDevPtrs;
    std::vector<cudaEvent_t> mySlotEvents;

    // CUDA Graphs (env-gated PLAN-005 §2.1, default OFF; see the capture-site comment in
    // CudaLinkOutTOP.cpp for the full design). myGraphsRequested is latched once from
    // CUDALINK_CPP_USE_GRAPHS at construction; myGraphsDisabled latches permanently (for the
    // rest of the session) on the first graph-path failure, falling back to the always-correct
    // legacy path. The three vectors are one-per-slot and stay empty until the first
    // reallocate(), which sizes them alongside mySlotDevPtrs/mySlotEvents.
    bool myGraphsRequested = false;
    bool myGraphsDisabled = false;
    std::vector<cudaGraphExec_t> myGraphExecs;
    std::vector<cudaGraphNode_t> myGraphCopyNodes;
    std::vector<bool> myGraphBuilt;

    // Status/error/warning surfaces.
    std::string myError;
    std::string myWarning;
    std::string myStatus = "Idle";

    // Sticky last-error capture (live-test finding: cookEveryFrame=true clears myError
    // every ~16ms, so a real one-cook error's badge vanishes before it can be read).
    // Updated only in getErrorString() -- never auto-cleared, only overwritten -- so it
    // survives across cooks and is visible via the Info DAT without needing the debug log.
    std::string myLastError;
    uint64_t myLastErrorFrame = 0;

    // Non-fatal diagnostic from CudaDeviceSession's ctor-time probe (skipped/failed IPC
    // capability query, WDDM/TCC support-envelope note, ...). Persists for the lifetime of
    // this instance (unlike myError/myWarning) so it stays reachable via the init_note Info
    // DAT row even long after the one-time debugLog() emission on the first cook.
    std::string myInitNote;
    bool myInitNoteLogged = false;

    // Debug-gated diagnostics (see debugLog()). Enabled state is refreshed once per cook
    // from Parameters::evalDebug() so reallocate()/teardown() can check it without needing
    // OP_Inputs threaded through.
    cudalink::common::DebugLogger myDebugLog{"cudalink_out_top_debug.log"};

    // Stats (Info CHOP channels): frames, cook_us, copy_us, begin_us, end_us,
    // write_idx, num_slots.
    uint64_t myFrameCount = 0;
    float myCookUs = 0.0f;
    float myCopyUs = 0.0f;

    // myCopyUs measures only the CUDA-call enqueues inside the CUDA-ops bracket;
    // myBeginUs/myEndUs separately time entering/leaving that bracket
    // (beginCUDAOperations()/endCUDAOperations()) so the two costs can be told apart.
    float myBeginUs = 0.0f;
    float myEndUs = 0.0f;

    // Periodic bench-log accumulator (Debug-gated, reported every 97 frames -- matches the
    // Python exporter's reporting cadence). Shared with CudaLinkInTOP (see bench_accumulator.h).
    cudalink::common::BenchAccumulator myBench;

    // Debug-gated GPU-side timing (see gpu_timer.h): the CPU-side channels above only time
    // async enqueues, so the actual GPU copy cost (4x bigger on float32) is invisible to them.
    // myIpcTimer brackets the IPC-slot copy + interprocess event-record (or, on the graph
    // path, the single cudaGraphLaunch replacing both); myPassTimer brackets the optional
    // pass-through display copy. Both are lazily created on the first Debug-on cook and are
    // slot-independent local events, untouched by reallocate(). When Debug is off, none of
    // this is created or recorded -- the hot path is byte-identical to before.
    cudalink::common::GpuTimerRing myIpcTimer;
    cudalink::common::GpuTimerRing myPassTimer;
    float myGpuIpcUs = 0.0f;
    float myGpuPassUs = 0.0f;
    bool myGpuTimingUnavailable = false; // sticky once-only "timing unavailable" log guard
};
