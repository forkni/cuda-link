// CudaLinkInTOP -- receiver TOP: attaches to a sender's shared-memory segment and imports
// its published texture via zero-copy CUDA IPC.
// opType "Cudalinkin", 0 inputs, TOP_ExecuteMode::CUDA.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "CPlusPlus_Common.h"
#include "TOP_CPlusPlusBase.h"

#include "../common/bench_accumulator.h"
#include "../common/cadence_counters.h"
#include "../common/debug_log.h"
#include "../common/gpu_timer.h"
#include "../common/raii_handles.h"
#include "../core/ring_reader.h"
#include "../core/shm_layout.h"

class CudaLinkInTOP : public TD::TOP_CPlusPlusBase {
public:
    CudaLinkInTOP(const TD::OP_NodeInfo* info, TD::TOP_Context* context);
    ~CudaLinkInTOP() override;

    // This class owns raw HANDLE/cudaStream_t/IPC resources with a user-declared
    // destructor -- an implicitly-defaulted copy would shallow-copy those and double-free
    // them in ~CudaLinkInTOP(). TD constructs this class exactly once and never copies or
    // moves it, so deleting all four is correct, not just defensive.
    CudaLinkInTOP(const CudaLinkInTOP&) = delete;
    CudaLinkInTOP& operator=(const CudaLinkInTOP&) = delete;
    CudaLinkInTOP(CudaLinkInTOP&&) = delete;
    CudaLinkInTOP& operator=(CudaLinkInTOP&&) = delete;

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
    // Parameter-change detection -- the Custom TOP API has no push notification for
    // parameter edits, so this diffs the values we care about against last cook's cache.
    void checkParameterChanges(const TD::OP_Inputs* inputs);

    // Full teardown: closes IPC handles, drops the SHM mapping, resets all connection
    // state to fresh-connect defaults. Called on Active On<->Off transitions and on
    // SHUTDOWN.
    void teardown();

    // Closes only the currently-open IPC mem/event handles (not the SHM mapping itself).
    // Called on VERSION_CHANGED before re-deriving the layout and reopening.
    void closeHandles();

    // Opens the SHM mapping named by 'name'. Returns false (status set to "Waiting for
    // producer") if the mapping doesn't exist yet -- normal, retried every cook.
    bool openSHM(const char* name);

    // C4 stutter fix (opt-in via Framewaitms > 0): bounded wait on the producer's publish
    // doorbell when acquire_slot() classified this cook as NoFrame. Re-polls acquire_slot()
    // after every doorbell wake and returns the latest classification -- NewFrame if the
    // producer published within the budget (a "rescued" frame that would otherwise have
    // been a visible repeat), or the original NoFrame on timeout. See the .cpp definition
    // for the auto-reset/stale-signal and single-waiter caveats.
    cudalink::core::AcquireResult waitForFreshFrame(cudalink::core::AcquireResult result, int waitMs);

    // Reads num_slots + metadata from the currently-mapped view, (re)builds mLayout,
    // and opens cudaIpcMemHandle/cudaIpcEventHandle for every slot. Covers both the
    // true first-connect case and the VERSION_CHANGED fall-through case uniformly --
    // handles are opened once per version and cached.
    bool openSlotHandlesIfNeeded();

    // Validates a wire-sourced num_slots value before it's used to construct a
    // SHMLayout: rejects 0 (division-by-zero in acquire_slot's modulo), rejects
    // anything above a sane cap (SHMLayout's offset math is uint32_t and can silently
    // wrap on pathological inputs), and rejects a layout whose total_size() would read
    // past the actual mapped view (queried via VirtualQuery at open time -- named SHM
    // is attachable by any process that knows the name, so its contents, including
    // num_slots, are treated as untrusted). Returns false (myError set) on rejection.
    bool validateNumSlots(uint32_t numSlots);

    // Validates the wire-sourced Metadata read in openSlotHandlesIfNeeded() before it's used
    // to size the D2D copy in execute(): rejects zero width/height/num_comps/bits_per_comp,
    // and rejects a data_size that doesn't match Metadata::expected_size() (width * height *
    // num_comps * bytes-per-comp). An honest sender always writes these consistently
    // (CudaLinkOutTOP::reallocate's Metadata construction); a mismatch means a corrupt or
    // adversarial SHM segment, since named SHM is attachable by any process that knows the
    // name (same untrusted-wire-input stance as validateNumSlots). Returns false (myError
    // set) on rejection.
    bool validateMetadata(const cudalink::core::Metadata& metadata);

    // CUDA Graphs path (opt-in via CUDALINK_CPP_USE_GRAPHS, PLAN-005 task #13 / ADR-0011):
    // folds the per-cook waitEvent + memcpy2DToArray pair into one cudaGraphLaunch via a
    // keyed graph-exec cache. Never calls cudaGraphExecMemcpyNodeSetParams -- exec-level
    // memcpy updates require 1D operands on BOTH sides (see cuda_graphs.py:249's docstring),
    // and this copy's destination is a TD-owned cudaArray_t, so both operands are instead
    // baked into a captured graph per (slot, dstArray) key. On cache hit: launch. On miss:
    // capture+instantiate+launch this cook. Any failure (or cache-cap breach, the signal
    // that TD's output arrays are NOT stable) latches myGraphsDisabled and the caller falls
    // through to the legacy pair the same cook -- no dropped frame, ever.
    bool tryGraphedCopy(uint32_t slot, cudaArray_t dstArray, size_t rowBytes, uint32_t height);

    // Destroys every cached graph exec. Called from closeHandles() (the single funnel for
    // VERSION_CHANGED and teardown()) because every cached exec bakes a slot device pointer
    // AND a slot event that closeHandles() is about to invalidate.
    void destroyGraphs();

    // Debug-gated diagnostics: the receiver's own transient error badge (e.g. during a
    // sender format/resolution switch) can disappear before it's read, so this gives an
    // opt-in persistent trace. Forwards to myDebugLog, which handles the lazy
    // file-open/gating itself; this wrapper just supplies the current frame number.
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

    // SHM connection state.
    void* myShmHandle = nullptr; // HANDLE, void* to avoid <windows.h> in the header

    // Producer's publish doorbell (named auto-reset event, "Local\cudalink_db_<Ipcmemname>"),
    // opened lazily by waitForFreshFrame() on the first Framewaitms-enabled NoFrame cook and
    // held until teardown(). The name depends only on Ipcmemname, so it deliberately survives
    // VERSION_CHANGED (closeHandles()) -- a restarted producer re-creates/reuses the same
    // named object. HANDLE as void*, same convention as myShmHandle.
    void* myDoorbellHandle = nullptr;
    uint8_t* myShmView = nullptr;
    size_t myShmMappedSize = 0; // actual mapped region size (VirtualQuery), for bounds checks
    cudalink::core::SHMLayout myLayout{0};
    uint64_t myLastVersion = 0;
    uint32_t myLastWriteIdx = 0;
    bool myHandlesOpen = false;

    // Per-slot IPC resources, sized to myLayout.num_slots() once handles are open.
    std::vector<void*> mySlotDevPtrs;
    std::vector<cudaEvent_t> mySlotEvents;

    // CUDA Graphs keyed-exec cache (see tryGraphedCopy()). Outer index = ring slot; each
    // entry bakes that slot's devptr+event plus one observed TD output array. The per-slot
    // cap is deliberately small: TD gives no cross-cook stability guarantee for the output
    // cudaArray_t (TOP_CPlusPlusBase.h's createCUDAArray docs), and a cache that keeps
    // growing past a plausible swap-chain depth means the pointer-identity assumption has
    // failed -- hard-disable (never evict) so that failure surfaces instead of being masked.
    struct GraphCacheEntry {
        cudaArray_t dstArray = nullptr;
        cudalink::common::CudaGraphExecGuard exec;
    };
    static constexpr size_t kGraphCacheCapPerSlot = 4;
    bool myGraphsRequested = false; // CUDALINK_CPP_USE_GRAPHS, read once in the ctor
    bool myGraphsDisabled = false;  // sticky latch -> legacy path for the rest of the session
    std::vector<std::vector<GraphCacheEntry>> myGraphCache;

    // Session-lifetime graph counters (Debug-gated increments, mirroring myTotalNoFrame):
    // builds should plateau at (num_slots x observed output arrays) within seconds; a
    // climbing count is the live pre-signal of the cache-cap latch. Feed the
    // graph_hits/graph_builds Info CHOP channels.
    uint64_t myGraphHits = 0;
    uint64_t myGraphBuilds = 0;

    cudalink::core::Metadata myMetadata;

    // Status/error/warning surfaces.
    std::string myError;
    std::string myWarning;
    std::string myStatus = "Waiting for producer";

    // Sticky last-error capture (live-test finding: cookEveryFrame=true clears myError
    // every ~16ms, so a real one-cook error's badge -- e.g. during the sender's
    // VERSION_CHANGED window -- vanishes before it can be read). Updated only in
    // getErrorString() -- never auto-cleared, only overwritten.
    std::string myLastError;
    uint64_t myLastErrorFrame = 0;

    // Debug-gated diagnostics (see debugLog()). Enabled state is refreshed once per cook
    // from Parameters::evalDebug() so other methods don't need OP_Inputs threaded through.
    cudalink::common::DebugLogger myDebugLog{"cudalink_in_top_debug.log"};

    // Stats (Info CHOP channels): frames, cook_us, copy_us, write_idx/read_slot,
    // ipc_version, and num_slots (a read-only receiver-side display -- surfaced here, not
    // as a parameter, since the Custom TOP API has no mechanism for a plugin to write a
    // parameter's displayed value).
    uint64_t myFrameCount = 0;
    float myCookUs = 0.0f;
    float myCopyUs = 0.0f;

    // myCopyUs measures only the wait-event + D2D-copy enqueue inside the CUDA-ops
    // bracket; myBeginUs/myEndUs separately time entering/leaving that bracket
    // (beginCUDAOperations()/endCUDAOperations()) so the two costs can be told apart.
    float myBeginUs = 0.0f;
    float myEndUs = 0.0f;

    // Periodic bench-log accumulator (Debug-gated, reported every 97 frames -- mirrors
    // CudaLinkOutTOP's cadence so sender and receiver logs line up for a side-by-side
    // comparison). Shared with CudaLinkOutTOP (see bench_accumulator.h).
    cudalink::common::BenchAccumulator myBench;

    // Debug-gated GPU-side timing (see gpu_timer.h). myWaitTimer brackets the
    // cudaStreamWaitEvent on the producer's interprocess ready-event -- event_wait_us is
    // how long that wait actually stalled this stream on the GPU, i.e. how late the
    // producer's event fired (the receiver-visible stutter suspect that no CPU-side channel
    // can see). myCopyTimer brackets the D2D copy into the output array (gpu_copy_us).
    // Both are lazily created on the first Debug-on cook; local, slot-independent events,
    // unrelated to the imported interprocess mySlotEvents (which disable timing).
    cudalink::common::GpuTimerRing myWaitTimer;
    cudalink::common::GpuTimerRing myCopyTimer;
    float myEventWaitUs = 0.0f;
    float myGpuCopyUs = 0.0f;
    bool myGpuTimingUnavailable = false; // sticky once-only "timing unavailable" log guard

    // Debug-gated frame-cadence tally (see cadence_counters.h): counts every classified
    // cook -- including the NoFrame/VersionChanged early returns that BenchAccumulator never
    // sees -- because a NoFrame cook re-displays the previous texture (a repeat frame, the
    // visible-stutter mechanism the timing channels can't show). The running totals feed the
    // noframe_count/version_changed_count Info CHOP channels.
    cudalink::common::CadenceCounters myCadence;
    uint64_t myTotalNoFrame = 0;
    uint64_t myTotalVersionChanged = 0;

    // Session-lifetime count of doorbell-rescued frames: cooks initially classified NoFrame
    // that waitForFreshFrame() converted into a fresh NewFrame within the Framewaitms budget
    // (each one a repeat frame the viewer never saw). Debug-gated increments, mirroring
    // myTotalNoFrame; feeds the rescued_count Info CHOP channel.
    uint64_t myTotalRescued = 0;

    uint32_t myReadSlot = 0;
};
