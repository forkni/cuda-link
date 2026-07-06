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
};
