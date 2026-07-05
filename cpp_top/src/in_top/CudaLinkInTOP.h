// CudaLinkInTOP -- receiver TOP per PLAN-001 D6/D7.
// opType "Cudalinkin", 0 inputs, TOP_ExecuteMode::CUDA.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "CPlusPlus_Common.h"
#include "TOP_CPlusPlusBase.h"

#include "../core/ring_reader.h"
#include "../core/shm_layout.hpp"

class CudaLinkInTOP : public TD::TOP_CPlusPlusBase {
public:
    CudaLinkInTOP(const TD::OP_NodeInfo* info, TD::TOP_Context* context);
    ~CudaLinkInTOP() override;

    void getGeneralInfo(TD::TOP_GeneralInfo* ginfo, const TD::OP_Inputs* inputs, void* reserved1) override;
    void execute(TD::TOP_Output* output, const TD::OP_Inputs* inputs, void* reserved1) override;

    int32_t getNumInfoCHOPChans(void* reserved1) override;
    void getInfoCHOPChan(int32_t index, TD::OP_InfoCHOPChan* chan, void* reserved1) override;
    bool getInfoDATSize(TD::OP_InfoDATSize* infoSize, void* reserved1) override;
    void getInfoDATEntries(int32_t index, int32_t nEntries, TD::OP_InfoDATEntries* entries, void* reserved1) override;

    void getErrorString(TD::OP_String* error, void* reserved1) override;
    void getWarningString(TD::OP_String* warning, void* reserved1) override;
    void getInfoPopupString(TD::OP_String* info, void* reserved1) override;

    void setupParameters(TD::OP_ParameterManager* manager, void* reserved1) override;

private:
    // D6 step 0: parameter-change detection (no push notification exists -- D7).
    void checkParameterChanges(const TD::OP_Inputs* inputs);

    // Full teardown: closes IPC handles, drops the SHM mapping, resets all connection
    // state to fresh-connect defaults. Called on Active On<->Off transitions (D6/D7) and
    // on SHUTDOWN.
    void teardown();

    // Closes only the currently-open IPC mem/event handles (not the SHM mapping itself).
    // Called on VERSION_CHANGED before re-deriving the layout and reopening.
    void closeHandles();

    // Opens the SHM mapping named by 'name'. Returns false (status set to "Waiting for
    // producer") if the mapping doesn't exist yet -- normal, retried every cook.
    bool openSHM(const char* name);

    // Reads num_slots + metadata from the currently-mapped view, (re)builds mLayout,
    // and opens cudaIpcMemHandle/cudaIpcEventHandle for every slot. Covers both the
    // true first-connect case and the VERSION_CHANGED fall-through case uniformly
    // (D6 step 3 "open-once-per-version (cached)").
    bool openSlotHandlesIfNeeded();

    // Validates a wire-sourced num_slots value before it's used to construct a
    // SHMLayout: rejects 0 (division-by-zero in acquire_slot's modulo), rejects
    // anything above a sane cap (SHMLayout's offset math is uint32_t and can silently
    // wrap on pathological inputs), and rejects a layout whose total_size() would read
    // past the actual mapped view (queried via VirtualQuery at open time -- named SHM
    // is attachable by any process that knows the name, so its contents, including
    // num_slots, are treated as untrusted). Returns false (myError set) on rejection.
    bool validateNumSlots(uint32_t numSlots);

    // Debug-gated diagnostics (live-test finding, sender-side investigation applies here
    // too: the receiver's own transient error badge during a sender format/resolution
    // switch disappeared before it could be read). Opens
    // %TEMP%\cudalink_in_top_debug.log lazily on first use, appends, flushes every line.
    // No-op when myDebugEnabled is false.
    void debugLog(const std::string& msg);

    TD::TOP_Context* myContext;
    cudaStream_t myStream = nullptr;

    // Cached parameter values for D6 step 0's change-detection diff.
    bool myFirstCook = true;
    bool myCachedActive = false;
    std::string myCachedIpcmemname;

    // SHM connection state.
    void* myShmHandle = nullptr; // HANDLE, void* to avoid <windows.h> in the header
    uint8_t* myShmView = nullptr;
    size_t myShmMappedSize = 0; // actual mapped region size (VirtualQuery), for bounds checks
    cudalink::core::SHMLayout myLayout{0};
    uint64_t myLastVersion = 0;
    uint32_t myLastWriteIdx = 0;
    bool myHandlesOpen = false;

    // Per-slot IPC resources, sized to myLayout.num_slots() once handles are open.
    std::vector<void*> mySlotDevPtrs;
    std::vector<cudaEvent_t> mySlotEvents;

    cudalink::core::Metadata myMetadata;

    // Status/error/warning surfaces (D7).
    std::string myError;
    std::string myWarning;
    std::string myStatus = "Waiting for producer";

    // Sticky last-error capture (live-test finding: cookEveryFrame=true clears myError
    // every ~16ms, so a real one-cook error's badge -- e.g. during the sender's
    // VERSION_CHANGED window -- vanishes before it can be read). Updated only in
    // getErrorString() -- never auto-cleared, only overwritten.
    std::string myLastError;
    uint64_t myLastErrorFrame = 0;

    // Debug-gated diagnostics (see debugLog()). Cached once per cook from
    // Parameters::evalDebug() so other methods don't need OP_Inputs threaded through.
    bool myDebugEnabled = false;
    std::ofstream myDebugLogFile;

    // Stats (D7 Info CHOP channels): frames, cook_us, copy_us, write_idx/read_slot,
    // ipc_version, and num_slots (the real .tox's read-only receiver-side Numslots
    // display -- surfaced here, not as a parameter, since the Custom TOP API has no
    // mechanism for a plugin to write a parameter's displayed value).
    uint64_t myFrameCount = 0;
    float myCookUs = 0.0f;
    float myCopyUs = 0.0f;
    uint32_t myReadSlot = 0;
};
