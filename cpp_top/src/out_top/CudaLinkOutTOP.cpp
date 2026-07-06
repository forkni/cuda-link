#include "CudaLinkOutTOP.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <chrono>
#include <cstring>

#include "Parameters.h"
#include "../common/cuda_check.h"
#include "../common/cuda_op_scope.h"
#include "../common/raii_handles.h"

using cudalink::core::Metadata;
using cudalink::core::SHMLayout;
using cudalink::core::clear_shutdown;
using cudalink::core::commit_version;
using cudalink::core::publish;
using cudalink::core::read_version;
using cudalink::core::set_shutdown;
using cudalink::core::write_init_header;
using cudalink::core::write_slot_handle;
using cudalink::out_top::WireFormat;
using cudalink::out_top::mapFromPixelFormat;

namespace {
HANDLE asHandle(void* h) {
    return static_cast<HANDLE>(h);
}

// Names the Sleep(100) grace-period literal used at both teardown() and reallocate()'s
// old-resource-free sites -- a timing-based grace period that gives a receiver time to
// close its imported IPC handles before this process frees the memory they reference
// (see the CUDA IPC undefined-behavior note near its call sites below).
constexpr DWORD kIpcCloseGracePeriodMs = 100;

// Names the Info CHOP channel count returned by getNumInfoCHOPChans() -- must stay in
// sync with the number of `case` labels in getInfoCHOPChan()'s switch below (frames, cook_us,
// copy_us, begin_us, end_us, write_idx, num_slots).
constexpr int32_t kNumInfoCHOPChans = 7;

// Names the Info DAT row count returned by getInfoDATSize() -- must stay in sync with
// the number of index branches in getInfoDATEntries() below (ipc_version, status, last_error,
// last_error_frame).
constexpr int32_t kNumInfoDATRows = 4;

// Monotonic seconds as a double, for the wire timestamp field consumed as a latency
// measurement (importer.py subtracts this from a same-domain "now" to get elapsed time --
// see importer.py's last_latency computation). steady_clock, NOT system_clock/wall-clock:
// on Windows both MSVC's steady_clock and CPython's time.perf_counter() resolve to
// QueryPerformanceCounter(), sharing the same boot-relative zero point, so this value is
// directly comparable to the perf_counter() the Python exporter (exporter.py::publish_frame
// caller) publishes and the Python importer reads back against. A wall-clock/epoch
// timestamp here would still decode fine (same 8-byte double on the wire) but would be
// off by ~ the epoch-vs-boot offset when read against perf_counter() -- exactly the
// multi-hundred-billion-ms latency bug this replaced. Also avoids NTP/DST step
// corruption landing in the 1-30 ms signal band that a wall-clock reading would risk.
double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Naive byte-widening, not a general UTF-8 decoder -- correct for the ASCII SHM names
// this protocol uses in practice (matches CudaLinkInTOP::openSHM's identical technique;
// names are verbatim and unprefixed).
std::wstring widen(const char* s) {
    return {s, s + std::strlen(s)};
}
} // namespace

extern "C" {

DLLEXPORT
void FillTOPPluginInfo(TD::TOP_PluginInfo* info) {
    if (!info->setAPIVersion(TD::TOPCPlusPlusAPIVersion)) return;

    info->executeMode = TD::TOP_ExecuteMode::CUDA;

    TD::OP_CustomOPInfo& customInfo = info->customOPInfo;
    customInfo.opType->setString("Cudalinkout");
    customInfo.opLabel->setString("CUDA Link Out");
    customInfo.opIcon->setString("CLO");
    customInfo.authorName->setString("forkni");
    customInfo.authorEmail->setString("forkni@gmail.com");

    customInfo.minInputs = 1;
    customInfo.maxInputs = 1;

    // Mirrors CudaLinkInTOP: harmless with an input present, and guarantees a first cook
    // even if the upstream input isn't already cooking every frame on its own.
    customInfo.cookOnStart = true;
}

DLLEXPORT
TD::TOP_CPlusPlusBase* CreateTOPInstance(const TD::OP_NodeInfo* info, TD::TOP_Context* context) {
    // A std::bad_alloc (or any other exception) out of the constructor must not cross the
    // ABI back into TD -- report failure as null rather than let it propagate.
    try {
        return new CudaLinkOutTOP(info, context);
    } catch (...) {
        return nullptr;
    }
}

DLLEXPORT
void DestroyTOPInstance(TD::TOP_CPlusPlusBase* instance, TD::TOP_Context* context) {
    // static_cast, not a C-style cast -- this is a safe downcast to the concrete type
    // CreateTOPInstance actually returned.
    delete static_cast<CudaLinkOutTOP*>(instance);
}

} // extern "C"

CudaLinkOutTOP::CudaLinkOutTOP(const TD::OP_NodeInfo*, TD::TOP_Context* context) : myContext(context) {
    // Align the runtime's current device with TD's own CUDA device selection before
    // creating a stream/allocating/exporting any IPC handles, so nothing here silently
    // lands on whatever device happens to be "current" by default (matters on multi-GPU
    // hosts). getCUDADeviceIndex() returns -1 when this node isn't in CUDA execute mode --
    // shouldn't happen for a TOP_ExecuteMode::CUDA plugin, handled defensively rather than
    // asserted against.
    const int cudaDevice = context->getCUDADeviceIndex(nullptr);
    if (cudaDevice >= 0 && cudaSetDevice(cudaDevice) != cudaSuccess) {
        myError = "cudaSetDevice(" + std::to_string(cudaDevice) + ") failed";
        myFatal = true;
        return;
    }

    // Verify the current device actually supports CUDA IPC before this TOP ever attempts
    // to export an IPC handle -- avoids a confusing failure deep inside reallocate() the
    // first time this node cooks. CUDA Runtime API docs: "Users can test their device for
    // IPC functionality by calling cudaDeviceGetAttribute with cudaDevAttrIpcEventSupport."
    int current = 0;
    if (cudaGetDevice(&current) == cudaSuccess) {
        int ipcSupport = 0;
        if (cudaDeviceGetAttribute(&ipcSupport, cudaDevAttrIpcEventSupport, current) == cudaSuccess &&
            ipcSupport == 0) {
            myError = "device " + std::to_string(current) +
                      " does not support CUDA IPC (cudaDevAttrIpcEventSupport == 0)";
            myFatal = true;
            return;
        }
    }

    // Stream creation is checked: an unchecked failure here would leave myStream null,
    // which CUDA silently treats as the default stream (0) instead of surfacing an error.
    //
    // High-priority parity with the Python exporter (exporter.py's high_priority_stream
    // policy creates its IPC stream via cudaStreamCreateWithPriority(..., greatest)): query the
    // device's priority range and request the greatest (numerically least) priority. Per the
    // CUDA docs this is only a scheduling *hint* for pending work -- it cannot preempt work
    // already running and "may not be respected for memory transfers" -- so it is not expected
    // to move the cook-time numbers on its own; it is restored purely for consistency with the
    // original design. Falls back to the plain non-priority creation if the range query fails
    // (older/limited drivers), so this can never turn a working device into a non-functional one.
    cudaError_t st = cudaErrorUnknown;
    int leastPriority = 0;
    int greatestPriority = 0;
    if (cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority) == cudaSuccess) {
        st = cudaStreamCreateWithPriority(&myStream, cudaStreamNonBlocking, greatestPriority);
    }
    if (st != cudaSuccess) {
        st = cudaStreamCreateWithFlags(&myStream, cudaStreamNonBlocking);
    }
    if (st != cudaSuccess) {
        myError = std::string("cudaStreamCreateWithFlags failed: ") + cudaGetErrorString(st);
        myFatal = true;
    }
}

CudaLinkOutTOP::~CudaLinkOutTOP() {
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

void CudaLinkOutTOP::getGeneralInfo(TD::TOP_GeneralInfo* ginfo, const TD::OP_Inputs*, void*) {
    // Publish every cook while Active, independent of whether the input's own contents
    // changed -- keeps write_idx advancing as a heartbeat and matches the receiver's
    // equally unconditional per-cook polling.
    ginfo->cookEveryFrame = true;
}

void CudaLinkOutTOP::setupParameters(TD::OP_ParameterManager* manager, void*) {
    // Parameters::setup() builds parameter strings; a std::bad_alloc must not cross the
    // ABI back into TD.
    try {
        Parameters::setup(manager);
    } catch (...) { // NOLINT(bugprone-empty-catch) -- deliberate ABI fence, see comment above
    }
}

// ---------------------------------------------------------------------------
// Parameter-change detection -- the Custom TOP API has no push notification for
// parameter edits, so this diffs the values we care about against last cook's cache
// (same approach as CudaLinkInTOP::checkParameterChanges).
// ---------------------------------------------------------------------------

void CudaLinkOutTOP::checkParameterChanges(const TD::OP_Inputs* inputs) {
    const bool active = Parameters::evalActive(inputs);
    const char* ipcmemnameRaw = Parameters::evalIpcmemname(inputs);
    const std::string ipcmemname = ipcmemnameRaw ? ipcmemnameRaw : "";
    const int numslots = Parameters::evalNumslots(inputs);

    // Refreshed once per cook so reallocate()/teardown() pick up the current Debug
    // toggle without needing OP_Inputs threaded through (they run from execute() and
    // from the destructor, neither of which otherwise has access to OP_Inputs).
    myDebugLog.setEnabled(Parameters::evalDebug(inputs));

    // Numslots is only meant to be edited while Active=Off; keep it disabled while
    // running so a live change can't be attempted through the UI (it would otherwise
    // silently do nothing until the next Active toggle).
    Parameters::setNumslotsEnabled(inputs, !active);

    if (myFirstCook) {
        myCachedActive = active;
        myCachedIpcmemname = ipcmemname;
        myCachedNumslots = numslots;
        myFirstCook = false;
        return;
    }

    // Active edge (either direction), or Ipcmemname changed while running: full
    // teardown (publishes shutdown first if a session was open) so the next cook starts
    // a fresh connection under whatever the current parameters say (mirrors
    // CudaLinkInTOP::checkParameterChanges' identical two-branch structure).
    if (active != myCachedActive) {
        teardown();
    } else if (ipcmemname != myCachedIpcmemname) {
        teardown();
    }
    // Numslots changes are picked up naturally by execute()'s needsAlloc check without a
    // full teardown -- no producer-exit signal is warranted for a live
    // format/geometry/slot-count change (the version bump alone tells the receiver to
    // reopen via its VERSION_CHANGED path).

    myCachedActive = active;
    myCachedIpcmemname = ipcmemname;
    myCachedNumslots = numslots;
}

// ---------------------------------------------------------------------------
// Teardown -- full stop (producer exit signal, then free everything)
// ---------------------------------------------------------------------------

void CudaLinkOutTOP::teardown() {
    debugLog("teardown: begin (allocated=" + std::string(myAllocated ? "true" : "false") + ")");
    if (myAllocated && myShmView) {
        // Signal producer exit before freeing anything it references: using an imported
        // IPC event after the exporter destroys the original is undefined behavior --
        // the shutdown flag must already be visible to the receiver before that happens.
        set_shutdown(myShmView, myLayout);
        if (myDoorbellHandle) {
            SetEvent(asHandle(myDoorbellHandle));
        }
        // Zero the IPC handle bytes so a receiver that reconnects to a stale mapping
        // (see the CreateFileMappingW name-reuse note in reallocate()) can't attempt to
        // open handles this process is about to invalidate. Mirrors
        // exporter.py::_do_cleanup step 1.
        for (uint32_t slot = 0; slot < myLayout.num_slots(); ++slot) {
            std::memset(myShmView + myLayout.slot_offset(slot), 0, cudalink::core::SLOT_SIZE);
        }
        // Grace period for the receiver to close its imported IPC handles before this
        // process frees the buffers/events they reference -- same 100ms safety margin
        // exporter.py::_do_cleanup uses for the identical UB class (CUDA IPC docs: using
        // an imported handle after the exporter frees the original is undefined).
        Sleep(kIpcCloseGracePeriodMs);
    }

    for (auto* evt : mySlotEvents) {
        if (evt) {
            cudaEventDestroy(evt);
        }
    }
    mySlotEvents.clear();
    for (auto* devPtr : mySlotDevPtrs) {
        if (devPtr) {
            cudaFree(devPtr);
        }
    }
    mySlotDevPtrs.clear();

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

    myLayout = SHMLayout(0);
    myAllocated = false;
    myWidth = 0;
    myHeight = 0;
    myWriteIdx = 0;
    myStatus = "Idle";
    debugLog("teardown: complete");
}

// ---------------------------------------------------------------------------
// (Re)allocation -- builds device buffers/IPC handles/SHM mapping for a new resolution,
// format, or slot count.
// ---------------------------------------------------------------------------

// None of this function's CUDA calls (cudaMalloc, cudaIpcGetMemHandle, cudaEventCreate,
// cudaIpcGetEventHandle, cudaFree) touch a TD-owned Vulkan-interop cudaArray, so they don't
// need to run inside a begin/endCUDAOperations() bracket. Verified against TD's own vendored
// sample (Samples/CPlusPlus/CudaTOP/CudaTOP.cpp in the TouchDesigner install): its ctor calls
// cudaStreamCreate() and its dtor calls cudaStreamDestroy()/cudaDestroySurfaceObject() with
// NO bracket at all; the only CUDA call the sample puts INSIDE begin/endCUDAOperations() is
// setupCudaSurface(), which wraps the cudaArray* obtained from getCUDAArray()/
// createCUDAArray() -- i.e. the bracket's documented purpose ("ensure the order of
// operations between Vulkan and CUDA is properly managed", CPlusPlus_Common.h:676-678)
// applies to operations on that shared interop resource, not to general CUDA resource
// management. This function never touches a cudaArray*, so it's correctly unbracketed;
// execute()'s Step 5 (the actual interop copy) is the only place that needs -- and has --
// the bracket.
bool CudaLinkOutTOP::reallocate(uint32_t width, uint32_t height, const WireFormat& fmt, int numSlots,
                                const char* name) {
    const auto reallocStart = std::chrono::steady_clock::now();
    debugLog("reallocate: begin " + std::to_string(width) + "x" + std::to_string(height) +
             " numSlots=" + std::to_string(numSlots) + " formatKind=" + std::to_string(fmt.format_kind) +
             " bits=" + std::to_string(fmt.bits_per_comp) + " flags=" + std::to_string(fmt.flags));

    // Build the entire NEW session into LOCAL storage first -- the live mySlotDevPtrs/
    // mySlotEvents/myShmView/myShmHandle members are left completely untouched until the
    // new session is fully written and committed below. This fixes a real, live-observed
    // race: the previous implementation freed the OLD device resources and bumped
    // 'version' on the wire BEFORE the new IPC handles/metadata were written, so a
    // receiver polling in that window could see VERSION_CHANGED and try to open handles
    // to memory that was stale or already freed (transient red error badge that
    // self-healed one poll later). On any failure below, only these locals are cleaned
    // up and the TOP keeps running its previous (still-valid, untouched) session.
    // Guard vectors free themselves automatically on any early return below -- no
    // hand-duplicated cleanup lambda needed (see src/common/raii_handles.h).
    std::vector<cudalink::common::CudaDeviceBuffer> newDevPtrs(numSlots);
    std::vector<cudalink::common::CudaEventGuard> newEvents(numSlots);

    const SHMLayout layout(static_cast<uint32_t>(numSlots));
    const size_t itemsize = fmt.bits_per_comp / cudalink::core::BITS_PER_BYTE;
    const size_t bufferSize = static_cast<size_t>(width) * height * fmt.num_comps * itemsize;

    for (int slot = 0; slot < numSlots; ++slot) {
        void* rawDevPtr = nullptr;
        cudaError_t st = cudaMalloc(&rawDevPtr, bufferSize);
        if (st != cudaSuccess) {
            myError = std::string("cudaMalloc failed: ") + cudaGetErrorString(st);
            return false;
        }
        newDevPtrs[slot].reset(rawDevPtr);

        cudaEvent_t rawEvent = nullptr;
        st = cudaEventCreateWithFlags(&rawEvent, cudaEventDisableTiming | cudaEventInterprocess);
        if (st != cudaSuccess) {
            myError = std::string("cudaEventCreateWithFlags failed: ") + cudaGetErrorString(st);
            return false;
        }
        newEvents[slot].reset(rawEvent);
    }

    // Reuse the existing SHM mapping when one is already open -- total_size() depends
    // only on num_slots, and Numslots is disabled while Active
    // (setNumslotsEnabled(!active) in checkParameterChanges), so a live reallocate (this
    // path) never actually needs a different-sized mapping; only a true first connect
    // needs to create one.
    const bool needNewMapping = (myShmView == nullptr);
    // Only populated inside the needNewMapping branch below -- when reusing the existing
    // mapping, this guard stays empty and its destructor is a no-op, so the live
    // myShmHandle/myShmView members are never at risk of being torn down by it.
    cudalink::common::ShmViewGuard newMapping;
    HANDLE mapping = needNewMapping ? nullptr : asHandle(myShmHandle);
    uint8_t* view = needNewMapping ? nullptr : myShmView;
    bool preExisting = false;

    if (needNewMapping) {
        const std::wstring wname = widen(name);
        LARGE_INTEGER sz;
        sz.QuadPart = static_cast<LONGLONG>(layout.total_size());
        HANDLE m = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, sz.HighPart, sz.LowPart,
                                      wname.c_str());
        if (!m) {
            myError = "CreateFileMappingW failed";
            return false;
        }
        // CreateFileMappingW returns a handle to an *existing* section (rather than
        // failing) when a previous/crashed session left one under this name. If so,
        // adopt its version so our upcoming commit is still guaranteed to look different
        // to any receiver that had seen that stale session.
        preExisting = (GetLastError() == ERROR_ALREADY_EXISTS);

        void* v = MapViewOfFile(m, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        if (!v) {
            CloseHandle(m);
            myError = "MapViewOfFile failed";
            return false;
        }
        newMapping = cudalink::common::ShmViewGuard(m, static_cast<uint8_t*>(v));
        mapping = newMapping.mapping();
        view = newMapping.view();
    }

    if (preExisting) {
        const uint64_t priorVersion = read_version(view);
        if (priorVersion > myVersion) {
            myVersion = priorVersion;
        }
    }

    // Write everything the new session needs BEFORE bumping version below. While this
    // executes, 'version' on the wire (when reusing the mapping) still names the OLD
    // session, so a concurrently-polling receiver correctly reports no change yet --
    // ring_writer::commit_version's release store is the only thing that makes this
    // visible.
    write_init_header(view, layout, static_cast<uint32_t>(numSlots));
    for (int slot = 0; slot < numSlots; ++slot) {
        cudaIpcMemHandle_t memHandle;
        // No manual "if (needNewMapping) { UnmapViewOfFile/CloseHandle }" here -- newMapping
        // (and newDevPtrs/newEvents) clean themselves up when this function returns,
        // whether or not a new mapping was created this call.
        cudaError_t st = cudaIpcGetMemHandle(&memHandle, newDevPtrs[slot].get());
        if (st != cudaSuccess) {
            myError = std::string("cudaIpcGetMemHandle failed: ") + cudaGetErrorString(st);
            return false;
        }
        cudaIpcEventHandle_t eventHandle;
        st = cudaIpcGetEventHandle(&eventHandle, newEvents[slot].get());
        if (st != cudaSuccess) {
            myError = std::string("cudaIpcGetEventHandle failed: ") + cudaGetErrorString(st);
            return false;
        }
        // Wire-protocol serialization: both handle structs are trivially-copyable, fixed-size
        // CUDA IPC types (opaque byte blobs by design) written verbatim into the SHM byte
        // buffer -- same category as ring_reader.cpp/ring_writer.cpp's atomic_ref casts.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        const auto* memHandleBytes = reinterpret_cast<const uint8_t*>(&memHandle);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        const auto* eventHandleBytes = reinterpret_cast<const uint8_t*>(&eventHandle);
        write_slot_handle(view, layout, static_cast<uint32_t>(slot), memHandleBytes, eventHandleBytes);
    }

    Metadata newMetadata{width,
                         height,
                         fmt.num_comps,
                         fmt.format_kind,
                         fmt.bits_per_comp,
                         fmt.flags,
                         static_cast<uint32_t>(bufferSize)};
    newMetadata.pack_into(view, layout);
    clear_shutdown(view, layout);

    if (!myDoorbellHandle) {
        const std::wstring wDoorbell = L"Local\\cudalink_db_" + widen(name);
        HANDLE db = CreateEventW(nullptr, FALSE, FALSE, wDoorbell.c_str());
        if (!db) {
            myWarning = "doorbell CreateEventW failed; consumer will fall back to poll-sleep";
        }
        myDoorbellHandle = db; // may remain null on failure -- SetEvent calls check for null (best-effort)
    }

    // Commit: the moment a receiver can observe VERSION_CHANGED (ring_reader's atomic
    // acquire load pairing with this release store), everything written above is already
    // guaranteed durable and visible -- this is the fix's load-bearing line.
    debugLog("reallocate: writes complete, committing version=" + std::to_string(myVersion + 1));
    commit_version(view, layout, ++myVersion);
    if (myDoorbellHandle) {
        SetEvent(asHandle(myDoorbellHandle));
    }

    // Only now swap the new resources into the live members and let the OLD ones go --
    // after a grace period, mirroring teardown()'s identical rationale: give a receiver
    // that was mid-cook against the old session (hasn't yet polled since the commit
    // above) one poll cycle (~16ms, well under 100ms) to notice VERSION_CHANGED and
    // relinquish its old imported handles before this process destroys what they refer to.
    std::vector<void*> oldDevPtrs = std::move(mySlotDevPtrs);
    std::vector<cudaEvent_t> oldEvents = std::move(mySlotEvents);
    // release() hands each guard's resource over to the raw-pointer members without
    // freeing it -- from here on the member vectors own them, same as before.
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
    myShmView = view;
    myShmHandle = mapping;
    newMapping.release(); // ownership transferred to myShmHandle/myShmView above
    myLayout = layout;
    myMetadata = newMetadata;
    myWidth = width;
    myHeight = height;
    myWriteIdx = 0;
    myAllocated = true;

    Sleep(kIpcCloseGracePeriodMs);
    for (auto* evt : oldEvents) {
        if (evt) cudaEventDestroy(evt);
    }
    for (auto* p : oldDevPtrs) {
        if (p) cudaFree(p);
    }

    debugLog("reallocate: complete, elapsed=" +
             std::to_string(
                 std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - reallocStart)
                     .count()) +
             "ms");
    return true;
}

// ---------------------------------------------------------------------------
// execute() -- per-cook: detect parameter changes, (re)allocate on geometry/format
// change, then copy the input texture into the current IPC slot and publish it.
// ---------------------------------------------------------------------------

// Live-tested: confirmed the .tox's own Sender-mode Script TOP also produces no output of
// its own (side-effects into SHM via cudaMemory(), never writes back) -- so a Custom TOP
// that skips output->createCUDAArray() entirely is byte-faithful to the reference and
// shows black/blank without erroring, as expected. Per explicit request, this TOP ALSO
// creates a pass-through output below (mirroring its input) so it can sit inline in an
// operator chain instead of being a dead end -- a deliberate UX improvement over the
// .tox, not a correctness requirement.
void CudaLinkOutTOP::execute(TD::TOP_Output* output, const TD::OP_Inputs* inputs, void*) {
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

        // Step 0/1: cached-diff parameter-change detection + Active gate.
        checkParameterChanges(inputs);
        if (!Parameters::evalActive(inputs)) {
            myStatus = "Idle";
            return;
        }

        // Step 2: read the input TOP + map its pixel format.
        const TD::OP_TOPInput* in = inputs->getInputTOP(0);
        if (!in) {
            myWarning = "no input connected";
            return;
        }
        const WireFormat fmt = mapFromPixelFormat(in->textureDesc.pixelFormat);
        if (!fmt.supported) {
            myWarning = "unsupported input pixel format (id=" +
                        std::to_string(static_cast<int>(in->textureDesc.pixelFormat)) + ")";
            return;
        }
        const uint32_t width = in->textureDesc.width;
        const uint32_t height = in->textureDesc.height;
        if (width == 0 || height == 0) {
            myWarning = "input has zero-sized geometry";
            return;
        }
        const int numSlots = Parameters::evalNumslots(inputs);

        // Step 3: (re)allocate on first cook / resolution / format / numslots change.
        const bool needsAlloc = !myAllocated || width != myWidth || height != myHeight ||
                                numSlots != static_cast<int>(myLayout.num_slots()) ||
                                fmt.format_kind != myMetadata.format_kind ||
                                fmt.bits_per_comp != myMetadata.bits_per_comp ||
                                fmt.flags != myMetadata.flags || fmt.num_comps != myMetadata.num_comps;
        if (needsAlloc) {
            const char* name = Parameters::evalIpcmemname(inputs);
            if (!name || !*name) {
                myError = "Ipcmemname is empty";
                return;
            }
            if (!reallocate(width, height, fmt, numSlots, name)) {
                return; // myError already set
            }
        }

        // Step 4: acquire the input's cudaArray (must be called before
        // beginCUDAOperations(); the returned pointer isn't valid until after).
        TD::OP_CUDAAcquireInfo acquireInfo;
        acquireInfo.stream = myStream;
        const TD::OP_CUDAArrayInfo* arr = in->getCUDAArray(acquireInfo, nullptr);
        if (!arr) {
            myError = "getCUDAArray failed";
            return;
        }

        // Pass-through output array must ALSO be created before beginCUDAOperations() --
        // same "cudaArray* null until beginCUDAOperations()" rule as the input side
        // (TOP_CPlusPlusBase.h:349-352), confirmed by the receiver's own working code
        // (CudaLinkInTOP.cpp), which creates its output array before its begin/end
        // bracket too. An earlier version of this code created it
        // AFTER beginCUDAOperations() had already run, so selfOut->cudaArray was never
        // populated -- passing that null array handle into cudaMemcpy2DToArrayAsync
        // produced cudaErrorInvalidResourceHandle ("pass-through output copy failed:
        // invalid resource handle"), confirmed live.
        TD::TOP_CUDAOutputInfo outInfo;
        outInfo.textureDesc = in->textureDesc;
        outInfo.stream = myStream;
        const TD::OP_CUDAArrayInfo* selfOut = output->createCUDAArray(outInfo, nullptr);

        // Step 5: GPU-side copy of the input texture into the current IPC slot, inside
        // the begin/end CUDA-operations bracket. CudaOpScope ties endCUDAOperations() to
        // the closing brace of this nested block -- scoped tightly to just the CUDA-ops
        // region so a CUDALINK_CUDA_CHECK_FATAL early `return` -- or an exception
        // unwinding through the catch(...) below -- can never leave the bracket
        // unbalanced, and the bracket isn't held open longer than TD's contract actually
        // requires.
        //
        // Three separate timestamps isolate begin/work/end costs: myBeginUs = CudaOpScope
        // construction (beginCUDAOperations()); myCopyUs = the actual CUDA calls below
        // (memcpy/event-record/pass-through); myEndUs = ~CudaOpScope()
        // (endCUDAOperations()), timed via 'workEnd' captured just before the closing
        // brace and read again immediately after it. Every early-return path below (the
        // !cudaOps check, CUDALINK_CUDA_CHECK_FATAL's bare `return;`) exits execute()
        // entirely before the post-block myEndUs line, so 'workEnd' is never read in an
        // unset state.
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

            const uint32_t slot = myWriteIdx % myLayout.num_slots(); // next slot to (over)write
            const size_t itemsize = myMetadata.bits_per_comp / cudalink::core::BITS_PER_BYTE;
            const size_t rowBytes = static_cast<size_t>(myMetadata.width) * myMetadata.num_comps * itemsize;
            CUDALINK_CUDA_CHECK_FATAL(
                cudaMemcpy2DFromArrayAsync(mySlotDevPtrs[slot], rowBytes, arr->cudaArray, 0, 0, rowBytes,
                                           myMetadata.height, cudaMemcpyDeviceToDevice, myStream),
                myError, myFatal);
            // Record the IPC-ready event right after the IPC-slot copy, before the
            // pass-through copy below, so the receiver's cudaStreamWaitEvent isn't delayed
            // by work this TOP does purely for its own on-screen display.
            CUDALINK_CUDA_CHECK_FATAL(cudaEventRecord(mySlotEvents[slot], myStream), myError, myFatal);

            // Pass-through output (this TOP's own display, per explicit request -- see the
            // comment above execute()). Non-fatal on failure -- the IPC export above is this
            // TOP's real job and must not be aborted just because the optional display copy
            // failed. 'selfOut' was created above, before beginCUDAOperations() (see that
            // comment for why the ordering matters).
            //
            // Deliberately does NOT copy directly from 'arr->cudaArray' (source) to
            // 'selfOut->cudaArray' (dest) -- that's an array-to-array copy, which requires
            // cudaMemcpy3DAsync with srcArray/dstArray and returned cudaErrorInvalidValue in
            // live testing (CUDA docs, cuda_runtime_api.h: array-to-array copies require
            // matching "element size" and array-specific depth/extent semantics that are
            // easy to get subtly wrong for a plain 2D array; cudaMemcpy2DArrayToArray would
            // avoid that ambiguity but has no stream parameter, forcing the legacy default
            // stream -- a real perf risk here given the <=90us budget, since the default
            // stream implicitly synchronizes with other streams unless
            // per-thread-default-stream compilation is used, which this project doesn't do).
            //
            // Instead, reuse 'mySlotDevPtrs[slot]' as the source: it already holds a
            // byte-identical LINEAR copy of this frame from the IPC-export copy just above,
            // so this is the exact same cudaMemcpy2DToArrayAsync(dst_array, ..., linear_src,
            // ...) shape the receiver (CudaLinkInTOP.cpp) already uses successfully in
            // production. Same-stream ordering guarantees the IPC-slot write above completes
            // before this read -- no extra event/wait needed.
            if (selfOut) {
                cudaError_t passThroughStatus =
                    cudaMemcpy2DToArrayAsync(selfOut->cudaArray, 0, 0, mySlotDevPtrs[slot], rowBytes,
                                             rowBytes, myMetadata.height, cudaMemcpyDeviceToDevice, myStream);
                if (passThroughStatus != cudaSuccess) {
                    myWarning = std::string("pass-through output copy failed: ") +
                                cudaGetErrorString(passThroughStatus);
                }
            } else {
                myWarning = "createCUDAArray failed (pass-through output unavailable this cook)";
            }

            // Deliberately no CPU-blocking cudaStreamSynchronize here. An earlier revision
            // synced the stream on every cook to protect the rule that the SDK guarantees
            // arr->cudaArray valid only until execute() returns -- but that rule is about
            // the *TD-side* Vulkan<->CUDA handoff, which endCUDAOperations() already orders
            // GPU-side (confirmed: TD's own bundled CudaTOP.cpp sample reads and writes
            // cudaArrays across this exact bracket with no CPU sync at all, and this TOP's
            // own end_us, ~65us measured, is far too cheap to be a stream drain).
            // Cross-process correctness for the receiver is independently guaranteed by the
            // per-slot IPC event recorded above (cudaEventRecord), which the receiver waits
            // on GPU-side via cudaStreamWaitEvent (CudaLinkInTOP.cpp) -- exactly mirroring
            // how this TOP's own In-TOP counterpart already runs with no sync of its own.
            // A blocking-export constraint applies to the *Python* exporter: that sender
            // runs outside any TD begin/endCUDAOperations bracket (src/cuda_link/
            // exporter.py), so it has no GPU-side ordering to lean on and must self-order
            // via record_source_sync(). That constraint does not apply here. If live
            // validation ever turns up torn/gray frames or receiver corruption, do NOT
            // restore this full-stream sync -- fall back to a narrower cudaEventSynchronize
            // on a dedicated copy-done event instead.
            workEnd = std::chrono::steady_clock::now();
            myCopyUs = std::chrono::duration<float, std::micro>(workEnd - beginEnd).count();
        }
        myEndUs =
            std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - workEnd).count();

        // Step 6: publish + doorbell.
        ++myWriteIdx;
        publish(myShmView, myLayout, myWriteIdx, now_seconds());
        if (myDoorbellHandle) {
            SetEvent(asHandle(myDoorbellHandle));
        }

        // Step 7 (stats recorded below, shared with the error path's cookUs update).
        ++myFrameCount;
        myStatus = std::to_string(width) + "x" + std::to_string(height);
    } catch (...) {
        // No exception ever crosses the ABI.
        myError = "unexpected exception in execute()";
    }
    myCookUs = std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - cookStart).count();

    // Periodic bench log (Debug-gated, every 97 frames -- matches the Python exporter's
    // reporting cadence): gives an offline average cook_us/copy_us/begin_us/end_us number
    // without needing to wire and eyeball an Info CHOP live.
    if (myDebugLog.enabled()) {
        mySumCookUs += myCookUs;
        mySumCopyUs += myCopyUs;
        mySumBeginUs += myBeginUs;
        mySumEndUs += myEndUs;
        if (++myBenchSamples >= 97) {
            debugLog("bench: avg_cook_us=" + std::to_string(mySumCookUs / myBenchSamples) +
                     " avg_copy_us=" + std::to_string(mySumCopyUs / myBenchSamples) +
                     " avg_begin_us=" + std::to_string(mySumBeginUs / myBenchSamples) +
                     " avg_end_us=" + std::to_string(mySumEndUs / myBenchSamples) + " samples=" +
                     std::to_string(myBenchSamples) + " frames=" + std::to_string(myFrameCount));
            mySumCookUs = 0.0f;
            mySumCopyUs = 0.0f;
            mySumBeginUs = 0.0f;
            mySumEndUs = 0.0f;
            myBenchSamples = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Info CHOP / DAT / status
// ---------------------------------------------------------------------------

int32_t CudaLinkOutTOP::getNumInfoCHOPChans(void*) {
    return kNumInfoCHOPChans; // frames, cook_us, copy_us, begin_us, end_us, write_idx, num_slots
}

void CudaLinkOutTOP::getInfoCHOPChan(int32_t index, TD::OP_InfoCHOPChan* chan, void*) {
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
                chan->value = static_cast<float>(myWriteIdx);
                break;
            case 6:
                chan->name->setString("num_slots");
                chan->value = static_cast<float>(myLayout.num_slots());
                break;
            default:
                break;
        }
    } catch (...) { // NOLINT(bugprone-empty-catch) -- deliberate ABI fence, see comment above
    }
}

bool CudaLinkOutTOP::getInfoDATSize(TD::OP_InfoDATSize* infoSize, void*) {
    infoSize->rows = kNumInfoDATRows; // ipc_version, status, last_error, last_error_frame
    infoSize->cols = 2;
    infoSize->byColumn = false;
    return true;
}

void CudaLinkOutTOP::getInfoDATEntries(int32_t index, int32_t, TD::OP_InfoDATEntries* entries, void*) {
    // std::to_string()/setString() may allocate; a std::bad_alloc must not cross the ABI.
    try {
        if (index == 0) {
            entries->values[0]->setString("ipc_version");
            entries->values[1]->setString(std::to_string(myVersion).c_str());
        } else if (index == 1) {
            entries->values[0]->setString("status");
            entries->values[1]->setString(myStatus.c_str());
        } else if (index == 2) {
            // Sticky (never auto-cleared, only overwritten) -- see getErrorString().
            // Survives across cooks, unlike the transient error badge (live-test finding).
            entries->values[0]->setString("last_error");
            entries->values[1]->setString(myLastError.c_str());
        } else if (index == 3) {
            entries->values[0]->setString("last_error_frame");
            entries->values[1]->setString(std::to_string(myLastErrorFrame).c_str());
        }
    } catch (...) { // NOLINT(bugprone-empty-catch) -- deliberate ABI fence, see comment above
    }
}

void CudaLinkOutTOP::getErrorString(TD::OP_String* error, void*) {
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

void CudaLinkOutTOP::getWarningString(TD::OP_String* warning, void*) {
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

void CudaLinkOutTOP::getInfoPopupString(TD::OP_String* info, void*) {
    // string copy/setString() may allocate; a std::bad_alloc must not cross the ABI.
    try {
        info->setString(myStatus.c_str());
    } catch (...) { // NOLINT(bugprone-empty-catch) -- deliberate ABI fence, see comment above
    }
}
