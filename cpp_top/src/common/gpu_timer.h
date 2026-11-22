// GpuTimerRing -- fixed-size ring of paired timing-enabled CUDA events for deferred,
// non-blocking GPU interval measurement on a single stream (shared by both TOPs' Debug-gated
// GPU-side timing; see PLAN Phase C).
//
// The existing cook_us/copy_us/begin_us/end_us channels are CPU-side steady_clock readings
// around *asynchronous enqueue* calls, so the actual GPU cost of the D2D copies (which scales
// 4x from uint8 to float32) never appears in any of them. This ring closes that blind spot:
// record a start/stop pair on the stream around the work, then read the interval back with
// cudaEventElapsedTime on a LATER cook -- reading the current cook's own pair would return
// cudaErrorNotReady (or force a sync, which this class never does). cudaEventQuery() guards
// every read; a not-ready pair just skips that cook's sample, it is never waited on.
//
// The ring is 3 deep so the pair being read is always kRingSize cooks old: one cook of slack
// beyond double-buffering absorbs WDDM/driver batching latency without dropping samples.
//
// These events are purely local scratch (created timing-enabled, with neither
// cudaEventDisableTiming nor cudaEventInterprocess) -- unrelated to the TOPs' interprocess
// mySlotEvents, which explicitly disable timing and so can never feed cudaEventElapsedTime.
// They are also independent of the IPC slot buffers, so they survive reallocate() untouched.
//
// Every method degrades silently: a CUDA failure leaves the affected pair invalid and the
// sample skipped. Nothing here ever sets a TOP's myError/myFatal -- timing is diagnostics,
// never a reason to fail a cook.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "raii_handles.h"

namespace cudalink::common {

class GpuTimerRing {
public:
    static constexpr uint32_t kRingSize = 3;

    // Lazily creates all events on the first call; no-ops (returns true) once created. Safe
    // to call every cook. Returns false and leaves the ring unusable if any creation fails --
    // callers should treat that as "timing unavailable" (log once, skip record/read).
    bool ensureCreated() {
        if (myCreated) {
            return true;
        }
        for (uint32_t i = 0; i < kRingSize; ++i) {
            cudaEvent_t startEvt = nullptr;
            if (cudaEventCreate(&startEvt) != cudaSuccess) {
                return false;
            }
            myStart[i].reset(startEvt);
            cudaEvent_t stopEvt = nullptr;
            if (cudaEventCreate(&stopEvt) != cudaSuccess) {
                return false;
            }
            myStop[i].reset(stopEvt);
        }
        myCreated = true;
        return true;
    }

    [[nodiscard]] bool created() const { return myCreated; }

    // The ring index the NEXT recordStart() will overwrite -- i.e. the pair that is kRingSize
    // cooks old right now. Callers must tryRead(peekReadIdx()) BEFORE recordStart(), which
    // invalidates that same slot for reuse.
    [[nodiscard]] uint32_t peekReadIdx() const { return myWriteIdx; }

    // Records the start event of the next ring slot on 'stream' and returns that slot's index
    // for the matching recordStop(). A failed record leaves the pair invalid (never read).
    [[nodiscard]] uint32_t recordStart(cudaStream_t stream) {
        const uint32_t idx = myWriteIdx;
        myWriteIdx = (myWriteIdx + 1) % kRingSize;
        myValid[idx] = cudaEventRecord(myStart[idx].get(), stream) == cudaSuccess;
        myStopped[idx] = false;
        return idx;
    }

    // Records the stop event for slot 'idx' (from recordStart()) on 'stream', arming the pair
    // for a later tryRead().
    void recordStop(uint32_t idx, cudaStream_t stream) {
        if (myValid[idx]) {
            myStopped[idx] = cudaEventRecord(myStop[idx].get(), stream) == cudaSuccess;
        }
    }

    // Non-blocking readback of slot 'idx': returns true and writes the interval (microseconds
    // -- cudaEventElapsedTime's float milliseconds converted here, once, centrally) only when
    // both events were recorded and the stop event has already completed on the GPU. Any other
    // outcome returns false with no waiting -- the caller just keeps its previous value.
    bool tryRead(uint32_t idx, float* outUs) {
        if (!myCreated || !myValid[idx] || !myStopped[idx]) {
            return false;
        }
        if (cudaEventQuery(myStop[idx].get()) != cudaSuccess) {
            return false;
        }
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, myStart[idx].get(), myStop[idx].get()) != cudaSuccess) {
            return false;
        }
        *outUs = ms * 1000.0f;
        myValid[idx] = false; // consumed -- never re-read a stale pair
        return true;
    }

private:
    bool myCreated = false;
    uint32_t myWriteIdx = 0;
    CudaEventGuard myStart[kRingSize];
    CudaEventGuard myStop[kRingSize];
    bool myValid[kRingSize] = {};
    bool myStopped[kRingSize] = {};
};

} // namespace cudalink::common
