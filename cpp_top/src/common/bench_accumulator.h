// BenchAccumulator -- shared periodic bench-log accumulator for both TOPs' execute() tails.
//
// Both TOPs report the same four per-cook timings (cook/copy/begin/end microseconds) averaged
// over a rolling window of 97 frames (matching the Python exporter's reporting cadence) when
// Debug is on. This holds the four running sums + sample counter so neither TOP has to carry
// its own copy of the accumulate-and-flush logic.

#pragma once

#include <cstdint>
#include <string>

#include "debug_log.h"

namespace cudalink::common {

class BenchAccumulator {
public:
    // Accumulates one cook's timings; once kWindowSize samples have been collected, emits one
    // "bench: avg_..." line via 'logger' (a no-op when logger is disabled -- see
    // DebugLogger::log) and resets the running sums. Callers are expected to only call this
    // from inside their own `if (myDebugLog.enabled())` gate, matching the original inline
    // accumulate-and-flush block this replaces.
    void record(float cookUs, float copyUs, float beginUs, float endUs, uint64_t frameCount,
                DebugLogger& logger) {
        mySumCookUs += cookUs;
        mySumCopyUs += copyUs;
        mySumBeginUs += beginUs;
        mySumEndUs += endUs;
        if (++mySamples >= kWindowSize) {
            logger.log("bench: avg_cook_us=" + std::to_string(mySumCookUs / mySamples) +
                           " avg_copy_us=" + std::to_string(mySumCopyUs / mySamples) +
                           " avg_begin_us=" + std::to_string(mySumBeginUs / mySamples) +
                           " avg_end_us=" + std::to_string(mySumEndUs / mySamples) +
                           " samples=" + std::to_string(mySamples) + " frames=" + std::to_string(frameCount),
                       frameCount);
            mySumCookUs = 0.0f;
            mySumCopyUs = 0.0f;
            mySumBeginUs = 0.0f;
            mySumEndUs = 0.0f;
            mySamples = 0;
        }
    }

    // Extended overload for the Debug-gated GPU-side timing fields: identical to record()
    // above, plus two labeled extras appended to the same log line as "avg_<label>=..."
    // (producer: gpu_ipc_us/gpu_pass_us; receiver: event_wait_us/gpu_copy_us). The extras
    // share mySamples and the four base sums with the plain overload, so a TOP must call
    // exactly one of the two overloads per cook, never mix them -- both TOPs have a single
    // record call site in their execute() tail, which satisfies that by construction.
    void record(float cookUs, float copyUs, float beginUs, float endUs, float extra1, const char* extra1Label,
                float extra2, const char* extra2Label, uint64_t frameCount, DebugLogger& logger) {
        mySumCookUs += cookUs;
        mySumCopyUs += copyUs;
        mySumBeginUs += beginUs;
        mySumEndUs += endUs;
        mySumExtra1 += extra1;
        mySumExtra2 += extra2;
        if (++mySamples >= kWindowSize) {
            logger.log("bench: avg_cook_us=" + std::to_string(mySumCookUs / mySamples) +
                           " avg_copy_us=" + std::to_string(mySumCopyUs / mySamples) +
                           " avg_begin_us=" + std::to_string(mySumBeginUs / mySamples) +
                           " avg_end_us=" + std::to_string(mySumEndUs / mySamples) + " avg_" + extra1Label +
                           "=" + std::to_string(mySumExtra1 / mySamples) + " avg_" + extra2Label + "=" +
                           std::to_string(mySumExtra2 / mySamples) + " samples=" + std::to_string(mySamples) +
                           " frames=" + std::to_string(frameCount),
                       frameCount);
            mySumCookUs = 0.0f;
            mySumCopyUs = 0.0f;
            mySumBeginUs = 0.0f;
            mySumEndUs = 0.0f;
            mySumExtra1 = 0.0f;
            mySumExtra2 = 0.0f;
            mySamples = 0;
        }
    }

private:
    static constexpr uint32_t kWindowSize = 97;
    float mySumCookUs = 0.0f;
    float mySumCopyUs = 0.0f;
    float mySumBeginUs = 0.0f;
    float mySumEndUs = 0.0f;
    float mySumExtra1 = 0.0f;
    float mySumExtra2 = 0.0f;
    uint32_t mySamples = 0;
};

} // namespace cudalink::common
