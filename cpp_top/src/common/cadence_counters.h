// CadenceCounters -- receiver-side per-97-cook-window tally of frame-classification outcomes
// (NewFrame / NoFrame / VersionChanged), the Debug-gated counterpart to BenchAccumulator for
// cooks that never reach it.
//
// BenchAccumulator's window counts record() *calls*, and record() only runs at the end of a
// full NewFrame cook -- but the stutter-relevant cooks are exactly the ones that exit early:
// a NoFrame cook returns before the CUDA block and TD silently re-displays the previous
// texture (a repeat frame, i.e. visible judder), and a VersionChanged cook also carries no
// frame. Counting those requires a wall-clock-cook window (every classified cook tallies
// exactly once, whatever the outcome), which is what this class provides. The flushed
// noframe_ratio is the direct signature of cadence stutter: a producer dipping below display
// rate shows up here even when every timing channel looks healthy.

#pragma once

#include <cstdint>
#include <string>

#include "debug_log.h"

namespace cudalink::common {

class CadenceCounters {
public:
    // Rescued: a cook first classified NoFrame that the doorbell wait (CudaLinkInTOP::
    // waitForFreshFrame, opt-in via Framewaitms) converted into a fresh frame within its
    // budget -- displays like a NewFrame (excluded from noframe_ratio) but tracked
    // separately so the log shows how many repeats the wait is actually absorbing.
    enum class Kind : uint8_t { NewFrame, NoFrame, VersionChanged, Rescued };

    // Call once per classified cook (any Kind), only from inside the caller's own
    // `if (myDebugLog.enabled())` gate -- mirrors BenchAccumulator::record()'s contract.
    // Every kWindowSize cooks, emits one "cadence: ..." line via 'logger' and resets.
    void tally(Kind kind, uint64_t frameCount, DebugLogger& logger) {
        ++myTotal;
        if (kind == Kind::NoFrame) {
            ++myNoFrame;
        } else if (kind == Kind::VersionChanged) {
            ++myVersionChanged;
        } else if (kind == Kind::Rescued) {
            ++myRescued;
        }
        if (myTotal >= kWindowSize) {
            const float ratio = static_cast<float>(myNoFrame) / static_cast<float>(myTotal);
            logger.log("cadence: total=" + std::to_string(myTotal) + " noframe=" + std::to_string(myNoFrame) +
                           " version_changed=" + std::to_string(myVersionChanged) + " rescued=" +
                           std::to_string(myRescued) + " noframe_ratio=" + std::to_string(ratio),
                       frameCount);
            myTotal = 0;
            myNoFrame = 0;
            myVersionChanged = 0;
            myRescued = 0;
        }
    }

private:
    // Same window as BenchAccumulator so the "bench:" and "cadence:" lines interleave at a
    // comparable cadence in the log (they will not align exactly -- this window counts every
    // cook, that one only full cooks -- and that difference is the point).
    static constexpr uint32_t kWindowSize = 97;
    uint32_t myTotal = 0;
    uint32_t myNoFrame = 0;
    uint32_t myVersionChanged = 0;
    uint32_t myRescued = 0;
};

} // namespace cudalink::common
