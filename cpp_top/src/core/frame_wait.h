// frame_wait.h -- timeout math for the In TOP's Framewaitms doorbell wait. Pure, so
// topcore_tests covers it; CudaLinkInTOP::waitForFreshFrame() is the only caller.

#pragma once

#include <cstdint>

namespace cudalink::core {

// Longest single doorbell wait. The doorbell is an auto-reset event, which wakes ONE waiter:
// a second consumer of the same Ipcmemname (a second In TOP, or a Python receiver with
// CUDALINK_DOORBELL) can take the signal meant for this one. The wait therefore runs in
// slices and re-polls write_idx after each, so a lost signal costs at most one slice instead
// of the whole budget.
constexpr uint32_t kFrameWaitSliceMs = 1;

// Timeout (ms) for the next doorbell wait, from the time left in the Framewaitms budget.
// Rounds UP, so a remainder under 1 ms still waits: the earlier truncating conversion made
// Framewaitms=1 never wait and every budget about 1 ms short. Returns 0 once the budget is
// spent. The total wait is therefore >= the budget and overshoots it by under one slice plus
// the OS wake latency (TD runs the Windows timer at 1 ms; measured wake ~0.5 ms late).
constexpr uint32_t next_wait_slice_ms(int64_t remaining_us, uint32_t slice_ms = kFrameWaitSliceMs) noexcept {
    if (remaining_us <= 0 || slice_ms == 0) {
        return 0;
    }
    const uint64_t ms = (static_cast<uint64_t>(remaining_us) + 999U) / 1000U;
    return ms < slice_ms ? static_cast<uint32_t>(ms) : slice_ms;
}

} // namespace cudalink::core
