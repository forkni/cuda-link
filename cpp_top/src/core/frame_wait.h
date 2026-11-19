// frame_wait.h -- timeout math and the publish-cadence guard for the In TOP's Framewaitms
// doorbell wait. Pure, so topcore_tests covers it; CudaLinkInTOP is the only caller. The
// .tox Receiver has a Python port (td_exporter/TDReceiver.py) with the same rules.

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

// Publish cadence of the sender, from the slot meta host_timestamp of each shown frame
// (seconds on the QueryPerformanceCounter clock, which all processes on a Windows machine
// share). The Framewaitms guard uses it to skip a wait that cannot get a frame: a sender
// slower than the receiver (a 30 fps camera into a 60 fps project) or a paused sender.
struct PublishCadence {
    double last_ts = 0.0;  // host_timestamp of the newest frame shown
    double period_s = 0.0; // smoothed seconds per published frame
    uint32_t samples = 0;  // accepted period samples
    uint32_t outliers = 0; // consecutive samples rejected as too long
};

// Samples before the guard trusts the period; below this the receiver waits as before.
constexpr uint32_t kCadenceMinSamples = 4;
// Consecutive too-long samples (> 3 periods) after which the estimate starts over: the
// sender rate really changed (for example 60 -> 10 fps), it is not one pause.
constexpr uint32_t kCadenceMaxOutliers = 4;
// A gap of 1 s or more is a pause, never a period sample.
constexpr double kCadenceMaxPeriodS = 1.0;
// Slack on the "next publish is within the budget" test: one wait slice.
constexpr double kCadenceSlackS = 0.001;

// Adds the shown frame with host_timestamp `ts`, `idx_step` ring writes after the frame
// shown before it (write_idx step; > 1 when the ring overwrote frames). ts <= 0 (a sender
// that writes no timestamp) leaves the cadence unknown, so the guard always waits.
inline void cadence_add(PublishCadence& c, double ts, uint64_t idx_step) noexcept {
    if (ts <= 0.0) {
        return;
    }
    if (c.last_ts > 0.0 && ts > c.last_ts && idx_step > 0) {
        const double dt = (ts - c.last_ts) / static_cast<double>(idx_step);
        if (dt < kCadenceMaxPeriodS) {
            if (c.samples >= kCadenceMinSamples && dt > 3.0 * c.period_s) {
                if (++c.outliers >= kCadenceMaxOutliers) {
                    c.samples = 0;
                    c.outliers = 0;
                }
            } else {
                c.period_s = c.samples == 0 ? dt : c.period_s + (dt - c.period_s) / 8.0;
                if (c.samples < kCadenceMinSamples) {
                    ++c.samples;
                }
                c.outliers = 0;
            }
        }
    }
    c.last_ts = ts;
}

// True when a Framewaitms wait of wait_ms can get a frame: the next publish is expected
// within the budget (plus one slice), and the newest frame is not older than two periods
// (a paused or stalled sender). An unknown cadence returns true: the classic wait.
inline bool cadence_wait_useful(const PublishCadence& c, double now_s, int wait_ms) noexcept {
    if (c.samples < kCadenceMinSamples || c.period_s <= 0.0 || c.last_ts <= 0.0) {
        return true;
    }
    const double since = now_s - c.last_ts;
    if (since > 2.0 * c.period_s) {
        return false;
    }
    const double ahead = c.period_s - since; // time to the next expected publish
    return ahead <= static_cast<double>(wait_ms) / 1000.0 + kCadenceSlackS;
}

} // namespace cudalink::core
