// frame_step.h -- classifies the step between two frames a receiver showed, from their
// write_idx and slot meta producer_frame. Pure, so topcore_tests covers it; the In TOP's
// frame_gap_count / producer_skip_count Info channels count its results.

#pragma once

#include <cstdint>

namespace cudalink::core {

enum class FrameStep : uint8_t {
    First,        // no previous frame on this connection (prev_write_idx == 0)
    Next,         // write_idx step 1 and producer_frame step <= 1 (or not set)
    FrameGap,     // write_idx step > 1: the ring overwrote frames before this receiver read them
    ProducerSkip, // write_idx step 1, producer_frame step > 1: the sender skipped TD frames
};

// producer_frame 0 means "not set" (a sender that writes no frame number): it never counts as a
// skip. write_idx wraps at 2^32 like the counter itself.
constexpr FrameStep classify_frame_step(uint32_t prev_write_idx, uint64_t prev_producer_frame,
                                        uint32_t write_idx, uint64_t producer_frame) noexcept {
    if (prev_write_idx == 0) {
        return FrameStep::First;
    }
    const uint32_t write_step = write_idx - prev_write_idx;
    if (write_step > 1) {
        return FrameStep::FrameGap;
    }
    if (write_step == 1 && prev_producer_frame != 0 && producer_frame > prev_producer_frame + 1) {
        return FrameStep::ProducerSkip;
    }
    return FrameStep::Next;
}

} // namespace cudalink::core
