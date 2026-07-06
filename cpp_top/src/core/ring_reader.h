// Receiver-side frame acquisition — mirrors shm_protocol.py::acquire_slot() exactly.
// TD-free, CUDA-free. C++20 required for std::atomic_ref.

#pragma once

#include <cstdint>

#include "shm_layout.h"

namespace cudalink::core {

enum class SlotState {
    NoFrame,
    NewFrame,
    Shutdown,
    VersionChanged,
};

struct AcquireResult {
    SlotState state = SlotState::NoFrame;
    uint32_t slot = 0;
    double timestamp = 0.0;
    uint64_t new_version = 0;
    uint32_t write_idx = 0;
};

// Read SHM state and classify the result for the consumer.
//
// - NoFrame: write_idx unchanged; nothing to consume.
// - NewFrame: new frame at .slot; caller reads it and updates last_write_idx to .write_idx.
// - Shutdown: shutdown_flag=1; producer has exited, consumer should clean up.
// - VersionChanged: SHM was re-initialised; caller must reopen IPC handles and re-read
//   num_slots to rebuild SHMLayout (the sender may have picked a different slot count).
//
// 'buf' must point at byte 0 of the mapped SHM region.
// write_idx is read via std::atomic_ref with acquire ordering — this is the paired
// operation to publish_frame()'s release store (shm_protocol.py) and is the only
// formally-correct way to observe it cross-process.
// [[nodiscard]]/noexcept: the returned AcquireResult is the entire point of the call --
// discarding it would silently drop a Shutdown/VersionChanged/NewFrame classification the
// caller must act on. noexcept is accurate: every path below is atomic_ref loads, memcpy,
// and aggregate returns -- nothing that can throw.
[[nodiscard]] AcquireResult acquire_slot(const uint8_t* buf, const SHMLayout& layout, uint32_t last_write_idx,
                                          uint64_t last_version) noexcept;

} // namespace cudalink::core
