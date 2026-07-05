#include "ring_reader.h"

#include <atomic>
#include <cstring>

namespace cudalink::core {

AcquireResult acquire_slot(const uint8_t* buf, const SHMLayout& layout, uint32_t last_write_idx,
                            uint64_t last_version) {
    // shutdown_flag is still read with a plain load, mirroring the Python producer's
    // actual synchronization discipline today (an OS-level release fence via
    // threading.Lock, not C++ atomics -- see shm_protocol.py's _release_fence comment).
    if (buf[layout.shutdown_offset()] != 0) {
        AcquireResult result;
        result.state = SlotState::Shutdown;
        return result;
    }

    // version IS promoted to a true atomic_ref acquire load -- the paired half of
    // ring_writer::commit_version()'s release store. This closes a real, live-observed
    // race: a plain load here could observe a new version before the writer's preceding
    // num_slots/handle/metadata stores were visible, letting a receiver try to open IPC
    // handles to memory the producer had already reallocated/freed. const_cast is safe:
    // this is a load-only operation, never a store.
    auto* version_ptr = const_cast<uint64_t*>(reinterpret_cast<const uint64_t*>(buf + VERSION_OFFSET));
    const uint64_t version = std::atomic_ref<uint64_t>(*version_ptr).load(std::memory_order_acquire);
    if (version != last_version && last_version != 0) {
        AcquireResult result;
        result.state = SlotState::VersionChanged;
        result.new_version = version;
        return result;
    }

    // Paired acquire load to the writer's release store on write_idx (D4's C3 ordering
    // contract) -- the only formally correct way to observe this field cross-process.
    // const_cast is safe here: this is a load-only operation, never a store.
    auto* write_idx_ptr = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(buf + WRITE_IDX_OFFSET));
    const uint32_t write_idx = std::atomic_ref<uint32_t>(*write_idx_ptr).load(std::memory_order_acquire);

    if (write_idx == 0 || write_idx == last_write_idx) {
        AcquireResult result;
        result.state = SlotState::NoFrame;
        return result;
    }

    const uint32_t slot = (write_idx - 1) % layout.num_slots();

    double timestamp = 0.0;
    std::memcpy(&timestamp, buf + layout.timestamp_offset(), sizeof(timestamp));

    AcquireResult result;
    result.state = SlotState::NewFrame;
    result.slot = slot;
    result.timestamp = timestamp;
    result.write_idx = write_idx;
    return result;
}

} // namespace cudalink::core
