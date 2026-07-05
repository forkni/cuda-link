#include "ring_writer.h"

#include <atomic>
#include <cstring>

namespace cudalink::core {

void write_init_header(uint8_t* buf, const SHMLayout& layout, uint32_t num_slots) {
    uint32_t magic = PROTOCOL_MAGIC;
    std::memcpy(buf + MAGIC_OFFSET, &magic, MAGIC_SIZE);
    std::memcpy(buf + NUM_SLOTS_OFFSET, &num_slots, NUM_SLOTS_SIZE);
    // A fresh/re-initialized session always restarts its frame counter at 0 (mirrors
    // _write_handles_to_shm's explicit WRITE_IDX_OFFSET reset) -- without this, a stale
    // reader's last_write_idx could numerically collide with this session's counter and
    // silently miss a real first frame (the same hazard D6/R8 documents on the read side).
    const uint32_t zero = 0;
    std::memcpy(buf + WRITE_IDX_OFFSET, &zero, WRITE_IDX_SIZE);
    // version is intentionally NOT written here -- see commit_version() and this
    // function's header comment.
    (void)layout; // layout not needed for header offsets (all fixed, pre-slot); kept for API symmetry
}

void commit_version(uint8_t* buf, const SHMLayout& layout, uint64_t version) {
    // Release barrier: everything the caller wrote before this call (num_slots, slot
    // handles, metadata, shutdown_flag) becomes visible-before the version store below,
    // to any receiver whose acquire load observes it -- the same C3 discipline publish()
    // already applies to write_idx, extended here to cover re-init/VERSION_CHANGED.
    std::atomic_thread_fence(std::memory_order_release);
    auto* version_ptr = reinterpret_cast<uint64_t*>(buf + VERSION_OFFSET);
    std::atomic_ref<uint64_t>(*version_ptr).store(version, std::memory_order_release);
    (void)layout; // version_offset is fixed (VERSION_OFFSET); kept for API symmetry
}

void write_slot_handle(uint8_t* buf, const SHMLayout& layout, uint32_t slot, const uint8_t* mem64,
                       const uint8_t* evt64) {
    std::memcpy(buf + layout.mem_handle_offset(slot), mem64, IPC_HANDLE_SIZE);
    std::memcpy(buf + layout.event_handle_offset(slot), evt64, IPC_HANDLE_SIZE);
}

void publish(uint8_t* buf, const SHMLayout& layout, uint32_t write_idx, double timestamp) {
    std::memcpy(buf + layout.timestamp_offset(), &timestamp, TIMESTAMP_SIZE);
    buf[layout.shutdown_offset()] = 0;
    // C3 release barrier: shutdown_flag (and timestamp) visible before write_idx. Only
    // write_idx itself is promoted to a true atomic object (std::atomic_ref) -- this
    // matches ring_reader.cpp's documented convention that write_idx is the one field
    // with a formal cross-process ordering guarantee today.
    std::atomic_thread_fence(std::memory_order_release);
    auto* write_idx_ptr = reinterpret_cast<uint32_t*>(buf + WRITE_IDX_OFFSET);
    std::atomic_ref<uint32_t>(*write_idx_ptr).store(write_idx, std::memory_order_release);
}

void set_shutdown(uint8_t* buf, const SHMLayout& layout) {
    buf[layout.shutdown_offset()] = 1;
    // Emits a release fence so the flag is visible to consumers before any subsequent
    // write_idx update -- mirrors shm_protocol.py::set_shutdown. Called once during
    // producer teardown, before freeing any IPC handles or device buffers (D6's
    // teardown-ordering note).
    std::atomic_thread_fence(std::memory_order_release);
}

void clear_shutdown(uint8_t* buf, const SHMLayout& layout) {
    // No fence required: callers either immediately follow with publish() (which owns
    // the fence), or make no write_idx advance -- mirrors shm_protocol.py::clear_shutdown.
    buf[layout.shutdown_offset()] = 0;
}

} // namespace cudalink::core
