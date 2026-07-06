// Sender-side frame publication — mirrors shm_protocol.py's write helpers exactly:
// _write_handles_to_shm (header + slot handles), _write_metadata_to_shm (via
// Metadata::pack_into, shm_layout.h), and publish_frame/set_shutdown/clear_shutdown
// (the acquire/release ordering contract below). TD-free, CUDA-free. C++20 required for
// std::atomic_ref, same as ring_reader.h.

#pragma once

#include <cstdint>

#include "shm_layout.h"

namespace cudalink::core {

// noexcept on every function below: each is pure memcpy/atomic-store/fence work against a
// caller-owned buffer -- nothing here can throw, and declaring that accurately lets callers
// (and the optimizer) rely on it, mirroring ring_reader.h::acquire_slot's identical treatment.

// Writes magic, num_slots, and write_idx=0 at the start of a fresh or re-initialized
// segment. Mirrors shm_protocol.py::SHMLayout.build_buffer (magic/num_slots) plus
// _write_handles_to_shm's own WRITE_IDX_OFFSET reset to 0 -- a fresh session always
// restarts its frame counter.
//
// Deliberately does NOT write 'version' -- see commit_version() below. A receiver may be
// polling concurrently and treats a version change as its sole signal that num_slots and
// everything this function writes is already valid; version must only become visible
// once the whole re-init sequence (this call, write_slot_handle x N, Metadata::pack_into,
// clear_shutdown) has completed, or a receiver landing mid-sequence can observe a new
// version paired with a stale/incomplete wire state (a receiver seeing VERSION_CHANGED
// before slot handles were rewritten would try to open handles to memory the producer had
// already freed -- exactly the bug this split fixes).
//
// 'buf' must point at byte 0 of the mapped SHM region and be at least
// layout.total_size() bytes long. Does not touch shutdown_flag, metadata, slot handles,
// version, or timestamp -- callers write those separately (write_slot_handle,
// Metadata::pack_into, clear_shutdown, commit_version) before the segment is considered
// ready.
void write_init_header(uint8_t* buf, const SHMLayout& layout, uint32_t num_slots) noexcept;

// Publishes a new protocol version -- the "this re-init is now complete and visible"
// signal, extending the acquire/release ordering contract (originally just write_idx) to
// cover VERSION_CHANGED re-inits too. Must be called LAST in any (re)initialization sequence,
// strictly after write_init_header + every write_slot_handle + Metadata::pack_into +
// clear_shutdown for this session have already completed -- everything sequenced before
// this release store is guaranteed visible to any receiver whose acquire load
// (ring_reader's version read) observes the new value, exactly mirroring how publish()
// already guarantees write_idx's readers see the frame data that precedes it.
void commit_version(uint8_t* buf, const SHMLayout& layout, uint64_t version) noexcept;

// Copies a 64-byte cudaIpcMemHandle_t and 64-byte cudaIpcEventHandle_t into slot 'slot'.
// 'mem64' and 'evt64' must each point at IPC_HANDLE_SIZE readable bytes. Mirrors
// shm_protocol.py::Exporter._write_handles_to_shm's per-slot handle copy loop.
void write_slot_handle(uint8_t* buf, const SHMLayout& layout, uint32_t slot, const uint8_t* mem64,
                       const uint8_t* evt64) noexcept;

// Publishes one frame: writes timestamp, clears shutdown_flag, fences, then stores
// write_idx LAST via std::atomic_ref with release ordering.
//
// Ordering is critical (mirroring shm_protocol.py::publish_frame verbatim): the consumer
// reads shutdown_flag BEFORE write_idx, so clearing shutdown_flag before the release store
// ensures the consumer always sees shutdown_flag=0 the first time it observes this
// write_idx. A bare release fence with a separate plain store (the naive C++ transcription)
// is formally UB -- atomic operations are required on both sides ([atomics.fences]);
// std::atomic_ref<uint32_t> is lock-free and address-free, valid for cross-process shared
// memory, and pairs with ring_reader's acquire load on write_idx. Callers must not
// replicate this sequence outside this function -- same rule as the Python original.
void publish(uint8_t* buf, const SHMLayout& layout, uint32_t write_idx, double timestamp) noexcept;

// Signals producer exit: shutdown_flag = 1. Mirrors shm_protocol.py::set_shutdown.
// Call once during producer teardown, before freeing any IPC handles or device buffers --
// the receiver treats SHUTDOWN as close-handles-first, so the flag must already be visible
// before anything it references is freed.
void set_shutdown(uint8_t* buf, const SHMLayout& layout) noexcept;

// Clears shutdown_flag = 0 at init or barrier-skip time. No fence required: callers
// either immediately follow with publish() (which owns the fence), or make no write_idx
// advance. Mirrors shm_protocol.py::clear_shutdown.
void clear_shutdown(uint8_t* buf, const SHMLayout& layout) noexcept;

} // namespace cudalink::core
