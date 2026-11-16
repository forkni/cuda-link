// SHM protocol v0.6.0 layout — verbatim transcription of src/cuda_link/shm_protocol.py.
//
// The Python file is the single source of truth. Any change there must be mirrored
// here; golden-byte tests enforce equality.
// TD-free, CUDA-free: no TD or cudart includes anywhere in this file.
//
// Binary layout (total = SHMLayout(num_slots).total_size() = 48 + N*160), little-endian:
//   [0-3]    magic        uint32 = PROTOCOL_MAGIC ("CIP6")
//   [4-7]    num_slots    uint32
//   [8-15]   version      uint64 (monotonic; incremented each sender init; 8-aligned)
//   [16-19]  write_idx    uint32 (monotonic; 0 = no frames written yet)
//   [20-23]  sender_pid   uint32 (sender process id; 0 = unknown)
//   [24 + i*160 ...]      slot i: 64B mem handle, 64B event handle, 32B slot meta
//   [24 + N*160]          shutdown_flag uint8
//   [25 + N*160 ...]      metadata 20B (width/height/num_comps/kind/bits/flags/data_size)
//   [45 + N*160 ...]      3B pad (total is a multiple of 8)
//
// Slot meta, relative to the slot base: +128 host_timestamp f64, +136 producer_frame u64,
// +144 source_timestamp_us u64, +152 stream_id u32, +156 meta_flags u32 (reserved, 0).

#pragma once

#include <cstdint>
#include <cstring>

namespace cudalink::core {

constexpr uint32_t PROTOCOL_MAGIC = 0x43495036;     // "CIP6"
constexpr uint32_t PROTOCOL_MAGIC_V05 = 0x43495044; // "CIPD": v0.5 (cuda-link 1.x), only to name the mismatch

constexpr uint32_t MAGIC_OFFSET = 0;
constexpr uint32_t MAGIC_SIZE = 4;
constexpr uint32_t NUM_SLOTS_OFFSET = 4;
constexpr uint32_t NUM_SLOTS_SIZE = 4;
// 8-aligned, so the std::atomic_ref<uint64_t> in ring_reader.cpp/ring_writer.cpp meets
// atomic_ref<uint64_t>::required_alignment (the mapping base is page-aligned).
constexpr uint32_t VERSION_OFFSET = 8;
constexpr uint32_t VERSION_SIZE = 8;
constexpr uint32_t WRITE_IDX_OFFSET = 16;
constexpr uint32_t WRITE_IDX_SIZE = 4;
constexpr uint32_t SENDER_PID_OFFSET = 20;
constexpr uint32_t SENDER_PID_SIZE = 4;
constexpr uint32_t SHM_HEADER_SIZE = 24; // magic + num_slots + version + write_idx + sender_pid

constexpr uint32_t IPC_HANDLE_SIZE = 64;           // cudaIpc{Mem,Event}Handle_t are each 64 bytes
constexpr uint32_t SLOT_META_OFFSET_IN_SLOT = 128; // after the two IPC handles
constexpr uint32_t SLOT_META_SIZE = 32;
constexpr uint32_t SLOT_SIZE = 160; // 64B mem handle + 64B event handle + 32B slot meta

// Slot-meta field offsets, relative to the start of the slot meta.
constexpr uint32_t SLOT_META_HOST_TIMESTAMP_OFFSET = 0;       // double
constexpr uint32_t SLOT_META_PRODUCER_FRAME_OFFSET = 8;       // uint64
constexpr uint32_t SLOT_META_SOURCE_TIMESTAMP_US_OFFSET = 16; // uint64
constexpr uint32_t SLOT_META_STREAM_ID_OFFSET = 24;           // uint32
constexpr uint32_t SLOT_META_FLAGS_OFFSET = 28;               // uint32, reserved (0)

constexpr uint32_t SHUTDOWN_FLAG_SIZE = 1;
constexpr uint32_t METADATA_SIZE =
    20; // 4B w + 4B h + 4B num_comps + 1B kind + 1B bits + 2B flags + 4B data_size
constexpr uint32_t TRAILER_PAD_SIZE = 3; // pads the total to a multiple of 8

// Names the `/ 8` in every bits_per_comp -> bytes-per-component conversion
// (Metadata::expected_size below; CudaLinkOutTOP/InTOP's itemsize computations), rather
// than a bare magic literal repeated at each call site.
constexpr uint32_t BITS_PER_BYTE = 8;

// DtypeCodec constants (cudaChannelFormatKind)
constexpr uint8_t FORMAT_KIND_SIGNED = 0;
constexpr uint8_t FORMAT_KIND_UNSIGNED = 1;
constexpr uint8_t FORMAT_KIND_FLOAT = 2;
constexpr uint16_t FLAGS_BFLOAT16 = 0x0001;   // bit0: bfloat16 (kind=Float, bits=16)
constexpr uint16_t FLAGS_MONO_ALPHA = 0x0002; // bit1: 2-channel source is mono+alpha, not RG
constexpr uint16_t FLAGS_BGRA = 0x0004;       // bit2: 4-channel uint8 source is BGRA-ordered, not RGBA

// Pre-computed byte offsets for a SHM region with num_slots IPC slots.
// Mirrors shm_protocol.py::SHMLayout exactly.
class SHMLayout {
public:
    explicit constexpr SHMLayout(uint32_t num_slots) : num_slots_(num_slots) {}

    [[nodiscard]] constexpr uint32_t num_slots() const noexcept { return num_slots_; }

    [[nodiscard]] constexpr uint32_t slot_offset(uint32_t i) const noexcept {
        return SHM_HEADER_SIZE + i * SLOT_SIZE;
    }

    [[nodiscard]] constexpr uint32_t mem_handle_offset(uint32_t slot) const noexcept {
        return slot_offset(slot);
    }

    [[nodiscard]] constexpr uint32_t event_handle_offset(uint32_t slot) const noexcept {
        return slot_offset(slot) + IPC_HANDLE_SIZE;
    }

    [[nodiscard]] constexpr uint32_t slot_meta_offset(uint32_t slot) const noexcept {
        return slot_offset(slot) + SLOT_META_OFFSET_IN_SLOT;
    }

    [[nodiscard]] constexpr uint32_t shutdown_offset() const noexcept {
        return SHM_HEADER_SIZE + num_slots_ * SLOT_SIZE;
    }

    [[nodiscard]] constexpr uint32_t metadata_offset() const noexcept {
        return shutdown_offset() + SHUTDOWN_FLAG_SIZE;
    }

    [[nodiscard]] constexpr uint32_t total_size() const noexcept {
        return metadata_offset() + METADATA_SIZE + TRAILER_PAD_SIZE;
    }

private:
    uint32_t num_slots_;
};

// Typed representation of the 20-byte metadata region. Mirrors shm_protocol.py::Metadata.
struct Metadata {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t num_comps = 0;
    uint8_t format_kind = FORMAT_KIND_FLOAT; // cudaChannelFormatKind
    uint8_t bits_per_comp = 0;
    uint16_t flags = 0;
    uint32_t data_size = 0;

    // width * height * num_comps * (bits_per_comp / 8)
    [[nodiscard]] constexpr uint64_t expected_size() const noexcept {
        return static_cast<uint64_t>(width) * height * num_comps * (bits_per_comp / BITS_PER_BYTE);
    }

    // Writes this metadata into a mapped SHM buffer at layout.metadata_offset(). 'buf'
    // must point at byte 0 of the SHM region and be at least layout.total_size() long.
    // Write counterpart of read_from(); mirrors shm_protocol.py::Metadata.pack_into.
    void pack_into(uint8_t* buf, const SHMLayout& layout) const noexcept {
        const uint32_t offset = layout.metadata_offset();
        std::memcpy(buf + offset, &width, 4);
        std::memcpy(buf + offset + 4, &height, 4);
        std::memcpy(buf + offset + 8, &num_comps, 4);
        buf[offset + 12] = format_kind;
        buf[offset + 13] = bits_per_comp;
        std::memcpy(buf + offset + 14, &flags, 2);
        std::memcpy(buf + offset + 16, &data_size, 4);
    }

    // Reads the metadata region out of a mapped SHM buffer at layout.metadata_offset().
    // 'buf' must point at byte 0 of the SHM region and be at least layout.total_size() long.
    static Metadata read_from(const uint8_t* buf, const SHMLayout& layout) noexcept {
        const uint32_t offset = layout.metadata_offset();
        Metadata m;
        std::memcpy(&m.width, buf + offset, 4);
        std::memcpy(&m.height, buf + offset + 4, 4);
        std::memcpy(&m.num_comps, buf + offset + 8, 4);
        m.format_kind = buf[offset + 12];
        m.bits_per_comp = buf[offset + 13];
        std::memcpy(&m.flags, buf + offset + 14, 2);
        std::memcpy(&m.data_size, buf + offset + 16, 4);
        return m;
    }
};

// Per-slot metadata the sender writes with each frame. Mirrors shm_protocol.py::SlotMeta.
struct SlotMeta {
    double host_timestamp = 0.0;      // sender steady_clock / perf_counter seconds at publish
    uint64_t producer_frame = 0;      // sender frame number; 0 = not set
    uint64_t source_timestamp_us = 0; // optional producer time, not interpreted
    uint32_t stream_id = 0;           // user-set stream name
    uint32_t meta_flags = 0;          // reserved, always 0

    // Writes this slot meta at layout.slot_meta_offset(slot).
    void pack_into(uint8_t* buf, const SHMLayout& layout, uint32_t slot) const noexcept {
        uint8_t* p = buf + layout.slot_meta_offset(slot);
        std::memcpy(p + SLOT_META_HOST_TIMESTAMP_OFFSET, &host_timestamp, 8);
        std::memcpy(p + SLOT_META_PRODUCER_FRAME_OFFSET, &producer_frame, 8);
        std::memcpy(p + SLOT_META_SOURCE_TIMESTAMP_US_OFFSET, &source_timestamp_us, 8);
        std::memcpy(p + SLOT_META_STREAM_ID_OFFSET, &stream_id, 4);
        std::memcpy(p + SLOT_META_FLAGS_OFFSET, &meta_flags, 4);
    }

    // Reads the slot meta at layout.slot_meta_offset(slot) (plain reads; call after the
    // acquire load of write_idx).
    static SlotMeta read_from(const uint8_t* buf, const SHMLayout& layout, uint32_t slot) noexcept {
        const uint8_t* p = buf + layout.slot_meta_offset(slot);
        SlotMeta m;
        std::memcpy(&m.host_timestamp, p + SLOT_META_HOST_TIMESTAMP_OFFSET, 8);
        std::memcpy(&m.producer_frame, p + SLOT_META_PRODUCER_FRAME_OFFSET, 8);
        std::memcpy(&m.source_timestamp_us, p + SLOT_META_SOURCE_TIMESTAMP_US_OFFSET, 8);
        std::memcpy(&m.stream_id, p + SLOT_META_STREAM_ID_OFFSET, 4);
        std::memcpy(&m.meta_flags, p + SLOT_META_FLAGS_OFFSET, 4);
        return m;
    }
};

// Header field readers — plain (non-atomic) reads for fields other than write_idx.
// write_idx itself must always be read via std::atomic_ref (see ring_reader.h) — never
// through these helpers, to preserve the acquire/release ordering contract.

inline uint32_t read_magic(const uint8_t* buf) noexcept {
    uint32_t v;
    std::memcpy(&v, buf + MAGIC_OFFSET, 4);
    return v;
}

inline uint64_t read_version(const uint8_t* buf) noexcept {
    uint64_t v;
    std::memcpy(&v, buf + VERSION_OFFSET, 8);
    return v;
}

inline uint32_t read_num_slots(const uint8_t* buf) noexcept {
    uint32_t v;
    std::memcpy(&v, buf + NUM_SLOTS_OFFSET, 4);
    return v;
}

inline uint32_t read_sender_pid(const uint8_t* buf) noexcept {
    uint32_t v;
    std::memcpy(&v, buf + SENDER_PID_OFFSET, 4);
    return v;
}

} // namespace cudalink::core
