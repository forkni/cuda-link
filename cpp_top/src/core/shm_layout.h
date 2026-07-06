// SHM protocol v0.5.0 layout — verbatim transcription of src/cuda_link/shm_protocol.py.
//
// The Python file is the single source of truth. Any change there must be mirrored
// here; golden-byte tests enforce equality.
// TD-free, CUDA-free: no TD or cudart includes anywhere in this file.
//
// Binary layout (total = SHMLayout(num_slots).total_size()):
//   [0-3]    magic        uint32 LE = PROTOCOL_MAGIC
//   [4-11]   version      uint64 LE (monotonic; incremented each sender init)
//   [12-15]  num_slots    uint32 LE
//   [16-19]  write_idx    uint32 LE (monotonic; 0 = no frames written yet)
//   [20 + slot*128 ...]   IPC handles (64B mem + 64B event per slot, N slots)
//   [20 + N*128]          shutdown_flag uint8
//   [21 + N*128 ...]      metadata 20B (width/height/num_comps/kind/bits/flags/data_size)
//   [41 + N*128 ...]      timestamp float64 LE (producer monotonic timestamp, perf_counter seconds)

#pragma once

#include <cstdint>
#include <cstring>

namespace cudalink::core {

constexpr uint32_t PROTOCOL_MAGIC = 0x43495044; // "CIPD"

constexpr uint32_t MAGIC_OFFSET = 0;
constexpr uint32_t MAGIC_SIZE = 4;
// Deliberately 4-byte-aligned (not 8), matching shm_protocol.py's packed layout exactly --
// re-aligning this field would break golden-byte wire parity with the Python side. This
// means ring_reader.cpp/ring_writer.cpp's std::atomic_ref<uint64_t> over this field runs
// under 4-byte alignment, one short of atomic_ref<uint64_t>::required_alignment (8) --
// formally UB per [atomics.ref.generic]. In practice this is benign on the x86-64 targets
// this project ships for: bytes [4,12) sit entirely inside the first 64-byte cache line
// (offsets 0-63), and the x86-64 SDM guarantees cacheable, naturally-contained unaligned
// 8-byte loads/stores are atomic. See the static_assert next to each atomic_ref<uint64_t>
// use for the accompanying lock-free/address-free compile-time guard.
constexpr uint32_t VERSION_OFFSET = 4;
constexpr uint32_t VERSION_SIZE = 8;
constexpr uint32_t NUM_SLOTS_OFFSET = 12;
constexpr uint32_t NUM_SLOTS_SIZE = 4;
constexpr uint32_t WRITE_IDX_OFFSET = 16;
constexpr uint32_t WRITE_IDX_SIZE = 4;
constexpr uint32_t SHM_HEADER_SIZE = 20; // 4B magic + 8B version + 4B num_slots + 4B write_idx

constexpr uint32_t SLOT_SIZE = 128;      // 64B cudaIpcMemHandle_t + 64B cudaIpcEventHandle_t
constexpr uint32_t IPC_HANDLE_SIZE = 64; // cudaIpc{Mem,Event}Handle_t are each 64 bytes

constexpr uint32_t SHUTDOWN_FLAG_SIZE = 1;
constexpr uint32_t METADATA_SIZE =
    20; // 4B w + 4B h + 4B num_comps + 1B kind + 1B bits + 2B flags + 4B data_size
constexpr uint32_t TIMESTAMP_SIZE = 8;

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

    [[nodiscard]] constexpr uint32_t shutdown_offset() const noexcept {
        return SHM_HEADER_SIZE + num_slots_ * SLOT_SIZE;
    }

    [[nodiscard]] constexpr uint32_t metadata_offset() const noexcept {
        return shutdown_offset() + SHUTDOWN_FLAG_SIZE;
    }

    [[nodiscard]] constexpr uint32_t timestamp_offset() const noexcept {
        return metadata_offset() + METADATA_SIZE;
    }

    [[nodiscard]] constexpr uint32_t total_size() const noexcept {
        return timestamp_offset() + TIMESTAMP_SIZE;
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

} // namespace cudalink::core
