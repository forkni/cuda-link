// topcore_tests -- golden-byte parity tests for the TD-free, CUDA-free protocol core.
//
// The fixtures in tests/fixtures/ are written by scripts/gen_golden_shm.py through
// src/cuda_link/shm_protocol.py (the source of truth). This binary runs the same
// operations through ring_writer / shm_layout.h and byte-compares, then runs
// ring_reader::acquire_slot on the Python-written buffers.
//
// The scenario values below are mirrored from gen_golden_shm.py. Change both together,
// then regenerate: python scripts/gen_golden_shm.py
//
// Usage: topcore_tests [fixture_dir]   (default: CUDALINK_FIXTURE_DIR from CMake)

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "ring_reader.h"
#include "ring_writer.h"
#include "shm_layout.h"

using namespace cudalink::core; // NOLINT(google-build-using-namespace)

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(expr)                                                                                          \
    do {                                                                                                     \
        ++g_checks;                                                                                          \
        if (!(expr)) {                                                                                       \
            ++g_failures;                                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s [%s]\n", __FILE__, __LINE__, #expr,                \
                         g_ctx.c_str());                                                                     \
        }                                                                                                    \
    } while (0)

std::string g_ctx;
std::filesystem::path g_fixture_dir;

using Bytes = std::vector<uint8_t>;

// --- Scenario values (mirrored from scripts/gen_golden_shm.py) ---------------------------
constexpr uint64_t kVersion = 0x0102030405060708ULL;
constexpr uint32_t kWriteIdx = 6;
constexpr double kTimestamp = 12345.678;
constexpr std::array<uint32_t, 3> kSlotCounts{2, 3, 4};
constexpr std::array<uint32_t, 3> kExpectedSlot{1, 2, 1}; // (6 - 1) % N for N = 2, 3, 4

struct MetaCase {
    const char* name;
    uint32_t width;
    uint32_t height;
    uint32_t num_comps;
    uint8_t kind;
    uint8_t bits;
    uint16_t flags;
};

// Same order as METADATA_CASES in gen_golden_shm.py; kind/bits/flags from DtypeCodec.
constexpr std::array<MetaCase, 9> kMetaCases{{
    {"float32_rgba", 1920, 1080, 4, FORMAT_KIND_FLOAT, 32, 0},
    {"float16_rgba", 1280, 720, 4, FORMAT_KIND_FLOAT, 16, 0},
    {"bfloat16_mono", 640, 480, 1, FORMAT_KIND_FLOAT, 16, FLAGS_BFLOAT16},
    {"uint8_rgba", 1920, 1080, 4, FORMAT_KIND_UNSIGNED, 8, 0},
    {"uint8_bgra", 1920, 1080, 4, FORMAT_KIND_UNSIGNED, 8, FLAGS_BGRA},
    {"uint8_mono_alpha", 512, 512, 2, FORMAT_KIND_UNSIGNED, 8, FLAGS_MONO_ALPHA},
    {"uint16_mono", 1024, 1024, 1, FORMAT_KIND_UNSIGNED, 16, 0},
    {"int8_mono", 256, 256, 1, FORMAT_KIND_SIGNED, 8, 0},
    {"int16_rg", 256, 256, 2, FORMAT_KIND_SIGNED, 16, 0},
}};
constexpr size_t kRingMetaCase = 4; // uint8_bgra

Metadata to_metadata(const MetaCase& c) {
    Metadata m;
    m.width = c.width;
    m.height = c.height;
    m.num_comps = c.num_comps;
    m.format_kind = c.kind;
    m.bits_per_comp = c.bits;
    m.flags = c.flags;
    m.data_size = static_cast<uint32_t>(m.expected_size());
    return m;
}

bool same_metadata(const Metadata& a, const Metadata& b) {
    return a.width == b.width && a.height == b.height && a.num_comps == b.num_comps &&
           a.format_kind == b.format_kind && a.bits_per_comp == b.bits_per_comp && a.flags == b.flags &&
           a.data_size == b.data_size;
}

std::array<uint8_t, IPC_HANDLE_SIZE> mem_handle(uint32_t slot, uint32_t salt = 0) {
    std::array<uint8_t, IPC_HANDLE_SIZE> h{};
    for (uint32_t k = 0; k < IPC_HANDLE_SIZE; ++k) {
        h[k] = static_cast<uint8_t>((slot * 64 + k + 1 + salt) & 0xFFU);
    }
    return h;
}

std::array<uint8_t, IPC_HANDLE_SIZE> event_handle(uint32_t slot, uint32_t salt = 0) {
    std::array<uint8_t, IPC_HANDLE_SIZE> h{};
    for (uint32_t k = 0; k < IPC_HANDLE_SIZE; ++k) {
        h[k] = static_cast<uint8_t>((((slot * 64 + k) * 7 + 0x80 + salt) & 0xFFU));
    }
    return h;
}

Bytes load_fixture(const char* name) {
    const std::filesystem::path path = g_fixture_dir / name;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open fixture %s\n", path.string().c_str());
        ++g_failures;
        return {};
    }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Reports the first differing offset so a failure names the field that moved.
bool bytes_equal(const Bytes& actual, const Bytes& expected) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "  size %zu != fixture size %zu [%s]\n", actual.size(), expected.size(),
                     g_ctx.c_str());
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "  first difference at byte %zu: 0x%02x != fixture 0x%02x [%s]\n", i,
                         actual[i], expected[i], g_ctx.c_str());
            return false;
        }
    }
    return true;
}

// The Out TOP's init sequence (CudaLinkOutTOP::reallocate) without the CUDA calls.
void write_session(Bytes& buf, const SHMLayout& layout, uint64_t version, uint32_t salt = 0) {
    write_init_header(buf.data(), layout, layout.num_slots());
    for (uint32_t slot = 0; slot < layout.num_slots(); ++slot) {
        const auto mem = mem_handle(slot, salt);
        const auto evt = event_handle(slot, salt);
        write_slot_handle(buf.data(), layout, slot, mem.data(), evt.data());
    }
    to_metadata(kMetaCases[kRingMetaCase]).pack_into(buf.data(), layout);
    clear_shutdown(buf.data(), layout);
    commit_version(buf.data(), layout, version);
}

Bytes cpp_ring(uint32_t num_slots, bool do_publish) {
    const SHMLayout layout(num_slots);
    Bytes buf(layout.total_size(), 0);
    write_session(buf, layout, kVersion);
    if (do_publish) {
        publish(buf.data(), layout, kWriteIdx, kTimestamp);
    }
    return buf;
}

// Offsets whose bytes differ between two equal-size buffers.
std::vector<size_t> changed_offsets(const Bytes& before, const Bytes& after) {
    std::vector<size_t> out;
    for (size_t i = 0; i < before.size(); ++i) {
        if (before[i] != after[i]) {
            out.push_back(i);
        }
    }
    return out;
}

bool all_within(const std::vector<size_t>& offsets, size_t begin, size_t end) {
    for (size_t off : offsets) {
        if (off < begin || off >= end) {
            return false;
        }
    }
    return true;
}

// --- Writer parity --------------------------------------------------------------------------

void test_layout_sizes() {
    g_ctx = "layout sizes";
    // Pinned in tests/core/test_shm_protocol.py::test_layout_total_size.
    CHECK(SHMLayout(2).total_size() == 305);
    CHECK(SHMLayout(3).total_size() == 433);
    CHECK(SHMLayout(4).total_size() == 561);
}

void test_writer_ring_fixtures() {
    for (size_t i = 0; i < kSlotCounts.size(); ++i) {
        const uint32_t n = kSlotCounts[i];
        const std::string name = "ring_n" + std::to_string(n) + ".bin";
        g_ctx = "writer " + name;
        CHECK(bytes_equal(cpp_ring(n, true), load_fixture(name.c_str())));
    }
}

void test_writer_fresh_fixture() {
    g_ctx = "writer fresh_n2.bin";
    CHECK(bytes_equal(cpp_ring(2, false), load_fixture("fresh_n2.bin")));
}

void test_writer_shutdown_fixture() {
    g_ctx = "writer shutdown_n3.bin";
    Bytes buf = cpp_ring(3, true);
    set_shutdown(buf.data(), SHMLayout(3));
    CHECK(bytes_equal(buf, load_fixture("shutdown_n3.bin")));
}

void test_metadata_formats() {
    const Bytes records = load_fixture("meta_formats.bin");
    g_ctx = "meta_formats.bin size";
    CHECK(records.size() == kMetaCases.size() * METADATA_SIZE);
    if (records.size() != kMetaCases.size() * METADATA_SIZE) {
        return;
    }
    const SHMLayout layout(2);
    const uint32_t off = layout.metadata_offset();
    for (size_t i = 0; i < kMetaCases.size(); ++i) {
        g_ctx = std::string("metadata ") + kMetaCases[i].name;
        const Metadata expected = to_metadata(kMetaCases[i]);
        const Bytes record(records.begin() + static_cast<std::ptrdiff_t>(i * METADATA_SIZE),
                           records.begin() + static_cast<std::ptrdiff_t>((i + 1) * METADATA_SIZE));

        // Write side: pack_into produces the Python record byte for byte.
        Bytes buf(layout.total_size(), 0);
        expected.pack_into(buf.data(), layout);
        const Bytes packed(buf.begin() + off, buf.begin() + off + METADATA_SIZE);
        CHECK(bytes_equal(packed, record));

        // Read side: read_from decodes the Python record into the same fields.
        Bytes scratch(layout.total_size(), 0);
        std::memcpy(scratch.data() + off, record.data(), METADATA_SIZE);
        CHECK(same_metadata(Metadata::read_from(scratch.data(), layout), expected));
    }
}

// --- Reader classification on Python-written buffers ----------------------------------------

void test_reader_ring_fixtures() {
    for (size_t i = 0; i < kSlotCounts.size(); ++i) {
        const uint32_t n = kSlotCounts[i];
        const std::string name = "ring_n" + std::to_string(n) + ".bin";
        g_ctx = "reader " + name;
        const Bytes buf = load_fixture(name.c_str());
        if (buf.empty()) {
            continue;
        }
        CHECK(read_magic(buf.data()) == PROTOCOL_MAGIC);
        CHECK(read_version(buf.data()) == kVersion);
        CHECK(read_num_slots(buf.data()) == n);
        const SHMLayout layout(read_num_slots(buf.data()));
        CHECK(same_metadata(Metadata::read_from(buf.data(), layout), to_metadata(kMetaCases[kRingMetaCase])));

        // First contact: last_version == 0 adopts the wire version (no VersionChanged).
        const AcquireResult first = acquire_slot(buf.data(), layout, 0, 0);
        CHECK(first.state == SlotState::NewFrame);
        CHECK(first.slot == kExpectedSlot[i]);
        CHECK(first.write_idx == kWriteIdx);
        CHECK(first.timestamp == kTimestamp);

        // Same version, same write_idx: nothing new.
        CHECK(acquire_slot(buf.data(), layout, kWriteIdx, kVersion).state == SlotState::NoFrame);

        // Same version, older write_idx: the frame is new.
        CHECK(acquire_slot(buf.data(), layout, kWriteIdx - 1, kVersion).state == SlotState::NewFrame);

        // A different nonzero last_version: the sender re-initialized.
        const AcquireResult changed = acquire_slot(buf.data(), layout, kWriteIdx, kVersion - 1);
        CHECK(changed.state == SlotState::VersionChanged);
        CHECK(changed.new_version == kVersion);
    }
}

void test_reader_fresh_fixture() {
    g_ctx = "reader fresh_n2.bin (write_idx == 0)";
    const Bytes buf = load_fixture("fresh_n2.bin");
    if (buf.empty()) {
        return;
    }
    const SHMLayout layout(read_num_slots(buf.data()));
    // write_idx == 0 means no frame yet, even when the caller's last_write_idx is also 0.
    CHECK(acquire_slot(buf.data(), layout, 0, 0).state == SlotState::NoFrame);
    CHECK(acquire_slot(buf.data(), layout, 0, kVersion).state == SlotState::NoFrame);
}

void test_reader_shutdown_fixture() {
    g_ctx = "reader shutdown_n3.bin";
    const Bytes buf = load_fixture("shutdown_n3.bin");
    if (buf.empty()) {
        return;
    }
    const SHMLayout layout(read_num_slots(buf.data()));
    CHECK(acquire_slot(buf.data(), layout, 0, 0).state == SlotState::Shutdown);
    // Shutdown wins over a version change: the reader must close handles first.
    CHECK(acquire_slot(buf.data(), layout, 0, kVersion - 1).state == SlotState::Shutdown);
}

void test_reader_slot_wraparound() {
    for (const uint32_t n : kSlotCounts) {
        g_ctx = "slot wraparound N=" + std::to_string(n);
        const SHMLayout layout(n);
        Bytes buf = cpp_ring(n, false);
        for (uint32_t w = 1; w <= 3 * n + 1; ++w) {
            publish(buf.data(), layout, w, kTimestamp);
            const AcquireResult r = acquire_slot(buf.data(), layout, w - 1, kVersion);
            CHECK(r.state == SlotState::NewFrame);
            CHECK(r.slot == (w - 1) % n);
            CHECK(r.slot < n);
        }
    }
}

// --- Regressions recorded in the TOP sources ------------------------------------------------

// CudaLinkInTOP once called acquire_slot with SHMLayout(0). Its shutdown_offset() is byte
// SHM_HEADER_SIZE, the first byte of slot 0's mem handle, so nonzero handle bytes read as
// shutdown on every cook. The layout must come from read_num_slots() first.
void test_regression_zero_slot_layout() {
    g_ctx = "regression: SHMLayout(0) shutdown offset";
    const Bytes buf = load_fixture("ring_n3.bin");
    if (buf.empty()) {
        return;
    }
    const SHMLayout real(read_num_slots(buf.data()));
    CHECK(SHMLayout(0).shutdown_offset() == real.mem_handle_offset(0));
    CHECK(buf[SHMLayout(0).shutdown_offset()] != 0);
    CHECK(acquire_slot(buf.data(), SHMLayout(0), 0, 0).state == SlotState::Shutdown); // the bug
    CHECK(acquire_slot(buf.data(), real, 0, 0).state == SlotState::NewFrame);         // the fix
}

// A receiver that saw VERSION_CHANGED before the slot handles were rewritten opened handles
// to freed memory. Single-threaded byte check of the fix: no re-init step before
// commit_version() touches the version bytes, and only commit_version() makes the change
// visible to acquire_slot.
void test_regression_version_commits_last() {
    g_ctx = "regression: version commits last";
    const SHMLayout layout(2);
    Bytes buf = cpp_ring(2, true);

    Bytes before = buf;
    write_init_header(buf.data(), layout, 2);
    CHECK(all_within(changed_offsets(before, buf), WRITE_IDX_OFFSET, WRITE_IDX_OFFSET + WRITE_IDX_SIZE));

    for (uint32_t slot = 0; slot < 2; ++slot) {
        const auto mem = mem_handle(slot, 11);
        const auto evt = event_handle(slot, 11);
        write_slot_handle(buf.data(), layout, slot, mem.data(), evt.data());
    }
    Metadata other = to_metadata(kMetaCases[0]);
    other.pack_into(buf.data(), layout);
    clear_shutdown(buf.data(), layout);

    CHECK(read_version(buf.data()) == kVersion);
    CHECK(acquire_slot(buf.data(), layout, kWriteIdx, kVersion).state != SlotState::VersionChanged);

    before = buf;
    commit_version(buf.data(), layout, kVersion + 1);
    CHECK(all_within(changed_offsets(before, buf), VERSION_OFFSET, VERSION_OFFSET + VERSION_SIZE));
    const AcquireResult r = acquire_slot(buf.data(), layout, kWriteIdx, kVersion);
    CHECK(r.state == SlotState::VersionChanged);
    CHECK(r.new_version == kVersion + 1);
}

// publish() writes only the timestamp, the shutdown flag and write_idx.
void test_publish_footprint() {
    g_ctx = "publish footprint";
    const SHMLayout layout(3);
    Bytes buf = cpp_ring(3, false);
    buf[layout.shutdown_offset()] = 1;
    const Bytes before = buf;
    publish(buf.data(), layout, kWriteIdx, kTimestamp);
    for (const size_t off : changed_offsets(before, buf)) {
        const bool in_write_idx = off >= WRITE_IDX_OFFSET && off < WRITE_IDX_OFFSET + WRITE_IDX_SIZE;
        const bool in_shutdown = off == layout.shutdown_offset();
        const bool in_timestamp = off >= layout.timestamp_offset() && off < layout.total_size();
        CHECK(in_write_idx || in_shutdown || in_timestamp);
    }
    CHECK(buf[layout.shutdown_offset()] == 0);
}

} // namespace

int main(int argc, char** argv) {
#ifdef CUDALINK_FIXTURE_DIR
    g_fixture_dir = CUDALINK_FIXTURE_DIR;
#endif
    if (argc > 1) {
        g_fixture_dir = argv[1];
    }
    if (g_fixture_dir.empty()) {
        std::fprintf(stderr, "usage: topcore_tests <fixture_dir>\n");
        return 2;
    }

    test_layout_sizes();
    test_writer_ring_fixtures();
    test_writer_fresh_fixture();
    test_writer_shutdown_fixture();
    test_metadata_formats();
    test_reader_ring_fixtures();
    test_reader_fresh_fixture();
    test_reader_shutdown_fixture();
    test_reader_slot_wraparound();
    test_regression_zero_slot_layout();
    test_regression_version_commits_last();
    test_publish_footprint();

    std::printf("topcore_tests: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
