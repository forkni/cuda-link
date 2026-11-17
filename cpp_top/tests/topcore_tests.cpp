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

#include "frame_step.h"
#include "frame_wait.h"
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
constexpr uint32_t kSenderPid = 0x0A0B0C0DU;
constexpr uint32_t kWriteIdx = 6;
constexpr std::array<uint32_t, 3> kSlotCounts{2, 3, 4};
constexpr std::array<uint32_t, 3> kExpectedSlot{1, 2, 1}; // (6 - 1) % N for N = 2, 3, 4

// Frame w carries slot_meta_for(w); frames 1..kWriteIdx are published in order.
constexpr double kHostTimestampBase = 12345.678;
constexpr uint64_t kProducerFrameBase = 0x1122334455667700ULL;
constexpr uint64_t kSourceTimestampUsBase = 0x0102030405060700ULL;
constexpr uint32_t kStreamId = 0x0000ABCDU;

SlotMeta slot_meta_for(uint32_t write_idx) {
    SlotMeta m;
    m.host_timestamp = kHostTimestampBase + write_idx;
    m.producer_frame = kProducerFrameBase + write_idx;
    m.source_timestamp_us = kSourceTimestampUsBase + write_idx;
    m.stream_id = kStreamId;
    return m;
}

bool same_slot_meta(const SlotMeta& a, const SlotMeta& b) {
    return a.host_timestamp == b.host_timestamp && a.producer_frame == b.producer_frame &&
           a.source_timestamp_us == b.source_timestamp_us && a.stream_id == b.stream_id &&
           a.meta_flags == b.meta_flags;
}

// The last frame in 1..kWriteIdx that went to 'slot' of an N-slot ring.
uint32_t last_write_idx_in_slot(uint32_t slot, uint32_t num_slots) {
    uint32_t last = 0;
    for (uint32_t w = 1; w <= kWriteIdx; ++w) {
        if ((w - 1) % num_slots == slot) {
            last = w;
        }
    }
    return last;
}

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
    write_init_header(buf.data(), layout, layout.num_slots(), kSenderPid);
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
        for (uint32_t w = 1; w <= kWriteIdx; ++w) {
            publish(buf.data(), layout, w, slot_meta_for(w));
        }
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
    CHECK(SHMLayout(2).total_size() == 368);
    CHECK(SHMLayout(3).total_size() == 528);
    CHECK(SHMLayout(4).total_size() == 688);
    // v0.6: the 64-bit fields are naturally aligned.
    CHECK(VERSION_OFFSET % 8 == 0);
    for (const uint32_t n : kSlotCounts) {
        const SHMLayout layout(n);
        CHECK(layout.total_size() % 8 == 0);
        for (uint32_t slot = 0; slot < n; ++slot) {
            CHECK(layout.slot_meta_offset(slot) % 8 == 0);
        }
    }
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
        CHECK(read_sender_pid(buf.data()) == kSenderPid);
        const SHMLayout layout(read_num_slots(buf.data()));
        CHECK(same_metadata(Metadata::read_from(buf.data(), layout), to_metadata(kMetaCases[kRingMetaCase])));

        // First contact: last_version == 0 adopts the wire version (no VersionChanged).
        const AcquireResult first = acquire_slot(buf.data(), layout, 0, 0);
        CHECK(first.state == SlotState::NewFrame);
        CHECK(first.slot == kExpectedSlot[i]);
        CHECK(first.write_idx == kWriteIdx);
        CHECK(same_slot_meta(first.slot_meta, slot_meta_for(kWriteIdx)));

        // Every slot holds the slot meta of the last frame written to it.
        for (uint32_t slot = 0; slot < n; ++slot) {
            CHECK(same_slot_meta(SlotMeta::read_from(buf.data(), layout, slot),
                                 slot_meta_for(last_write_idx_in_slot(slot, n))));
        }

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

// A v0.5 (cuda-link 1.x) segment keeps magic 0x43495044 at offset 0, so a v0.6 receiver
// can name the version instead of "unrelated process".
void test_v05_segment_magic() {
    g_ctx = "ring_v05_n2.bin";
    const Bytes buf = load_fixture("ring_v05_n2.bin");
    if (buf.empty()) {
        return;
    }
    CHECK(buf.size() == 305); // v0.5 total for N = 2
    CHECK(read_magic(buf.data()) == PROTOCOL_MAGIC_V05);
    CHECK(read_magic(buf.data()) != PROTOCOL_MAGIC);
}

void test_reader_slot_wraparound() {
    for (const uint32_t n : kSlotCounts) {
        g_ctx = "slot wraparound N=" + std::to_string(n);
        const SHMLayout layout(n);
        Bytes buf = cpp_ring(n, false);
        for (uint32_t w = 1; w <= 3 * n + 1; ++w) {
            publish(buf.data(), layout, w, slot_meta_for(w));
            const AcquireResult r = acquire_slot(buf.data(), layout, w - 1, kVersion);
            CHECK(r.state == SlotState::NewFrame);
            CHECK(r.slot == (w - 1) % n);
            CHECK(r.slot < n);
            CHECK(same_slot_meta(r.slot_meta, slot_meta_for(w)));
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
    write_init_header(buf.data(), layout, 2, kSenderPid);
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

// publish() writes only the slot meta of slot (write_idx - 1) % N, the shutdown flag and
// write_idx.
void test_publish_footprint() {
    g_ctx = "publish footprint";
    const SHMLayout layout(3);
    Bytes buf = cpp_ring(3, false);
    buf[layout.shutdown_offset()] = 1;
    const Bytes before = buf;
    publish(buf.data(), layout, kWriteIdx, slot_meta_for(kWriteIdx));
    const uint32_t meta = layout.slot_meta_offset((kWriteIdx - 1) % 3);
    for (const size_t off : changed_offsets(before, buf)) {
        const bool in_write_idx = off >= WRITE_IDX_OFFSET && off < WRITE_IDX_OFFSET + WRITE_IDX_SIZE;
        const bool in_shutdown = off == layout.shutdown_offset();
        const bool in_slot_meta = off >= meta && off < meta + SLOT_META_SIZE;
        CHECK(in_write_idx || in_shutdown || in_slot_meta);
    }
    CHECK(buf[layout.shutdown_offset()] == 0);
}

// The In TOP's frame_gap_count / producer_skip_count classify each shown frame against the previous.
void test_frame_step_classification() {
    // First frame of a connection: nothing to compare, whatever the stale producer_frame.
    CHECK(classify_frame_step(0, 0, 1, 100) == FrameStep::First);
    CHECK(classify_frame_step(0, 999, 7, 100) == FrameStep::First);
    // Steady state: write_idx +1, producer_frame +1.
    CHECK(classify_frame_step(5, 100, 6, 101) == FrameStep::Next);
    // Same producer_frame twice (two publishes in one sender frame): not a skip.
    CHECK(classify_frame_step(5, 100, 6, 100) == FrameStep::Next);
    // Ring overwrote frames before the read.
    CHECK(classify_frame_step(5, 100, 7, 102) == FrameStep::FrameGap);
    CHECK(classify_frame_step(5, 100, 9, 101) == FrameStep::FrameGap);
    // Sender skipped TD frames between two consecutive publishes.
    CHECK(classify_frame_step(5, 100, 6, 102) == FrameStep::ProducerSkip);
    // producer_frame 0 = not set (sender writes none): never a skip.
    CHECK(classify_frame_step(5, 0, 6, 50) == FrameStep::Next);
    CHECK(classify_frame_step(5, 100, 6, 0) == FrameStep::Next);
    // Sender restarted its frame count (smaller value): not a skip.
    CHECK(classify_frame_step(5, 100, 6, 3) == FrameStep::Next);
    // write_idx wraps at 2^32 like the counter.
    CHECK(classify_frame_step(0xFFFFFFFFU, 100, 1, 101) == FrameStep::FrameGap); // step 2 across the wrap
    CHECK(classify_frame_step(0xFFFFFFFEU, 100, 0xFFFFFFFFU, 101) == FrameStep::Next);
    // A repeated write_idx (step 0) is not a gap.
    CHECK(classify_frame_step(5, 100, 5, 100) == FrameStep::Next);
}

// The In TOP's Framewaitms wait: timeout of the next doorbell wait from the time left.
void test_frame_wait_slice() {
    // Budget spent (or overrun): no wait.
    CHECK(next_wait_slice_ms(0) == 0);
    CHECK(next_wait_slice_ms(-1) == 0);
    CHECK(next_wait_slice_ms(-5000) == 0);
    // Regression: a remainder under 1 ms still waits (truncation made Framewaitms=1 a no-op).
    CHECK(next_wait_slice_ms(1) == 1);
    CHECK(next_wait_slice_ms(999) == 1);
    CHECK(next_wait_slice_ms(1000) == 1);
    // Default 1 ms slice caps every wait (a lost auto-reset signal costs one slice).
    CHECK(kFrameWaitSliceMs == 1);
    CHECK(next_wait_slice_ms(1001) == 1);
    CHECK(next_wait_slice_ms(4000) == 1);
    CHECK(next_wait_slice_ms(8000) == 1);
    // Wider slice: rounds up, then caps.
    CHECK(next_wait_slice_ms(1001, 8) == 2);
    CHECK(next_wait_slice_ms(4000, 8) == 4);
    CHECK(next_wait_slice_ms(7999, 8) == 8);
    CHECK(next_wait_slice_ms(100000, 8) == 8);
    // A zero slice never waits (no busy wait of 0 ms timeouts).
    CHECK(next_wait_slice_ms(4000, 0) == 0);
}

// Feeds `n` frames at `period` seconds from t0; returns the time of the last one.
double feed_cadence(PublishCadence& c, double t0, double period, int n) {
    double t = t0;
    for (int i = 0; i < n; ++i) {
        t = t0 + period * i;
        cadence_add(c, t, 1);
    }
    return t;
}

void test_frame_wait_cadence() {
    constexpr double k60 = 1.0 / 60.0;
    constexpr double k30 = 1.0 / 30.0;
    // Unknown cadence (no samples, or fewer than kCadenceMinSamples) always waits.
    PublishCadence empty;
    CHECK(cadence_wait_useful(empty, 100.0, 3));
    PublishCadence few;
    feed_cadence(few, 100.0, k60, 3);
    CHECK(few.samples == 2);
    CHECK(cadence_wait_useful(few, 1000.0, 3));
    // No timestamp (0) leaves the cadence unknown.
    PublishCadence nots;
    for (int i = 0; i < 10; ++i) {
        cadence_add(nots, 0.0, 1);
    }
    CHECK(nots.samples == 0 && nots.last_ts == 0.0);

    // 60 fps sender into a 60 fps receiver: the publish lands 2.7 ms after this cook.
    PublishCadence c60;
    const double last60 = feed_cadence(c60, 100.0, k60, 20);
    CHECK(c60.samples == kCadenceMinSamples);
    CHECK(c60.period_s > 0.0166 && c60.period_s < 0.0167);
    CHECK(cadence_wait_useful(c60, last60 + k60 - 0.0027, 3));
    // Next publish 5 ms away: a 3 ms budget cannot get it.
    CHECK(!cadence_wait_useful(c60, last60 + k60 - 0.005, 3));
    // Overdue publish (late sender, under two periods): wait.
    CHECK(cadence_wait_useful(c60, last60 + 1.5 * k60, 3));
    // Paused sender (newest frame older than two periods): no wait.
    CHECK(!cadence_wait_useful(c60, last60 + 2.5 * k60, 3));
    // Framewaitms 0 is handled by the caller; the guard alone still answers.
    CHECK(!cadence_wait_useful(c60, last60 + k60 - 0.0027, 0));

    // 30 fps sender into 60 fps: the repeat cook one receiver frame after a publish skips.
    PublishCadence c30;
    const double last30 = feed_cadence(c30, 100.0, k30, 20);
    CHECK(c30.period_s > 0.0333 && c30.period_s < 0.0334);
    CHECK(!cadence_wait_useful(c30, last30 + k60, 3));
    // The cook just before the next publish waits.
    CHECK(cadence_wait_useful(c30, last30 + k30 - 0.002, 3));

    // Ring overwrote frames (write_idx step 2): the period stays per publish.
    PublishCadence gap;
    feed_cadence(gap, 100.0, k60, 10);
    cadence_add(gap, gap.last_ts + 2 * k60, 2);
    CHECK(gap.period_s > 0.0166 && gap.period_s < 0.0167);

    // One pause (an outlier over 3 periods) keeps the period.
    PublishCadence pause;
    feed_cadence(pause, 100.0, k60, 10);
    cadence_add(pause, pause.last_ts + 0.2, 1);
    CHECK(pause.period_s > 0.0166 && pause.period_s < 0.0167);
    CHECK(pause.outliers == 1);
    cadence_add(pause, pause.last_ts + k60, 1);
    CHECK(pause.outliers == 0);
    // A gap of 1 s or more is never a sample.
    cadence_add(pause, pause.last_ts + 5.0, 1);
    CHECK(pause.period_s > 0.0166 && pause.period_s < 0.0167);

    // A real rate change (60 -> 10 fps): after kCadenceMaxOutliers long samples the
    // estimate starts over and then learns the new period.
    PublishCadence slow;
    double t = feed_cadence(slow, 100.0, k60, 10);
    for (uint32_t i = 0; i < kCadenceMaxOutliers; ++i) {
        t += 0.1;
        cadence_add(slow, t, 1);
    }
    CHECK(slow.samples == 0);
    for (int i = 0; i < 6; ++i) {
        t += 0.1;
        cadence_add(slow, t, 1);
    }
    CHECK(slow.samples == kCadenceMinSamples);
    CHECK(slow.period_s > 0.099 && slow.period_s < 0.101);
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
    test_v05_segment_magic();
    test_reader_slot_wraparound();
    test_regression_zero_slot_layout();
    test_regression_version_commits_last();
    test_publish_footprint();
    test_frame_step_classification();
    test_frame_wait_slice();
    test_frame_wait_cadence();

    std::printf("topcore_tests: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
