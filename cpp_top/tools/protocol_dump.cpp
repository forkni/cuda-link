// protocol_dump -- read-only dump of a cuda-link SHM segment (protocol v0.5).
//
// Prints the header, the slot handles (hex), the shutdown flag, the metadata and the
// timestamp, plus what ring_reader::acquire_slot would report to a fresh receiver.
// Use it when a receiver stays on "Waiting for producer" or shows wrong frames.
//
// Usage:
//   protocol_dump <segment_name>        open a live named segment read-only (Windows only)
//   protocol_dump --file <path.bin>     dump a saved image, e.g. a golden fixture
//   add --full to print all 64 bytes of each handle instead of the first 16
//
// Exit codes: 0 dumped, 1 bad arguments, 2 cannot open, 3 too small or bad header.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "ring_reader.h"
#include "shm_layout.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace cudalink::core; // NOLINT(google-build-using-namespace)

namespace {

void print_hex(const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        std::printf("%02x", p[i]);
    }
}

const char* state_name(SlotState s) {
    switch (s) {
        case SlotState::NoFrame:
            return "NoFrame";
        case SlotState::NewFrame:
            return "NewFrame";
        case SlotState::Shutdown:
            return "Shutdown";
        case SlotState::VersionChanged:
            return "VersionChanged";
    }
    return "?";
}

const char* kind_name(uint8_t kind) {
    switch (kind) {
        case FORMAT_KIND_SIGNED:
            return "signed";
        case FORMAT_KIND_UNSIGNED:
            return "unsigned";
        case FORMAT_KIND_FLOAT:
            return "float";
        default:
            return "unknown";
    }
}

// 'size' is the number of readable bytes at 'buf'.
int dump(const uint8_t* buf, size_t size, bool full) {
    if (size < SHM_HEADER_SIZE) {
        std::printf("segment is %zu bytes, smaller than the %u-byte header\n", size, SHM_HEADER_SIZE);
        return 3;
    }
    const uint32_t magic = read_magic(buf);
    const uint64_t version = read_version(buf);
    const uint32_t num_slots = read_num_slots(buf);
    uint32_t write_idx = 0;
    std::memcpy(&write_idx, buf + WRITE_IDX_OFFSET, WRITE_IDX_SIZE);

    std::printf("size        %zu bytes\n", size);
    std::printf("magic       0x%08x %s\n", magic, magic == PROTOCOL_MAGIC ? "(v0.5, ok)" : "(MISMATCH)");
    std::printf("version     %llu\n", static_cast<unsigned long long>(version));
    std::printf("num_slots   %u\n", num_slots);
    std::printf("write_idx   %u%s\n", write_idx, write_idx == 0 ? " (no frame published yet)" : "");
    if (magic != PROTOCOL_MAGIC) {
        return 3;
    }

    // Cap num_slots before building a layout: a garbage header must not drive reads.
    constexpr uint32_t kMaxSlots = 64;
    if (num_slots == 0 || num_slots > kMaxSlots) {
        std::printf("num_slots out of range 1..%u; not reading further\n", kMaxSlots);
        return 3;
    }
    const SHMLayout layout(num_slots);
    if (size < layout.total_size()) {
        std::printf("segment is %zu bytes, layout needs %u\n", size, layout.total_size());
        return 3;
    }

    const size_t shown = full ? IPC_HANDLE_SIZE : 16;
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        std::printf("slot %u mem   ", slot);
        print_hex(buf + layout.mem_handle_offset(slot), shown);
        std::printf("%s\nslot %u event ", full ? "" : "...", slot);
        print_hex(buf + layout.event_handle_offset(slot), shown);
        std::printf("%s\n", full ? "" : "...");
    }

    std::printf("shutdown    %u\n", buf[layout.shutdown_offset()]);

    const Metadata m = Metadata::read_from(buf, layout);
    std::printf("metadata    %ux%u, %u comps, %s %u-bit, flags 0x%04x", m.width, m.height, m.num_comps,
                kind_name(m.format_kind), m.bits_per_comp, m.flags);
    if (m.flags & FLAGS_BFLOAT16) {
        std::printf(" bfloat16");
    }
    if (m.flags & FLAGS_MONO_ALPHA) {
        std::printf(" mono_alpha");
    }
    if (m.flags & FLAGS_BGRA) {
        std::printf(" bgra");
    }
    std::printf("\ndata_size   %u (expected %llu)%s\n", m.data_size,
                static_cast<unsigned long long>(m.expected_size()),
                m.data_size == m.expected_size() ? "" : " MISMATCH");

    double timestamp = 0.0;
    std::memcpy(&timestamp, buf + layout.timestamp_offset(), TIMESTAMP_SIZE);
    std::printf("timestamp   %.6f\n", timestamp);

    const AcquireResult r = acquire_slot(buf, layout, 0, 0);
    std::printf("new reader  %s", state_name(r.state));
    if (r.state == SlotState::NewFrame) {
        std::printf(" slot %u (write_idx %u)", r.slot, r.write_idx);
    }
    std::printf("\n");
    return 0;
}

int dump_file(const char* path, bool full) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::printf("cannot open file %s\n", path);
        return 2;
    }
    const std::vector<uint8_t> data{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    return dump(data.data(), data.size(), full);
}

#ifdef _WIN32
int dump_segment(const char* name, bool full) {
    const std::wstring wname(name, name + std::strlen(name)); // ASCII names, as the TOPs use
    HANDLE mapping = OpenFileMappingW(FILE_MAP_READ, FALSE, wname.c_str());
    if (!mapping) {
        std::printf("cannot open segment '%s' (error %lu): no producer has created it\n", name,
                    GetLastError());
        return 2;
    }
    const void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        std::printf("MapViewOfFile failed (error %lu)\n", GetLastError());
        CloseHandle(mapping);
        return 2;
    }
    // The view spans the whole section, rounded up to a page; RegionSize is readable.
    MEMORY_BASIC_INFORMATION info{};
    const size_t size = VirtualQuery(view, &info, sizeof(info)) != 0 ? info.RegionSize : 0;
    const int rc = dump(static_cast<const uint8_t*>(view), size, full);
    UnmapViewOfFile(view);
    CloseHandle(mapping);
    return rc;
}
#endif

} // namespace

int main(int argc, char** argv) {
    bool full = false;
    const char* file = nullptr;
    const char* segment = nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--full") {
            full = true;
        } else if (arg == "--file" && i + 1 < argc) {
            file = argv[++i];
        } else if (!arg.empty() && arg[0] != '-' && !segment) {
            segment = argv[i];
        } else {
            segment = nullptr;
            file = nullptr;
            break;
        }
    }
    if (file) {
        return dump_file(file, full);
    }
#ifdef _WIN32
    if (segment) {
        return dump_segment(segment, full);
    }
#endif
    std::printf("usage: protocol_dump [--full] <segment_name>   (Windows)\n"
                "       protocol_dump [--full] --file <path.bin>\n");
    return 1;
}
