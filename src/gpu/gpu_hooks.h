// SPDX-License-Identifier: GPL-3.0-or-later
// Interface between the Gnm HLE layer (PM4 interpreter) and the GPU backend. No Vulkan types here so bbhle stays
// independent of the renderer; the application wires the hooks at start-up.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

namespace bb::gpu {

// Guest memory written by the port itself on the CPU (PM4 WRITE_DATA/DMA_DATA/DUMP_CONST_RAM, CPU-executed helper shaders): the
// write watch (write_watch.h) re-polls these ranges even within one submit.
inline std::mutex g_local_writes_mutex;
inline std::vector<std::pair<uint64_t, uint64_t>> g_local_writes;
inline bool g_local_writes_overflow = false;  // too many to keep: the write watch re-polls everything
inline std::atomic<bool> g_local_writes_any{false};  // set with the first entry: readers skip the lock while there is none

// Debug: BB_GPU_WATCH=<hex guest address>[,<hex bytes>] logs every GPU-side write (PM4 WRITE_DATA/DMA_DATA/DUMP_CONST_RAM, shader stores,
// storage images and render targets over the range; the backend also logs sampled T#s there). Every such write is also recorded for the
// write watch (above).
inline bool gpu_watched(uint64_t dst, uint64_t bytes) {
    static const char* const env = std::getenv("BB_GPU_WATCH");
    static const uint64_t w = env ? std::strtoull(env, nullptr, 16) : 0;
    static const uint64_t wn = env && std::strchr(env, ',') ? std::strtoull(std::strchr(env, ',') + 1, nullptr, 16) : 1;
    return w && dst < w + wn && w < dst + bytes;
}
inline void gpu_watch(const char* what, uint64_t dst, uint64_t bytes, const char* detail = "") {
    if (gpu_watched(dst, bytes)) std::fprintf(stderr, "gpu-watch: %s writes [0x%llx, +0x%llx) %s\n", what, (unsigned long long)dst, (unsigned long long)bytes, detail);
#ifdef _WIN32  // only the Windows write watch consumes them (elsewhere the vertex cache hashes)
    std::lock_guard<std::mutex> lk(g_local_writes_mutex);
    if (g_local_writes.size() >= 65536) { g_local_writes.clear(); g_local_writes_overflow = true; }
    g_local_writes.emplace_back(dst, dst + bytes);
    g_local_writes_any.store(true, std::memory_order_release);
#endif
}

// Pointers into the register shadow maintained by the PM4 interpreter (valid only during the hook call).
struct RegView {
    const uint32_t* sh;       // SET_SH_REG space (0xB000 + 4*i): shader program/user-data registers
    const uint32_t* context;  // SET_CONTEXT_REG space (0x28000 + 4*i)
    const uint32_t* uconfig;  // SET_UCONFIG_REG space (0x30000 + 4*i)
};

struct DrawCmd {
    bool indexed = false;
    uint32_t count = 0;        // vertices / indices
    uint32_t instances = 1;
    uint64_t index_addr = 0;   // indexed only (guest address)
    uint32_t index_bytes = 2;  // 2 or 4
    bool host_indices = false;  // index_addr is host memory owned by the backend (generated tessellation grid), not a guest address
};

struct Hooks {
    bool (*mem_valid)(uint64_t addr, uint64_t size) = nullptr;  // guest range is mapped and readable
    void (*draw)(const RegView&, const DrawCmd&) = nullptr;
    void (*dispatch)(const RegView&, uint32_t x, uint32_t y, uint32_t z) = nullptr;
    void (*end_submit)() = nullptr;                              // all command buffers of one submit were processed
    void (*gds_copy)(bool to_gds, uint64_t mem, uint32_t gds_off, uint32_t bytes) = nullptr;  // GDS <-> guest memory, in stream order
    // make pending GPU writes to a guest range visible to the CPU (may wait); why (statistics): 4 indirect arguments, 5 WRITE_DATA,
    // 6 DMA_DATA destination, 7 DMA_DATA source, 8 LOAD_CONST_RAM, 9 DUMP_CONST_RAM
    void (*sync_read)(uint64_t addr, uint64_t bytes, int why) = nullptr;
    // DMA_DATA memory -> memory: true if the backend recorded the copy in stream order (its source has GPU writes still pending), so
    // the PM4 scan must not copy on the CPU; false: copy on the CPU now (after sync_read)
    bool (*mem_copy)(uint64_t dst, uint64_t src, uint64_t bytes) = nullptr;
    // DISPATCH_INDIRECT: true if the backend recorded it with the arguments read on the GPU at `args` (pending GPU writes there);
    // false: the PM4 scan reads the arguments and calls dispatch
    bool (*dispatch_indirect)(const RegView& r, uint64_t args) = nullptr;
    void (*label)(uint64_t addr, uint64_t value, uint32_t bytes) = nullptr;  // completion label: written once the preceding GPU work ran
    void (*hw_watch)(const uint64_t* addrs, int n) = nullptr;  // debug (BB_HWWATCH_VS): re-arm the hardware write watchpoints (up to 4) on every thread
};

inline Hooks& hooks() {
    static Hooks h;
    return h;
}

}  // namespace bb::gpu
