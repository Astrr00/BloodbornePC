// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpu/write_watch.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gpu/gpu_hooks.h"

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace bb::gpu {
namespace {

constexpr uint64_t kPage = 4096;
constexpr uint64_t kChunk = 64 << 10;  // poll state unit: the Windows allocation granularity, so a chunk never spans two allocations
// A poll covers the chunk's whole 1 MiB block when that lies in one write-watched allocation: a call costs ~0.45 us for 64 KiB and
// ~1.2 us for 1 MiB (measured), and the clinic's ~1700 chunk polls per frame fell in ~190 blocks.
constexpr uint64_t kBlock = 1 << 20;
uint64_t g_epoch = 1;
uint64_t g_submit = 1;
#ifdef _WIN32
// Poll state per 64 KiB chunk, with the epoch of the last write seen on each of its 16 pages. Chunks sit in 16 MiB leaves (never
// freed) found through a hash map of leaves and a direct-mapped memo: the former page map (unordered_map per written page) cost its
// inserts in every poll and a find per page queried.
struct Chunk { uint64_t polled = 0, max_epoch = 0; uint64_t page[kChunk / kPage] = {}; };  // submit of the last poll; newest write
bool due(const Chunk& ch) {  // the chunk must be polled before a query: once per submit (BB_WW_PER_CALL=1: always)
    static const bool per_call = std::getenv("BB_WW_PER_CALL") && std::getenv("BB_WW_PER_CALL")[0] == '1';
    return per_call || ch.polled != g_submit;
}
constexpr uint64_t kLeafChunks = 256;
struct Leaf { Chunk c[kLeafChunks]; };
std::unordered_map<uint64_t, std::unique_ptr<Leaf>> g_leaves;  // leaf number (address >> 24) -> chunks
Chunk& chunk(uint64_t c) {
    static uint64_t tag[256];
    static Leaf* ptr[256];
    const uint64_t l = c / kLeafChunks;
    const size_t i = l & 255;
    if (!ptr[i] || tag[i] != l) {
        auto& p = g_leaves[l];
        if (!p) p = std::make_unique<Leaf>();
        tag[i] = l, ptr[i] = p.get();
    }
    return ptr[i]->c[c % kLeafChunks];
}
std::unordered_set<uint64_t> g_split;    // blocks a single call cannot poll (several allocations, unwatched parts): polled per chunk
std::unordered_set<uint64_t> g_nomerge;  // blocks whose poll cannot extend into the next block (another allocation follows)
#endif
uint64_t g_polls = 0, g_poll_pages = 0, g_poll_ns = 0;

#ifdef _WIN32
// Polls chunk c, with its block when possible and the following blocks due up to chunk `last` in the same call (one allocation; a
// call costs ~0.4 us plus a few ns per page); marks the polled chunks and gives their written pages a new epoch (once per query).
bool poll(uint64_t c, uint64_t last, bool& found) {
    thread_local std::vector<void*> addrs;
    const uint64_t b = c * kChunk / kBlock;
    uint64_t lo = c * kChunk, len = kChunk;
    if (!g_split.count(b)) {
        lo = b * kBlock, len = kBlock;
        if (!g_nomerge.count(b))
            for (uint64_t n = b + 1; n <= last * kChunk / kBlock && !g_split.count(n) && due(chunk(n * (kBlock / kChunk))); ++n) len += kBlock;
    }
    ULONG_PTR count = 0;
    for (;;) {
        count = ULONG_PTR(len / kPage);
        if (addrs.size() < count) addrs.resize(count);
        ULONG gran = 0;
        ++g_polls, g_poll_pages += len / kPage;
        const auto t0 = std::chrono::steady_clock::now();
        const UINT r = GetWriteWatch(WRITE_WATCH_FLAG_RESET, reinterpret_cast<void*>(lo), SIZE_T(len), addrs.data(), &count, &gran);
        g_poll_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        if (r == 0) break;
        if (len == kChunk) return false;
        if (len > kBlock) g_nomerge.insert(b), len = kBlock;  // (the blocks lie in several allocations)
        else g_split.insert(b), lo = c * kChunk, len = kChunk;
    }
    for (uint64_t k = lo / kChunk; k < (lo + len) / kChunk; ++k) chunk(k).polled = g_submit;
    if (count && !found) ++g_epoch, found = true;
    for (ULONG_PTR i = 0; i < count; ++i) {
        const uint64_t a = reinterpret_cast<uint64_t>(addrs[i]);
        Chunk& ch = chunk(a / kChunk);
        ch.page[a % kChunk / kPage] = ch.max_epoch = g_epoch;
    }
    return true;
}
#endif

}  // namespace

void ww_new_submit() {
    ++g_submit;  // (BB_WW_PER_CALL: see due)
}

uint64_t ww_polls() { return g_polls; }
uint64_t ww_poll_pages() { return g_poll_pages; }
uint64_t ww_poll_ns() { return g_poll_ns; }

int ww_clean_since(uint64_t lo, uint64_t hi, uint64_t since, uint64_t& now) {
#ifdef _WIN32
    if (g_local_writes_any.load(std::memory_order_acquire)) {  // the port's own CPU writes: their chunks are polled again
        std::lock_guard<std::mutex> lk(g_local_writes_mutex);
        g_local_writes_any.store(false, std::memory_order_relaxed);
        for (const auto& [a, b] : g_local_writes)
            for (uint64_t c = a / kChunk; c <= (b - 1) / kChunk && b > a; ++c) chunk(c).polled = 0;
        g_local_writes.clear();
        if (g_local_writes_overflow) { ++g_submit; g_local_writes_overflow = false; }  // (every chunk unpolled)
    }
    bool found = false;
    const uint64_t plo = lo & ~(kPage - 1), phi = (hi + kPage - 1) & ~(kPage - 1);
    bool dirty = false;
    for (uint64_t c = lo / kChunk; c <= (hi - 1) / kChunk; ++c) {
        Chunk& ch = chunk(c);  // (leaves are never freed: the reference survives poll's lookups)
        if (due(ch) && !poll(c, (hi - 1) / kChunk, found)) {
            ch.polled = 0;
            return -1;
        }
        // pages are looked at only in chunks with a write newer than `since`
        if (!dirty && ch.max_epoch > since)
            for (uint64_t p = std::max(plo, c * kChunk) / kPage, pe = std::min(phi, (c + 1) * kChunk) / kPage; p < pe && !dirty; ++p)
                dirty = ch.page[p % (kChunk / kPage)] > since;
    }
    now = g_epoch;
    return dirty ? 0 : 1;
#else
    (void)lo; (void)hi; (void)since; (void)now;
    return -1;
#endif
}

int ww_repoll(uint64_t lo, uint64_t hi, uint64_t since) {
#ifdef _WIN32
    for (uint64_t c = lo / kChunk; c <= (hi - 1) / kChunk && hi > lo; ++c) chunk(c).polled = 0;
#endif
    uint64_t now = 0;
    return ww_clean_since(lo, hi, since, now);
}

void ww_mark(uint64_t lo, uint64_t hi) {
    if (hi <= lo) return;
    ++g_epoch;
#ifdef _WIN32  // the written pages get the new epoch, as if a poll had found them (elsewhere nothing is tracked)
    for (uint64_t c = lo / kChunk; c <= (hi - 1) / kChunk; ++c) {
        Chunk& ch = chunk(c);
        ch.max_epoch = g_epoch;
        for (uint64_t p = std::max(lo, c * kChunk) / kPage, pe = (std::min(hi, (c + 1) * kChunk) + kPage - 1) / kPage; p < pe; ++p)
            ch.page[p % (kChunk / kPage)] = g_epoch;
    }
#endif
}

void fine_sleep_us(uint32_t us) {
#ifdef _WIN32
    thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (timer) {
        LARGE_INTEGER due;
        due.QuadPart = -LONGLONG(us) * 10;  // relative, 100 ns units
        if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) { WaitForSingleObject(timer, INFINITE); return; }
    }
#endif
    std::this_thread::sleep_for(std::chrono::microseconds(us));
}

}  // namespace bb::gpu
