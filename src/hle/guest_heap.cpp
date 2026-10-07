// SPDX-License-Identifier: GPL-3.0-or-later
#include "hle/guest_heap.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

#include "hle/hle.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace bb::hle {
namespace {

// ponytail: power-of-two size classes without coalescing; the engine uses its own allocator over direct memory, libc
// malloc churn in the clinic is ~2k blocks/s at ~3 MB live (measured). Switch to a real allocator if malloc becomes hot.
struct Header {
    uint64_t tag;    // class index, kLarge, or kAligned
    uint64_t extra;  // large: mapping length; aligned: original block
};
constexpr uint64_t kLarge = 0xFE, kAligned = 0xFD;
constexpr size_t kMinClass = 5, kMaxClass = 20;  // 32 B .. 1 MiB blocks (header included)
constexpr size_t kChunk = 64ull << 20;

struct FreeNode { FreeNode* next; };

std::mutex g_m;
FreeNode* g_free[kMaxClass + 1];
uint8_t *g_chunk_cur = nullptr, *g_chunk_end = nullptr;
std::vector<std::pair<uintptr_t, uintptr_t>> g_chunks;  // small-block chunks [begin, end)
std::set<uintptr_t> g_large;                             // live large blocks (header addresses)

// A pointer the engine passes to free() may come from its own allocator (direct memory); freeing it here would put an
// arbitrary address on a free list. Only blocks this heap handed out are accepted.
bool owned(uintptr_t block) {  // block = header address
    std::lock_guard lk(g_m);
    if (g_large.count(block)) return true;
    for (const auto& c : g_chunks)
        if (block >= c.first && block < c.second) return true;
    return false;
}

void* small_block(size_t cls) {
    if (FreeNode* n = g_free[cls]) {
        g_free[cls] = n->next;
        return n;
    }
    const size_t bytes = size_t(1) << cls;
    if (g_chunk_cur + bytes > g_chunk_end) {
        g_chunk_cur = static_cast<uint8_t*>(host_map(nullptr, kChunk));
        if (!g_chunk_cur) return nullptr;
        g_chunk_end = g_chunk_cur + kChunk;
        g_chunks.push_back({reinterpret_cast<uintptr_t>(g_chunk_cur), reinterpret_cast<uintptr_t>(g_chunk_end)});
    }
    void* p = g_chunk_cur;
    g_chunk_cur += bytes;
    return p;
}

void* alloc_plain(size_t size) {
    const size_t need = size + sizeof(Header);
    size_t cls = kMinClass;
    while (cls <= kMaxClass && (size_t(1) << cls) < need) ++cls;
    Header* h;
    if (cls <= kMaxClass) {
        std::lock_guard lk(g_m);
        h = static_cast<Header*>(small_block(cls));
        if (!h) return nullptr;
        h->tag = cls;
        h->extra = 0;
    } else {
        const size_t len = (need + 0xFFFF) & ~size_t(0xFFFF);
        h = static_cast<Header*>(host_map(nullptr, len));
        if (!h) return nullptr;
        h->tag = kLarge;
        h->extra = len;
        std::lock_guard lk(g_m);
        g_large.insert(reinterpret_cast<uintptr_t>(h));
    }
    return h + 1;
}

void free_plain(Header* h) {
    if (h->tag == kLarge) {
        {
            std::lock_guard lk(g_m);
            g_large.erase(reinterpret_cast<uintptr_t>(h));
        }
#ifdef _WIN32
        VirtualFree(h, 0, MEM_RELEASE);
#else
        munmap(h, h->extra);
#endif
        return;
    }
    std::lock_guard lk(g_m);
    const uint64_t cls = h->tag;  // (read first: `next` overwrites the tag)
    auto* n = reinterpret_cast<FreeNode*>(h);
    n->next = g_free[cls];
    g_free[cls] = n;
}

} // namespace

void* guest_alloc(size_t size, size_t align) {
    if (align <= 16) return alloc_plain(size);
    void* raw = alloc_plain(size + align + sizeof(Header));
    if (!raw) return nullptr;
    const uintptr_t user = (reinterpret_cast<uintptr_t>(raw) + sizeof(Header) + align - 1) & ~(uintptr_t(align) - 1);
    auto* h = reinterpret_cast<Header*>(user) - 1;
    h->tag = kAligned;
    h->extra = reinterpret_cast<uintptr_t>(raw);
    return reinterpret_cast<void*>(user);
}

void guest_free(void* p) {
    if (!p) return;
    Header* h = static_cast<Header*>(p) - 1;
    if (!owned(reinterpret_cast<uintptr_t>(h))) {  // aligned blocks keep their header just before the user pointer, which lies inside an owned block
        const uintptr_t inside = reinterpret_cast<uintptr_t>(p);
        bool ok = false;
        {
            std::lock_guard lk(g_m);
            for (const auto& c : g_chunks) ok = ok || (inside >= c.first && inside < c.second);
        }
        if (!ok) {
            static int logged = 0;
            if (logged++ < 16) std::fprintf(stderr, "heap: free(%p) ignored: not a block of the HLE heap\n", p);
            return;
        }
    }
    if (h->tag == kAligned) h = reinterpret_cast<Header*>(h->extra) - 1;
    free_plain(h);
}

size_t guest_usable_size(void* p) {
    Header* h = static_cast<Header*>(p) - 1;
    if (h->tag == kAligned) {
        auto* raw = reinterpret_cast<uint8_t*>(h->extra);
        Header* rh = reinterpret_cast<Header*>(raw) - 1;
        const size_t whole = rh->tag == kLarge ? rh->extra - sizeof(Header) : (size_t(1) << rh->tag) - sizeof(Header);
        return whole - size_t(static_cast<uint8_t*>(p) - raw);
    }
    return h->tag == kLarge ? h->extra - sizeof(Header) : (size_t(1) << h->tag) - sizeof(Header);
}

int* guest_errno() {
    static int* slot = [] {
        int* p = static_cast<int*>(guest_alloc(16));
        *p = 0;
        return p;
    }();
    return slot;
}

} // namespace bb::hle
