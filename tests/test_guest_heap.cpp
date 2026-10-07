// SPDX-License-Identifier: GPL-3.0-or-later
// guest_heap reuse/leak regression test; guest_heap.cpp is compiled in with a counting host_map stub.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "hle/guest_heap.h"

#define NOMINMAX
#include <windows.h>

namespace bb::hle {
int g_maps = 0;
void* host_map(void*, uint64_t len) {
    ++g_maps;
    return VirtualAlloc(nullptr, len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}
} // namespace bb::hle

using namespace bb::hle;

#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            std::exit(1);                                                \
        }                                                                \
    } while (0)

int main() {
    // (1) free + malloc of the same class returns the freed block.
    for (size_t n : {1, 16, 40, 100, 500, 4000, 100000}) {
        void* a = guest_alloc(n);
        CHECK(a && guest_usable_size(a) >= n);
        std::memset(a, 0xAB, n);
        guest_free(a);
        CHECK(guest_alloc(n) == a);
        guest_free(a);
    }

    // (2) churn of mixed small sizes stays within the first chunk (64 MiB; no reuse would need ~GBs).
    const int maps = g_maps;
    void* live[64] = {};
    unsigned rng = 1;
    for (int i = 0; i < 200000; ++i) {
        rng = rng * 1103515245u + 12345u;
        const unsigned slot = (rng >> 16) % 64;
        guest_free(live[slot]);
        live[slot] = guest_alloc(1 + (rng >> 8) % 2000);
        CHECK(live[slot]);
    }
    for (void* p : live) guest_free(p);
    CHECK(g_maps - maps <= 1);

    // (3) aligned and large round-trip.
    for (size_t al : {32, 64, 4096, 65536}) {
        void* p = guest_alloc(300, al);
        CHECK(p && reinterpret_cast<uintptr_t>(p) % al == 0 && guest_usable_size(p) >= 300);
        std::memset(p, 1, 300);
        guest_free(p);
    }
    for (size_t n : {size_t(2) << 20, size_t(5) << 20}) {
        void* p = guest_alloc(n);
        CHECK(p && guest_usable_size(p) >= n);
        std::memset(p, 2, n);
        guest_free(p);
        void* q = guest_alloc(n, 4096);
        CHECK(q && reinterpret_cast<uintptr_t>(q) % 4096 == 0 && guest_usable_size(q) >= n);
        guest_free(q);
    }
    std::puts("guest_heap OK");
    return 0;
}
