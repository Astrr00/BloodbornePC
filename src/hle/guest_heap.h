// SPDX-License-Identifier: GPL-3.0-or-later
// Allocator for everything HLE hands to the guest (libc malloc, thread/mutex/semaphore objects).
// The guest assumes user pointers fit in 40 bits (e.g. the engine's mutex owner field masks scePthreadSelf()
// to 0xFFFFFFFFFF but compares the unmasked value), so these blocks come from host_map's low address window
// instead of the host heap.
#pragma once
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

namespace bb::hle {

void* guest_alloc(size_t size, size_t align = 16);  // nullptr on failure; memory is NOT zeroed
void guest_free(void* p);
size_t guest_usable_size(void* p);
int* guest_errno();  // the (process-wide) errno cell the guest reads through __error / sceNetErrnoLoc

// Placement-new a host object inside guest memory (its address is what the guest sees as the handle).
template <class T, class... A> T* guest_new(A&&... a) {
    void* p = guest_alloc(sizeof(T), alignof(T) < 16 ? 16 : alignof(T));
    return p ? new (p) T(std::forward<A>(a)...) : nullptr;
}
template <class T> void guest_delete(T* p) {
    if (!p) return;
    p->~T();
    guest_free(p);
}

} // namespace bb::hle
