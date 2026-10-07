// SPDX-License-Identifier: GPL-3.0-or-later
// Which guest pages were written since a cached copy was taken: Windows write watch (guest memory is allocated with MEM_WRITE_WATCH,
// see hle::host_map). Every written page found by a poll gets the next epoch in a page table, so overlapping cache entries that poll the
// same pages later still see the write. GPU stores into imported guest memory bypass the watch and are recorded with ww_mark.
// Not called concurrently (backend lock).
#pragma once
#include <cstdint>

namespace bb::gpu {

// 1: no page of [lo, hi) was written after epoch `since`; 0: written; -1: tracking unavailable here (hash instead).
// `now` receives the current epoch (store it with a copy taken after this call).
int ww_clean_since(uint64_t lo, uint64_t hi, uint64_t since, uint64_t& now);
// Debug (BB_WW_VERIFY): the same query after polling [lo, hi) again now. Written now = a write after this submit's poll (the
// documented once-per-submit window); still clean = a write the watch never saw.
int ww_repoll(uint64_t lo, uint64_t hi, uint64_t since);
void ww_mark(uint64_t lo, uint64_t hi);  // [lo, hi) changed outside the CPU (GPU stores): its pages count as written (page granularity)
// A new guest submit begins: each 64 KiB chunk is polled at most once per submit (with its 1 MiB block where one call can; again after
// the port's own CPU writes to it, gpu_hooks.h g_local_writes). ponytail: guest-thread writes during one submit show up with the next
// one; on the console the GPU reads even later, so only racing game code could notice. Under load a submit takes up to ~100 ms; the
// game rewrites a few dynamic vertex buffers in that window (BB_WW_VERIFY "late write", ~1 per 10 M checks). BB_WW_PER_CALL=1: poll
// on every call.
void ww_new_submit();
uint64_t ww_polls();       // GetWriteWatch calls so far (statistics)
uint64_t ww_poll_pages();  // pages those calls covered (statistics)
uint64_t ww_poll_ns();     // time in those calls (statistics)

// Sleeps about `us` microseconds with sub-millisecond resolution (Windows: high-resolution waitable timer; else sleep_for).
void fine_sleep_us(uint32_t us);

}  // namespace bb::gpu