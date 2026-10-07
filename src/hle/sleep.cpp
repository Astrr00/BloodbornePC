// SPDX-License-Identifier: GPL-3.0-or-later
// Sub-millisecond sleeps. std::this_thread::sleep_for rounds up to the Windows timer tick (about 15.6 ms), which made the guest's
// polling loops (sceKernelUsleep(1000), audio output pacing) run several times slower than on the console.
#include <chrono>
#include <thread>

#include "hle/hle.h"

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace bb::hle {

void precise_sleep_us(uint64_t us) {
    if (!us) return;
#ifdef _WIN32
    // one high-resolution waitable timer per thread (about 0.5 ms resolution); falls back to sleep_for where unavailable
    thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (timer) {
        LARGE_INTEGER due;
        due.QuadPart = -int64_t(us) * 10;  // relative, 100 ns units
        if (SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0)) {
            WaitForSingleObject(timer, INFINITE);
            return;
        }
    }
#endif
    std::this_thread::sleep_for(std::chrono::microseconds(us));
}

}  // namespace bb::hle
