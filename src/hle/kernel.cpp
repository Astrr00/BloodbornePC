// SPDX-License-Identifier: GPL-3.0-or-later
// libkernel slice (non-threading): direct memory mapping, time, and misc system stubs.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "core/sfo.h"

#include "hle/guest_heap.h"
#include "hle/hle.h"
#include "hle/vfs.h"

namespace bb::rt { std::vector<ThreadDebug> debug_threads_snapshot(); }  // guest.cpp (not in guest.h)

namespace bb::hle {
namespace {

void zero(Context& c) { ret(c, 0); }

// 5 GiB direct memory like retail hardware; offsets are handed out by a bump allocator, never reused.
constexpr uint64_t kDirectSize = 5ull << 30;
uint64_t g_direct_next = 0;

void direct_size(Context& c) { ret(c, kDirectSize); }
void allocate_direct(Context& c) {  // (searchStart, searchEnd, len, align, type, uint64_t* phys)
    const uint64_t len = arg(c, 2), align = arg(c, 3) ? arg(c, 3) : 0x4000;
    uint64_t off = (g_direct_next + align - 1) / align * align;
    if (off + len > kDirectSize) return ret(c, kErrNoMem);
    g_direct_next = off + len;
    rt::st<uint64_t>(c, arg(c, 5), off);
    if (std::getenv("BB_MAP_LOG")) std::fprintf(stderr, "direct: allocate off=0x%llx len=0x%llx align=0x%llx type=%d\n", (unsigned long long)off, (unsigned long long)len, (unsigned long long)align, int(arg(c, 4)));
    ret(c, 0);
}
void map_direct(Context& c) {  // (void** addr, len, prot, flags, directStart, align); flag 0x10 = FIXED
    const uint64_t at = rt::ld<uint64_t>(c, arg(c, 0)), len = arg(c, 1), flags = arg(c, 3);
    void* p = host_map((flags & 0x10) ? ptr(c, at) : nullptr, len);
    if (!p) return ret(c, kErrNoMem);
    rt::st<uint64_t>(c, arg(c, 0), reinterpret_cast<uintptr_t>(p) - c.base);
    note_mapping(reinterpret_cast<uintptr_t>(p) - c.base, len, int(arg(c, 2)));
    if (std::getenv("BB_MAP_LOG")) std::fprintf(stderr, "direct: map addr=0x%llx len=0x%llx directStart=0x%llx prot=%d flags=0x%llx\n", (unsigned long long)(reinterpret_cast<uintptr_t>(p) - c.base), (unsigned long long)len, (unsigned long long)arg(c, 4), int(arg(c, 2)), (unsigned long long)flags);
    ret(c, 0);
}

uint64_t now_us() {
    return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}
// Debug BB_UNCAP=<seconds>: real throughput above the engine's 30 fps cap. From <seconds> after start the limiter f_2035ac0 (guest
// 0x2435ac0, ROADMAP 45) never waits: its first gettimeofday of a frame (return 0x2435ccc; object in r12, callee-saved) returns at least
// frame start +0x20 (µs) + target +0x18 (float s) + 1 µs, so the wait is over before it starts and the limiter's state (+0x264, ring,
// late flag) matches an on-time frame. The jump is kept as an offset for every later gettimeofday of that thread (monotonic, consistent
// with the engine's other frame stamps); other threads see real time. The logic steps a fixed 1/30 s per frame (ROADMAP 52/62), so the
// game runs in fast-forward. Above 60 flips/s also set BB_FLIP_PACE=0. ponytail: 1.00 addresses only; 1.09 needs its own call site.
const auto g_start = std::chrono::steady_clock::now();
const double g_uncap = std::getenv("BB_UNCAP") ? std::atof(std::getenv("BB_UNCAP")) : -1;
uint64_t g_lim_off = 0;              // only touched by the limiter's thread
thread_local bool t_limiter = false;
void gettimeofday_(Context& c) {  // struct timeval {int64 sec; int64 usec}
    uint64_t t = now_us();
    if (g_uncap >= 0) {
        const bool frame = rt::ld<uint64_t>(c, c.r[4]) == 0x2435ccc;
        t_limiter |= frame;
        if (t_limiter) t += g_lim_off;
        if (frame && std::chrono::steady_clock::now() - g_start >= std::chrono::duration<double>(g_uncap)) {
            if (static bool said = false; !said) {  // once: thread ids for per-thread CPU accounting (build/uncap_stats.py)
                said = true;
                std::fprintf(stderr, "uncap: limiter unbounded from now; threads:");
                for (const auto& th : rt::debug_threads_snapshot()) std::fprintf(stderr, " [%llu %s]", (unsigned long long)th.os_thread_id, th.name);
                std::fprintf(stderr, "\n");
            }
            const uint64_t due = rt::ld<uint64_t>(c, c.r[12] + 0x20) + uint64_t(double(rt::ld<float>(c, c.r[12] + 0x18)) * 1e6) + 1;
            if (t < due) g_lim_off += due - t, t = due;
        }
    }
    if (arg(c, 0)) {
        rt::st<uint64_t>(c, arg(c, 0), t / 1000000);
        rt::st<uint64_t>(c, arg(c, 0) + 8, t % 1000000);
    }
    ret(c, 0);
}
void usleep_(Context& c) { precise_sleep_us(arg32(c, 0)); ret(c, 0); }  // useconds_t is 32 bit
uint64_t steady_us() {
    static const auto t0 = std::chrono::steady_clock::now();
    return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
}
void process_time(Context& c) { ret(c, steady_us()); }
void process_time_counter(Context& c) { ret(c, steady_us()); }
void process_time_freq(Context& c) { ret(c, 1000000); }

// sceAppContentAppParamGetInt(paramId, int* out): 0 = SKU flag (3 = full game, 1 = trial), 1..4 = USER_DEFINED_PARAM_1..4 of
// /app0/sce_sys/param.sfo (PSF integer entries).
void appcontent_param_int(Context& c) {
    int32_t val = 0;
    const unsigned id = unsigned(argi(c, 0));
    if (id == 0) val = 3;
    else if (auto p = vfs_resolve("/app0/sce_sys/param.sfo"); p && id <= 4)
        if (auto sfo = load_sfo(*p))
            if (auto it = sfo->entries.find("USER_DEFINED_PARAM_" + std::to_string(id)); it != sfo->entries.end())
                if (auto* v = std::get_if<uint32_t>(&it->second)) val = int32_t(*v);
    if (std::getenv("BB_SAVE_LOG")) std::fprintf(stderr, "appcontent: AppParamGetInt(%u) = %d\n", id, val);
    rt::st<int32_t>(c, arg(c, 1), val);
    ret(c, 0);
}

// FreeBSD clock ids: REALTIME 0, MONOTONIC 4 (also UPTIME 5 and friends map to the steady clock).
void clock_gettime_(Context& c) {  // (clockid, struct timespec*)
    uint64_t ns;
    if (arg(c, 0) == 0) {
        ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    } else {
        ns = steady_us() * 1000;
    }
    rt::st<uint64_t>(c, arg(c, 1), ns / 1000000000);
    rt::st<uint64_t>(c, arg(c, 1) + 8, ns % 1000000000);
    ret(c, 0);
}
// (addr, void** start, void** end, int* prot): everything is one RW block as far as the guest can tell.
void query_protection(Context& c) {
    Mapping m{arg(c, 0) & ~uint64_t(0xFFF), (arg(c, 0) & ~uint64_t(0xFFF)) + 0x1000, 3};  // default: RW (stacks, image)
    find_mapping(arg(c, 0), m);
    if (arg(c, 1)) rt::st<uint64_t>(c, arg(c, 1), m.start);
    if (arg(c, 2)) rt::st<uint64_t>(c, arg(c, 2), m.end);
    if (arg(c, 3)) rt::st<int32_t>(c, arg(c, 3), m.prot);
    ret(c, 0);
}

} // namespace

void register_kernel() {
    reg("sceKernelGetDirectMemorySize", direct_size);
    reg("sceKernelAllocateDirectMemory", allocate_direct);
    reg("sceKernelMapDirectMemory", map_direct);
    reg("sceKernelReleaseDirectMemory", zero);  // ponytail: direct memory offsets are never reused
    reg("gettimeofday", gettimeofday_);
    reg("__error", [](Context& c) { ret(c, reinterpret_cast<uintptr_t>(guest_errno()) - c.base); });
    reg("sceKernelReadTsc", [](Context& c) { ret(c, steady_us() * 1600); });  // 1.6 GHz TSC like the console
    reg("sceKernelClockGettime", clock_gettime_);
    reg("sceKernelQueryMemoryProtection", query_protection);
    // (addr, len, prot): guest memory is host RW already and the GPU reads any noted mapping, so nothing changes. The AvPlayer
    // allocator (f_2912da0) asks for GPU access on blocks QueryMemoryProtection reports CPU-only. ponytail: the new protection is
    // not recorded (later queries repeat the old one; the allocator then simply asks again).
    reg("sceKernelMprotect", zero);
    reg("sceKernelUsleep", usleep_);
    reg("usleep", usleep_);
    reg("sleep", [](Context& c) { precise_sleep_us(uint64_t(arg32(c, 0)) * 1000000); ret(c, 0); });  // returns the unslept seconds: 0
    reg("getpagesize", [](Context& c) { ret(c, 0x4000); });  // 16 KiB pages on Orbis
    reg("sceKernelGetProcessTime", process_time);
    reg("sceKernelGetProcessTimeCounter", process_time_counter);
    reg("sceKernelGetProcessTimeCounterFrequency", process_time_freq);
    reg("sceSysmoduleLoadModule", zero);
    reg("sceAppContentInitialize", zero);
    reg("sceAppContentAppParamGetInt", appcontent_param_int);
}

} // namespace bb::hle
