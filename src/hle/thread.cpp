// SPDX-License-Identifier: GPL-3.0-or-later
// libkernel/libScePosix threading: pthreads, mutexes, conds, rwlocks, semaphores. Guest threads are host threads
// with their own Context, TLS block and guest stack. Handles written to guest memory are host object pointers
// (valid guest addresses under identity mapping); zero handles (static initialisers) are created lazily.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <system_error>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "hle/guest_heap.h"
#include "hle/hle.h"
#include "runtime/process.h"

namespace bb::rt { void unregister_debug_thread(); }

namespace bb::hle {
namespace {

using Clock = std::chrono::steady_clock;

std::mutex g_create_mutex;  // serialises lazy creation of statically initialised objects

template <class T>
T* get_or_create(Context& c, uint64_t handle_addr) {
    std::lock_guard lk(g_create_mutex);
    uint64_t h = rt::ld<uint64_t>(c, handle_addr);
    if (h <= 1) {  // FreeBSD libthr static initialisers: NULL (PTHREAD_MUTEX_INITIALIZER) and 1 (adaptive)
        h = reinterpret_cast<uintptr_t>(guest_new<T>());
        rt::st<uint64_t>(c, handle_addr, h);
    } else if (h < 0x10000) {  // 2 = destroyed, 3 = replaced: a use after destroy; libthr answers EINVAL instead of crashing
        static int logged = 0;
        if (logged++ < 8) {
            std::fprintf(stderr, "thread: sync object at 0x%llx holds %llu (destroyed/invalid); words:", (unsigned long long)handle_addr, (unsigned long long)h);
            for (int i = -2; i < 14; ++i) std::fprintf(stderr, " %llx", (unsigned long long)rt::ld<uint64_t>(c, handle_addr + 8 * i));
            std::fprintf(stderr, "\n");
        }
        return nullptr;
    }
    return reinterpret_cast<T*>(h);
}
template <class T>
T* get(Context&, uint64_t handle) { return handle < 0x10000 ? nullptr : reinterpret_cast<T*>(handle); }

// ponytail: every mutex is recursive (superset of normal/errorcheck); real type semantics only if a game relies on EDEADLK.
// Recursive mutex with FreeBSD libthr semantics for condition waits: pthread_cond_wait releases the mutex *completely*
// (all recursion levels) and restores the depth afterwards. std::recursive_mutex would release only one level, leaving
// the mutex held and every other thread that needs it blocked while the waiter sleeps.
struct Mutex {
    std::mutex m;
    std::atomic<const void*> owner{nullptr};
    uint32_t count = 0;  // only touched by the owner
    static const void* me() { static thread_local char tag; return &tag; }
    void lock() {
        if (owner.load(std::memory_order_relaxed) == me()) { ++count; return; }
        try {
            m.lock();
        } catch (const std::system_error& e) {
            std::fprintf(stderr, "mutex %p lock failed: %s (owner=%p me=%p count=%u)\n", (void*)this, e.what(), owner.load(), me(), count);
            throw;
        }
        owner.store(me(), std::memory_order_relaxed);
        count = 1;
    }
    bool try_lock() {
        if (owner.load(std::memory_order_relaxed) == me()) { ++count; return true; }
        if (!m.try_lock()) return false;
        owner.store(me(), std::memory_order_relaxed);
        count = 1;
        return true;
    }
    void unlock() {
        if (owner.load(std::memory_order_relaxed) != me()) return;  // not the owner (EPERM on the real system)
        if (--count == 0) {
            owner.store(nullptr, std::memory_order_relaxed);
            m.unlock();
        }
    }
    uint32_t release_all() {  // 0 if the caller did not hold the mutex
        if (owner.load(std::memory_order_relaxed) != me()) return 0;
        const uint32_t d = count;
        count = 0;
        owner.store(nullptr, std::memory_order_relaxed);
        m.unlock();
        return d;
    }
    void reacquire(uint32_t depth) {
        if (!depth) return;
        m.lock();
        owner.store(me(), std::memory_order_relaxed);
        count = depth;
    }
};
// BasicLockable view for std::condition_variable_any: unlock() drops every level, lock() restores them.
struct CondLock {
    Mutex* mu;
    uint32_t depth = 0;
    void unlock() { depth = mu->release_all(); }
    void lock() { mu->reacquire(depth); }
};
struct Cond { std::condition_variable_any cv; };
struct RwLock {
    std::mutex m;
    std::condition_variable cv;
    int readers = 0;
    bool writer = false;
};
struct Sema {
    std::mutex m;
    std::condition_variable cv;
    int64_t count = 0, max = 0x7fffffff;
    std::string name;  // for BB_TRACE_HLE
};
struct Attr { uint64_t stack_size = 256 << 10; int detach = 0; };

void zero(Context& c) { ret(c, 0); }

// ---- mutex / cond / rwlock ---------------------------------------------------------------------------------------
void attr_init(Context& c) { rt::st<uint64_t>(c, arg(c, 0), 1); ret(c, 0); }  // opaque, contents never read

void mutex_init(Context& c) { rt::st<uint64_t>(c, arg(c, 0), 0); get_or_create<Mutex>(c, arg(c, 0)); ret(c, 0); }
// Debug BB_REPLAY=<s> (DoD 4, experiment E1): from <s> seconds after start the GXRenderThread plays its finished GX list once more
// and flips again before it reports "frame done". Hook: the scePthreadMutexLock at the start of f_207fb70(GXRT+0x68, 1), called by
// the render loop f_21b7a30 right after the flip job; "done" is not set yet, so the main thread still waits in f_219a0f0 and the
// list stays valid. Every such point logs "gxrt t=.. gx600=.. gx608=.. replay=0|1" (render thread, so in order with "flip" lines).
// The first replayed list is dumped before (the lock in f_207fb70(GXRT+0x48, 0) right before f_219ac10), after its original play
// and after the replay. Result (clinic): the replay emits 48 of ~1520 draws: 45 are the post/composite pass 0x25d6e20 working on
// the scene targets the original left behind; the scene's draw packets (type 0) and the HUD are consumed by their first play.
// Attribution / fallback: BB_REPLAY_DROP=<hex fn> mutes the list callbacks fn (from <s> on) in the original play and the replay,
// BB_REPLAY_SKIP=<hex fn> only in the replay (2364740 resets the HUD command buffer; skipping it keeps the HUD). BB_REPLAY_KEEP=1:
// f_216bbc0 / f_216bd70 free a draw packet's chunk list through its allocator [D] vt+0x70 and clear D+0x8..+0xb0; before the
// original play each packet D gets a no-op allocator and a snapshot of D[0, 0xb0), restored right before the replay, which then
// frees for real. ponytail: the replay then emits ~1440-1490 draws, but a GX recording worker crashes in the chunk allocator
// (f_26aa2c0) within 1-2 frames; a usable version needs the allocator's ownership rules.
const double g_replay_from = std::getenv("BB_REPLAY") ? std::atof(std::getenv("BB_REPLAY")) : -1.0;
const uint64_t g_replay_drop = std::getenv("BB_REPLAY_DROP") ? std::strtoull(std::getenv("BB_REPLAY_DROP"), nullptr, 16) : 0;
const uint64_t g_replay_skip = std::getenv("BB_REPLAY_SKIP") ? std::strtoull(std::getenv("BB_REPLAY_SKIP"), nullptr, 16) : 0;
const bool g_replay_keep = std::getenv("BB_REPLAY_KEEP") != nullptr;
const Clock::time_point g_replay_t0 = Clock::now();

// Allocator whose every virtual returns its this (f_26d5c50 = `mov rax, rdi; ret`); host memory is guest memory (identity mapping).
uint64_t g_noop_vt[32];
uint64_t g_noop_alloc = [] { std::fill(std::begin(g_noop_vt), std::end(g_noop_vt), 0x2ad5c50); return reinterpret_cast<uint64_t>(g_noop_vt); }();

// BB_REPLAY_KEEP: snapshot D[0, 0xb0) of every draw packet D = [pkt + 4 + len] and give it the no-op allocator.
std::vector<std::pair<uint64_t, std::array<uint64_t, 22>>> keep_packets(Context& c, uint64_t list) {
    std::vector<std::pair<uint64_t, std::array<uint64_t, 22>>> kept;
    const uint64_t ents = rt::ld<uint64_t>(c, list + 0x30);
    for (uint32_t i = 0, n = rt::ld<uint32_t>(c, list + 0x38); i < n; ++i)
        if (const uint64_t p = rt::ld<uint64_t>(c, ents + 16 * i); p && rt::ld<uint8_t>(c, p) == 0)
            if (const uint64_t d = rt::ld<uint64_t>(c, p + 4 + rt::ld<uint8_t>(c, p + 1))) {
                auto& [at, w] = kept.emplace_back(d, std::array<uint64_t, 22>{});
                for (size_t k = 0; k < w.size(); ++k) w[k] = rt::ld<uint64_t>(c, d + 8 * k);
                rt::st<uint64_t>(c, d, reinterpret_cast<uint64_t>(&g_noop_alloc));
            }
    return kept;
}

// Turns the list callbacks fn into no-ops (fn := 0x2364750 with rdx = 0, which returns at once). Returns (slot, rdx) to undo it.
std::vector<std::pair<uint64_t, uint64_t>> mute(Context& c, uint64_t list, uint64_t fn) {
    std::vector<std::pair<uint64_t, uint64_t>> undo;
    const uint64_t ents = rt::ld<uint64_t>(c, list + 0x30);
    for (uint32_t i = 0, n = rt::ld<uint32_t>(c, list + 0x38); i < n; ++i)
        if (const uint64_t p = rt::ld<uint64_t>(c, ents + 16 * i); p && rt::ld<uint8_t>(c, p) == 1)
            if (const uint64_t q = p + 4 + rt::ld<uint8_t>(c, p + 1); rt::ld<uint64_t>(c, q) == fn) {
                undo.emplace_back(q, rt::ld<uint64_t>(c, q + 16));
                rt::st<uint64_t>(c, q, 0x2364750);
                rt::st<uint64_t>(c, q + 16, 0);
            }
    return undo;
}

// PM4 draws (DRAW_INDEX_2 / _AUTO / _OFFSET_2) and INDIRECT_BUFFERs in a mapped command buffer, following IBs two levels deep.
void pm4_count(Context& c, uint64_t a, uint64_t dw, int depth, uint64_t& draws, uint64_t& ibs) {
    if (Mapping m; !dw || !find_mapping(a, m) || m.end < a + 4 * dw) return;
    for (uint64_t i = 0; i < dw;) {
        const uint32_t h = rt::ld<uint32_t>(c, a + 4 * i), len = ((h >> 16) & 0x3FFF) + 2;
        if (h >> 30 != 3) { i += (h >> 30 == 0 && h) ? len : 1; continue; }  // type 0 skips its body, type 2 / zero / type 1: one dword
        const uint32_t op = (h >> 8) & 0xFF;
        if (op == 0x27 || op == 0x2D || op == 0x35) ++draws;
        if (op == 0x3F && i + 3 < dw) {
            ++ibs;
            if (depth < 2)
                pm4_count(c, rt::ld<uint32_t>(c, a + 4 * i + 4) | (uint64_t(rt::ld<uint32_t>(c, a + 4 * i + 8) & 0xFFFF) << 32), rt::ld<uint32_t>(c, a + 4 * i + 12) & 0xFFFFF, depth + 1, draws, ibs);
        }
        i += len;
    }
}

// A Gnmx-style command buffer d as f_109bf60 submits it: segments (count +0x2b0, table +0x130, 24 B each: dword offset from [+0],
// dword size) plus the tail [+0x100, +0x10).
void cb_dump(Context& c, const char* tag, const char* what, uint64_t d) {
    const uint64_t tail = rt::ld<uint64_t>(c, d + 0x100), end = rt::ld<uint64_t>(c, d + 0x10);
    const uint32_t segs = rt::ld<uint32_t>(c, d + 0x2b0);
    uint64_t draws = 0, ibs = 0;
    for (uint32_t s = 0; s < segs && s < 64; ++s)
        pm4_count(c, rt::ld<uint64_t>(c, d) + 4ull * rt::ld<uint32_t>(c, d + 24 * s + 0x130), rt::ld<uint32_t>(c, d + 24 * s + 0x134), 0, draws, ibs);
    pm4_count(c, tail, (end - tail) / 4, 0, draws, ibs);
    std::fprintf(stderr, "replay: %s %s cb 0x%llx segs=%u tail=0x%llx draws=%llu ibs=%llu\n", tag, what, (unsigned long long)d, segs, (unsigned long long)(end - tail),
                 (unsigned long long)draws, (unsigned long long)ibs);
}

// Packet types of a GX list and per callback (type 1: fn = [pkt + 4 + len], rsi/rdx = [+8]/[+0x10]) the state known consume-once
// callbacks read: 0x2364750 its pending count [rdx+0x88] (x), 0x2364760 its (HUD) command buffer rsi+0xc08.
void list_dump(Context& c, const char* tag, uint64_t list) {
    const uint64_t ents = rt::ld<uint64_t>(c, list + 0x30);
    const uint32_t n = rt::ld<uint32_t>(c, list + 0x38);
    uint32_t types[256] = {};
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t p = rt::ld<uint64_t>(c, ents + 16 * i);
        const uint8_t ty = p ? rt::ld<uint8_t>(c, p) : 0xff;
        ++types[ty];
        if (ty != 1) continue;
        const uint64_t q = p + 4 + rt::ld<uint8_t>(c, p + 1), fn = rt::ld<uint64_t>(c, q), a1 = rt::ld<uint64_t>(c, q + 8), a2 = rt::ld<uint64_t>(c, q + 16);
        const uint64_t x = fn == 0x2364750 && a2 ? rt::ld<uint32_t>(c, a2 + 0x88) : 0;
        if (fn == 0x2364760 && a1) cb_dump(c, tag, "hud", a1 + 0xc08);
        std::fprintf(stderr, "replay: %s #%u fn=0x%llx rsi=0x%llx rdx=0x%llx x=0x%llx\n", tag, i, (unsigned long long)fn, (unsigned long long)a1, (unsigned long long)a2, (unsigned long long)x);
    }
    std::fprintf(stderr, "replay: %s list 0x%llx n=%u type0=%u type1=%u other=%u\n", tag, (unsigned long long)list, n, types[0], types[1], n - types[0] - types[1]);
}

void gx_replay(Context& c) {
    // return addresses: the lock call in f_207fb70, and f_207fb70's caller (after its 5 pushes) in the render loop f_21b7a30:
    // 0x25b7b8d after the flip job (replay point), 0x25b7ae2 before f_219ac10 plays the list (snapshot point)
    if (rt::ld<uint64_t>(c, c.r[4]) != 0x247fbac) return;
    const uint64_t caller = rt::ld<uint64_t>(c, c.r[4] + 48);
    if (caller != 0x25b7b8d && caller != 0x25b7ae2) return;
    const uint64_t gx = rt::ld<uint64_t>(c, 0x5940698), gxrt = rt::ld<uint64_t>(c, gx + 0x348);
    const double t = std::chrono::duration<double>(Clock::now() - g_replay_t0).count();
    static int dumps = 0;  // render thread only
    static std::vector<std::pair<uint64_t, std::array<uint64_t, 22>>> kept;  // BB_REPLAY_KEEP snapshot of the list being played
    if (caller == 0x25b7ae2) {
        if (t < g_replay_from || c.r[7] != gxrt + 0x50) return;  // rdi = event GXRT+0x48, +8 (its mutex)
        const uint64_t list = rt::ld<uint64_t>(c, gxrt + 0x18);
        if (dumps == 0 && ++dumps) list_dump(c, "pre", list);
        mute(c, list, g_replay_drop);  // fn 0 (unset) matches nothing
        if (g_replay_keep) kept = keep_packets(c, list);
        return;
    }
    const uint64_t cfg = rt::ld<uint64_t>(c, gxrt + 0x88), job = rt::ld<uint64_t>(c, gx + 0xb0);
    const bool on = t >= g_replay_from && c.r[7] == gxrt + 0x70 && rt::ld<uint8_t>(c, cfg + 0x72) && job;
    const uint64_t done = rt::ld<uint64_t>(c, gx + 0x608);
    std::fprintf(stderr, "gxrt t=%.3f gx600=%llu gx608=%llu replay=%d\n", t, (unsigned long long)rt::ld<uint64_t>(c, gx + 0x600), (unsigned long long)done, int(on));
    // put the draw packets back (reverse: a packet listed twice ends with its first, real snapshot); ponytail: a frame without
    // replay (no flip job) restores too and leaks that frame's chunks
    for (auto it = kept.rbegin(); it != kept.rend(); ++it)
        for (size_t k = 0; k < it->second.size(); ++k) rt::st<uint64_t>(c, it->first + 8 * k, it->second[k]);
    kept.clear();
    if (!on) return;
    // the list just played (GX+0x380 flips only after the main thread's wait); f_219ac10 already appended its end callback and sorted it
    const uint64_t list = rt::ld<uint64_t>(c, gx + 0x360 + 8 * (rt::ld<uint32_t>(c, gx + 0x380) ^ 1));
    const uint64_t ents = rt::ld<uint64_t>(c, list + 0x30);
    const uint32_t n = rt::ld<uint32_t>(c, list + 0x38);
    if (dumps == 1 && ++dumps) list_dump(c, "post", list);
    const Context saved = c;  // the guest code after the lock expects its registers; call_guest restores only rsp
    const uint64_t ctx = rt::ld<uint64_t>(c, gx + 0x10);
    if (const uint64_t prof = rt::ld<uint64_t>(c, gx + 0x640)) call_guest(c, 0x25c28d0, prof, ctx);  // as f_219ac10 does
    call_guest(c, 0x2599d30, gx, ctx, 0);       // prepare the context (reset, open the section the end callback closes)
    const auto undo = mute(c, list, g_replay_skip);
    call_guest(c, 0x25b5e20, 0, ctx, ents, n);  // executor
    for (const auto& [q, rdx] : undo) rt::st<uint64_t>(c, q, g_replay_skip), rt::st<uint64_t>(c, q + 16, rdx);
    if (dumps == 2 && ++dumps) list_dump(c, "replayed", list);
    if (const uint64_t now = rt::ld<uint64_t>(c, gx + 0x608); now != done)
        std::fprintf(stderr, "replay: GX+0x608 %llu -> %llu, restored\n", (unsigned long long)done, (unsigned long long)now);
    rt::st<uint64_t>(c, gx + 0x608, done);
    call_guest(c, rt::ld<uint64_t>(c, rt::ld<uint64_t>(c, job)), job, rt::ld<uint32_t>(c, cfg + 0x40));  // flip job vt[0]
    c = saved;
}

// (kErrInval comes from hle.h)
void mutex_lock(Context& c) {
    if (g_replay_from >= 0) gx_replay(c);
    Mutex* m = get_or_create<Mutex>(c, arg(c, 0));
    if (!m) return ret(c, kErrInval);
    try { m->lock(); } catch (const std::exception& e) { std::fprintf(stderr, "mutex_lock: %s\n", e.what()); throw; }
    ret(c, 0);
}
void mutex_trylock(Context& c) {
    Mutex* m = get_or_create<Mutex>(c, arg(c, 0));
    ret(c, !m ? kErrInval : m->try_lock() ? 0 : kErrBusy);
}
void mutex_unlock(Context& c) {
    Mutex* m = get_or_create<Mutex>(c, arg(c, 0));
    if (!m) return ret(c, kErrInval);
    m->unlock();
    ret(c, 0);
}
void mutex_destroy(Context& c) {
    std::lock_guard lk(g_create_mutex);  // vs. a concurrent lazy create/lookup of the same slot
    Mutex* m = get<Mutex>(c, rt::ld<uint64_t>(c, arg(c, 0)));
    if (m && m->owner.load()) return ret(c, kErrBusy);  // libthr: EBUSY, object stays valid
    guest_delete(m);
    rt::st<uint64_t>(c, arg(c, 0), 0);
    ret(c, 0);
}

void cond_init(Context& c) { rt::st<uint64_t>(c, arg(c, 0), 0); get_or_create<Cond>(c, arg(c, 0)); ret(c, 0); }
// BB_TRACE_COND: signals per cond slot and the slot each thread currently waits on; the hang watchdog prints both
// (tells "nobody signals" from "signalled but the predicate never holds").
std::mutex g_cond_trace_mutex;
std::map<uint64_t, uint64_t> g_sigcount;       // slot -> signals so far
std::map<uint64_t, std::pair<uint64_t, uint64_t>> g_waiters;  // fs_base -> (slot, signals at wait start)
bool cond_trace() {
    static const bool on = std::getenv("BB_TRACE_COND") != nullptr;
    return on;
}
void count_signal(uint64_t slot) {
    if (!cond_trace()) return;
    std::lock_guard lk(g_cond_trace_mutex);
    ++g_sigcount[slot];
}
void cond_wait(Context& c) {
    Cond* cv = get_or_create<Cond>(c, arg(c, 0));
    Mutex* m = get_or_create<Mutex>(c, arg(c, 1));
    if (!cv || !m) return ret(c, kErrInval);
    CondLock cl{m};
    if (cond_trace()) {
        std::lock_guard lk(g_cond_trace_mutex);
        g_waiters[c.fs_base] = {arg(c, 0), g_sigcount[arg(c, 0)]};
    }
    cv->cv.wait(cl);
    if (cond_trace()) {
        std::lock_guard lk(g_cond_trace_mutex);
        g_waiters.erase(c.fs_base);
    }
    ret(c, 0);
}
void cond_timedwait(Context& c) {  // (cond, mutex, usec): relative timeout on Orbis scePthreadCondTimedwait
    Cond* cv = get_or_create<Cond>(c, arg(c, 0));
    Mutex* m = get_or_create<Mutex>(c, arg(c, 1));
    if (!cv || !m) return ret(c, kErrInval);
    CondLock cl{m};
    ret(c, cv->cv.wait_for(cl, std::chrono::microseconds(arg(c, 2))) == std::cv_status::timeout ? kErrTimedOut : 0);
}
void cond_signal(Context& c) { Cond* cv = get_or_create<Cond>(c, arg(c, 0)); count_signal(arg(c, 0)); if (cv) cv->cv.notify_one(); ret(c, cv ? 0 : kErrInval); }
void cond_broadcast(Context& c) { Cond* cv = get_or_create<Cond>(c, arg(c, 0)); count_signal(arg(c, 0)); if (cv) cv->cv.notify_all(); ret(c, cv ? 0 : kErrInval); }
void cond_destroy(Context& c) {
    guest_delete(get<Cond>(c, rt::ld<uint64_t>(c, arg(c, 0))));
    rt::st<uint64_t>(c, arg(c, 0), 0);
    ret(c, 0);
}

void rw_init(Context& c) { rt::st<uint64_t>(c, arg(c, 0), 0); get_or_create<RwLock>(c, arg(c, 0)); ret(c, 0); }
void rw_rdlock(Context& c) {
    RwLock* l = get_or_create<RwLock>(c, arg(c, 0));
    if (!l) return ret(c, kErrInval);
    std::unique_lock lk(l->m);
    l->cv.wait(lk, [&] { return !l->writer; });
    ++l->readers;
    ret(c, 0);
}
void rw_wrlock(Context& c) {
    RwLock* l = get_or_create<RwLock>(c, arg(c, 0));
    if (!l) return ret(c, kErrInval);
    std::unique_lock lk(l->m);
    l->cv.wait(lk, [&] { return !l->writer && l->readers == 0; });
    l->writer = true;
    ret(c, 0);
}
void rw_unlock(Context& c) {
    RwLock* l = get_or_create<RwLock>(c, arg(c, 0));
    if (!l) return ret(c, kErrInval);
    {
        std::lock_guard lk(l->m);
        if (l->writer) l->writer = false; else --l->readers;
    }
    l->cv.notify_all();
    ret(c, 0);
}
void rw_destroy(Context& c) {
    guest_delete(get<RwLock>(c, rt::ld<uint64_t>(c, arg(c, 0))));
    rt::st<uint64_t>(c, arg(c, 0), 0);
    ret(c, 0);
}
// Orbis try/timed variants: kErrBusy when not free now, kErrTimedOut after the relative timeout (u32 usec).
void rw_tryrdlock(Context& c) {
    RwLock* l = get_or_create<RwLock>(c, arg(c, 0));
    if (!l) return ret(c, kErrInval);
    std::lock_guard lk(l->m);
    if (l->writer) return ret(c, kErrBusy);
    ++l->readers;
    ret(c, 0);
}
void rw_trywrlock(Context& c) {
    RwLock* l = get_or_create<RwLock>(c, arg(c, 0));
    if (!l) return ret(c, kErrInval);
    std::lock_guard lk(l->m);
    if (l->writer || l->readers) return ret(c, kErrBusy);
    l->writer = true;
    ret(c, 0);
}
void rw_timedrdlock(Context& c) {  // (rwlock, usec)
    RwLock* l = get_or_create<RwLock>(c, arg(c, 0));
    if (!l) return ret(c, kErrInval);
    std::unique_lock lk(l->m);
    if (!l->cv.wait_for(lk, std::chrono::microseconds(arg32(c, 1)), [&] { return !l->writer; })) return ret(c, kErrTimedOut);
    ++l->readers;
    ret(c, 0);
}
void rw_timedwrlock(Context& c) {  // (rwlock, usec)
    RwLock* l = get_or_create<RwLock>(c, arg(c, 0));
    if (!l) return ret(c, kErrInval);
    std::unique_lock lk(l->m);
    if (!l->cv.wait_for(lk, std::chrono::microseconds(arg32(c, 1)), [&] { return !l->writer && l->readers == 0; }))
        return ret(c, kErrTimedOut);
    l->writer = true;
    ret(c, 0);
}
// ponytail: polls try_lock every 100 us (Mutex wraps a plain std::mutex); a timed_mutex if a hot path ever waits here.
void mutex_timedlock(Context& c) {  // (mutex, usec)
    Mutex* m = get_or_create<Mutex>(c, arg(c, 0));
    if (!m) return ret(c, kErrInval);
    const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(arg32(c, 1));
    while (!m->try_lock()) {
        if (std::chrono::steady_clock::now() >= until) return ret(c, kErrTimedOut);
        precise_sleep_us(100);
    }
    ret(c, 0);
}

// ---- semaphores --------------------------------------------------------------------------------------------------
void sema_create(Context& c) {  // (SceKernelSema* out, name, attr, initCount, maxCount, opt)
    auto* s = guest_new<Sema>();
    s->count = int64_t(argi(c, 3));
    s->max = int64_t(argi(c, 4));
    if (arg(c, 1)) s->name = std::string(ptr<const char>(c, arg(c, 1)), strnlen(ptr<const char>(c, arg(c, 1)), 64));
    if (std::getenv("BB_TRACE_HLE"))
        std::fprintf(stderr, "sema %p created '%s' init=%lld max=%lld\n", (void*)s, s->name.c_str(), (long long)s->count, (long long)s->max);
    rt::st<uint64_t>(c, arg(c, 0), reinterpret_cast<uintptr_t>(s));
    ret(c, 0);
}
void sema_wait(Context& c) {  // (sema, need, uint32_t* timeoutUs)
    auto* s = get<Sema>(c, arg(c, 0));
    const int64_t need = argi(c, 1);
    std::unique_lock lk(s->m);
    if (arg(c, 2)) {
        const auto us = std::chrono::microseconds(rt::ld<uint32_t>(c, arg(c, 2)));
        if (!s->cv.wait_for(lk, us, [&] { return s->count >= need; })) {
            rt::st<uint32_t>(c, arg(c, 2), 0);
            return ret(c, kErrTimedOut);
        }
    } else {
        s->cv.wait(lk, [&] { return s->count >= need; });
    }
    s->count -= need;
    ret(c, 0);
}
void sema_poll(Context& c) {
    auto* s = get<Sema>(c, arg(c, 0));
    const int64_t need = argi(c, 1);
    std::lock_guard lk(s->m);
    if (s->count < need) return ret(c, kErrBusy);
    s->count -= need;
    ret(c, 0);
}
void sema_signal(Context& c) {
    auto* s = get<Sema>(c, arg(c, 0));
    {
        std::lock_guard lk(s->m);
        s->count += argi(c, 1);
    }
    s->cv.notify_all();
    ret(c, 0);
}
void sema_delete(Context& c) { guest_delete(get<Sema>(c, arg(c, 0))); ret(c, 0); }

// ---- threads -----------------------------------------------------------------------------------------------------
struct ThreadExit { uint64_t value; };

struct GuestThread {
    HostThread host;
    uint64_t tcb = 0;
    uint64_t result = 0;
    uint64_t entry = 0, arg = 0;
    std::atomic<bool> done{false};
};

std::mutex g_threads_m;
std::map<uint64_t, std::unique_ptr<GuestThread>> g_threads;  // handle -> thread (handles are object addresses)

// Guest stacks need room for the guest frames plus red zone; the host stack recursion is sized separately.
constexpr uint64_t kHostStack = 256ull << 20;
constexpr uint64_t kThreadObjectSize = 0x400;
uint64_t new_thread_object() {
    void* p = guest_alloc(kThreadObjectSize);
    std::memset(p, 0, kThreadObjectSize);
    return reinterpret_cast<uintptr_t>(p);
}

void attr_obj_init(Context& c) { rt::st<uint64_t>(c, arg(c, 0), reinterpret_cast<uintptr_t>(guest_new<Attr>())); ret(c, 0); }
void attr_destroy(Context& c) { guest_delete(get<Attr>(c, rt::ld<uint64_t>(c, arg(c, 0)))); rt::st<uint64_t>(c, arg(c, 0), 0); ret(c, 0); }
void attr_setstacksize(Context& c) { get<Attr>(c, rt::ld<uint64_t>(c, arg(c, 0)))->stack_size = arg(c, 1); ret(c, 0); }
void attr_setdetach(Context& c) { get<Attr>(c, rt::ld<uint64_t>(c, arg(c, 0)))->detach = int(arg(c, 1)); ret(c, 0); }

void thread_create(Context& c) {  // (ScePthread* out, const ScePthreadAttr* attr, entry, arg, name)
    const uint64_t attr_h = arg(c, 1) ? rt::ld<uint64_t>(c, arg(c, 1)) : 0;
    const uint64_t stack_size = attr_h ? get<Attr>(c, attr_h)->stack_size : 256 << 10;
    auto t = std::make_unique<GuestThread>();
    GuestThread* th = t.get();
    th->entry = arg(c, 2);
    th->arg = arg(c, 3);
    // The handle is a zeroed block, not the host object: guest code may read fields of its ScePthread.
    const uint64_t handle = new_thread_object();
    std::string name = arg(c, 4) ? ptr<const char>(c, arg(c, 4)) : "";
    char label[96];
    std::snprintf(label, sizeof label, "worker '%s' entry 0x%llx", name.c_str(),
                  (unsigned long long)(th->entry - rt::kLoadBias));
    const char* debug_name = (new std::string(label))->c_str();  // leaked on purpose  // lives as long as the process; shown in crash/hang dumps
    {  // libthr stores *thread before the new thread can run; the guest may read its own handle slot immediately
        std::lock_guard lk(g_threads_m);
        g_threads.emplace(handle, std::move(t));
    }
    rt::st<uint64_t>(c, arg(c, 0), handle);
    const bool ok = start_guest_thread(th->host, stack_size, debug_name, handle, [th](Context& tc) {
        th->tcb = tc.fs_base;
        tc.r[4] -= 8;  // return address slot: rsp = 8 (mod 16) at entry
        rt::st<uint64_t>(tc, tc.r[4], 0);
        tc.r[7] = th->arg;
        try {
            rt::call_indirect(tc, th->entry);
            th->result = tc.r[0];
        } catch (const ThreadExit& e) {
            th->result = e.value;
        }
        th->done = true;
    });
    if (!ok) {
        std::lock_guard lk(g_threads_m);
        g_threads.erase(handle);
        rt::st<uint64_t>(c, arg(c, 0), 0);
        return ret(c, kErrNoMem);
    }
    ret(c, 0);
}
void thread_self(Context& c) {
    uint64_t h = rt::ld<uint64_t>(c, c.fs_base + rt::GuestTcb::kThread);
    if (!h) {  // main thread: handle created on first use
        h = new_thread_object();
        rt::st<uint64_t>(c, c.fs_base + rt::GuestTcb::kThread, h);
    }
    ret(c, h);
}
void thread_exit(Context& c) { throw ThreadExit{arg(c, 0)}; }
void thread_join(Context& c) {  // (thread, void** value)
    std::unique_ptr<GuestThread> t;
    {
        std::lock_guard lk(g_threads_m);
        auto it = g_threads.find(arg(c, 0));
        if (it == g_threads.end()) return ret(c, kErrSrch);
        t = std::move(it->second);
        g_threads.erase(it);
    }
    t->host.join();
    guest_free(ptr(c, arg(c, 0)));
    if (arg(c, 1)) rt::st<uint64_t>(c, arg(c, 1), t->result);
    ret(c, 0);
}
void thread_detach(Context& c) { ret(c, 0); }  // ponytail: detached threads are never reaped
void sched_yield_(Context& c) { std::this_thread::yield(); ret(c, 0); }

} // namespace

bool start_guest_thread(HostThread& host, uint64_t stack_size, const char* debug_name, uint64_t handle, std::function<void(Context&)> body) {
    const uint64_t tls_bytes = (rt::tls_block_size(*process().image) + 0xFFF) & ~uint64_t(0xFFF);
    const uint64_t area = reinterpret_cast<uintptr_t>(host_map(nullptr, tls_bytes + stack_size));
    if (!area) return false;
    if (!handle) handle = new_thread_object();
    return host.start(kHostStack, [=, body = std::move(body)] {
        Context tc;
        tc.base = 0;
        std::string err;
        if (!rt::build_tls_block(tc, *process().image, area, process().canary, err)) {
            std::fprintf(stderr, "bbrt: thread TLS setup failed: %s\n", err.c_str());
            std::abort();
        }
        rt::current_context = &tc;
        rt::register_debug_thread(debug_name, current_os_thread_id());
        rt::st<uint64_t>(tc, tc.fs_base + rt::GuestTcb::kThread, handle);
        tc.r[4] = (area + tls_bytes + stack_size) & ~uint64_t(15);
        body(tc);
        rt::unregister_debug_thread();
    });
}

void print_cond_waiters() {
    if (!cond_trace()) return;
    std::lock_guard lk(g_cond_trace_mutex);
    for (const auto& [fs, w] : g_waiters)
        std::fprintf(stderr, "cond waiter: fs_base 0x%llx waits on slot 0x%llx, signals since wait start: %llu\n", (unsigned long long)fs,
                     (unsigned long long)w.first, (unsigned long long)(g_sigcount[w.first] - w.second));
}

void register_thread() {
    struct Pair { const char* sce; const char* posix; rt::GuestFn fn; };
    static const Pair kPairs[] = {
        {"scePthreadMutexattrInit", "pthread_mutexattr_init", attr_init},
        {"scePthreadMutexattrSettype", "pthread_mutexattr_settype", zero},
        {"scePthreadMutexattrDestroy", "pthread_mutexattr_destroy", zero},
        {"scePthreadMutexInit", "pthread_mutex_init", mutex_init},
        {"scePthreadMutexLock", "pthread_mutex_lock", mutex_lock},
        {"scePthreadMutexTrylock", "pthread_mutex_trylock", mutex_trylock},
        {"scePthreadMutexUnlock", "pthread_mutex_unlock", mutex_unlock},
        {"scePthreadMutexDestroy", "pthread_mutex_destroy", mutex_destroy},
        {"scePthreadCondInit", "pthread_cond_init", cond_init},
        {"scePthreadCondWait", "pthread_cond_wait", cond_wait},
        {"scePthreadCondTimedwait", "pthread_cond_timedwait", cond_timedwait},
        {"scePthreadCondSignal", "pthread_cond_signal", cond_signal},
        {"scePthreadCondBroadcast", "pthread_cond_broadcast", cond_broadcast},
        {"scePthreadCondDestroy", "pthread_cond_destroy", cond_destroy},
        {"scePthreadRwlockInit", "pthread_rwlock_init", rw_init},
        {"scePthreadRwlockRdlock", "pthread_rwlock_rdlock", rw_rdlock},
        {"scePthreadRwlockWrlock", "pthread_rwlock_wrlock", rw_wrlock},
        {"scePthreadRwlockUnlock", "pthread_rwlock_unlock", rw_unlock},
        {"scePthreadRwlockDestroy", "pthread_rwlock_destroy", rw_destroy},
        {"scePthreadSelf", "pthread_self", thread_self},
        {"scePthreadExit", "pthread_exit", thread_exit},
        {"scePthreadJoin", "pthread_join", thread_join},
        {"scePthreadDetach", "pthread_detach", thread_detach},
    };
    for (const Pair& p : kPairs) {
        reg(p.sce, p.fn);
        reg(p.posix, p.fn);
    }
    reg("scePthreadRwlockTryrdlock", rw_tryrdlock);
    reg("scePthreadRwlockTrywrlock", rw_trywrlock);
    reg("scePthreadRwlockTimedrdlock", rw_timedrdlock);
    reg("scePthreadRwlockTimedwrlock", rw_timedwrlock);
    reg("scePthreadMutexTimedlock", mutex_timedlock);
    // POSIX differs from the Orbis variants in two return conventions: errno values and absolute cond timeouts.
    reg("pthread_mutex_trylock", [](Context& c) { mutex_trylock(c); if (c.r[0]) c.r[0] = 16; });  // EBUSY
    reg("pthread_cond_timedwait", [](Context& c) {  // (cond, mutex, const timespec* abstime)
        Cond* cv = get_or_create<Cond>(c, arg(c, 0));
        Mutex* m = get_or_create<Mutex>(c, arg(c, 1));
        if (!cv || !m) return ret(c, kErrInval);
        const int64_t sec = rt::ld<int64_t>(c, arg(c, 2)), nsec = rt::ld<int64_t>(c, arg(c, 2) + 8);
        const std::chrono::system_clock::time_point at{
            std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::nanoseconds(sec * 1000000000 + nsec))};
        CondLock cl{m};
        ret(c, cv->cv.wait_until(cl, at) == std::cv_status::timeout ? 60 : 0);  // ETIMEDOUT
    });
    reg("scePthreadAttrInit", attr_obj_init);
    reg("scePthreadAttrDestroy", attr_destroy);
    reg("scePthreadAttrSetstacksize", attr_setstacksize);
    reg("scePthreadAttrSetdetachstate", attr_setdetach);
    reg("scePthreadAttrSetschedpolicy", zero);
    reg("scePthreadAttrSetinheritsched", zero);
    reg("scePthreadAttrSetschedparam", zero);
    reg("scePthreadAttrSetaffinity", zero);
    reg("scePthreadSetaffinity", zero);
    reg("scePthreadSetprio", zero);
    reg("scePthreadRename", zero);
    reg("scePthreadCreate", thread_create);
    reg("pthread_create", thread_create);
    reg("pthread_create_name_np", thread_create);  // (thread*, attr*, entry, arg, name): same layout
    reg("pthread_attr_init", attr_obj_init);
    reg("pthread_attr_destroy", attr_destroy);
    reg("pthread_attr_setstacksize", attr_setstacksize);
    reg("pthread_attr_setdetachstate", attr_setdetach);
    reg("pthread_attr_setschedparam", zero);
    reg("sceKernelCreateSema", sema_create);
    reg("sceKernelWaitSema", sema_wait);
    reg("sceKernelPollSema", sema_poll);
    reg("sceKernelSignalSema", sema_signal);
    reg("sceKernelDeleteSema", sema_delete);
    reg("sched_yield", sched_yield_);
    reg("scePthreadYield", sched_yield_);
}

} // namespace bb::hle
