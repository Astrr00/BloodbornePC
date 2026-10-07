// SPDX-License-Identifier: GPL-3.0-or-later
#include "hle/hle.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <pthread.h>
#include <sys/mman.h>
#endif

#include <memory>
#include <vector>
#include <atomic>
#include <mutex>

#include "core/nid.h"

namespace bb::hle {

void reg(const char* symbol, rt::GuestFn fn) { rt::register_hle(generate_nid(symbol).c_str(), fn); }

Process& process() {
    static Process p;
    return p;
}

namespace {
void* map_at(void* hint, uint64_t len) {
#ifdef _WIN32
    // MEM_WRITE_WATCH: the GPU backend asks which pages were written since it last looked (GetWriteWatch) instead of hashing buffers
    return VirtualAlloc(hint, len, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
#else
    void* p = mmap(hint, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | (hint ? MAP_FIXED_NOREPLACE : 0), -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}
} // namespace

// Anonymous mappings are placed in [64 GiB, 960 GiB): the guest masks pointers to 40 bits in places (see guest_heap.h).
void* host_map(void* hint, uint64_t len) {
    if (hint) return map_at(hint, len);
    static std::mutex m;
    static uint64_t next = 0x1000000000ull;
    constexpr uint64_t kLimit = 0xF000000000ull, kStep = 0x200000;  // 2 MiB: the engine's pools mask pointers to find chunk headers
    len = (len + kStep - 1) & ~(kStep - 1);
    std::lock_guard lk(m);
    for (uint64_t a = next; a + len < kLimit; a += kStep) {
        if (void* p = map_at(reinterpret_cast<void*>(a), len)) {
            next = a + len;
            return p;
        }
    }
    return nullptr;
}

namespace {
struct Start { std::function<void()> fn; };
#ifdef _WIN32
DWORD WINAPI thread_main(void* p) {
    std::unique_ptr<Start> s(static_cast<Start*>(p));
    s->fn();
    return 0;
}
#else
void* thread_main(void* p) {
    std::unique_ptr<Start> s(static_cast<Start*>(p));
    s->fn();
    return nullptr;
}
#endif
} // namespace

bool HostThread::start(uint64_t stack_bytes, std::function<void()> fn) {
    auto* s = new Start{std::move(fn)};
#ifdef _WIN32
    handle_ = CreateThread(nullptr, stack_bytes, thread_main, s, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (!handle_) delete s;
    return handle_ != nullptr;
#else
    pthread_attr_t a;
    pthread_attr_init(&a);
    pthread_attr_setstacksize(&a, stack_bytes);
    pthread_t t;
    const bool ok = pthread_create(&t, &a, thread_main, s) == 0;
    pthread_attr_destroy(&a);
    if (!ok) { delete s; return false; }
    handle_ = reinterpret_cast<void*>(t);
    return true;
#endif
}

void HostThread::join() {
    if (!handle_) return;
#ifdef _WIN32
    WaitForSingleObject(handle_, INFINITE);
    CloseHandle(handle_);
#else
    pthread_join(reinterpret_cast<pthread_t>(handle_), nullptr);
#endif
    handle_ = nullptr;
}

HostThread::~HostThread() { join(); }

uint64_t call_guest(Context& c, uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    const uint64_t saved_rsp = c.r[4];
    c.r[7] = a0;
    c.r[6] = a1;
    c.r[2] = a2;
    c.r[1] = a3;
    c.r[4] = ((saved_rsp - 256) & ~uint64_t(15)) - 8;  // callee entry: rsp = 8 mod 16 with the return address on top
    rt::st<uint64_t>(c, c.r[4], rt::kExitAddress);
    rt::call_indirect(c, fn);
    const uint64_t r = c.r[0];
    c.r[4] = saved_rsp;
    return r;
}

namespace {
std::mutex g_map_mutex;
std::vector<Mapping> g_mappings;
std::atomic<uint64_t> g_map_gen{1};  // bumped by every new mapping: invalidates the per-thread last hits in find_mapping
} // namespace

void note_mapping(uint64_t start, uint64_t len, int prot) {
    std::lock_guard lk(g_map_mutex);
    for (const Mapping& m : g_mappings)
        if (start < m.end && m.start < start + len)
            std::fprintf(stderr, "map: new mapping [0x%llx, 0x%llx) overlaps [0x%llx, 0x%llx)\n", (unsigned long long)start, (unsigned long long)(start + len), (unsigned long long)m.start, (unsigned long long)m.end);
    if (std::getenv("BB_MAP_LOG"))
        std::fprintf(stderr, "map: [0x%llx, 0x%llx) prot=%d\n", (unsigned long long)start, (unsigned long long)(start + len), prot);
    g_mappings.push_back({start, start + len, prot});
    g_map_gen.fetch_add(1);
}
bool find_mapping(uint64_t addr, Mapping& out) {
    // the GPU backend asks ~20 times per draw, mostly about a few pools: remember the last hits per thread (until any new mapping)
    thread_local Mapping last[4]{};
    thread_local uint64_t last_gen = 0;
    const uint64_t gen = g_map_gen.load(std::memory_order_acquire);
    if (last_gen != gen) {
        for (Mapping& m : last) m = {};
        last_gen = gen;
    }
    for (const Mapping& m : last)
        if (addr >= m.start && addr < m.end) { out = m; return true; }
    std::lock_guard lk(g_map_mutex);
    for (auto it = g_mappings.rbegin(); it != g_mappings.rend(); ++it)  // newest first: remaps win
        if (addr >= it->start && addr < it->end) {
            for (int i = 3; i > 0; --i) last[i] = last[i - 1];
            out = last[0] = *it;
            return true;
        }
    return false;
}

uint64_t current_os_thread_id() {
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return 0;
#endif
}

void register_all() {
    register_libc();
    register_kernel();
    register_thread();
    register_system();
    register_net();
    register_np();
    register_audio();
    register_vfs();
    register_gnm();
    register_input();
    register_services();
    register_savedata();
    register_ajm();
    register_ime();
}

} // namespace bb::hle
