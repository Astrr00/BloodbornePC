// SPDX-License-Identifier: GPL-3.0-or-later
#include "runtime/guest.h"

#ifdef _MSC_VER
#include <excpt.h>
#endif
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <utility>
#include <vector>

namespace bb::rt {

thread_local Context* current_context = nullptr;
thread_local const ImportEntry* last_import = nullptr;
thread_local TraceRing trace_ring;

uint64_t watch_addr = [] {
    const char* e = std::getenv("BB_WATCH");
    return e ? std::strtoull(e, nullptr, 16) : ~0ull;
}();
uint64_t watch_value = [] {
    const char* e = std::getenv("BB_WATCH_VALUE");
    return e ? std::strtoull(e, nullptr, 16) : ~0ull;
}();

uint64_t trace_fn = [] {
    const char* e = std::getenv("BB_TRACE_FN");
    return e ? std::strtoull(e, nullptr, 16) + kLoadBias : ~0ull;
}();

void trace_fn_report(const Context& c, uint64_t addr, bool entry) {
    if (entry)
        std::fprintf(stderr, "fn 0x%llx enter: rdi=%llx rsi=%llx rdx=%llx rcx=%llx r8=%llx r9=%llx rsp=%llx\n",
                     (unsigned long long)(addr - kLoadBias), (unsigned long long)c.r[7], (unsigned long long)c.r[6],
                     (unsigned long long)c.r[2], (unsigned long long)c.r[1], (unsigned long long)c.r[8],
                     (unsigned long long)c.r[9], (unsigned long long)c.r[4]);
    else
        std::fprintf(stderr, "fn 0x%llx leave: rax=%llx rdx=%llx\n", (unsigned long long)(addr - kLoadBias),
                     (unsigned long long)c.r[0], (unsigned long long)c.r[2]);
}

void bad_access(Context& c, uint64_t addr, size_t size, bool write) {
    std::fprintf(stderr, "\nbbrt: guest %s of %zu bytes at low address 0x%llx; recent entries (vaddr):", write ? "store" : "load",
                 size, (unsigned long long)addr);
    const TraceRing& t = trace_ring;
    for (uint32_t i = t.next > 12 ? t.next - 12 : 0; i < t.next; ++i)
        std::fprintf(stderr, " %llx", (unsigned long long)(t.entries[i & 63] - kLoadBias));
    std::fprintf(stderr, "\n  rax=%llx rcx=%llx rdx=%llx rbx=%llx rsp=%llx rbp=%llx rsi=%llx rdi=%llx\n  r8=%llx r9=%llx r10=%llx r11=%llx r12=%llx r13=%llx r14=%llx r15=%llx\n",
                 (unsigned long long)c.r[0], (unsigned long long)c.r[1], (unsigned long long)c.r[2], (unsigned long long)c.r[3],
                 (unsigned long long)c.r[4], (unsigned long long)c.r[5], (unsigned long long)c.r[6], (unsigned long long)c.r[7],
                 (unsigned long long)c.r[8], (unsigned long long)c.r[9], (unsigned long long)c.r[10], (unsigned long long)c.r[11],
                 (unsigned long long)c.r[12], (unsigned long long)c.r[13], (unsigned long long)c.r[14], (unsigned long long)c.r[15]);
    std::fprintf(stderr, "  [rsp]=%llx; rbp chain (return vaddrs):", (unsigned long long)(ld<uint64_t>(c, c.r[4]) - kLoadBias));
    for (uint64_t bp = c.r[5], i = 0; i < 12 && bp > 0x10000 && bp < 0x7e0000000000ull; ++i) {
        std::fprintf(stderr, " %llx", (unsigned long long)(ld<uint64_t>(c, bp + 8) - kLoadBias));
        const uint64_t next = ld<uint64_t>(c, bp);
        if (next <= bp) break;
        bp = next;
    }
    std::fputc('\n', stderr);
    *static_cast<volatile int*>(nullptr) = 0;  // provoke the crash handler (it also runs the BB_DUMP/BB_STR hooks); abort() would bypass it
    std::abort();
}

void watch_hit(Context& c, uint64_t addr, size_t size, uint64_t value) {
    std::fprintf(stderr, "bbrt: watch hit: store %zu bytes at 0x%llx value 0x%llx; recent entries:", size,
                 (unsigned long long)addr, (unsigned long long)value);
    const TraceRing& t = trace_ring;
    for (uint32_t i = t.next > 12 ? t.next - 12 : 0; i < t.next; ++i)
        std::fprintf(stderr, " %llx", (unsigned long long)(t.entries[i & 63] - kLoadBias));
    std::fprintf(stderr, "  rsp=%llx\n", (unsigned long long)c.r[4]);
}

void unsupported(Context& c, uint64_t ip, const char* mnemonic) {
    std::fprintf(stderr, "bbrt: unsupported instruction '%s' at guest vaddr 0x%llx (rsp=0x%llx)\n", mnemonic,
                 (unsigned long long)ip, (unsigned long long)c.r[4]);
    std::abort();
}

void divide_error(Context& c) {
    std::fprintf(stderr, "bbrt: guest divide error (#DE), rsp=0x%llx\n", (unsigned long long)c.r[4]);
    std::abort();
}

namespace {
std::mutex g_debug_mutex;
std::vector<ThreadDebug> g_debug_threads;
}
void register_debug_thread(const char* name, uint64_t os_thread_id) {
    std::lock_guard lk(g_debug_mutex);
    g_debug_threads.push_back({name, &current_context, &trace_ring, &last_import, os_thread_id});
}
const std::vector<ThreadDebug>& debug_threads() { return g_debug_threads; }
// Not declared in guest.h (included by every generated shard): thread exit removes the entry whose thread_locals die,
// and reports iterate over a snapshot taken under the same lock.
void unregister_debug_thread() {
    std::lock_guard lk(g_debug_mutex);
    std::erase_if(g_debug_threads, [](const ThreadDebug& t) { return t.context == &current_context; });
}
std::vector<ThreadDebug> debug_threads_snapshot() {
    std::lock_guard lk(g_debug_mutex);
    return g_debug_threads;
}

namespace {
// Guest address -> V by open addressing (Fibonacci hash, linear probing, load <= 1/2): about one probe where a binary
// search over 179k kFunctions took 17 mispredicted steps. Built during static init and read-only afterwards, so lookups
// take no locks. Address 0 marks empty slots (nothing lives there); their V() is the miss result.
template <typename V>
struct AddrTable {
    struct Slot {
        uint64_t addr;
        V v;
    };
    std::vector<Slot> slots;
    int shift = 63;
    explicit AddrTable(size_t n) {
        while ((size_t(1) << (64 - shift)) < 2 * n) --shift;
        slots.resize(size_t(1) << (64 - shift));
    }
    size_t home(uint64_t a) const { return size_t((a * 0x9E3779B97F4A7C15ull) >> shift); }
    void insert(uint64_t a, V v) {  // duplicates: the first insert wins, as lower_bound did
        size_t i = home(a);
        while (slots[i].addr) i = (i + 1) & (slots.size() - 1);
        slots[i] = {a, v};
    }
    V find(uint64_t a) const {
        for (size_t i = home(a);; i = (i + 1) & (slots.size() - 1))
            if (slots[i].addr == a || !slots[i].addr) return slots[i].v;
    }
};

const AddrTable<GuestFn> g_functions = [] {
    AddrTable<GuestFn> t(kFunctionCount);
    for (size_t i = 0; i < kFunctionCount; ++i) t.insert(kFunctions[i].addr, kFunctions[i].fn);
    return t;
}();
// PLT entries and synthetic import addresses (kImportBase + symbol * kImportStride) -> kImports entry.
const AddrTable<const ImportEntry*> g_imports = [] {
    AddrTable<const ImportEntry*> t(2 * kImportCount);
    for (const ImportEntry* it = kImports; it != kImports + kImportCount; ++it) {
        t.insert(kImportBase + uint64_t(it->symbol) * kImportStride, it);
        if (it->plt) t.insert(it->plt, it);
    }
    return t;
}();
// PLT stubs and synthetic import addresses lie above every recompiled function: import calls skip the function table.
const uint64_t g_imports_lo = [] {
    uint64_t lo = kImportBase;
    for (const ImportEntry* it = kImports; it != kImports + kImportCount; ++it)
        if (it->plt && it->plt < lo) lo = it->plt;
    return lo;
}();
// Host implementation per kImports entry, resolved on its first call (the registry is a string-keyed map).
std::vector<std::atomic<GuestFn>> g_hle(kImportCount);
// Options served by call_hle_debug; BB_HANG_SECS because its watchdog prints the per-import call counts.
const bool g_hle_debug =
    std::getenv("BB_TRACE_HLE") || std::getenv("BB_TRACE_ARG0") || std::getenv("BB_TRACE_SLOW") || std::getenv("BB_HANG_SECS");
}  // namespace

GuestFn lookup(uint64_t addr) { return g_functions.find(addr); }

const ImportEntry* find_import(uint64_t target) { return g_imports.find(target); }

std::atomic<uint64_t>* import_call_counts() {
    static std::vector<std::atomic<uint64_t>> counts(kImportCount);
    return counts.data();
}

namespace {
void call_hle_debug(Context& c, const ImportEntry* imp, GuestFn hle) {
    ++import_call_counts()[size_t(imp - kImports)];
    static const bool trace = std::getenv("BB_TRACE_HLE") != nullptr;  // debug: log every HLE call
    // BB_TRACE_HLE_AFTER=<s>: only after that many seconds, and only sce* calls without the noisy sync/Gnm ones
    static const double trace_after = std::getenv("BB_TRACE_HLE_AFTER") ? std::atof(std::getenv("BB_TRACE_HLE_AFTER")) : -1.0;
    static const auto t0 = std::chrono::steady_clock::now();
    const bool trace_now = trace && (trace_after < 0 ||
        (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() >= trace_after && !std::strncmp(imp->name, "sce", 3) &&
         !std::strstr(imp->name, "Mutex") && !std::strstr(imp->name, "Rwlock") && !std::strstr(imp->name, "Gnm") && !std::strstr(imp->name, "Self")));
    // BB_WATCH + BB_TRACE_HLE: report the HLE call after which the watched qword changed (host-side writes bypass the store watch)
    auto read_watch = [&](uint64_t& v) {
#ifdef _MSC_VER
        __try { v = ld<uint64_t>(c, watch_addr); return true; } __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
#else
        v = ld<uint64_t>(c, watch_addr);
        return true;
#endif
    };
    // BB_TRACE_ARG0=<hex>: additionally log every HLE call whose first argument has that value (a semaphore/cond handle)
    static const uint64_t arg0_watch = std::getenv("BB_TRACE_ARG0") ? std::strtoull(std::getenv("BB_TRACE_ARG0"), nullptr, 16) : 0;
    if (trace_now || (arg0_watch && c.r[7] == arg0_watch))
        std::fprintf(stderr, "hle[%llx] %.3fs %s(%llx, %llx, %llx, %llx)\n", (unsigned long long)c.fs_base,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), imp->name[0] ? imp->name : imp->nid,
                     (unsigned long long)c.r[7], (unsigned long long)c.r[6], (unsigned long long)c.r[2], (unsigned long long)c.r[1]);
    uint64_t wbefore = 0;
    const bool wb = trace && watch_addr != ~0ull && read_watch(wbefore);
    // BB_TRACE_SLOW=<ms>: log HLE calls that block at least that long (which wait stalls a thread)
    static const double slow_ms = std::getenv("BB_TRACE_SLOW") ? std::atof(std::getenv("BB_TRACE_SLOW")) : 0.0;
    if (slow_ms > 0) {
        const uint64_t a0 = c.r[7], a1 = c.r[6], a2 = c.r[2], a3 = c.r[1];
        const auto s0 = std::chrono::steady_clock::now();
        hle(c);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
        if (ms >= slow_ms)
            std::fprintf(stderr, "slow[%llx] %s(%llx, %llx, %llx, %llx) %.0f ms\n", (unsigned long long)c.fs_base, imp->name[0] ? imp->name : imp->nid,
                         (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2, (unsigned long long)a3, ms);
    } else
    hle(c);
    if (wb) {
        uint64_t v;
        if (read_watch(v) && v != wbefore)
            std::fprintf(stderr, "  [watch] %llx -> %llx inside the HLE call above\n", (unsigned long long)wbefore, (unsigned long long)v);
    }
}
}  // namespace

void call_indirect(Context& c, uint64_t target) {
    if (target < g_imports_lo)
        if (GuestFn fn = g_functions.find(target)) return fn(c);
    if (const ImportEntry* imp = g_imports.find(target)) {
        std::atomic<GuestFn>& slot = g_hle[size_t(imp - kImports)];
        GuestFn hle = slot.load(std::memory_order_relaxed);
        if (!hle && (hle = find_hle(imp->nid))) slot.store(hle, std::memory_order_relaxed);
        if (hle) {
            last_import = imp;
            if (g_hle_debug)
                call_hle_debug(c, imp, hle);
            else
                hle(c);
            c.r[4] += 8;  // HLE functions return like the guest callee would: pop the pushed return address
            return;
        }
        static const bool stub_mode = std::getenv("BB_STUB_IMPORTS") != nullptr;  // debug: map the boot path
        if (stub_mode) {
            static std::vector<std::pair<const ImportEntry*, uint64_t>> hits;
            if (hits.empty()) std::atexit([] {
                std::fprintf(stderr, "bbrt: %zu distinct stubbed imports (first-hit order):\n", hits.size());
                for (auto& h : hits)
                    std::fprintf(stderr, "  %s %s %s x%llu\n", h.first->library, h.first->nid, h.first->name,
                                 (unsigned long long)h.second);
            });
            auto it = std::find_if(hits.begin(), hits.end(), [&](auto& h) { return h.first == imp; });
            if (it == hits.end()) {
                hits.emplace_back(imp, 1);
                std::fprintf(stderr, "bbrt: stub %s %s %s\n", imp->library, imp->nid, imp->name);
            } else ++it->second;
            c.r[0] = 0;
            c.r[4] += 8;
            return;
        }
        std::fprintf(stderr, "bbrt: unimplemented import %s %s %s\n", imp->library, imp->nid,
                     imp->name[0] ? imp->name : "(name unknown)");
        std::abort();
    }
    if (target == kExitAddress) {
        std::fprintf(stderr, "bbrt: guest called the exit function\n");
        std::exit(int(c.r[7]));
    }
    const FunctionEntry* end = kFunctions + kFunctionCount;
    const FunctionEntry* it = std::upper_bound(kFunctions, end, target, [](uint64_t v, const FunctionEntry& e) { return v < e.addr; });
    std::fprintf(stderr, "bbrt: indirect transfer to unknown guest address 0x%llx (inside function 0x%llx) rsp=0x%llx [rsp]=0x%llx\n",
                 (unsigned long long)target, it != kFunctions ? (unsigned long long)(it - 1)->addr : 0ull,
                 (unsigned long long)c.r[4], (unsigned long long)ld<uint64_t>(c, c.r[4]));
    std::abort();
}

} // namespace bb::rt
