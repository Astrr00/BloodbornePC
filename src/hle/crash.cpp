// SPDX-License-Identifier: GPL-3.0-or-later
// Fatal exception and hang reports: guest registers, recent function entries (needs -DBB_TRACE_CALLS) and a heuristic
// guest backtrace (stack words that point into recompiled functions). Host RIPs say nothing useful about guest code.
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <tuple>
#include <utility>

#include "hle/hle.h"
#include "gpu/gpu_hooks.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#include <tlhelp32.h>
#ifdef _MSC_VER
#pragma comment(lib, "dbghelp.lib")
#endif
#else
#include <csignal>
#endif

namespace bb::rt {
std::atomic<uint64_t>* import_call_counts();
std::vector<ThreadDebug> debug_threads_snapshot();  // guest.cpp (not in guest.h: that header is included by every generated shard)
}

namespace bb::hle {
namespace {

bool readable(uint64_t addr) {
#ifdef _WIN32
    return !IsBadReadPtr(reinterpret_cast<void*>(addr), 8);
#else
    return addr != 0;  // ponytail: no cheap probe on POSIX; a corrupt rsp can fault again inside the handler
#endif
}

// Absolute host RIP of another thread, 0 if unavailable; *ret_out = the stack word at [rsp] (the caller of a leaf routine).
uint64_t sample_rip_abs(uint64_t tid, uint64_t* ret_out = nullptr) {
#ifdef _WIN32
    if (!tid) return 0;
    HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE, DWORD(tid));
    if (!h) return 0;
    uint64_t rip = 0;
    if (SuspendThread(h) != DWORD(-1)) {
        CONTEXT ctx = {};
        ctx.ContextFlags = CONTEXT_CONTROL;
        if (GetThreadContext(h, &ctx)) {
            rip = ctx.Rip;
            if (ret_out && !IsBadReadPtr(reinterpret_cast<void*>(ctx.Rsp), 8)) *ret_out = *reinterpret_cast<const uint64_t*>(ctx.Rsp);
        }
        ResumeThread(h);
    }
    CloseHandle(h);
    return rip;
#else
    (void)tid; (void)ret_out;
    return 0;
#endif
}

// Host RIP of another thread as offset in the exe (resolve against the link map), 0 if unavailable.
uint64_t sample_rip(uint64_t tid) {
#ifdef _WIN32
    const auto exe = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
    const uint64_t rip = sample_rip_abs(tid);
    return rip > exe ? rip - exe : 0;
#else
    return sample_rip_abs(tid);
#endif
}

// "module!export+0xoff" (dbghelp falls back to the export table without PDBs), or "module+0xoff" for a RIP outside the exe.
std::string describe_rip(uint64_t rip) {
#ifdef _WIN32
    HMODULE mod = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(rip), &mod) || !mod)
        return "?+0x" + std::to_string(rip);
    char path[MAX_PATH] = "?";
    GetModuleFileNameA(mod, path, sizeof path);
    const char* base = std::strrchr(path, '\\');
    char out[512];
    static bool sym_init = SymInitialize(GetCurrentProcess(), nullptr, TRUE);  // ponytail: debug-only, one-time module scan
    alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 256] = {};
    auto* si = reinterpret_cast<SYMBOL_INFO*>(buf);
    si->SizeOfStruct = sizeof(SYMBOL_INFO);
    si->MaxNameLen = 255;
    DWORD64 disp = 0;
    if (sym_init && SymFromAddr(GetCurrentProcess(), rip, &disp, si))
        std::snprintf(out, sizeof out, "%s!%s+0x%llx", base ? base + 1 : path, si->Name, (unsigned long long)disp);
    else
        std::snprintf(out, sizeof out, "%s+0x%llx", base ? base + 1 : path, (unsigned long long)(rip - reinterpret_cast<uint64_t>(mod)));
    return out;
#else
    return std::to_string(rip);
#endif
}

void print_thread(const char* name, const Context* c, const rt::TraceRing& ring, const rt::ImportEntry* last) {
    std::fprintf(stderr, "--- guest thread %s\n", name);
    if (last) std::fprintf(stderr, "  last HLE import: %s %s\n", last->library, last->name);
    if (!c) { std::fputs("  (no context)\n", stderr); return; }
    std::fprintf(stderr, "  fs_base=%llx thread handle=%llx\n", (unsigned long long)c->fs_base,
                 (unsigned long long)*reinterpret_cast<const uint64_t*>(c->fs_base + c->base + 0x10));
    static const char* kNames[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                     "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};
    for (int i = 0; i < 16; ++i)
        std::fprintf(stderr, "  %-3s=%016llx%s", kNames[i], (unsigned long long)c->r[i], i % 4 == 3 ? "\n" : "");
    for (int i = 0; i < 16; ++i) {  // memory around register values that point into the anonymous-map heap (corruption forensics)
        const uint64_t v = c->r[i];
        if (i == 4 || v < 0x1000000000ull || v >= 0xF000000000ull || !readable(v - 0x20) || !readable(v + 0x5f)) continue;
        std::fprintf(stderr, "  mem around %s=%llx:", kNames[i], (unsigned long long)v);
        for (int o = -0x20; o < 0x60; o += 8) std::fprintf(stderr, "%s%llx", o % 32 ? " " : "\n    ", (unsigned long long)*reinterpret_cast<const uint64_t*>(v + o));
        std::fputc('\n', stderr);
    }
    if (ring.next) {
        std::fputs("  recent function entries (oldest first, vaddr):", stderr);
        for (uint32_t i = ring.next > 24 ? ring.next - 24 : 0; i < ring.next; ++i)
            std::fprintf(stderr, " %llx", (unsigned long long)(ring.entries[i & 63] - rt::kLoadBias));
        std::fputc('\n', stderr);
    }
    // Guest backtrace along the rbp chain (the engine keeps frame pointers): [rbp] = caller rbp, [rbp+8] = return address.
    const rt::FunctionEntry* end = rt::kFunctions + rt::kFunctionCount;
    auto describe = [&](uint64_t v) {
        auto it = std::upper_bound(rt::kFunctions, end, v, [](uint64_t a, const rt::FunctionEntry& e) { return a < e.addr; });
        if (it == rt::kFunctions) return;
        std::fprintf(stderr, "  frame: vaddr 0x%llx = function 0x%llx + 0x%llx\n", (unsigned long long)(v - rt::kLoadBias),
                     (unsigned long long)((it - 1)->addr - rt::kLoadBias), (unsigned long long)(v - (it - 1)->addr));
    };
    if (readable(c->r[4] + c->base)) {
        uint64_t top;
        std::memcpy(&top, reinterpret_cast<void*>(c->r[4] + c->base), 8);
        std::fputs("  [rsp] ", stderr);
        describe(top);
    }
    uint64_t bp = c->r[5];
    for (int i = 0; i < 24 && bp && readable(bp + c->base) && readable(bp + 8 + c->base); ++i) {
        uint64_t ret_addr, next;
        std::memcpy(&ret_addr, reinterpret_cast<void*>(bp + 8 + c->base), 8);
        std::memcpy(&next, reinterpret_cast<void*>(bp + c->base), 8);
        describe(ret_addr);
        if (next <= bp) break;
        bp = next;
    }
}

void print_import_counts() {
    {  // most frequent HLE imports so far
        std::vector<std::pair<uint64_t, const rt::ImportEntry*>> top;
        for (size_t i = 0; i < rt::kImportCount; ++i)
            if (uint64_t n = rt::import_call_counts()[i].load()) top.emplace_back(n, rt::kImports + i);
        std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.first > b.first; });
        std::fputs("  HLE call counts:", stderr);
        for (size_t i = 0; i < top.size() && i < 120; ++i)
            std::fprintf(stderr, " %s=%llu", top[i].second->name[0] ? top[i].second->name : top[i].second->nid, (unsigned long long)top[i].first);
        std::fputc('\n', stderr);
    }
}

void dump_env() {
    if (const char* dump = std::getenv("BB_DUMP")) {  // "hexaddr:hexbytes" guest memory at the hang
        unsigned long long a = 0, n = 0;
        if (std::sscanf(dump, "%llx:%llx", &a, &n) == 2) {
            for (unsigned long long o = 0; o < n; o += 8)
                std::fprintf(stderr, "  [%llx] %016llx\n", a + o, *reinterpret_cast<const unsigned long long*>(a + o));
        }
    }
    if (const char* strs = std::getenv("BB_STR")) {  // "hexaddr,hexaddr,...": print C strings (ASCII or UTF-16)
        for (const char* p = strs; *p;) {
            char* end = nullptr;
            const unsigned long long a = std::strtoull(p, &end, 16);
            const char* q = reinterpret_cast<const char*>(a);
            std::fprintf(stderr, "  str[%llx] = \"", a);
            if (q[1] == 0)
                for (const char16_t* w = reinterpret_cast<const char16_t*>(q); *w && w - reinterpret_cast<const char16_t*>(q) < 200; ++w)
                    std::fputc(*w < 0x80 ? int(*w) : '?', stderr);
            else
                for (int i = 0; q[i] && i < 200; ++i) std::fputc(q[i], stderr);
            std::fputs("\"\n", stderr);
            p = (*end == ',') ? end + 1 : end;
            if (end == p && *p) break;
        }
    }
    if (const char* chain = std::getenv("BB_CHAIN")) {  // "hexaddr:hexoffset:count": follow a linked list
        unsigned long long a = 0, off = 0, n = 0;
        if (std::sscanf(chain, "%llx:%llx:%llu", &a, &off, &n) == 3) {
            unsigned long long p = *reinterpret_cast<const unsigned long long*>(a);
            for (unsigned long long i = 0; i < n && p; ++i) {
                std::fprintf(stderr, "  chain[%llu] = %llx\n", i, p);
                p = *reinterpret_cast<const unsigned long long*>(p + off);
            }
        }
    }
}

void report(const char* what, uint64_t fault_addr) {
    std::fprintf(stderr, "\nbbrt: fatal %s, access address 0x%llx\n", what, (unsigned long long)fault_addr);
    print_thread("(faulting)", rt::current_context, rt::trace_ring, rt::last_import);
    for (const auto& t : rt::debug_threads_snapshot())  // racy snapshot of the other threads
        if (*t.context != rt::current_context) print_thread(t.name, *t.context, *t.ring, *t.last_import);
    dump_env();
}

#ifdef _WIN32
LONG WINAPI filter(EXCEPTION_POINTERS* e) {
    const auto& r = *e->ExceptionRecord;
    char what[48];
    std::snprintf(what, sizeof what, "exception 0x%08lx", r.ExceptionCode);
    report(what, r.NumberParameters >= 2 ? r.ExceptionInformation[1] : 0);
    {
        HMODULE mod = nullptr;
        const bool found = GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                              reinterpret_cast<LPCSTR>(e->ContextRecord->Rip), &mod);
        char modname[260] = "?";
        if (found && mod) GetModuleFileNameA(mod, modname, sizeof modname);
        std::fprintf(stderr, "  host rip = %s+0x%llx (%s; exe: resolve against bbgame.map)\n",
                     found && mod == GetModuleHandleA(nullptr) ? "exe" : "other-module",
                     (unsigned long long)(e->ContextRecord->Rip - reinterpret_cast<uintptr_t>(mod)), modname);
        // host call chain through a driver: stack words that point into the exe (return addresses; some are stale), resolve against bbgame.map
        const auto exe = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(exe + reinterpret_cast<const IMAGE_DOS_HEADER*>(exe)->e_lfanew);
        const uint64_t* sp = reinterpret_cast<const uint64_t*>(e->ContextRecord->Rsp);
        int shown = 0;
        for (int i = 0; i < 2048 && shown < 10 && !IsBadReadPtr(sp + i, 8); ++i)
            if (sp[i] > exe && sp[i] < exe + nt->OptionalHeader.SizeOfImage) {
                std::fprintf(stderr, "  host stack[%d] = exe+0x%llx\n", i, (unsigned long long)(sp[i] - exe));
                ++shown;
            }
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

// BB_HWWATCH=<hex guest address>[,<address>...] (up to 4): hardware write watchpoints (DR0-DR3, 8 bytes, 8-aligned) on every thread of
// the process. Reports writes whose value is not a plausible pointer/zero (unaligned or < 0x10000), i.e. the corruption of a free-list
// link; see ROADMAP. BB_HWWATCH_ALL=1: every write (first 60), e.g. to find who fills a texture. The addresses can also be re-armed at
// run time (hw_watch_set, used by BB_HWWATCH_VS in the GPU backend for per-frame pool addresses that are only known at draw time).
uint64_t g_hw_addr[4] = {};
std::atomic<int> g_hw_n{0};
std::atomic<uint32_t> g_hw_gen{0};
bool g_hw_started = false;
LONG WINAPI hw_handler(EXCEPTION_POINTERS* e) {
    const uint64_t hit = e->ContextRecord->Dr6 & 0xF;
    if (e->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP || !hit) return EXCEPTION_CONTINUE_SEARCH;
    e->ContextRecord->Dr6 = 0;
    static const bool all = std::getenv("BB_HWWATCH_ALL") != nullptr;
    const uint64_t a = g_hw_addr[std::countr_zero(hit)];
    const uint64_t v = *reinterpret_cast<const uint64_t*>(a);
    static std::atomic<int> shown{0};
    if ((all || (v & 7) || (v && v < 0x10000)) && shown++ < (all ? 150 : 6)) {
        std::fprintf(stderr, "\nhwwatch: write of 0x%llx to 0x%llx, host rip = exe+0x%llx\n", (unsigned long long)v, (unsigned long long)a,
                     (unsigned long long)(e->ContextRecord->Rip - reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr))));
        print_thread("(hwwatch)", rt::current_context, rt::trace_ring, rt::last_import);
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}
void hw_watch_set(const uint64_t* addrs, int n) {
    static std::mutex m;
    std::lock_guard lk(m);
    n = std::min(n, 4);
    for (int i = 0; i < n; ++i) g_hw_addr[i] = addrs[i] & ~7ull;
    g_hw_n = n;
    ++g_hw_gen;
    if (std::exchange(g_hw_started, true)) return;
    AddVectoredExceptionHandler(1, hw_handler);
    std::thread([] {
        std::vector<std::pair<DWORD, uint32_t>> done;  // thread id, generation applied
        for (;;) {
            const uint32_t gen = g_hw_gen;
            const int n = g_hw_n;
            HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            THREADENTRY32 te = {sizeof te};
            if (snap != INVALID_HANDLE_VALUE && Thread32First(snap, &te)) {
                do {
                    if (te.th32OwnerProcessID != GetCurrentProcessId() || te.th32ThreadID == GetCurrentThreadId()) continue;
                    auto it = std::find_if(done.begin(), done.end(), [&](auto& d) { return d.first == te.th32ThreadID; });
                    if (it != done.end() && it->second == gen) continue;
                    HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE, te.th32ThreadID);
                    if (!h) continue;
                    if (SuspendThread(h) != DWORD(-1)) {
                        CONTEXT ctx = {};
                        ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
                        if (GetThreadContext(h, &ctx)) {
                            DWORD64* dr[4] = {&ctx.Dr0, &ctx.Dr1, &ctx.Dr2, &ctx.Dr3};
                            ctx.Dr7 = 0;
                            for (int i = 0; i < n; ++i) *dr[i] = g_hw_addr[i], ctx.Dr7 |= (1ull << (2 * i)) | (1ull << (16 + 4 * i)) | (3ull << (18 + 4 * i));  // Li, write, 8 bytes
                            SetThreadContext(h, &ctx);
                            if (it != done.end()) it->second = gen; else done.emplace_back(te.th32ThreadID, gen);
                        }
                        ResumeThread(h);
                    }
                    CloseHandle(h);
                } while (Thread32Next(snap, &te));
            }
            if (snap != INVALID_HANDLE_VALUE) CloseHandle(snap);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }).detach();
}
void start_hw_watch(const char* list) {
    uint64_t a[4];
    int n = 0;
    for (const char* p = list; p && *p && n < 4; p = std::strchr(p, ',') ? std::strchr(p, ',') + 1 : nullptr) a[n++] = std::strtoull(p, nullptr, 16);
    hw_watch_set(a, n);
}
#else
void handler(int sig, siginfo_t* si, void*) {
    report(sig == SIGSEGV ? "SIGSEGV" : "signal", reinterpret_cast<uintptr_t>(si->si_addr));
    std::signal(sig, SIG_DFL);
}
#endif

} // namespace

void install_crash_handler() {
#ifdef _WIN32
    SetUnhandledExceptionFilter(filter);
    if (const char* hw = std::getenv("BB_HWWATCH")) start_hw_watch(hw);
    gpu::hooks().hw_watch = hw_watch_set;
#else
    struct sigaction sa = {};
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
#endif
}

void start_profiler(unsigned seconds) {
    std::thread([seconds] {
        std::this_thread::sleep_for(std::chrono::seconds(2 + (std::getenv("BB_PROFILE_FROM") ? std::atoi(std::getenv("BB_PROFILE_FROM")) : 0)));  // let the threads register
        std::map<std::tuple<std::string, uint64_t, uint64_t>, unsigned> hist;
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
        while (std::chrono::steady_clock::now() < end) {
            for (const auto& t : rt::debug_threads_snapshot())
                if (t.name)
                    { uint64_t ret = 0; if (uint64_t rip = sample_rip_abs(t.os_thread_id, &ret)) ++hist[{std::string(t.name) + " #" + std::to_string(t.os_thread_id), rip, ret}]; }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        for (auto& [key, n] : hist) {
#ifdef _WIN32
            const auto exe = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
            HMODULE mod = nullptr;
            GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(std::get<1>(key)), &mod);
            if (reinterpret_cast<uint64_t>(mod) != exe) {
                // caller = word at [rsp] when the sample hit a leaf routine of a DLL (memmove etc.): exe RVA to resolve against the link map
                const uint64_t ret = std::get<2>(key);
                std::fprintf(stderr, "PROFMOD %s %u [%s] caller=%llx\n", describe_rip(std::get<1>(key)).c_str(), n, std::get<0>(key).c_str(),
                             (unsigned long long)(ret > exe ? ret - exe : 0));
                continue;
            }
            std::fprintf(stderr, "PROFILE %llx %u [%s]\n", (unsigned long long)(std::get<1>(key) - exe), n, std::get<0>(key).c_str());
#endif
        }
        std::fflush(stderr);
        std::_Exit(4);
    }).detach();
}

void start_hang_watchdog(unsigned seconds) {
    std::thread([seconds] {
        std::this_thread::sleep_for(std::chrono::seconds(seconds));
        std::fprintf(stderr, "\nbbrt: no exit after %u s, dumping guest threads (racy snapshot)\n", seconds);
        print_cond_waiters();
        for (const auto& t : rt::debug_threads_snapshot()) {
            print_thread(t.name, *t.context, *t.ring, *t.last_import);
            if (uint64_t rva = sample_rip(t.os_thread_id))
                std::fprintf(stderr, "  host rip = exe+0x%llx (see bbgame.map)\n", (unsigned long long)rva);
        }
        print_import_counts();
        print_gnm_stats();
        dump_env();
        std::fflush(stderr);
        std::_Exit(3);
    }).detach();
}

} // namespace bb::hle
