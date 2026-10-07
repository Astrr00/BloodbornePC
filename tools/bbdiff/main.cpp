// SPDX-License-Identifier: GPL-3.0-or-later
// bbdiff: differential test of recompiled leaf functions against the same guest code executed natively.
// Local developer tool: needs the user's own eboot (CMake -DBB_EBOOT=<path>); nothing game-derived is committed.
//
// Guest memory is identity-mapped (Context::base = 0) at the eboot's load address, so native and recompiled runs
// see the same addresses: pointers in registers, RIP-relative data and relocated pointers are valid in both.
// Each sampled function runs with randomised registers whose pointer-shaped values point into a scratch region
// filled with a mix of pointers and integers. Native faults (bad pointer chases) are skipped, not failures.
#define NOMINMAX
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/orbis_elf.h"
#include "core/util.h"
#include "runtime/guest.h"
#include "runtime/loader.h"

struct NativeState {  // layout shared with native_call_win64.asm
    uint64_t r[16];
    uint64_t rflags;
    uint8_t xmm[16][16];
    uint64_t host_rsp;
    uint64_t fn;
};
static_assert(offsetof(NativeState, rflags) == 128 && offsetof(NativeState, xmm) == 136 &&
              offsetof(NativeState, host_rsp) == 392 && offsetof(NativeState, fn) == 400);

extern "C" int bb_native_call(NativeState* s);  // 1 = returned normally
extern "C" char bb_native_after_call[];         // return site inside bb_native_call
extern "C" void bb_native_recover();             // resume point after a native fault (returns 0 to the caller)

namespace {

// Guest code has no unwind data, so SEH cannot unwind from a native guest fault to a __try in our frame. A vectored
// handler redirects the faulting thread to bb_native_recover, which restores the host frame saved by the trampoline.
volatile bool g_in_native = false;
volatile ULONGLONG g_phase_start = 0;  // tick count when the current native/recompiled run started, 0 = idle
volatile uint64_t g_current_fn = 0;
volatile uint64_t g_fault_addr = 0, g_fault_ip = 0;  // last fault of a recompiled run

LONG record_fault(EXCEPTION_POINTERS* ep) {
    g_fault_ip = reinterpret_cast<uint64_t>(ep->ExceptionRecord->ExceptionAddress);
    g_fault_addr = ep->ExceptionRecord->NumberParameters > 1 ? ep->ExceptionRecord->ExceptionInformation[1] : 0;
    return EXCEPTION_EXECUTE_HANDLER;
}

LONG CALLBACK native_fault_handler(EXCEPTION_POINTERS* ep) {
    if (!g_in_native) return EXCEPTION_CONTINUE_SEARCH;
    g_in_native = false;
    ep->ContextRecord->Rip = reinterpret_cast<DWORD64>(&bb_native_recover);
    return EXCEPTION_CONTINUE_EXECUTION;
}

bool run_native(NativeState* s) {
    g_phase_start = GetTickCount64();
    g_in_native = true;
    int ok = bb_native_call(s);
    g_in_native = false;
    g_phase_start = 0;
    return ok != 0;
}

// Watchdog: a native run that does not return within kHangMs is redirected to bb_native_recover (skipped like a
// fault). A recompiled run that hangs although the native one terminated is a translation bug: report and exit.
constexpr ULONGLONG kHangMs = 2000;
DWORD WINAPI watchdog(LPVOID main_thread) {
    for (;;) {
        Sleep(250);
        const ULONGLONG start = g_phase_start;
        if (!start || GetTickCount64() - start < kHangMs) continue;
        HANDLE t = static_cast<HANDLE>(main_thread);
        if (g_in_native) {
            SuspendThread(t);
            CONTEXT ctx{};
            ctx.ContextFlags = CONTEXT_CONTROL;
            if (g_in_native && GetThreadContext(t, &ctx)) {
                g_in_native = false;
                g_phase_start = 0;
                ctx.Rip = reinterpret_cast<DWORD64>(&bb_native_recover);
                SetThreadContext(t, &ctx);
            }
            ResumeThread(t);
        } else {
            std::printf("FAIL: recompiled 0x%llx hangs although the native run terminated\n",
                        (unsigned long long)g_current_fn);
            std::fflush(stdout);
            ExitProcess(1);
        }
    }
}

bool run_recompiled(bb::rt::GuestFn fn, bb::rt::Context* c) {
    g_phase_start = GetTickCount64();
    bool ok = true;
    __try {
        fn(*c);
    } __except (record_fault(GetExceptionInformation())) {
        ok = false;
    }
    g_phase_start = 0;
    return ok;
}

struct Region {
    uint64_t lo, hi;
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fputs("usage: bbdiff <eboot.bin> [inputs per function=3]\n", stderr);
        return 2;
    }
    const int inputs = argc > 2 ? std::atoi(argv[2]) : 3;
    auto file = bb::read_file(argv[1]);
    if (!file) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    auto parsed = bb::elf::parse(*file);
    if (auto* e = std::get_if<bb::elf::Error>(&parsed)) {
        std::fprintf(stderr, "error: %s\n", e->message.c_str());
        return 1;
    }
    const auto& img = std::get<bb::elf::Image>(parsed);
    const uint64_t bias = bb::rt::kLoadBias;

    // Identity-mapped guest space: [bias, image end) + import cells + 1 MiB scratch (data + stack), executable for
    // the native run.
    const uint64_t image_hi = (bb::rt::import_cells_end(img, bias) + 0xFFFF) & ~0xFFFFull;
    const uint64_t scratch = image_hi, scratch_size = 1 << 20, stack_top = scratch + scratch_size - 0x1000;
    const uint64_t total = scratch + scratch_size - bias;
    void* at = VirtualAlloc(reinterpret_cast<void*>(bias), total, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
    if (at != reinterpret_cast<void*>(bias)) {
        std::fprintf(stderr, "cannot reserve guest memory at 0x%llx (%llu MiB)\n", (unsigned long long)bias,
                     (unsigned long long)(total >> 20));
        return 1;
    }
    bb::rt::Context c;
    c.base = 0;
    AddVectoredExceptionHandler(1, native_fault_handler);
    HANDLE self = nullptr;
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &self, 0, FALSE,
                    DUPLICATE_SAME_ACCESS);
    CreateThread(nullptr, 0, watchdog, self, 0, nullptr);
    std::string error;
    if (!bb::rt::load_image(c, img, bias, scratch + scratch_size, error)) {
        std::fprintf(stderr, "load_image: %s\n", error.c_str());
        return 1;
    }

    // Regions whose contents are restored before and compared after every run: scratch and writable segments.
    std::vector<Region> regions{{scratch, scratch + scratch_size}};
    for (auto& s : img.segments)
        if (s.type == bb::elf::PT_LOAD && (s.flags & 2)) regions.push_back({s.vaddr + bias, s.vaddr + bias + s.memsz});
    auto snapshot = [&] {
        std::vector<std::vector<uint8_t>> v;
        for (auto& r : regions) v.emplace_back(bb::rt::host(c, r.lo), bb::rt::host(c, r.hi));
        return v;
    };
    auto restore = [&](const std::vector<std::vector<uint8_t>>& v) {
        for (size_t i = 0; i < regions.size(); ++i) std::memcpy(bb::rt::host(c, regions[i].lo), v[i].data(), v[i].size());
    };

    uint64_t seed = 0x243F6A8885A308D3ull;
    auto rnd = [&] { return seed = seed * 6364136223846793005ull + 1442695040888963407ull, seed >> 11; };
    // Scratch layout: A = [scratch, scratch+half) holds pointers into B and small integers; B = second half up to the
    // stack is all zero, so pointer chases end at null and loop counters stay small.
    const uint64_t half = scratch + scratch_size / 2;
    auto ptr_a = [&] { return (scratch + 0x100 + rnd() % (scratch_size / 2 - 0x200)) & ~7ull; };
    auto ptr_b = [&] { return (half + rnd() % (scratch_size / 4)) & ~7ull; };
    auto small_int = [&] { return rnd() % 64; };

    size_t tested = 0, skipped = 0, passed = 0, failed = 0, fn_failed = 0;
    std::map<uint64_t, std::string> first_failure;
    for (size_t i = 0; i < bb::rt::kFunctionCount; ++i) {
        const auto& entry = bb::rt::kFunctions[i];
        g_current_fn = entry.addr;
        bool fn_ok = true;
        for (int k = 0; k < inputs; ++k) {
            for (uint64_t a = scratch; a < half; a += 8) bb::rt::st<uint64_t>(c, a, rnd() & 1 ? ptr_b() : small_int());
            std::memset(bb::rt::host(c, half), 0, stack_top - half);
            const auto pristine = snapshot();
            NativeState ns{};
            for (int r = 0; r < 16; ++r) ns.r[r] = rnd() & 1 ? ptr_a() : small_int();
            for (int r : {1, 2, 6, 7, 8, 9}) ns.r[r] = rnd() % 4 ? ptr_a() : small_int();  // args: mostly pointers
            ns.r[4] = stack_top;                               // 16-aligned before the call pushes
            for (auto& x : ns.xmm)
                for (int j = 0; j < 4; ++j) {
                    float f = float(int32_t(rnd())) / 65536.0f;
                    std::memcpy(x + 4 * j, &f, 4);
                }
            ns.fn = entry.addr;
            NativeState in = ns;
            if (!run_native(&ns)) {
                ++skipped;
                restore(pristine);
                continue;
            }
            // The native run left RFLAGS (trampoline pushfq) in the return-address slot. During the run the slot held
            // the real native return address: the recompiled run sees that same value (loops over stack words stop at
            // the same place), and the slot is set to the native final value before the memory comparison.
            const uint64_t final_slot = bb::rt::ld<uint64_t>(c, stack_top - 8);
            const uint64_t real_ret = reinterpret_cast<uint64_t>(&bb_native_after_call);
            const auto want_mem = snapshot();
            restore(pristine);

            for (int r = 0; r < 16; ++r) c.r[r] = in.r[r];
            for (int r = 0; r < 16; ++r) {
                std::memset(c.ymm[r], 0, 32);
                std::memcpy(c.ymm[r], in.xmm[r], 16);
            }
            c.r[4] = stack_top - 8;
            bb::rt::st<uint64_t>(c, c.r[4], real_ret);
            ++tested;
            std::string why;
            if (!run_recompiled(entry.fn, &c)) {
                char b[360];
                std::snprintf(b, sizeof b,
                              "recompiled faulted: access 0x%llx (host ip 0x%llx), input rdi=0x%llx [rdi+0x20]=0x%llx "
                              "[rdi+0x28]=0x%llx; native final rax=0x%llx rcx=0x%llx rdi=0x%llx",
                              (unsigned long long)g_fault_addr, (unsigned long long)g_fault_ip,
                              (unsigned long long)in.r[7], (unsigned long long)bb::rt::ld<uint64_t>(c, in.r[7] + 0x20),
                              (unsigned long long)bb::rt::ld<uint64_t>(c, in.r[7] + 0x28), (unsigned long long)ns.r[0],
                              (unsigned long long)ns.r[1], (unsigned long long)ns.r[7]);
                why = b;
            } else {
                bb::rt::st<uint64_t>(c, stack_top - 8, final_slot);
                for (int r = 0; r < 16 && why.empty(); ++r)
                    if (c.r[r] != ns.r[r]) {
                        char b[96];
                        std::snprintf(b, sizeof b, "r%d native 0x%llx recompiled 0x%llx", r,
                                      (unsigned long long)ns.r[r], (unsigned long long)c.r[r]);
                        why = b;
                    }
                for (int r = 0; r < 16 && why.empty(); ++r)
                    if (std::memcmp(c.ymm[r], ns.xmm[r], 16) != 0) why = "xmm" + std::to_string(r);
                for (size_t g = 0; g < regions.size() && why.empty(); ++g)
                    if (std::memcmp(bb::rt::host(c, regions[g].lo), want_mem[g].data(), want_mem[g].size()) != 0)
                        why = g == 0 ? "scratch/stack memory" : "data segment memory";
            }
            restore(pristine);
            if (why.empty()) {
                ++passed;
            } else {
                ++failed;
                fn_ok = false;
                first_failure.emplace(entry.addr, why);
            }
        }
        if (!fn_ok) ++fn_failed;
    }

    std::printf("functions %zu, runs tested %zu (passed %zu, failed %zu), skipped (native fault) %zu; "
                "functions with a failure: %zu\n",
                bb::rt::kFunctionCount, tested, passed, failed, skipped, fn_failed);
    size_t shown = 0;
    for (auto& [addr, why] : first_failure) {
        if (shown++ == 25) break;
        std::printf("  FAIL 0x%llx (vaddr 0x%llx): %s\n", (unsigned long long)addr, (unsigned long long)(addr - bias),
                    why.c_str());
    }
    return failed ? 1 : 0;
}
