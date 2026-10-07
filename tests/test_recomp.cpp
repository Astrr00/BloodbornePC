// SPDX-License-Identifier: GPL-3.0-or-later
// End-to-end recompiler check: hand-assembled guest functions (tests/make_fixtures.py, RECOMP_TESTS) are translated
// by bbrecomp at build time; this runs the generated C++ and compares against the expected guest results.
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>

#include "core/util.h"
#include "recomp_addrs.h"
#include "runtime/guest.h"
#include "runtime/loader.h"


// 64-byte aligned guest memory: some fixtures use 32-byte aligned non-temporal stores (vmovntps ymm), which fault (#GP) on a heap block
// that is only 16-byte aligned (glibc); on Windows the same bytes happened to be aligned.
template <class T>
struct Aligned64 {
    using value_type = T;
    Aligned64() = default;
    template <class U> Aligned64(const Aligned64<U>&) {}
    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(64))); }
    void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t(64)); }
    template <class U> bool operator==(const Aligned64<U>&) const { return true; }
};
static int failures = 0;
#define CHECK(c)                                                    \
    do {                                                            \
        if (!(c)) {                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                             \
        }                                                           \
    } while (0)

// Host features for the native side of the differential runs (AVX2 integer ops and FMA are not in AVX1/Jaguar).
static bool host_avx2 = false, host_fma = false;
static void detect_cpu() {
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    const int max_leaf = r[0];
    __cpuid(r, 1);
    host_fma = (r[2] >> 12) & 1;
    if (max_leaf >= 7) {
        __cpuidex(r, 7, 0);
        host_avx2 = (r[1] >> 5) & 1;
    }
#else
    __builtin_cpu_init();
    host_avx2 = __builtin_cpu_supports("avx2");
    host_fma = __builtin_cpu_supports("fma");
#endif
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep output if a translation bug crashes the process
    detect_cpu();
    if (argc != 2) {
        std::printf("usage: test_recomp <fixture eboot.elf>\n");
        return 2;
    }
    auto file = bb::read_file(argv[1]);
    auto parsed = file ? bb::elf::parse(*file) : std::variant<bb::elf::Image, bb::elf::Error>(bb::elf::Error{"read"});
    if (auto* e = std::get_if<bb::elf::Error>(&parsed)) {
        std::printf("FAIL: cannot load %s: %s\n", argv[1], e->message.c_str());
        return 1;
    }
    const auto& img = std::get<bb::elf::Image>(parsed);
    // Guest memory: a heap buffer standing in for the guest address space (Context::base), image at vaddr+kLoadBias,
    // then 1 MiB of stack above it.
    std::vector<uint8_t, Aligned64<uint8_t>> mem(bb::rt::image_end(img) + bb::rt::kLoadBias + (1 << 20));
    bb::rt::Context c;
    c.base = reinterpret_cast<uintptr_t>(mem.data());
    std::string error;
    if (!bb::rt::load_image(c, img, bb::rt::kLoadBias, mem.size(), error)) {
        std::printf("FAIL: load_image: %s\n", error.c_str());
        return 1;
    }
    const uint64_t stack_top = mem.size() - 64;
    // Guest call: SysV args in rdi/rsi, fake return address pushed like a `call` would.
    auto call = [&](uint64_t fn, uint64_t a, uint64_t b = 0) {
        c.r[4] = stack_top - 8;
        bb::rt::st<uint64_t>(c, c.r[4], 0xDEADull);
        c.r[7] = a;
        c.r[6] = b;
        c.r[3] = 0x1111;  // rbx: callee-saved, must survive fact()
        bb::rt::GuestFn f = bb::rt::lookup(fn + bb::rt::kLoadBias);  // RECOMP_* are vaddrs
        if (!f) {
            std::printf("FAIL: no recompiled function at 0x%llx\n", (unsigned long long)fn);
            ++failures;
            return uint64_t(0);
        }
        f(c);
        CHECK(c.r[4] == stack_top);  // ret popped exactly the return address
        return c.r[0];
    };

    // ---- Native ground truth: run the same fixture bytes on the host CPU and diff against the recompiled run. ----
    const bb::elf::Segment* text = nullptr;
    for (auto& s : img.segments)
        if (s.type == bb::elf::PT_LOAD && (s.flags & 1)) text = &s;
    if (!text) {
        std::printf("FAIL: fixture has no executable segment\n");
        return 1;
    }
    constexpr size_t kCodeOffset = 0x100;  // trampoline lives below the copied text
    const size_t page_size = kCodeOffset + text->filesz;
#ifdef _WIN32
    auto* page = static_cast<uint8_t*>(VirtualAlloc(nullptr, page_size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
#else
    auto* page = static_cast<uint8_t*>(
        mmap(nullptr, page_size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (page == MAP_FAILED) page = nullptr;
#endif
    if (!page) {
        std::printf("FAIL: cannot allocate executable memory\n");
        return 1;
    }
    std::memcpy(page + kCodeOffset, img.elf.data() + text->offset, text->filesz);
#ifdef _WIN32
    // Win64 -> SysV: push rdi/rsi/rbx; mov rdi,rcx; mov rsi,rdx; sub rsp,32; call r8; add rsp,32; pop; ret
    static const uint8_t kTrampoline[] = {0x57, 0x56, 0x53, 0x48, 0x89, 0xCF, 0x48, 0x89, 0xD6, 0x48, 0x83, 0xEC, 0x20,
                                          0x41, 0xFF, 0xD0, 0x48, 0x83, 0xC4, 0x20, 0x5B, 0x5E, 0x5F, 0xC3};
    std::memcpy(page, kTrampoline, sizeof kTrampoline);
    using Tramp = uint64_t (*)(uint64_t, uint64_t, const void*);
    auto native = [&](uint64_t fn, uint64_t a, uint64_t b) {
        return reinterpret_cast<Tramp>(page)(a, b, page + kCodeOffset + (fn - text->vaddr));
    };
#else
    using Fn = uint64_t (*)(uint64_t, uint64_t);  // host is SysV already
    auto native = [&](uint64_t fn, uint64_t a, uint64_t b) {
        return reinterpret_cast<Fn>(page + kCodeOffset + (fn - text->vaddr))(a, b);
    };
#endif
    // Pointer arguments are guest addresses for the recompiled run and the same bytes' host address natively.
    size_t diffs = 0;
    auto diff = [&](const char* name, uint64_t fn, uint64_t a, uint64_t b, bool ptr, uint64_t mask) {
        char where[64] = "";
        const size_t lo = 0x1000, hi = 0x3000;  // scratch region both runs may touch
        std::vector<uint8_t> before(mem.begin() + lo, mem.begin() + hi);
        uint64_t want = native(fn, ptr ? uint64_t(mem.data() + a) : a, b) & mask;
        std::vector<uint8_t> want_mem(mem.begin() + lo, mem.begin() + hi);
        std::copy(before.begin(), before.end(), mem.begin() + lo);
        uint64_t got = call(fn, a, b) & mask;
        ++diffs;
        const auto mm = std::mismatch(want_mem.begin(), want_mem.end(), mem.begin() + lo);
        if (mm.first != want_mem.end())
            std::snprintf(where, sizeof where, " (memory differs at 0x%zx: native 0x%02x, recompiled 0x%02x)",
                          lo + size_t(mm.first - want_mem.begin()), unsigned(*mm.first), unsigned(*mm.second));
        if (got != want || mm.first != want_mem.end()) {
            std::printf("FAIL native diff %s(0x%llx, 0x%llx): native 0x%llx, recompiled 0x%llx%s\n", name,
                        (unsigned long long)a, (unsigned long long)b, (unsigned long long)want,
                        (unsigned long long)got, where);
            ++failures;
        }
    };
    constexpr uint64_t kAll = ~0ull, kFlags = 0x8C7, kFlagsNoOf = 0x0C7, kNone = 0;
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    auto rnd = [&] { return seed = seed * 6364136223846793005ull + 1442695040888963407ull, uint32_t(seed >> 32); };
    for (int k = 0; k < 300; ++k) {
        uint32_t a = rnd(), b = rnd();
        if (k % 3 == 0) b = a;  // equal operands: ZF paths
        if (k % 5 == 0) a = 0x80000000u, b = k & 8 ? 1 : 0x7FFFFFFFu;  // overflow corners
        diff("flags_add", RECOMP_FLAGS_ADD, a, b, false, kFlags);
        diff("flags_sub", RECOMP_FLAGS_SUB, a, b, false, kFlags);
        diff("flags_inc", RECOMP_FLAGS_INC, a, b, false, kFlags);
        unsigned n = 1 + b % 31;  // count 0 leaves flags untouched (caller state differs); OF undefined if n > 1
        diff("flags_shl", RECOMP_FLAGS_SHL, a, n, false, n == 1 ? kFlags : kFlagsNoOf);
        diff("flags_sar", RECOMP_FLAGS_SAR, a, n, false, n == 1 ? kFlags : kFlagsNoOf);
        diff("max", RECOMP_MAX, a, b, false, kAll);
        diff("sum", RECOMP_SUM, a % 2000, 0, false, kAll);
        diff("fact", RECOMP_FACT, a % 13, 0, false, kAll);
        diff("sel", RECOMP_SEL, a % 5, 0, false, kAll);
        bb::rt::st<uint32_t>(c, 0x2400, b);
        diff("atom_xadd", RECOMP_ATOM_XADD, 0x2400, a, true, kAll);
        bb::rt::st<uint32_t>(c, 0x2410, k & 1 ? 5 : a);
        diff("atom_cas", RECOMP_ATOM_CAS, 0x2410, b, true, kAll);
        diff("mem", RECOMP_MEM, 0x1000, a, true, kAll);
        float fa = float(int32_t(a)) / 7.0f, fb = float(int32_t(b)) / 3.0f;
        std::memcpy(&mem[0x2100], &fa, 4);
        std::memcpy(&mem[0x2104], k % 7 == 0 ? &fa : &fb, 4);
        diff("vec_less", RECOMP_VEC_LESS, 0x2100, 0, true, kAll);
        for (int i = 0; i < 8; ++i) {
            float f = float(int32_t(rnd())) / 65536.0f;
            std::memcpy(&mem[0x2000 + 4 * i], &f, 4);
        }
        diff("vec_math", RECOMP_VEC_MATH, 0x2000, 0, true, kNone);  // result is in memory
        for (int i = 0; i < 4; ++i) bb::rt::st<uint32_t>(c, 0x2200 + 4 * i, rnd());
        diff("vec_shuf", RECOMP_VEC_SHUF, 0x2200, 0, true, kAll);
        // Scalar float compares incl. NaN: all modelled flags must match the host CPU (OF/SF preset to 1).
        float x = k % 4 == 0 ? std::numeric_limits<float>::quiet_NaN() : fa, y = k % 6 == 0 ? fa : fb;
        std::memcpy(&mem[0x2100], &x, 4);
        std::memcpy(&mem[0x2104], &y, 4);
        diff("vec_comiss", RECOMP_VEC_COMISS, 0x2100, 0, true, kFlags);
        diff("vec_ucomiss", RECOMP_VEC_UCOMISS, 0x2100, 0, true, kFlags);
        // Integer long tail (values; flags only where architecturally defined).
        uint32_t d = b | 1;                                    // non-zero divisor
        if (a == 0x80000000u && d == 0xFFFFFFFFu) d = 3;       // INT_MIN / -1 raises #DE
        diff("int_div", RECOMP_INT_DIV, a, d, false, kAll);
        diff("int_idiv", RECOMP_INT_IDIV, a, d, false, kAll);
        uint64_t wa = (uint64_t(rnd()) << 32) | rnd(), wb = (uint64_t(rnd()) << 32) | rnd();
        diff("int_mul", RECOMP_INT_MUL, wa, wb, false, kAll);
        diff("int_imul1", RECOMP_INT_IMUL1, wa, wb, false, kAll);
        diff("int_adcsbb", RECOMP_INT_ADCSBB, wa, wb, false, kAll);
        diff("int_adcflags", RECOMP_INT_ADCFLAGS, a, b, false, kFlags);
        diff("int_rol", RECOMP_INT_ROL, a, b, false, kAll);
        diff("int_ror1", RECOMP_INT_ROR1, a, 0, false, 0x801);  // ror leaves SF/ZF/PF as before: CF and OF only
        diff("int_bt", RECOMP_INT_BT, a, b, false, kAll);
        diff("int_bextr", RECOMP_INT_BEXTR, a, b & 0x3F3F, false, kAll);
        diff("int_andn", RECOMP_INT_ANDN, a, b, false, kAll);
        diff("int_lzcnt", RECOMP_INT_LZCNT, k % 9 == 0 ? 0 : a >> (k % 32), 0, false, kAll);
        diff("int_tzcnt", RECOMP_INT_TZCNT, k % 9 == 0 ? 0 : a << (k % 32), 0, false, kAll);
        diff("int_popcnt", RECOMP_INT_POPCNT, a, 0, false, kAll);
        bb::rt::st<uint32_t>(c, 0x2500, a);
        diff("int_movbe", RECOMP_INT_MOVBE, 0x2500, 0, true, kAll);
        // ---- Long tail: string ops, scalar odds and ends, AVX/SSE/MMX/x87 (see make_fixtures.py) ----
        const uint64_t base = 0x2000;
        auto fill = [&](size_t off, size_t n) {
            for (size_t i = 0; i < n; ++i) mem[off + i] = uint8_t(rnd() >> 7);
        };
        auto putd = [&](size_t off, double v) { std::memcpy(&mem[base + off], &v, 8); };
        constexpr uint64_t kFlagsHi = 0xFFFFFFFF00000000ull | kFlags;  // flags in the low dword, state above
        constexpr uint64_t kFlags4 = kFlags * 0x1000100010001ull;      // RFLAGS image at bit 0/16/32/48
        fill(base, 0x800);
        diff("str_movs", RECOMP_STR_MOVS, base, 0, true, kAll);
        fill(base, 0x800);
        diff("str_stos", RECOMP_STR_STOS, base, wb, true, kAll);
        fill(base, 0x100);
        diff("str_lods", RECOMP_STR_LODS, base, 0, true, kAll);
        fill(base, 0x200);
        std::memcpy(&mem[base + 0x100], &mem[base], k % 40);  // equal prefix of 0..39 bytes
        diff("str_cmps_e", RECOMP_STR_CMPS_E, base, 0, true, kFlagsHi);
        fill(base, 0x200);
        std::memcpy(&mem[base + 0x100 + 2 * (k % 20)], &mem[base + 2 * (k % 20)], 2);  // equal word at 0..19
        diff("str_cmps_ne", RECOMP_STR_CMPS_NE, base, 0, true, kFlagsHi);
        fill(base, 0x200);
        std::memcpy(&mem[base + 0x100], &mem[base], k % 3 == 0 ? 15 : k % 3 == 1 ? 1 : 0);
        diff("str_cmps_one", RECOMP_STR_CMPS_ONE, base, 0, true, kFlags4);
        fill(base, 0x200);
        std::memcpy(&mem[base + 0x100 + 4 * (k % 10)], &mem[base + 4 * (k % 10)], 4);
        for (int j = 0; j < k % 7; ++j) std::memcpy(&mem[base + 0x80 - 8 * j], &mem[base + 0x180 - 8 * j], 8);
        diff("str_cmps_d", RECOMP_STR_CMPS_D, base, 0, true, 0xFFFFFF0000000000ull | kFlags | (kFlags << 16));
        fill(base, 0x200);
        mem[base + k % 80] = uint8_t(b);
        diff("str_scas_ne", RECOMP_STR_SCAS_NE, base, b & 0xFF, true, kFlagsHi);
        fill(base, 0x200);
        for (int j = 0; j < k % 40; ++j) std::memcpy(&mem[base + 2 * j], &b, 2);  // first words equal ax
        diff("str_scas_e", RECOMP_STR_SCAS_E, base, b & 0xFFFF, true, kFlagsHi);
        fill(base, 0x200);
        for (int j = 0; j < 4; ++j)
            if (k >> j & 1) std::memcpy(&mem[base + (j == 0 ? 0 : j == 1 ? 1 : j == 2 ? 3 : 7)], &wb, size_t(1) << j);
        diff("str_scas_one", RECOMP_STR_SCAS_ONE, base, wb, true, kFlags4);
        diff("int_bswap", RECOMP_INT_BSWAP, wa, b, false, kAll);
        diff("int_cstc", RECOMP_INT_CSTC, a, b, false, kFlags * 0x100010001ull);
        diff("int_cwd", RECOMP_INT_CWD, a, b, false, kAll);
        diff("int_lahf", RECOMP_INT_LAHF, a, b, false, 0xEF);  // AF (bit 4 of ah) is not modelled
        diff("int_sahf", RECOMP_INT_SAHF, a, b, false, kFlags);
        fill(base, 0x100);
        diff("int_xlat", RECOMP_INT_XLAT, base, b & 0xFF, true, kAll);
        diff("int_bsf", RECOMP_INT_BSF, k % 9 == 0 ? 0 : a >> (k % 32), 0, false, kAll);
        diff("int_bsr", RECOMP_INT_BSR, k % 9 == 0 ? 0 : a >> (k % 32), 0, false, kAll);
        diff("int_bs64", RECOMP_INT_BS64, wa | 1, wb | 1, false, kAll);
        constexpr uint64_t kBls = 0xFFFFFFFFull | (0x8C3ull << 32);  // PF/AF undefined
        diff("int_blsi", RECOMP_INT_BLSI, k % 9 == 0 ? 0 : a, 0, false, kBls);
        diff("int_blsr", RECOMP_INT_BLSR, k % 9 == 0 ? 0 : a, 0, false, kBls);
        diff("int_blsmsk", RECOMP_INT_BLSMSK, k % 9 == 0 ? 0 : a, 0, false, kBls);
        diff("int_bls64", RECOMP_INT_BLS64, k % 9 == 0 ? 0 : wa, 0, false, kAll);
        // rotate through carry: OF is defined for a masked count of 1 only
        const uint64_t kRot = 0xFFFFFFFFull | ((b & 31) == 1 ? kFlags : kFlagsNoOf) << 32;
        diff("int_rcl8", RECOMP_INT_RCL8, a, b, false, kRot);
        diff("int_rcr16", RECOMP_INT_RCR16, a, b, false, kRot);
        diff("int_rcl32", RECOMP_INT_RCL32, a, b, false, kRot);
        diff("int_rcr1", RECOMP_INT_RCR1, a, b, false, 0xFFFFFFFFull | kFlags << 32);
        diff("int_rcl64", RECOMP_INT_RCL64, wa, wb, false, kAll);
        const uint64_t sn = 1 + b % 31;  // shld/shrd: count 0 would expose undefined SF/ZF/PF left by imul
        const uint64_t kDbl = 0xFFFFFFFFull | (sn == 1 ? kFlags : kFlagsNoOf) << 32;
        diff("int_shld32", RECOMP_INT_SHLD32, a, sn, false, kDbl);
        diff("int_shrd32", RECOMP_INT_SHRD32, a, sn, false, kDbl);
        diff("int_shld64", RECOMP_INT_SHLD64, wa, 1 + wb % 63, false, kAll);
        diff("int_loop", RECOMP_INT_LOOP, 1 + a % 200, 0, false, kAll);
        diff("int_loope", RECOMP_INT_LOOPE, 1 + a % 50, b, false, kAll);
        diff("int_jrcxz", RECOMP_INT_JRCXZ, k % 3 == 0 ? 0 : k % 3 == 1 ? 0x100000000ull : (wa | 1), 0, false, kAll);
        diff("int_leave", RECOMP_INT_LEAVE, wa, 0, false, kAll);
        diff("int_enter", RECOMP_INT_ENTER, 0, 0, false, kAll);
        diff("int_fence", RECOMP_INT_FENCE, a, 0, false, kAll);
        fill(base, 0x100);
        diff("v_ps", RECOMP_V_PS, base, 0, true, kAll);
        fill(base, 0x100);
        diff("v_pd", RECOMP_V_PD, base, 0, true, kAll);
        fill(base, 0x100);
        diff("v_scalar", RECOMP_V_SCALAR, base, 0, true, kAll);
        fill(base, 0x100);
        if (host_fma) diff("v_fma", RECOMP_V_FMA, base, 0, true, kNone);
        fill(base, 0x100);
        diff("v_int_a", RECOMP_V_INT_A, base, 0, true, kNone);
        fill(base, 0x100);
        diff("v_int_b", RECOMP_V_INT_B, base, 0, true, kNone);
        fill(base, 0x100);
        if (host_avx2) diff("v_avx2", RECOMP_V_AVX2, base, 0, true, kNone);
        fill(base, 0x100);
        diff("v_misc", RECOMP_V_MISC, base, 0, true, kNone);
        fill(base, 0x100);
        diff("v_zeroupper", RECOMP_V_ZEROUPPER, base, 0, true, kNone);
        fill(base, 0x100);
        diff("mmx_ops", RECOMP_MMX_OPS, base, b, true, kAll);
        // x87 (control word is forced to 53-bit precision inside the fixtures so double math is exact)
        const double xa = k % 3 == 0   ? double(int16_t(rnd())) / 8.0
                          : k % 7 == 0 ? double(int(rnd() % 2000) - 1000) + 0.5  // rounding ties
                                       : double(int32_t(rnd())) / 1024.0;
        const double xb = k % 4 == 0 ? xa : k % 11 == 0 ? std::numeric_limits<double>::quiet_NaN() : double(int32_t(rnd() | 1)) / 512.0;
        const float xc = k % 6 == 0 ? float(xa) : float(int32_t(rnd())) / 64.0f;
        fill(base, 0x100);
        putd(0, xa);
        putd(8, xb);
        std::memcpy(&mem[base + 16], &xc, 4);
        const int16_t xi16 = int16_t(rnd() | 1);
        const int32_t xi32 = int32_t(rnd() | 1);
        const int64_t xi64 = int64_t(((uint64_t(rnd()) << 32 | rnd()) >> 12)) - (int64_t(1) << 51);  // exact as double
        std::memcpy(&mem[base + 24], &xi16, 2);
        std::memcpy(&mem[base + 28], &xi32, 4);
        std::memcpy(&mem[base + 32], &xi64, 8);
        putd(40, double(int(rnd() % 121) - 60) + 0.25);
        if (k % 6 == 0) putd(0, double(float(xa)));
        diff("x87_arith", RECOMP_X87_ARITH, base, 0, true, kNone);
        diff("x87_cmp", RECOMP_X87_CMP, base, 0, true, kAll);
        putd(16, k % 5 == 0 ? xa : double(int32_t(rnd())) / 256.0);
        diff("x87_comi", RECOMP_X87_COMI, base, 0, true, kAll);
        std::memcpy(&mem[base + 16], &xc, 4);
        diff("x87_int", RECOMP_X87_INT, base, rnd() % 4, true, kNone);
    }
    std::printf("native differential runs: %zu\n", diffs);

    // Imports: HLE registered for the NID of symbol 2, reached via PLT stub and via `call [GOT]`; GOT slots hold
    // synthetic import addresses written by the loader.
    bb::rt::register_hle("1jfXLRVzisc", [](bb::rt::Context& cx) { cx.r[0] = cx.r[7] * 2 + 1; });
    CHECK(bb::rt::ld<uint64_t>(c, 0x402010 + bb::rt::kLoadBias) == bb::rt::kImportBase + 2 * bb::rt::kImportStride);
    CHECK(call(RECOMP_IMP_PLT, 20) == 41);
    CHECK(call(RECOMP_IMP_GOT, 5) == 11);

    CHECK(uint32_t(call(RECOMP_SUM, 10)) == 55);
    CHECK(uint32_t(call(RECOMP_SUM, 0)) == 0);
    CHECK(uint32_t(call(RECOMP_FACT, 5)) == 120);
    CHECK(uint32_t(call(RECOMP_FACT, 10)) == 3628800);
    CHECK(c.r[3] == 0x1111);
    CHECK(uint32_t(call(RECOMP_SEL, 0)) == 0);
    CHECK(uint32_t(call(RECOMP_SEL, 1)) == 10);
    CHECK(uint32_t(call(RECOMP_SEL, 2)) == 20);
    CHECK(uint32_t(call(RECOMP_SEL, 3)) == 0xFFFFFFFFu);
    CHECK(uint32_t(call(RECOMP_MEM, 0x1000, 0x01020304)) == 0x01020310u);
    CHECK(bb::rt::ld<uint32_t>(c, 0x1000) == 0x01020304u && mem[0x1004] == 0x08);
    CHECK(c.r[0] >> 32 == 0);  // 32-bit writes zero-extend
    CHECK(uint32_t(call(RECOMP_MAX, 3, 7)) == 7 + 256);
    CHECK(uint32_t(call(RECOMP_MAX, 9, 2)) == 9);
    CHECK(uint32_t(call(RECOMP_MAX, uint32_t(-1), 1)) == 1 + 256);  // signed compare

    // RFLAGS after the operation, masked to CF|1|PF|ZF|SF|OF (AF is not modelled).
    auto flags = [&](uint64_t fn, uint32_t a, uint32_t b) { return call(fn, a, b) & 0x8C7; };
    CHECK(flags(RECOMP_FLAGS_ADD, 0xFFFFFFFFu, 1) == 0x047);           // 0: CF ZF PF
    CHECK(flags(RECOMP_FLAGS_ADD, 0x7FFFFFFFu, 1) == 0x886);           // 0x80000000: SF OF PF
    CHECK(flags(RECOMP_FLAGS_SUB, 1, 2) == 0x087);                     // 0xFFFFFFFF: CF SF PF
    CHECK(flags(RECOMP_FLAGS_SUB, 0x80000000u, 1) == 0x806);           // 0x7FFFFFFF: OF PF
    CHECK(flags(RECOMP_FLAGS_SHL, 0x80000001u, 1) == 0x803);           // 2: CF (msb out), OF = msb(r) != CF
    CHECK(flags(RECOMP_FLAGS_SAR, 0xFFFFFFF0u, 4) == 0x086);           // -1: SF PF, CF = bit 3 = 0
    CHECK(flags(RECOMP_FLAGS_INC, 0xFFFFFFFFu, 1) == 0x003);           // add sets CF; inc 0 -> 1 keeps it

    // AVX: packed math through memory, scalar compare into flags, dword shuffle + vmovd.
    float in[8] = {1, 2, 3, 4, 10, 20, 30, 40};
    std::memcpy(&mem[0x2000], in, sizeof in);
    call(RECOMP_VEC_MATH, 0x2000);
    float outv[4];
    std::memcpy(outv, &mem[0x2020], sizeof outv);
    CHECK(outv[0] == 121.0f && outv[1] == 484.0f && outv[2] == 1089.0f && outv[3] == 1936.0f);
    auto less = [&](float a, float b) {
        std::memcpy(&mem[0x2100], &a, 4);
        std::memcpy(&mem[0x2104], &b, 4);
        return uint32_t(call(RECOMP_VEC_LESS, 0x2100));
    };
    CHECK(less(1.5f, 2.5f) == 1);
    CHECK(less(3.0f, 2.0f) == 0);
    CHECK(less(2.0f, 2.0f) == 0);
    CHECK(less(std::numeric_limits<float>::quiet_NaN(), 1.0f) == 1);  // unordered sets CF
    uint32_t lanes[4] = {0x11, 0x22, 0x33, 0x44};
    std::memcpy(&mem[0x2200], lanes, sizeof lanes);
    CHECK(uint32_t(call(RECOMP_VEC_SHUF, 0x2200)) == 0x44);
    uint32_t rev[4];
    std::memcpy(rev, &mem[0x2210], sizeof rev);
    CHECK(rev[0] == 0x44 && rev[1] == 0x33 && rev[2] == 0x22 && rev[3] == 0x11);

    // Atomics.
    bb::rt::st<uint32_t>(c, 0x2300, 40);
    CHECK(uint32_t(call(RECOMP_ATOM_XADD, 0x2300, 2)) == 40 && bb::rt::ld<uint32_t>(c, 0x2300) == 42);
    bb::rt::st<uint32_t>(c, 0x2310, 5);
    CHECK(uint32_t(call(RECOMP_ATOM_CAS, 0x2310, 9)) == 0x10005 && bb::rt::ld<uint32_t>(c, 0x2310) == 9);
    bb::rt::st<uint32_t>(c, 0x2310, 7);
    CHECK(uint32_t(call(RECOMP_ATOM_CAS, 0x2310, 9)) == 7 && bb::rt::ld<uint32_t>(c, 0x2310) == 7);
    // Long-tail cases that cannot be diffed natively (fixed guest identity, constant selectors/MXCSR, libm vs x87
    // microcode, instruction-pointer fields in fnstenv).
    CHECK(uint32_t(call(RECOMP_INT_CPUID, 0)) == 0x444D4163u);  // "cAMD" (ecx of leaf 0)
    CHECK(uint32_t(call(RECOMP_INT_CPUID, 1)) == 0x3ED8220Bu);
    CHECK(call(RECOMP_INT_RDTSCP, 0) == 0);
    CHECK(uint32_t(call(RECOMP_INT_SREG, 0)) == (0x43u << 16 | 0x3Bu));
    call(RECOMP_V_MXCSR, 0x2000);
    CHECK(bb::rt::ld<uint32_t>(c, 0x2000) == 0x1F80 && bb::rt::ld<uint32_t>(c, 0x2004) == 0x1F80);
    for (double v : {0.0, 0.5, 1.0, -2.5, 7.25, -9.99}) {
        std::memcpy(&mem[0x2000], &v, 8);
        call(RECOMP_X87_TRIG, 0x2000);
        double r[4];
        std::memcpy(r, &mem[0x2100], sizeof r);  // sin, cos, then fsincos' cos, sin
        CHECK(std::fabs(r[0] - std::sin(v)) < 1e-12 && std::fabs(r[1] - std::cos(v)) < 1e-12);
        CHECK(std::fabs(r[2] - std::cos(v)) < 1e-12 && std::fabs(r[3] - std::sin(v)) < 1e-12);
    }
    call(RECOMP_X87_ENV, 0x2000);
    CHECK(bb::rt::ld<uint16_t>(c, 0x2000) == 0x37F && (bb::rt::ld<uint16_t>(c, 0x2004) & 0x3800) == 0 &&
          bb::rt::ld<uint16_t>(c, 0x2008) == 0xFFFF);

    std::printf(failures ? "%d failure(s)\n" : "all recompiler checks passed\n", failures);
    return failures ? 1 : 0;
}
