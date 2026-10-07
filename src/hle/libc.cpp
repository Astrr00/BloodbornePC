// SPDX-License-Identifier: GPL-3.0-or-later
// libc.prx slice: string/memory helpers, allocator, printf family, C++ runtime guards.
// Allocator results are host pointers, which are valid guest addresses because guest memory is identity-mapped.
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hle/hle.h"

#include "hle/guest_heap.h"

namespace bb::hle {
namespace {

// ---- varargs -----------------------------------------------------------------------------------------------------
// One reader for both `printf(fmt, ...)` (registers spilled into a synthetic register save area) and
// `vprintf(fmt, va_list)` (guest va_list in memory).
struct VaReader {
    Context& c;
    GuestVaList va;
    uint64_t next_int() {
        if (va.gp_offset < 48) {
            uint64_t v = rt::ld<uint64_t>(c, va.reg_save_area + va.gp_offset);
            va.gp_offset += 8;
            return v;
        }
        uint64_t v = rt::ld<uint64_t>(c, va.overflow_arg_area);
        va.overflow_arg_area += 8;
        return v;
    }
    double next_double() {
        uint64_t bits;
        if (va.fp_offset < 176) {
            bits = rt::ld<uint64_t>(c, va.reg_save_area + va.fp_offset);
            va.fp_offset += 16;
        } else {
            bits = rt::ld<uint64_t>(c, va.overflow_arg_area);
            va.overflow_arg_area += 8;
        }
        double d;
        std::memcpy(&d, &bits, 8);
        return d;
    }
};

std::string format(VaReader& va, uint64_t fmt) {
    std::string out;
    Context& c = va.c;
    const char* f = ptr<const char>(c, fmt);
    char buf[512];
    while (*f) {
        if (*f != '%') { out += *f++; continue; }
        const char* start = f++;
        if (*f == '%') { out += '%'; ++f; continue; }
        std::string spec = "%";
        while (*f && std::strchr("-+ #0", *f)) spec += *f++;
        auto number = [&] {  // width / precision, possibly '*'
            if (*f == '*') { spec += std::to_string(int(va.next_int())); ++f; return; }
            while (*f >= '0' && *f <= '9') spec += *f++;
        };
        number();
        if (*f == '.') { spec += *f++; number(); }
        std::string len;
        while (*f && std::strchr("hlLzjtq", *f)) len += *f++;
        const char conv = *f ? *f++ : '\0';
        const bool wide = len == "l" || len == "ll" || len == "z" || len == "j" || len == "t" || len == "q";
        switch (conv) {
        case 'd': case 'i': {
            uint64_t v = va.next_int();
            long long s = wide ? (long long)v : (long long)(int32_t)v;
            if (len == "hh") s = (int8_t)v; else if (len == "h") s = (int16_t)v;
            std::snprintf(buf, sizeof buf, (spec + "lld").c_str(), s);
            out += buf;
            break;
        }
        case 'u': case 'x': case 'X': case 'o': {
            uint64_t v = va.next_int();
            unsigned long long u = wide ? v : (uint32_t)v;
            if (len == "hh") u = (uint8_t)v; else if (len == "h") u = (uint16_t)v;
            std::snprintf(buf, sizeof buf, (spec + "ll" + conv).c_str(), u);
            out += buf;
            break;
        }
        case 'c': std::snprintf(buf, sizeof buf, (spec + "c").c_str(), int(va.next_int())); out += buf; break;
        case 'p': std::snprintf(buf, sizeof buf, "0x%llx", (unsigned long long)va.next_int()); out += buf; break;
        case 's': {
            uint64_t a = va.next_int();
            std::vector<char> big(std::strlen(ptr<const char>(c, a ? a : fmt)) + 600);
            std::snprintf(big.data(), big.size(), (spec + "s").c_str(), a ? ptr<const char>(c, a) : "(null)");
            out += big.data();
            break;
        }
        case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': case 'a': case 'A':
            std::snprintf(buf, sizeof buf, (spec + conv).c_str(), va.next_double());
            out += buf;
            break;
        case 'n': va.next_int(); break;  // not supported: consume the argument only
        default: out.append(start, f); break;  // unknown conversion: emit verbatim
        }
    }
    return out;
}

} // namespace

// Variadic call: `fixed` integer arguments precede the va part; the save area mirrors the SysV prologue.
std::string format_variadic(Context& c, int fmt_index) {
    uint8_t save[176];
    static constexpr int kReg[6] = {7, 6, 2, 1, 8, 9};
    for (int i = 0; i < 6; ++i) std::memcpy(save + 8 * i, &c.r[kReg[i]], 8);
    for (int i = 0; i < 8; ++i) std::memcpy(save + 48 + 16 * i, c.ymm[i], 16);
    const uint64_t fmt = arg(c, fmt_index);
    VaReader va{c, {uint32_t(8 * (fmt_index + 1)), 48, c.r[4] + 8, uint64_t(reinterpret_cast<uintptr_t>(save)) - c.base}};
    return format(va, fmt);
}

std::string format_guest(Context& c, uint64_t fmt, uint64_t va_list_addr) {
    VaReader va{c, rt::ld<GuestVaList>(c, va_list_addr)};
    return format(va, fmt);
}

namespace {

void emit(const std::string& s) { std::fwrite(s.data(), 1, s.size(), stdout); }

void cxa_guard_acquire(Context& c) { ret(c, rt::ld<uint8_t>(c, arg(c, 0)) == 0); }
void cxa_guard_release(Context& c) { rt::st<uint8_t>(c, arg(c, 0), 1); }
void cxa_guard_abort(Context&) {}
void cxa_atexit(Context& c) { ret(c, 0); }  // ponytail: destructors never run (process exit = kill); add a list if shutdown matters

// qsort(base, n, size, cmp): the comparator is guest code. Sorting is done on an index permutation so the guest only ever
// sees pointers into the untouched original array; stable (equal elements keep their order), a valid qsort result.
void h_qsort(Context& c) {
    const uint64_t base = arg(c, 0), n = arg(c, 1), size = arg(c, 2), cmp = arg(c, 3);
    if (n < 2 || !size) return;
    const uint64_t saved_rsp = c.r[4];
    std::vector<uint64_t> order(n);
    for (uint64_t i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) {
        c.r[7] = base + a * size;
        c.r[6] = base + b * size;
        c.r[4] = ((saved_rsp - 256) & ~uint64_t(15)) - 8;  // callee entry: rsp = 8 mod 16, return address on top
        rt::st<uint64_t>(c, c.r[4], rt::kExitAddress);
        rt::call_indirect(c, cmp);
        return int32_t(c.r[0]) < 0;
    });
    c.r[4] = saved_rsp;
    std::vector<uint8_t> tmp(n * size);
    for (uint64_t i = 0; i < n; ++i) std::memcpy(&tmp[i * size], ptr(c, base + order[i] * size), size);
    std::memcpy(ptr(c, base), tmp.data(), n * size);
}

void h_memset(Context& c) { std::memset(ptr(c, arg(c, 0)), int(arg(c, 1)), arg(c, 2)); ret(c, arg(c, 0)); }
void h_memcpy(Context& c) { std::memcpy(ptr(c, arg(c, 0)), ptr(c, arg(c, 1)), arg(c, 2)); ret(c, arg(c, 0)); }
void h_memmove(Context& c) { std::memmove(ptr(c, arg(c, 0)), ptr(c, arg(c, 1)), arg(c, 2)); ret(c, arg(c, 0)); }
void h_memcmp(Context& c) { ret(c, uint64_t(int64_t(std::memcmp(ptr(c, arg(c, 0)), ptr(c, arg(c, 1)), arg(c, 2))))); }
void h_strlen(Context& c) { ret(c, std::strlen(ptr<const char>(c, arg(c, 0)))); }
void h_strcmp(Context& c) { ret(c, uint64_t(int64_t(std::strcmp(ptr<const char>(c, arg(c, 0)), ptr<const char>(c, arg(c, 1)))))); }

void* alloc_or_null(size_t n, size_t align = 16) { return guest_alloc(n ? n : 1, align); }
void h_malloc(Context& c) { ret(c, reinterpret_cast<uintptr_t>(alloc_or_null(arg(c, 0))) - c.base); }
void h_calloc(Context& c) {
    const uint64_t n = arg(c, 0) * arg(c, 1);
    void* p = alloc_or_null(n);
    if (p) std::memset(p, 0, n);
    ret(c, reinterpret_cast<uintptr_t>(p) - c.base);
}
void h_free(Context& c) { if (arg(c, 0)) guest_free(ptr(c, arg(c, 0))); }
void h_memalign(Context& c) { ret(c, reinterpret_cast<uintptr_t>(alloc_or_null(arg(c, 1), arg(c, 0))) - c.base); }  // (align, size)
void h_realloc(Context& c) {
    void* old = arg(c, 0) ? ptr(c, arg(c, 0)) : nullptr;
    const uint64_t n = arg(c, 1);
    void* p = alloc_or_null(n);
    if (p && old) {
        const size_t have = guest_usable_size(old);
        std::memcpy(p, old, have < n ? have : n);
    }
    if (old) guest_free(old);
    ret(c, reinterpret_cast<uintptr_t>(p) - c.base);
}

void h_strncpy(Context& c) {
    char* d = ptr<char>(c, arg(c, 0));
    std::strncpy(d, ptr<const char>(c, arg(c, 1)), arg(c, 2));
    ret(c, arg(c, 0));
}
// wchar_t is 16-bit on Orbis (-fshort-wchar): guest wide strings are UTF-16.
void h_wcslen(Context& c) {
    const uint16_t* w = ptr<const uint16_t>(c, arg(c, 0));
    uint64_t n = 0;
    while (w[n]) ++n;
    ret(c, n);
}
void h_wcstol(Context& c) {  // (const wchar_t*, wchar_t** end, base)
    const uint16_t* w = ptr<const uint16_t>(c, arg(c, 0));
    std::string narrow;
    for (const uint16_t* p = w; *p && *p < 0x80; ++p) narrow += char(*p);
    char* end = nullptr;
    const long v = std::strtol(narrow.c_str(), &end, int(arg(c, 2)));
    if (arg(c, 1)) rt::st<uint64_t>(c, arg(c, 1), arg(c, 0) + 2 * uint64_t(end - narrow.c_str()));
    ret(c, uint64_t(int64_t(v)));
}
void h_nop(Context& c) { ret(c, 0); }
// operator new/delete: failure of the throwing forms is fatal (the guest's bad_alloc path would need C++ unwinding).
void h_new(Context& c) {
    void* p = alloc_or_null(arg(c, 0));
    if (!p) { std::fputs("bbrt: operator new failed\n", stderr); std::abort(); }
    ret(c, reinterpret_cast<uintptr_t>(p) - c.base);
}
void h_new_nothrow(Context& c) { ret(c, reinterpret_cast<uintptr_t>(alloc_or_null(arg(c, 0))) - c.base); }
// Writes at most `cap` bytes (including the NUL) of `s` to guest `buf`; returns the untruncated length like snprintf.
uint64_t put_string(Context& c, uint64_t buf, uint64_t cap, const std::string& s) {
    if (cap) {
        const uint64_t n = s.size() < cap - 1 ? s.size() : cap - 1;
        std::memcpy(ptr(c, buf), s.data(), n);
        ptr<char>(c, buf)[n] = 0;
    }
    return s.size();
}
void h_sprintf(Context& c) { ret(c, put_string(c, arg(c, 0), ~0ull, format_variadic(c, 1))); }
void h_snprintf(Context& c) { ret(c, put_string(c, arg(c, 0), arg(c, 1), format_variadic(c, 2))); }
void h_vsprintf(Context& c) { ret(c, put_string(c, arg(c, 0), ~0ull, format_guest(c, arg(c, 1), arg(c, 2)))); }
void h_vsnprintf(Context& c) { ret(c, put_string(c, arg(c, 0), arg(c, 1), format_guest(c, arg(c, 2), arg(c, 3)))); }
void h_assert(Context& c) {  // _Assert(const char* message)
    std::fprintf(stderr, "bbrt: guest assertion failed: %s\n", ptr<const char>(c, arg(c, 0)));
    std::abort();
}
void h_printf(Context& c) { std::string s = format_variadic(c, 0); emit(s); ret(c, s.size()); }
void h_vprintf(Context& c) { std::string s = format_guest(c, arg(c, 0), arg(c, 1)); emit(s); ret(c, s.size()); }

} // namespace

void register_libc() {
    reg("__cxa_guard_acquire", cxa_guard_acquire);
    reg("__cxa_guard_release", cxa_guard_release);
    reg("__cxa_guard_abort", cxa_guard_abort);
    reg("__cxa_atexit", cxa_atexit);
    reg("qsort", h_qsort);
    reg("memset", h_memset);
    reg("memcpy", h_memcpy);
    reg("memmove", h_memmove);
    reg("memcmp", h_memcmp);
    reg("strlen", h_strlen);
    reg("strcmp", h_strcmp);
    reg("malloc", h_malloc);
    reg("calloc", h_calloc);
    reg("free", h_free);
    reg("realloc", h_realloc);
    reg("memalign", h_memalign);
    register_libc_wrapped();
    reg("strncpy", h_strncpy);
    reg("wcslen", h_wcslen);
    reg("wcstol", h_wcstol);
    reg("atexit", h_nop);  // see __cxa_atexit
    reg("_init_env", h_nop);
    for (const char* n : {"_Znwm", "_Znam"}) reg(n, h_new);
    for (const char* n : {"_ZnwmRKSt9nothrow_t", "_ZnamRKSt9nothrow_t"}) reg(n, h_new_nothrow);
    for (const char* n : {"_ZdlPv", "_ZdaPv", "_ZdlPvm", "_ZdaPvm", "_ZdlPvRKSt9nothrow_t", "_ZdaPvRKSt9nothrow_t"}) reg(n, h_free);
    reg("sprintf", h_sprintf);
    reg("snprintf", h_snprintf);
    reg("vsprintf", h_vsprintf);
    reg("vsnprintf", h_vsnprintf);
    reg("_Assert", h_assert);
    reg("printf", h_printf);
    reg("vprintf", h_vprintf);
}

} // namespace bb::hle
