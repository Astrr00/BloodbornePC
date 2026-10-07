// SPDX-License-Identifier: GPL-3.0-or-later
// libc.prx functions that map directly onto host C functions (strings, wide strings, math, conversions).
// wchar_t is 16-bit on Orbis (-fshort-wchar; guest text in memory is UTF-16), so wide-string helpers are written out.
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#ifndef _WIN32
#include <strings.h>
#endif

#include "hle/abi.h"

#ifdef _WIN32
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#endif

namespace bb::hle {
namespace {

using W = uint16_t;

W* w_chr(const W* s, W ch) {
    for (;; ++s) {
        if (*s == ch) return const_cast<W*>(s);
        if (!*s) return nullptr;
    }
}
int w_cmp(const W* a, const W* b, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i)
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}
W* w_cpy(W* d, const W* s, uint64_t n) { std::memmove(d, s, n * sizeof(W)); return d; }
W* w_set(W* d, W v, uint64_t n) { for (uint64_t i = 0; i < n; ++i) d[i] = v; return d; }
int w_scmp(const W* a, const W* b) {
    for (; *a && *a == *b; ++a, ++b) {}
    return *a < *b ? -1 : *a > *b;
}
W* w_scpy(W* d, const W* s) { W* r = d; while ((*d++ = *s++)) {} return r; }

int w_ncmp(const W* a, const W* b, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
        if (!a[i]) break;
    }
    return 0;
}
W* w_ncpy(W* d, const W* s, uint64_t n) {
    uint64_t i = 0;
    for (; i < n && s[i]; ++i) d[i] = s[i];
    for (; i < n; ++i) d[i] = 0;
    return d;
}
W* w_cat(W* d, const W* s) { w_scpy(d + std::char_traits<W>::length(d), s); return d; }
W* w_rchr(const W* s, W ch) {
    const W* r = nullptr;
    for (;; ++s) {
        if (*s == ch) r = s;
        if (!*s) return const_cast<W*>(r);
    }
}
// wcsspn / wcspbrk over 16-bit wide strings.
uint64_t w_spn(const W* s, const W* set) {
    uint64_t n = 0;
    for (; s[n] && w_chr(set, s[n]); ++n) {}
    return n;
}
W* w_pbrk(const W* s, const W* set) {
    for (; *s; ++s)
        if (w_chr(set, *s)) return const_cast<W*>(s);
    return nullptr;
}

// struct tm is nine ints on Orbis and on the host CRT, so guest tm pointers are used directly. time_t is 64-bit.
std::tm* tm_local(const int64_t* t, std::tm* out) {
    const std::time_t tt = std::time_t(*t);
#ifdef _WIN32
    return localtime_s(out, &tt) ? nullptr : out;
#else
    return localtime_r(&tt, out);
#endif
}
std::tm* tm_utc(const int64_t* t, std::tm* out) {
    const std::time_t tt = std::time_t(*t);
#ifdef _WIN32
    return gmtime_s(out, &tt) ? nullptr : out;
#else
    return gmtime_r(&tt, out);
#endif
}
thread_local std::tm t_tm;  // localtime/gmtime result buffer (per thread; guest addresses are host addresses)

// Dinkumware ctype tables for the C locale (index -1..255; the returned pointer addresses entry 0).
// Class bits (xctype.h): XD 0x01 UP 0x02 SP 0x04 PU 0x08 LO 0x10 DI 0x20 CN 0x40 BB 0x80.
struct CTables {
    int16_t type[257], lower[257], upper[257];
    CTables() {
        for (int i = -1; i < 256; ++i) {
            int16_t t = 0, lo = int16_t(i), up = int16_t(i);
            if (i >= 0 && i < 128) {
                if (i < 32 || i == 127) t |= 0x40;
                if (i >= 9 && i <= 13) t |= 0x04;
                if (i == 32) t |= 0x04 | 0x80;
                if (i >= '0' && i <= '9') t |= 0x20 | 0x01;
                if (i >= 'A' && i <= 'Z') { t |= 0x02; lo = int16_t(i + 32); }
                if (i >= 'a' && i <= 'z') { t |= 0x10; up = int16_t(i - 32); }
                if ((i >= 'A' && i <= 'F') || (i >= 'a' && i <= 'f')) t |= 0x01;
                if (std::ispunct(i)) t |= 0x08;
            }
            type[i + 1] = t;
            lower[i + 1] = lo;
            upper[i + 1] = up;
        }
    }
};
CTables& ctables() {
    static CTables t;
    return t;
}

// Dinkumware string-to-float helpers: (string, end pointer, power of ten to scale by).
template <class Ch, class F>
F dink_to_float(const Ch* s, const Ch** end, int64_t pten) {
    std::string narrow;
    const Ch* p = s;
    for (; *p && *p < 0x80; ++p) narrow += char(*p);
    char* e = nullptr;
    const F v = F(std::strtod(narrow.c_str(), &e));
    if (end) *end = s + (e - narrow.c_str());
    return pten ? F(v * std::pow(10.0, double(pten))) : v;
}

// Dinkumware's internal trig entry points: qoff selects sin (0) or cos (1).
float f_sin(float x, uint32_t qoff) { return qoff ? std::cos(x) : std::sin(x); }
double d_sin(double x, uint32_t qoff) { return qoff ? std::cos(x) : std::sin(x); }
// Dinkumware's system locks (_LOCK_LOCALE, _LOCK_MALLOC, _LOCK_STREAM, ...; _MAX_LOCK = 8), recursive like the original.
std::recursive_mutex& syslock(int i) {
    static std::recursive_mutex locks[8];
    return locks[unsigned(i) & 7];
}

} // namespace

void register_libc_wrapped() {
    reg("strncmp", wrap<+[](const char* a, const char* b, uint64_t n) { return std::strncmp(a, b, n); }>);
    reg("strcpy", wrap<+[](char* d, const char* s) { return std::strcpy(d, s); }>);
    reg("strcat", wrap<+[](char* d, const char* s) { return std::strcat(d, s); }>);
    reg("strncat", wrap<+[](char* d, const char* s, uint64_t n) { return std::strncat(d, s, n); }>);
    reg("strchr", wrap<+[](const char* s, int ch) { return const_cast<char*>(std::strchr(s, ch)); }>);
    reg("strrchr", wrap<+[](const char* s, int ch) { return const_cast<char*>(std::strrchr(s, ch)); }>);
    reg("strstr", wrap<+[](const char* a, const char* b) { return const_cast<char*>(std::strstr(a, b)); }>);
    reg("memchr", wrap<+[](const void* s, int ch, uint64_t n) { return const_cast<void*>(std::memchr(s, ch, n)); }>);
    reg("strcasecmp", wrap<+[](const char* a, const char* b) { return strcasecmp(a, b); }>);
    reg("strncasecmp", wrap<+[](const char* a, const char* b, uint64_t n) { return strncasecmp(a, b, n); }>);
    reg("atoi", wrap<+[](const char* s) { return std::atoi(s); }>);
    reg("atof", wrap<+[](const char* s) { return std::atof(s); }>);
    reg("strtol", wrap<+[](const char* s, char** e, int b) -> int64_t { return std::strtoll(s, e, b); }>);
    reg("strtoul", wrap<+[](const char* s, char** e, int b) -> uint64_t { return std::strtoull(s, e, b); }>);
    reg("strtoll", wrap<+[](const char* s, char** e, int b) -> int64_t { return std::strtoll(s, e, b); }>);
    reg("strtoull", wrap<+[](const char* s, char** e, int b) -> uint64_t { return std::strtoull(s, e, b); }>);
    reg("_Stoul", wrap<+[](const char* s, char** e, int b) -> uint64_t { return std::strtoull(s, e, b); }>);
    reg("_Stoull", wrap<+[](const char* s, char** e, int b) -> uint64_t { return std::strtoull(s, e, b); }>);
    reg("_Stoll", wrap<+[](const char* s, char** e, int b) -> int64_t { return std::strtoll(s, e, b); }>);
    reg("strtod", wrap<+[](const char* s, char** e) { return std::strtod(s, e); }>);
    reg("strtof", wrap<+[](const char* s, char** e) { return std::strtof(s, e); }>);
    reg("toupper", wrap<+[](int ch) { return std::toupper(ch); }>);
    reg("tolower", wrap<+[](int ch) { return std::tolower(ch); }>);
    reg("abs", wrap<+[](int v) { return v < 0 ? -v : v; }>);
    reg("labs", wrap<+[](int64_t v) { return v < 0 ? -v : v; }>);
    reg("strpbrk", wrap<+[](const char* s, const char* set) { return const_cast<char*>(std::strpbrk(s, set)); }>);
    reg("strspn", wrap<+[](const char* s, const char* set) -> uint64_t { return std::strspn(s, set); }>);
    reg("strcspn", wrap<+[](const char* s, const char* set) -> uint64_t { return std::strcspn(s, set); }>);
    reg("strcoll", wrap<+[](const char* a, const char* b) { return std::strcmp(a, b); }>);  // C locale: strcoll == strcmp
    reg("strtok", wrap<+[](char* s, const char* delim) { return std::strtok(s, delim); }>);  // host per-thread state; guest addresses are host addresses
    reg("strerror", wrap<+[](int e) -> const char* { return std::strerror(e); }>);  // errno 1..34 share their meaning between FreeBSD and the host CRT
    reg("localtime_s", wrap<tm_local>);  // C11 Annex K order: (const time_t*, struct tm*) -> tm* or NULL
    reg("localtime", wrap<+[](const int64_t* t) { return tm_local(t, &t_tm); }>);
    reg("gmtime", wrap<+[](const int64_t* t) { return tm_utc(t, &t_tm); }>);
    reg("mktime", wrap<+[](std::tm* tm) -> int64_t { return int64_t(std::mktime(tm)); }>);
    reg("difftime", wrap<+[](int64_t a, int64_t b) { return double(a - b); }>);
    // Dinkumware atomics (locale facet refcounts): (u32* p, u32 v, memory_order) -> old value; always seq_cst.
    reg("_Atomic_fetch_add_4", wrap<+[](uint32_t* p, uint32_t v, int) { return std::atomic_ref<uint32_t>(*p).fetch_add(v); }>);
    reg("_Atomic_fetch_sub_4", wrap<+[](uint32_t* p, uint32_t v, int) { return std::atomic_ref<uint32_t>(*p).fetch_sub(v); }>);
    // wide strings
    reg("wcschr", wrap<w_chr>);
    reg("wmemcmp", wrap<w_cmp>);
    reg("wmemcpy", wrap<w_cpy>);
    reg("wmemmove", wrap<w_cpy>);
    reg("wmemset", wrap<w_set>);
    reg("wcscmp", wrap<w_scmp>);
    reg("wcscpy", wrap<w_scpy>);
    reg("wcsstr", wrap<+[](const W* h, const W* n) -> W* {
        if (!*n) return const_cast<W*>(h);
        for (; *h; ++h) {
            const W *a = h, *b = n;
            while (*a && *b && *a == *b) ++a, ++b;
            if (!*b) return const_cast<W*>(h);
        }
        return nullptr;
    }>);
    reg("wcsncmp", wrap<w_ncmp>);
    reg("wcsncpy", wrap<w_ncpy>);
    reg("wcscat", wrap<w_cat>);
    reg("wcsrchr", wrap<w_rchr>);
    reg("wcsspn", wrap<w_spn>);
    reg("wcspbrk", wrap<w_pbrk>);
    reg("_Getpctype", wrap<+[]() { return &ctables().type[1]; }>);
    reg("_Getptolower", wrap<+[]() { return &ctables().lower[1]; }>);
    reg("_Getptoupper", wrap<+[]() { return &ctables().upper[1]; }>);
    // math (float and double)
    reg("sinf", wrap<+[](float x) { return std::sin(x); }>);
    reg("cosf", wrap<+[](float x) { return std::cos(x); }>);
    reg("tanf", wrap<+[](float x) { return std::tan(x); }>);
    reg("asinf", wrap<+[](float x) { return std::asin(x); }>);
    reg("acosf", wrap<+[](float x) { return std::acos(x); }>);
    reg("atanf", wrap<+[](float x) { return std::atan(x); }>);
    reg("atan2f", wrap<+[](float y, float x) { return std::atan2(y, x); }>);
    reg("sqrtf", wrap<+[](float x) { return std::sqrt(x); }>);
    reg("powf", wrap<+[](float x, float y) { return std::pow(x, y); }>);
    reg("expf", wrap<+[](float x) { return std::exp(x); }>);
    reg("logf", wrap<+[](float x) { return std::log(x); }>);
    reg("log10f", wrap<+[](float x) { return std::log10(x); }>);
    reg("fmodf", wrap<+[](float x, float y) { return std::fmod(x, y); }>);
    reg("floorf", wrap<+[](float x) { return std::floor(x); }>);
    reg("ceilf", wrap<+[](float x) { return std::ceil(x); }>);
    reg("fabsf", wrap<+[](float x) { return std::fabs(x); }>);
    reg("atan", wrap<+[](double x) { return std::atan(x); }>);
    reg("asin", wrap<+[](double x) { return std::asin(x); }>);
    reg("acos", wrap<+[](double x) { return std::acos(x); }>);
    reg("tanhf", wrap<+[](float x) { return std::tanh(x); }>);
    reg("exp2", wrap<+[](double x) { return std::exp2(x); }>);
    reg("frexp", wrap<+[](double x, int* e) { return std::frexp(x, e); }>);
    reg("frexpf", wrap<+[](float x, int* e) { return std::frexp(x, e); }>);
    // Dinkumware internals behind coshf/sinhf: (x, y) -> y * cosh(x) / y * sinh(x).
    reg("_FCosh", wrap<+[](float x, float y) { return y * std::cosh(x); }>);
    reg("_FSinh", wrap<+[](float x, float y) { return y * std::sinh(x); }>);
    reg("sin", wrap<+[](double x) { return std::sin(x); }>);
    reg("cos", wrap<+[](double x) { return std::cos(x); }>);
    reg("tan", wrap<+[](double x) { return std::tan(x); }>);
    reg("atan2", wrap<+[](double y, double x) { return std::atan2(y, x); }>);
    reg("sqrt", wrap<+[](double x) { return std::sqrt(x); }>);
    reg("pow", wrap<+[](double x, double y) { return std::pow(x, y); }>);
    reg("exp", wrap<+[](double x) { return std::exp(x); }>);
    reg("log", wrap<+[](double x) { return std::log(x); }>);
    reg("floor", wrap<+[](double x) { return std::floor(x); }>);
    reg("ceil", wrap<+[](double x) { return std::ceil(x); }>);
    reg("fmod", wrap<+[](double x, double y) { return std::fmod(x, y); }>);
    reg("modf", wrap<+[](double x, double* ip) { return std::modf(x, ip); }>);
    reg("modff", wrap<+[](float x, float* ip) { return std::modf(x, ip); }>);
    reg("ldexp", wrap<+[](double x, int e) { return std::ldexp(x, e); }>);
    reg("ldexpf", wrap<+[](float x, int e) { return std::ldexp(x, e); }>);
    reg("exp2f", wrap<+[](float x) { return std::exp2(x); }>);
    reg("srand", wrap<+[](uint32_t seed) { std::srand(seed); }>);
    reg("rand", wrap<+[]() -> int { return std::rand(); }>);
    reg("_FSin", wrap<f_sin>);
    reg("_FLog", wrap<+[](float x, int base10) { return base10 ? std::log10(x) : std::log(x); }>);
    reg("_Log", wrap<+[](double x, int base10) { return base10 ? std::log10(x) : std::log(x); }>);
    reg("_WStof", wrap<dink_to_float<W, float>>);
    reg("_WStod", wrap<dink_to_float<W, double>>);
    reg("_Stof", wrap<dink_to_float<char, float>>);
    reg("_Stod", wrap<dink_to_float<char, double>>);
    reg("_Sin", wrap<d_sin>);
    // Dinkumware C++ runtime: the iostream static-init constructors/destructors (the game prints nothing through iostreams) and the system locks.
    reg("_ZNSt8ios_base4InitC1Ev", wrap<+[](void*) {}>);
    reg("_ZNSt6_WinitC1Ev", wrap<+[](void*) {}>);
    reg("_ZNSt8ios_base4InitD1Ev", wrap<+[](void*) {}>);
    reg("_ZNSt6_WinitD1Ev", wrap<+[](void*) {}>);
    reg("_Locksyslock", wrap<+[](int i) { syslock(i).lock(); }>);
    reg("_Unlocksyslock", wrap<+[](int i) { syslock(i).unlock(); }>);
}

} // namespace bb::hle
