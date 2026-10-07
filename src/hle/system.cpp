// SPDX-License-Identifier: GPL-3.0-or-later
// System service libraries: UserService, SystemService, CommonDialog, Np* offline answers, memory queries.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "hle/abi.h"

namespace bb::hle {
namespace {

// Dinkumware floating-point classification: _NANCODE 2, _INFCODE 1, _FINITE -1, _DENORM -2, zero 0 (short result)
uint64_t fp_code(int k) { return uint64_t(int64_t(k == FP_NAN ? 2 : k == FP_INFINITE ? 1 : k == FP_ZERO ? 0 : k == FP_SUBNORMAL ? -2 : -1)); }

void ok(Context& c) { ret(c, 0); }

void flexible_map(Context& c) {  // (void** addr, len, prot, flags)
    const uint64_t at = rt::ld<uint64_t>(c, arg(c, 0)), flags = arg(c, 3);
    void* p = host_map((flags & 0x10) ? ptr(c, at) : nullptr, arg(c, 1));
    if (!p) return ret(c, kErrNoMem);
    rt::st<uint64_t>(c, arg(c, 0), reinterpret_cast<uintptr_t>(p) - c.base);
    note_mapping(reinterpret_cast<uintptr_t>(p) - c.base, arg(c, 1), int(arg(c, 2)));
    ret(c, 0);
}

// SceKernelVirtualQueryInfo: start, end, offset, protection, memoryType, bit flags, name[32].
void virtual_query(Context& c) {  // (addr, flags, info*, infoSize)
    const uint64_t page = arg(c, 0) & ~uint64_t(0xFFF), info = arg(c, 2);
    Mapping m{page, page + 0x1000, 3};
    find_mapping(arg(c, 0), m);
    std::memset(ptr(c, info), 0, arg(c, 3));
    rt::st<uint64_t>(c, info, m.start);
    rt::st<uint64_t>(c, info + 8, m.end);
    rt::st<int32_t>(c, info + 24, m.prot);
    rt::st<uint8_t>(c, info + 32, 1 << 4);   // isCommitted
    ret(c, 0);
}

// sceSystemServiceParamGetInt: id 1 = system language (1 = English US), id 1000 = enter-button assignment
// (0 circle, 1 cross: western default); everything else 0.
void system_param_int(Context& c) {
    const uint32_t id = arg32(c, 0);
    if (std::getenv("BB_SAVE_LOG")) std::fprintf(stderr, "service: ParamGetInt(%u)\n", id);
    rt::st<int32_t>(c, arg(c, 1), id == 1 || id == 1000 ? 1 : 0);
    ret(c, 0);
}

} // namespace

void register_system() {
    reg("sceUserServiceInitialize", ok);
    reg("sceCommonDialogInitialize", ok);
    reg("sceNpSetContentRestriction", ok);
    reg("sceKernelMapFlexibleMemory", flexible_map);
    reg("sceKernelVirtualQuery", virtual_query);
    reg("sceSystemServiceParamGetInt", system_param_int);
    reg("time", wrap<+[](int64_t* t) -> int64_t {
        const int64_t v = int64_t(std::time(nullptr));
        if (t) *t = v;
        return v;
    }>);
    reg("clock", wrap<+[]() -> int64_t { return int64_t(std::clock()) * (1000000 / CLOCKS_PER_SEC); }>);  // CLOCKS_PER_SEC = 1e6
    reg("scePthreadGetprio", [](Context& c) { rt::st<int32_t>(c, arg(c, 1), 700); ret(c, 0); });
    // setjmp is emitted inline by the recompiler (see FunctionEmitter); this stub only serves indirect calls (returns 0).
    reg("setjmp", [](Context& c) { ret(c, 0); });
    reg("_setjmp", [](Context& c) { ret(c, 0); });
    reg("sigsetjmp", [](Context& c) { ret(c, 0); });
    for (const char* n : {"longjmp", "_longjmp", "siglongjmp"}) reg(n, [](Context& c) {
        static int logged = 0;
        if (logged++ < 16)
            std::fprintf(stderr, "longjmp(jb=0x%llx, val=%d) called from guest 0x%llx\n", (unsigned long long)arg(c, 0), int(arg(c, 1)), (unsigned long long)rt::ld<uint64_t>(c, c.r[4]));
        rt::longjmp_restore(c, arg(c, 0), arg(c, 1));
    });
    reg("_Dtest", [](Context& c) { ret(c, fp_code(std::fpclassify(rt::ld<double>(c, arg(c, 0))))); });  // (double*)
    reg("_FDtest", [](Context& c) { ret(c, fp_code(std::fpclassify(rt::ld<float>(c, arg(c, 0))))); });  // (float*)
    reg("ldiv", [](Context& c) {  // struct return in rax:rdx
        const int64_t n = int64_t(arg(c, 0)), d = int64_t(arg(c, 1));
        c.r[0] = uint64_t(n / d);
        c.r[2] = uint64_t(n % d);
    });
    reg("wmemchr", wrap<+[](const uint16_t* s, uint16_t ch, uint64_t n) -> uint16_t* {
        for (uint64_t i = 0; i < n; ++i)
            if (s[i] == ch) return const_cast<uint16_t*>(s + i);
        return nullptr;
    }>);
}

} // namespace bb::hle
