// SPDX-License-Identifier: GPL-3.0-or-later
// Process start-up on a synthetic image (no eboot): TLS/TCB layout, TLS relocations, entry stack, init + entry calls.
#include <cstdio>
#include <cstring>
#include <vector>

#include "runtime/process.h"

using namespace bb::rt;

static int failures = 0;
#define CHECK(c)                                                    \
    do {                                                            \
        if (!(c)) {                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                             \
        }                                                           \
    } while (0)

// Guest functions: `ret` is "rsp += 8; return", as emitted by bbrecomp.
static int order[4], n_order;
static uint64_t entry_rsp, entry_rdi, entry_rsi, entry_fs0;
static void f_init(Context& c) { order[n_order++] = 1; c.r[7] = 0; c.r[4] += 8; }
static void f_init_array(Context& c) { order[n_order++] = 2; c.r[4] += 8; }
static void f_entry(Context& c) {
    order[n_order++] = 3;
    entry_rsp = c.r[4];
    entry_rdi = c.r[7];
    entry_rsi = c.r[6];
    entry_fs0 = ld<uint64_t>(c, c.fs_base);
    c.r[4] += 8;
}
constexpr uint64_t kBias = 0x1000;
namespace bb::rt {
const FunctionEntry kFunctions[] = {{kBias + 0x20, f_init}, {kBias + 0x30, f_init_array}, {kBias + 0x100, f_entry}};
const size_t kFunctionCount = 3;
const uint64_t kLoadBias = kBias;
const ImportEntry kImports[] = {{0, 0, "", "", ""}};
const size_t kImportCount = 0;
}  // namespace bb::rt

int main() {
    bb::elf::Image img;
    img.entry = 0x100;
    img.elf = {0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD};
    img.segments.push_back({bb::elf::PT_TLS, 4, 1, 0x500, 12, 0x45, 0x20});  // 12 bytes data, 0x45 total, align 0x20
    img.dynamic = {{bb::elf::DT_INIT, 0x20}, {bb::elf::DT_INIT_ARRAY, 0x200}, {bb::elf::DT_INIT_ARRAYSZ, 8}};
    img.relocations = {{0x300, bb::elf::R_X86_64_TPOFF64, 0, 8, false},
                       {0x308, bb::elf::R_X86_64_DTPMOD64, 0, 0, false},
                       {0x310, bb::elf::R_X86_64_DTPOFF64, 0, 5, false}};
    CHECK(bb::rt::tls_aligned_size(img) == 0x60);

    std::vector<uint8_t> mem(0x10000, 0xCC);  // 0xCC: stale bytes must not leak into the zero tail
    Context c;
    c.base = reinterpret_cast<uintptr_t>(mem.data());
    const uint64_t tls_area = 0x4008, stack_top = 0xF000 + 3;  // deliberately unaligned inputs
    CHECK(tls_area + tls_block_size(img) <= 0x8000);
    st<uint64_t>(c, kBias + 0x200, kBias + 0x30);  // relocated init-array slot

    ProcessConfig cfg;
    cfg.args = {"eboot.bin", "-x"};
    std::string err;
    CHECK(setup_main_thread(c, img, kBias, stack_top, tls_area, err, cfg));
    CHECK(err.empty());

    // TCB: aligned to the TLS alignment, self pointer, dtv, canary.
    const uint64_t fs = c.fs_base;
    CHECK(fs % 0x20 == 0 && fs >= tls_area);
    CHECK(ld<uint64_t>(c, fs + 0x00) == fs);
    const uint64_t dtv = ld<uint64_t>(c, fs + 0x08);
    CHECK(dtv == fs + GuestTcb::kSize);
    CHECK(ld<uint64_t>(c, fs + 0x28) == cfg.canary);
    CHECK(ld<uint64_t>(c, fs + 0x10) == 0);
    // TLS image: static TLS ends at fs; data copied, tail (including padding to the aligned size) zeroed.
    const uint64_t tls = fs - 0x60;
    CHECK(tls % 0x20 == 0);
    CHECK(std::memcmp(host(c, tls), img.elf.data() + 1, 12) == 0);
    for (uint64_t i = 12; i < 0x60; ++i) CHECK(*host(c, tls + i) == 0);
    CHECK(ld<uint64_t>(c, dtv + 16) == tls);  // dtv[module 1]
    CHECK(ld<uint8_t>(c, fs - 0x60) == 0xA2);  // negative-offset access, like `mov rax,fs:0; mov al,[rax-0x60]`
    // Relocations: TPOFF = addend - tls size, DTPMOD = module 1, DTPOFF = addend.
    CHECK(int64_t(ld<uint64_t>(c, kBias + 0x300)) == 8 - 0x60);
    CHECK(ld<uint8_t>(c, fs + int64_t(ld<uint64_t>(c, kBias + 0x300))) == *host(c, tls + 8));
    CHECK(ld<uint64_t>(c, kBias + 0x308) == 1);
    CHECK(ld<uint64_t>(c, kBias + 0x310) == 5);

    // Entry stack: rsp -> argc, argv[], NULL, NULL(envp); 16-byte aligned; rdi = rsp; rsi = exit function.
    const uint64_t sp = c.r[4];
    CHECK(sp % 16 == 0 && sp < stack_top);
    CHECK(ld<uint64_t>(c, sp) == 2);
    const uint64_t a0 = ld<uint64_t>(c, sp + 8), a1 = ld<uint64_t>(c, sp + 16);
    CHECK(std::strcmp(reinterpret_cast<char*>(host(c, a0)), "eboot.bin") == 0);
    CHECK(std::strcmp(reinterpret_cast<char*>(host(c, a1)), "-x") == 0);
    CHECK(ld<uint64_t>(c, sp + 24) == 0 && ld<uint64_t>(c, sp + 32) == 0);
    CHECK(a0 > sp && a1 < stack_top);
    CHECK(c.r[7] == sp && c.r[6] == kExitAddress);

    // run_entry: only the entry point (`_start` runs the init functions itself on the real system; running DT_INIT /
    // the init array here constructed every static twice). The entry sees a return address above argc and a SysV rsp.
    run_entry(c, img, kBias);
    CHECK(n_order == 1 && order[0] == 3);
    CHECK(entry_rsp == sp - 8 && entry_rsp % 16 == 8);
    CHECK(ld<uint64_t>(c, entry_rsp) == kExitAddress);
    CHECK(entry_rdi == sp && entry_rsi == kExitAddress);
    CHECK(entry_fs0 == fs);
    CHECK(c.r[4] == sp);

    // No PT_TLS: still a valid TCB.
    bb::elf::Image bare;
    Context c2;
    c2.base = c.base;
    CHECK(setup_main_thread(c2, bare, 0, stack_top, 0x2000, err));
    CHECK(c2.fs_base % 16 == 0 && ld<uint64_t>(c2, c2.fs_base) == c2.fs_base);

    // Bad alignment is rejected.
    img.segments[0].align = 24;
    CHECK(!setup_main_thread(c2, img, kBias, stack_top, 0x2000, err) && !err.empty());

    std::printf(failures ? "%d failure(s)\n" : "all process checks passed\n", failures);
    return failures ? 1 : 0;
}
