// SPDX-License-Identifier: GPL-3.0-or-later
// Process start for the recompiled eboot: TLS/TCB block, entry stack, TLS relocations, init functions, entry call.
// Layout follows the PS4/FreeBSD x86-64 TLS variant II (fs_base -> TCB, static TLS at negative offsets); field order
// of the TCB matches shadPS4's Tcb (src/core/tls.h, GPL-2.0+; reimplemented, not copied).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "core/orbis_elf.h"
#include "runtime/guest.h"

namespace bb::rt {

// Guest TCB at fs_base. canary lives at fs:0x28 (stack protector), tcb_self at fs:0, dtv at fs:8.
struct GuestTcb {
    static constexpr uint64_t kSelf = 0x00, kDtv = 0x08, kThread = 0x10, kCanary = 0x28, kFiber = 0x30, kSize = 0x40;
};

struct ProcessConfig {
    std::vector<std::string> args{"eboot.bin"};  // argv[0..]; envp is empty
    uint64_t canary = 0x5BB5BB5BB5BB5BB5ull;     // value at fs:0x28
};

// Bytes to reserve for one thread's TLS block: TLS image + TCB + dtv + alignment slack.
uint64_t tls_block_size(const elf::Image& img);

// Static TLS size of the image rounded to max(PT_TLS align, 16); TPOFF of image offset `o` is o - tls_size.
uint64_t tls_aligned_size(const elf::Image& img);

// Per-thread part of setup_main_thread: [TLS image][TCB][dtv] inside `tls_area` (tls_block_size bytes), sets
// c.fs_base and the canary at fs:0x28.
bool build_tls_block(Context& c, const elf::Image& img, uint64_t tls_area, uint64_t canary, std::string& err);

// Builds the TLS block at `tls_area` (guest address, tls_block_size bytes) and sets c.fs_base, patches the image's
// TLS relocations (DTPMOD64 = module 1, DTPOFF64 = addend, TPOFF64 = addend - tls size), then builds the entry stack
// below `stack_top`: rsp -> [argc][argv0..argvN][0][envp: 0], rdi = rsp, rsi = kExitAddress. Image must be loaded.
bool setup_main_thread(Context& c, const elf::Image& img, uint64_t bias, uint64_t stack_top, uint64_t tls_area,
                       std::string& err, const ProcessConfig& cfg = {});

// Runs DT_INIT and DT_INIT_ARRAY, then the entry point, each as a guest call. Returns when the entry returns
// (normally it ends in the exit function instead).
void run_entry(Context& c, const elf::Image& img, uint64_t bias);

}  // namespace bb::rt
