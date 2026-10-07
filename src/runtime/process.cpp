// SPDX-License-Identifier: GPL-3.0-or-later
#include "runtime/process.h"

#include <algorithm>
#include <bit>
#include <cstring>

namespace bb::rt {
namespace {

constexpr uint64_t kDtvBytes = 24;  // FreeBSD dtv: [generation][slot count][module 1 TLS block]

const elf::Segment* tls_segment(const elf::Image& img) {
    for (auto& s : img.segments)
        if (s.type == elf::PT_TLS) return &s;
    return nullptr;
}
uint64_t tls_align(const elf::Segment* s) { return std::max<uint64_t>(s ? s->align : 0, 16); }
uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

bool patch_tls_relocations(Context& c, const elf::Image& img, uint64_t bias, uint64_t tls_size, std::string& err) {
    for (auto& r : img.relocations) {
        if (r.type != elf::R_X86_64_DTPMOD64 && r.type != elf::R_X86_64_DTPOFF64 && r.type != elf::R_X86_64_TPOFF64)
            continue;
        if (r.symbol) {  // TLS symbols of other modules would need their own TLS block
            err = "TLS relocation against an imported symbol is not supported";
            return false;
        }
        uint64_t v = r.type == elf::R_X86_64_DTPMOD64 ? 1 : r.type == elf::R_X86_64_DTPOFF64
                                                              ? uint64_t(r.addend)
                                                              : uint64_t(r.addend) - tls_size;
        st<uint64_t>(c, r.offset + bias, v);
    }
    return true;
}

uint64_t dyn(const elf::Image& img, int64_t tag) {
    for (auto& [k, v] : img.dynamic)
        if (k == tag) return v;
    return 0;
}

// Guest call: push a return address, dispatch, rsp is back to its old value after the callee's `ret`.
void guest_call(Context& c, uint64_t target) {
    c.r[4] -= 8;
    st<uint64_t>(c, c.r[4], kExitAddress);
    call_indirect(c, target);
}

}  // namespace

uint64_t tls_aligned_size(const elf::Image& img) {
    const elf::Segment* s = tls_segment(img);
    return s ? align_up(s->memsz, tls_align(s)) : 0;
}

uint64_t tls_block_size(const elf::Image& img) {
    return tls_align(tls_segment(img)) + tls_aligned_size(img) + GuestTcb::kSize + kDtvBytes;
}

bool build_tls_block(Context& c, const elf::Image& img, uint64_t tls_area, uint64_t canary, std::string& err) {
    const elf::Segment* tls = tls_segment(img);
    const uint64_t align = tls_align(tls);
    if (!std::has_single_bit(align)) {
        err = "PT_TLS alignment is not a power of two";
        return false;
    }
    const uint64_t tls_size = tls_aligned_size(img);
    if (tls && (tls->filesz > tls->memsz || tls->offset > img.elf.size() || tls->filesz > img.elf.size() - tls->offset)) {
        err = "PT_TLS segment out of file bounds";
        return false;
    }

    // [TLS image][TCB][dtv]; the TCB is aligned like the TLS image so negative offsets keep their alignment.
    const uint64_t start = align_up(tls_area, align), tcb = start + tls_size, dtv = tcb + GuestTcb::kSize;
    if (tls) {
        std::memcpy(host(c, start), img.elf.data() + tls->offset, tls->filesz);
        std::memset(host(c, start + tls->filesz), 0, tls_size - tls->filesz);
    }
    std::memset(host(c, tcb), 0, GuestTcb::kSize);
    st<uint64_t>(c, tcb + GuestTcb::kSelf, tcb);
    st<uint64_t>(c, tcb + GuestTcb::kDtv, dtv);
    st<uint64_t>(c, tcb + GuestTcb::kCanary, canary);
    st<uint64_t>(c, dtv, 1);
    st<uint64_t>(c, dtv + 8, 1);
    st<uint64_t>(c, dtv + 16, start);
    c.fs_base = tcb;
    return true;
}

bool setup_main_thread(Context& c, const elf::Image& img, uint64_t bias, uint64_t stack_top, uint64_t tls_area,
                       std::string& err, const ProcessConfig& cfg) {
    if (!build_tls_block(c, img, tls_area, cfg.canary, err)) return false;
    if (!patch_tls_relocations(c, img, bias, tls_aligned_size(img), err)) return false;

    // Entry stack: strings at the top, then a 16-aligned block [argc][argv...][0][envp 0].
    uint64_t p = stack_top & ~uint64_t(15);
    std::vector<uint64_t> argv;
    for (auto& a : cfg.args) {
        p -= a.size() + 1;
        std::memcpy(host(c, p), a.c_str(), a.size() + 1);
        argv.push_back(p);
    }
    const uint64_t words = 1 + argv.size() + 2;
    const uint64_t params = (p - words * 8) & ~uint64_t(15);
    st<uint64_t>(c, params, argv.size());
    for (size_t i = 0; i < argv.size(); ++i) st<uint64_t>(c, params + 8 + i * 8, argv[i]);
    st<uint64_t>(c, params + 8 + argv.size() * 8, 0);
    st<uint64_t>(c, params + 16 + argv.size() * 8, 0);
    c.r[4] = params;
    c.r[7] = params;
    c.r[6] = kExitAddress;
    return true;
}

void run_entry(Context& c, const elf::Image& img, uint64_t bias) {
    const uint64_t params = c.r[4];
    // ponytail: the entry point (`_start`) runs the module's init functions itself; calling DT_INIT/DT_INIT_ARRAY here
    // as well constructed every static object twice (cyclic registration lists, observed with the watchpoint).
    c.r[7] = params;
    c.r[6] = kExitAddress;
    guest_call(c, img.entry + bias);
}

}  // namespace bb::rt
