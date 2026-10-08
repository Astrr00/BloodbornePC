// SPDX-License-Identifier: GPL-3.0-or-later
// libSceGnmDriver / libSceVideoOut, first slice: enough for the engine's GX device creation to succeed. Command
// buffers are only filled with PM4 NOPs here; a real PM4 interpreter + Vulkan backend replace this (M2/M3).
#include <atomic>
#include <set>
#include "gpu/gcn.h"
#include "gpu/gcn_spirv.h"
#include "gpu/gpu_hooks.h"
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <deque>
#include <algorithm>
#include <mutex>
#include <thread>
#include <vector>

#include "hle/abi.h"
#include "hle/guest_heap.h"

namespace bb::hle {
namespace {

constexpr uint32_t kPm4Nop(uint32_t dwords) { return 0xC0001000u | ((dwords - 2) << 16); }  // type-3 NOP, count = dwords-2

// Writes one NOP packet of min(n, used) dwords and returns that count: the engine advances its write pointer by the
// return value, so returning the whole buffer size would mark the command buffer full.
uint32_t fill_nop(Context& c, uint64_t buf, uint64_t n, uint64_t used) {
    if (n > used) n = used;
    uint32_t* p = ptr<uint32_t>(c, buf);
    if (n >= 2) {
        p[0] = kPm4Nop(uint32_t(n));
        for (uint64_t i = 1; i < n; ++i) p[i] = 0;
    }
    return uint32_t(n);
}

// ---- PM4 statistics (what does the engine actually record?) -----------------------------------------------------------
std::mutex g_pm4_mutex;
std::map<uint32_t, uint64_t> g_pm4_opcodes;  // type-3 opcode -> packet count
uint64_t g_pm4_type0 = 0, g_pm4_bad = 0, g_pm4_dwords = 0, g_pm4_submits = 0, g_pm4_overrun = 0, g_pm4_zero = 0;

void pm4_scan(Context& c, uint64_t addr, uint64_t dwords) {
    const uint32_t* p = ptr<const uint32_t>(c, addr);
    std::lock_guard lk(g_pm4_mutex);
    ++g_pm4_submits;
    g_pm4_dwords += dwords;
    for (uint64_t i = 0; i < dwords;) {
        const uint32_t h = p[i];
        const uint32_t type = h >> 30;
        if (type == 3) {
            ++g_pm4_opcodes[(h >> 8) & 0xFF];
            i += ((h >> 16) & 0x3FFF) + 2;
            if (i > dwords) ++g_pm4_overrun;  // a packet that runs past the buffer end = lost sync
        } else if (type == 2) {
            ++i;
        } else if (type == 0 && h == 0) {
            ++g_pm4_zero;
            ++i;  // zero dword = unused buffer space (as a real type-0 packet it would swallow the next header)
        } else if (type == 0) {
            ++g_pm4_type0;
            i += ((h >> 16) & 0x3FFF) + 2;
        } else {
            if (++g_pm4_bad <= 4) {
                std::fprintf(stderr, "PM4 bad header at dword %llu of %llu (buffer %llx): ", (unsigned long long)i, (unsigned long long)dwords, (unsigned long long)addr);
                for (uint64_t k = i > 6 ? i - 6 : 0; k < i + 6 && k < dwords; ++k) std::fprintf(stderr, "%s%08x", k == i ? "[" : " ", p[k]), std::fputs(k == i ? "]" : "", stderr);
                std::fputc('\n', stderr);
            }
            ++i;
        }
    }
}
// Effects the CPU-visible side of the stream must have even without a GPU: fences. The engine polls memory that the
// GPU writes at end of pipe (EVENT_WRITE_EOP) / via WRITE_DATA; nothing else would ever write it.
uint64_t gpu_clock() {
    static const auto t0 = std::chrono::steady_clock::now();
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()) * 100 / 1000 + 1;
}
// Register shadow (GCN register file as written by SET_*_REG): indices are register offsets relative to each block base.
struct RegState {
    uint32_t config[0x1000] = {}, sh[0x400] = {}, context[0x400] = {}, uconfig[0x10000] = {};
    uint64_t draws = 0, dispatches = 0;
} g_regs;
std::mutex g_regs_mutex;
constexpr uint32_t kCbColor0Base = 0x318, kCbColor0Info = 0x31C;                 // context regs (0xA000 + n)
constexpr uint32_t kSpiPgmLoPs = 0x08, kSpiPgmLoVs = 0x48, kComputePgmLo = 0x20C;                        // SH regs (0x2C00 + n)
// Tessellation state: sceGnmSetLsShader/HsShader carry their register block in a NOP (see set_tess_shader); the PM4 interpreter keeps it in
// SH slots no real register uses (real SH registers end near index 0x240), so the backend finds it through RegView::sh.
constexpr uint32_t kTessMagic = 0x53544242;  // 'BBTS'
constexpr uint32_t kShTessLs = 0x3E0, kShTessHs = 0x3C0;

uint64_t shader_addr(const RegState& r, uint32_t lo_idx) { return (uint64_t(r.sh[lo_idx + 1]) << 40) | (uint64_t(r.sh[lo_idx]) << 8); }

// Debug: BB_DRAW_LOG prints the first draws; BB_SHADER_DIR=<dir> writes every distinct shader (<kind>_<crc>.bin = exact
// code bytes, .txt = disassembly, .env = translation environment for `bbspv --replay`) once, which is the corpus for the GCN->SPIR-V translator.
void note_shader(Context& c, const char* kind, uint64_t a, const RegState& r) {
    static const char* dir = std::getenv("BB_SHADER_DIR");
    static std::mutex m;
    static std::set<uint64_t> seen, seen_at;
    if (!dir || !a) return;
    const uint32_t* code = ptr<const uint32_t>(c, a);
    gcn::ShaderInfo si;
    if (!gcn::shader_info(code, 16384, si)) return;
    std::lock_guard<std::mutex> g(m);
    if (seen_at.insert(a).second)  // index.txt: every address -> the crc file that holds its code (one file per distinct code)
        if (FILE* f = std::fopen((std::string(dir) + "/index.txt").c_str(), "a")) { std::fprintf(f, "%s@%llx %s_%08x\n", kind, (unsigned long long)a, kind, si.crc); std::fclose(f); }
    if (!seen.insert((uint64_t(si.crc) << 32) | si.code_bytes).second) return;
    char name[256];
    std::snprintf(name, sizeof name, "%s/%s_%08x", dir, kind, si.crc);
    std::string bin = std::string(name) + ".bin", txt = std::string(name) + ".txt";
    if (FILE* f = std::fopen(bin.c_str(), "wb")) { std::fwrite(code, 4, si.trailer_dw + 16, f); std::fclose(f); }
    if (FILE* f = std::fopen(txt.c_str(), "w")) {
        std::fprintf(f, "; %s @0x%llx type=%u code_bytes=%u\n", kind, (unsigned long long)a, si.type, si.code_bytes);
        for (size_t k = 0; k < si.code_bytes / 4;) {
            const gcn::Insn in = gcn::decode(code + k, si.code_bytes / 4 - k);
            std::fprintf(f, "%04zx: %s\n", k * 4, gcn::disasm(in).c_str());
            k += in.len;
        }
        std::fclose(f);
    }
    // Translate with the live register state; .spv on success, otherwise the reason is appended to the listing.
    gpu::GcnEnv env;
    const bool is_ps = kind[0] == 'p', is_cs = kind[0] == 'c';
    env.stage = is_ps ? gpu::ShStage::PS : is_cs ? gpu::ShStage::CS : gpu::ShStage::VS;
    const bool is_hs = kind[0] == 'h', is_ls = kind[0] == 'l';  // HS / LS: their own SH register blocks (0x108 / 0x148 + 0x0B / 0x0C), not the VS's
    const uint32_t ud = is_cs ? 0x240 : is_ps ? 0x0C : is_hs ? 0x10C : is_ls ? 0x14C : 0x4C, rsrc2 = r.sh[is_cs ? 0x213 : is_ps ? 0x0B : is_hs ? 0x10B : is_ls ? 0x14B : 0x4B];
    env.user_sgprs = (rsrc2 >> 1) & 0x1F;
    for (int i = 0; i < 16; ++i) env.user[i] = r.sh[ud + i];
    env.ps_input_addr = r.context[0x1B4];
    if (!is_ps && !is_cs) env.vs_out_cntl = r.context[0x207];  // PA_CL_VS_OUT_CNTL
    env.cs_tgid_en = (rsrc2 >> 7) & 7;
    env.cs_tidig_comps = (rsrc2 >> 11) & 3;
    for (int i = 0; i < 3; ++i) env.cs_local[i] = r.sh[0x207 + i];
    std::vector<std::pair<uint64_t, std::vector<uint32_t>>> reads;  // guest memory the translation read (s_load, fetch shader)
    env.read_mem = [&c, &reads](uint64_t addr, uint32_t* dst, uint32_t n) {
        if (addr < 0x10000 || (gpu::hooks().mem_valid && !gpu::hooks().mem_valid(addr, uint64_t(n) * 4))) return false;  // (garbage user data: an HS/LS read through the wrong registers crashed here)
        std::memcpy(dst, ptr<const uint32_t>(c, addr), size_t(n) * 4);
        reads.emplace_back(addr, std::vector<uint32_t>(dst, dst + n));
        return true;
    };
    gpu::Translation t = gpu::translate(code, si.code_bytes / 4, env);
    // <name>.env: the environment and the memory read, so `bbspv --replay <name>.bin` translates the shader offline the same way
    if (FILE* f = std::fopen((std::string(name) + ".env").c_str(), "wb")) {
        const uint32_t h[] = {0x56454242u /* "BBEV" */, uint32_t(env.stage), env.user_sgprs, env.ps_input_addr, env.vs_out_cntl, env.cs_tgid_en,
                              env.cs_tidig_comps, env.cs_local[0], env.cs_local[1], env.cs_local[2], uint32_t(reads.size())};
        std::fwrite(h, 4, sizeof h / 4, f);
        std::fwrite(env.user, 4, 16, f);
        for (const auto& [at, d] : reads) {
            const uint32_t n = uint32_t(d.size());
            std::fwrite(&at, 8, 1, f);
            std::fwrite(&n, 4, 1, f);
            std::fwrite(d.data(), 4, n, f);
        }
        std::fclose(f);
    }
    if (FILE* f = std::fopen(txt.c_str(), "a")) {
        if (t.error.empty()) {
            std::fprintf(f, "; translated: %zu words, %zu resources, %zu loads\n", t.spirv.size(), t.resources.size(), t.loads.size());
            for (const auto& res : t.resources) {
                std::fprintf(f, "; res binding=%u type=%d src=(load %d, word %u) words:", res.binding, int(res.type), res.ref.load, res.ref.word);
                for (uint32_t k = 0; k < res.dwords; ++k) std::fprintf(f, " %08x", res.words[k]);
                std::fputc('\n', f);
            }
            for (size_t k = 0; k < t.fetch_code.size();) {  // the inlined fetch shader (VS)
                const gcn::Insn in = gcn::decode(t.fetch_code.data() + k, t.fetch_code.size() - k);
                std::fprintf(f, "; fetch @0x%llx %04zx: %s\n", (unsigned long long)t.fetch_addr, k * 4, gcn::disasm(in).c_str());
                k += in.len;
            }
        }
        else std::fprintf(f, "; TRANSLATION FAILED: %s\n", t.error.c_str());
        std::fclose(f);
    }
    if (t.error.empty())
        if (FILE* f = std::fopen((std::string(name) + ".spv").c_str(), "wb")) { std::fwrite(t.spirv.data(), 4, t.spirv.size(), f); std::fclose(f); }
}

void log_draw(Context& c, const char* kind, const RegState& r, uint32_t count) {
    note_shader(c, "vs", shader_addr(r, kSpiPgmLoVs), r);
    note_shader(c, "ps", shader_addr(r, kSpiPgmLoPs), r);
    note_shader(c, "hs", shader_addr(r, 0x108), r);  // SPI_SHADER_PGM_LO_HS / _LS: tessellation stages (only meaningful for patch draws)
    note_shader(c, "ls", shader_addr(r, 0x148), r);
    note_shader(c, "cs", shader_addr(r, kComputePgmLo), r);
    static const bool on = std::getenv("BB_DRAW_LOG") != nullptr;
    static std::atomic<int> shown{0};
    if (!on || shown++ >= 12) return;
    std::fprintf(stderr, "%s #%llu: count=%u color0 base=0x%llx info=0x%08x ps=0x%llx vs=0x%llx\n", kind,
                 (unsigned long long)(r.draws + r.dispatches), count, (unsigned long long)r.context[kCbColor0Base] << 8,
                 r.context[kCbColor0Info], (unsigned long long)shader_addr(r, kSpiPgmLoPs), (unsigned long long)shader_addr(r, kSpiPgmLoVs));
}

uint32_t g_index_type = 0, g_num_instances = 1;
uint64_t g_index_base = 0;  // draw state set by INDEX_TYPE / NUM_INSTANCES packets
uint64_t g_indirect_base = 0;  // SET_BASE index 1: the argument buffer of DRAW_INDIRECT / DISPATCH_INDIRECT

// Completion labels (EOP / EOS / RELEASE_MEM): the guest reuses the memory the GPU read once it sees them, so they must not appear before
// the GPU work recorded ahead of them has run. The backend writes them when that work completes; without a backend they are written now.
void label_write(Context& c, uint64_t at, uint64_t value, uint32_t bytes) {
    if (const auto& h = gpu::hooks(); h.label) { h.label(at, value, bytes); return; }
    if (bytes == 8) rt::st<uint64_t>(c, at, value); else rt::st<uint32_t>(c, at, uint32_t(value));
}

// PM4 packets carry raw guest addresses. Anything outside the guest address space is a corrupt/stale command stream (never
// dereference it); the first few offenders are logged with their packet for diagnosis.
bool mem_ok(uint64_t a, uint32_t op, const uint32_t* pkt) {
    if (a >= 0x400000 && a < 0xF000000000ull) return true;
    static std::atomic<int> logged[256];  // per opcode: one noisy opcode must not use up the budget of the others
    if (op != 0x50 && logged[op & 0xFF]++ < 8) {  // DMA_DATA to small "addresses" are GDS/register targets: expected, silent
        std::fprintf(stderr, "gnm: PM4 op 0x%x with bad address 0x%llx ignored; packet:", op, (unsigned long long)a);
        for (int k = 0; k < 12; ++k) std::fprintf(stderr, " %08x", pkt[k]);
        std::fputc('\n', stderr);
    }
    return false;
}

constexpr uint32_t kCeDwords = 0xC000 / 4;  // constant-engine RAM, 48 KB
uint32_t g_ce_ram[kCeDwords];
std::mutex g_ce_mutex;

// The constant engine (CCB) fills descriptor tables in memory ahead of the draw buffer (DCB); the two streams are ordered by the
// CE counter: every INCREMENT_CE_COUNTER of the CCB releases one WAIT_ON_CE_COUNTER of the DCB. The CCB runs lazily, on those waits.
struct CeStream { uint64_t addr = 0, dwords = 0, pos = 0, incs = 0, waits = 0; } g_ceq;
void ce_advance(Context& c, uint64_t until_incs);

// Executes a packet stream; returns the dwords consumed (a CCB run stops after one INCREMENT_CE_COUNTER).
uint64_t pm4_execute(Context& c, uint64_t addr, uint64_t dwords, int depth = 0, bool ce_mode = false) {
    // a stream outside mapped guest memory (stale/corrupt pointer in a submit or INDIRECT_BUFFER) is skipped, not dereferenced
    if (const auto& hk = gpu::hooks(); hk.mem_valid && dwords && !hk.mem_valid(addr, dwords * 4)) {
        static std::atomic<int> logged{0};
        if (logged++ < 8) std::fprintf(stderr, "gnm: PM4 stream 0x%llx (%llu dwords) is not mapped; skipped\n", (unsigned long long)addr, (unsigned long long)dwords);
        return dwords;
    }
    const uint32_t* p = ptr<const uint32_t>(c, addr);
    for (uint64_t i = 0; i < dwords;) {
        const uint32_t h = p[i];
        if (h >> 30 != 3) { ++i; continue; }  // filler / padding (zero dwords)
        const uint32_t op = (h >> 8) & 0xFF, n = ((h >> 16) & 0x3FFF) + 1;  // body dwords
        const uint32_t* b = p + i + 1;
        if (i + 1 + n > dwords) break;
        switch (op) {
        case 0x10:  // NOP: the LS/HS setters carry their register block here
            if (n >= 3 && b[0] == kTessMagic && (b[1] == 1 || b[1] == 2)) {
                std::lock_guard lk(g_regs_mutex);
                uint32_t* dst = &g_regs.sh[b[1] == 1 ? kShTessLs : kShTessHs];
                for (uint32_t k = 0; k < 28 && 2 + k < n; ++k) dst[k] = b[2 + k];
            }
            break;
        case 0x47: {  // EVENT_WRITE_EOP: [event][addr lo][addr hi | int_sel<<24 | data_sel<<29][data lo][data hi]
            const uint32_t data_sel = b[2] >> 29;
            const uint64_t at = b[1] | (uint64_t(b[2] & 0xFFFF) << 32);
            if (at && data_sel && mem_ok(at, op, p + i)) {
                if (data_sel == 1) label_write(c, at, b[3], 4);
                else if (data_sel == 2) label_write(c, at, b[3] | (uint64_t(b[4]) << 32), 8);
                else if (data_sel == 3) label_write(c, at, gpu_clock(), 8);
            }
            break;
        }
        case 0x49: {  // RELEASE_MEM (compute / newer gfx): [event][data_sel<<29|int_sel<<24|dst_sel<<16][addr lo][addr hi][data lo][data hi][ctx id]
            if (n < 6) break;
            const uint32_t data_sel = b[1] >> 29;
            const uint64_t at = b[2] | (uint64_t(b[3] & 0xFFFF) << 32);
            if (at && data_sel && mem_ok(at, op, p + i)) {
                if (data_sel == 1) label_write(c, at, b[4], 4);
                else if (data_sel == 2) label_write(c, at, b[4] | (uint64_t(b[5]) << 32), 8);
                else if (data_sel == 3) label_write(c, at, gpu_clock(), 8);
            }
            break;
        }
        case 0x37: {  // WRITE_DATA: [control][addr lo][addr hi][data...]; dst_sel 5 = memory (async), 2 = TC L2
            const uint32_t dst = (b[0] >> 8) & 0xF;
            const uint64_t at = b[1] | (uint64_t(b[2]) << 32);
            if ((dst == 5 || dst == 2 || dst == 1) && n >= 4 && mem_ok(at, op, p + i) && mem_ok(at + 4 * (n - 4), op, p + i)) {
                if (gpu::hooks().sync_read) gpu::hooks().sync_read(at, 4ull * (n - 3), 5);  // a pending GPU write there must not land after this one
                gpu::gpu_watch("WRITE_DATA", at, 4ull * (n - 3));
                for (uint32_t k = 3; k < n; ++k) rt::st<uint32_t>(c, at + 4 * (k - 3), b[k]);
            }
            break;
        }
        case 0x69:  // SET_CONTEXT_REG: [reg - 0xA000][values...]
        case 0x76:  // SET_SH_REG: [reg - 0x2C00][values...]
        case 0x79: {  // SET_UCONFIG_REG: [reg - 0xC000][values...]
            std::lock_guard lk(g_regs_mutex);
            uint32_t* file = op == 0x69 ? g_regs.context : op == 0x76 ? g_regs.sh : g_regs.uconfig;
            const uint32_t size = op == 0x69 ? 0x400 : op == 0x76 ? 0x400 : 0x10000;
            for (uint32_t k = 1; k < n; ++k)
                if (b[0] + k - 1 < size) file[b[0] + k - 1] = b[k];
            break;
        }
        case 0x2A: g_index_type = b[0] & 3; break;   // INDEX_TYPE: 0 = 16 bit, 1 = 32 bit
        case 0x26: if (n >= 2) g_index_base = b[0] | (uint64_t(b[1] & 0xFFFF) << 32); break;  // INDEX_BASE
        case 0x11: if (n >= 3 && (b[0] & 0xF) == 1) g_indirect_base = b[1] | (uint64_t(b[2] & 0xFFFF) << 32); break;  // SET_BASE: [index][addr lo][addr hi]
        case 0x16:    // DISPATCH_INDIRECT: [data offset][initiator]; args {x, y, z} at the SET_BASE address + offset
        case 0x24: {  // DRAW_INDIRECT: [data offset][base vertex SH reg][start instance SH reg][initiator]; args {count, instances, first vertex, first instance}
            // The arguments are usually written by a compute shader just before. A dispatch whose arguments the GPU still writes reads them
            // on the GPU (backend); otherwise (and for draws, whose first vertex/instance go into user SGPRs) the backend makes those GPU
            // writes visible and they are read here. ponytail: draws still read synchronously.
            const uint64_t at = g_indirect_base + b[0];
            const uint32_t nargs = op == 0x16 ? 3 : 4;
            if (!g_indirect_base || !mem_ok(at, op, p + i) || !mem_ok(at + 4 * nargs - 1, op, p + i)) break;
            const auto& hk = gpu::hooks();
            if (op == 0x16 && hk.dispatch_indirect) {
                std::lock_guard lk(g_regs_mutex);
                if (hk.dispatch_indirect({g_regs.sh, g_regs.context, g_regs.uconfig}, at)) { ++g_regs.dispatches; break; }
            }
            if (hk.sync_read) hk.sync_read(at, 4 * nargs, 4);
            uint32_t a[4] = {};
            for (uint32_t k = 0; k < nargs; ++k) a[k] = rt::ld<uint32_t>(c, at + 4 * k);
            std::lock_guard lk(g_regs_mutex);
            if (op == 0x16) {
                log_draw(c, "dispatch", g_regs, a[0] * a[1] * a[2]);
                ++g_regs.dispatches;
                if (hk.dispatch && a[0] && a[1] && a[2]) hk.dispatch({g_regs.sh, g_regs.context, g_regs.uconfig}, a[0], a[1], a[2]);
                break;
            }
            if (n >= 3) {  // the CP stores the first vertex / instance into the shader's user SGPRs named by the packet
                if (b[1] && (b[1] & 0xFFFF) < 0x400) g_regs.sh[b[1] & 0xFFFF] = a[2];
                if (b[2] && (b[2] & 0xFFFF) < 0x400) g_regs.sh[b[2] & 0xFFFF] = a[3];
            }
            log_draw(c, "draw", g_regs, a[0]);
            ++g_regs.draws;
            if (hk.draw && a[0] && a[1]) {
                gpu::DrawCmd d;
                d.count = a[0];
                d.instances = a[1];
                hk.draw({g_regs.sh, g_regs.context, g_regs.uconfig}, d);
            }
            break;
        }
        case 0x35: {  // DRAW_INDEX_OFFSET_2: [max size][index offset][index count][draw initiator]; base from INDEX_BASE
            std::lock_guard lk(g_regs_mutex);
            log_draw(c, "draw_indexed", g_regs, b[2]);
            ++g_regs.draws;
            if (const auto& hk = gpu::hooks(); hk.draw && n >= 4) {
                gpu::DrawCmd d;
                d.indexed = true;
                d.count = b[2];
                d.instances = g_num_instances ? g_num_instances : 1;
                d.index_bytes = g_index_type == 1 ? 4 : 2;
                d.index_addr = g_index_base + uint64_t(b[1]) * d.index_bytes;
                hk.draw({g_regs.sh, g_regs.context, g_regs.uconfig}, d);
            }
            break;
        }
        case 0x50: {  // DMA_DATA: [control][src lo][src hi][dst lo][dst hi][byte count | flags]; memory copy / fill
            if (n < 6) break;
            const uint32_t src_sel = (b[0] >> 29) & 3, dst_sel = (b[0] >> 20) & 3;  // src: 0/3 address (3 = through L2), 1 GDS, 2 immediate data; dst: 0/3 address, 1 GDS
            const uint64_t dst = b[3] | (uint64_t(b[4]) << 32), bytes = b[5] & 0x1FFFFF;
            if (dst_sel == 1 || src_sel == 1) {  // GDS end (the address word is the GDS byte offset): the backend copies in stream order
                const bool to_gds = dst_sel == 1;
                const uint64_t mem = to_gds ? b[1] | (uint64_t(b[2]) << 32) : dst;
                if ((to_gds ? src_sel == 0 || src_sel == 3 : dst_sel == 0 || dst_sel == 3) && bytes && gpu::hooks().gds_copy)
                    gpu::hooks().gds_copy(to_gds, mem, to_gds ? b[3] : b[1], uint32_t(bytes));
                break;
            }
            if ((dst_sel != 0 && dst_sel != 3) || !dst || !bytes || !mem_ok(dst, op, p + i) || !mem_ok(dst + bytes - 1, op, p + i)) break;
            gpu::gpu_watch(src_sel == 2 ? "DMA_DATA fill" : "DMA_DATA copy", dst, bytes);
            // executed now, at scan time: GPU work recorded before must have written (and not later overwrite) both ranges. A source the
            // GPU is still writing is copied by the backend in stream order instead (no wait for the GPU to drain).
            if (const auto& hk = gpu::hooks(); hk.sync_read) {
                if (src_sel == 0 || src_sel == 3) {
                    const uint64_t src = b[1] | (uint64_t(b[2]) << 32);
                    if (src && mem_ok(src, op, p + i) && mem_ok(src + bytes - 1, op, p + i) && hk.mem_copy && hk.mem_copy(dst, src, bytes)) break;
                }
                hk.sync_read(dst, bytes, 6);
                if (src_sel == 0 || src_sel == 3) hk.sync_read(b[1] | (uint64_t(b[2]) << 32), bytes, 7);
            }
            if (src_sel == 2) {  // fill with the 32-bit immediate in the src_lo word
                uint8_t* d = ptr<uint8_t>(c, dst);
                for (uint64_t k = 0; k < bytes; ++k) d[k] = uint8_t(b[1] >> (8 * (k & 3)));
            } else if (src_sel == 0 || src_sel == 3) {
                const uint64_t src = b[1] | (uint64_t(b[2]) << 32);
                if (src && mem_ok(src, op, p + i) && mem_ok(src + bytes - 1, op, p + i)) std::memmove(ptr<void>(c, dst), ptr<const void>(c, src), size_t(bytes));
            }
            break;
        }
        case 0x81: {  // WRITE_CONST_RAM: [byte offset][data...]
            std::lock_guard lk(g_ce_mutex);
            if (n >= 2 && (b[0] & 0xFFFF) / 4 + (n - 1) <= kCeDwords) std::memcpy(&g_ce_ram[(b[0] & 0xFFFF) / 4], b + 1, 4ull * (n - 1));
            break;
        }
        case 0x80: {  // LOAD_CONST_RAM: [addr lo][addr hi][num dwords][byte offset]
            const uint64_t src = b[0] | (uint64_t(b[1] & 0xFFFF) << 32);
            const uint32_t num = b[2] & 0x7FFF, off = (b[3] & 0xFFFF) / 4;
            if (num && gpu::hooks().sync_read) gpu::hooks().sync_read(src, 4ull * num, 8);  // read at scan time: earlier GPU writes must have landed
            std::lock_guard lk(g_ce_mutex);
            if (n >= 4 && off + num <= kCeDwords && mem_ok(src, op, p + i) && mem_ok(src + 4ull * num - 1, op, p + i)) std::memcpy(&g_ce_ram[off], ptr<const void>(c, src), 4ull * num);
            break;
        }
        case 0x83: {  // DUMP_CONST_RAM: [byte offset][num dwords][addr lo][addr hi]
            const uint64_t dst = b[2] | (uint64_t(b[3]) << 32);
            const uint32_t num = b[1] & 0x7FFF, off = (b[0] & 0xFFFF) / 4;
            if (num && gpu::hooks().sync_read) gpu::hooks().sync_read(dst, 4ull * num, 9);  // a pending GPU write there must not land after this one
            std::lock_guard lk(g_ce_mutex);
            if (n >= 4 && num && off + num <= kCeDwords && mem_ok(dst, op, p + i) && mem_ok(dst + 4ull * num - 1, op, p + i)) { gpu::gpu_watch("DUMP_CONST_RAM", dst, 4ull * num); std::memcpy(ptr<void>(c, dst), &g_ce_ram[off], 4ull * num); }
            break;
        }
        case 0x2F: g_num_instances = b[0]; break;    // NUM_INSTANCES
        case 0x2D: {  // DRAW_INDEX_AUTO: [index count][draw initiator]
            std::lock_guard lk(g_regs_mutex);
            log_draw(c, "draw", g_regs, b[0]);
            ++g_regs.draws;
            if (const auto& hk = gpu::hooks(); hk.draw) {
                gpu::DrawCmd d;
                d.count = b[0];
                d.instances = g_num_instances ? g_num_instances : 1;
                hk.draw({g_regs.sh, g_regs.context, g_regs.uconfig}, d);
            }
            break;
        }
        case 0x27: {  // DRAW_INDEX_2: [max size][index base lo][index base hi][index count][draw initiator]
            std::lock_guard lk(g_regs_mutex);
            log_draw(c, "draw_indexed", g_regs, b[3]);
            ++g_regs.draws;
            if (const auto& hk = gpu::hooks(); hk.draw && n >= 5) {
                gpu::DrawCmd d;
                d.indexed = true;
                d.count = b[3];
                d.instances = g_num_instances ? g_num_instances : 1;
                d.index_addr = b[1] | (uint64_t(b[2] & 0xFFFF) << 32);
                d.index_bytes = g_index_type == 1 ? 4 : 2;
                hk.draw({g_regs.sh, g_regs.context, g_regs.uconfig}, d);
            }
            break;
        }
        case 0x15: {  // DISPATCH_DIRECT: [x][y][z][initiator]
            std::lock_guard lk(g_regs_mutex);
            log_draw(c, "dispatch", g_regs, b[0] * b[1] * b[2]);
            ++g_regs.dispatches;
            if (const auto& hk = gpu::hooks(); hk.dispatch) hk.dispatch({g_regs.sh, g_regs.context, g_regs.uconfig}, b[0], b[1], b[2]);
            break;
        }
        case 0x84:  // INCREMENT_CE_COUNTER
            ++g_ceq.incs;
            if (ce_mode) return i + 1 + n;
            break;
        case 0x86:  // WAIT_ON_CE_COUNTER (draw buffer): let the constant engine run up to its next counter increment
            if (!ce_mode) ce_advance(c, ++g_ceq.waits);
            break;
        case 0x48: {  // EVENT_WRITE_EOS: [event][addr lo][addr hi | command<<29][data]; command 2 = store the 32-bit data, 1 = store GDS data
            const uint32_t cmd = b[2] >> 29;
            const uint64_t at = b[1] | (uint64_t(b[2] & 0xFFFF) << 32);
            if (n >= 4 && cmd == 2 && at && mem_ok(at, op, p + i)) label_write(c, at, b[3], 4);
            else if (static bool said = false; !said) { said = true; std::fprintf(stderr, "gnm: EVENT_WRITE_EOS command %u not implemented\n", cmd); }  // ponytail: GDS readback form unseen in the game
            break;
        }
        case 0x46: {  // EVENT_WRITE: [event type 5:0 | index 11:8][addr lo][addr hi]
            // PIXEL_PIPE_STAT_DUMP (0x39): the occlusion-query sample counters of every render backend land at the address as 64-bit values
            // (bit 63 = valid) with a 16-byte stride (begin/end pairs, PAL's OcclusionQueryResultPair; observed: begin dump at a 16-byte
            // aligned X, end dump at X+8, 8 RBs). The guest waits for the valid bits and culls on end - begin.
            // ponytail: no real sample counting. Every query reads as complete and visible as soon as it begins (the begin dump also
            // fills the end slots), so a query read before its end dump is never "occluded".
            if ((b[0] & 0x3F) == 0x39 && n >= 3) {
                constexpr uint64_t kValid = 1ull << 63, kVisible = 1u << 20;
                const uint64_t at = (b[1] & ~7u) | (uint64_t(b[2] & 0xFFFF) << 32), pair = at & ~15ull;
                if (at && mem_ok(pair, op, p + i) && mem_ok(pair + 16 * 7 + 15, op, p + i))
                    for (int k = 0; k < 8; ++k) {
                        if (at == pair) rt::st<uint64_t>(c, pair + 16 * k, kValid);
                        rt::st<uint64_t>(c, pair + 16 * k + 8, kValid | kVisible);
                    }
            }
            break;
        }
        case 0x3F:  // INDIRECT_BUFFER: [addr lo][addr hi | vmid<<24][size dwords | flags]
            if (depth < 4 && n >= 3 && mem_ok(b[0] | (uint64_t(b[1] & 0xFFFF) << 32), op, p + i)) pm4_execute(c, b[0] | (uint64_t(b[1] & 0xFFFF) << 32), b[2] & 0xFFFFF, depth + 1);
            break;
        default:
            if (static std::atomic<bool> seen[256]; !seen[op & 0xFF].exchange(true)) std::fprintf(stderr, "gnm: PM4 op 0x%x not handled (first seen; ignored)\n", op);
            break;
        }
        i += 1 + n;
    }
    return dwords;
}

void ce_advance(Context& c, uint64_t until_incs) {
    while (g_ceq.pos < g_ceq.dwords && g_ceq.incs < until_incs) g_ceq.pos += pm4_execute(c, g_ceq.addr + 4 * g_ceq.pos, g_ceq.dwords - g_ceq.pos, 0, true);
}

// (count, void** dcb, uint32_t* dcbBytes, void** ccb, uint32_t* ccbBytes)
void submit_scan(Context& c) {
    for (uint64_t i = 0; i < arg32(c, 0); ++i) {
        const uint64_t ccb = arg(c, 3) ? rt::ld<uint64_t>(c, arg(c, 3) + 8 * i) : 0;
        g_ceq = {ccb, ccb ? rt::ld<uint32_t>(c, arg(c, 4) + 4 * i) / 4 : 0, 0, 0, 0};
        pm4_execute(c, rt::ld<uint64_t>(c, arg(c, 1) + 8 * i), rt::ld<uint32_t>(c, arg(c, 2) + 4 * i) / 4);
        ce_advance(c, ~0ull);  // whatever the draw buffer did not wait for
        g_ceq = {};
    }
    if (const auto& h = gpu::hooks(); h.end_submit) h.end_submit();
    if (static const bool stats = std::getenv("BB_PM4_STATS") != nullptr; !stats) return;
    for (uint64_t i = 0; i < arg32(c, 0); ++i) {
        pm4_scan(c, rt::ld<uint64_t>(c, arg(c, 1) + 8 * i), rt::ld<uint32_t>(c, arg(c, 2) + 4 * i) / 4);
        if (arg(c, 3) && rt::ld<uint64_t>(c, arg(c, 3) + 8 * i))
            pm4_scan(c, rt::ld<uint64_t>(c, arg(c, 3) + 8 * i), rt::ld<uint32_t>(c, arg(c, 4) + 4 * i) / 4);
    }
}

// ---- command-buffer writers of the driver library ------------------------------------------------------------------------
// sceGnmSet*Shader write fixed-length packet sequences (the engine advances its write pointer by a constant): register
// sets followed by one trailing NOP that pads to the fixed length. Register numbers per the GCN register map
// (verified against shadPS4's gnmdriver.cpp, GPL-2.0+, read for facts only).
constexpr uint32_t kOpSetCtx = 0x69, kOpSetSh = 0x76, kOpNop = 0x10;
inline uint32_t pm4_header3(uint32_t op, uint32_t body_dwords) { return 0xC0000000u | ((body_dwords - 1) << 16) | (op << 8); }

template <class... V>
uint32_t* emit_set(uint32_t* p, uint32_t op, uint32_t reg, V... values) {
    *p++ = pm4_header3(op, 1 + sizeof...(V));
    *p++ = reg;
    ((*p++ = values), ...);
    return p;
}
uint32_t* emit_trailing_nop(uint32_t* p, uint32_t data_dwords) {
    *p++ = pm4_header3(kOpNop, data_dwords);
    *p++ = 0;
    for (uint32_t i = 1; i < data_dwords; ++i) *p++ = 0;  // contents are ignored by the CP
    return p;
}

void set_ps_shader(Context& c) {  // (cmdbuf, size, ps_regs): fixed 0x28 dwords
    uint32_t* p = ptr<uint32_t>(c, arg(c, 0));
    const uint32_t* r = arg(c, 2) ? ptr<const uint32_t>(c, arg(c, 2)) : nullptr;
    if (!p || arg32(c, 1) <= 0x27) return ret(c, uint64_t(int64_t(-1)));
    if (!r) {
        p = emit_set(p, kOpSetSh, 8u, 0u, 0u);
        p = emit_set(p, kOpSetCtx, 0x203u, 0u);
        emit_trailing_nop(p, 0x20);
    } else {
        if (r[1] != 0) return ret(c, uint64_t(int64_t(-1)));
        p = emit_set(p, kOpSetSh, 8u, r[0], 0u);          // SPI_SHADER_PGM_LO/HI_PS
        p = emit_set(p, kOpSetSh, 10u, r[2], r[3]);       // SPI_SHADER_PGM_RSRC1/2_PS
        p = emit_set(p, kOpSetCtx, 0x1c4u, r[4], r[5]);   // SPI_SHADER_Z_FORMAT / COL_FORMAT
        p = emit_set(p, kOpSetCtx, 0x1b3u, r[6], r[7]);   // SPI_PS_INPUT_ENA / ADDR
        p = emit_set(p, kOpSetCtx, 0x1b6u, r[8]);         // SPI_PS_IN_CONTROL
        p = emit_set(p, kOpSetCtx, 0x1b8u, r[9]);         // SPI_BARYC_CNTL
        p = emit_set(p, kOpSetCtx, 0x203u, r[10]);        // DB_SHADER_CONTROL
        p = emit_set(p, kOpSetCtx, 0x8fu, r[11]);         // CB_SHADER_MASK
        emit_trailing_nop(p, 11);
    }
    ret(c, 0);
}
void set_vs_shader(Context& c) {  // (cmdbuf, size, vs_regs, modifier): fixed 0x1d dwords
    uint32_t* p = ptr<uint32_t>(c, arg(c, 0));
    const uint32_t* r = arg(c, 2) ? ptr<const uint32_t>(c, arg(c, 2)) : nullptr;
    const uint32_t modifier = arg32(c, 3);
    if (!p || !r || arg32(c, 1) <= 0x1c || (modifier & 0xfcfffc3fu) || r[1] != 0) return ret(c, uint64_t(int64_t(-1)));
    const uint32_t rsrc1 = modifier == 0 ? r[2] : (r[2] & 0xfcfffc3fu) | modifier;
    p = emit_set(p, kOpSetSh, 0x48u, r[0], 0u);       // SPI_SHADER_PGM_LO/HI_VS
    p = emit_set(p, kOpSetSh, 0x4au, rsrc1, r[3]);    // SPI_SHADER_PGM_RSRC1/2_VS
    p = emit_set(p, kOpSetCtx, 0x207u, r[6]);         // PA_CL_VS_OUT_CNTL
    p = emit_set(p, kOpSetCtx, 0x1b1u, r[4]);         // SPI_VS_OUT_CONFIG
    p = emit_set(p, kOpSetCtx, 0x1c3u, r[5]);         // SPI_SHADER_POS_FORMAT
    emit_trailing_nop(p, 11);
    ret(c, 0);
}
// Tessellation / geometry stages (LS, HS, ES, GS) are not translated yet (their draws are skipped, see backend). The guest still
// advances its write pointer by `size` dwords, so the reserved space MUST hold valid packets: one NOP covering all of it.
void fill_reserved_nop(Context& c) {  // (cmdbuf, size, ...)
    uint32_t* p = ptr<uint32_t>(c, arg(c, 0));
    const uint32_t n = arg32(c, 1);
    if (!p || n < 2) return ret(c, uint64_t(int64_t(-1)));
    emit_trailing_nop(p, n - 1);
    ret(c, 0);
}
// LS / HS: the first four register words follow the VS/CS layout (PGM_LO, PGM_HI, RSRC1, RSRC2) and become real SET_SH_REG packets; the
// whole register block is also carried in a trailing NOP (magic, stage, raw words) so tessellation state can be read at draw time.
// (kTessMagic: see the register shadow above)
void set_tess_shader(Context& c, uint32_t stage) {  // stage 1 = LS, 2 = HS; (cmdbuf, size, regs, ...)
    uint32_t* p = ptr<uint32_t>(c, arg(c, 0));
    const uint32_t* r = arg(c, 2) ? ptr<const uint32_t>(c, arg(c, 2)) : nullptr;
    const uint32_t n = arg32(c, 1);
    if (!p || !r || n < 12) return ret(c, uint64_t(int64_t(-1)));
    static const bool log = std::getenv("BB_TESS_LOG") != nullptr;
    static std::atomic<int> shown{0};
    if (log && shown++ < 8) {
        std::fprintf(stderr, "gnm: %s shader regs (size %u dwords, arg3 %llx):", stage == 1 ? "LS" : "HS", n, (unsigned long long)arg(c, 3));
        for (int k = 0; k < 16; ++k) std::fprintf(stderr, " %08x", r[k]);
        std::fputc('\n', stderr);
    }
    const uint32_t base = stage == 1 ? 0x148u : 0x108u;
    p = emit_set(p, kOpSetSh, base, r[0], 0u);          // SPI_SHADER_PGM_LO/HI_LS|HS
    p = emit_set(p, kOpSetSh, base + 2, r[2], r[3]);    // SPI_SHADER_PGM_RSRC1/2_LS|HS
    const uint32_t data = n - 9;                        // 8 dwords used above, the NOP is header + data
    uint32_t* nop = p;
    emit_trailing_nop(p, data);
    nop[1] = kTessMagic;
    nop[2] = stage;
    for (uint32_t k = 0; k < 28 && 3 + k < 1 + data; ++k) nop[3 + k] = r[k];
    ret(c, 0);
}
void set_cs_shader(Context& c) {  // (cmdbuf, size, cs_regs): fixed 0x19 dwords
    uint32_t* p = ptr<uint32_t>(c, arg(c, 0));
    const uint32_t* r = arg(c, 2) ? ptr<const uint32_t>(c, arg(c, 2)) : nullptr;
    if (!p || !r || arg32(c, 1) <= 0x18 || r[1] != 0) return ret(c, uint64_t(int64_t(-1)));
    p = emit_set(p, kOpSetSh, 0x20cu, r[0], 0u);      // COMPUTE_PGM_LO/HI
    p = emit_set(p, kOpSetSh, 0x212u, r[2], r[3]);    // COMPUTE_PGM_RSRC1/2
    p = emit_set(p, kOpSetSh, 0x207u, r[4], r[5], r[6]);  // COMPUTE_NUM_THREAD_X/Y/Z
    emit_trailing_nop(p, 11);
    ret(c, 0);
}

void gnm_zero(Context& c) { ret(c, 0); }
void gnm_one(Context& c) { ret(c, 1); }

// Compute queues: the guest writes PM4 into a ring (sceGnmMapComputeQueue: pipe, queue, ring base, ring dwords, read-pointer
// address) and rings sceGnmDingDong(vqueue, next write offset in dwords). Work is executed synchronously there; the read
// pointer is then advanced to the write offset, which is what the engine's completion/space checks poll.
struct ComputeQueue { uint64_t base, rptr; uint32_t size, last; };
std::mutex g_cq_mutex;
std::vector<ComputeQueue> g_cqs;

void map_queue(Context& c) {
    std::lock_guard lk(g_cq_mutex);
    g_cqs.push_back({arg(c, 2), arg(c, 4), arg32(c, 3), 0});
    ret(c, g_cqs.size());  // vqueue id = index + 1
}
void ding_dong(Context& c) {  // (vqueue, next start offset in dwords)
    std::lock_guard lk(g_cq_mutex);
    const uint64_t id = arg(c, 0);
    const uint32_t next = arg32(c, 1);
    if (!id || id > g_cqs.size()) return ret(c, 0);
    ComputeQueue& q = g_cqs[id - 1];
    if (!q.size || next > q.size) return ret(c, 0);
    if (next >= q.last) pm4_execute(c, q.base + 4ull * q.last, next - q.last);
    else {  // wrapped
        pm4_execute(c, q.base + 4ull * q.last, q.size - q.last);
        pm4_execute(c, q.base, next);
    }
    q.last = next;
    if (q.rptr) rt::st<uint32_t>(c, q.rptr, next);
    ret(c, 0);
}

void tess_ring(Context& c) {  // 2 MiB-aligned work area handed out once, sized for the maximum ring (0x20000 + 0x40000)
    static void* area = host_map(nullptr, 0x100000);
    ret(c, reinterpret_cast<uintptr_t>(area) - c.base);
}

// ---- event queues (flip events) ------------------------------------------------------------------------------------
struct Event {  // SceKernelEvent
    uint64_t ident;
    int16_t filter;
    uint16_t flags;
    uint32_t fflags;
    int64_t data;
    uint64_t udata;
};
struct Equeue {
    std::mutex m;
    std::condition_variable cv;
    std::deque<Event> events;
};
constexpr int16_t kEvfiltVideoOut = -13;

struct VideoOut {
    std::mutex m;
    std::vector<std::pair<Equeue*, uint64_t>> flip_listeners;  // (queue, udata)
    uint64_t flip_count = 0;
    uint64_t flip_arg = 0;
    uint32_t flip_rate = 0;  // sceVideoOutSetFlipRate: 0 = 60 Hz, 1 = 30 Hz, 2 = 20 Hz
    std::chrono::steady_clock::time_point last_flip{};
};
VideoOut& vo() {
    static VideoOut v;
    return v;
}

void eq_create(Context& c) {  // (SceKernelEqueue* out, name)
    Equeue* q = guest_new<Equeue>();
    if (!q) return ret(c, kErrNoMem);
    rt::st<uint64_t>(c, arg(c, 0), reinterpret_cast<uintptr_t>(q));
    ret(c, 0);
}
void eq_delete(Context& c) { guest_delete(reinterpret_cast<Equeue*>(arg(c, 0))); ret(c, 0); }
void eq_wait(Context& c) {  // (eq, SceKernelEvent* ev, int num, int* out, SceKernelUseconds* timeout)
    auto* q = reinterpret_cast<Equeue*>(arg(c, 0));
    std::unique_lock lk(q->m);
    const auto ready = [&] { return !q->events.empty(); };
    if (arg(c, 4)) {
        if (!q->cv.wait_for(lk, std::chrono::microseconds(rt::ld<uint32_t>(c, arg(c, 4))), ready)) {
            rt::st<int32_t>(c, arg(c, 3), 0);
            return ret(c, kErrTimedOut);
        }
    } else {
        q->cv.wait(lk, ready);
    }
    int n = 0;
    while (n < int(arg(c, 2)) && !q->events.empty()) {
        std::memcpy(ptr(c, arg(c, 1) + 32 * uint64_t(n)), &q->events.front(), sizeof(Event));
        q->events.pop_front();
        ++n;
    }
    rt::st<int32_t>(c, arg(c, 3), n);
    ret(c, 0);
}

// ---- VideoOut -----------------------------------------------------------------------------------------------------
void vo_attr(Context& c) {  // (attr*, pixelFormat, tilingMode, aspect, width, height, pitchInPixel)
    uint32_t* a = ptr<uint32_t>(c, arg(c, 0));
    std::memset(a, 0, 64);
    a[0] = arg32(c, 1);
    a[1] = arg32(c, 2);
    a[2] = arg32(c, 3);
    a[3] = arg32(c, 4);
    a[4] = arg32(c, 5);
    a[5] = arg32(c, 6);
    ret(c, 0);
}
// Registered display buffers (handle, startIndex, void** addresses, bufferNum, attr*).
struct DisplayBuffers {
    std::vector<uint64_t> address;
    uint32_t width = 0, height = 0, pitch = 0, tiling = 0, format = 0;
} g_display;

void vo_register_buffers(Context& c) {
    std::lock_guard lk(vo().m);
    const uint64_t start = arg32(c, 1), n = arg32(c, 3);  // int parameters
    if (g_display.address.size() < start + n) g_display.address.resize(start + n);
    for (uint64_t i = 0; i < n; ++i) g_display.address[start + i] = rt::ld<uint64_t>(c, arg(c, 2) + 8 * i);
    if (arg(c, 4)) {
        g_display.format = rt::ld<uint32_t>(c, arg(c, 4));
        g_display.tiling = rt::ld<uint32_t>(c, arg(c, 4) + 4);
        g_display.width = rt::ld<uint32_t>(c, arg(c, 4) + 12);
        g_display.height = rt::ld<uint32_t>(c, arg(c, 4) + 16);
        g_display.pitch = rt::ld<uint32_t>(c, arg(c, 4) + 20);
    }
    if (std::getenv("BB_FLIP_STATS"))
        std::fprintf(stderr, "VideoOut: registered %llu buffers from index %llu: %ux%u pitch %u tiling %u format 0x%x, first at 0x%llx\n",
                     (unsigned long long)n, (unsigned long long)start, g_display.width, g_display.height, g_display.pitch,
                     g_display.tiling, g_display.format, (unsigned long long)g_display.address[start]);
    ret(c, 0);
}

void vo_add_flip_event(Context& c) {  // (eq, handle, udata)
    std::lock_guard lk(vo().m);
    vo().flip_listeners.emplace_back(reinterpret_cast<Equeue*>(arg(c, 0)), arg(c, 2));
    ret(c, 0);
}
FlipHook g_flip_hook = nullptr;

void vo_submit_flip(Context& c) {  // (handle, bufferIndex, flipMode, flipArg)
    std::vector<std::pair<Equeue*, uint64_t>> targets;
    uint64_t count;
    {
        std::lock_guard lk(vo().m);
        vo().flip_arg = arg(c, 3);
        count = ++vo().flip_count;
        targets = vo().flip_listeners;
    }
    if (static const bool flog = std::getenv("BB_FLIP_LOG") != nullptr; flog) {  // flip timestamps only (BB_FLIP_STATS also scans the buffer)
        static const auto t0 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "flip %llu @%.3f\n", (unsigned long long)count, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    if (std::getenv("BB_FLIP_STATS")) {  // does the display buffer contain anything? (nothing draws into it yet)
        std::lock_guard lk(vo().m);
        const uint64_t idx = arg32(c, 1);  // int argument: the upper register half is garbage
        if (idx < g_display.address.size() && g_display.address[idx]) {
            const uint32_t* px = ptr<const uint32_t>(c, g_display.address[idx]);
            const uint64_t total = std::min<uint64_t>(uint64_t(g_display.pitch) * g_display.height, 1u << 20);
            uint64_t nonzero = 0;
            for (uint64_t i = 0; i < total; ++i) nonzero += px[i] != 0;
            static const auto t0 = std::chrono::steady_clock::now();
            std::fprintf(stderr, "flip %llu @%.3f: buffer %llu nonzero words %llu/%llu\n", (unsigned long long)count, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), (unsigned long long)idx,
                         (unsigned long long)nonzero, (unsigned long long)total);
        } else {
            std::fprintf(stderr, "flip %llu: buffer index %llu (handle %llx, mode %llu, arg %llx) has no registered display buffer\n", (unsigned long long)count,
                         (unsigned long long)idx, (unsigned long long)arg(c, 0), (unsigned long long)arg(c, 2), (unsigned long long)arg(c, 3));
        }
    }
    if (g_flip_hook) {
        std::lock_guard lk(vo().m);
        const uint64_t idx = arg32(c, 1);
        g_flip_hook(uint32_t(idx), idx < g_display.address.size() ? g_display.address[idx] : 0, g_display.width, g_display.height);
    }
    // The engine paces itself to 30 fps (ROADMAP 44/45). A flip completes no earlier than one refresh interval (60/(rate+1) Hz) after the
    // previous one; a late flip restarts the cadence instead of catching up. BB_FLIP_PACE=0 lets flips complete immediately (diagnosis).
    static const bool pace = !(std::getenv("BB_FLIP_PACE") && std::getenv("BB_FLIP_PACE")[0] == '0');
    if (pace) {
        using namespace std::chrono;
        std::chrono::steady_clock::time_point due;
        {
            std::lock_guard lk(vo().m);
            due = vo().last_flip + microseconds(16667 * (vo().flip_rate + 1));
        }
        const auto now = steady_clock::now();
        if (due > now) std::this_thread::sleep_until(due);
        std::lock_guard lk(vo().m);
        vo().last_flip = due > now ? due : now;
    }
    for (auto& [q, udata] : targets) {
        {
            std::lock_guard lk(q->m);
            q->events.push_back({1, kEvfiltVideoOut, 0x20, 0, int64_t(count), udata});
        }
        q->cv.notify_all();
    }
    ret(c, 0);
}
void vo_flip_status(Context& c) {  // SceVideoOutFlipStatus: count, processTime, tsc, flipArg, gcQueueNum, flipPendingNum, ...
    std::memset(ptr(c, arg(c, 1)), 0, 48);
    std::lock_guard lk(vo().m);
    rt::st<uint64_t>(c, arg(c, 1), vo().flip_count);
    rt::st<int64_t>(c, arg(c, 1) + 24, int64_t(vo().flip_arg));
    ret(c, 0);
}

} // namespace

void register_gnm() {
    reg("sceKernelCreateEqueue", eq_create);
    reg("sceKernelDeleteEqueue", eq_delete);
    reg("sceKernelWaitEqueue", eq_wait);
    reg("sceVideoOutOpen", gnm_one);
    reg("sceVideoOutClose", gnm_zero);
    reg("sceVideoOutSetFlipRate", [](Context& c) {  // (handle, rate)
        std::lock_guard lk(vo().m);
        vo().flip_rate = std::min<uint32_t>(arg32(c, 1), 2);
        if (std::getenv("BB_FLIP_STATS")) std::fprintf(stderr, "VideoOut: flip rate %u\n", vo().flip_rate);
        ret(c, 0);
    });
    reg("sceVideoOutSetBufferAttribute", vo_attr);
    reg("sceVideoOutRegisterBuffers", vo_register_buffers);
    reg("sceVideoOutAddFlipEvent", vo_add_flip_event);
    reg("sceVideoOutSubmitFlip", vo_submit_flip);
    reg("sceVideoOutGetFlipStatus", vo_flip_status);
    reg("sceGnmSetCsShader", set_cs_shader);
    reg("sceGnmSetPsShader", set_ps_shader);
    reg("sceGnmSetVsShader", set_vs_shader);
    reg("sceGnmSetLsShader", [](Context& c) { set_tess_shader(c, 1); });
    reg("sceGnmSetHsShader", [](Context& c) { set_tess_shader(c, 2); });
    for (const char* s : {"sceGnmSetEsShader", "sceGnmSetGsShader"}) reg(s, fill_reserved_nop);
    // Update*Shader: same packet length as Set*; the context registers go into NOPs because the engine only calls them when those
    // registers equal the bound shader's (f_10900a0 compares first). Emitting real SET_CONTEXT_REG with the same values is equivalent.
    reg("sceGnmUpdatePsShader", set_ps_shader);
    reg("sceGnmUpdateVsShader", set_vs_shader);
    reg("sceGnmUpdateHsShader", [](Context& c) { set_tess_shader(c, 2); });
    reg("sceGnmUpdateGsShader", fill_reserved_nop);
    reg("sceGnmSubmitCommandBuffers", [](Context& c) { submit_scan(c); ret(c, 0); });  // (count, dcb**, dcbSizes*, ccb**, ccbSizes*): PM4 interpreter comes with M2
    reg("sceGnmSubmitDone", gnm_zero);
    reg("sceGnmSubmitAndFlipCommandBuffers", [](Context& c) {  // (count, dcb**, dcbSz*, ccb**, ccbSz*, handle, bufIdx, mode, arg)
        submit_scan(c);
        c.r[7] = arg(c, 5);
        c.r[6] = arg(c, 6);
        c.r[2] = arg(c, 7);
        c.r[1] = arg(c, 8);
        vo_submit_flip(c);
    });
    reg("sceGnmGetTheTessellationFactorRingBufferBaseAddress", tess_ring);
    reg("sceGnmValidateCommandBuffers", gnm_zero);
    reg("sceGnmAreSubmitsAllowed", gnm_one);
    // Returns the virtual queue id (>= 1, 0 means failure to the engine; verified: with 0 it tears the device down).
    reg("sceGnmMapComputeQueue", map_queue);
    reg("sceGnmMapComputeQueueWithPriority", map_queue);
    reg("sceGnmDebugHardwareStatus", gnm_zero);
    reg("sceGnmDriverCaptureInProgress", gnm_zero);
    reg("sceGnmDriverTriggerCapture", gnm_zero);
    reg("sceGnmIsUserPaEnabled", gnm_zero);
    reg("sceGnmDisableMipStatsReport", gnm_zero);
    reg("sceGnmRequestMipStatsReportAndReset", gnm_zero);
    reg("sceGnmFlushGarlic", gnm_zero);
    reg("sceGnmDingDong", ding_dong);
    reg("sceGnmAddEqEvent", gnm_zero);
    reg("sceGnmDeleteEqEvent", gnm_zero);
    reg("sceGnmLogicalCuIndexToPhysicalCuIndex", [](Context& c) { ret(c, arg(c, 0) & 0xFFFFFFFF); });
    reg("sceGnmLogicalCuMaskToPhysicalCuMask", [](Context& c) { ret(c, arg(c, 0) & 0xFFFFFFFF); });
    reg("sceGnmDrawInitDefaultHardwareState200", [](Context& c) { ret(c, fill_nop(c, arg(c, 0), arg(c, 1), 0x100)); });
    reg("sceGnmDispatchInitDefaultHardwareState", [](Context& c) { ret(c, fill_nop(c, arg(c, 0), arg(c, 1), 0x80)); });
    // Debug markers: the guest reserves (cmdbuf, dwords) and expects the call to fill it; one NOP covers the space, text is dropped.
    for (const char* n : {"sceGnmInsertPushMarker", "sceGnmInsertPopMarker", "sceGnmInsertPushColorMarker", "sceGnmInsertSetMarker",
                          "sceGnmInsertSetColorMarker"})
        reg(n, fill_reserved_nop);
    reg("sceGnmInsertWaitFlipDone", gnm_zero);
}

void set_flip_hook(FlipHook hook) { g_flip_hook = hook; }

void install_gpu_mem_hook() {  // guest ranges the GPU backend may read: every byte must lie in a noted mapping
    gpu::hooks().mem_valid = [](uint64_t addr, uint64_t size) {
        Mapping m;
        for (uint64_t at = addr, end = addr + size; at < end;) {
            if (!find_mapping(at, m) || m.end <= at) return false;
            at = m.end;
        }
        return true;
    };
}

void print_gnm_stats() {
    std::lock_guard lk(g_pm4_mutex);
    std::fprintf(stderr, "PM4: %llu buffers, %llu dwords (%llu zero), type0=%llu, bad=%llu, overrun=%llu; type-3 opcodes:", (unsigned long long)g_pm4_submits,
                 (unsigned long long)g_pm4_dwords, (unsigned long long)g_pm4_zero, (unsigned long long)g_pm4_type0, (unsigned long long)g_pm4_bad, (unsigned long long)g_pm4_overrun);
    for (auto& [op, n] : g_pm4_opcodes) std::fprintf(stderr, " 0x%x=%llu", op, (unsigned long long)n);
    std::fputc('\n', stderr);
}

} // namespace bb::hle
