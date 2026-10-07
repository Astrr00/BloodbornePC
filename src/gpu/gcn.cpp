#include "gpu/gcn.h"

#include <cstdio>
#include <cstring>

namespace bb::gcn {
namespace {

struct OpRow { GcnFmt fmt; unsigned op; const char* name; };
const OpRow kOps[] = {
#include "gpu/gcn_ops.inc"
};

// SMRD (GFX6/7) is not in ACO (it models SMEM): s_load_* / s_buffer_load_*.
const char* smrd_name(unsigned op) {
    static const char* const n[] = {"s_load_dword", "s_load_dwordx2", "s_load_dwordx4", "s_load_dwordx8",
                                    "s_load_dwordx16", nullptr, nullptr, nullptr, "s_buffer_load_dword",
                                    "s_buffer_load_dwordx2", "s_buffer_load_dwordx4", "s_buffer_load_dwordx8",
                                    "s_buffer_load_dwordx16"};
    if (op < sizeof n / sizeof *n) return n[op];
    return op == 0x1d ? "s_dcache_inv_vol" : op == 0x1f ? "s_dcache_inv" : nullptr;
}

}  // namespace

// VOP2 opcodes the ACO tables omit for GFX6/7 (they moved to VOP3-only on GFX8); numbering from the Sea Islands ISA.
const OpRow kVop2Extra[] = {
    {GcnFmt::VOP2, 0x1e, "v_bfm_b32"},          {GcnFmt::VOP2, 0x22, "v_bcnt_u32_b32"},       {GcnFmt::VOP2, 0x23, "v_mbcnt_lo_u32_b32"},
    {GcnFmt::VOP2, 0x2b, "v_ldexp_f32"},        {GcnFmt::VOP2, 0x2c, "v_cvt_pkaccum_u8_f32"}, {GcnFmt::VOP2, 0x2d, "v_cvt_pknorm_i16_f32"},
    {GcnFmt::VOP2, 0x2e, "v_cvt_pknorm_u16_f32"}, {GcnFmt::VOP2, 0x30, "v_cvt_pk_u16_u32"},   {GcnFmt::VOP2, 0x31, "v_cvt_pk_i16_i32"},
};

const char* mnemonic(GcnFmt fmt, unsigned op) {
    if (fmt == GcnFmt::SMRD) return smrd_name(op);
    if (fmt == GcnFmt::VOP2)
        for (const OpRow& r : kVop2Extra)
            if (r.op == op) return r.name;
    for (const OpRow& r : kOps)
        if (r.fmt == fmt && r.op == op) return r.name;
    return nullptr;
}

Insn decode(const uint32_t* w, size_t avail) {
    Insn i;
    if (!avail) return i;
    const uint32_t d = w[0];
    i.w[0] = d;
    auto need2 = [&] {
        if (avail < 2) return false;
        i.w[1] = w[1];
        i.len = 2;
        return true;
    };
    if (d >> 31 == 0) {  // VOP2 / VOP1 / VOPC / (VOP3 and others have bit31=1 or 0b110)
        const unsigned top7 = d >> 25;
        if (top7 == 0x3F) { i.fmt = GcnFmt::VOP1; i.op = (d >> 9) & 0xFF; }
        else if (top7 == 0x3E) { i.fmt = GcnFmt::VOPC; i.op = (d >> 17) & 0xFF; }
        else { i.fmt = GcnFmt::VOP2; i.op = (d >> 25) & 0x3F; }
        // v_madmk_f32 (0x20) / v_madak_f32 (0x21) always carry the constant K in the following dword
        const bool implicit_k = i.fmt == GcnFmt::VOP2 && (i.op == 0x20 || i.op == 0x21);
        if (((d & 0x1FF) == 255 || implicit_k) && avail >= 2) { i.has_literal = true; i.literal = w[1]; i.len = 2; }
        return i;
    }
    const unsigned t6 = d >> 26;
    if ((d >> 30) == 2) {  // 10xxxxxx: SOP2 / SOPK / SOP1 / SOPC / SOPP
        const unsigned t9 = d >> 23;
        unsigned src0 = 0;
        if (t9 == 0x17D) { i.fmt = GcnFmt::SOP1; i.op = (d >> 8) & 0xFF; src0 = d & 0xFF; }
        else if (t9 == 0x17E) { i.fmt = GcnFmt::SOPC; i.op = (d >> 16) & 0x7F; src0 = d & 0xFF; }
        else if (t9 == 0x17F) { i.fmt = GcnFmt::SOPP; i.op = (d >> 16) & 0x7F; return i; }
        else if ((d >> 28) == 0xB) { i.fmt = GcnFmt::SOPK; i.op = (d >> 23) & 0x1F; return i; }
        else { i.fmt = GcnFmt::SOP2; i.op = (d >> 23) & 0x7F; src0 = d & 0xFF; }
        const unsigned s1 = (d >> 8) & 0xFF;
        if ((src0 == 255 || (i.fmt != GcnFmt::SOP1 && s1 == 255)) && avail >= 2) {
            i.has_literal = true; i.literal = w[1]; i.len = 2;
        }
        return i;
    }
    switch (t6) {
        case 0x32: i.fmt = GcnFmt::VINTRP; i.op = (d >> 16) & 3; return i;                       // 110010
        case 0x34: need2(); i.fmt = GcnFmt::VOP3; i.op = (d >> 17) & 0x1FF; return i;            // 110100
        case 0x36: need2(); i.fmt = GcnFmt::DS; i.op = (d >> 18) & 0xFF; return i;               // 110110
        case 0x38: need2(); i.fmt = GcnFmt::MUBUF; i.op = (d >> 18) & 0x7F; return i;            // 111000
        case 0x3A: need2(); i.fmt = GcnFmt::MTBUF; i.op = (d >> 16) & 7; return i;               // 111010
        case 0x3C: need2(); i.fmt = GcnFmt::MIMG; i.op = (d >> 18) & 0x7F; return i;             // 111100
        case 0x3E: need2(); i.fmt = GcnFmt::EXP; return i;                                       // 111110
        default: break;
    }
    if ((d >> 27) == 0x18) {  // 11000
        i.fmt = GcnFmt::SMRD; i.op = (d >> 22) & 0x1F;
        if (!(d & 0x100) && (d & 0xFF) == 0xFF && avail >= 2) { i.has_literal = true; i.literal = w[1]; i.len = 2; }  // 32-bit dword offset follows
        return i;
    }
    return i;
}

namespace {

std::string sreg(unsigned n, unsigned cnt) {
    char b[32];
    if (cnt <= 1) std::snprintf(b, sizeof b, "s%u", n);
    else std::snprintf(b, sizeof b, "s[%u:%u]", n, n + cnt - 1);
    return b;
}

std::string vreg(unsigned n, unsigned cnt) {
    char b[32];
    if (cnt <= 1) std::snprintf(b, sizeof b, "v%u", n);
    else std::snprintf(b, sizeof b, "v[%u:%u]", n, n + cnt - 1);
    return b;
}

// Generic source operand (9 bits: SGPR/special/inline const/VGPR). cnt = register count for pairs.
std::string src(unsigned s, const Insn& i, unsigned cnt = 1) {
    char b[48];
    if (s < 104) return sreg(s, cnt);
    if (s >= 256) return vreg(s - 256, cnt);
    if (s >= 129 && s <= 192) { std::snprintf(b, sizeof b, "%d", int(s) - 128); return b; }
    if (s >= 193 && s <= 208) { std::snprintf(b, sizeof b, "-%d", int(s) - 192); return b; }
    switch (s) {
        case 104: return cnt > 1 ? "flat_scratch" : "flat_scratch_lo";
        case 105: return "flat_scratch_hi";
        case 106: return cnt > 1 ? "vcc" : "vcc_lo";
        case 107: return "vcc_hi";
        case 124: return "m0";
        case 126: return cnt > 1 ? "exec" : "exec_lo";
        case 127: return "exec_hi";
        case 128: return "0";
        case 240: return "0.5";
        case 241: return "-0.5";
        case 242: return "1.0";
        case 243: return "-1.0";
        case 244: return "2.0";
        case 245: return "-2.0";
        case 246: return "4.0";
        case 247: return "-4.0";
        case 251: return "vccz";
        case 252: return "execz";
        case 253: return "scc";
        case 255:
            if (i.has_literal) { std::snprintf(b, sizeof b, "0x%x", i.literal); return b; }
            return "lit?";
        default: std::snprintf(b, sizeof b, "src%u", s); return b;
    }
}

// Register count of the destination/sources from the mnemonic suffix (b64/f64/u64/i64, x2/x4/x8).
unsigned width_of(const char* m, bool dst) {
    if (!m) return 1;
    const size_t n = std::strlen(m);
    auto ends = [&](const char* s) { size_t k = std::strlen(s); return n >= k && !std::strcmp(m + n - k, s); };
    if (ends("x2")) return 2;
    if (ends("x4")) return 4;
    if (ends("x8")) return 8;
    if (ends("x16")) return 16;
    if (ends("64")) return (dst && (std::strstr(m, "cmp") || std::strstr(m, "bcnt"))) ? 1 : 2;
    return 1;
}

}  // namespace

std::string disasm(const Insn& i) {
    char b[256];
    const uint32_t d = i.w[0], e = i.w[1];
    const char* m = mnemonic(i.fmt, i.op);
    char fallback[24];
    auto name = [&](GcnFmt f, unsigned op, unsigned bias = 0) {
        const char* r = mnemonic(f, op);
        if (r) return r;
        (void)bias;
        std::snprintf(fallback, sizeof fallback, "op_%u_0x%x", unsigned(f), op);
        return static_cast<const char*>(fallback);
    };
    switch (i.fmt) {
        case GcnFmt::SOP1: {
            const unsigned w_ = width_of(m, true);
            std::snprintf(b, sizeof b, "%s %s, %s", name(i.fmt, i.op), src((d >> 16) & 0x7F, i, w_).c_str(),
                          src(d & 0xFF, i, width_of(m, false)).c_str());
            break;
        }
        case GcnFmt::SOP2:
            std::snprintf(b, sizeof b, "%s %s, %s, %s", name(i.fmt, i.op), src((d >> 16) & 0x7F, i, width_of(m, true)).c_str(),
                          src(d & 0xFF, i, width_of(m, false)).c_str(), src((d >> 8) & 0xFF, i, width_of(m, false)).c_str());
            break;
        case GcnFmt::SOPC:
            std::snprintf(b, sizeof b, "%s %s, %s", name(i.fmt, i.op), src(d & 0xFF, i, width_of(m, false)).c_str(),
                          src((d >> 8) & 0xFF, i, width_of(m, false)).c_str());
            break;
        case GcnFmt::SOPK:
            std::snprintf(b, sizeof b, "%s %s, 0x%x", name(i.fmt, i.op), src((d >> 16) & 0x7F, i).c_str(), d & 0xFFFF);
            break;
        case GcnFmt::SOPP:
            std::snprintf(b, sizeof b, "%s 0x%x", name(i.fmt, i.op), d & 0xFFFF);
            break;
        case GcnFmt::SMRD: {
            const unsigned sd = (d >> 15) & 0x7F, base = ((d >> 9) & 0x3F) * 2;
            const unsigned cnt = i.op & 7 ? (1u << ((i.op & 7) - 0)) : 1;  // 0:1 1:2 2:4 3:8 4:16
            const unsigned regs = (i.op & 7) == 0 ? 1 : (i.op & 7) == 1 ? 2 : (i.op & 7) == 2 ? 4 : (i.op & 7) == 3 ? 8 : 16;
            (void)cnt;
            const bool buf = i.op >= 8;
            if (d & 0x100) std::snprintf(b, sizeof b, "%s %s, %s, 0x%x", name(i.fmt, i.op), sreg(sd, regs).c_str(),
                                         sreg(base, buf ? 4 : 2).c_str(), d & 0xFF);
            else if (i.has_literal) std::snprintf(b, sizeof b, "%s %s, %s, 0x%x", name(i.fmt, i.op), sreg(sd, regs).c_str(),
                                                  sreg(base, buf ? 4 : 2).c_str(), i.literal);
            else std::snprintf(b, sizeof b, "%s %s, %s, %s", name(i.fmt, i.op), sreg(sd, regs).c_str(),
                               sreg(base, buf ? 4 : 2).c_str(), sreg(d & 0xFF, 1).c_str());
            break;
        }
        case GcnFmt::VOP1:
            std::snprintf(b, sizeof b, "%s v%u, %s", name(i.fmt, i.op), (d >> 17) & 0xFF, src(d & 0x1FF, i).c_str());
            break;
        case GcnFmt::VOP2:
            std::snprintf(b, sizeof b, "%s v%u, %s, v%u", name(i.fmt, i.op), (d >> 17) & 0xFF, src(d & 0x1FF, i).c_str(), (d >> 9) & 0xFF);
            break;
        case GcnFmt::VOPC:
            std::snprintf(b, sizeof b, "%s vcc, %s, v%u", name(i.fmt, i.op), src(d & 0x1FF, i).c_str(), (d >> 9) & 0xFF);
            break;
        case GcnFmt::VINTRP: {
            static const char* const n[] = {"v_interp_p1_f32", "v_interp_p2_f32", "v_interp_mov_f32", "?"};
            std::snprintf(b, sizeof b, "%s v%u, v%u, attr%u.%c", n[i.op & 3], (d >> 18) & 0xFF, d & 0xFF, (d >> 10) & 0x3F, "xyzw"[(d >> 8) & 3]);
            break;
        }
        case GcnFmt::VOP3: {
            unsigned op = i.op;
            const char* mn = nullptr;
            const char* kind = "";
            if (op < 0x100) { mn = mnemonic(GcnFmt::VOPC, op); kind = "c"; }
            else if (op < 0x140) { mn = mnemonic(GcnFmt::VOP2, op - 0x100); kind = "2"; }
            else if (op >= 0x180 && op < 0x200) { mn = mnemonic(GcnFmt::VOP1, op - 0x180); kind = "1"; }
            else mn = mnemonic(GcnFmt::VOP3, op);
            char nb[40];
            if (!mn) { std::snprintf(nb, sizeof nb, "vop3_0x%x", op); mn = nb; }
            std::snprintf(b, sizeof b, "%s_e64 v%u, %s, %s, %s%s%s%s", mn, d & 0xFF, src(e & 0x1FF, i).c_str(),
                          src((e >> 9) & 0x1FF, i).c_str(), src((e >> 18) & 0x1FF, i).c_str(),
                          (d & 0x800) ? " clamp" : "", (e >> 27 & 3) ? " omod" : "", (d >> 8 & 7) || (e >> 29) ? " abs/neg" : "");
            (void)kind;
            break;
        }
        case GcnFmt::DS:
            std::snprintf(b, sizeof b, "%s vdst=v%u addr=v%u d0=v%u d1=v%u off=%u,%u%s", name(i.fmt, i.op), e >> 24, e & 0xFF,
                          (e >> 8) & 0xFF, (e >> 16) & 0xFF, d & 0xFF, (d >> 8) & 0xFF, (d & 0x20000) ? " gds" : "");
            break;
        case GcnFmt::MUBUF:
            std::snprintf(b, sizeof b, "%s v%u, v%u, s[%u:%u], %s offset:%u%s%s%s", name(i.fmt, i.op), (e >> 8) & 0xFF, e & 0xFF,
                          ((e >> 16) & 0x1F) * 4, ((e >> 16) & 0x1F) * 4 + 3, src(e >> 24, i).c_str(), d & 0xFFF,
                          (d & 0x1000) ? " offen" : "", (d & 0x2000) ? " idxen" : "", (d & 0x4000) ? " glc" : "");
            break;
        case GcnFmt::MTBUF:
            std::snprintf(b, sizeof b, "%s v%u, v%u, s[%u:%u], %s offset:%u dfmt:%u nfmt:%u%s%s", name(i.fmt, i.op), (e >> 8) & 0xFF, e & 0xFF,
                          ((e >> 16) & 0x1F) * 4, ((e >> 16) & 0x1F) * 4 + 3, src(e >> 24, i).c_str(), d & 0xFFF, (d >> 19) & 15,
                          (d >> 23) & 7, (d & 0x1000) ? " offen" : "", (d & 0x2000) ? " idxen" : "");
            break;
        case GcnFmt::MIMG:
            std::snprintf(b, sizeof b, "%s v[%u..], v[%u..], s[%u:%u], s[%u:%u] dmask:0x%x%s", name(i.fmt, i.op), (e >> 8) & 0xFF, e & 0xFF,
                          ((e >> 16) & 0x1F) * 4, ((e >> 16) & 0x1F) * 4 + 7, ((e >> 21) & 0x1F) * 4, ((e >> 21) & 0x1F) * 4 + 3,  // SRSRC/SSAMP: units of 4 SGPRs
                          (d >> 8) & 0xF, (d & 0x8000) ? " r128" : "");
            break;
        case GcnFmt::EXP: {
            const unsigned tgt = (d >> 4) & 0x3F;
            char t[16];
            if (tgt < 8) std::snprintf(t, sizeof t, "mrt%u", tgt);
            else if (tgt == 8) std::snprintf(t, sizeof t, "mrtz");
            else if (tgt == 9) std::snprintf(t, sizeof t, "null");
            else if (tgt >= 12 && tgt < 16) std::snprintf(t, sizeof t, "pos%u", tgt - 12);
            else if (tgt >= 32 && tgt < 64) std::snprintf(t, sizeof t, "param%u", tgt - 32);
            else std::snprintf(t, sizeof t, "tgt%u", tgt);
            std::snprintf(b, sizeof b, "exp %s v%u, v%u, v%u, v%u en:0x%x%s%s%s", t, e & 0xFF, (e >> 8) & 0xFF, (e >> 16) & 0xFF, e >> 24,
                          d & 15, (d & 0x400) ? " compr" : "", (d & 0x800) ? " done" : "", (d & 0x1000) ? " vm" : "");
            break;
        }
        default:
            std::snprintf(b, sizeof b, ".word 0x%08x", d);
    }
    return b;
}

bool shader_info(const uint32_t* w, size_t max_dw, ShaderInfo& out) {
    static const char sig[7] = {'O', 'r', 'b', 'S', 'h', 'd', 'r'};
    for (size_t k = 0; k + 5 <= max_dw; ++k)
        if (!std::memcmp(w + k, sig, 7)) {
            out.trailer_dw = k;
            out.code_bytes = w[k + 2] >> 8;
            out.type = (w[k + 2] >> 2) & 0xF;
            out.crc = w[k + 4];
            return true;
        }
    return false;
}

}  // namespace bb::gcn
