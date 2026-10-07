// SPDX-License-Identifier: GPL-3.0-or-later
// GCN -> SPIR-V translation; see gcn_spirv.h for the model.
#include "gpu/gcn_spirv.h"

#include <algorithm>
#include <bit>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>

#include <spirv/unified1/GLSL.std.450.h>

#include "gpu/gcn.h"
#include "gpu/spirv_builder.h"

namespace bb::gpu {
namespace {

using Id = SpvBuilder::Id;
using gcn::GcnFmt;
using gcn::Insn;
using spv::Op;

struct Lane { Id var = 0; bool kt = false; };  // per-invocation bool standing for a 64-bit lane mask
struct Sym { bool known = false; ScalarVal v; };

constexpr unsigned kVcc = 106, kM0 = 124, kExec = 126;

bool ends_with(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && !s.compare(s.size() - n, n, suf);
}
bool starts_with(const std::string& s, const char* pre) { return !s.compare(0, std::strlen(pre), pre); }

class Xlat {
public:
    Xlat(const uint32_t* code, size_t n, const GcnEnv& env, Translation& t) : code_(code), n_(n), env_(env), T(t) {}
    bool run();
    std::vector<uint32_t> finish() const { return b.finish(); }
    const std::string& error() const { return err_; }

private:
    // ---- plumbing -----------------------------------------------------------------------------------------------------
    bool fail(const std::string& why, const Insn* in = nullptr) {
        if (err_.empty()) err_ = why + (in ? "  [" + gcn::disasm(*in) + "]" : "");
        return false;
    }
    Id op(Op o, Id t, std::vector<uint32_t> a) { return b.op(o, t, a); }
    Id bits(Id f) { return op(Op::OpBitcast, U, {f}); }
    Id flt(Id u) { return op(Op::OpBitcast, F, {u}); }
    Id ld(Id var, Id t) { return op(Op::OpLoad, t, {var}); }
    void st(Id var, Id v) { b.op0(Op::OpStore, {var, v}); }
    Id glsl(Id t, GLSLstd450 i, std::vector<uint32_t> a) { return b.glsl(t, uint32_t(i), a); }
    Id cu(uint32_t v) { return b.c_u32(v); }
    Id cf(float v) { return b.c_f32(v); }

    Id sgpr_var(unsigned i) {
        if (!sg_[i]) sg_[i] = b.local_var_init(U, cu(0));
        return sg_[i];
    }
    Id vgpr_var(unsigned i) {
        if (!vg_[i]) vg_[i] = b.local_var_init(U, cu(0));
        return vg_[i];
    }
    Lane& lane(unsigned s) {
        auto it = lm_.find(s);
        if (it == lm_.end()) {
            Lane l;
            l.var = b.local_var_init(B, b.c_bool(false));
            it = lm_.emplace(s, l).first;
        }
        return it->second;
    }
    bool exec_kt() { return lane(kExec).kt; }

    Id rd_s(unsigned s, const Insn& in, bool is_float, Sym* sym, bool* ok);
    int get_load_resource(uint32_t load);
    Id load_word(const ScalarVal& v, const Insn& in, bool* ok);
    std::map<std::pair<unsigned, uint32_t>, Id> spill_;
    std::map<std::pair<unsigned, uint32_t>, Sym> spill_sym_;  // what is known about a spilled SGPR (a descriptor table pointer stays usable after the fill)
    Id rd_v(unsigned s9, const Insn& in, bool is_float, bool* ok);
    void wr_s(unsigned i, Id v, const Sym& sym = {});
    void wr_v(unsigned i, Id v);
    Id rd_lane(unsigned s, const Insn& in, bool* ok);
    void wr_lane(unsigned s, Id v, bool kt = false);
    Id builtin_var(spv::StorageClass sc, Id type, spv::BuiltIn bi);
    Id lane_index();  // this invocation's GCN lane id (0..63)
    Id exec_bits();   // this lane's exec mask dword
    Id lds_var();     // Workgroup array holding the LDS, 0 if the shader has none
    Id sf_unpack(Id raw, unsigned w);
    Id sf_pack(Id f, unsigned w);
    bool is_ls() const { return env_.tess_ls != 0; }
    bool is_ds() const { return env_.tess_ds_level != 0; }
    bool tess() const { return is_ls() || is_ds(); }
    Id tess_var(int slot);  // 0: TessIndices, 1: LdsBuffer, 2: GdsBuffer (described at the end of the translation)
    bool finish_tess();

    struct Pred { Id merge = 0; };
    Pred pred_begin() {
        if (exec_kt()) return {};
        Id cond = ld(lane(kExec).var, B), then_l = b.label(), merge = b.label();
        b.op0(Op::OpSelectionMerge, {merge, 0});
        b.op0(Op::OpBranchConditional, {cond, then_l, merge});
        b.place_label(then_l);
        return {merge};
    }
    void pred_end(Pred p) {
        if (!p.merge) return;
        b.op0(Op::OpBranch, {p.merge});
        b.place_label(p.merge);
    }

    bool desc_ref(unsigned sfirst, unsigned cnt, DescRef& out);
    uint32_t eval_sval(const ScalarVal& v) const;
    int get_resource(Resource::Type type, const DescRef& ref, unsigned dwords, bool write);
    bool read_desc_words(const DescRef& ref, unsigned dwords, uint32_t* out);
    Id buffer_var(int res);
    Id image_var(int res);
    Id sampler_var(int res);
    Id image_type_of(const Resource& r, bool storage);
    Id push_user(unsigned i);
    bool copy_s(unsigned dst, unsigned src, const Insn& in) {  // register copy that keeps symbolic (descriptor) values
        if (src < 104 && sym_[src].known && sym_[src].v.kind == ScalarVal::LoadWord) { sym_[dst] = sym_[src]; return true; }
        bool ok = true;
        Sym sy;
        Id v = rd_s(src, in, false, &sy, &ok);
        if (!ok) return false;
        wr_s(dst, v, sy);
        return true;
    }

    bool step(const Insn& in);
    bool salu(const Insn& in, const std::string& m);
    bool smrd(const Insn& in, const std::string& m);
    bool valu(const Insn& in);
    bool mubuf(const Insn& in, const std::string& m);
    bool mimg(const Insn& in, const std::string& m);
    bool vintrp(const Insn& in);
    bool ds(const Insn& in);
    bool lds(const Insn& in);
    bool gds(const Insn& in);
    bool exp(const Insn& in);
    bool setup();
    void setup_ls();
    void setup_ds();

    const uint32_t* code_;
    size_t n_;
    const GcnEnv& env_;
    Translation& T;
    SpvBuilder b;
    std::string err_;

    Id U = 0, F = 0, B = 0, I = 0, V4F = 0, V2F = 0, UV3 = 0, PU = 0;
    Id sg_[128] = {}, vg_[256] = {};
    Sym sym_[128];
    std::map<unsigned, Lane> lm_;
    Id scc_ = 0;
    Id push_var_ = 0;
    std::vector<Id> iface_;
    Id frag_coord_ = 0, front_face_ = 0, pos_out_ = 0, psize_out_ = 0, clip_out_ = 0, cull_out_ = 0, depth_out_ = 0, bary_ = 0;
    Id lds_ = 0, li_ptr_ = 0, sub_ptr_ = 0;
    // GcnEnv::res_scale > 1: the push-constant scale bits, "the targets are scaled" (PS) and the fragment's sub-pixel (int(FragCoord) mod s
    // per axis, 0 without scaled targets), set up in the entry block (setup)
    Id sc_info_ = 0, sc_t_ = 0, sc_sub_[2] = {0, 0};
    Id tess_ptr_[3] = {};
    bool lds_written_ = false;
    uint64_t flat_attrs_ = 0, interp_attrs_ = 0;  // PS attributes read by v_interp_mov (Flat) / by v_interp_p1
    std::map<unsigned, Id> attr_in_, param_out_, mrt_out_;
    std::map<std::tuple<int, int, uint32_t>, int> res_index_;
    std::map<int, Id> res_var_;
    std::vector<std::vector<uint32_t>> load_vals_;
    size_t pc_ = 0, skip_until_ = 0;
    // sym_reads: SGPRs whose symbolic (descriptor / s_load base) value the body used; such a register must not change inside the loop
    struct Loop { size_t back_pc; Id header, body, cont, merge; Sym syms[128]; std::bitset<128> sym_reads; };
    void note_sym_read(unsigned first, unsigned cnt) { if (!loops_.empty()) for (unsigned k = first; k < first + cnt && k < 128; ++k) loops_.back().sym_reads.set(k); }
    std::map<size_t, std::vector<size_t>> loop_heads_;  // header pc -> pcs of the backward branches targeting it
    std::vector<Loop> loops_;
    void scan_loops();
    bool begin_loop();
    bool loop_cond(const Insn& in, const std::string& m, Id* taken);
    bool end_loop(const Insn& in, const std::string& m);
    bool break_loop(const Insn& in, const std::string& m, size_t target);
    // Forward s_cbranch_scc0/1: a structured selection (SCC is wave-uniform, so the region really is skipped - scalar writes in it must not
    // happen otherwise). `alt`: the not-taken path's first block (no else-region: an empty block placed when the selection closes); after an
    // if/else switch (see s_branch) the then-path's deferred exit block. `syms`/`kts`: compile-time register knowledge on entry to the
    // other path (then: at the branch; after the switch: at the end of the then-region).
    struct Join { size_t pc; Id merge, alt; Sym syms[128]; std::map<unsigned, bool> kts; bool els = false; };
    std::vector<Join> joins_;
    std::map<unsigned, bool> lane_kts() const {
        std::map<unsigned, bool> m;
        for (const auto& [s, l] : lm_) m[s] = l.kt;
        return m;
    }
    static bool same_sym(const Sym& x, const Sym& y) {
        return x.known == y.known && (!x.known || (x.v.kind == y.v.kind && x.v.a == y.v.a && x.v.b == y.v.b && x.v.add == y.v.add && x.v.mask == y.v.mask));
    }
    // End of one path: s_load results exist only symbolically; where the other path disagrees, the merge reads the register, so store it.
    void flush_loads(const Sym* other) {
        bool ok = true;
        for (unsigned i = 0; i < 128; ++i)
            if (sym_[i].known && sym_[i].v.kind == ScalarVal::LoadWord && !same_sym(sym_[i], other[i])) st(sgpr_var(i), load_word(sym_[i].v, Insn{}, &ok));
    }
    void close_join() {
        Join j = std::move(joins_.back());
        joins_.pop_back();
        flush_loads(j.syms);
        b.op0(Op::OpBranch, {j.merge});
        b.place_label(j.alt);
        std::swap(sym_, j.syms);
        flush_loads(j.syms);
        b.op0(Op::OpBranch, {j.merge});
        b.place_label(j.merge);
        for (unsigned i = 0; i < 128; ++i) if (!same_sym(sym_[i], j.syms[i])) sym_[i] = {};
        for (auto& [s, l] : lm_) { auto it = j.kts.find(s); l.kt = l.kt && it != j.kts.end() && it->second; }
    }
    std::string prev_mn_;
    bool ended_ = false, jumped_ = false, buf_type_done_ = false;
    std::vector<uint32_t> fetch_;  // inlined fetch-shader code (s_swappc_b64 s[0:1], s[0:1] convention)
    const uint32_t* ret_code_ = nullptr;
    size_t ret_n_ = 0, ret_pc_ = 0;
    Id fn_ = 0;
    // BB_SPV_PROBE=<shader crc hex>:<pc hex>:<vgpr>: the PS exports v[vgpr..vgpr+3] (as floats, before that instruction) to mrt0 instead of its colour
    Id probe_var_ = 0;
    size_t probe_pc_ = ~size_t(0);
    unsigned probe_reg_ = 0;
    float probe_scale_ = 1.f;
};

// ---- registers ------------------------------------------------------------------------------------------------------
Id Xlat::push_user(unsigned i) {
    Id ptr = op(Op::OpAccessChain, PU, {push_var_, cu(0), cu(i)});
    return ld(ptr, U);
}

Id Xlat::builtin_var(spv::StorageClass sc, Id type, spv::BuiltIn bi) {
    Id v = b.global_var(b.t_ptr(sc, type), sc);
    b.decorate(v, spv::Decoration::BuiltIn, {uint32_t(bi)});
    iface_.push_back(v);
    return v;
}

// One SPIR-V invocation models one GCN lane, so the lane id is the invocation's position in the 64-lane wave: a GCN wave is
// 64 consecutive invocations of the workgroup. Outside compute the subgroup is the wave (ponytail: exact only where the
// subgroup is 64 wide, which is what the driver is asked for).
// The value is re-emitted per use: an id from an earlier block would not dominate the use.
Id Xlat::lane_index() {
    if (env_.stage == ShStage::CS) {
        if (!li_ptr_) li_ptr_ = builtin_var(spv::StorageClass::Input, I, spv::BuiltIn::LocalInvocationIndex);
        return op(Op::OpBitwiseAnd, U, {op(Op::OpBitcast, U, {ld(li_ptr_, I)}), cu(63)});
    }
    if (!sub_ptr_) sub_ptr_ = builtin_var(spv::StorageClass::Input, U, spv::BuiltIn::SubgroupLocalInvocationId);
    return ld(sub_ptr_, U);
}

// exec_lo/exec_hi read as a 32-bit lane mask: under the per-invocation exec model this lane's bits are all set while it is active.
Id Xlat::exec_bits() {
    if (exec_kt()) return cu(~0u);
    return op(Op::OpSelect, U, {ld(lane(kExec).var, B), cu(~0u), cu(0)});
}

// The LDS: Workgroup memory in a compute shader, a global storage buffer in the emulated tessellation passes (one buffer for the
// whole draw: the host runs the LS over every control point before the DS reads the records back).
Id Xlat::lds_var() {
    if (!lds_ && (tess() || env_.lds_bytes)) {  // the tessellation passes address a host buffer, its size is the host's business
        if (tess()) lds_ = tess_var(1);
        else {
            Id arr = b.t_array(U, cu(env_.lds_bytes / 4));
            lds_ = b.global_var(b.t_ptr(spv::StorageClass::Workgroup, arr), spv::StorageClass::Workgroup);
            iface_.push_back(lds_);
        }
    }
    return lds_;
}

// The host buffers (tessellation, GDS) are declared here (the shader body needs them) but described in finish_tess(): their
// bindings are only known once the shader's own resources are, and they have to be the last ones.
Id Xlat::tess_var(int slot) {
    if (!tess_ptr_[slot]) {
        Id arr = b.t_rtarray(U), st_t = b.t_struct({arr});
        if (!buf_type_done_) {  // the runtime-array-in-a-struct layout of a descriptor buffer, deduplicated with it
            buf_type_done_ = true;
            b.decorate(arr, spv::Decoration::ArrayStride, {4});
            b.decorate(st_t, spv::Decoration::Block);
            b.member_decorate(st_t, 0, spv::Decoration::Offset, {0});
        }
        tess_ptr_[slot] = b.global_var(b.t_ptr(spv::StorageClass::StorageBuffer, st_t), spv::StorageClass::StorageBuffer);
    }
    return tess_ptr_[slot];
}

bool Xlat::finish_tess() {
    for (int slot : {0, 2, 1}) {  // TessIndices first, LdsBuffer always last; GdsBuffer (compute only) never meets the other two
        if (!tess_ptr_[slot]) continue;
        Resource r;
        r.type = slot == 1 ? Resource::LdsBuffer : slot == 2 ? Resource::GdsBuffer : Resource::TessIndices;
        r.dwords = 0;
        r.written = (slot == 1 && lds_written_) || slot == 2;
        r.binding = 2 * uint32_t(T.resources.size()) + env_.binding_base;
        T.resources.push_back(r);
        b.decorate(tess_ptr_[slot], spv::Decoration::DescriptorSet, {0});
        b.decorate(tess_ptr_[slot], spv::Decoration::Binding, {r.binding});
        iface_.push_back(tess_ptr_[slot]);
    }
    return true;
}

// dfmt 6/7: unsigned small float, w bits = one exponent bit above a (w-1) bit mantissa, value = (E + M / 2^(w-1)) in [0, 2).
// ponytail: no sign bit (AMD's ISA also documents one) and no Inf/NaN; nothing in the game corpus uses these formats.
Id Xlat::sf_unpack(Id raw, unsigned w) { return op(Op::OpFMul, F, {op(Op::OpConvertUToF, F, {raw}), cf(1.f / float(1u << (w - 1)))}); }
Id Xlat::sf_pack(Id f, unsigned w) {
    Id v = glsl(F, GLSLstd450FClamp, {flt(f), cf(0.f), cf(2.f)});
    return op(Op::OpConvertFToU, U, {glsl(F, GLSLstd450RoundEven, {op(Op::OpFMul, F, {v, cf(float(1u << (w - 1)))})})});
}

// Value of an s_load result dword. The data is read at run time from a small storage buffer the host fills per draw with the
// guest memory the load addresses (Resource::load_data); descriptor uses of the same registers stay symbolic.
int Xlat::get_load_resource(uint32_t load) {
    const DescRef ref{int32_t(load), 0x40000000u};
    const auto key = std::make_tuple(int(Resource::Buffer), ref.load, ref.word);
    auto it = res_index_.find(key);
    if (it != res_index_.end()) return it->second;
    Resource r;
    r.type = Resource::Buffer;
    r.ref = ref;
    r.dwords = 4;
    r.load_data = true;
    r.binding = 2 * uint32_t(T.resources.size()) + env_.binding_base;
    T.resources.push_back(r);
    res_index_[key] = int(T.resources.size()) - 1;
    return int(T.resources.size()) - 1;
}

Id Xlat::load_word(const ScalarVal& v, const Insn& in, bool* ok) {
    (void)in; (void)ok;
    Id var = buffer_var(get_load_resource(v.a));
    Id ptr = op(Op::OpAccessChain, b.t_ptr(spv::StorageClass::StorageBuffer, U), {var, cu(0), cu(v.b)});
    Id x = ld(ptr, U);
    if (v.add) x = op(Op::OpIAdd, U, {x, cu(v.add)});
    if (v.mask != ~0u) x = op(Op::OpBitwiseAnd, U, {x, cu(v.mask)});
    return x;
}

Id Xlat::rd_s(unsigned s, const Insn& in, bool is_float, Sym* sym, bool* ok) {
    if (sym) *sym = {};
    if (s < 104 || s == kM0) {
        const Sym& k = sym_[s];
        if (sym) *sym = k;
        if (k.known && k.v.kind == ScalarVal::Const && !k.v.add && k.v.mask == ~0u) return cu(k.v.a);
        if (k.known && k.v.kind == ScalarVal::LoadWord) { if (s < 128) note_sym_read(s, 1); return load_word(k.v, in, ok); }
        return ld(sgpr_var(s), U);
    }
    uint32_t c = 0;
    if (s == 128) c = 0;
    else if (s >= 129 && s <= 192) c = is_float ? 0 : s - 128;
    else if (s >= 193 && s <= 208) c = is_float ? 0 : uint32_t(-int(s - 192));
    if (s >= 128 && s <= 208) {
        if (sym) { sym->known = true; sym->v.kind = ScalarVal::Const; sym->v.a = is_float ? 0 : c; }
        if (!is_float || s == 128) {
            if (sym) sym->v.a = c;
            return cu(c);
        }
        const float v = s <= 192 ? float(int(s) - 128) : -float(int(s) - 192);
        uint32_t u;
        std::memcpy(&u, &v, 4);
        return cu(u);
    }
    static const float kF[8] = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f};
    if (s >= 240 && s <= 247) {
        uint32_t u;
        std::memcpy(&u, &kF[s - 240], 4);
        if (sym) { sym->known = true; sym->v.kind = ScalarVal::Const; sym->v.a = u; }
        return cu(u);
    }
    if (s == 255) {
        if (sym) { sym->known = true; sym->v.kind = ScalarVal::Const; sym->v.a = in.literal; }
        return cu(in.literal);
    }
    if (s == 253) {  // SCC as 0/1
        if (!scc_) { *ok = fail("SCC read before any write", &in); return cu(0); }
        return op(Op::OpSelect, U, {ld(scc_, B), cu(1), cu(0)});
    }
    *ok = fail("unsupported scalar source " + std::to_string(s), &in);
    return cu(0);
}

Id Xlat::rd_v(unsigned s9, const Insn& in, bool is_float, bool* ok) {
    if (s9 >= 256) return ld(vgpr_var(s9 - 256), U);
    return rd_s(s9, in, is_float, nullptr, ok);
}

void Xlat::wr_s(unsigned i, Id v, const Sym& sym) {
    if (i == 107) return;  // vcc_hi: the shader-start marker (s_mov_b32 vcc_hi, <hash>) carries no state
    st(sgpr_var(i), v);
    sym_[i] = sym;
}

void Xlat::wr_v(unsigned i, Id v) {
    Id var = vgpr_var(i);
    if (!exec_kt()) v = op(Op::OpSelect, U, {ld(lane(kExec).var, B), v, ld(var, U)});
    st(var, v);
}

Id Xlat::rd_lane(unsigned s, const Insn& in, bool* ok) {
    if (s == 128 || (s == 255 && in.literal == 0)) return b.c_bool(false);
    if (s == 193 || (s == 255 && in.literal == 0xFFFFFFFFu)) return b.c_bool(true);
    if (s == kExec && lane(kExec).kt) return b.c_bool(true);
    if (s != kExec && s != kVcc && !lm_.count(s)) {
        *ok = fail("scalar pair s" + std::to_string(s) + " read as lane mask but never written as one", &in);
        return b.c_bool(false);
    }
    return ld(lane(s).var, B);
}

void Xlat::wr_lane(unsigned s, Id v, bool kt) {
    Lane& l = lane(s);
    st(l.var, v);
    l.kt = kt;
}

// ---- descriptors ----------------------------------------------------------------------------------------------------
uint32_t Xlat::eval_sval(const ScalarVal& v) const {
    uint32_t base = 0;
    switch (v.kind) {
        case ScalarVal::User: base = env_.user[v.a & 15]; break;
        case ScalarVal::Const: base = v.a; break;
        case ScalarVal::LoadWord: base = v.a < load_vals_.size() && v.b < load_vals_[v.a].size() ? load_vals_[v.a][v.b] : 0; break;
    }
    return (base + v.add) & v.mask;
}

bool Xlat::desc_ref(unsigned sfirst, unsigned cnt, DescRef& out) {
    note_sym_read(sfirst, cnt);
    for (unsigned k = 0; k < cnt; ++k) {
        const Sym& s = sym_[sfirst + k];
        const Sym& s0 = sym_[sfirst];
        if (!s.known || s.v.kind != s0.v.kind || s.v.add || s.v.mask != ~0u) return false;
        if (s.v.kind == ScalarVal::User && s.v.a != s0.v.a + k) return false;
        if (s.v.kind == ScalarVal::LoadWord && (s.v.a != s0.v.a || s.v.b != s0.v.b + k)) return false;
        if (s.v.kind == ScalarVal::Const) return false;
    }
    const Sym& s0 = sym_[sfirst];
    out = s0.v.kind == ScalarVal::User ? DescRef{-1, s0.v.a} : DescRef{int32_t(s0.v.a), s0.v.b};
    return true;
}

bool Xlat::read_desc_words(const DescRef& ref, unsigned dwords, uint32_t* out) {
    for (unsigned k = 0; k < dwords; ++k) {
        if (ref.load < 0) {
            if (ref.word + k >= 16) return false;
            out[k] = env_.user[ref.word + k];
        } else {
            if (size_t(ref.load) >= load_vals_.size() || ref.word + k >= load_vals_[ref.load].size()) return false;
            out[k] = load_vals_[ref.load][ref.word + k];
        }
    }
    return true;
}

int Xlat::get_resource(Resource::Type type, const DescRef& ref, unsigned dwords, bool write) {
    const auto key = std::make_tuple(int(type), ref.load, ref.word);
    auto it = res_index_.find(key);
    if (it != res_index_.end()) {
        if (write) T.resources[it->second].written = true;
        return it->second;
    }
    Resource r;
    r.type = type;
    r.ref = ref;
    r.dwords = dwords;
    r.written = write;
    r.binding = 2 * uint32_t(T.resources.size()) + env_.binding_base;
    if (!read_desc_words(ref, dwords, r.words)) return -1;
    T.resources.push_back(r);
    res_index_[key] = int(T.resources.size()) - 1;
    return int(T.resources.size()) - 1;
}

Id Xlat::buffer_var(int res) {
    auto it = res_var_.find(res);
    if (it != res_var_.end()) return it->second;
    Id arr = b.t_rtarray(U);
    Id st_t = b.t_struct({arr});
    if (!buf_type_done_) {  // both types are deduplicated: decorate once
        buf_type_done_ = true;
        b.decorate(arr, spv::Decoration::ArrayStride, {4});
        b.decorate(st_t, spv::Decoration::Block);
        b.member_decorate(st_t, 0, spv::Decoration::Offset, {0});
    }
    Id var = b.global_var(b.t_ptr(spv::StorageClass::StorageBuffer, st_t), spv::StorageClass::StorageBuffer);
    b.decorate(var, spv::Decoration::DescriptorSet, {0});
    b.decorate(var, spv::Decoration::Binding, {T.resources[res].binding});
    iface_.push_back(var);
    res_var_[res] = var;
    return var;
}

Id Xlat::image_type_of(const Resource& r, bool storage) {
    const uint32_t type = r.words[3] >> 28, nfmt = (r.words[1] >> 26) & 15;
    Id sampled = nfmt == 4 ? U : nfmt == 5 ? I : F;
    spv::Dim dim;
    bool arrayed = false;
    switch (type) {
        case 8: dim = spv::Dim::Dim1D; b.capability(spv::Capability::Sampled1D); b.capability(spv::Capability::Image1D); break;
        case 9: dim = spv::Dim::Dim2D; break;
        case 10: dim = spv::Dim::Dim3D; break;
        case 11: dim = spv::Dim::Cube; break;
        case 12: dim = spv::Dim::Dim1D; arrayed = true; b.capability(spv::Capability::Sampled1D); b.capability(spv::Capability::Image1D); break;
        case 13: dim = spv::Dim::Dim2D; arrayed = true; break;
        default: return 0;
    }
    return b.t_image(sampled, dim, arrayed, storage ? 2 : 1);
}

Id Xlat::image_var(int res) {
    auto it = res_var_.find(res);
    if (it != res_var_.end()) return it->second;
    const Resource& r = T.resources[res];
    Id t = image_type_of(r, r.type == Resource::StorageImage);
    if (!t) return 0;
    Id var = b.global_var(b.t_ptr(spv::StorageClass::UniformConstant, t), spv::StorageClass::UniformConstant);
    b.decorate(var, spv::Decoration::DescriptorSet, {0});
    b.decorate(var, spv::Decoration::Binding, {r.binding});
    iface_.push_back(var);
    res_var_[res] = var;
    return var;
}

Id Xlat::sampler_var(int res) {
    auto it = res_var_.find(res);
    if (it != res_var_.end()) return it->second;
    Id var = b.global_var(b.t_ptr(spv::StorageClass::UniformConstant, b.t_sampler()), spv::StorageClass::UniformConstant);
    b.decorate(var, spv::Decoration::DescriptorSet, {0});
    b.decorate(var, spv::Decoration::Binding, {T.resources[res].binding});
    iface_.push_back(var);
    res_var_[res] = var;
    return var;
}

// ---- scalar ALU -----------------------------------------------------------------------------------------------------
bool Xlat::salu(const Insn& in, const std::string& m) {
    const uint32_t d = in.w[0];
    bool ok = true;
    // 64-bit lane-mask / pointer moves
    if (in.fmt == GcnFmt::SOP1 && (m == "s_mov_b64" || m == "s_wqm_b64")) {
        const unsigned dst = (d >> 16) & 0x7F, src = d & 0xFF;
        if (m == "s_wqm_b64" && src == kExec && dst == kExec) { wr_lane(kExec, b.c_bool(true), true); return true; }
        if (lm_.count(src) || src == kExec || src == kVcc || src == 128 || src == 193 || (src == 255 && (in.literal == 0 || in.literal == ~0u))) {
            const bool kt = (src == 193) || (src == kExec && lane(kExec).kt) || (lm_.count(src) && lane(src).kt);
            wr_lane(dst, rd_lane(src, in, &ok), kt);
            return ok;
        }
        if (dst >= 103 || src >= 103) return fail("unsupported 64-bit scalar move", &in);
        for (unsigned k = 0; k < 2; ++k) {
            if (!copy_s(dst + k, src + k, in)) return false;
        }
        return true;
    }
    if (in.fmt == GcnFmt::SOP1 && m == "s_and_saveexec_b64") {
        const unsigned dst = (d >> 16) & 0x7F, src = d & 0xFF;
        Id old = rd_lane(kExec, in, &ok), s = rd_lane(src, in, &ok);
        if (!ok) return false;
        wr_lane(dst, old, lane(kExec).kt);
        wr_lane(kExec, op(Op::OpLogicalAnd, B, {old, s}), false);
        return true;
    }
    if (in.fmt == GcnFmt::SOP2 && (m == "s_and_b64" || m == "s_or_b64" || m == "s_xor_b64" || m == "s_andn2_b64" || m == "s_orn2_b64")) {
        const unsigned dst = (d >> 16) & 0x7F, s0 = d & 0xFF, s1 = (d >> 8) & 0xFF;
        Id a = rd_lane(s0, in, &ok), c = rd_lane(s1, in, &ok);
        if (!ok) return false;
        if (m == "s_andn2_b64") c = op(Op::OpLogicalNot, B, {c});
        if (m == "s_orn2_b64") c = op(Op::OpLogicalNot, B, {c});
        Id r = op(m == "s_or_b64" || m == "s_orn2_b64" ? Op::OpLogicalOr : m == "s_xor_b64" ? Op::OpLogicalNotEqual : Op::OpLogicalAnd, B, {a, c});
        wr_lane(dst, r, false);
        return true;
    }
    if (in.fmt == GcnFmt::SOP1 || in.fmt == GcnFmt::SOP2 || in.fmt == GcnFmt::SOPK || in.fmt == GcnFmt::SOPC) {
        const unsigned dst = in.fmt == GcnFmt::SOPC ? 0 : (d >> 16) & 0x7F;
        Sym sa, sb;
        Id a = 0, c = 0;
        if (in.fmt == GcnFmt::SOPK) {
            const int32_t imm = int16_t(d & 0xFFFF);
            if (m == "s_addk_i32" || m == "s_mulk_i32") {  // dst = dst op simm16; ponytail: SCC (signed overflow of s_addk) is not produced
                Id cur = rd_s(dst, in, false, &sa, &ok);
                if (!ok) return false;
                wr_s(dst, op(m == "s_addk_i32" ? Op::OpIAdd : Op::OpIMul, U, {cur, cu(uint32_t(imm))}));
                return true;
            }
            if (starts_with(m, "s_cmpk_")) {  // s_cmpk_{eq,lg,gt,ge,lt,le}_{i32,u32} sdst, simm16 (sign-extended for i32, zero-extended for u32)
                const std::string p = m.substr(7, 2), ty = m.substr(10);
                Id cur = rd_s(dst, in, false, &sa, &ok);
                if (!ok) return false;
                const bool sg = ty == "i32";
                Id aa = sg ? op(Op::OpBitcast, I, {cur}) : cur, cc = sg ? b.c_i32(imm) : cu(d & 0xFFFF);
                Op co = p == "eq" ? Op::OpIEqual : p == "lg" ? Op::OpINotEqual : p == "gt" ? (sg ? Op::OpSGreaterThan : Op::OpUGreaterThan)
                      : p == "ge" ? (sg ? Op::OpSGreaterThanEqual : Op::OpUGreaterThanEqual) : p == "lt" ? (sg ? Op::OpSLessThan : Op::OpULessThan)
                      : (sg ? Op::OpSLessThanEqual : Op::OpULessThanEqual);
                if (!scc_) scc_ = b.local_var_init(B, b.c_bool(false));
                st(scc_, op(co, B, {aa, cc}));
                return true;
            }
            if (m != "s_movk_i32") return fail("unsupported SOPK", &in);
            sb.known = true; sb.v.kind = ScalarVal::Const; sb.v.a = uint32_t(imm);
            wr_s(dst, cu(uint32_t(imm)), sb);
            return true;
        }
        if (m == "s_mov_b32") return copy_s(dst, d & 0xFF, in);
        a = rd_s(d & 0xFF, in, false, &sa, &ok);
        if (in.fmt != GcnFmt::SOP1) c = rd_s((d >> 8) & 0xFF, in, false, &sb, &ok);
        if (!ok) return false;
        if (m == "s_mov_b32") return copy_s(dst, d & 0xFF, in);
        if (m == "s_not_b32") { wr_s(dst, op(Op::OpNot, U, {a})); return true; }
        const bool both_const = sa.known && sb.known && sa.v.kind == ScalarVal::Const && sb.v.kind == ScalarVal::Const && !sa.v.add && !sb.v.add;
        Sym r;
        auto fold = [&](uint32_t v) { r.known = true; r.v.kind = ScalarVal::Const; r.v.a = v; };
        Op o = Op::OpNop;
        if (m == "s_add_u32" || m == "s_add_i32") {
            o = Op::OpIAdd;
            if (both_const) fold(sa.v.a + sb.v.a);
            else if (sa.known && sb.known && sb.v.kind == ScalarVal::Const && sa.v.mask == ~0u) { r = sa; r.v.add += sb.v.a; }
            if (m == "s_add_u32") {
                Id sum = op(o, U, {a, c});
                if (!scc_) scc_ = b.local_var_init(B, b.c_bool(false));
                st(scc_, op(Op::OpULessThan, B, {sum, a}));
                wr_s(dst, sum, r);
                return true;
            }
        } else if (m == "s_sub_u32" || m == "s_sub_i32") {
            o = Op::OpISub;
            if (both_const) fold(sa.v.a - sb.v.a);
            if (m == "s_sub_u32") {
                Id diff = op(o, U, {a, c});
                if (!scc_) scc_ = b.local_var_init(B, b.c_bool(false));
                st(scc_, op(Op::OpULessThan, B, {a, c}));
                wr_s(dst, diff, r);
                return true;
            }
        } else if (m == "s_and_b32") {
            o = Op::OpBitwiseAnd;
            if (both_const) fold(sa.v.a & sb.v.a);
            else if (sa.known && sb.known && sb.v.kind == ScalarVal::Const && sa.v.mask == ~0u) { r = sa; r.v.mask = sb.v.a; }
        } else if (m == "s_or_b32") { o = Op::OpBitwiseOr; if (both_const) fold(sa.v.a | sb.v.a); }
        else if (m == "s_xor_b32") { o = Op::OpBitwiseXor; if (both_const) fold(sa.v.a ^ sb.v.a); }
        else if (m == "s_lshl_b32") { o = Op::OpShiftLeftLogical; c = op(Op::OpBitwiseAnd, U, {c, cu(31)}); if (both_const) fold(sa.v.a << (sb.v.a & 31)); }
        else if (m == "s_lshr_b32") { o = Op::OpShiftRightLogical; c = op(Op::OpBitwiseAnd, U, {c, cu(31)}); if (both_const) fold(sa.v.a >> (sb.v.a & 31)); }
        else if (m == "s_ashr_i32") { o = Op::OpShiftRightArithmetic; c = op(Op::OpBitwiseAnd, U, {c, cu(31)}); }
        else if (m == "s_mul_i32") { o = Op::OpIMul; if (both_const) fold(sa.v.a * sb.v.a); }
        else if (m == "s_andn2_b32") { o = Op::OpBitwiseAnd; c = op(Op::OpNot, U, {c}); }
        else if (m == "s_min_u32" || m == "s_max_u32" || m == "s_min_i32" || m == "s_max_i32") {
            const bool is_signed = m.back() == '2' && m[m.size() - 3] == 'i', is_min = m.find("min") != std::string::npos;
            Id aa = is_signed ? op(Op::OpBitcast, I, {a}) : a, cc = is_signed ? op(Op::OpBitcast, I, {c}) : c;
            Id res = glsl(is_signed ? I : U, is_signed ? (is_min ? GLSLstd450SMin : GLSLstd450SMax) : (is_min ? GLSLstd450UMin : GLSLstd450UMax), {aa, cc});
            wr_s(dst, is_signed ? bits(res) : res);
            return true;
        } else if (m == "s_bfe_u32") {  // offset = c[4:0], width = c[22:16]
            Id off = op(Op::OpBitwiseAnd, U, {c, cu(31)}), wid = op(Op::OpBitwiseAnd, U, {op(Op::OpShiftRightLogical, U, {c, cu(16)}), cu(0x7F)});
            wr_s(dst, op(Op::OpBitFieldUExtract, U, {a, off, wid}));
            return true;
        } else if (m == "s_cselect_b32") {
            if (!scc_) return fail("s_cselect_b32 without SCC", &in);
            wr_s(dst, op(Op::OpSelect, U, {ld(scc_, B), a, c}));
            return true;
        } else if (starts_with(m, "s_cmp_")) {
            // s_cmp_{eq,lg,gt,ge,lt,le}_{i32,u32}
            const std::string p = m.substr(6, 2), ty = m.substr(9);
            Id aa = ty == "i32" ? op(Op::OpBitcast, I, {a}) : a, cc = ty == "i32" ? op(Op::OpBitcast, I, {c}) : c;
            const bool sg = ty == "i32";
            Op co = p == "eq" ? Op::OpIEqual : p == "lg" ? Op::OpINotEqual : p == "gt" ? (sg ? Op::OpSGreaterThan : Op::OpUGreaterThan)
                  : p == "ge" ? (sg ? Op::OpSGreaterThanEqual : Op::OpUGreaterThanEqual) : p == "lt" ? (sg ? Op::OpSLessThan : Op::OpULessThan)
                  : (sg ? Op::OpSLessThanEqual : Op::OpULessThanEqual);
            if (!scc_) scc_ = b.local_var_init(B, b.c_bool(false));
            st(scc_, op(co, B, {aa, cc}));
            return true;
        } else return fail("unsupported scalar instruction", &in);
        wr_s(dst, op(o, U, {a, c}), r);
        return true;
    }
    return fail("unsupported scalar format", &in);
}

bool Xlat::smrd(const Insn& in, const std::string& m) {
    const uint32_t d = in.w[0];
    const unsigned sdst = (d >> 15) & 0x7F, sbase = ((d >> 9) & 0x3F) * 2, off = in.has_literal ? in.literal : d & 0xFF;
    const bool imm = (d & 0x100) || in.has_literal, buf = starts_with(m, "s_buffer_load");
    const unsigned opn = in.op & 7, cnt = opn == 0 ? 1 : opn == 1 ? 2 : opn == 2 ? 4 : opn == 3 ? 8 : 16;
    if (!starts_with(m, "s_load") && !buf) return fail("unsupported SMRD", &in);
    bool ok = true;
    if (!buf) {
        if (!imm) return fail("s_load with SGPR offset", &in);
        const Sym &lo = sym_[sbase], &hi = sym_[sbase + 1];
        note_sym_read(sbase, 2);
        if (!lo.known || !hi.known) return fail("s_load base pointer is not user-data derived", &in);
        SLoad l;
        l.lo = lo.v; l.hi = hi.v; l.offset = off * 4; l.dwords = cnt;
        const uint64_t addr = (uint64_t(eval_sval(l.hi)) << 32 | eval_sval(l.lo)) + l.offset;
        std::vector<uint32_t> vals(cnt);
        if (!env_.read_mem || !env_.read_mem(addr, vals.data(), cnt)) return fail("s_load: guest memory unreadable at translation time", &in);
        load_vals_.push_back(vals);
        T.loads.push_back(l);
        for (unsigned k = 0; k < cnt; ++k) {
            Sym s;
            s.known = true; s.v.kind = ScalarVal::LoadWord; s.v.a = uint32_t(T.loads.size() - 1); s.v.b = k;
            sym_[sdst + k] = s;
        }
        return true;
    }
    DescRef ref;
    if (!desc_ref(sbase, 4, ref)) return fail("s_buffer_load: V# is not a user-data/s_load descriptor", &in);
    const int res = get_resource(Resource::Buffer, ref, 4, false);
    if (res < 0) return fail("s_buffer_load: descriptor unreadable", &in);
    T.resources[size_t(res)].scalar = true;
    Id word0 = imm ? cu(off) : op(Op::OpShiftRightLogical, U, {rd_s(off, in, false, nullptr, &ok), cu(2)});
    if (!ok) return false;
    Id var = buffer_var(res);
    for (unsigned k = 0; k < cnt; ++k) {
        Id idx = op(Op::OpIAdd, U, {word0, cu(k)});
        Id ptr = op(Op::OpAccessChain, b.t_ptr(spv::StorageClass::StorageBuffer, U), {var, cu(0), idx});
        wr_s(sdst + k, ld(ptr, U));
    }
    return true;
}

// ---- vector ALU -----------------------------------------------------------------------------------------------------
struct VOp {
    std::string name;
    int vdst = -1, sdst = -1;
    unsigned s[3] = {0, 0, 0};
    bool neg[3] = {}, abs[3] = {}, clamp = false, e64 = false;
    unsigned omod = 0;
};

bool Xlat::valu(const Insn& in) {
    const uint32_t d = in.w[0], e = in.w[1];
    VOp v;
    const char* mn = nullptr;
    switch (in.fmt) {
        case GcnFmt::VOP1: mn = gcn::mnemonic(in.fmt, in.op); v.vdst = (d >> 17) & 0xFF; v.s[0] = d & 0x1FF; break;
        case GcnFmt::VOP2: mn = gcn::mnemonic(in.fmt, in.op); v.vdst = (d >> 17) & 0xFF; v.s[0] = d & 0x1FF; v.s[1] = 256 + ((d >> 9) & 0xFF); break;
        case GcnFmt::VOPC: mn = gcn::mnemonic(in.fmt, in.op); v.sdst = kVcc; v.s[0] = d & 0x1FF; v.s[1] = 256 + ((d >> 9) & 0xFF); break;
        case GcnFmt::VOP3: {
            v.e64 = true;
            const unsigned o = in.op;
            mn = o < 0x100 ? gcn::mnemonic(GcnFmt::VOPC, o) : o < 0x140 ? gcn::mnemonic(GcnFmt::VOP2, o - 0x100)
               : o >= 0x180 && o < 0x200 ? gcn::mnemonic(GcnFmt::VOP1, o - 0x180) : gcn::mnemonic(GcnFmt::VOP3, o);
            v.s[0] = e & 0x1FF; v.s[1] = (e >> 9) & 0x1FF; v.s[2] = (e >> 18) & 0x1FF;
            for (int k = 0; k < 3; ++k) { v.abs[k] = (d >> (8 + k)) & 1; v.neg[k] = (e >> (29 + k)) & 1; }
            v.clamp = (d >> 11) & 1;
            v.omod = (e >> 27) & 3;
            if (o < 0x100) v.sdst = d & 0x7F;
            else {
                v.vdst = d & 0xFF;
                if (mn && (!std::strcmp(mn, "v_add_co_u32") || !std::strcmp(mn, "v_sub_co_u32") || !std::strcmp(mn, "v_subrev_co_u32") ||
                           !std::strcmp(mn, "v_addc_co_u32") || !std::strcmp(mn, "v_subb_co_u32") || !std::strcmp(mn, "v_subbrev_co_u32")))
                    v.sdst = (d >> 8) & 0x7F;
            }
            break;
        }
        default: return fail("not a vector ALU format", &in);
    }
    if (!mn) return fail("unknown vector opcode", &in);
    v.name = mn;
    if (!v.e64 && (in.fmt == GcnFmt::VOP2 || in.fmt == GcnFmt::VOP1 || in.fmt == GcnFmt::VOPC) && v.name.find("_e64") != std::string::npos) return fail("bad name", &in);
    const std::string& m = v.name;
    bool ok = true;

    const bool float_src = ends_with(m, "_f32");
    auto src = [&](int i) -> Id {  // raw bits
        return rd_v(v.s[i], in, float_src, &ok);
    };
    auto fsrc = [&](int i) -> Id {  // float with VOP3 input modifiers
        Id x = flt(rd_v(v.s[i], in, true, &ok));
        if (v.abs[i]) x = glsl(F, GLSLstd450FAbs, {x});
        if (v.neg[i]) x = op(Op::OpFNegate, F, {x});
        return x;
    };
    auto fres = [&](Id x) -> Id {  // omod/clamp then bits
        if (v.omod == 1) x = op(Op::OpFMul, F, {x, cf(2.f)});
        else if (v.omod == 2) x = op(Op::OpFMul, F, {x, cf(4.f)});
        else if (v.omod == 3) x = op(Op::OpFMul, F, {x, cf(0.5f)});
        if (v.clamp) x = glsl(F, GLSLstd450FClamp, {x, cf(0.f), cf(1.f)});
        return bits(x);
    };
    auto put = [&](Id r) { if (ok) wr_v(unsigned(v.vdst), r); return ok; };

    // ---- compares -------------------------------------------------------------------------------------------------
    if (starts_with(m, "v_cmp_") || starts_with(m, "v_cmpx_")) {
        const bool x = starts_with(m, "v_cmpx_");
        const std::string rest = m.substr(x ? 7 : 6);  // e.g. "gt_u32"
        const size_t us = rest.find('_');
        const std::string p = rest.substr(0, us), ty = rest.substr(us + 1);
        if (ty != "f32" && ty != "i32" && ty != "u32") return fail("unsupported compare type " + ty, &in);
        Id a, c;
        if (ty == "f32") { a = fsrc(0); c = fsrc(1); }
        else { a = src(0); c = src(1); if (ty == "i32") { a = op(Op::OpBitcast, I, {a}); c = op(Op::OpBitcast, I, {c}); } }
        if (!ok) return false;
        Id r = 0;
        if (ty == "f32") {
            Op ord = Op::OpNop;
            bool neg_unord = false, unord_variant = false;
            if (p == "f") r = b.c_bool(false);
            else if (p == "tru") r = b.c_bool(true);
            else if (p == "o") r = op(Op::OpLogicalNot, B, {op(Op::OpLogicalOr, B, {op(Op::OpIsNan, B, {a}), op(Op::OpIsNan, B, {c})})});
            else if (p == "u") r = op(Op::OpLogicalOr, B, {op(Op::OpIsNan, B, {a}), op(Op::OpIsNan, B, {c})});
            else {
                unord_variant = p[0] == 'n';
                const std::string q = unord_variant ? p.substr(1) : p;
                ord = q == "lt" ? Op::OpFOrdLessThan : q == "eq" ? Op::OpFOrdEqual : q == "le" ? Op::OpFOrdLessThanEqual : q == "gt" ? Op::OpFOrdGreaterThan
                    : q == "lg" ? Op::OpFOrdNotEqual : q == "ge" ? Op::OpFOrdGreaterThanEqual : Op::OpNop;
                if (ord == Op::OpNop) return fail("unsupported float compare " + p, &in);
                // n<op> = !<op> (true if unordered)
                r = op(ord, B, {a, c});
                if (unord_variant) r = op(Op::OpLogicalNot, B, {r});
                (void)neg_unord;
            }
        } else {
            const bool sg = ty == "i32";
            Op o = p == "f" ? Op::OpNop : p == "lt" ? (sg ? Op::OpSLessThan : Op::OpULessThan) : p == "eq" ? Op::OpIEqual : p == "le" ? (sg ? Op::OpSLessThanEqual : Op::OpULessThanEqual)
                 : p == "gt" ? (sg ? Op::OpSGreaterThan : Op::OpUGreaterThan) : p == "lg" || p == "ne" ? Op::OpINotEqual : p == "ge" ? (sg ? Op::OpSGreaterThanEqual : Op::OpUGreaterThanEqual) : Op::OpNop;
            if (p == "f") r = b.c_bool(false);
            else if (p == "tru") r = b.c_bool(true);
            else if (o == Op::OpNop) return fail("unsupported integer compare " + p, &in);
            else r = op(o, B, {a, c});
        }
        if (!exec_kt()) r = op(Op::OpLogicalAnd, B, {r, ld(lane(kExec).var, B)});
        if (x) wr_lane(kExec, r, false);
        else wr_lane(unsigned(v.sdst), r, false);
        return true;
    }


    // ---- moves, conversions, float math ----------------------------------------------------------------------------
    if (m == "v_readfirstlane_b32") {  // VOP1: the destination field names an SGPR; per-invocation = own value (uniform assumption)
        Id x = src(0);
        if (!ok) return false;
        wr_s((d >> 17) & 0xFF, x);
        return true;
    }
    if (m == "v_mov_b32") return put(src(0));
    if (m == "v_not_b32") return put(op(Op::OpNot, U, {src(0)}));
    if (m == "v_cvt_f32_i32") return put(fres(op(Op::OpConvertSToF, F, {op(Op::OpBitcast, I, {src(0)})})));
    if (m == "v_cvt_f32_u32") return put(fres(op(Op::OpConvertUToF, F, {src(0)})));
    if (m == "v_cvt_i32_f32") {
        Id x = fsrc(0);
        Id c = glsl(F, GLSLstd450FClamp, {x, cf(-2147483648.f), cf(2147483520.f)});
        Id r = op(Op::OpConvertFToS, I, {c});
        return put(op(Op::OpSelect, U, {op(Op::OpIsNan, B, {x}), cu(0), op(Op::OpBitcast, U, {r})}));
    }
    if (m == "v_cvt_u32_f32") {
        Id x = fsrc(0);
        Id c = glsl(F, GLSLstd450FClamp, {x, cf(0.f), cf(4294967040.f)});
        return put(op(Op::OpSelect, U, {op(Op::OpIsNan, B, {x}), cu(0), op(Op::OpConvertFToU, U, {c})}));
    }
    if (starts_with(m, "v_cvt_f32_ubyte")) {
        const unsigned k = m.back() - '0';
        return put(fres(op(Op::OpConvertUToF, F, {op(Op::OpBitwiseAnd, U, {op(Op::OpShiftRightLogical, U, {src(0), cu(8 * k)}), cu(0xFF)})})));
    }
    if (m == "v_cvt_pkrtz_f16_f32") {
        // Round toward zero: PackHalf2x16 rounds to nearest, so a half whose magnitude came out above |x| steps one ulp down. This also
        // keeps |x| > 65504 finite (65504, not inf) as on GCN: HDR effects (lamp glare, bright particles) exceed it, and an inf in the
        // scene target turned into blocky green blooms over the whole post chain.
        auto rtz = [&](Id x) {
            Id h = op(Op::OpBitwiseAnd, U, {glsl(U, GLSLstd450PackHalf2x16, {op(Op::OpCompositeConstruct, V2F, {x, cf(0.f)})}), cu(0xFFFF)});
            Id back = op(Op::OpCompositeExtract, F, {glsl(V2F, GLSLstd450UnpackHalf2x16, {h}), 0});
            Id over = op(Op::OpFOrdGreaterThan, B, {glsl(F, GLSLstd450FAbs, {back}), glsl(F, GLSLstd450FAbs, {x})});
            return op(Op::OpSelect, U, {over, op(Op::OpISub, U, {h, cu(1)}), h});
        };
        return put(op(Op::OpBitwiseOr, U, {rtz(fsrc(0)), op(Op::OpShiftLeftLogical, U, {rtz(fsrc(1)), cu(16)})}));
    }
    if (m == "v_cvt_f32_f16") return put(fres(op(Op::OpCompositeExtract, F, {glsl(V2F, GLSLstd450UnpackHalf2x16, {src(0)}), 0})));  // low half
    if (m == "v_cvt_f16_f32") return put(glsl(U, GLSLstd450PackHalf2x16, {op(Op::OpCompositeConstruct, V2F, {fsrc(0), cf(0.f)})}));  // high half 0
    static const std::pair<const char*, GLSLstd450> kF1[] = {
        {"v_floor_f32", GLSLstd450Floor}, {"v_ceil_f32", GLSLstd450Ceil}, {"v_trunc_f32", GLSLstd450Trunc}, {"v_fract_f32", GLSLstd450Fract},
        {"v_rndne_f32", GLSLstd450RoundEven}, {"v_sqrt_f32", GLSLstd450Sqrt}, {"v_rsq_f32", GLSLstd450InverseSqrt}, {"v_exp_f32", GLSLstd450Exp2},
        {"v_log_f32", GLSLstd450Log2}};
    for (const auto& [n, g] : kF1)
        if (m == n) return put(fres(glsl(F, g, {fsrc(0)})));
    if (m == "v_rcp_f32" || m == "v_rcp_iflag_f32") return put(fres(op(Op::OpFDiv, F, {cf(1.f), fsrc(0)})));
    if (m == "v_sin_f32" || m == "v_cos_f32")  // inputs are in revolutions
        return put(fres(glsl(F, m == "v_sin_f32" ? GLSLstd450Sin : GLSLstd450Cos, {op(Op::OpFMul, F, {fsrc(0), cf(6.2831853071795864f)})})));

    // ---- float arithmetic ---------------------------------------------------------------------------------------------
    // *_legacy_f32 multiplies follow the D3D9 rule 0 * anything = 0 (also inf / NaN). Shaders rely on it, e.g. pow as
    // exp2(log2(x) * y) with x = 0 and y = 0; a plain multiply makes NaN there, which bloom/adaptation passes spread over the screen.
    auto lmul = [&](Id a, Id c) {
        Id z = op(Op::OpLogicalOr, B, {op(Op::OpFOrdEqual, B, {a, cf(0.f)}), op(Op::OpFOrdEqual, B, {c, cf(0.f)})});
        return op(Op::OpSelect, F, {z, cf(0.f), op(Op::OpFMul, F, {a, c})});
    };
    if (m == "v_add_f32" || m == "v_sub_f32" || m == "v_subrev_f32" || m == "v_mul_f32" || m == "v_mul_legacy_f32" || m == "v_min_f32" || m == "v_max_f32") {
        Id a = fsrc(0), c = fsrc(1);
        if (m == "v_subrev_f32") std::swap(a, c);
        Id r = m == "v_add_f32" ? op(Op::OpFAdd, F, {a, c}) : m == "v_mul_f32" ? op(Op::OpFMul, F, {a, c}) : m == "v_mul_legacy_f32" ? lmul(a, c)
             : m == "v_min_f32" ? glsl(F, GLSLstd450FMin, {a, c}) : m == "v_max_f32" ? glsl(F, GLSLstd450FMax, {a, c}) : op(Op::OpFSub, F, {a, c});
        return put(fres(r));
    }
    if (m == "v_mac_f32" || m == "v_mac_legacy_f32") {
        Id a = fsrc(0), c = fsrc(1);
        Id r = op(Op::OpFAdd, F, {m == "v_mac_f32" ? op(Op::OpFMul, F, {a, c}) : lmul(a, c), flt(ld(vgpr_var(unsigned(v.vdst)), U))});
        return put(fres(r));
    }
    if (m == "v_madmk_f32") return put(fres(op(Op::OpFAdd, F, {op(Op::OpFMul, F, {fsrc(0), flt(cu(in.literal))}), fsrc(1)})));
    if (m == "v_madak_f32") return put(fres(op(Op::OpFAdd, F, {op(Op::OpFMul, F, {fsrc(0), fsrc(1)}), flt(cu(in.literal))})));
    if (m == "v_mad_f32") return put(fres(op(Op::OpFAdd, F, {op(Op::OpFMul, F, {fsrc(0), fsrc(1)}), fsrc(2)})));
    if (m == "v_mad_legacy_f32") return put(fres(op(Op::OpFAdd, F, {lmul(fsrc(0), fsrc(1)), fsrc(2)})));
    if (m == "v_fma_f32") return put(fres(glsl(F, GLSLstd450Fma, {fsrc(0), fsrc(1), fsrc(2)})));
    if (m == "v_cubeid_f32" || m == "v_cubesc_f32" || m == "v_cubetc_f32" || m == "v_cubema_f32") {  // cube face selection, see cube_dir() for the inverse
        Id x = fsrc(0), y = fsrc(1), z = fsrc(2);
        auto ab = [&](Id v) { return glsl(F, GLSLstd450FAbs, {v}); };
        auto neg = [&](Id v) { return op(Op::OpFNegate, F, {v}); };
        auto sel = [&](Id c, Id a, Id d) { return op(Op::OpSelect, F, {c, a, d}); };
        auto isneg = [&](Id v) { return op(Op::OpFOrdLessThan, B, {v, cf(0.f)}); };
        Id ax = ab(x), ay = ab(y), az = ab(z);
        Id zm = op(Op::OpLogicalAnd, B, {op(Op::OpFOrdGreaterThanEqual, B, {az, ax}), op(Op::OpFOrdGreaterThanEqual, B, {az, ay})});
        Id ym = op(Op::OpFOrdGreaterThanEqual, B, {ay, ax});
        Id r;
        if (m == "v_cubema_f32") r = sel(zm, op(Op::OpFMul, F, {z, cf(2.f)}), sel(ym, op(Op::OpFMul, F, {y, cf(2.f)}), op(Op::OpFMul, F, {x, cf(2.f)})));
        else if (m == "v_cubeid_f32") r = sel(zm, sel(isneg(z), cf(5.f), cf(4.f)), sel(ym, sel(isneg(y), cf(3.f), cf(2.f)), sel(isneg(x), cf(1.f), cf(0.f))));
        else if (m == "v_cubesc_f32") r = sel(zm, sel(isneg(z), neg(x), x), sel(ym, x, sel(isneg(x), z, neg(z))));
        else r = sel(zm, neg(y), sel(ym, sel(isneg(y), neg(z), z), neg(y)));
        return put(fres(r));
    }

    // ---- integer / bit ops ----------------------------------------------------------------------------------------------
    auto bin = [&](Op o, bool swap = false, bool mask_shift = false) {
        Id a = src(0), c = src(1);
        if (swap) std::swap(a, c);
        if (mask_shift) c = op(Op::OpBitwiseAnd, U, {c, cu(31)});
        return put(op(o, U, {a, c}));
    };
    if (m == "v_and_b32") return bin(Op::OpBitwiseAnd);
    if (m == "v_or_b32") return bin(Op::OpBitwiseOr);
    if (m == "v_xor_b32") return bin(Op::OpBitwiseXor);
    if (m == "v_lshlrev_b32") return bin(Op::OpShiftLeftLogical, true, true);
    if (m == "v_lshrrev_b32") return bin(Op::OpShiftRightLogical, true, true);
    if (m == "v_ashrrev_i32") return bin(Op::OpShiftRightArithmetic, true, true);
    if (m == "v_lshl_b32") return bin(Op::OpShiftLeftLogical, false, true);
    if (m == "v_lshr_b32") return bin(Op::OpShiftRightLogical, false, true);
    if (m == "v_ashr_i32") return bin(Op::OpShiftRightArithmetic, false, true);
    if (m == "v_mul_lo_u32" || m == "v_mul_lo_i32") return bin(Op::OpIMul);
    if (m == "v_rsq_clamp_f32")
        return put(fres(glsl(F, GLSLstd450FClamp, {glsl(F, GLSLstd450InverseSqrt, {fsrc(0)}), cf(-3.40282347e38f), cf(3.40282347e38f)})));
    if (m == "v_mul_hi_u32" || m == "v_mul_hi_i32") {
        const bool sg = m == "v_mul_hi_i32";
        Id t = sg ? I : U;
        Id r = op(sg ? Op::OpSMulExtended : Op::OpUMulExtended, b.t_struct({t, t}),
                  {sg ? op(Op::OpBitcast, I, {src(0)}) : src(0), sg ? op(Op::OpBitcast, I, {src(1)}) : src(1)});
        Id hi = op(Op::OpCompositeExtract, t, {r, 1});
        return put(sg ? op(Op::OpBitcast, U, {hi}) : hi);
    }
    if (m == "v_add_co_u32" || m == "v_sub_co_u32" || m == "v_subrev_co_u32") {
        Id a = src(0), c = src(1);
        if (m == "v_subrev_co_u32") std::swap(a, c);
        Id r = op(m == "v_add_co_u32" ? Op::OpIAdd : Op::OpISub, U, {a, c});
        Id carry = m == "v_add_co_u32" ? op(Op::OpULessThan, B, {r, a}) : op(Op::OpULessThan, B, {a, c});  // carry / borrow
        if (!ok) return false;
        if (!exec_kt()) carry = op(Op::OpLogicalAnd, B, {carry, ld(lane(kExec).var, B)});
        wr_lane(unsigned(v.sdst >= 0 ? v.sdst : int(kVcc)), carry, false);
        return put(r);
    }
    if (m == "v_addc_co_u32" || m == "v_subb_co_u32" || m == "v_subbrev_co_u32") {  // carry in: VCC (VOP2) or SGPR pair src2 (VOP3)
        Id a = src(0), c = src(1);
        if (m == "v_subbrev_co_u32") std::swap(a, c);
        Id ci = rd_lane(in.fmt == GcnFmt::VOP3 ? unsigned(v.s[2]) : unsigned(kVcc), in, &ok);
        if (!ok) return false;
        const bool add = m == "v_addc_co_u32";
        const Op o = add ? Op::OpIAdd : Op::OpISub;
        Id r = op(o, U, {op(o, U, {a, c}), op(Op::OpSelect, U, {ci, cu(1), cu(0)})});
        Id lo = add ? r : a, hi = add ? a : c;  // carry = r < a (add) / a < c (sub); with carry in the comparison includes equality
        Id carry = op(Op::OpSelect, B, {ci, op(Op::OpULessThanEqual, B, {lo, hi}), op(Op::OpULessThan, B, {lo, hi})});
        if (!exec_kt()) carry = op(Op::OpLogicalAnd, B, {carry, ld(lane(kExec).var, B)});
        wr_lane(unsigned(v.sdst >= 0 ? v.sdst : int(kVcc)), carry, false);
        return put(r);
    }
    if (m == "v_lshl_b64" || m == "v_lshlrev_b64") {  // 64-bit value as register pair; inline constants extend to 64 bits
        const int vi = m == "v_lshlrev_b64" ? 1 : 0, si = 1 - vi;
        auto hi_of = [&](int i) -> Id {
            const unsigned code = unsigned(v.s[i]);
            if (code >= 256 || code <= 107) return rd_v(code + 1, in, false, &ok);
            return cu(code >= 193 && code <= 208 ? 0xFFFFFFFFu : 0u);
        };
        Id lo = src(vi), hi = hi_of(vi), n = op(Op::OpBitwiseAnd, U, {src(si), cu(63)});
        if (!ok) return false;
        Id n32 = op(Op::OpBitwiseAnd, U, {n, cu(31)}), back = op(Op::OpBitwiseAnd, U, {op(Op::OpISub, U, {cu(32), n32}), cu(31)});
        Id big = op(Op::OpUGreaterThanEqual, B, {n, cu(32)}), zero = op(Op::OpIEqual, B, {n32, cu(0)});
        Id lo_s = op(Op::OpShiftLeftLogical, U, {lo, n32});
        Id hi_s = op(Op::OpSelect, U, {zero, hi, op(Op::OpBitwiseOr, U, {op(Op::OpShiftLeftLogical, U, {hi, n32}), op(Op::OpShiftRightLogical, U, {lo, back})})});
        wr_v(unsigned(v.vdst), op(Op::OpSelect, U, {big, cu(0), lo_s}));
        wr_v(unsigned(v.vdst) + 1, op(Op::OpSelect, U, {big, lo_s, hi_s}));
        return true;
    }
    // SGPR spill/fill: v_writelane/v_readlane with a compile-time lane index keep the lane's value in a side variable per (vgpr, lane).
    // ponytail: exact only while the lane index is a constant and the VGPR is not also used as ordinary per-lane data.
    if (m == "v_writelane_b32" || m == "v_readlane_b32") {
        Sym sl;
        // VOP2 encoding: the 8-bit src1 field is an SGPR/inline-constant code here (not a VGPR); decoded as 256 + field
        const unsigned lane_code = !v.e64 && v.s[1] >= 256 ? v.s[1] - 256 : v.s[1];
        if (lane_code >= 256) return fail("v_writelane/v_readlane with a VGPR lane index", &in);
        rd_s(lane_code, in, false, &sl, &ok);
        if (!ok) return false;
        if (!sl.known || sl.v.kind != ScalarVal::Const || sl.v.add || sl.v.mask != ~0u) return fail("v_writelane/v_readlane with a non-constant lane index", &in);
        const uint32_t lane_i = sl.v.a & 63;
        const bool wr = m == "v_writelane_b32";
        const unsigned vreg = wr ? unsigned(v.vdst) : (v.s[0] >= 256 ? v.s[0] - 256 : 0);
        if (!wr && v.s[0] < 256) return fail("v_readlane_b32 source is not a VGPR", &in);
        auto key = std::make_pair(vreg, lane_i);
        auto it = spill_.find(key);
        if (it == spill_.end()) it = spill_.emplace(key, b.local_var_init(U, cu(0))).first;
        if (wr) {
            Id x = src(0);
            if (!ok) return false;
            st(it->second, x);
            spill_sym_[key] = v.s[0] < 104 ? sym_[v.s[0]] : Sym{};
        } else {
            auto ks = spill_sym_.find(key);
            wr_s(unsigned(v.vdst), ld(it->second, U), ks != spill_sym_.end() ? ks->second : Sym{});
        }
        return true;
    }
    if (m == "v_min_u32" || m == "v_max_u32") return put(glsl(U, m == "v_min_u32" ? GLSLstd450UMin : GLSLstd450UMax, {src(0), src(1)}));
    if (m == "v_min_i32" || m == "v_max_i32") {
        Id a = op(Op::OpBitcast, I, {src(0)}), c = op(Op::OpBitcast, I, {src(1)});
        return put(op(Op::OpBitcast, U, {glsl(I, m == "v_min_i32" ? GLSLstd450SMin : GLSLstd450SMax, {a, c})}));
    }
    if (m == "v_mul_u32_u24") return put(op(Op::OpIMul, U, {op(Op::OpBitwiseAnd, U, {src(0), cu(0xFFFFFF)}), op(Op::OpBitwiseAnd, U, {src(1), cu(0xFFFFFF)})}));
    if (m == "v_mul_i32_i24" || m == "v_mad_i32_i24") {  // signed 24-bit multiply (low 32 bits)
        auto s24 = [&](Id x) { return op(Op::OpBitFieldSExtract, I, {op(Op::OpBitcast, I, {x}), cu(0), cu(24)}); };
        Id p = op(Op::OpIMul, I, {s24(src(0)), s24(src(1))});
        Id r = op(Op::OpBitcast, U, {p});
        return put(m == "v_mad_i32_i24" ? op(Op::OpIAdd, U, {r, src(2)}) : r);
    }
    if (m == "v_mad_u32_u24")
        return put(op(Op::OpIAdd, U, {op(Op::OpIMul, U, {op(Op::OpBitwiseAnd, U, {src(0), cu(0xFFFFFF)}), op(Op::OpBitwiseAnd, U, {src(1), cu(0xFFFFFF)})}), src(2)}));
    if (m == "v_bfe_u32" || m == "v_bfe_i32") {
        Id a = src(0), off = op(Op::OpBitwiseAnd, U, {src(1), cu(31)}), w = op(Op::OpBitwiseAnd, U, {src(2), cu(31)});
        if (m == "v_bfe_u32") return put(op(Op::OpBitFieldUExtract, U, {a, off, w}));
        return put(op(Op::OpBitcast, U, {op(Op::OpBitFieldSExtract, I, {op(Op::OpBitcast, I, {a}), off, w})}));
    }
    if (m == "v_bfi_b32") {  // (a & b) | (~a & c)
        Id a = src(0), c1 = src(1), c2 = src(2);
        return put(op(Op::OpBitwiseOr, U, {op(Op::OpBitwiseAnd, U, {a, c1}), op(Op::OpBitwiseAnd, U, {op(Op::OpNot, U, {a}), c2})}));
    }
    if (m == "v_bfm_b32") {  // ((1 << (a & 31)) - 1) << (b & 31)
        Id w = op(Op::OpBitwiseAnd, U, {src(0), cu(31)}), off = op(Op::OpBitwiseAnd, U, {src(1), cu(31)});
        return put(op(Op::OpShiftLeftLogical, U, {op(Op::OpISub, U, {op(Op::OpShiftLeftLogical, U, {cu(1), w}), cu(1)}), off}));
    }
    if (m == "v_ldexp_f32") return put(fres(glsl(F, GLSLstd450Ldexp, {fsrc(0), op(Op::OpBitcast, I, {src(1)})})));
    if (m == "v_cvt_pknorm_u16_f32" || m == "v_cvt_pknorm_i16_f32") {
        Id a = fsrc(0), c = fsrc(1);
        return put(glsl(U, m == "v_cvt_pknorm_u16_f32" ? GLSLstd450PackUnorm2x16 : GLSLstd450PackSnorm2x16, {op(Op::OpCompositeConstruct, V2F, {a, c})}));
    }
    if (m == "v_bcnt_u32_b32") return put(op(Op::OpIAdd, U, {op(Op::OpBitCount, U, {src(0)}), src(1)}));
    if (m == "v_mbcnt_lo_u32_b32" || m == "v_mbcnt_hi_u32_b32") {
        // popcount of src0's set bits in the lanes below this one, plus src1. With exec all ones the lo/hi pair (the lane-id
        // idiom: v_mbcnt_lo v, exec_lo, v + v_mbcnt_hi v, exec_hi, 0, s0) counts this lane's own index.
        // ponytail: src0 is exec/ballot-like, so only its bits below the lane are counted; exec is read per the exec model above.
        Id s0 = v.s[0] == 126 || v.s[0] == 127 ? exec_bits() : src(0), s1 = src(1);
        if (!ok) return false;
        const bool hi = m == "v_mbcnt_hi_u32_b32";
        Id lane = lane_index();
        // lo counts lanes [0, min(lane, 32)), hi counts lanes [32, lane); the shift count stays < 32 so the idle
        // side of the select can never shift out of range
        Id n = hi ? op(Op::OpSelect, U, {op(Op::OpUGreaterThan, B, {lane, cu(32)}), op(Op::OpISub, U, {lane, cu(32)}), cu(0)})
                  : op(Op::OpBitwiseAnd, U, {lane, cu(31)});
        Id mask = op(Op::OpISub, U, {op(Op::OpShiftLeftLogical, U, {cu(1), n}), cu(1)});
        if (!hi) mask = op(Op::OpSelect, U, {op(Op::OpUGreaterThanEqual, B, {lane, cu(32)}), cu(~0u), mask});
        return put(op(Op::OpIAdd, U, {op(Op::OpBitCount, U, {op(Op::OpBitwiseAnd, U, {s0, mask})}), s1}));
    }
    if (m == "v_cndmask_b32") {
        Id cond = rd_lane(in.fmt == GcnFmt::VOP3 ? v.s[2] : kVcc, in, &ok);
        Id a = src(0), c = src(1);
        if (!ok) return false;
        return put(op(Op::OpSelect, U, {cond, c, a}));
    }
    return fail("unsupported vector instruction " + m, &in);
}

// ---- DS ------------------------------------------------------------------------------------------------------------
// ds_swizzle_b32: pixel shaders use it for screen-space derivatives and quad broadcasts. Only the quad-local forms are
// supported, mapped to the Vulkan subgroup quad operations (quad lane = (x & 1) | (y & 1) << 1, the same layout as GCN
// lanes 0..3 of a quad). Everything else is an LDS access (lds()).
bool Xlat::ds(const Insn& in) {
    if (in.op != 0x35 || env_.stage != ShStage::PS) return lds(in);
    const uint32_t d = in.w[0], e = in.w[1];
    const uint32_t off = d & 0xFFFF;
    const unsigned addr = e & 0xFF, vdst = e >> 24;
    b.capability(spv::Capability::GroupNonUniformQuad);
    const Id sub = cu(3);  // Subgroup scope
    Id val = ld(vgpr_var(addr), U), res = 0;
    if (off & 0x8000) {  // quad permute: lane q reads lane (perm >> 2q) & 3 of its quad
        const uint32_t perm = off & 0xFF;
        if (!frag_coord_) return fail("ds_swizzle without FragCoord", &in);
        const Id fc = ld(frag_coord_, V4F);
        const Id x = op(Op::OpConvertFToU, U, {op(Op::OpCompositeExtract, F, {fc, 0})});
        const Id y = op(Op::OpConvertFToU, U, {op(Op::OpCompositeExtract, F, {fc, 1})});
        const Id qi = op(Op::OpBitwiseOr, U, {op(Op::OpBitwiseAnd, U, {x, cu(1)}), op(Op::OpShiftLeftLogical, U, {op(Op::OpBitwiseAnd, U, {y, cu(1)}), cu(1)})});
        const Id src = op(Op::OpBitwiseAnd, U, {op(Op::OpShiftRightLogical, U, {cu(perm), op(Op::OpShiftLeftLogical, U, {qi, cu(1)})}), cu(3)});
        res = op(Op::OpGroupNonUniformQuadBroadcast, U, {sub, val, cu(3)});
        for (uint32_t k = 3; k-- > 0;)
            res = op(Op::OpSelect, U, {op(Op::OpIEqual, B, {src, cu(k)}), op(Op::OpGroupNonUniformQuadBroadcast, U, {sub, val, cu(k)}), res});
    } else {  // bit-mask mode: lane' = ((lane & and) | or) ^ xor, supported when it only permutes inside the quad
        const uint32_t am = off & 0x1F, om = (off >> 5) & 0x1F, xm = (off >> 10) & 0x1F;
        if ((am & 0x1F) != 0x1F || (om & 0x1C) || (xm & 0x1C) || (om & 3)) return fail("unsupported ds_swizzle bit-mask mode", &in);
        res = xm & 3 ? op(Op::OpGroupNonUniformQuadSwap, U, {sub, val, cu((xm & 3) - 1)}) : val;  // xor 1/2/3 = horizontal/vertical/diagonal
    }
    wr_v(vdst, res);
    return true;
}

// ---- LDS -----------------------------------------------------------------------------------------------------------
// The GCN LDS is workgroup memory: a Workgroup-storage uint array of GcnEnv::lds_bytes bytes. Word 0 holds offset0[7:0],
// offset1[15:8], the GDS bit (17) and the opcode[25:18], word 1 addr[7:0], data0[15:8], data1[23:16], vdst[31:24].
// Sub-word widths are read out of / merged into the covering dword (LDS is workgroup memory, so the extra traffic is cheap).
// ponytail: relaxed atomics (GCN's LDS atomics are ordered by s_barrier, not by themselves); no 64-bit atomics or rsub.
bool Xlat::lds(const Insn& in) {
    if (in.op == 0x35) return fail("ds_swizzle_b32 outside a pixel shader", &in);
    const uint32_t d = in.w[0], e = in.w[1];
    if ((d >> 17) & 1) return gds(in);
    const bool read_only = (in.op >= 0x36 && in.op <= 0x3C) || (in.op >= 0x76 && in.op <= 0x78);  // the ds_read* forms
    if (is_ds() && !read_only) return fail("LDS write in a domain shader (a DS only reads the records the LS wrote)", &in);
    Id var = lds_var();
    if (!var) return fail("LDS instruction but the shader was compiled without LDS (GcnEnv::lds_bytes = 0)", &in);
    const unsigned off0 = d & 0xFF, off1 = (d >> 8) & 0xFF, iop = in.op;
    const unsigned addr = e & 0xFF, d0 = (e >> 8) & 0xFF, d1 = (e >> 16) & 0xFF, vdst = e >> 24;
    const bool stride64 = iop == 0x38 || iop == 0x0F || iop == 0x4F || iop == 0x78;
    Id base = ld(vgpr_var(addr), U);
    const Id a0 = op(Op::OpIAdd, U, {base, cu(off0)}), a1 = op(Op::OpIAdd, U, {base, cu(stride64 ? off1 * 64 : off1)});
    // Workgroup memory is a bare array, the emulated LDS buffer a runtime array inside a Block struct (one index more)
    const Id pt = b.t_ptr(tess() ? spv::StorageClass::StorageBuffer : spv::StorageClass::Workgroup, U);
    auto cell = [&](Id byte) {
        Id idx = op(Op::OpShiftRightLogical, U, {byte, cu(2)});
        return op(Op::OpAccessChain, pt, tess() ? std::vector<uint32_t>{var, cu(0), idx} : std::vector<uint32_t>{var, idx});
    };
    auto get = [&](Id byte) { return ld(cell(byte), U); };
    auto put = [&](Id byte, Id v) { lds_written_ = true; st(cell(byte), v); };
    auto reg = [&](unsigned r) { return ld(vgpr_var(r), U); };
    Pred p = pred_begin();
    switch (iop) {
        case 0x0D: put(a0, reg(d0)); break;                                                        // ds_write_b32
        case 0x0E: case 0x0F: put(a0, reg(d0)); put(a1, reg(d1)); break;                           // ds_write2(_st64)_b32
        case 0x4D: put(a0, reg(d0)); put(op(Op::OpIAdd, U, {a0, cu(4)}), reg(d0 + 1)); break;       // ds_write_b64
        case 0x1E: case 0x1F: {                                                                    // ds_write_b8/b16
            const uint32_t m = iop == 0x1E ? 0xFF : 0xFFFF;
            put(a0, op(Op::OpBitwiseOr, U, {op(Op::OpBitwiseAnd, U, {get(a0), cu(~m)}), op(Op::OpBitwiseAnd, U, {reg(d0), cu(m)})}));
            break;
        }
        case 0x36: wr_v(vdst, get(a0)); break;                                                    // ds_read_b32
        case 0x37: case 0x38: wr_v(vdst, get(a0)); wr_v(vdst + 1, get(a1)); break;                // ds_read2(_st64)_b32
        case 0x39: case 0x3A: case 0x3B: case 0x3C: {                                              // ds_read_i8/u8/i16/u16
            const unsigned w = iop <= 0x3A ? 8 : 16;
            wr_v(vdst, iop == 0x39 || iop == 0x3B ? op(Op::OpBitcast, U, {op(Op::OpBitFieldSExtract, I, {op(Op::OpBitcast, I, {get(a0)}), cu(0), cu(w)})})
                                                : op(Op::OpBitFieldUExtract, U, {get(a0), cu(0), cu(w)}));
            break;
        }
        case 0x76: case 0x77: case 0x78: {                                                          // ds_read_b64 / ds_read2(_st64)_b64
            const unsigned n = iop == 0x76 ? 1 : 2;
            for (unsigned k = 0; k < n; ++k) {
                Id at = iop == 0x76 || !k ? a0 : a1;
                wr_v(vdst + 2 * k, get(at));
                wr_v(vdst + 2 * k + 1, get(op(Op::OpIAdd, U, {at, cu(4)})));
            }
            break;
        }
        default: {  // ds_{add,sub,inc,dec,min_i32,max_i32,min_u32,max_u32,and,or,xor}_b32[u32] and their _rtn forms (0x00-0x0b, 0x20-0x2b)
            const unsigned k = iop & 0x0F;
            if (k > 0x0B || iop > 0x2B) { pred_end(p); return fail("unsupported DS instruction", &in); }
            const unsigned rtn = iop >= 0x20 ? 1u : 0u;
            Op ao = k == 0x00 || k == 0x03 ? Op::OpAtomicIAdd : k == 0x01 || k == 0x04 ? Op::OpAtomicISub
                   : k == 0x05 ? Op::OpAtomicSMin : k == 0x06 ? Op::OpAtomicSMax : k == 0x07 ? Op::OpAtomicUMin
                   : k == 0x08 ? Op::OpAtomicUMax : k == 0x09 ? Op::OpAtomicAnd : k == 0x0A ? Op::OpAtomicOr : Op::OpAtomicXor;
            const bool sg = k == 0x05 || k == 0x06;
            Id data = k == 0x03 || k == 0x04 ? cu(1) : reg(d0);
            if (sg) data = op(Op::OpBitcast, I, {data});
            Id old = op(ao, sg ? I : U, {cell(a0), cu(uint32_t(spv::Scope::Workgroup)), cu(0), data});
            if (rtn) wr_v(vdst, sg ? op(Op::OpBitcast, U, {old}) : old);
            break;
        }
    }
    pred_end(p);
    return true;
}

// GDS (global data sharing): a host storage buffer that lives across dispatches (Resource::GdsBuffer), addressed in bytes by
// M0[15:0] + the instruction offset. Only ds_append is used by the game: a wave-wide counter bump by the active lane count that
// returns the pre-op value to every lane, which the shader then offsets by v_mbcnt(exec) to get one slot per lane.
// ponytail: one atomic per lane and the lane id subtracted again, so old + mbcnt is still a unique dense slot without wave ops;
// exact for the "append result + mbcnt" idiom, wrong for a shader that reads the result as a wave-uniform value.
bool Xlat::gds(const Insn& in) {
    if (in.op != 0x3E) return fail("GDS instruction other than ds_append is not implemented", &in);
    bool ok = true;
    Id m0 = rd_s(kM0, in, false, nullptr, &ok);
    if (!ok) return false;
    const Id byte = op(Op::OpIAdd, U, {op(Op::OpBitwiseAnd, U, {m0, cu(0xFFFF)}), cu(in.w[0] & 0xFFFF)});
    const Id var = tess_var(2);
    const Id cell = op(Op::OpAccessChain, b.t_ptr(spv::StorageClass::StorageBuffer, U), {var, cu(0), op(Op::OpShiftRightLogical, U, {byte, cu(2)})});
    Pred p = pred_begin();
    const Id old = op(Op::OpAtomicIAdd, U, {cell, cu(uint32_t(spv::Scope::Device)), cu(0), cu(1)});
    wr_v(in.w[1] >> 24, op(Op::OpISub, U, {old, lane_index()}));
    pred_end(p);
    return true;
}

// ---- interpolation / exports ------------------------------------------------------------------------------------------
bool Xlat::vintrp(const Insn& in) {
    const uint32_t d = in.w[0];
    const unsigned vdst = (d >> 18) & 0xFF, attr = (d >> 10) & 0x3F, chan = (d >> 8) & 3;
    if (in.op == 1) return true;  // v_interp_p2_f32: p1 already produced the fully interpolated value
    // v_interp_mov_f32 with P0 (vsrc 2) reads the attribute of the triangle's first vertex: a Flat input (Vulkan's provoking vertex is the
    // first one too). P10/P20 (vertex differences) and mixing flat and interpolated reads of one attribute are not supported.
    const uint32_t cntl = env_.ps_input_map ? env_.ps_input_cntl[attr & 31] : attr;
    if (cntl & 0x20) {  // no VS param behind this input: the DEFAULT_VAL constant
        const uint32_t dv = (cntl >> 8) & 3;
        const bool one = chan == 3 ? (dv & 1) : (dv & 2);
        wr_v(vdst, cu(one ? 0x3F800000u : 0u));
        return true;
    }
    const unsigned loc = cntl & 0x1F;
    const bool flat = in.op == 2 || (cntl & 0x400);
    if (in.op == 2 && (d & 0xFF) != 2) return fail("v_interp_mov_f32 with P10/P20 not implemented", &in);
    if ((flat ? interp_attrs_ : flat_attrs_) & (1ull << loc)) return fail("attribute read both flat and interpolated", &in);
    (flat ? flat_attrs_ : interp_attrs_) |= 1ull << loc;
    T.ps_attrs |= 1u << loc;
    if (!flat && env_.ps_bary) {  // GCN interpolation: P0 + i*(P1-P0) + j*(P2-P0), exact when all three vertices agree
        Id& pv = attr_in_[loc];
        if (!pv) {
            if (!bary_) {
                b.extension("SPV_KHR_fragment_shader_barycentric");
                b.capability(spv::Capability::FragmentBarycentricKHR);
                bary_ = builtin_var(spv::StorageClass::Input, b.t_vec(F, 3), spv::BuiltIn::BaryCoordKHR);
            }
            pv = b.global_var(b.t_ptr(spv::StorageClass::Input, b.t_array(V4F, cu(3))), spv::StorageClass::Input);
            b.decorate(pv, spv::Decoration::Location, {loc});
            b.decorate(pv, spv::Decoration::PerVertexKHR, {});
            iface_.push_back(pv);
        }
        const Id pf = b.t_ptr(spv::StorageClass::Input, F), bc = ld(bary_, b.t_vec(F, 3));
        auto vert = [&](uint32_t k) { return ld(op(Op::OpAccessChain, pf, {pv, cu(k), cu(chan)}), F); };
        const Id p0 = vert(0), d1 = op(Op::OpFSub, F, {vert(1), p0}), d2 = op(Op::OpFSub, F, {vert(2), p0});
        const Id i = op(Op::OpCompositeExtract, F, {bc, 1}), j = op(Op::OpCompositeExtract, F, {bc, 2});
        wr_v(vdst, bits(op(Op::OpFAdd, F, {op(Op::OpFAdd, F, {p0, op(Op::OpFMul, F, {i, d1})}), op(Op::OpFMul, F, {j, d2})})));
        return true;
    }
    Id& var = attr_in_[loc];
    if (!var) {
        var = b.global_var(b.t_ptr(spv::StorageClass::Input, V4F), spv::StorageClass::Input);
        b.decorate(var, spv::Decoration::Location, {loc});
        if (flat) b.decorate(var, spv::Decoration::Flat, {});
        iface_.push_back(var);
    }
    Id v4 = ld(var, V4F);
    wr_v(vdst, bits(op(Op::OpCompositeExtract, F, {v4, chan})));
    return true;
}

bool Xlat::exp(const Insn& in) {
    const uint32_t d = in.w[0], e = in.w[1];
    const unsigned tgt = (d >> 4) & 0x3F, en = d & 15;
    if (is_ls()) return fail("export in an LS: the emulated LS only fills the LDS records the DS reads", &in);
    const bool compr = d & 0x400;
    Pred p = pred_begin();
    auto reg = [&](unsigned k) { return ld(vgpr_var((e >> (8 * k)) & 0xFF), U); };
    auto out_var = [&](std::map<unsigned, Id>& mp, unsigned loc) {
        Id& var = mp[loc];
        if (!var) {
            var = b.global_var(b.t_ptr(spv::StorageClass::Output, V4F), spv::StorageClass::Output);
            b.decorate(var, spv::Decoration::Location, {loc});
            iface_.push_back(var);
        }
        return var;
    };
    Id zero = cf(0.f);
    if (tgt < 8) {
        Id comps[4];
        if (compr) {
            Id lo = glsl(V2F, GLSLstd450UnpackHalf2x16, {reg(0)}), hi = glsl(V2F, GLSLstd450UnpackHalf2x16, {reg(1)});
            comps[0] = op(Op::OpCompositeExtract, F, {lo, 0}); comps[1] = op(Op::OpCompositeExtract, F, {lo, 1});
            comps[2] = op(Op::OpCompositeExtract, F, {hi, 0}); comps[3] = op(Op::OpCompositeExtract, F, {hi, 1});
            if (!(en & 3)) comps[0] = comps[1] = zero;
            if (!(en & 0xC)) comps[2] = comps[3] = zero;
        } else {
            for (unsigned k = 0; k < 4; ++k) comps[k] = (en >> k) & 1 ? flt(reg(k)) : zero;
        }
        Id val = op(Op::OpCompositeConstruct, V4F, {comps[0], comps[1], comps[2], comps[3]});
        if (probe_var_ && tgt == 0) val = op(Op::OpCompositeInsert, V4F, {cf(1.f), ld(probe_var_, V4F), 3});
        st(out_var(mrt_out_, tgt), val);
        T.ps_mrts |= 1u << tgt;
    } else if (is_ls() && (tgt == 12 || (tgt >= 32 && tgt < 64))) {
        return fail("an LS (compute pass) cannot export", &in);
    } else if (tgt == 8) {
        if (!depth_out_) {
            depth_out_ = b.global_var(b.t_ptr(spv::StorageClass::Output, F), spv::StorageClass::Output);
            b.decorate(depth_out_, spv::Decoration::BuiltIn, {uint32_t(spv::BuiltIn::FragDepth)});
            iface_.push_back(depth_out_);
        }
        st(depth_out_, flt(reg(0)));
        T.ps_mrts |= 1u << 8;
    } else if (tgt == 12) {
        Id c[4];
        for (unsigned k = 0; k < 4; ++k) c[k] = (en >> k) & 1 ? flt(reg(k)) : zero;
        st(pos_out_, op(Op::OpCompositeConstruct, V4F, {c[0], c[1], c[2], c[3]}));
    } else if (tgt >= 13 && tgt <= 15) {
        // POS1-3 carry, in this order and only if enabled in PA_CL_VS_OUT_CNTL: the misc vector (x point size, y edge flag, z RT index,
        // w viewport index), clip/cull distances 0-3, 4-7. CLIP_DIST_ENA_i [7:0] / CULL_DIST_ENA_i [15:8] say what distance i is.
        // ponytail: of the misc vector only the point size is used (one viewport, single-layer targets, no wireframe edge flags).
        const uint32_t oc = env_.vs_out_cntl;
        unsigned kinds[3], nk = 0;
        for (unsigned bit = 21; bit <= 23; ++bit) if (oc & (1u << bit)) kinds[nk++] = bit - 21;
        const unsigned k = tgt - 13;
        if (k < nk && kinds[k] == 0 && (oc & (1u << 16)) && (en & 1)) st(psize_out_, flt(reg(0)));
        if (k < nk && kinds[k] != 0) {
            auto dist = [&](Id& var, spv::BuiltIn bi, uint32_t ena, unsigned i, Id v) {
                if (!((ena >> i) & 1)) return;
                if (!var) {
                    b.capability(bi == spv::BuiltIn::ClipDistance ? spv::Capability::ClipDistance : spv::Capability::CullDistance);
                    var = builtin_var(spv::StorageClass::Output, b.t_array(F, cu(uint32_t(std::popcount(ena)))), bi);
                }
                st(op(Op::OpAccessChain, b.t_ptr(spv::StorageClass::Output, F), {var, cu(uint32_t(std::popcount(ena & ((1u << i) - 1))))}), v);
            };
            for (unsigned c = 0; c < 4; ++c) {
                if (!((en >> c) & 1)) continue;
                const unsigned i = (kinds[k] - 1) * 4 + c;
                const Id v = flt(reg(c));
                dist(clip_out_, spv::BuiltIn::ClipDistance, oc & 0xFF, i, v);
                dist(cull_out_, spv::BuiltIn::CullDistance, (oc >> 8) & 0xFF, i, v);
            }
        }
    } else if (tgt >= 32 && tgt < 64) {
        Id c[4];
        for (unsigned k = 0; k < 4; ++k) c[k] = (en >> k) & 1 ? flt(reg(k)) : zero;
        st(out_var(param_out_, tgt - 32), op(Op::OpCompositeConstruct, V4F, {c[0], c[1], c[2], c[3]}));
        T.vs_params |= 1u << (tgt - 32);
    } else {
        pred_end(p);
        return fail("unsupported export target " + std::to_string(tgt), &in);
    }
    pred_end(p);
    // PS: the final export with VM set uses EXEC as the valid mask - pixels whose lane is off there are killed for every MRT (alpha test:
    // the shader clears killed lanes from its saved exec and restores it right before this export). Without this they wrote garbage.
    if (env_.stage == ShStage::PS && tgt < 8 && (d & 0x1000) && !exec_kt()) {
        T.ps_kill = true;
        Id dead = op(Op::OpLogicalNot, B, {ld(lane(kExec).var, B)}), kill_l = b.label(), rest = b.label();
        b.op0(Op::OpSelectionMerge, {rest, 0});
        b.op0(Op::OpBranchConditional, {dead, kill_l, rest});
        b.place_label(kill_l);
        b.op0(Op::OpKill, {});
        b.place_label(rest);
    }
    return true;
}

// ---- buffer memory ----------------------------------------------------------------------------------------------------
namespace {
struct BufFmt { unsigned comps, width; };
bool buf_fmt(unsigned dfmt, BufFmt& f) {
    switch (dfmt) {
        case 1: f = {1, 8}; return true;
        case 2: f = {1, 16}; return true;
        case 3: f = {2, 8}; return true;
        case 4: f = {1, 32}; return true;
        case 5: f = {2, 16}; return true;
        case 10: f = {4, 8}; return true;
        case 11: f = {2, 32}; return true;
        case 12: f = {4, 16}; return true;
        case 13: f = {3, 32}; return true;
        case 14: f = {4, 32}; return true;
        default: return false;
    }
}
}  // namespace

bool Xlat::mubuf(const Insn& in, const std::string& m) {
    const uint32_t d = in.w[0], e = in.w[1];
    const bool offen = (d >> 12) & 1, idxen = (d >> 13) & 1;
    const unsigned imm = d & 0xFFF, vaddr = e & 0xFF, vdata = (e >> 8) & 0xFF, srsrc = ((e >> 16) & 0x1F) * 4, soff = e >> 24;
    const bool store = starts_with(m, "buffer_store") || starts_with(m, "tbuffer_store"), fmt = m.find("_format_") != std::string::npos;
    unsigned comps = 0;
    if (fmt) comps = ends_with(m, "xyzw") ? 4 : ends_with(m, "xyz") ? 3 : ends_with(m, "xy") ? 2 : ends_with(m, "_x") ? 1 : 0;
    else comps = ends_with(m, "dwordx4") ? 4 : ends_with(m, "dwordx2") ? 2 : ends_with(m, "_dword") ? 1 : 0;
    const bool atomic = starts_with(m, "buffer_atomic_") && m.find("_x2") == std::string::npos && m.find("cmpswap") == std::string::npos &&
                        m.find("_inc") == std::string::npos && m.find("_dec") == std::string::npos;  // 32-bit integer atomics
    if (atomic) comps = 1;
    if (!comps || (!atomic && !starts_with(m, "buffer_load") && !starts_with(m, "tbuffer_load") && !store)) return fail("unsupported MUBUF opcode", &in);
    DescRef ref;
    if (!desc_ref(srsrc, 4, ref)) return fail("MUBUF: V# is not a user-data/s_load descriptor", &in);
    const int res = get_resource(Resource::Buffer, ref, 4, store || atomic);
    if (res < 0) return fail("MUBUF: descriptor unreadable", &in);
    T.resources[size_t(res)].vertex_use |= (idxen && !offen && vaddr == 0 && !store && !atomic) ? 1 : 2;
    const uint32_t* w = T.resources[res].words;
    // tbuffer instructions carry the data/number format themselves (dfmt bits 22:19, nfmt 25:23); buffer_*_format use the V#.
    const bool typed = in.fmt == GcnFmt::MTBUF;
    const uint32_t stride = (w[1] >> 16) & 0x3FFF, dfmt = typed ? (d >> 19) & 15 : (w[3] >> 15) & 15, nfmt = typed ? (d >> 23) & 7 : (w[3] >> 12) & 7;
    if (w[0] == ~0u && w[1] == ~0u && w[2] == ~0u && w[3] == ~0u) {  // unbound V# (all ones): loads return 0, stores vanish
        if (!store) for (unsigned k = 0; k < comps; ++k) wr_v(vdata + k, cu(0));
        return true;
    }
    if (fmt && dfmt == 0) {  // BUF_DATA_FORMAT 0 = invalid: loads return 0, stores vanish
        if (!store) for (unsigned k = 0; k < comps; ++k) wr_v(vdata + k, cu(0));
        return true;
    }
    // dfmt 6/7 = 10_11_11 / 11_11_10: three unsigned small floats packed into one dword. AMD names list the fields from the MSB, so X (R)
    // is in the low bits: 10_11_11 is R11G11B10_FLOAT (Vulkan B10G11R11, Mesa ac_formats), 11_11_10 has a 10-bit R and an 11-bit B.
    const bool small = dfmt == 6 || dfmt == 7;
    const unsigned swid[3] = {dfmt == 6 ? 11u : 10u, 11u, dfmt == 6 ? 10u : 11u}, ssh[3] = {0u, swid[0], swid[0] + 11u};
    if (fmt && small && nfmt != 7) return fail("buffer format dfmt=6/7 needs num_format 7", &in);
    BufFmt bf{comps, 32};
    if (fmt && !small && !buf_fmt(dfmt, bf)) {
        char dw[64];
        std::snprintf(dw, sizeof dw, " V#=%08x %08x %08x %08x", w[0], w[1], w[2], w[3]);
        return fail("buffer format dfmt=" + std::to_string(dfmt) + " not implemented" + dw, &in);
    }
    bool ok = true;
    unsigned va = vaddr;
    Id index = idxen ? ld(vgpr_var(va++), U) : 0, voff = offen ? ld(vgpr_var(va++), U) : 0;
    Id bytes = cu(imm);
    if (idxen) bytes = op(Op::OpIAdd, U, {bytes, op(Op::OpIMul, U, {index, cu(stride)})});
    if (offen) bytes = op(Op::OpIAdd, U, {bytes, voff});
    bytes = op(Op::OpIAdd, U, {bytes, rd_s(soff, in, false, nullptr, &ok)});
    if (!ok) return false;
    Id var = buffer_var(res);
    Id pt = b.t_ptr(spv::StorageClass::StorageBuffer, U);
    auto word_ptr = [&](Id dword_index) { return op(Op::OpAccessChain, pt, {var, cu(0), dword_index}); };
    if (atomic) {  // buffer_atomic_<op> vdata, ...: glc set = the pre-op value returns in vdata
        const std::string opn = m.substr(14);
        Op ao = opn == "add" ? Op::OpAtomicIAdd : opn == "sub" ? Op::OpAtomicISub : opn == "smin" ? Op::OpAtomicSMin : opn == "umin" ? Op::OpAtomicUMin
              : opn == "smax" ? Op::OpAtomicSMax : opn == "umax" ? Op::OpAtomicUMax : opn == "and" ? Op::OpAtomicAnd : opn == "or" ? Op::OpAtomicOr
              : opn == "xor" ? Op::OpAtomicXor : opn == "swap" ? Op::OpAtomicExchange : Op::OpNop;
        if (ao == Op::OpNop) return fail("unsupported MUBUF atomic " + m, &in);
        Pred pa = pred_begin();
        Id old = op(ao, U, {word_ptr(op(Op::OpShiftRightLogical, U, {bytes, cu(2)})), cu(1), cu(0), ld(vgpr_var(vdata), U)});  // device scope, relaxed
        if ((d >> 14) & 1) st(vgpr_var(vdata), old);
        pred_end(pa);
        return true;
    }
    const unsigned elem_bits = bf.comps * bf.width;
    const uint32_t one_bits = nfmt == 7 ? 0x3F800000u : 1u;
    const unsigned swz[4] = {(w[3] >> 0) & 7, (w[3] >> 3) & 7, (w[3] >> 6) & 7, (w[3] >> 9) & 7};
    Pred p = pred_begin();
    if (!store) {
        // Raw (non-format) loads read dwords; format loads decode each stored channel then apply the V# dst_sel swizzle.
        std::vector<Id> chan(4, 0);
        if (!fmt) {
            Id dw = op(Op::OpShiftRightLogical, U, {bytes, cu(2)});
            for (unsigned k = 0; k < comps; ++k) st(vgpr_var(vdata + k), ld(word_ptr(op(Op::OpIAdd, U, {dw, cu(k)})), U));
            pred_end(p);
            return true;
        }
        if (small) {  // one dword, three packed channels; the dst_sel swizzle below applies as usual
            Id word = ld(word_ptr(op(Op::OpShiftRightLogical, U, {bytes, cu(2)})), U);
            for (unsigned k = 0; k < 3 && k < comps; ++k)
                chan[k] = bits(sf_unpack(op(Op::OpBitFieldUExtract, U, {word, cu(ssh[k]), cu(swid[k])}), swid[k]));
        } else {
            const uint32_t maxv = bf.width == 32 ? 0xFFFFFFFFu : (1u << bf.width) - 1;
            for (unsigned k = 0; k < bf.comps; ++k) {
                Id bitpos = op(Op::OpIAdd, U, {op(Op::OpShiftLeftLogical, U, {bytes, cu(3)}), cu(k * bf.width)});
                Id dwv = ld(word_ptr(op(Op::OpShiftRightLogical, U, {bitpos, cu(5)})), U);
                Id raw = bf.width == 32 ? dwv : op(Op::OpBitFieldUExtract, U, {dwv, op(Op::OpBitwiseAnd, U, {bitpos, cu(31)}), cu(bf.width)});
                Id val = raw;
                if (bf.width < 32 && nfmt != 4 && nfmt != 5 && nfmt != 7 && nfmt != 0) { pred_end(p); return fail("buffer num_format " + std::to_string(nfmt) + " not implemented", &in); }
                if (nfmt == 7) {  // float
                    if (bf.width == 16) val = bits(op(Op::OpCompositeExtract, F, {glsl(V2F, GLSLstd450UnpackHalf2x16, {raw}), 0}));
                    else if (bf.width != 32) { pred_end(p); return fail("8-bit float channel", &in); }
                } else if (nfmt == 0 && bf.width < 32) {  // unorm
                    val = bits(op(Op::OpFDiv, F, {op(Op::OpConvertUToF, F, {raw}), cf(float(maxv))}));
                } else if (nfmt == 5 && bf.width < 32) {  // sint: sign-extend
                    val = op(Op::OpBitcast, U, {op(Op::OpBitFieldSExtract, I, {op(Op::OpBitcast, I, {raw}), cu(0), cu(bf.width)})});
                }
                chan[k] = val;
            }
        }
        for (unsigned k = 0; k < comps; ++k) {
            const unsigned sel = swz[k];
            Id v;
            if (sel == 0) v = cu(0);
            else if (sel == 1) v = cu(one_bits);
            else if (sel >= 4 && sel - 4 < bf.comps) v = chan[sel - 4];
            else v = cu(sel == 7 ? one_bits : 0);  // channel not present in the format: 0,0,0,1
            st(vgpr_var(vdata + k), v);
        }
        pred_end(p);
        return true;
    }
    // ---- stores ----------------------------------------------------------------------------------------------------------
    if (!fmt) {
        Id dw = op(Op::OpShiftRightLogical, U, {bytes, cu(2)});
        for (unsigned k = 0; k < comps; ++k) st(word_ptr(op(Op::OpIAdd, U, {dw, cu(k)})), ld(vgpr_var(vdata + k), U));
        pred_end(p);
        return true;
    }
    const unsigned n = bf.comps < comps ? bf.comps : comps;
    if (small) {  // merge the written channels into the dword (a partial _x/_xy store keeps the others)
        const unsigned ns = comps < 3 ? comps : 3;
        Id dw = op(Op::OpShiftRightLogical, U, {bytes, cu(2)});
        Id word = ld(word_ptr(dw), U);
        for (unsigned k = 0; k < ns; ++k) {
            Id mask = op(Op::OpShiftLeftLogical, U, {cu((1u << swid[k]) - 1), cu(ssh[k])});
            Id enc = op(Op::OpShiftLeftLogical, U, {sf_pack(ld(vgpr_var(vdata + k), U), swid[k]), cu(ssh[k])});
            word = op(Op::OpBitwiseOr, U, {op(Op::OpBitwiseAnd, U, {word, op(Op::OpNot, U, {mask})}), enc});
        }
        st(word_ptr(dw), word);
        pred_end(p);
        return true;
    }
    const uint32_t maxv = bf.width == 32 ? 0xFFFFFFFFu : (1u << bf.width) - 1;
    std::vector<Id> encs(n);
    for (unsigned k = 0; k < n; ++k) {
        Id v = ld(vgpr_var(vdata + k), U), enc = v;
        if (bf.width < 32) {
            if (nfmt == 7 && bf.width == 16) enc = op(Op::OpBitwiseAnd, U, {glsl(U, GLSLstd450PackHalf2x16, {op(Op::OpCompositeConstruct, V2F, {flt(v), cf(0.f)})}), cu(0xFFFF)});
            else if (nfmt == 0) {
                Id f = glsl(F, GLSLstd450FClamp, {flt(v), cf(0.f), cf(1.f)});
                enc = op(Op::OpConvertFToU, U, {glsl(F, GLSLstd450RoundEven, {op(Op::OpFMul, F, {f, cf(float(maxv))})})});
            } else if (nfmt == 4) enc = glsl(U, GLSLstd450UMin, {v, cu(maxv)});
            else { pred_end(p); return fail("buffer store num_format " + std::to_string(nfmt) + " not implemented", &in); }
        }
        encs[k] = enc;
    }
    if (bf.width < 32 && n == bf.comps && (bf.comps * bf.width) % 32 == 0) {
        // whole elements made of full dwords: assemble each dword and store it without a read-modify-write
        // (the RMW loads from guest memory, which is slow when the buffer is imported host memory)
        Id dw0 = op(Op::OpShiftRightLogical, U, {bytes, cu(2)});
        for (unsigned dwi = 0; dwi < bf.comps * bf.width / 32; ++dwi) {
            Id word = 0;
            for (unsigned k = 0; k < n; ++k)
                if (k * bf.width / 32 == dwi) {
                    Id part = op(Op::OpShiftLeftLogical, U, {encs[k], cu(k * bf.width % 32)});
                    word = word ? op(Op::OpBitwiseOr, U, {word, part}) : part;
                }
            st(word_ptr(op(Op::OpIAdd, U, {dw0, cu(dwi)})), word);
        }
    } else {
        for (unsigned k = 0; k < n; ++k) {
            Id bitpos = op(Op::OpIAdd, U, {op(Op::OpShiftLeftLogical, U, {bytes, cu(3)}), cu(k * bf.width)});
            Id ptr = word_ptr(op(Op::OpShiftRightLogical, U, {bitpos, cu(5)}));
            if (bf.width == 32) st(ptr, encs[k]);
            else {
                Id sh = op(Op::OpBitwiseAnd, U, {bitpos, cu(31)});
                Id mask = op(Op::OpShiftLeftLogical, U, {cu(maxv), sh});
                Id merged = op(Op::OpBitwiseOr, U, {op(Op::OpBitwiseAnd, U, {ld(ptr, U), op(Op::OpNot, U, {mask})}), op(Op::OpShiftLeftLogical, U, {encs[k], sh})});
                st(ptr, merged);
            }
        }
    }
    (void)elem_bits;
    pred_end(p);
    return true;
}

// ---- image memory -----------------------------------------------------------------------------------------------------
bool Xlat::mimg(const Insn& in, const std::string& m) {
    const uint32_t d = in.w[0], e = in.w[1];
    const unsigned dmask = (d >> 8) & 0xF, vaddr = e & 0xFF, vdata = (e >> 8) & 0xFF, sres = ((e >> 16) & 0x1F) * 4, ssmp = ((e >> 21) & 0x1F) * 4;
    const bool is_store = starts_with(m, "image_store"), is_load = starts_with(m, "image_load"), is_sample = starts_with(m, "image_sample");
    if (!is_store && !is_load && !is_sample) return fail("unsupported MIMG opcode", &in);
    DescRef tref;
    if (!desc_ref(sres, 8, tref)) return fail("MIMG: T# is not a user-data/s_load descriptor", &in);
    const int res = get_resource(is_store ? Resource::StorageImage : Resource::Image, tref, 8, is_store);
    if (res < 0) return fail("MIMG: descriptor unreadable", &in);
    const Resource& r = T.resources[res];
    const uint32_t type = r.words[3] >> 28, nfmt = (r.words[1] >> 26) & 15;
    const bool null_t = type == 15 && r.words[3] == ~0u;  // unbound T# (all ones): loads/samples return 0, stores vanish
    if (null_t) {
        if (!is_store) for (unsigned c = 0, k = 0; c < 4; ++c) if ((dmask >> c) & 1) wr_v(vdata + k++, cu(0));
        return true;
    }
    const unsigned ncoord = type == 8 ? 1 : type == 9 ? 2 : type == 10 ? 3 : type == 11 ? 3 : type == 12 ? 2 : type == 13 ? 3 : 0;
    if (!ncoord) return fail("image type " + std::to_string(type) + " not implemented", &in);
    Id itype = image_type_of(r, is_store);
    Id ivar = image_var(res);
    const Id comp_t = nfmt == 4 ? U : nfmt == 5 ? I : F;
    const Id vec4_t = b.t_vec(comp_t, 4);
    unsigned va = vaddr;
    Pred p = pred_begin();
    // cube sampling: GCN takes (s, t, face) after v_cube*; Vulkan wants a direction. s,t arrive as coord/(2|ma|) + 1.5.
    auto cube_dir = [&](Id s_, Id t_, Id face) {
        Id sc = op(Op::OpFMul, F, {op(Op::OpFSub, F, {s_, cf(1.5f)}), cf(2.f)}), tc = op(Op::OpFMul, F, {op(Op::OpFSub, F, {t_, cf(1.5f)}), cf(2.f)});
        Id fi = op(Op::OpConvertFToS, I, {glsl(F, GLSLstd450RoundEven, {face})});
        auto neg = [&](Id v) { return op(Op::OpFNegate, F, {v}); };
        auto pick = [&](Id v0, Id v1, Id v2, Id v3, Id v4, Id v5) {
            Id r = v5;
            Id vs[5] = {v4, v3, v2, v1, v0};
            for (int k = 4; k >= 0; --k) r = op(Op::OpSelect, F, {op(Op::OpIEqual, B, {fi, b.c_i32(k)}), vs[4 - k], r});
            return r;
        };
        Id one = cf(1.f), m1 = cf(-1.f);
        Id x = pick(one, m1, sc, sc, sc, neg(sc));
        Id y = pick(neg(tc), neg(tc), one, m1, neg(tc), neg(tc));
        Id z = pick(neg(sc), sc, tc, neg(tc), one, m1);
        return op(Op::OpCompositeConstruct, b.t_vec(F, 3), {x, y, z});
    };
    // Padded render targets sampled in place (the backend binds e.g. a 1920x1088 target for a 1920x1080 T#): normalized coordinates
    // address the T# size, so they are scaled by T# size / image size and clamped to the centre of the last T# texel, which is what
    // clamp-to-edge samples there (the backend binds a larger image only for clamp-to-edge axes). Texel space is unchanged, so LOD,
    // filtering and the min/mag choice match a T#-sized image. An image of the T# size (s = 1) keeps the coordinates exactly.
    // BB_RES_SCALE (GcnEnv::res_scale): is this image s times its guest size (push-constant bit `res`)? Its guest texel / size factor.
    const bool scalable = env_.res_scale > 1 && (type == 9 || type == 13);
    Id sc_img = 0;
    auto sc_bit = [&]() -> Id {
        if (!sc_img) sc_img = res < 31 ? op(Op::OpINotEqual, B, {op(Op::OpBitwiseAnd, U, {sc_info_, cu(1u << res)}), cu(0)}) : b.c_bool(false);
        return sc_img;
    };
    auto sc_f = [&]() { return op(Op::OpSelect, F, {sc_bit(), cf(float(env_.res_scale)), cf(1.f)}); };
    // integer coordinates of a scaled image: guest texel c -> c * s + (dx, dy) in x and y
    auto sc_int = [&](Id v, Id dx, Id dy) -> Id {
        std::vector<uint32_t> o;
        for (unsigned k = 0; k < ncoord; ++k) {
            Id x = op(Op::OpCompositeExtract, I, {v, k});
            if (k < 2) x = op(Op::OpSelect, I, {sc_bit(), op(Op::OpIAdd, I, {op(Op::OpIMul, I, {x, b.c_i32(int32_t(env_.res_scale))}), k ? dy : dx}), x});
            o.push_back(x);
        }
        return op(Op::OpCompositeConstruct, b.t_vec(I, ncoord), o);
    };
    Id pad_s[2] = {0, 0}, pad_m[2] = {0, 0}, pad_on[2] = {0, 0};
    auto pad = [&](Id c, bool clamp) -> Id {
        if (!pad_s[0]) {
            b.capability(spv::Capability::ImageQuery);
            Id w2 = rd_s(sres + 2, in, false, nullptr, nullptr);  // T# word 2 at run time: WIDTH-1 [13:0], HEIGHT-1 [27:14]
            Id q = op(Op::OpImageQuerySizeLod, b.t_vec(I, type == 13 ? 3 : 2), {ld(ivar, itype), b.c_i32(0)});
            for (unsigned a = 0; a < 2; ++a) {
                Id tw = op(Op::OpBitwiseAnd, U, {a ? op(Op::OpShiftRightLogical, U, {w2, cu(14)}) : w2, cu(0x3FFF)});
                Id t = op(Op::OpConvertUToF, F, {op(Op::OpIAdd, U, {tw, cu(1)})});
                if (scalable) t = op(Op::OpFMul, F, {t, sc_f()});  // (a scaled image holds the T# picture at s times its size)
                Id qd = op(Op::OpConvertSToF, F, {op(Op::OpCompositeExtract, I, {q, a})});
                pad_s[a] = op(Op::OpFDiv, F, {t, qd});
                pad_m[a] = op(Op::OpFDiv, F, {op(Op::OpFSub, F, {t, cf(0.5f)}), qd});
                pad_on[a] = op(Op::OpFOrdLessThan, B, {t, qd});
            }
        }
        std::vector<uint32_t> o;
        for (unsigned k = 0; k < ncoord; ++k) {
            Id x = op(Op::OpCompositeExtract, F, {c, k});
            if (k < 2) {
                Id y = op(Op::OpFMul, F, {x, pad_s[k]});
                if (clamp) y = glsl(F, GLSLstd450FMin, {y, pad_m[k]});
                x = op(Op::OpSelect, F, {pad_on[k], y, x});
            }
            o.push_back(x);
        }
        return op(Op::OpCompositeConstruct, b.t_vec(F, ncoord), o);
    };
    const bool padable = type == 9 || type == 13;  // 2D / 2D array
    // *_o: the first address VGPR packs texel offsets x [5:0], y [13:8], z [21:16] (6-bit signed). Vulkan allows a non-constant Offset
    // only on gathers, so the coordinates move by offset / size of the level sampled: lz 0, l the LOD VGPR, implicit (PS) the level
    // OpImageQueryLod picks. ponytail: nearest level only (a trilinear blend uses one texel size for both levels), bias ignored for the
    // level, _d and VS implicit use level 0; on padded targets the image size (not the T# size) scales the offset (< 1 % off).
    Id off = 0, off_simg = 0;
    std::string off_mode;  // "lz" / "l" / "i" (implicit)
    auto offset = [&](Id v) {
        b.capability(spv::Capability::ImageQuery);
        Id lvl = b.c_i32(0);
        if (off_mode == "l" || (off_mode == "i" && env_.stage == ShStage::PS)) {
            Id lf = off_mode == "l" ? flt(ld(vgpr_var(va), U))
                                    : op(Op::OpCompositeExtract, F, {op(Op::OpImageQueryLod, V2F, {off_simg, v}), 0});
            lvl = op(Op::OpConvertFToS, I, {glsl(F, GLSLstd450Floor, {op(Op::OpFAdd, F, {lf, cf(0.5f)})})});
            lvl = glsl(I, GLSLstd450SMax, {lvl, b.c_i32(0)});
        }
        const Id q = op(Op::OpImageQuerySizeLod, ncoord == 1 ? I : b.t_vec(I, ncoord), {ld(ivar, itype), lvl});
        const unsigned dims = type == 8 || type == 12 ? 1 : type == 10 ? 3 : 2;  // 1D (array), 3D, 2D (array); slices are not offset
        std::vector<uint32_t> o;
        for (unsigned k = 0; k < ncoord; ++k) {
            Id x = ncoord == 1 ? v : op(Op::OpCompositeExtract, F, {v, k});
            if (k < dims) {
                Id t = op(Op::OpConvertSToF, F, {op(Op::OpBitFieldSExtract, I, {op(Op::OpBitcast, I, {off}), cu(8 * k), cu(6)})});
                Id sz = op(Op::OpConvertSToF, F, {ncoord == 1 ? q : op(Op::OpCompositeExtract, I, {q, k})});
                if (scalable) sz = op(Op::OpFDiv, F, {sz, sc_f()});  // (offsets are guest texels)
                x = op(Op::OpFAdd, F, {x, op(Op::OpFDiv, F, {t, sz})});
            }
            o.push_back(x);
        }
        return ncoord == 1 ? o[0] : op(Op::OpCompositeConstruct, b.t_vec(F, ncoord), o);
    };
    auto coordv = [&](bool as_float) {
        std::vector<uint32_t> c;
        for (unsigned k = 0; k < ncoord; ++k) {
            Id u = ld(vgpr_var(va + k), U);
            c.push_back(as_float ? flt(u) : op(Op::OpBitcast, I, {u}));
        }
        va += ncoord;
        if (type == 11 && as_float) return cube_dir(uint32_t(c[0]), uint32_t(c[1]), uint32_t(c[2]));
        const Id ct = as_float ? F : I;
        Id v = ncoord == 1 ? c[0] : op(Op::OpCompositeConstruct, b.t_vec(ct, ncoord), c);
        if (as_float && off) v = offset(v);
        return as_float && padable ? pad(v, true) : v;
    };
    if (is_store) {
        Id coord = coordv(false);
        if (starts_with(m, "image_store_mip")) return fail("image_store_mip not implemented", &in);
        std::vector<uint32_t> t;
        unsigned k = 0;
        for (unsigned c = 0; c < 4; ++c) {
            if ((dmask >> c) & 1) {
                Id u = ld(vgpr_var(vdata + k++), U);
                t.push_back(comp_t == F ? flt(u) : comp_t == I ? op(Op::OpBitcast, I, {u}) : u);
            } else t.push_back(comp_t == F ? cf(0.f) : comp_t == I ? b.c_i32(0) : cu(0));
        }
        b.capability(spv::Capability::StorageImageWriteWithoutFormat);
        Id img = ld(ivar, itype);
        if (!scalable) b.op0(Op::OpImageWrite, {img, coord, op(Op::OpCompositeConstruct, vec4_t, t)});
        else {  // the guest texel's s x s block; a PS into scaled targets writes its own sub-pixel (sc_sub_) s * s times instead
            const Id val = op(Op::OpCompositeConstruct, vec4_t, t);
            for (uint32_t j = 0; j < env_.res_scale * env_.res_scale; ++j) {
                Id dx = op(Op::OpSelect, I, {sc_t_, sc_sub_[0], b.c_i32(int32_t(j % env_.res_scale))});
                Id dy = op(Op::OpSelect, I, {sc_t_, sc_sub_[1], b.c_i32(int32_t(j / env_.res_scale))});
                b.op0(Op::OpImageWrite, {img, sc_int(coord, dx, dy), val});
            }
        }
        pred_end(p);
        return true;
    }
    Id img = ld(ivar, itype);
    Id result = 0;
    if (is_load) {
        Id coord = coordv(false);
        if (scalable) coord = sc_int(coord, sc_sub_[0], sc_sub_[1]);  // (the fragment's own sub-pixel: a guest texel of its own pixel is its own physical one)
        if (m == "image_load_mip") {
            Id lod = op(Op::OpBitcast, I, {ld(vgpr_var(va), U)});
            result = op(Op::OpImageFetch, vec4_t, {img, coord, 0x2, lod});
        } else if (m == "image_load") result = op(Op::OpImageFetch, vec4_t, {img, coord});
        else { pred_end(p); return fail("unsupported image load variant", &in); }
    } else {
        DescRef sref;
        if (!desc_ref(ssmp, 4, sref)) { pred_end(p); return fail("MIMG: S# is not a user-data/s_load descriptor", &in); }
        const int sres_i = get_resource(Resource::Sampler, sref, 4, false);
        if (sres_i < 0) { pred_end(p); return fail("MIMG: sampler unreadable", &in); }
        int16_t& rs = T.resources[res].sampler;  // (the backend binds a padded target in place only for clamp-to-edge samplers)
        rs = rs == -1 || rs == sres_i ? int16_t(sres_i) : int16_t(-2);
        Id smp = ld(sampler_var(sres_i), b.t_sampler());
        Id sit = b.t_sampled_image(itype);
        Id simg = op(Op::OpSampledImage, sit, {img, smp});
        const bool has_off = ends_with(m, "_o");
        const std::string mb = has_off ? m.substr(0, m.size() - 2) : m;  // the variant without the offset
        if (has_off) {
            if (type == 11) { pred_end(p); return fail("image sample offset on a cube", &in); }
            off = ld(vgpr_var(va++), U);  // address order: offset, bias, z-compare, gradients, coordinates, LOD
            off_simg = simg;
            off_mode = ends_with(mb, "_lz") ? "lz" : ends_with(mb, "_l") ? "l" : ends_with(mb, "_d") ? "lz" : "i";
        }
        if (mb == "image_sample") result = op(Op::OpImageSampleImplicitLod, vec4_t, {simg, coordv(true)});
        else if (mb == "image_sample_b") {
            Id bias = flt(ld(vgpr_var(va++), U));
            result = op(Op::OpImageSampleImplicitLod, vec4_t, {simg, coordv(true), 0x1, bias});
        } else if (mb == "image_sample_l") {
            Id c = coordv(true);
            result = op(Op::OpImageSampleExplicitLod, vec4_t, {simg, c, 0x2, flt(ld(vgpr_var(va), U))});
        } else if (mb == "image_sample_lz") result = op(Op::OpImageSampleExplicitLod, vec4_t, {simg, coordv(true), 0x2, cf(0.f)});
        else if (mb == "image_sample_d" && type != 11) {  // explicit gradients: dx(ncoord), dy(ncoord), then the coordinates
            const Id gt = ncoord == 1 ? F : b.t_vec(F, ncoord);
            auto grad = [&]() {
                std::vector<uint32_t> g;
                for (unsigned k = 0; k < ncoord; ++k) g.push_back(flt(ld(vgpr_var(va + k), U)));
                va += ncoord;
                return ncoord == 1 ? g[0] : op(Op::OpCompositeConstruct, gt, g);
            };
            Id dx = grad(), dy = grad();
            if (padable) dx = pad(dx, false), dy = pad(dy, false);
            result = op(Op::OpImageSampleExplicitLod, vec4_t, {simg, coordv(true), 0x4, dx, dy});
        } else if (starts_with(mb, "image_sample_c") && type != 11 && comp_t == F &&
                   (mb == "image_sample_c" || mb == "image_sample_c_lz" || mb == "image_sample_c_l" || mb == "image_sample_c_b")) {
            // depth compare (the S# compare function becomes a Vulkan compare sampler): [bias] z_ref coords [lod]
            Id bias = mb == "image_sample_c_b" ? flt(ld(vgpr_var(va++), U)) : 0;
            Id dref = flt(ld(vgpr_var(va++), U)), c = coordv(true), cmp;
            if (mb == "image_sample_c") cmp = op(Op::OpImageSampleDrefImplicitLod, F, {simg, c, dref});
            else if (mb == "image_sample_c_b") cmp = op(Op::OpImageSampleDrefImplicitLod, F, {simg, c, dref, 0x1, bias});
            else cmp = op(Op::OpImageSampleDrefExplicitLod, F, {simg, c, dref, 0x2, mb == "image_sample_c_lz" ? cf(0.f) : flt(ld(vgpr_var(va), U))});
            result = op(Op::OpCompositeConstruct, vec4_t, {cmp, cmp, cmp, cmp});
        }
        else { pred_end(p); return fail("unsupported image sample variant " + m, &in); }
    }
    unsigned k = 0;
    for (unsigned c = 0; c < 4; ++c)
        if ((dmask >> c) & 1) {
            Id x = op(Op::OpCompositeExtract, comp_t, {result, c});
            st(vgpr_var(vdata + k++), comp_t == U ? x : op(Op::OpBitcast, U, {x}));
        }
    pred_end(p);
    return true;
}

// ---- driver -----------------------------------------------------------------------------------------------------------
// ---- loops -------------------------------------------------------------------------------------------------------------
// A backward s_branch / s_cbranch_* closes a loop whose header is the branch target. Each invocation runs its own copy of the loop
// (exec masking inside the body already makes inactive lanes do nothing), so the loop becomes a structured SPIR-V loop: the exit
// branch (forward, just past the back edge) is a break, the back edge a continue. Scalar/vector state lives in variables, so no
// phi nodes are needed; only compile-time knowledge about registers (constants, exec == all ones) is dropped at the header.
// ponytail: cross-lane operations inside a loop (readlane, ds_swizzle, barriers) see each lane at its own iteration; exact only for
// loops whose trip count is uniform. Per-lane copies of a loop with divergent trip counts need subgroup-aware execution.
void Xlat::scan_loops() {
    for (size_t pc = 0; pc < n_;) {
        const Insn in = gcn::decode(code_ + pc, n_ - pc);
        if (in.fmt == GcnFmt::Invalid) break;
        if (in.fmt == GcnFmt::SOPP) {
            const char* mc = gcn::mnemonic(in.fmt, in.op);
            const std::string m = mc ? mc : "";
            const int off = int16_t(in.w[0] & 0xFFFF);
            if (off < 0 && (m == "s_branch" || starts_with(m, "s_cbranch_"))) {
                const int64_t target = int64_t(pc) + in.len + off;
                if (target >= 0) loop_heads_[size_t(target)].push_back(pc);
            }
        }
        pc += in.len;
    }
}

bool Xlat::begin_loop() {
    std::vector<size_t> backs = loop_heads_[pc_];
    std::sort(backs.begin(), backs.end(), [](size_t a, size_t c) { return a > c; });  // outermost (farthest back edge) first
    for (size_t back : backs) {
        if (!loops_.empty() && back > loops_.back().back_pc) return fail("loop crosses an enclosing loop");
        if (!joins_.empty()) return fail("loop inside a scalar-branch region");
        if (!scc_) scc_ = b.local_var_init(B, b.c_bool(false));  // a back edge may test SCC set later in the body
        Loop l;
        l.back_pc = back;
        l.header = b.label(); l.body = b.label(); l.cont = b.label(); l.merge = b.label();
        b.op0(Op::OpBranch, {l.header});
        b.place_label(l.header);
        b.op0(Op::OpLoopMerge, {l.merge, l.cont, 0});
        b.op0(Op::OpBranch, {l.body});
        b.place_label(l.body);
        for (unsigned i = 0; i < 128; ++i) l.syms[i] = sym_[i];
        for (auto& [s, ln] : lm_) ln.kt = false;
        for (unsigned i = 0; i < 128; ++i) if (sym_[i].known && sym_[i].v.kind == ScalarVal::Const) sym_[i] = {};
        loops_.push_back(l);
    }
    return true;
}

bool Xlat::loop_cond(const Insn& in, const std::string& m, Id* taken) {
    if (m == "s_cbranch_scc0" || m == "s_cbranch_scc1") {
        if (!scc_) return fail("SCC read before any write", &in);
        Id v = ld(scc_, B);
        *taken = m == "s_cbranch_scc1" ? v : op(Op::OpLogicalNot, B, {v});
        return true;
    }
    const bool vcc = m == "s_cbranch_vccz" || m == "s_cbranch_vccnz", z = m == "s_cbranch_vccz" || m == "s_cbranch_execz";
    if (!vcc && m != "s_cbranch_execz" && m != "s_cbranch_execnz") return fail("unsupported control flow", &in);
    bool ok = true;
    Id v = rd_lane(vcc ? kVcc : kExec, in, &ok);
    if (!ok) return false;
    *taken = z ? op(Op::OpLogicalNot, B, {v}) : v;
    return true;
}

bool Xlat::end_loop(const Insn& in, const std::string& m) {
    const Loop l = loops_.back();
    if (!joins_.empty()) return fail("loop back edge inside a scalar-branch region", &in);
    // A register whose symbolic value changes in the body is only a problem if the body used that value symbolically (it would be wrong from the
    // second iteration on); otherwise it simply becomes unknown after the loop (e.g. data loaded at a loop-dependent offset into a former T# slot).
    std::bitset<128> changed;
    for (unsigned i = 0; i < 128; ++i) {
        const Sym &h = l.syms[i], &s = sym_[i];
        if (h.known && h.v.kind != ScalarVal::Const && (!s.known || s.v.kind != h.v.kind || s.v.a != h.v.a || s.v.b != h.v.b || s.v.add != h.v.add || s.v.mask != h.v.mask)) {
            if (l.sym_reads.test(i)) return fail("loop changes a descriptor register", &in);
            changed.set(i);
        }
    }
    Id taken = 0;
    if (m != "s_branch" && !loop_cond(in, m, &taken)) return false;
    b.op0(Op::OpBranch, {l.cont});
    b.place_label(l.cont);
    if (m == "s_branch") b.op0(Op::OpBranch, {l.header});
    else b.op0(Op::OpBranchConditional, {taken, l.header, l.merge});
    b.place_label(l.merge);
    for (unsigned i = 0; i < 128; ++i) if ((sym_[i].known && sym_[i].v.kind == ScalarVal::Const) || changed.test(i)) sym_[i] = {};
    for (auto& [s, ln] : lm_) ln.kt = false;
    const std::bitset<128> reads = l.sym_reads;
    loops_.pop_back();
    if (!loops_.empty()) loops_.back().sym_reads |= reads;  // an outer loop re-runs this body: its reads count there too
    return true;
}

bool Xlat::break_loop(const Insn& in, const std::string& m, size_t target) {
    const Loop& l = loops_.back();
    const Insn back = gcn::decode(code_ + l.back_pc, n_ - l.back_pc);
    if (target != l.back_pc + back.len) return fail("loop exit does not land behind the back edge", &in);
    if (m == "s_branch") {
        b.op0(Op::OpBranch, {l.merge});
        b.place_label(b.label());  // unreachable remainder of the body
        return true;
    }
    Id cond = 0;
    if (!loop_cond(in, m, &cond)) return false;
    Id brk = b.label(), rest = b.label();
    b.op0(Op::OpSelectionMerge, {rest, 0});
    b.op0(Op::OpBranchConditional, {cond, brk, rest});
    b.place_label(brk);
    b.op0(Op::OpBranch, {l.merge});
    b.place_label(rest);
    return true;
}

bool Xlat::step(const Insn& in) {
    const char* mc = gcn::mnemonic(in.fmt, in.op);
    const std::string m = mc ? mc : "";
    if (in.fmt == GcnFmt::SOPP && !ret_code_ && (m == "s_branch" || starts_with(m, "s_cbranch_"))) {
        const int off = int16_t(in.w[0] & 0xFFFF);
        if (!loops_.empty() && pc_ == loops_.back().back_pc) return end_loop(in, m);
        if (!loops_.empty() && off >= 0 && pc_ + in.len + size_t(off) > loops_.back().back_pc) return break_loop(in, m, pc_ + in.len + size_t(off));
    }
    switch (in.fmt) {
        case GcnFmt::SOPP: {
            if (m == "s_endpgm") { ended_ = true; return true; }
            if (m == "s_waitcnt" || m == "s_nop" || m == "s_sendmsg" || m == "s_setprio" || m == "s_icache_inv" || m == "s_ttracedata") return true;
            // Workgroup execution/Workgroup memory scope, AcquireRelease (0x8 << 8) | WorkgroupMemory (0x4): 0x108 was
            // CrossDeviceMemory with an invalid semantic, which ordered nothing and did not cover the LDS.
            if (m == "s_barrier") { b.op0(Op::OpControlBarrier, {cu(2), cu(2), cu(0x804)}); return true; }
            const int off = int16_t(in.w[0] & 0xFFFF);
            if (starts_with(m, "s_cbranch_") && off >= 0 && (m == "s_cbranch_execz" || m == "s_cbranch_execnz" || m == "s_cbranch_vccz" || m == "s_cbranch_vccnz"))
                return true;  // forward skip of exec-predicated work: executing it with exec=0 is a no-op
            if (m == "s_branch" && off >= 0) {
                const size_t target = pc_ + in.len + size_t(off);
                // if/else: a then-region ending in a jump over the else-region (the else-region runs iff the then-region did not; skipping it
                // dropped a particle PS's default blend path - the clinic lamp glow came out green).
                if (!ret_code_ && !joins_.empty() && joins_.back().pc == pc_ + in.len && !joins_.back().els && target > joins_.back().pc &&
                    (joins_.size() < 2 || joins_[joins_.size() - 2].pc >= target)) {
                    Join& j = joins_.back();
                    const Id fix = b.label();
                    b.op0(Op::OpBranch, {fix});
                    b.place_label(j.alt);
                    j.alt = fix; j.pc = target; j.els = true;
                    std::swap(sym_, j.syms);  // the else-region starts from the branch state; the join keeps the then-state
                    auto kts = lane_kts();
                    for (auto& [s, l] : lm_) { auto it = j.kts.find(s); l.kt = it != j.kts.end() && it->second; }
                    j.kts = std::move(kts);
                    return true;
                }
                skip_until_ = target;
                return true;
            }
            if ((m == "s_cbranch_scc0" || m == "s_cbranch_scc1") && off >= 0 && !ret_code_) {
                // After a 64-bit logical op (the kill pattern) scc0 means "no lane left": the skipped code runs with exec=0 anyway.
                if (m == "s_cbranch_scc0" && (prev_mn_ == "s_andn2_b64" || prev_mn_ == "s_and_b64")) return true;
                if (!scc_) return fail("SCC read before any write", &in);
                Id taken = ld(scc_, B);
                if (m == "s_cbranch_scc1") taken = op(Op::OpLogicalNot, B, {taken});  // run the region iff the branch is not taken
                joins_.emplace_back();
                Join& j = joins_.back();
                j.pc = pc_ + in.len + size_t(off);
                j.merge = b.label(); j.alt = b.label();
                std::copy(std::begin(sym_), std::end(sym_), j.syms);
                j.kts = lane_kts();
                const Id then_l = b.label();
                b.op0(Op::OpSelectionMerge, {j.merge, 0});
                b.op0(Op::OpBranchConditional, {taken, then_l, j.alt});
                b.place_label(then_l);
                return true;
            }
            return fail("unsupported control flow", &in);
        }
        case GcnFmt::SOP1:
            if (m == "s_swappc_b64") {  // call into the fetch shader whose address sits in the source pair (user data)
                const unsigned src = in.w[0] & 0xFF;
                const auto host = [&](unsigned s) { return sym_[s].known && sym_[s].v.kind != ScalarVal::LoadWord; };  // (evaluable per draw from user data alone)
                if (ret_code_ || src >= 103 || !host(src) || !host(src + 1)) return fail("s_swappc_b64: call target is not a user-data pointer", &in);
                T.fetch_lo = sym_[src].v; T.fetch_hi = sym_[src + 1].v;
                const uint64_t addr = (uint64_t(eval_sval(T.fetch_hi)) << 32) | eval_sval(T.fetch_lo);
                fetch_.assign(512, 0);
                if (!env_.read_mem || !env_.read_mem(addr, fetch_.data(), 512)) return fail("fetch shader unreadable", &in);
                T.fetch_addr = addr;
                ret_code_ = code_; ret_n_ = n_; ret_pc_ = pc_ + in.len;
                code_ = fetch_.data(); n_ = fetch_.size(); pc_ = 0; jumped_ = true;
                sym_[src] = {}; sym_[src + 1] = {};  // return address: not a value the shader may use
                return true;
            }
            if (m == "s_setpc_b64" && ret_code_) {
                T.fetch_code.assign(fetch_.begin(), fetch_.begin() + pc_ + in.len);
                code_ = ret_code_; n_ = ret_n_; pc_ = ret_pc_; ret_code_ = nullptr; jumped_ = true;
                return true;
            }
            return salu(in, m);
        case GcnFmt::SOP2: case GcnFmt::SOPK: case GcnFmt::SOPC: return salu(in, m);
        case GcnFmt::SMRD: return smrd(in, m);
        case GcnFmt::VOP1: case GcnFmt::VOP2: case GcnFmt::VOPC: case GcnFmt::VOP3: return valu(in);
        case GcnFmt::VINTRP: return vintrp(in);
        case GcnFmt::MUBUF: case GcnFmt::MTBUF: return mubuf(in, m);
        case GcnFmt::MIMG: return mimg(in, m);
        case GcnFmt::EXP: return exp(in);
        case GcnFmt::DS: return ds(in);
        default: return fail("unsupported instruction format", &in);
    }
}

bool Xlat::setup() {
    b.capability(spv::Capability::Shader);
    if (env_.lds_bytes > 32768) return fail("GcnEnv::lds_bytes = " + std::to_string(env_.lds_bytes) + " exceeds the 32 KB LDS limit");
    U = b.t_u32(); I = b.t_i32(); F = b.t_f32(); B = b.t_bool();
    V4F = b.t_vec(F, 4); V2F = b.t_vec(F, 2); UV3 = b.t_vec(U, 3);
    PU = b.t_ptr(spv::StorageClass::PushConstant, U);
    Id arr = b.t_array(U, cu(16));
    b.decorate(arr, spv::Decoration::ArrayStride, {4});
    Id pcs = env_.res_scale > 1 ? b.t_struct({arr, U}) : b.t_struct({arr});
    b.decorate(pcs, spv::Decoration::Block);
    b.member_decorate(pcs, 0, spv::Decoration::Offset, {env_.push_offset});
    if (env_.res_scale > 1) b.member_decorate(pcs, 1, spv::Decoration::Offset, {env_.push_offset + 64});
    push_var_ = b.global_var(b.t_ptr(spv::StorageClass::PushConstant, pcs), spv::StorageClass::PushConstant);
    iface_.push_back(push_var_);
    fn_ = b.begin_function(b.t_void(), b.t_fn(b.t_void()));

    wr_lane(kExec, b.c_bool(true), true);
    if (env_.res_scale > 1) {
        sc_info_ = ld(op(Op::OpAccessChain, PU, {push_var_, cu(1)}), U);
        sc_t_ = env_.stage == ShStage::PS ? op(Op::OpINotEqual, B, {op(Op::OpBitwiseAnd, U, {sc_info_, cu(0x80000000u)}), cu(0)}) : b.c_bool(false);
    }
    if (env_.stage == ShStage::PS)
        if (const char* pe = std::getenv("BB_SPV_PROBE")) {
            unsigned crc = 0, pc = 0, reg = 0;
            float scale = 1.f;  // optional 4th field: multiply the probed values (to read magnitudes outside [0,1] through an 8-bit target)
            gcn::ShaderInfo si;
            if (std::sscanf(pe, "%x:%x:%u:%f", &crc, &pc, &reg, &scale) >= 3 && gcn::shader_info(code_, n_ + 64, si) && si.crc == crc) {
                probe_pc_ = pc / 4; probe_reg_ = reg; probe_scale_ = scale;
                probe_var_ = b.global_var(b.t_ptr(spv::StorageClass::Private, V4F), spv::StorageClass::Private);
                st(probe_var_, op(Op::OpCompositeConstruct, V4F, {cf(0.f), cf(0.f), cf(0.f), cf(0.f)}));
                std::fprintf(stderr, "gcn: probe armed in ps %08x at pc 0x%x v%u\n", crc, pc, reg);
            }
        }
    for (unsigned i = 0; i < env_.user_sgprs && i < 16; ++i) {
        Sym s;
        s.known = true; s.v.kind = ScalarVal::User; s.v.a = i;
        wr_s(i, push_user(i), s);
    }
    auto builtin = [&](spv::StorageClass sc, Id type, spv::BuiltIn bi) { return builtin_var(sc, type, bi); };
    switch (env_.stage) {
        case ShStage::VS: {
            if (!is_ls()) pos_out_ = builtin(spv::StorageClass::Output, V4F, spv::BuiltIn::Position);  // an LS is a compute pass: no vertex outputs (its exports, if any, fail below)
            // point-list draws need PointSize written (validation VUID-VkGraphicsPipelineCreateInfo-topology-08773); GCN points default to 1 px
            if (!is_ls()) st(psize_out_ = builtin(spv::StorageClass::Output, F, spv::BuiltIn::PointSize), cf(1.f));
            if (is_ls()) setup_ls();
            else if (is_ds()) setup_ds();
            else {  // v0 = vertex id, v3 = instance id (instanced VS address their per-instance records with it, e.g. instance * 13 + 5)
                // ponytail: v1/v2 (instance id / VGT_INSTANCE_STEP_RATE_0/1) are not set; no shader seen reads them
                st(vgpr_var(0), op(Op::OpBitcast, U, {ld(builtin(spv::StorageClass::Input, I, spv::BuiltIn::VertexIndex), I)}));
                st(vgpr_var(3), op(Op::OpBitcast, U, {ld(builtin(spv::StorageClass::Input, I, spv::BuiltIn::InstanceIndex), I)}));
            }
            break;
        }
        case ShStage::PS: {
            if (!frag_coord_) frag_coord_ = builtin(spv::StorageClass::Input, V4F, spv::BuiltIn::FragCoord);  // quad lane id for ds_swizzle
            static const unsigned kCount[16] = {2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
            unsigned v = 0;
            for (unsigned bit = 0; bit < 16; ++bit) {
                if (!((env_.ps_input_addr >> bit) & 1)) continue;
                if (bit >= 8 && bit <= 11) {
                    if (!frag_coord_) frag_coord_ = builtin(spv::StorageClass::Input, V4F, spv::BuiltIn::FragCoord);
                    Id f = op(Op::OpCompositeExtract, F, {ld(frag_coord_, V4F), bit - 8});
                    if (sc_t_ && bit <= 9) f = op(Op::OpSelect, F, {sc_t_, op(Op::OpFMul, F, {f, cf(1.0f / float(env_.res_scale))}), f});  // guest pixels
                    st(vgpr_var(v), bits(f));
                } else if (bit == 12) {
                    if (!front_face_) front_face_ = builtin(spv::StorageClass::Input, B, spv::BuiltIn::FrontFacing);
                    st(vgpr_var(v), op(Op::OpSelect, U, {ld(front_face_, B), cu(~0u), cu(0)}));
                }
                v += kCount[bit];
            }
            break;
        }
        case ShStage::CS: {
            Id lid = builtin(spv::StorageClass::Input, UV3, spv::BuiltIn::LocalInvocationId);
            Id wg = builtin(spv::StorageClass::Input, UV3, spv::BuiltIn::WorkgroupId);
            for (unsigned k = 0; k <= env_.cs_tidig_comps && k < 3; ++k) st(vgpr_var(k), op(Op::OpCompositeExtract, U, {ld(lid, UV3), k}));
            unsigned s = env_.user_sgprs;
            for (unsigned k = 0; k < 3; ++k)
                if ((env_.cs_tgid_en >> k) & 1) wr_s(s++, op(Op::OpCompositeExtract, U, {ld(wg, UV3), k}));
            break;
        }
    }
    if (sc_info_) {
        for (unsigned a = 0; a < 2; ++a) {
            if (env_.stage != ShStage::PS) { sc_sub_[a] = b.c_i32(0); continue; }
            Id p = op(Op::OpConvertFToS, I, {op(Op::OpCompositeExtract, F, {ld(frag_coord_, V4F), a})});
            sc_sub_[a] = op(Op::OpSelect, I, {sc_t_, op(Op::OpSMod, I, {p, b.c_i32(int32_t(env_.res_scale))}), b.c_i32(0)});
        }
    }
    return true;
}

// An LS is run as a compute pass over all control points of the draw: one invocation per point, the index buffer comes from the
// TessIndices resource (count at dword 0, ids from dword 1). v0 = the vertex id the fetch shader / idxen loads use, v1 = the
// relative point index (the shader scales it by 128 to address its record), v2 = the instance id.
void Xlat::setup_ls() {
    const Id pt = b.t_ptr(spv::StorageClass::StorageBuffer, U);
    Id gid = op(Op::OpCompositeExtract, U, {ld(builtin_var(spv::StorageClass::Input, UV3, spv::BuiltIn::GlobalInvocationId), UV3), 0});
    Id count = ld(op(Op::OpAccessChain, pt, {tess_var(0), cu(0), cu(0)}), U);
    Id out_of_range = op(Op::OpUGreaterThanEqual, B, {gid, count});  // computed before the merge instruction: it must directly precede its branch
    Id then_l = b.label(), rest = b.label();
    b.op0(Op::OpSelectionMerge, {rest, 0});
    b.op0(Op::OpBranchConditional, {out_of_range, then_l, rest});
    b.place_label(then_l);
    b.op0(Op::OpReturn, {});
    b.place_label(rest);
    st(vgpr_var(0), ld(op(Op::OpAccessChain, pt, {tess_var(0), cu(0), op(Op::OpIAdd, U, {gid, cu(1)})}), U));
    st(vgpr_var(1), gid);
    st(vgpr_var(2), cu(0));
}

// A DS runs as a vertex pass over a generated (N+1)x(N+1) grid per patch: the vertex index selects the cell, the instance index is
// the patch. v0/v1 = the domain location, v2 = v3 = the patch id (the shader computes the record address from it).
void Xlat::setup_ds() {
    const uint32_t side = env_.tess_ds_level + 1;
    Id vid = op(Op::OpBitcast, U, {ld(builtin_var(spv::StorageClass::Input, I, spv::BuiltIn::VertexIndex), I)});
    Id patch = op(Op::OpBitcast, U, {ld(builtin_var(spv::StorageClass::Input, I, spv::BuiltIn::InstanceIndex), I)});
    auto frac = [&](Id k) { return bits(op(Op::OpFDiv, F, {op(Op::OpConvertUToF, F, {k}), cf(float(env_.tess_ds_level))})); };
    st(vgpr_var(0), frac(op(Op::OpUMod, U, {vid, cu(side)})));
    st(vgpr_var(1), frac(op(Op::OpUDiv, U, {vid, cu(side)})));
    st(vgpr_var(2), patch);
    st(vgpr_var(3), patch);
}

bool Xlat::run() {
    if (!setup()) return false;
    T.tess_ls = env_.tess_ls;
    T.tess_ds_level = env_.tess_ds_level;
    scan_loops();
    while (pc_ < n_ && !ended_ && err_.empty()) {
        const Insn in = gcn::decode(code_ + pc_, n_ - pc_);
        if (in.fmt == GcnFmt::Invalid) { fail("undecodable instruction word " + std::to_string(code_[pc_])); break; }
        jumped_ = false;
        while (!joins_.empty() && pc_ >= joins_.back().pc) close_join();
        if (!ret_code_ && pc_ >= skip_until_ && loop_heads_.count(pc_) && !begin_loop()) break;
        if (pc_ == probe_pc_ && probe_var_) {
            Id c[4];
            for (unsigned k = 0; k < 4; ++k) c[k] = op(Op::OpFMul, F, {flt(ld(vgpr_var(probe_reg_ + k), U)), cf(probe_scale_)});
            st(probe_var_, op(Op::OpCompositeConstruct, V4F, {c[0], c[1], c[2], c[3]}));
        }
        if (pc_ >= skip_until_ || ret_code_) {
            step(in);
            if (in.fmt != GcnFmt::SOPP) prev_mn_ = gcn::mnemonic(in.fmt, in.op) ? gcn::mnemonic(in.fmt, in.op) : "";
        }
        if (!jumped_) pc_ += in.len;
    }
    if (err_.empty() && !ended_) fail("no s_endpgm");
    while (err_.empty() && !joins_.empty()) close_join();  // s_endpgm inside a region: close the open selections
    if (!err_.empty()) return false;
    finish_tess();
    b.op0(Op::OpReturn, {});
    b.end_function();
    // an LS is a VS-slot program that runs as compute, a DS is a VS-slot program that runs as vertex
    const spv::ExecutionModel model = env_.stage == ShStage::PS ? spv::ExecutionModel::Fragment
                                 : (env_.stage == ShStage::VS && !is_ls()) ? spv::ExecutionModel::Vertex : spv::ExecutionModel::GLCompute;
    b.entry_point(model, fn_, "main", iface_);
    if (env_.stage == ShStage::PS) {
        b.exec_mode(fn_, spv::ExecutionMode::OriginUpperLeft);
        if (depth_out_) b.exec_mode(fn_, spv::ExecutionMode::DepthReplacing);
    }
    if (is_ls()) b.exec_mode(fn_, spv::ExecutionMode::LocalSize, {64, 1, 1});  // one invocation per control point
    else if (env_.stage == ShStage::CS) b.exec_mode(fn_, spv::ExecutionMode::LocalSize, {env_.cs_local[0], env_.cs_local[1], env_.cs_local[2]});
    return true;
}

}  // namespace

Translation translate(const uint32_t* code, size_t dwords, const GcnEnv& env) {
    Translation t;
    Xlat x(code, dwords, env, t);
    if (x.run()) t.spirv = x.finish();
    else t.error = x.error();
    return t;
}


bool eval_resources(const Translation& t, const uint32_t user[16],
                    const std::function<bool(uint64_t, uint32_t*, uint32_t)>& read_mem, std::vector<std::array<uint32_t, 8>>& words) {
    // per-thread scratch: called twice per draw; fresh vectors were several heap allocations each time
    thread_local std::vector<std::vector<uint32_t>> loads;
    const size_t nl = t.loads.size();
    if (loads.size() < nl) loads.resize(nl);
    for (size_t i = 0; i < nl; ++i) loads[i].clear();  // a load not read yet must look empty (as with fresh vectors)
    auto ev = [&](const ScalarVal& v) -> uint32_t {
        uint32_t base = 0;
        switch (v.kind) {
            case ScalarVal::User: base = user[v.a & 15]; break;
            case ScalarVal::Const: base = v.a; break;
            case ScalarVal::LoadWord: base = v.a < nl && v.b < loads[v.a].size() ? loads[v.a][v.b] : 0; break;
        }
        return (base + v.add) & v.mask;
    };
    for (size_t i = 0; i < nl; ++i) {
        const SLoad& l = t.loads[i];
        const uint64_t addr = ((uint64_t(ev(l.hi)) << 32) | ev(l.lo)) + l.offset;
        loads[i].resize(l.dwords);
        if (!read_mem(addr, loads[i].data(), l.dwords)) return false;
    }
    words.assign(t.resources.size(), {});
    for (size_t i = 0; i < t.resources.size(); ++i) {
        const Resource& r = t.resources[i];
        if (r.load_data) {  // synthesize a V# over the loaded dwords: base address, size in bytes
            if (r.ref.load < 0 || size_t(r.ref.load) >= t.loads.size()) return false;
            const SLoad& l = t.loads[size_t(r.ref.load)];
            const uint64_t addr = ((uint64_t(ev(l.hi)) << 32) | ev(l.lo)) + l.offset;
            words[i][0] = uint32_t(addr);
            words[i][1] = uint32_t(addr >> 32) & 0xFFF;
            words[i][2] = std::max<uint32_t>(l.dwords * 4, 16);
            continue;
        }
        for (uint32_t k = 0; k < r.dwords; ++k) {
            if (r.ref.load < 0) words[i][k] = user[(r.ref.word + k) & 15];
            else if (size_t(r.ref.load) < nl && r.ref.word + k < loads[r.ref.load].size()) words[i][k] = loads[r.ref.load][r.ref.word + k];
            else return false;
        }
    }
    return true;
}

uint32_t shape_mask(Resource::Type type, unsigned i) {
    switch (type) {
        case Resource::Buffer: return i == 1 ? 0x3FFF0000u : i == 3 ? 0x0007FFFFu : 0;  // stride | dst_sel, nfmt, dfmt
        case Resource::Image:
        case Resource::StorageImage: return i == 1 ? 0x3C000000u : i == 3 ? 0xF0000000u : 0;  // num_format class | image type
        case Resource::Sampler: return 0;
    }
    return 0;
}

uint64_t fetch_address(const Translation& t, const uint32_t user[16]) {
    auto ev = [&](const ScalarVal& v) { return ((v.kind == ScalarVal::User ? user[v.a & 15] : v.a) + v.add) & v.mask; };  // (never LoadWord, see s_swappc_b64)
    return uint64_t(ev(t.fetch_hi)) << 32 | ev(t.fetch_lo);
}

}  // namespace bb::gpu
