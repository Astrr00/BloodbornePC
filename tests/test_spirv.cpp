// GCN -> SPIR-V: hand-assembled shaders must translate and the SPIR-V must validate for Vulkan 1.2.
#undef NDEBUG
#include <spirv/unified1/spirv.hpp11>
#include <spirv-tools/libspirv.hpp>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
#include <tuple>
#include <vector>

#include "gpu/gcn_spirv.h"
#include "gpu/fsr1_spv.inc"  // the presenter's FSR 1 passes (tools/gen_fsr1_spv.py): must stay valid Vulkan 1.3 SPIR-V

using namespace bb::gpu;

static void check(const char* what, const Translation& t) {
    if (!t.error.empty()) { std::fprintf(stderr, "%s: translate failed: %s\n", what, t.error.c_str()); std::abort(); }
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* m) { std::fprintf(stderr, "  %s\n", m); });
    if (!tools.Validate(t.spirv)) { std::fprintf(stderr, "%s: invalid SPIR-V\n", what); std::abort(); }
}

// First instruction with opcode `op` (walks the instruction stream after the 5-word header); nullptr if absent.
static const uint32_t* find_op(const std::vector<uint32_t>& spv, spv::Op op, const std::function<bool(const uint32_t*)>& pred = {}) {
    for (size_t i = 5; i < spv.size();) {
        const uint32_t n = spv[i] >> 16;
        if (!n) break;
        if ((spv[i] & 0xFFFF) == uint32_t(op) && (!pred || pred(&spv[i]))) return &spv[i];
        i += n;
    }
    return nullptr;
}
// the module declares OpDecorate <id> BuiltIn <n>: the GCN lane index comes from this invocation's built-in id
static bool has_builtin(const std::vector<uint32_t>& spv, uint32_t which) {
    return find_op(spv, spv::Op::OpDecorate, [&](const uint32_t* w) { return (w[0] >> 16) == 4 && w[2] == uint32_t(spv::Decoration::BuiltIn) && w[3] == which; }) != nullptr;
}
// the module's OpEntryPoint execution model (Vertex = 0, GLCompute = 5)
static uint32_t entry_model(const std::vector<uint32_t>& spv) {
    const uint32_t* w = find_op(spv, spv::Op::OpEntryPoint);
    return w ? w[1] : ~0u;
}

int main() {
    {  // VS: v1 = 1.0; exp pos0 v1,v1,v1,v1; exp param0; s_endpgm
        const uint32_t c[] = {0x7E0202F2, 0xF80008CF, 0x01010101, 0xF8000200, 0x01010101, 0xBF810000};
        GcnEnv env; env.stage = ShStage::VS;
        Translation t = translate(c, 6, env);
        check("vs", t);
        assert(t.vs_params == 1);
    }
    {  // PS: v0 = 1.0; exp mrt0 compr (v0,v0); s_endpgm
        const uint32_t c[] = {0x7E0002F2, 0xF8001C0F, 0x00000000, 0xBF810000};
        GcnEnv env; env.stage = ShStage::PS;
        Translation t = translate(c, 4, env);
        check("ps", t);
        assert(t.ps_mrts == 1);
    }
    {  // CS: buffer_store_dword v1, v0, s[0:3] idxen with the V# in user data (stride 4)
        const uint32_t c[] = {0xE0702000, 0x80000100, 0xBF810000};
        GcnEnv env; env.stage = ShStage::CS; env.user_sgprs = 4;
        env.user[0] = 0x1000; env.user[1] = 4u << 16; env.user[2] = 64; env.user[3] = 0x00027000;
        Translation t = translate(c, 3, env);
        check("cs", t);
        assert(t.resources.size() == 1 && t.resources[0].written);
    }
    {  // s_buffer_load_dword s4, s[0:3], 0x0: a constant buffer, flagged scalar (the backend must not clamp it to the vertex count); a vertex fetch is not
        const uint32_t cb[] = {0xC2020100, 0xBF810000};
        GcnEnv env; env.stage = ShStage::CS; env.user_sgprs = 4;
        env.user[0] = 0x1000; env.user[1] = 16u << 16; env.user[2] = 64; env.user[3] = 0x00027000;
        Translation t = translate(cb, 2, env);
        check("cbuffer", t);
        assert(t.resources.size() == 1 && t.resources[0].scalar);
        const uint32_t st[] = {0xE0702000, 0x80000100, 0xBF810000};
        Translation u = translate(st, 3, env);
        check("store", u);
        assert(u.resources.size() == 1 && !u.resources[0].scalar);
    }
    {  // unsupported input is reported, not silently mistranslated
        const uint32_t bad[] = {0xBF850001, 0xBF810000, 0xBF810000};  // s_cbranch_scc1 +1 (needs real control flow)
        GcnEnv env; env.stage = ShStage::CS;
        assert(!translate(bad, 3, env).error.empty());
    }
    {  // if/else on a scalar condition: s_cmp_eq_i32 s0, s1; s_cbranch_scc0 ELSE; v0 = 1.0; s_branch END; ELSE: v0 = 2.0; END: exp mrt0
        // The else-region must be translated (it used to be skipped after the then-region's s_branch).
        const uint32_t c[] = {0xBF000100, 0xBF840002, 0x7E0002F2, 0xBF820001, 0x7E0002F4, 0xF8001C0F, 0x00000000, 0xBF810000};
        GcnEnv env; env.stage = ShStage::PS; env.user_sgprs = 2;
        Translation t = translate(c, 8, env);
        check("if/else", t);
        assert(find_op(t.spirv, spv::Op::OpConstant, [](const uint32_t* w) { return (w[0] >> 16) == 4 && w[3] == 0x40000000u; }));
    }
    {  // if/else, the then-region clobbers the else-region's s_load base: s_cmp_eq_i32 s0, s1; s_cbranch_scc0 ELSE; s_not_b32 s2, s0; s_branch END;
       // ELSE: s_load_dwordx4 s[4:7], s[2:3], 0; END: v_mov_b32 v0, s4; exp mrt0. Both regions start from the branch state (the else-region
       // failed with "s_load base pointer is not user-data derived"); s4 is read after the merge, so the else path stores its load result.
        static uint32_t data[4] = {1, 2, 3, 4};
        const uint32_t c[] = {0xBF000100, 0xBF840002, 0xBE820700, 0xBF820001, 0xC0820300, 0x7E000204, 0xF8001C0F, 0x00000000, 0xBF810000};
        GcnEnv env; env.stage = ShStage::PS; env.user_sgprs = 4;
        env.user[2] = uint32_t(uintptr_t(data)); env.user[3] = uint32_t(uint64_t(uintptr_t(data)) >> 32);
        env.read_mem = [](uint64_t a, uint32_t* d, uint32_t n) { std::memcpy(d, reinterpret_cast<const void*>(a), n * 4); return true; };
        Translation t = translate(c, 9, env);
        check("if/else scalar state", t);
        assert(t.loads.size() == 1 && t.resources.size() == 1 && t.resources[0].load_data);
    }
    {  // an unreadable s_load table fails with read_failed and the failed read as the last load: the host retries once a draw reads it
        static uint32_t data[4] = {1, 2, 3, 4};
        const uint32_t c[] = {0xC0820300, 0x7E000204, 0xF8001C0F, 0x00000000, 0xBF810000};  // s_load_dwordx4 s[4:7], s[2:3], 0; v_mov_b32 v0, s4; exp mrt0
        GcnEnv env; env.stage = ShStage::PS; env.user_sgprs = 4;
        env.user[2] = uint32_t(uintptr_t(data)); env.user[3] = uint32_t(uint64_t(uintptr_t(data)) >> 32);
        env.read_mem = [](uint64_t, uint32_t*, uint32_t) { return false; };
        const Translation t = translate(c, 5, env);
        assert(!t.error.empty() && t.read_failed && t.loads.size() == 1 && t.loads[0].dwords == 4);
        std::vector<std::array<uint32_t, 8>> w;
        assert(!eval_resources(t, env.user, env.read_mem, w));
        const std::function<bool(uint64_t, uint32_t*, uint32_t)> ok = [](uint64_t a, uint32_t* d, uint32_t n) { std::memcpy(d, reinterpret_cast<const void*>(a), n * 4); return true; };
        assert(eval_resources(t, env.user, ok, w));
    }
    {  // scalar-branch shapes a structured selection cannot express are rejected with a reason, never translated into wrong/invalid SPIR-V
        GcnEnv env; env.stage = ShStage::PS; env.user_sgprs = 2;
        auto fails = [&](const char* what, const uint32_t* c, size_t n, const char* reason) {
            const Translation t = translate(c, n, env);
            if (t.error.find(reason) == std::string::npos) { std::fprintf(stderr, "%s: expected \"%s\", got \"%s\"\n", what, reason, t.error.c_str()); std::abort(); }
        };
        // overlapping ranges: s_cmp; s_cbranch_scc0 A(5); s_cmp; s_cbranch_scc1 B(6); v0 = 1.0; A: v0 = 2.0; B: exp mrt0
        const uint32_t overlap[] = {0xBF000100, 0xBF840003, 0xBF000100, 0xBF850002, 0x7E0002F2, 0x7E0002F4, 0xF8001C0F, 0x00000000, 0xBF810000};
        fails("overlapping regions", overlap, 9, "crosses the end of the enclosing branch region");
        // jump out of a region (not the if/else shape): s_cmp; s_cbranch_scc0 A(4); s_branch B(5); v0 = 1.0; A: v0 = 2.0; B: exp mrt0
        const uint32_t out[] = {0xBF000100, 0xBF840002, 0xBF820002, 0x7E0002F2, 0x7E0002F4, 0xF8001C0F, 0x00000000, 0xBF810000};
        fails("s_branch out of a region", out, 8, "s_branch leaves a scalar-branch region");
        // backward jump into a region: s_cmp; s_cbranch_scc0 A(5); L: v0 = 1.0; s_cmp; s_cbranch_scc1 L; A: exp mrt0
        const uint32_t loop[] = {0xBF000100, 0xBF840003, 0x7E0002F2, 0xBF000100, 0xBF85FFFD, 0xF8001C0F, 0x00000000, 0xBF810000};
        fails("loop inside a region", loop, 8, "loop inside a scalar-branch region");
    }
    {  // s_endpgm inside a region with code behind it ends only that path: s_cmp; s_cbranch_scc0 A; v0 = 1.0; exp; s_endpgm; A: v0 = 2.0; exp
        const uint32_t c[] = {0xBF000100, 0xBF840004, 0x7E0002F2, 0xF8001C0F, 0x00000000, 0xBF810000, 0x7E0002F4, 0xF8001C0F, 0x00000000, 0xBF810000};
        GcnEnv env; env.stage = ShStage::PS; env.user_sgprs = 2;
        Translation t = translate(c, 10, env);
        check("s_endpgm in a region", t);
        assert(find_op(t.spirv, spv::Op::OpConstant, [](const uint32_t* w) { return (w[0] >> 16) == 4 && w[3] == 0x40000000u; }));  // A was translated
    }
    {  // jump threading (a clinic CS): the inner then-region jumps past the outer region to where the outer region's closing s_branch goes.
       // s_cmp; s_cbranch_scc1 X; s_cmp; s_cbranch_scc0 E; v0 = 1.0; s_branch T; E: v0 = 2.0; s_branch T; X: v0 = 4.0; T: exp mrt0
        const uint32_t c[] = {0xBF000100, 0xBF850006, 0xBF000100, 0xBF840002, 0x7E0002F2, 0xBF820003, 0x7E0002F4, 0xBF820001, 0x7E0002F6,
                              0xF8001C0F, 0x00000000, 0xBF810000};
        GcnEnv env; env.stage = ShStage::PS; env.user_sgprs = 2;
        Translation t = translate(c, 12, env);
        check("jump threading", t);
        assert(find_op(t.spirv, spv::Op::OpConstant, [](const uint32_t* w) { return (w[0] >> 16) == 4 && w[3] == 0x40800000u; }));  // X was translated
    }
    {  // lane id: v_mbcnt_lo_u32_b32 v25, exec_lo, v25 + v_mbcnt_hi_u32_b32_e64 v25, exec_hi, 0, s0
        const uint32_t c[] = {0x4632327E, 0xD2480019, 0x0001007F, 0xBF810000};
        GcnEnv env; env.stage = ShStage::CS;
        Translation t = translate(c, 4, env);
        check("mbcnt", t);
        assert(has_builtin(t.spirv, uint32_t(spv::BuiltIn::LocalInvocationIndex)));
    }
    {  // LDS: ds_write2_b32, s_barrier, ds_read_b32/ds_read2_b32, ds_add_u32/ds_add_rtn_u32 (256 bytes of LDS)
        const uint32_t c[] = {0x7E000280, 0x7E0202F2, 0x7E0402F2, 0xD8380400, 0x00010000, 0xBF8A0000, 0xD8D80000, 0x03000000,
                              0xD8DC0400, 0x04000000, 0xD8000000, 0x00000000, 0xD8800000, 0x06000100, 0xBF810000};
        GcnEnv env; env.stage = ShStage::CS; env.lds_bytes = 256;
        Translation t = translate(c, 15, env);
        check("lds", t);
        GcnEnv no_lds; no_lds.stage = ShStage::CS;
        assert(!translate(c, 15, no_lds).error.empty());  // an LDS access without an LDS size is reported, not ignored
        GcnEnv huge = env; huge.lds_bytes = 65536;
        assert(!translate(c, 15, huge).error.empty());
    }
    {  // dfmt 6 (10_11_11): buffer_store_format_xyz + buffer_load_format_xyz of packed small floats
        const uint32_t c[] = {0x7E0002F2, 0x7E0202F2, 0x7E0402F0, 0x7E060280, 0xE0180000, 0x80000003, 0xE0080000, 0x80000403, 0xBF810000};
        GcnEnv env; env.stage = ShStage::CS; env.user_sgprs = 4;
        env.user[0] = 0x1000; env.user[1] = 4u << 16; env.user[2] = 64; env.user[3] = (6u << 15) | (7u << 12) | 0x654;
        Translation t = translate(c, 9, env);
        check("small float", t);
        assert(t.resources.size() == 1 && t.resources[0].written);
        env.user[3] = (7u << 15) | (7u << 12) | 0x654;  // dfmt 7 (11_11_10)
        check("small float 7", translate(c, 9, env));
        env.user[3] = (0u << 15) | (7u << 12) | 0x654;  // dfmt 0 = invalid: loads return 0, stores vanish
        check("dfmt 0", translate(c, 9, env));
    }
    {  // LS: v2,v3 = 1.0; ds_write2_b32 at (v0, v0+4) (the shader scales v1 by 128 for the record address); s_endpgm
        const uint32_t c[] = {0x7E0402F2, 0x7E0602F2, 0xD8380400, 0x00030204, 0xBF810000};
        GcnEnv env; env.stage = ShStage::VS; env.tess_ls = 1; env.lds_bytes = 512;
        Translation t = translate(c, 5, env);
        check("ls", t);
        assert(entry_model(t.spirv) == uint32_t(spv::ExecutionModel::GLCompute) && t.tess_ls == 1);
        assert(t.resources.size() == 2 && t.resources[0].type == Resource::TessIndices && t.resources[1].type == Resource::LdsBuffer);
        assert(t.resources[1].written && !t.resources[0].written);
    }
    {  // DS: v2,v3 = 1.0; v2 = 0; ds_read2_b32 from (v2, v2+4) into v6,v7; exp pos0 v6; s_endpgm
        const uint32_t c[] = {0x7E0402F2, 0x7E0602F2, 0x7E040280, 0xD8DC0400, 0x06060002, 0xF80008CF, 0x07060606, 0xBF810000};
        GcnEnv env; env.stage = ShStage::VS; env.tess_ds_level = 3; env.lds_bytes = 512;
        Translation t = translate(c, 8, env);
        check("ds", t);
        assert(entry_model(t.spirv) == uint32_t(spv::ExecutionModel::Vertex) && t.tess_ds_level == 3);
        assert(t.resources.size() == 1 && t.resources[0].type == Resource::LdsBuffer && !t.resources[0].written);
        const uint32_t wr[] = {0x7E0402F2, 0xD8380400, 0x00030204, 0xBF810000};  // an LDS write in a DS is reported, not silently dropped
        assert(!translate(wr, 4, env).error.empty());
    }
    {  // GDS: s_mov_b32 m0, 0; ds_append v1 offset:4 gds; s_endpgm -> one host GDS buffer, written
        const uint32_t c[] = {0xBEFC0380, 0xD8FA0004, 0x01000000, 0xBF810000};
        GcnEnv env; env.stage = ShStage::CS;
        Translation t = translate(c, 4, env);
        check("gds append", t);
        assert(t.resources.size() == 1 && t.resources[0].type == Resource::GdsBuffer && t.resources[0].written);
        const uint32_t add[] = {0xBEFC0380, 0xD8020000, 0x00000000, 0xBF810000};  // other GDS ops are reported, not run as LDS
        assert(!translate(add, 4, env).error.empty());
    }
    {  // v_interp_mov_f32 v0, p0, attr0.x (flat); exp mrt0 v0 done vm -> valid SPIR-V; P10 and mixing with p1 are reported
        const uint32_t c[] = {0xC8020002, 0xF8001801, 0x00000000, 0xBF810000};
        GcnEnv env; env.stage = ShStage::PS;
        Translation t = translate(c, 4, env);
        check("flat", t);
        assert(t.ps_attrs == 1);
        const uint32_t p10[] = {0xC8020000, 0xF8001801, 0x00000000, 0xBF810000};
        assert(!translate(p10, 4, env).error.empty());
        const uint32_t mix[] = {0xC8000100, 0xC8020002, 0xF8001801, 0x00000000, 0xBF810000};  // v_interp_p1_f32 v0, v0, attr0.x, then flat
        assert(!translate(mix, 5, env).error.empty());
        GcnEnv bary = env; bary.ps_bary = true;  // per-vertex inputs + BaryCoordKHR: still valid SPIR-V
        const uint32_t p1[] = {0xC8000100, 0xF8001801, 0x00000000, 0xBF810000};  // v_interp_p1_f32 v0, v0, attr0.y; exp mrt0 v0
        check("bary", translate(p1, 4, bary));
    }
    {  // VS: v1 = 1.0; exp pos0 v1 x4; exp pos1 v1,v1 en:0x3 done; PA_CL_VS_OUT_CNTL decides what pos1 is
        const uint32_t c[] = {0x7E0202F2, 0xF80000CF, 0x01010101, 0xF80008D3, 0x01010101, 0xBF810000};
        GcnEnv env; env.stage = ShStage::VS;
        env.vs_out_cntl = (1u << 22) | 0x1 | (0x2 << 8);  // CCDIST0 vector: distance 0 clips, distance 1 culls
        Translation t = translate(c, 6, env);
        check("pos1 clip/cull", t);
        assert(has_builtin(t.spirv, uint32_t(spv::BuiltIn::ClipDistance)) && has_builtin(t.spirv, uint32_t(spv::BuiltIn::CullDistance)));
        env.vs_out_cntl = (1u << 21) | (1u << 16);  // misc vector with the point size
        t = translate(c, 6, env);
        check("pos1 misc", t);
        assert(!has_builtin(t.spirv, uint32_t(spv::BuiltIn::ClipDistance)));
        env.vs_out_cntl = 0;  // nothing enabled: the export is ignored like on hardware
        check("pos1 off", translate(c, 6, env));
    }
    {  // PS: image_sample_lz_o / image_sample_o v[4:7], v[0:2] (offset, s, t), T# s[0:7], S# s[8:11]; exp mrt0 v[4:7] done vm
        GcnEnv env; env.stage = ShStage::PS; env.user_sgprs = 12;
        env.user[0] = 0x1000; env.user[1] = 10u << 20; env.user[2] = 63 | (63u << 14); env.user[3] = (9u << 28) | 0xFAC;
        for (uint32_t op : {55u, 48u, 52u}) {  // lz_o, o (implicit LOD), l_o
            const uint32_t c[] = {0xF0000F00 | (op << 18), 0x00400400, 0xF800180F, 0x07060504, 0xBF810000};
            Translation t = translate(c, 5, env);
            check("sample offset", t);
            assert(t.resources.size() == 2 && find_op(t.spirv, spv::Op::OpBitFieldSExtract));  // the offset moved the coordinates
        }
    }
    {  // PS: v0 = 1e5; v_cvt_pkrtz_f16_f32 v0, v0, v0 (round toward zero, stays finite); exp mrt0 compr done vm
        const uint32_t c[] = {0x7E0002FF, 0x47C35000, 0x5E000100, 0xF8001C0F, 0x00000000, 0xBF810000};
        GcnEnv env; env.stage = ShStage::PS;
        check("pkrtz", translate(c, 6, env));
    }
    {  // BB_RES_SCALE 2: image_load / image_store / image_sample_lz_o of a 2D T# with FragCoord (POS_X/Y) in a PS, image_store in a CS:
       // valid SPIR-V with the scale bits after the user data; the PS store writes s * s times
        GcnEnv env; env.stage = ShStage::PS; env.user_sgprs = 12; env.res_scale = 2; env.push_offset = 80; env.ps_input_addr = 0x300;
        env.user[0] = 0x1000; env.user[1] = 10u << 20; env.user[2] = 63 | (63u << 14); env.user[3] = (9u << 28) | 0xFAC;
        for (uint32_t op : {0u, 8u, 55u}) {  // image_load, image_store, image_sample_lz_o
            const uint32_t c[] = {0xF0000F00 | (op << 18), 0x00400400, 0xF800180F, 0x07060504, 0xBF810000};
            Translation t = translate(c, 5, env);
            check("res_scale ps", t);
            if (op == 8) {
                size_t writes = 0;
                for (size_t i = 5; i < t.spirv.size() && (t.spirv[i] >> 16); i += t.spirv[i] >> 16) writes += (t.spirv[i] & 0xFFFF) == uint32_t(spv::Op::OpImageWrite);
                assert(writes == 4);
            }
        }
        env.stage = ShStage::CS; env.push_offset = 0; env.ps_input_addr = 0;
        const uint32_t cs[] = {0xF0000F00 | (8u << 18), 0x00400400, 0xBF810000};
        check("res_scale cs store", translate(cs, 3, env));
    }
    for (const auto& [what, b, e] : {std::tuple{"fsr easu", std::begin(kFsrEasuFs), std::end(kFsrEasuFs)}, std::tuple{"fsr rcas", std::begin(kFsrRcasFs), std::end(kFsrRcasFs)}}) {
        spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
        tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* m) { std::fprintf(stderr, "  %s\n", m); });
        if (!tools.Validate(std::vector<uint32_t>(b, e))) { std::fprintf(stderr, "%s: invalid SPIR-V\n", what); std::abort(); }
    }
    std::puts("spirv ok");
}
