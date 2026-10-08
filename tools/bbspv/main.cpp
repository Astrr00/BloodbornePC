// bbspv: validate (and optionally disassemble) SPIR-V files produced by the GCN translator, or translate a raw
// shader dump (build/sh2/*.bin) the way the backend does, so a shader can be checked without running the game.
// Usage: bbspv [-d] file.spv...                                  exit 0 iff all files validate against Vulkan 1.2.
//        bbspv --tess-ls [--user-sgprs n] [--fetch-stub] [-d] f   translate f as an LS (one invocation per control point)
//        bbspv --tess-ds N [--user-sgprs n] [-d] f                translate f as a DS (vertex pass, N x N cells per patch)
//        bbspv --replay [-d] [--write-spv] f.bin...               translate f.bin with the environment and guest memory recorded in f.env
// The shader code length comes from the OrbShdr trailer; user data is zero, guest memory is unreadable unless
// --fetch-stub serves a one-instruction fetch shader (a real fetch shader lives in guest memory, not in the dump). --replay takes both
// from the .env file BB_SHADER_DIR writes next to each dump (see note_shader).
#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gpu/gcn.h"
#include "gpu/gcn_spirv.h"

using namespace bb;

static spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);

static bool report(const char* what, const std::vector<uint32_t>& spv, const std::string& err, bool dis) {
    std::printf("%s: ", what);
    if (!err.empty()) { std::printf("TRANSLATE FAILED: %s\n", err.c_str()); return false; }
    const bool ok = tools.Validate(spv);
    std::puts(ok ? "valid" : "INVALID");
    if (dis) {
        std::string text;
        tools.Disassemble(spv, &text, SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES | SPV_BINARY_TO_TEXT_OPTION_INDENT);
        std::puts(text.c_str());
    }
    return ok;
}

static bool read_file(const char* path, std::vector<uint32_t>& w) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return false; }
    uint32_t x;
    while (std::fread(&x, 4, 1, f) == 1) w.push_back(x);
    std::fclose(f);
    return true;
}

// f.env (note_shader): header {magic "BBEV", stage, user_sgprs, ps_input_addr, vs_out_cntl, cs_tgid_en, cs_tidig_comps, cs_local[3], reads},
// user[16], then per read {u64 address, u32 dwords, dwords}. A replayed read must lie inside one recorded read.
struct Replay { std::vector<std::pair<uint64_t, std::vector<uint32_t>>> mem; };
static bool load_env(const std::string& path, gpu::GcnEnv& env, Replay& rp) {
    std::vector<uint32_t> w;
    if (!read_file(path.c_str(), w) || w.size() < 27 || w[0] != 0x56454242u) return false;
    env.stage = gpu::ShStage(w[1]);
    env.user_sgprs = w[2]; env.ps_input_addr = w[3]; env.vs_out_cntl = w[4]; env.cs_tgid_en = w[5]; env.cs_tidig_comps = w[6];
    for (int i = 0; i < 3; ++i) env.cs_local[i] = w[7 + i];
    for (int i = 0; i < 16; ++i) env.user[i] = w[11 + i];
    size_t p = 27;
    for (uint32_t r = 0; r < w[10]; ++r) {
        if (p + 3 > w.size() || p + 3 + w[p + 2] > w.size()) return false;
        const uint64_t a = w[p] | uint64_t(w[p + 1]) << 32;
        rp.mem.emplace_back(a, std::vector<uint32_t>(w.begin() + p + 3, w.begin() + p + 3 + w[p + 2]));
        p += 3 + w[p + 2];
    }
    env.read_mem = [&rp](uint64_t a, uint32_t* dst, uint32_t n) {
        for (const auto& [b, d] : rp.mem)
            if (a >= b && (a - b) % 4 == 0 && (a - b) / 4 + n <= d.size()) { std::memcpy(dst, d.data() + (a - b) / 4, size_t(n) * 4); return true; }
        return false;
    };
    return true;
}

int main(int argc, char** argv) {
    tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t& p, const char* m) { std::fprintf(stderr, "  %zu: %s\n", p.index, m); });
    bool dis = false, fetch_stub = false, write_spv = false;
    int bad = 0, ds_level = 0, translate = 0;
    unsigned user_sgprs = 16;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!std::strcmp(a, "-d")) { dis = true; continue; }
        if (!std::strcmp(a, "--fetch-stub")) { fetch_stub = true; continue; }
        if (!std::strcmp(a, "--write-spv")) { write_spv = true; continue; }
        if (!std::strcmp(a, "--tess-ls")) { translate = 1; continue; }
        if (!std::strcmp(a, "--replay")) { translate = 3; continue; }
        if (!std::strcmp(a, "--tess-ds") && i + 1 < argc) { translate = 2; ds_level = std::atoi(argv[++i]); continue; }
        if (!std::strcmp(a, "--user-sgprs") && i + 1 < argc) { user_sgprs = unsigned(std::atoi(argv[++i])); continue; }
        std::vector<uint32_t> w;
        if (!read_file(a, w)) { ++bad; continue; }
        if (!translate) { if (!report(a, w, {}, dis)) ++bad; continue; }
        gcn::ShaderInfo si;
        if (!gcn::shader_info(w.data(), w.size(), si) || !si.code_bytes || si.code_bytes > w.size() * 4) {
            std::fprintf(stderr, "%s: no OrbShdr trailer\n", a);
            ++bad;
            continue;
        }
        gpu::GcnEnv env;
        Replay rp;
        if (translate == 3) {
            std::string ep = a;
            ep = ep.substr(0, ep.rfind('.')) + ".env";
            if (!load_env(ep, env, rp)) { std::fprintf(stderr, "%s: no valid %s\n", a, ep.c_str()); ++bad; continue; }
        } else {
            env.stage = gpu::ShStage::VS;
            env.tess_ls = translate == 1;
            env.tess_ds_level = translate == 2 ? uint32_t(ds_level) : 0;
            env.user_sgprs = user_sgprs;
            env.read_mem = [fetch_stub](uint64_t, uint32_t* dst, uint32_t dwords) {  // s_swappc_b64 asks for the fetch shader here
                if (!fetch_stub || dwords < 1) return false;
                for (uint32_t k = 0; k < dwords; ++k) dst[k] = k == 0 ? 0xBE802000u : 0;  // s_setpc_b64 s[0:1], s[0:1]
                return true;
            };
        }
        gpu::Translation t = gpu::translate(w.data(), si.code_bytes / 4, env);
        if (write_spv && t.error.empty()) {  // <f>.new.spv beside the dump, to diff translator versions word by word
            const std::string sp = std::string(a).substr(0, std::string(a).rfind('.')) + ".new.spv";
            if (FILE* f = std::fopen(sp.c_str(), "wb")) { std::fwrite(t.spirv.data(), 4, t.spirv.size(), f); std::fclose(f); }
        }
        if (!report(a, t.spirv, t.error, dis)) ++bad;
    }
    return bad ? 1 : 0;
}
