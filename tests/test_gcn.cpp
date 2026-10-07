// GCN decoder: encodings checked against the AMD Sea Islands ISA field layout.
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <string>

#include <vector>
#include "gpu/tiling.h"
#include "gpu/gcn.h"

using namespace bb::gcn;

static std::string d(std::initializer_list<uint32_t> w, size_t expect_len) {
    const uint32_t buf[4] = {*w.begin(), w.size() > 1 ? *(w.begin() + 1) : 0, 0, 0};
    const Insn i = decode(buf, w.size());
    assert(i.len == expect_len);
    return disasm(i);
}

// Macro-tiled addressing must map every element of a padded surface to a distinct byte range inside the surface.
static void check_tiling() {
    for (uint32_t tile : {10u, 11u, 12u, 14u, 15u, 16u, 17u, 18u})
        for (uint32_t bpp : {8u, 16u, 32u, 64u, 128u}) {
            bb::gpu::tiling::Macro m;
            assert(bb::gpu::tiling::macro_mode(tile, bpp, m));
            const uint32_t pitch = bb::gpu::tiling::macro_pitch(m) * 2, height = bb::gpu::tiling::macro_height(m) * 2, slices = 4;
            const uint64_t bytes = uint64_t(pitch) * height * (bpp / 8) * slices;
            std::vector<uint8_t> seen(bytes / (bpp / 8), 0);
            for (uint32_t s = 0; s < slices; ++s)
                for (uint32_t y = 0; y < height; ++y)
                    for (uint32_t x = 0; x < pitch; ++x) {
                        const uint64_t a = bb::gpu::tiling::macro_addr(m, bpp, x, y, s, pitch, height);
                        assert(a % (bpp / 8) == 0 && a + bpp / 8 <= bytes);
                        assert(!seen[a / (bpp / 8)]++);
                    }
        }
    bb::gpu::tiling::Macro thick;
    assert(!bb::gpu::tiling::macro_mode(19, 32, thick));  // thick modes stay unsupported
}

// Mip chains: offsets and padding per level (addrlib rules, see tiling::mip_chain).
static void check_mips() {
    using namespace bb::gpu::tiling;
    Level lv[16];
    // 256x256 RGBA8 1D thin: levels shrink by 4x down to the 8x8-element / 256-byte minimum
    assert(mip_chain(lv, 9, 256, 256, 256, 1, 1, 4, k1D, kThin, false, false, nullptr) == 350208);
    assert(lv[1].off == 262144 && lv[5].off == 349184 && lv[5].slice == 256 && lv[6].pe == 8 && lv[6].he == 8 && lv[6].ew == 4 && lv[8].off == 349952);
    // 2D thin 32 bpp (macro tile 128x64): 2D while a level covers one macro tile, then 1D with the mode's micro order
    Macro m;
    assert(macro_mode(14, 32, m) && macro_pitch(m) == 128 && macro_height(m) == 64);
    assert(mip_chain(lv, 4, 512, 512, 512, 2, 1, 4, k2D, kDisplay, false, false, &m) == (1048576 + 262144 + 65536 + 16384) * 2);
    assert(lv[2].mode == k2D && lv[3].mode == k1D && lv[3].micro == kThin && lv[3].off == (1048576 + 262144 + 65536) * 2);
    // BC1 (4x4 blocks of 8 bytes): elements are blocks, small levels pad to 8x8 blocks
    assert(mip_chain(lv, 4, 64, 64, 64, 1, 4, 8, k1D, kThin, false, false, nullptr) == 2048 + 512 * 3 && lv[2].ew == 4 && lv[3].w == 8);
    // linear aligned, 8 bpp: level 0 keeps the descriptor pitch, level 1 pads rows to 64 bytes and the slice to 256
    mip_chain(lv, 2, 100, 10, 100, 1, 1, 1, kLinear, kThin, true, false, nullptr);
    assert(lv[0].pe == 100 && lv[1].off == 1000 && lv[1].pe == 256 && lv[1].ew == 50);
    // POW2_PAD: a 100x60 chain reads level 1 as 64x32 elements of memory
    mip_chain(lv, 2, 100, 60, 100, 1, 1, 4, k1D, kThin, false, true, nullptr);
    assert(lv[0].pe == 128 && lv[0].he == 64 && lv[1].pe == 64 && lv[1].he == 32 && lv[1].ew == 50);
    // cube map (6 faces), 128x128 BC6H 1D thin: every level's slices pad to 8 (addrlib); the bytes read end after face 5 of the last level
    assert(mip_chain(lv, 3, 128, 128, 128, 6, 4, 16, k1D, kThin, false, false, nullptr, true) == 131072 + 32768 + 6144);
    assert(lv[1].off == 131072 && lv[1].slice == 4096 && lv[2].off == 163840);
    // array (6 slices): level 0 unpadded, from level 1 on padded to 8
    assert(mip_chain(lv, 3, 128, 128, 128, 6, 4, 16, k1D, kThin, false, false, nullptr) == 98304 + 32768 + 6144);
    assert(lv[1].off == 98304 && lv[2].off == 131072);
}

int main() {
    check_tiling();
    check_mips();
    assert(d({0xBEEB03FF, 0x14}, 2) == "s_mov_b32 vcc_hi, 0x14");
    assert(d({0xBF810000}, 1) == "s_endpgm 0x0");
    assert(d({0x7E000280}, 1) == "v_mov_b32 v0, 0");
    assert(d({0x40000000, 0x3F800000}, 2).rfind("v_madmk_f32", 0) == 0);  // implicit literal K follows madmk/madak
    assert(d({0xC0820700}, 1).rfind("s_load_dwordx4", 0) == 0);  // SMRD op 2
    assert(d({0xF800080F, 0x01010101}, 2).rfind("exp mrt0", 0) == 0);
    // Real VS trailer seen in the game: sig, then type=1 (VS), length 0x40 bytes, crc 0x89cda9e4.
    const uint32_t code[] = {0xBEEB03FF, 1, 0xBF810000, 0x5362724F, 0x07726468, 0x00004045, 0x00080000, 0x89cda9e4};
    ShaderInfo si;
    assert(shader_info(code, 8, si) && si.trailer_dw == 3 && si.code_bytes == 0x40 && si.type == 1 && si.crc == 0x89cda9e4);
    std::puts("gcn ok");
}
