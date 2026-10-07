// SPDX-License-Identifier: GPL-3.0-or-later
// GCN (Sea Islands / PS4 Liverpool) surface addressing: micro tiles (8x8 elements) and 2D/3D macro tiles over pipes and banks.
// Facts follow AMD's public addrlib behaviour (MIT) and the PS4 default tile-mode table; one sample, bank swizzle 0; mip layout: mip_chain.
#pragma once
#include <algorithm>
#include <bit>
#include <cstdint>

namespace bb::gpu::tiling {

enum Micro : uint32_t { kDisplay = 0, kThin = 1 };
enum Pipe : uint32_t { kP8_32x32_8x16 = 10, kP8_32x32_16x16 = 12 };

// Order of the 6 address bits inside an 8x8 micro tile (bit 0 first). Display micro tiling depends on the element size.
inline uint32_t micro_pixel_index(uint32_t micro, uint32_t bpp, uint32_t x, uint32_t y) {
    const uint32_t x0 = x & 1, x1 = (x >> 1) & 1, x2 = (x >> 2) & 1, y0 = y & 1, y1 = (y >> 1) & 1, y2 = (y >> 2) & 1;
    uint32_t p[6];
    if (micro == kDisplay) {
        switch (bpp) {
            case 8: p[0] = x0; p[1] = x1; p[2] = x2; p[3] = y1; p[4] = y0; p[5] = y2; break;
            case 16: p[0] = x0; p[1] = x1; p[2] = x2; p[3] = y0; p[4] = y1; p[5] = y2; break;
            case 32: p[0] = x0; p[1] = x1; p[2] = y0; p[3] = x2; p[4] = y1; p[5] = y2; break;
            case 64: p[0] = x0; p[1] = y0; p[2] = x1; p[3] = x2; p[4] = y1; p[5] = y2; break;
            default: p[0] = x0; p[1] = y0; p[2] = x1; p[3] = y1; p[4] = x2; p[5] = y2; break;  // 128
        }
    } else {
        p[0] = x0; p[1] = y0; p[2] = x1; p[3] = y1; p[4] = x2; p[5] = y2;
    }
    return p[0] | (p[1] << 1) | (p[2] << 2) | (p[3] << 3) | (p[4] << 4) | (p[5] << 5);
}

// Pixel order inside an 8x8x4 thick micro tile (volume textures, Thick1DThick): z0/z1 interleave per element size.
inline uint32_t micro_pixel_index_thick(uint32_t bpp, uint32_t x, uint32_t y, uint32_t z) {
    const uint32_t x0 = x & 1, x1 = (x >> 1) & 1, x2 = (x >> 2) & 1, y0 = y & 1, y1 = (y >> 1) & 1, y2 = (y >> 2) & 1, z0 = z & 1, z1 = (z >> 1) & 1;
    uint32_t p[6];
    if (bpp <= 16) { p[0] = x0; p[1] = y0; p[2] = x1; p[3] = y1; p[4] = z0; p[5] = z1; }
    else if (bpp == 32) { p[0] = x0; p[1] = y0; p[2] = x1; p[3] = z0; p[4] = y1; p[5] = z1; }
    else { p[0] = x0; p[1] = y0; p[2] = z0; p[3] = x1; p[4] = y1; p[5] = z1; }
    return p[0] | (p[1] << 1) | (p[2] << 2) | (p[3] << 3) | (p[4] << 4) | (p[5] << 5) | (x2 << 6) | (y2 << 7);
}

// Parameters of one macro-tiled mode (tile index = GB_TILE_MODE index of the T# / CB / DB descriptor).
struct Macro {
    uint32_t array_mode;        // 4 = 2D thin, 5 = PRT thin, 6 = PRT 2D thin, 11 = PRT 3D thin, 12 = 3D thin
    uint32_t micro;             // Micro
    uint32_t pipe_cfg;          // Pipe
    uint32_t sample_split;
    uint32_t num_pipes = 8, pipe_bits = 3;
    uint32_t bank_w = 1, bank_h = 1, num_banks = 16, bank_bits = 4, aspect = 2;
    uint32_t tile_split_bytes = 256;
    bool prt = false;
};

// ponytail: thin (non-thick, non-depth) modes only; thick/depth modes return false and the texture stays empty.
inline bool macro_mode(uint32_t tile_index, uint32_t bpp, Macro& m) {
    switch (tile_index) {
        case 10: m = {4, kDisplay, kP8_32x32_16x16, 2}; break;                                   // Display2DThin
        case 11: m = {5, kDisplay, kP8_32x32_8x16, 2}; m.prt = true; break;                      // DisplayThinPrt
        case 12: m = {6, kDisplay, kP8_32x32_16x16, 2}; m.prt = true; break;                     // Display2DThinPrt
        case 14: m = {4, kThin, kP8_32x32_16x16, 2}; break;                                      // Thin2DThin
        case 15: m = {12, kThin, kP8_32x32_8x16, 2}; break;                                                  // Thin3DThin
        case 16: m = {5, kThin, kP8_32x32_8x16, 2}; m.prt = true; break;                         // ThinThinPrt
        case 17: m = {6, kThin, kP8_32x32_16x16, 2}; m.prt = true; break;                        // Thin2DThinPrt
        case 18: m = {11, kThin, kP8_32x32_16x16, 2}; m.prt = true; break;                                      // Thin3DThinPrt
        default: return false;
    }
    // macro tile mode index: log2(bytes of one tile, split-limited / 64); PRT modes use the upper half of the table
    const uint32_t tile_bytes_1x = bpp * 64 / 8;
    m.tile_split_bytes = std::min<uint32_t>(1024, std::max<uint32_t>(256, m.sample_split * tile_bytes_1x));
    const uint32_t idx = std::bit_width(std::min(m.tile_split_bytes, tile_bytes_1x) / 64) - 1;
    static const uint8_t bank_h[8] = {4, 2, 1, 1, 1, 1, 1, 1};
    static const uint8_t banks[8] = {16, 16, 16, 16, 8, 4, 2, 2};
    static const uint8_t aspect[8] = {4, 2, 2, 2, 1, 1, 1, 1};
    static const uint8_t bank_h_prt[8] = {8, 4, 2, 1, 1, 1, 1, 1};
    static const uint8_t banks_prt[8] = {16, 16, 16, 16, 8, 4, 2, 2};
    static const uint8_t aspect_prt[8] = {4, 4, 2, 2, 1, 1, 1, 1};
    const uint32_t i = std::min<uint32_t>(idx, 7);
    m.bank_h = m.prt ? bank_h_prt[i] : bank_h[i];
    m.num_banks = m.prt ? banks_prt[i] : banks[i];
    m.aspect = m.prt ? aspect_prt[i] : aspect[i];
    m.bank_bits = std::countr_zero(m.num_banks);
    return true;
}

inline uint32_t macro_pitch(const Macro& m) { return 8 * m.bank_w * m.num_pipes * m.aspect; }
inline uint32_t macro_height(const Macro& m) { return 8 * m.bank_h * m.num_banks / m.aspect; }

// Byte offset of element (x, y, slice) in a macro-tiled surface; pitch/height are in elements and multiples of the macro tile.
inline uint64_t macro_addr(const Macro& m, uint32_t bpp, uint32_t x, uint32_t y, uint32_t slice, uint32_t pitch, uint32_t height) {
    const uint32_t pixel_index = micro_pixel_index(m.micro, bpp, x, y);
    uint32_t element_offset = pixel_index * bpp / 8;
    const uint32_t micro_tile_bytes_full = 64 * bpp / 8;
    uint32_t micro_tile_bytes = micro_tile_bytes_full, slices_per_tile = 1, tile_split_slice = 0;
    if (micro_tile_bytes_full > m.tile_split_bytes) {
        slices_per_tile = micro_tile_bytes_full / m.tile_split_bytes;
        tile_split_slice = element_offset / m.tile_split_bytes;
        element_offset %= m.tile_split_bytes;
        micro_tile_bytes = m.tile_split_bytes;
    }
    const uint32_t mt_pitch = macro_pitch(m), mt_height = macro_height(m);
    const uint64_t macro_tile_bytes = uint64_t(micro_tile_bytes) * (mt_pitch / 8) * (mt_height / 8) / (m.num_pipes * m.num_banks);
    const uint32_t tiles_per_row = pitch / mt_pitch;
    const uint64_t macro_tile_offset = (uint64_t(y / mt_height) * tiles_per_row + x / mt_pitch) * macro_tile_bytes;
    const uint64_t slice_bytes = uint64_t(tiles_per_row) * (height / mt_height) * macro_tile_bytes;
    const uint64_t slice_offset = slice_bytes * (tile_split_slice + slices_per_tile * slice);
    const uint32_t tile_row = (y / 8) % m.bank_h, tile_col = ((x / 8) / m.num_pipes) % m.bank_w;
    const uint64_t total = slice_offset + macro_tile_offset + element_offset + uint64_t(tile_row * m.bank_w + tile_col) * micro_tile_bytes;

    uint32_t px = x, py = y;
    if (m.prt) { px %= mt_pitch; py %= mt_height; }
    const uint32_t tx = px / 8, ty = py / 8;
    const auto bit = [](uint32_t v, uint32_t b) { return (v >> b) & 1u; };
    const uint32_t x3 = bit(tx, 0), x4 = bit(tx, 1), x5 = bit(tx, 2), y3 = bit(ty, 0), y4 = bit(ty, 1), y5 = bit(ty, 2);
    uint32_t pipe = m.pipe_cfg == kP8_32x32_8x16 ? (x4 ^ y3 ^ x5) | ((x3 ^ y4) << 1) | ((x5 ^ y5) << 2) : (x3 ^ y3 ^ x4) | ((x4 ^ y4) << 1) | ((x5 ^ y5) << 2);
    if (m.array_mode == 12) pipe ^= (std::max(1u, m.num_pipes / 2 - 1) * slice) & (m.num_pipes - 1);

    const uint32_t bx = px / 8 / (m.bank_w * m.num_pipes), by = py / 8 / m.bank_h;
    const uint32_t bx3 = bit(bx, 0), bx4 = bit(bx, 1), bx5 = bit(bx, 2), bx6 = bit(bx, 3), by3 = bit(by, 0), by4 = bit(by, 1), by5 = bit(by, 2), by6 = bit(by, 3);
    uint32_t bank = 0;
    if (m.num_banks == 16) bank = (bx3 ^ by6) | ((bx4 ^ by5 ^ by6) << 1) | ((bx5 ^ by4) << 2) | ((bx6 ^ by3) << 3);
    else if (m.num_banks == 8) bank = (bx3 ^ by5) | ((bx4 ^ by4 ^ by5) << 1) | ((bx5 ^ by3) << 2);
    else if (m.num_banks == 4) bank = (bx3 ^ by4) | ((bx4 ^ by3) << 1);
    else if (m.num_banks == 2) bank = bx3 ^ by3;
    uint32_t slice_rotation = 0;
    if (m.array_mode == 4) slice_rotation = (m.num_banks / 2 - 1) * slice;
    else if (m.array_mode == 12) slice_rotation = std::max(1u, m.num_pipes / 2 - 1) * slice / m.num_pipes;
    const uint32_t tile_split_rotation = (m.array_mode == 4 || m.array_mode == 12 || m.array_mode == 6 || m.array_mode == 11) ? (m.num_banks / 2 + 1) * tile_split_slice : 0;
    bank = (bank ^ slice_rotation ^ tile_split_rotation) & (m.num_banks - 1);

    constexpr uint32_t kInterleaveBits = 8;  // 256-byte pipe interleave
    return (total & ((1u << kInterleaveBits) - 1)) | (uint64_t(pipe) << kInterleaveBits) | (uint64_t(bank) << (kInterleaveBits + m.pipe_bits)) |
           ((total >> kInterleaveBits) << (kInterleaveBits + m.pipe_bits + m.bank_bits));
}

// One mip level in guest memory: texels; elements of the image level; memory pitch/height in elements; storage; slice bytes; offset.
enum Mode : uint32_t { kLinear = 0, k1D = 1, k2D = 2, kThick = 3 };
struct Level { uint32_t w, h, ew, eh, pe, he, mode, micro; uint64_t slice, off; };

// Levels 0..n-1 of a surface; returns the bytes read. Levels follow level 0, each with all its slices (addrlib order). Per level: the
// level-0 pitch and height >> level, rounded up to a power of two with POW2_PAD, then aligned for the level's mode like addrlib:
// linear-aligned rows to 64 bytes / slices to 256 (level 0: the descriptor's pitch as is), 1D tiles to 8x8 elements and 256-byte
// slices, 2D to whole macro tiles; a 2D level smaller than one macro tile is stored 1D. Slice count padded to a power of two: cube maps
// on every level (addrlib PadDimensions, padDims 3, no noCubeMipSlicesPad: a cube's 6 faces take 8 slots; the Dream's BC6H probe has
// float junk exactly in slots 6-7 of each level), arrays from level 1 on or with POW2_PAD (PostComputeMipLevel). The padding only moves
// the next level: the bytes read end after the last used slice. ponytail: no PRT mip tail.
// `block`: texels per element side (4 for BC), `bytes`: per element; `mt` for k2D only; `micro`: the 1D micro order (k2D: mt->micro).
inline uint64_t mip_chain(Level* lv, uint32_t n, uint32_t width, uint32_t height, uint32_t pitch, uint32_t slices, uint32_t block, uint32_t bytes,
                          uint32_t mode, uint32_t micro, bool linear_aligned, bool pow2, const Macro* mt, bool cube = false) {
    auto align = [](uint32_t v, uint32_t a) { return (v + a - 1) / a * a; };
    uint64_t total = 0, end = 0;
    for (uint32_t i = 0; i < n; ++i) {
        Level& v = lv[i];
        v.w = std::max(1u, width >> i), v.h = std::max(1u, height >> i);
        v.ew = (v.w + block - 1) / block, v.eh = (v.h + block - 1) / block;
        v.pe = (std::max(1u, pitch >> i) + block - 1) / block, v.he = v.eh;
        if (pow2) v.pe = std::bit_ceil(v.pe), v.he = std::bit_ceil(v.he);
        v.mode = mode, v.micro = mode == k2D ? mt->micro : micro;
        if (mode == k2D && i > 0 && (v.pe < macro_pitch(*mt) || v.he < macro_height(*mt))) v.mode = k1D;
        if (v.mode == kLinear && linear_aligned && i > 0) {
            const uint32_t pa = std::max(8u, 64 / bytes);
            v.pe = align(v.pe, pa);
            while (uint64_t(v.pe) * v.he % std::max(64u, 256 / bytes)) v.pe += pa;
        } else if (v.mode == k1D || v.mode == kThick) {
            v.pe = align(v.pe, 8), v.he = align(v.he, 8);
            while (uint64_t(v.pe) * v.he * bytes * (v.mode == kThick ? 4 : 1) % 256) v.pe += 8;
        } else if (v.mode == k2D) v.pe = align(v.pe, macro_pitch(*mt)), v.he = align(v.he, macro_height(*mt));
        v.slice = uint64_t(v.pe) * v.he * bytes;
        v.off = total;
        const uint32_t used = v.mode == kThick ? align(slices, 4) : slices;
        end = total + v.slice * used;
        total += v.slice * (v.mode == kThick ? used : cube || pow2 || i > 0 ? std::bit_ceil(slices) : slices);
    }
    return end;
}

}  // namespace bb::gpu::tiling
