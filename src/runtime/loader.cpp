// SPDX-License-Identifier: GPL-3.0-or-later
#include "runtime/loader.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace bb::rt {

uint64_t image_end(const elf::Image& img) {
    uint64_t end = 0;
    for (auto& s : img.segments)
        if (s.type == elf::PT_LOAD) end = std::max(end, s.vaddr + s.memsz);
    return end;
}
namespace {
bool is_data_import(const elf::Symbol& s) { return s.shndx == 0 && s.type == 1 /*STT_OBJECT*/ && !s.nid.empty(); }
constexpr uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
// NID of __stack_chk_guard: the cell is initialised with a fixed non-zero canary.
constexpr const char* kStackChkGuardNid = "f7uOxY9mM1U";
constexpr uint64_t kStackChkGuardValue = 0x5BB5BB5BB5BB5BB5ull;
} // namespace

uint64_t import_cells_base(const elf::Image& img, uint64_t bias) { return align_up(image_end(img) + bias, 0x10000); }

uint64_t import_cells_end(const elf::Image& img, uint64_t bias) {
    uint64_t n = 0;
    for (auto& s : img.symbols) n += is_data_import(s);
    return import_cells_base(img, bias) + n * kImportCellSize;
}

bool load_image(Context& c, const elf::Image& img, uint64_t bias, uint64_t mapped_end, std::string& error) {
    if (import_cells_end(img, bias) > mapped_end) {
        error = "guest memory does not cover the image and its import cells";
        return false;
    }
    for (auto& s : img.segments) {
        if (s.type != elf::PT_LOAD) continue;
        if (s.filesz > s.memsz || s.offset > img.elf.size() || s.filesz > img.elf.size() - s.offset) {
            error = "PT_LOAD segment out of file bounds";
            return false;
        }
        std::memcpy(host(c, s.vaddr + bias), img.elf.data() + s.offset, s.filesz);
        std::memset(host(c, s.vaddr + bias + s.filesz), 0, s.memsz - s.filesz);
    }
    // Address a symbol resolves to: its cell (data) or a synthetic import address (functions).
    std::vector<uint64_t> target(img.symbols.size());
    uint64_t next_cell = import_cells_base(img, bias);
    for (size_t i = 0; i < img.symbols.size(); ++i) {
        const auto& s = img.symbols[i];
        if (is_data_import(s)) {
            std::memset(host(c, next_cell), 0, kImportCellSize);
            if (s.nid == kStackChkGuardNid) st<uint64_t>(c, next_cell, kStackChkGuardValue);
            target[i] = next_cell;
            next_cell += kImportCellSize;
        } else {
            target[i] = kImportBase + i * kImportStride;
        }
    }
    for (auto& r : img.relocations) {
        if (r.offset + bias + 8 > mapped_end) {
            error = "relocation outside guest memory";
            return false;
        }
        if (r.symbol >= img.symbols.size()) {
            error = "relocation symbol index out of range";
            return false;
        }
        switch (r.type) {
        case elf::R_X86_64_RELATIVE: st<uint64_t>(c, r.offset + bias, uint64_t(r.addend) + bias); break;
        case elf::R_X86_64_GLOB_DAT:
        case elf::R_X86_64_JUMP_SLOT: st<uint64_t>(c, r.offset + bias, target[r.symbol]); break;
        case elf::R_X86_64_64:
            if (r.symbol) st<uint64_t>(c, r.offset + bias, target[r.symbol] + uint64_t(r.addend));
            else st<uint64_t>(c, r.offset + bias, uint64_t(r.addend) + bias);
            break;
        default: break;  // TLS relocations: handled by setup_main_thread
        }
    }
    return true;
}

} // namespace bb::rt
