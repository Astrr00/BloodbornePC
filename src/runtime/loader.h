// SPDX-License-Identifier: GPL-3.0-or-later
// Maps the game image into guest memory for recompiled code. Guest address = link-time vaddr + load bias; the bias is
// fixed at recompile time (bb::rt::kLoadBias) because recompiled code embeds absolute guest addresses.
// Plan for the game (M1.2): guest memory is reserved at its own host addresses (Context::base = 0) and the eboot
// (linked at vaddr 0) is loaded at bias 0x400000 - the address the PS4 uses and community patches assume.
#pragma once
#include <string>

#include "core/orbis_elf.h"
#include "runtime/guest.h"

namespace bb::rt {

// Highest vaddr (exclusive) covered by PT_LOAD segments.
uint64_t image_end(const elf::Image& img);

// Data imports (ELF symbol type OBJECT: __stack_chk_guard, _Stdout, vtables, typeinfo, ...) get a zeroed cell of
// kImportCellSize bytes in guest memory right behind the image; their GOT slots point there instead of at the
// synthetic function address. Cells occupy [import_cells_base, import_cells_end).
inline constexpr uint64_t kImportCellSize = 256;
uint64_t import_cells_base(const elf::Image& img, uint64_t bias);
uint64_t import_cells_end(const elf::Image& img, uint64_t bias);

// Copies PT_LOAD file data to guest vaddr+bias, zero-fills memsz tails (.bss), applies R_X86_64_RELATIVE relocations
// (value = addend + bias) and links imports: function imports get synthetic addresses (kImportBase + symbol*16),
// data imports their cell. Guest addresses [0, mapped_end) must be backed by memory at c.base and cover the cells.
bool load_image(Context& c, const elf::Image& img, uint64_t bias, uint64_t mapped_end, std::string& error);

} // namespace bb::rt
