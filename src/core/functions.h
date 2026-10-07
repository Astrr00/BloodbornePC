// SPDX-License-Identifier: GPL-3.0-or-later
// Function discovery for the recompiler.
//  1. .eh_frame_hdr / .eh_frame (PT_GNU_EH_FRAME): exact boundaries for every function with unwind info. Format:
//     https://refspecs.linuxfoundation.org/LSB_5.0.0/LSB-Core-generic/LSB-Core-generic/ehframechpt.html
//  2. Direct call/jump targets and R_X86_64_RELATIVE targets that land in the code span but outside every known
//     function (functions without unwind info). Decoded with Zydis.
#pragma once
#include <cstdint>
#include <variant>
#include <vector>

#include "core/orbis_elf.h"

namespace bb {

enum class FunctionSource : uint8_t { EhFrame, CallTarget, AddressTaken, Relocation };

struct Function {
    uint64_t start, size;  // vaddr, bytes
    FunctionSource source = FunctionSource::EhFrame;
};

// Sorted by start. Error if the image has no usable PT_GNU_EH_FRAME.
std::variant<std::vector<Function>, elf::Error> find_functions_eh(const elf::Image& img);

// Adds functions referenced by direct call/jmp, by RIP-relative `lea` (address taken without a relocation in a
// position-independent image; found by decoding every known function, transitively) or by R_X86_64_RELATIVE
// relocations, restricted to the code span of `known`. The size of a new function is the highest instruction end
// reached by recursive descent before the next known start; indirect jumps end a path (under-size possible).
// Candidates whose first instruction does not decode are dropped. Returns all functions sorted by start.
std::vector<Function> discover_functions(const elf::Image& img, std::vector<Function> known);

const char* function_source_name(FunctionSource s);

// Indirect `jmp` sites inside known functions, classified. Resolved switch tables follow Clang's PIC pattern
//   lea B,[rip+T]; movsxd R,dword [B+I*4]; add R,B; jmp R
// with the entry count taken from a preceding `cmp I,imm` + `ja`/`jae` bound check.
struct JumpTable {
    uint64_t jmp;    // vaddr of the jmp instruction
    uint64_t table;  // vaddr of the int32 offset table (0 if unresolved)
    uint64_t function;
    std::vector<uint64_t> targets;  // absolute case addresses (empty if unresolved)
};

struct IndirectJumps {
    std::vector<JumpTable> tables;           // resolved switch tables
    std::vector<JumpTable> unresolved_reg;   // jmp reg not matching the pattern
    size_t memory_tail_calls = 0;            // jmp [reg+disp] (vtable / function-pointer tail calls)
    size_t rip_memory = 0;                   // jmp [rip+disp] (import/GOT style)
    size_t register_tail_calls = 0;          // jmp R where R was last loaded from memory (function pointer / vtable)
    size_t inside_table_data = 0;            // "jmp reg" decoded from switch-table data inside a body (not code)
    size_t bound_rejected = 0;               // tables whose cmp bound produced targets outside the function
    // Every non-immediate jmp lands in exactly one bucket:
    // total == tables + unresolved_reg + register_tail_calls + memory_tail_calls + rip_memory + inside_table_data
    size_t total = 0;
};

IndirectJumps find_indirect_jumps(const elf::Image& img, const std::vector<Function>& functions);

} // namespace bb
