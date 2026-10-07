// SPDX-License-Identifier: GPL-3.0-or-later
// x86-64 -> C++ emitter. Each guest function becomes `void f_<vaddr>(bb::rt::Context& c)` (see runtime/guest.h).
// Reachable instructions are found by recursive descent from the function start and its switch-table cases;
// branches become gotos, calls push the guest return address and call the target function directly.
// Instructions without a translation become a runtime trap (bb::rt::unsupported) and are counted.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/functions.h"

namespace bb {

struct EmitStats {
    uint64_t instructions = 0;  // reachable instructions emitted
    uint64_t unsupported = 0;   // of those, emitted as runtime trap
    std::map<std::string, uint64_t> unsupported_by_mnemonic;
    // Functions whose reachable code is fully translated and self-contained: no calls, no tail transfers, no
    // indirect jumps other than resolved switch tables, no fs:, no `ret imm`. Candidates for native differential tests.
    std::vector<uint64_t> leaves;
};

// One C++ translation unit for all `functions`, including the kFunctions dispatch table and kLoadBias. All absolute
// guest addresses in the output are vaddr + `bias` (see runtime/loader.h). If `out` is null, only the statistics
// are computed (used to measure coverage on the full eboot without writing gigabytes of code).
void emit_translation_unit(const elf::Image& img, const std::vector<Function>& functions, const IndirectJumps& jumps,
                           uint64_t bias, std::string* out, EmitStats& stats);

// Same code split into translation units of `per_shard` functions ("shard_NNNN.cpp", each declaring only what it
// defines or calls) plus "table.cpp" (kFunctions, kFunctionCount, kLoadBias). For the full eboot.
void emit_sharded(const elf::Image& img, const std::vector<Function>& functions, const IndirectJumps& jumps,
                  uint64_t bias, size_t per_shard, std::vector<std::pair<std::string, std::string>>& files,
                  EmitStats& stats);

// "kImports"/"kImportCount" definitions (see runtime/guest.h): one entry per imported symbol, with the runtime address
// of its PLT stub (found by decoding `jmp [rip+GOT]` stubs and matching the GOT slot to a JUMP_SLOT relocation).
// `names` maps NID -> symbol name for log output (may be empty).
std::string emit_import_table(const elf::Image& img, uint64_t bias,
                              const std::unordered_map<std::string, std::string>& names);
} // namespace bb
