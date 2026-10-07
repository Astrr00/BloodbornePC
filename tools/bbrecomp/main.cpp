// SPDX-License-Identifier: GPL-3.0-or-later
// bbrecomp: static recompiler x86-64 (Orbis eboot) -> C++.
//   --stats                     instruction census over all discovered functions (linear sweep)
//   --coverage                  run the emitter without writing; report translated vs unsupported instructions
//   --emit <file>               write one C++ translation unit (all functions + dispatch table)
//   --emit-leaves <file> <max>  only fully translated leaf functions (differential testing with bbdiff)
//   --emit-dir <dir> <n>        full output sharded into <n> functions per file + table.cpp + sources.cmake
//   [--bias <hex>]              override the load bias (default: elf::default_load_bias)
#include <Zydis/Zydis.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <optional>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>

#include "core/functions.h"
#include "core/nid.h"
#include "core/orbis_elf.h"
#include "core/util.h"
#include "recomp/emitter.h"

using namespace bb;

namespace {
constexpr const char* kUsage =
    "usage: bbrecomp <eboot.bin> --stats | --coverage | --emit <out.cpp> | --emit-leaves <out.cpp> <max>"
    " | --emit-dir <dir> <functions per shard>  [--bias <hex load bias>] [--names <symbols.txt>]\n";

void census(const elf::Image& img, const std::vector<Function>& fns) {
    ZydisDecoder dec;
    ZydisDecoderInit(&dec, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    ZydisDecodedInstruction insn;
    ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
    std::map<ZydisMnemonic, uint64_t> by_mnemonic;
    std::map<ZydisISASet, uint64_t> by_isa;
    uint64_t total = 0, fs_access = 0, syscalls = 0;
    for (auto& fn : fns) {
        auto body = img.at_vaddr(fn.start, fn.size);
        for (size_t off = 0; off < body.size();) {
            if (!ZYAN_SUCCESS(ZydisDecoderDecodeFull(&dec, body.data() + off, body.size() - off, &insn, ops))) {
                ++off;
                continue;
            }
            off += insn.length;
            ++total;
            ++by_mnemonic[insn.mnemonic];
            ++by_isa[insn.meta.isa_set];
            if (insn.mnemonic == ZYDIS_MNEMONIC_SYSCALL) ++syscalls;
            for (uint8_t k = 0; k < insn.operand_count_visible; ++k)
                if (ops[k].type == ZYDIS_OPERAND_TYPE_MEMORY && ops[k].mem.segment == ZYDIS_REGISTER_FS) ++fs_access;
        }
    }
    auto print_sorted = [&](const auto& m, auto name, size_t limit) {
        std::vector<std::pair<uint64_t, std::string>> v;
        for (auto& [k, n] : m) v.emplace_back(n, name(k));
        std::sort(v.rbegin(), v.rend());
        double cum = 0;
        for (size_t i = 0; i < v.size() && i < limit; ++i) {
            cum += 100.0 * double(v[i].first) / double(total);
            std::printf("  %-20s %10llu  %6.2f%%  cum %6.2f%%\n", v[i].second.c_str(), (unsigned long long)v[i].first,
                        100.0 * double(v[i].first) / double(total), cum);
        }
        std::printf("  (%zu distinct)\n", v.size());
    };
    std::printf("instructions: %llu in %zu functions, fs: accesses %llu, syscall %llu\n", (unsigned long long)total,
                fns.size(), (unsigned long long)fs_access, (unsigned long long)syscalls);
    std::printf("\nISA sets:\n");
    print_sorted(by_isa, [](ZydisISASet s) { return std::string(ZydisISASetGetString(s)); }, 40);
    std::printf("\nmnemonics:\n");
    print_sorted(by_mnemonic, [](ZydisMnemonic m) { return std::string(ZydisMnemonicGetString(m)); }, SIZE_MAX);
}

void print_emit_stats(const EmitStats& s) {
    std::printf("emitted %llu reachable instructions, %llu unsupported (%.2f%% translated)\n",
                (unsigned long long)s.instructions, (unsigned long long)s.unsupported,
                s.instructions ? 100.0 * double(s.instructions - s.unsupported) / double(s.instructions) : 0.0);
    std::vector<std::pair<uint64_t, std::string>> v;
    for (auto& [k, n] : s.unsupported_by_mnemonic) v.emplace_back(n, k);
    std::sort(v.rbegin(), v.rend());
    for (size_t i = 0; i < v.size() && i < 40; ++i)
        std::printf("  unsupported %-16s %llu\n", v[i].second.c_str(), (unsigned long long)v[i].first);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fputs(kUsage, stderr);
        return 2;
    }
    const std::string mode = argv[2];
    // Optional trailing "--bias <hex>" overrides the default load bias (tests exercise a non-zero bias).
    std::optional<uint64_t> bias_override;
    std::unordered_map<std::string, std::string> names;
    // Trailing options in any order: "--bias <hex>", "--names <symbols.txt>" (NID names for import logging).
    while (argc >= 5) {
        const std::string opt = argv[argc - 2];
        if (opt == "--bias") bias_override = std::strtoull(argv[argc - 1], nullptr, 16);
        else if (opt == "--names") names = load_nid_names(argv[argc - 1]);
        else break;
        argc -= 2;
    }
    if (!(mode == "--stats" || mode == "--coverage" || (mode == "--emit" && argc == 4) ||
          (mode == "--emit-leaves" && argc == 5) || (mode == "--emit-dir" && argc == 5))) {
        std::fputs(kUsage, stderr);
        return 2;
    }
    auto data = read_file(argv[1]);
    if (!data) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    auto parsed = elf::parse(*data);
    if (auto* e = std::get_if<elf::Error>(&parsed)) {
        std::fprintf(stderr, "error: %s\n", e->message.c_str());
        return 1;
    }
    const auto& img = std::get<elf::Image>(parsed);
    auto eh = find_functions_eh(img);
    if (auto* e = std::get_if<elf::Error>(&eh)) {
        std::fprintf(stderr, "error: %s\n", e->message.c_str());
        return 1;
    }
    auto fns = discover_functions(img, std::get<std::vector<Function>>(eh));
    if (mode == "--stats") {
        census(img, fns);
        return 0;
    }

    auto jumps = find_indirect_jumps(img, fns);
    EmitStats stats;
    std::string code;
    const uint64_t bias = bias_override.value_or(elf::default_load_bias(img));
    std::printf("load bias 0x%llx\n", (unsigned long long)bias);
    const std::string imports_tu = emit_import_table(img, bias, names);
    if (mode == "--emit-dir") {
        // Sharded output for the full eboot: shard_NNNN.cpp + table.cpp + sources.cmake (list for the build).
        std::vector<std::pair<std::string, std::string>> files;
        emit_sharded(img, fns, jumps, bias, std::strtoull(argv[4], nullptr, 10), files, stats);
        files.emplace_back("imports.cpp", imports_tu);
        print_emit_stats(stats);
        const std::filesystem::path dir = argv[3];
        std::filesystem::create_directories(dir);
        std::string list = "set(BB_GENERATED_SOURCES\n";
        uint64_t bytes = 0;
        for (auto& [name, content] : files) {
            // Rewrite only changed files so incremental builds stay incremental.
            const auto path = dir / name;
            auto old = read_file(path);
            if (!old || old->size() != content.size() || std::memcmp(old->data(), content.data(), content.size())) {
                std::ofstream f(path, std::ios::binary);
                f << content;
                if (!f) {
                    std::fprintf(stderr, "cannot write %s\n", path.string().c_str());
                    return 1;
                }
            }
            bytes += content.size();
            list += "    ${CMAKE_CURRENT_LIST_DIR}/" + name + "\n";
        }
        list += ")\n";
        std::ofstream(dir / "sources.cmake", std::ios::binary) << list;
        std::printf("wrote %zu files (%.1f MiB) to %s\n", files.size(), double(bytes) / (1 << 20), argv[3]);
        return 0;
    }
    if (mode == "--emit-leaves") {
        // Pass 1 classifies, pass 2 emits an evenly spaced sample of the leaf functions only.
        emit_translation_unit(img, fns, jumps, bias, nullptr, stats);
        const size_t max = std::strtoull(argv[4], nullptr, 10);
        std::vector<Function> pick;
        const size_t n = stats.leaves.size(), step = std::max<size_t>(1, n / std::max<size_t>(1, max));
        std::map<uint64_t, const Function*> by_start;
        for (auto& f : fns) by_start[f.start] = &f;
        for (size_t i = 0; i < n && pick.size() < max; i += step) pick.push_back(*by_start.at(stats.leaves[i]));
        EmitStats sub;
        emit_translation_unit(img, pick, jumps, bias, &code, sub);
        std::printf("%zu leaf functions of %zu, emitting %zu\n", n, fns.size(), pick.size());
    } else {
        emit_translation_unit(img, fns, jumps, bias, mode == "--emit" ? &code : nullptr, stats);
        print_emit_stats(stats);
        std::printf("leaf functions: %zu\n", stats.leaves.size());
    }
    if (mode == "--emit" || mode == "--emit-leaves") {
        code += "\n" + imports_tu.substr(imports_tu.find("const bb::rt::ImportEntry"));
        std::ofstream f(argv[3], std::ios::binary);
        f << code;
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", argv[3]);
            return 1;
        }
        std::printf("wrote %s\n", argv[3]);
    }
    return 0;
}
