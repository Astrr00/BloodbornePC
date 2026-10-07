// SPDX-License-Identifier: GPL-3.0-or-later
// bbelf: inspect an Orbis eboot.bin - segments, modules/libraries, imports (NIDs), relocations.
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>

#include "core/functions.h"
#include "core/hash.h"
#include "core/nid.h"
#include "core/orbis_elf.h"
#include "core/util.h"

namespace fs = std::filesystem;
using namespace bb;

namespace {

constexpr const char* kUsage = R"(usage: bbelf <eboot.bin> [--names <symbols.txt>] [--csv <imports.csv>] [--relocs]
                         [--functions <functions.csv>] [--jumptables <jumptables.csv>]
  --names       symbol name list (one per line); NIDs are computed and matched
  --csv         write symbol relocations as CSV (vaddr,type,library,module,nid,name) for Ghidra import
  --relocs      print every relocation instead of a per-type summary
  --functions   list functions (.eh_frame + discovery) as CSV (start,size,source) and print coverage
  --jumptables  classify indirect jmps; CSV (jmp,function,table,entries) of switch tables
)";

struct Options {
    fs::path input;
    std::optional<fs::path> names, csv, functions, jumptables;
    bool relocs = false;
};

std::optional<Options> parse_args(int argc, char** argv) {
    if (argc < 2) return std::nullopt;
    Options o{argv[1]};
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--relocs") o.relocs = true;
        else if (k == "--names" && i + 1 < argc) o.names = argv[++i];
        else if (k == "--csv" && i + 1 < argc) o.csv = argv[++i];
        else if (k == "--functions" && i + 1 < argc) o.functions = argv[++i];
        else if (k == "--jumptables" && i + 1 < argc) o.jumptables = argv[++i];
        else return std::nullopt;
    }
    return o;
}

struct Resolved {
    std::string library, module, name;
};

} // namespace

int main(int argc, char** argv) {
    auto opt = parse_args(argc, argv);
    if (!opt) {
        std::fputs(kUsage, stderr);
        return 2;
    }
    auto data = read_file(opt->input);
    if (!data) {
        std::fprintf(stderr, "cannot read %s\n", opt->input.string().c_str());
        return 1;
    }
    auto parsed = elf::parse(*data);
    if (auto* e = std::get_if<elf::Error>(&parsed)) {
        std::fprintf(stderr, "error: %s\n", e->message.c_str());
        return 1;
    }
    const auto& img = std::get<elf::Image>(parsed);
    std::unordered_map<std::string, std::string> names;
    if (opt->names) names = load_nid_names(*opt->names);

    std::printf("file      %s (%s, %zu bytes)\n", opt->input.string().c_str(), img.from_self ? "SELF" : "ELF",
                data->size());
    std::printf("type      %s (0x%04X)\n", elf::elf_type_name(img.elf_type), img.elf_type);
    std::printf("entry     0x%llx\n", (unsigned long long)img.entry);
    if (img.sdk_version) std::printf("sdk       0x%08llx\n", (unsigned long long)img.sdk_version);
    if (!img.original_filename.empty()) std::printf("origname  %s\n", img.original_filename.c_str());
    if (!img.fingerprint.empty()) std::printf("fingerpr  %s\n", to_hex(img.fingerprint).c_str());

    std::printf("\nsegments (%zu)\n  %-16s %-4s %-18s %-12s %-12s %s\n", img.segments.size(), "type", "rwx", "vaddr",
                "filesz", "memsz", "offset");
    for (auto& s : img.segments) {
        char rwx[4] = {s.flags & 4 ? 'r' : '-', s.flags & 2 ? 'w' : '-', s.flags & 1 ? 'x' : '-', 0};
        std::printf("  %-16s %-4s 0x%016llx 0x%010llx 0x%010llx 0x%llx\n", elf::segment_type_name(s.type), rwx,
                    (unsigned long long)s.vaddr, (unsigned long long)s.filesz, (unsigned long long)s.memsz,
                    (unsigned long long)s.offset);
    }

    for (auto& m : img.self_module) std::printf("\nmodule    %s v%u.%u\n", m.name.c_str(), m.major, m.minor);
    std::printf("\nneeded modules (%zu)\n", img.needed_modules.size());
    for (auto& m : img.needed_modules) std::printf("  [%u] %s v%u.%u\n", m.id, m.name.c_str(), m.major, m.minor);

    auto resolve = [&](const elf::Symbol& s) {
        bool imported = s.shndx == 0;
        const elf::Library* lib = img.find_library(s.library_id, !imported);
        const elf::Module* mod = img.find_module(s.module_id);
        auto it = names.find(s.nid);
        return Resolved{lib ? lib->name : "?", mod ? mod->name : "?", it != names.end() ? it->second : ""};
    };

    // Imports grouped by library.
    std::map<std::string, std::vector<const elf::Symbol*>> by_lib;
    size_t imports = 0, exports = 0, resolved = 0;
    for (auto& s : img.symbols) {
        if (s.nid.empty() || s.library_id < 0) continue;
        if (s.shndx != 0) {
            ++exports;
            continue;
        }
        ++imports;
        auto r = resolve(s);
        if (!r.name.empty()) ++resolved;
        by_lib[r.library + " (" + r.module + ")"].push_back(&s);
    }
    std::printf("\nlibraries (%zu)\n", img.libraries.size());
    for (auto& l : img.libraries)
        std::printf("  [%u] %-28s v%u %s\n", l.id, l.name.c_str(), l.version, l.exported ? "export" : "import");
    std::printf("\nimports: %zu symbols, %zu named; exports: %zu\n", imports, resolved, exports);
    for (auto& [lib, syms] : by_lib) {
        std::printf("  %s: %zu\n", lib.c_str(), syms.size());
        for (auto* s : syms) {
            auto it = names.find(s->nid);
            std::printf("    %s %s\n", s->nid.c_str(), it != names.end() ? it->second.c_str() : "");
        }
    }

    std::map<uint32_t, size_t> per_type;
    for (auto& r : img.relocations) ++per_type[r.type];
    std::printf("\nrelocations (%zu)\n", img.relocations.size());
    if (opt->relocs) {
        for (auto& r : img.relocations) {
            std::string sym = r.symbol && r.symbol < img.symbols.size() ? img.symbols[r.symbol].raw : "";
            std::printf("  0x%016llx %-20s %s %s%s0x%llx\n", (unsigned long long)r.offset, elf::reloc_type_name(r.type),
                        r.plt ? "plt" : "   ", sym.c_str(), r.addend < 0 ? "-" : "+",
                        (unsigned long long)(r.addend < 0 ? 0 - uint64_t(r.addend) : uint64_t(r.addend)));
        }
    } else {
        for (auto& [t, n] : per_type) std::printf("  %-20s %zu\n", elf::reloc_type_name(t), n);
    }

    if (opt->csv) {
        std::ofstream f(*opt->csv, std::ios::binary);
        f << "vaddr,type,library,module,nid,name,stt\n";
        size_t rows = 0;
        for (auto& r : img.relocations) {
            if (!r.symbol || r.symbol >= img.symbols.size()) continue;
            auto& s = img.symbols[r.symbol];
            auto res = resolve(s);
            char addr[32];
            std::snprintf(addr, sizeof addr, "0x%llx", (unsigned long long)r.offset);
            f << addr << ',' << elf::reloc_type_name(r.type) << ',' << res.library << ',' << res.module << ','
              << s.nid << ',' << res.name << ',' << unsigned(s.type) << '\n';
            ++rows;
        }
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", opt->csv->string().c_str());
            return 1;
        }
        std::printf("\nwrote %zu rows to %s\n", rows, opt->csv->string().c_str());
    }

    if (opt->functions) {
        auto found = find_functions_eh(img);
        if (auto* e = std::get_if<elf::Error>(&found)) {
            std::fprintf(stderr, "error: functions: %s\n", e->message.c_str());
            return 1;
        }
        const auto& fns = std::get<std::vector<Function>>(found);
        uint64_t text_lo = 0, text_hi = 0;
        for (auto& s : img.segments)
            if (s.type == elf::PT_LOAD && (s.flags & 1)) text_lo = s.vaddr, text_hi = s.vaddr + s.filesz;
        // Coverage is measured over the function span [first start, last end]: the executable segment also holds
        // read-only data after the code, which has no FDEs. fns is sorted by start.
        uint64_t span_lo = fns.empty() ? text_lo : fns.front().start, span_hi = span_lo;
        for (auto& fn : fns) span_hi = std::max(span_hi, fn.start + fn.size);
        uint64_t covered = 0, end = span_lo, largest_gap = 0, gap_at = 0;
        for (auto& fn : fns) {
            if (fn.start > end && fn.start - end > largest_gap) largest_gap = fn.start - end, gap_at = end;
            uint64_t lo = std::max(fn.start, end), hi = fn.start + fn.size;
            if (hi > lo) covered += hi - lo;
            end = std::max(end, hi);
        }
        // Second pass: functions without unwind info, reached via direct call/jmp, RIP-relative lea or relocations.
        auto all = discover_functions(img, fns);
        size_t by_call = 0, by_lea = 0, by_reloc = 0;
        uint64_t new_bytes = 0, all_covered = 0, all_end = span_lo, largest_new = 0;
        for (auto& fn : all) {
            if (fn.source == FunctionSource::CallTarget) ++by_call;
            if (fn.source == FunctionSource::AddressTaken) ++by_lea;
            if (fn.source == FunctionSource::Relocation) ++by_reloc;
            if (fn.source != FunctionSource::EhFrame) new_bytes += fn.size, largest_new = std::max(largest_new, fn.size);
            uint64_t lo = std::max(fn.start, all_end), hi = fn.start + fn.size;
            if (hi > lo) all_covered += hi - lo;
            all_end = std::max(all_end, hi);
        }
        std::ofstream f(*opt->functions, std::ios::binary);
        f << "start,size,source\n";
        for (auto& fn : all) {
            char row[80];
            std::snprintf(row, sizeof row, "0x%llx,%llu,%s\n", (unsigned long long)fn.start,
                          (unsigned long long)fn.size, function_source_name(fn.source));
            f << row;
        }
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", opt->functions->string().c_str());
            return 1;
        }
        double pct = span_hi > span_lo ? 100.0 * double(covered) / double(span_hi - span_lo) : 0.0;
        std::printf("\nfunctions (.eh_frame): %zu, span 0x%llx-0x%llx covered %.2f%% (%llu of %llu bytes), "
                    "largest gap 0x%llx at 0x%llx\n",
                    fns.size(), (unsigned long long)span_lo, (unsigned long long)span_hi, pct,
                    (unsigned long long)covered, (unsigned long long)(span_hi - span_lo),
                    (unsigned long long)largest_gap, (unsigned long long)gap_at);
        std::printf("functions without unwind info: %zu via call/jmp, %zu via lea, %zu via relocation (%llu bytes, "
                    "largest %llu, sizes by recursive descent); total %zu, span covered %.2f%% (rest: padding or "
                    "unreached code)\n",
                    by_call, by_lea, by_reloc, (unsigned long long)new_bytes, (unsigned long long)largest_new,
                    all.size(),
                    span_hi > span_lo ? 100.0 * double(all_covered) / double(span_hi - span_lo) : 0.0);
        if (text_hi > span_hi)
            std::printf("executable segment beyond last function: 0x%llx-0x%llx (%llu bytes, no FDEs)\n",
                        (unsigned long long)span_hi, (unsigned long long)text_hi,
                        (unsigned long long)(text_hi - span_hi));
        std::printf("wrote %s\n", opt->functions->string().c_str());
    }

    if (opt->jumptables) {
        auto found = find_functions_eh(img);
        if (auto* e = std::get_if<elf::Error>(&found)) {
            std::fprintf(stderr, "error: functions: %s\n", e->message.c_str());
            return 1;
        }
        auto fns = discover_functions(img, std::get<std::vector<Function>>(found));
        auto ij = find_indirect_jumps(img, fns);
        size_t targets = 0, outside = 0;
        std::map<uint64_t, const Function*> by_start;
        for (auto& fn : fns) by_start[fn.start] = &fn;
        std::ofstream f(*opt->jumptables, std::ios::binary);
        f << "jmp,function,table,entries\n";
        for (auto& jt : ij.tables) {
            const Function* fn = by_start.at(jt.function);
            for (uint64_t t : jt.targets)
                if (t < fn->start || t >= fn->start + fn->size) ++outside;
            targets += jt.targets.size();
            char row[96];
            std::snprintf(row, sizeof row, "0x%llx,0x%llx,0x%llx,%zu\n", (unsigned long long)jt.jmp,
                          (unsigned long long)jt.function, (unsigned long long)jt.table, jt.targets.size());
            f << row;
        }
        for (auto& jt : ij.unresolved_reg) {
            char row[96];
            std::snprintf(row, sizeof row, "0x%llx,0x%llx,unresolved,0\n", (unsigned long long)jt.jmp,
                          (unsigned long long)jt.function);
            f << row;
        }
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", opt->jumptables->string().c_str());
            return 1;
        }
        std::printf("\nindirect jmps: %zu switch tables (%zu case targets, %zu outside their function, %zu bounds "
                    "rejected), %zu unresolved jmp reg, %zu jmp reg tail calls, %zu jmp [mem] tail calls, %zu jmp "
                    "[rip]; %zu decode artifacts inside table data dropped\n",
                    ij.tables.size(), targets, outside, ij.bound_rejected, ij.unresolved_reg.size(),
                    ij.register_tail_calls, ij.memory_tail_calls, ij.rip_memory, ij.inside_table_data);
        size_t sum = ij.tables.size() + ij.unresolved_reg.size() + ij.register_tail_calls + ij.memory_tail_calls +
                     ij.rip_memory + ij.inside_table_data;
        std::printf("classified %zu of %zu indirect jmps%s\n", sum, ij.total,
                    sum == ij.total ? "" : " - CLASSIFICATION MISMATCH");
        if (sum != ij.total) return 1;
        std::printf("wrote %s\n", opt->jumptables->string().c_str());
    }
    for (auto& w : img.warnings) std::fprintf(stderr, "warning: %s\n", w.c_str());
    return 0;
}
