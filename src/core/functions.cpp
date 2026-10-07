// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/functions.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <Zydis/Zydis.h>

namespace bb {
namespace {

constexpr uint8_t DW_EH_PE_omit = 0xFF;

// Cursor over guest memory: reads are bounds-checked through Image::at_vaddr.
struct Reader {
    const elf::Image& img;
    uint64_t pos;  // vaddr
    bool ok = true;

    template <typename T>
    T get() {
        auto s = img.at_vaddr(pos, sizeof(T));
        if (s.size() != sizeof(T)) {
            ok = false;
            return T{};
        }
        T v;
        std::memcpy(&v, s.data(), sizeof(T));
        pos += sizeof(T);
        return v;
    }
    uint64_t uleb() {
        uint64_t v = 0;
        for (int shift = 0; ok && shift < 64; shift += 7) {
            uint8_t b = get<uint8_t>();
            v |= uint64_t(b & 0x7F) << shift;
            if (!(b & 0x80)) break;
        }
        return v;
    }
    int64_t sleb() {
        int64_t v = 0;
        int shift = 0;
        uint8_t b = 0x80;
        while (ok && (b & 0x80) && shift < 64) {
            b = get<uint8_t>();
            v |= int64_t(b & 0x7F) << shift;
            shift += 7;
        }
        if (shift < 64 && (b & 0x40)) v |= -(int64_t(1) << shift);
        return v;
    }
    // DWARF EH pointer encoding: low nibble = format, high nibble = application (pcrel / datarel).
    std::optional<uint64_t> encoded(uint8_t enc, uint64_t datarel_base) {
        uint64_t field = pos;
        uint64_t v;
        switch (enc & 0x0F) {
        case 0x00: v = get<uint64_t>(); break;
        case 0x01: v = uleb(); break;
        case 0x02: v = get<uint16_t>(); break;
        case 0x03: v = get<uint32_t>(); break;
        case 0x04: v = get<uint64_t>(); break;
        case 0x09: v = uint64_t(sleb()); break;
        case 0x0A: v = uint64_t(int64_t(get<int16_t>())); break;
        case 0x0B: v = uint64_t(int64_t(get<int32_t>())); break;
        case 0x0C: v = get<uint64_t>(); break;
        default: return std::nullopt;
        }
        switch (enc & 0x70) {
        case 0x00: break;
        case 0x10: v += field; break;
        case 0x30: v += datarel_base; break;
        default: return std::nullopt;
        }
        return ok ? std::optional(v) : std::nullopt;
    }
};

// Returns the FDE pointer encoding ('R' augmentation) of the CIE at `cie`, or nullopt if unparsable.
std::optional<uint8_t> cie_fde_encoding(const elf::Image& img, uint64_t cie) {
    Reader r{img, cie};
    uint32_t len = r.get<uint32_t>();
    if (!r.ok || len == 0 || len == 0xFFFFFFFF) return std::nullopt;
    if (r.get<uint32_t>() != 0) return std::nullopt;  // CIE id in .eh_frame is 0
    uint8_t version = r.get<uint8_t>();
    std::string aug;
    for (char c; r.ok && (c = char(r.get<uint8_t>())) != 0;) aug.push_back(c);
    r.uleb();  // code alignment
    r.sleb();  // data alignment
    if (version == 1) r.get<uint8_t>();
    else r.uleb();  // return address register
    uint8_t fde_enc = 0x00;  // absptr if no 'R'
    if (!aug.empty() && aug[0] == 'z') {
        r.uleb();  // augmentation data length
        for (size_t i = 1; i < aug.size() && r.ok; ++i) {
            switch (aug[i]) {
            case 'R': fde_enc = r.get<uint8_t>(); break;
            case 'L': r.get<uint8_t>(); break;
            case 'P': {
                uint8_t penc = r.get<uint8_t>();
                if (!r.encoded(penc & 0x7F, 0)) return std::nullopt;
                break;
            }
            case 'S': break;
            default: return std::nullopt;
            }
        }
    }
    return r.ok ? std::optional(fde_enc) : std::nullopt;
}

} // namespace

std::variant<std::vector<Function>, elf::Error> find_functions_eh(const elf::Image& img) {
    const elf::Segment* seg = nullptr;
    for (auto& s : img.segments)
        if (s.type == elf::PT_GNU_EH_FRAME) seg = &s;
    if (!seg) return elf::Error{"no PT_GNU_EH_FRAME segment"};

    // .eh_frame_hdr: version, eh_frame_ptr_enc, fde_count_enc, table_enc, then the encoded values.
    const uint64_t hdr = seg->vaddr;
    Reader r{img, hdr};
    uint8_t version = r.get<uint8_t>(), ptr_enc = r.get<uint8_t>(), count_enc = r.get<uint8_t>(),
            table_enc = r.get<uint8_t>();
    if (!r.ok || version != 1) return elf::Error{".eh_frame_hdr has unsupported version"};
    if (!r.encoded(ptr_enc, hdr)) return elf::Error{".eh_frame_hdr: bad eh_frame_ptr"};
    if (count_enc == DW_EH_PE_omit || table_enc == DW_EH_PE_omit)
        return elf::Error{".eh_frame_hdr has no binary search table"};
    auto count = r.encoded(count_enc, hdr);
    if (!count) return elf::Error{".eh_frame_hdr: bad fde_count"};

    std::unordered_map<uint64_t, std::optional<uint8_t>> cie_cache;
    std::vector<Function> out;
    out.reserve(size_t(*count));
    for (uint64_t i = 0; i < *count; ++i) {
        auto loc = r.encoded(table_enc, hdr), fde = r.encoded(table_enc, hdr);
        if (!loc || !fde) return elf::Error{".eh_frame_hdr table truncated at entry " + std::to_string(i)};

        // FDE: length, CIE pointer (relative to its own field), pc_begin, pc_range.
        Reader f{img, *fde};
        uint32_t len = f.get<uint32_t>();
        uint64_t cie_field = f.pos;
        uint32_t cie_off = f.get<uint32_t>();
        if (!f.ok || len == 0 || len == 0xFFFFFFFF || cie_off == 0)
            return elf::Error{"malformed FDE at 0x" + std::to_string(*fde)};
        uint64_t cie = cie_field - cie_off;
        auto [it, inserted] = cie_cache.try_emplace(cie);
        if (inserted) it->second = cie_fde_encoding(img, cie);
        if (!it->second) return elf::Error{"unparsable CIE referenced by FDE " + std::to_string(i)};
        auto begin = f.encoded(*it->second, hdr);
        auto range = f.encoded(*it->second & 0x0F, hdr);  // pc_range: format only, no application
        if (!begin || !range) return elf::Error{"FDE " + std::to_string(i) + " has unreadable pc range"};
        if (*begin != *loc)
            return elf::Error{"FDE " + std::to_string(i) + " pc_begin disagrees with .eh_frame_hdr table"};
        out.push_back({*begin, *range});
    }
    std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.start < b.start; });
    return out;
}

const char* function_source_name(FunctionSource s) {
    switch (s) {
    case FunctionSource::EhFrame: return "eh_frame";
    case FunctionSource::CallTarget: return "call";
    case FunctionSource::AddressTaken: return "lea";
    case FunctionSource::Relocation: return "reloc";
    }
    return "?";
}

std::vector<Function> discover_functions(const elf::Image& img, std::vector<Function> known) {
    if (known.empty()) return known;
    std::sort(known.begin(), known.end(), [](auto& a, auto& b) { return a.start < b.start; });
    // Entry, DT_INIT/DT_FINI and the init/fini arrays can lie before the first unwind entry (a `.init` stub).
    std::vector<uint64_t> seeds{img.entry};
    for (auto& [tag, val] : img.dynamic)
        if (tag == elf::DT_INIT || tag == elf::DT_FINI) seeds.push_back(val);
    uint64_t lo = known.front().start;
    uint64_t hi = lo;
    for (auto& f : known) hi = std::max(hi, f.start + f.size);
    for (uint64_t s : seeds)
        if (s < lo && img.at_vaddr(s, 1).size() == 1) lo = s;

    // start -> {end, source}. Ends of discovered functions come from recursive descent when they are added, so a
    // later target inside a gap is only rejected if it really lies inside an already-traced body.
    std::map<uint64_t, std::pair<uint64_t, FunctionSource>> fns;
    for (auto& f : known) fns.emplace(f.start, std::pair{f.start + f.size, f.source});
    auto contained = [&](uint64_t a) {
        auto it = fns.upper_bound(a);
        if (it == fns.begin()) return false;
        --it;
        return a < it->second.first;
    };
    // Relocation targets (vtables, init arrays) are function entries: they bound a neighbour's recursive descent so a
    // FDE-less function cannot swallow the next entry (it would then be unreachable as a separate function).
    std::set<uint64_t> seed_starts;
    auto next_start = [&](uint64_t a) {
        auto it = fns.upper_bound(a);
        uint64_t n = it == fns.end() ? hi : it->first;
        auto s = seed_starts.upper_bound(a);
        return s != seed_starts.end() && *s < n ? *s : n;
    };

    ZydisDecoder dec;
    ZydisDecoderInit(&dec, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    struct Insn {
        ZydisDecodedInstruction i;
        ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
    };
    auto decode = [&](uint64_t a, uint64_t limit, Insn& out) {
        auto bytes = img.at_vaddr(a, std::min<uint64_t>(ZYDIS_MAX_INSTRUCTION_LENGTH, limit - a));
        return !bytes.empty() && ZYAN_SUCCESS(ZydisDecoderDecodeFull(&dec, bytes.data(), bytes.size(), &out.i, out.ops));
    };
    auto rel_target = [](const Insn& in, uint64_t ip, ZyanU64& target) {
        return in.i.operand_count_visible >= 1 && in.ops[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE &&
               in.ops[0].imm.is_relative && ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&in.i, &in.ops[0], ip, &target));
    };

    // Visits instructions of [s, limit). FDE functions (exact bounds) are swept linearly; functions without unwind
    // info are followed by recursive descent (conditional branches inside the bound, stop at ret/jmp/int3/ud2/hlt),
    // so trailing padding or a neighbouring unreferenced function is not swallowed. Returns the highest end reached.
    // Decode buffers are local, so on_insn may re-enter (add -> walk).
    // ponytail: indirect jumps (switch tables) end a path, so a function using one may be under-sized until 0.10.
    auto walk = [&](uint64_t s, uint64_t limit, bool linear, auto&& on_insn) {
        uint64_t max_end = s;
        std::vector<uint64_t> todo{s};
        std::unordered_set<uint64_t> seen;
        Insn in;
        while (!todo.empty()) {
            uint64_t a = todo.back();
            todo.pop_back();
            while (a < limit && seen.insert(a).second && decode(a, limit, in)) {
                const uint64_t ip = a;
                a += in.i.length;
                max_end = std::max(max_end, a);
                const Insn cur = in;  // on_insn may decode further instructions
                on_insn(ip, cur);
                if (linear) continue;
                const auto cat = cur.i.meta.category;
                ZyanU64 t;
                if (cat == ZYDIS_CATEGORY_RET || cur.i.mnemonic == ZYDIS_MNEMONIC_INT3 ||
                    cur.i.mnemonic == ZYDIS_MNEMONIC_UD2 || cur.i.mnemonic == ZYDIS_MNEMONIC_HLT)
                    break;
                if (cat == ZYDIS_CATEGORY_COND_BR && rel_target(cur, ip, t) && t >= s && t < limit) todo.push_back(t);
                if (cat == ZYDIS_CATEGORY_UNCOND_BR) {
                    if (rel_target(cur, ip, t) && t >= s && t < limit) todo.push_back(t);
                    break;
                }
            }
        }
        return max_end;
    };

    std::vector<std::pair<uint64_t, FunctionSource>> candidates;
    for (uint64_t s : seeds) candidates.emplace_back(s, FunctionSource::Relocation);
    for (auto& r : img.relocations)
        if (r.type == elf::R_X86_64_RELATIVE && uint64_t(r.addend) >= lo && uint64_t(r.addend) < hi)
            candidates.emplace_back(uint64_t(r.addend), FunctionSource::Relocation);
    for (auto& [t, src] : candidates) if (!contained(t)) seed_starts.insert(t);

    // Worklist: decode each function once, collect direct call/jmp and RIP-relative lea targets.
    std::vector<uint64_t> work;
    for (auto& [s, v] : fns) work.push_back(s);
    auto add = [&](uint64_t t, FunctionSource src) {
        if (t < lo || t >= hi || contained(t)) return;
        Insn probe;
        if (!decode(t, hi, probe)) return;
        uint64_t end = walk(t, next_start(t), false, [](uint64_t, const Insn&) {});
        fns.emplace(t, std::pair{end, src});
        work.push_back(t);
    };
    for (auto& [t, src] : candidates) add(t, src);
    while (!work.empty()) {
        uint64_t s = work.back();
        work.pop_back();
        auto [e, src] = fns.at(s);
        walk(s, e, src == FunctionSource::EhFrame, [&](uint64_t ip, const Insn& in) {
            ZyanU64 target;
            if (in.i.mnemonic == ZYDIS_MNEMONIC_LEA && in.i.operand_count_visible >= 2 &&
                in.ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY && in.ops[1].mem.base == ZYDIS_REGISTER_RIP) {
                if (ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&in.i, &in.ops[1], ip, &target)) && (target < s || target >= e))
                    add(target, FunctionSource::AddressTaken);
                return;
            }
            if (in.i.mnemonic != ZYDIS_MNEMONIC_CALL && in.i.mnemonic != ZYDIS_MNEMONIC_JMP) return;
            if (!rel_target(in, ip, target)) return;
            // An intra-function jmp is control flow, not a new function; a jmp leaving the body is a tail call.
            if (in.i.mnemonic == ZYDIS_MNEMONIC_JMP && target >= s && target < e) return;
            add(target, FunctionSource::CallTarget);
        });
    }

    std::vector<Function> out;
    out.reserve(fns.size());
    for (auto& [s, v] : fns) out.push_back({s, v.first - s, v.second});
    return out;
}

IndirectJumps find_indirect_jumps(const elf::Image& img, const std::vector<Function>& functions) {
    ZydisDecoder dec;
    ZydisDecoderInit(&dec, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    struct Decoded {
        uint64_t ip;
        ZydisDecodedInstruction i;
        ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
    };
    auto widest = [](ZydisRegister r) { return ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, r); };
    auto is_reg = [](const ZydisDecodedOperand& o) { return o.type == ZYDIS_OPERAND_TYPE_REGISTER; };

    IndirectJumps result;
    // ponytail: pattern window of the last 24 instructions in linear order; a bound check hoisted further away
    // falls back to reading entries while they point into the function.
    constexpr size_t kWindow = 24;
    std::vector<Decoded> hist;
    std::unordered_map<ZydisRegister, uint64_t> rip_lea;  // register -> target of its last `lea r,[rip+T]`
    for (auto& fn : functions) {
        auto body = img.at_vaddr(fn.start, fn.size);
        hist.clear();
        rip_lea.clear();
        for (size_t off = 0; off < body.size();) {
            Decoded d;
            d.ip = fn.start + off;
            if (!ZYAN_SUCCESS(ZydisDecoderDecodeFull(&dec, body.data() + off, body.size() - off, &d.i, d.ops))) {
                ++off;  // undecodable byte (padding/data inside a body): resync
                hist.clear();
                continue;
            }
            off += d.i.length;
            // Pattern checks read ops[0..1] directly; hidden/unused slots must not look like real operands.
            for (size_t k = d.i.operand_count_visible; k < ZYDIS_MAX_OPERAND_COUNT; ++k)
                d.ops[k].type = ZYDIS_OPERAND_TYPE_UNUSED;
            if (hist.size() == kWindow) hist.erase(hist.begin());
            hist.push_back(d);
            // Track the last definition of every written register (linear order) for hoisted table bases.
            for (uint8_t k = 0; k < d.i.operand_count_visible; ++k) {
                const auto& o = d.ops[k];
                if (o.type != ZYDIS_OPERAND_TYPE_REGISTER || !(o.actions & ZYDIS_OPERAND_ACTION_MASK_WRITE)) continue;
                ZyanU64 t;
                if (d.i.mnemonic == ZYDIS_MNEMONIC_LEA && d.ops[1].mem.base == ZYDIS_REGISTER_RIP &&
                    ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&d.i, &d.ops[1], d.ip, &t)))
                    rip_lea[widest(o.reg.value)] = t;
                else
                    rip_lea.erase(widest(o.reg.value));
            }
            if (d.i.mnemonic != ZYDIS_MNEMONIC_JMP || d.i.operand_count_visible < 1) continue;
            const auto& op = d.ops[0];
            if (op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) continue;
            ++result.total;
            if (op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
                (op.mem.base == ZYDIS_REGISTER_RIP ? result.rip_memory : result.memory_tail_calls)++;
                continue;
            }
            // jmp R: look back for add R,B / movsxd R,[B+I*4] / lea B,[rip+T] / cmp I,imm + ja|jae.
            const ZydisRegister r = widest(op.reg.value);
            ZydisRegister b = ZYDIS_REGISTER_NONE, idx = ZYDIS_REGISTER_NONE;
            uint64_t table = 0;
            std::vector<uint64_t> candidates;
            struct Cmp { ZydisRegister reg; uint64_t count; bool to_default; };
            std::vector<Cmp> cmps;
            std::vector<std::pair<ZydisRegister, ZydisRegister>> copies;
            std::vector<ZydisRegister> aliases;
            bool def_seen = false, loaded_from_memory = false;
            for (auto it = hist.rbegin() + 1; it != hist.rend(); ++it) {
                const auto& h = *it;
                const auto m = h.i.mnemonic;
                // Most recent definition of R: a load from memory means a function-pointer/vtable tail call.
                if (!def_seen && is_reg(h.ops[0]) && widest(h.ops[0].reg.value) == r && m != ZYDIS_MNEMONIC_CMP &&
                    m != ZYDIS_MNEMONIC_TEST && m != ZYDIS_MNEMONIC_PUSH) {
                    def_seen = true;
                    loaded_from_memory = m == ZYDIS_MNEMONIC_MOV && h.ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY;
                }
                if (b == ZYDIS_REGISTER_NONE && m == ZYDIS_MNEMONIC_ADD && is_reg(h.ops[0]) && is_reg(h.ops[1]) &&
                    widest(h.ops[0].reg.value) == r)
                    b = widest(h.ops[1].reg.value);
                else if (b != ZYDIS_REGISTER_NONE && !table && m == ZYDIS_MNEMONIC_MOV && is_reg(h.ops[0]) &&
                         is_reg(h.ops[1]) && widest(h.ops[0].reg.value) == b && idx != ZYDIS_REGISTER_NONE)
                    b = widest(h.ops[1].reg.value);  // table base copied: lea X,[rip+T]; mov B,X
                else if (b != ZYDIS_REGISTER_NONE && idx == ZYDIS_REGISTER_NONE && m == ZYDIS_MNEMONIC_MOVSXD &&
                         h.ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY && h.ops[1].mem.scale == 4 &&
                         widest(h.ops[1].mem.base) == b)
                    idx = widest(h.ops[1].mem.index);
                else if (b != ZYDIS_REGISTER_NONE && !table && m == ZYDIS_MNEMONIC_LEA && is_reg(h.ops[0]) &&
                         widest(h.ops[0].reg.value) == b && h.ops[1].mem.base == ZYDIS_REGISTER_RIP) {
                    ZyanU64 t;
                    if (ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&h.i, &h.ops[1], h.ip, &t))) table = t;
                }
                // Bound compares and register copies are collected over the whole window; which of them constrain the
                // table index (directly or through `mov idx, X`) is decided after the scan.
                if (m == ZYDIS_MNEMONIC_CMP && is_reg(h.ops[0]) && h.ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE && it != hist.rbegin()) {
                    // ja/jae jump to the default case; jbe/jb jump *into* the table code (the default is the fallthrough).
                    const auto& jc = *std::prev(it);
                    const auto nm = jc.i.mnemonic;
                    ZyanU64 jt_target = 0;
                    // jbe/jb only bound the table when they branch to the table code that precedes the jmp; a branch
                    // past the jmp is a lower-bound test whose target is the default case.
                    const bool into_table = (nm == ZYDIS_MNEMONIC_JBE || nm == ZYDIS_MNEMONIC_JB) &&
                                            ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&jc.i, &jc.ops[0], jc.ip, &jt_target)) && jt_target <= d.ip;
                    if (nm == ZYDIS_MNEMONIC_JNBE || (nm == ZYDIS_MNEMONIC_JBE && into_table)) cmps.push_back({widest(h.ops[0].reg.value), h.ops[1].imm.value.u + 1, nm == ZYDIS_MNEMONIC_JNBE});
                    else if (nm == ZYDIS_MNEMONIC_JNB || (nm == ZYDIS_MNEMONIC_JB && into_table)) cmps.push_back({widest(h.ops[0].reg.value), h.ops[1].imm.value.u, nm == ZYDIS_MNEMONIC_JNB});
                } else if (m == ZYDIS_MNEMONIC_MOV && is_reg(h.ops[0]) && is_reg(h.ops[1])) {
                    copies.push_back({widest(h.ops[0].reg.value), widest(h.ops[1].reg.value)});
                }
            }
            for (bool grew = idx != ZYDIS_REGISTER_NONE; grew;) {  // closure of registers equal to the index register
                grew = false;
                if (std::find(aliases.begin(), aliases.end(), idx) == aliases.end()) aliases.push_back(idx);
                for (const auto& [dst, src] : copies)
                    for (auto [a, b2] : {std::pair{dst, src}, std::pair{src, dst}})
                        if (std::find(aliases.begin(), aliases.end(), a) != aliases.end() && std::find(aliases.begin(), aliases.end(), b2) == aliases.end())
                            aliases.push_back(b2), grew = true;
            }
            for (const auto& cm : cmps)
                if (std::find(aliases.begin(), aliases.end(), cm.reg) != aliases.end()) candidates.push_back(cm.count);
            uint64_t direct_first = 0;  // nearest bound compare on the index register that jumps to the default
            for (const auto& cm : cmps)
                if (cm.reg == idx && cm.to_default) { direct_first = cm.count; break; }
            // Table base hoisted out of the window (e.g. kept in a callee-saved register across a loop).
            if (!table && b != ZYDIS_REGISTER_NONE && idx != ZYDIS_REGISTER_NONE)
                if (auto it = rip_lea.find(b); it != rip_lea.end()) table = it->second;
            JumpTable jt{d.ip, 0, fn.start, {}};
            if (b == ZYDIS_REGISTER_NONE || idx == ZYDIS_REGISTER_NONE || !table) {
                // jmp right after an epilogue (pop / add rsp / leave) leaves the frame: indirect tail call.
                const auto prev = hist.size() >= 2 ? hist[hist.size() - 2].i.mnemonic : ZYDIS_MNEMONIC_INVALID;
                const bool after_epilogue = prev == ZYDIS_MNEMONIC_POP || prev == ZYDIS_MNEMONIC_LEAVE ||
                                            (prev == ZYDIS_MNEMONIC_ADD && hist[hist.size() - 2].ops[0].type ==
                                                                               ZYDIS_OPERAND_TYPE_REGISTER &&
                                             hist[hist.size() - 2].ops[0].reg.value == ZYDIS_REGISTER_RSP);
                if (loaded_from_memory || after_epilogue) ++result.register_tail_calls;
                else result.unresolved_reg.push_back(std::move(jt));
                continue;
            }
            jt.table = table;
            auto read_entries = [&](uint64_t limit, bool stop_outside) {
                jt.targets.clear();
                for (uint64_t k = 0; k < limit; ++k) {
                    auto e = img.at_vaddr(table + 4 * k, 4);
                    if (e.size() != 4) return true;
                    int32_t rel;
                    std::memcpy(&rel, e.data(), 4);
                    uint64_t target = table + int64_t(rel);
                    if (target < fn.start || target >= fn.start + fn.size) {
                        if (stop_outside) return true;
                        return false;  // bound is wrong (cmp of another value)
                    }
                    jt.targets.push_back(target);
                }
                return true;
            };
            // Every case target must lie inside the function. A bound that yields outside targets came from an
            // unrelated cmp; then read entries until the first one that leaves the function.
            // A table that is too small aborts the game at run time, one that is too large only adds unused case labels,
            // so the result is the larger of (a) the best bound compare tied to the index (directly or via mov) and
            // (b) the nearest direct bound / reading entries until one leaves the function.
            std::sort(candidates.begin(), candidates.end(), std::greater<>());
            std::vector<uint64_t> best_cand, baseline;
            for (uint64_t cand : candidates)
                if (cand > 0 && cand <= 4096 && read_entries(cand, false)) { best_cand = jt.targets; break; }
            if (direct_first > 0 && direct_first <= 4096 && read_entries(direct_first, false)) baseline = jt.targets;
            else {
                if (direct_first) ++result.bound_rejected;
                read_entries(4096, true);
                baseline = jt.targets;
            }
            jt.targets = best_cand.size() > baseline.size() ? best_cand : baseline;
            result.tables.push_back(std::move(jt));
        }
    }
    // Linear decoding also walks switch tables that Clang placed inside function bodies; "jmp reg" decoded from
    // their int32 entries is not code. Drop unresolved sites that lie inside a resolved table.
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (auto& t : result.tables) ranges.emplace_back(t.table, t.table + 4 * t.targets.size());
    std::sort(ranges.begin(), ranges.end());
    std::erase_if(result.unresolved_reg, [&](const JumpTable& j) {
        auto it = std::upper_bound(ranges.begin(), ranges.end(), std::pair{j.jmp, UINT64_MAX});
        if (it == ranges.begin()) return false;
        --it;
        if (j.jmp >= it->second) return false;
        ++result.inside_table_data;
        return true;
    });
    return result;
}

} // namespace bb
