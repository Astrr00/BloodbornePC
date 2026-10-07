// SPDX-License-Identifier: GPL-3.0-or-later
#include "recomp/emitter.h"

#include <Zydis/Zydis.h>

#include "core/nid.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace bb {
namespace {

struct Insn {
    uint64_t ip;
    ZydisDecodedInstruction i;
    ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
    uint64_t next() const { return ip + i.length; }
};

std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "0x%" PRIx64, v);
    return b;
}

std::string fn_name(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "f_%" PRIx64, v);
    return b;
}

std::string label(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "L_%" PRIx64, v);
    return b;
}

const char* utype(unsigned bits) {
    switch (bits) {
    case 8: return "uint8_t";
    case 16: return "uint16_t";
    case 32: return "uint32_t";
    default: return "uint64_t";
    }
}

const char* stype(unsigned bits) {
    switch (bits) {
    case 8: return "int8_t";
    case 16: return "int16_t";
    case 32: return "int32_t";
    default: return "int64_t";
    }
}

// Condition code of jcc/setcc/cmovcc as a C++ expression over Context flags; nullptr if not a condition.
const char* condition(ZydisMnemonic m) {
    switch (m) {
    case ZYDIS_MNEMONIC_JO: case ZYDIS_MNEMONIC_SETO: case ZYDIS_MNEMONIC_CMOVO: return "c.of";
    case ZYDIS_MNEMONIC_JNO: case ZYDIS_MNEMONIC_SETNO: case ZYDIS_MNEMONIC_CMOVNO: return "!c.of";
    case ZYDIS_MNEMONIC_JB: case ZYDIS_MNEMONIC_SETB: case ZYDIS_MNEMONIC_CMOVB: return "c.cf";
    case ZYDIS_MNEMONIC_JNB: case ZYDIS_MNEMONIC_SETNB: case ZYDIS_MNEMONIC_CMOVNB: return "!c.cf";
    case ZYDIS_MNEMONIC_JZ: case ZYDIS_MNEMONIC_SETZ: case ZYDIS_MNEMONIC_CMOVZ: return "c.zf";
    case ZYDIS_MNEMONIC_JNZ: case ZYDIS_MNEMONIC_SETNZ: case ZYDIS_MNEMONIC_CMOVNZ: return "!c.zf";
    case ZYDIS_MNEMONIC_JBE: case ZYDIS_MNEMONIC_SETBE: case ZYDIS_MNEMONIC_CMOVBE: return "(c.cf || c.zf)";
    case ZYDIS_MNEMONIC_JNBE: case ZYDIS_MNEMONIC_SETNBE: case ZYDIS_MNEMONIC_CMOVNBE: return "(!c.cf && !c.zf)";
    case ZYDIS_MNEMONIC_JS: case ZYDIS_MNEMONIC_SETS: case ZYDIS_MNEMONIC_CMOVS: return "c.sf";
    case ZYDIS_MNEMONIC_JNS: case ZYDIS_MNEMONIC_SETNS: case ZYDIS_MNEMONIC_CMOVNS: return "!c.sf";
    case ZYDIS_MNEMONIC_JP: case ZYDIS_MNEMONIC_SETP: case ZYDIS_MNEMONIC_CMOVP: return "c.pf";
    case ZYDIS_MNEMONIC_JNP: case ZYDIS_MNEMONIC_SETNP: case ZYDIS_MNEMONIC_CMOVNP: return "!c.pf";
    case ZYDIS_MNEMONIC_JL: case ZYDIS_MNEMONIC_SETL: case ZYDIS_MNEMONIC_CMOVL: return "(c.sf != c.of)";
    case ZYDIS_MNEMONIC_JNL: case ZYDIS_MNEMONIC_SETNL: case ZYDIS_MNEMONIC_CMOVNL: return "(c.sf == c.of)";
    case ZYDIS_MNEMONIC_JLE: case ZYDIS_MNEMONIC_SETLE: case ZYDIS_MNEMONIC_CMOVLE: return "(c.zf || c.sf != c.of)";
    case ZYDIS_MNEMONIC_JNLE: case ZYDIS_MNEMONIC_SETNLE: case ZYDIS_MNEMONIC_CMOVNLE: return "(!c.zf && c.sf == c.of)";
    default: return nullptr;
    }
}

// String instruction: element bits (0 = not a string instruction) and kind m(ovs) s(tos) l(ods) c(mps) a(scas).
unsigned string_op(ZydisMnemonic m, char& kind) {
    switch (m) {
    case ZYDIS_MNEMONIC_MOVSB: kind = 'm'; return 8;
    case ZYDIS_MNEMONIC_MOVSW: kind = 'm'; return 16;
    case ZYDIS_MNEMONIC_MOVSD: kind = 'm'; return 32;
    case ZYDIS_MNEMONIC_MOVSQ: kind = 'm'; return 64;
    case ZYDIS_MNEMONIC_STOSB: kind = 's'; return 8;
    case ZYDIS_MNEMONIC_STOSW: kind = 's'; return 16;
    case ZYDIS_MNEMONIC_STOSD: kind = 's'; return 32;
    case ZYDIS_MNEMONIC_STOSQ: kind = 's'; return 64;
    case ZYDIS_MNEMONIC_LODSB: kind = 'l'; return 8;
    case ZYDIS_MNEMONIC_LODSW: kind = 'l'; return 16;
    case ZYDIS_MNEMONIC_LODSD: kind = 'l'; return 32;
    case ZYDIS_MNEMONIC_LODSQ: kind = 'l'; return 64;
    case ZYDIS_MNEMONIC_CMPSB: kind = 'c'; return 8;
    case ZYDIS_MNEMONIC_CMPSW: kind = 'c'; return 16;
    case ZYDIS_MNEMONIC_CMPSD: kind = 'c'; return 32;
    case ZYDIS_MNEMONIC_CMPSQ: kind = 'c'; return 64;
    case ZYDIS_MNEMONIC_SCASB: kind = 'a'; return 8;
    case ZYDIS_MNEMONIC_SCASW: kind = 'a'; return 16;
    case ZYDIS_MNEMONIC_SCASD: kind = 'a'; return 32;
    case ZYDIS_MNEMONIC_SCASQ: kind = 'a'; return 64;
    default: return 0;
    }
}

// General-purpose register access. Returns false for registers the emitter does not model (segment, x87, ...).
struct Gpr {
    int index;
    unsigned bits;
    bool high8;  // ah, ch, dh, bh
};

bool gpr(ZydisRegister r, Gpr& out) {
    if (r >= ZYDIS_REGISTER_AH && r <= ZYDIS_REGISTER_BH) {
        out = {int(r - ZYDIS_REGISTER_AH), 8, true};
        return true;
    }
    ZydisRegister wide = ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, r);
    if (wide < ZYDIS_REGISTER_RAX || wide > ZYDIS_REGISTER_R15) return false;
    out = {int(wide - ZYDIS_REGISTER_RAX), unsigned(ZydisRegisterGetWidth(ZYDIS_MACHINE_MODE_LONG_64, r)), false};
    return true;
}

// PLT stubs (vaddr, no bias) of setjmp/_setjmp/sigsetjmp. A call to one of them is emitted inline in the caller so the
// caller's frame can be resumed by a later longjmp (see FunctionEmitter::emit / bb::rt::LongJmp).
std::unordered_set<uint64_t> find_setjmp_plts(const elf::Image& img) {
    std::unordered_set<std::string> nids{generate_nid("setjmp"), generate_nid("_setjmp"), generate_nid("sigsetjmp")};
    std::unordered_map<uint64_t, uint32_t> slot_symbol;
    for (auto& r : img.relocations)
        if (r.type == elf::R_X86_64_JUMP_SLOT && r.symbol && r.symbol < img.symbols.size() && nids.count(img.symbols[r.symbol].nid)) slot_symbol.emplace(r.offset, r.symbol);
    std::unordered_set<uint64_t> plts;
    if (slot_symbol.empty()) return plts;
    for (auto& s : img.segments) {
        if (s.type != elf::PT_LOAD || !(s.flags & 1)) continue;
        for (uint64_t a = s.vaddr; a + 12 <= s.vaddr + s.filesz; ++a) {
            auto b = img.at_vaddr(a, 12);
            if (b.size() != 12 || b[0] != 0xFF || b[1] != 0x25 || b[6] != 0x68 || b[11] != 0xE9) continue;
            int32_t disp;
            std::memcpy(&disp, b.data() + 2, 4);
            if (slot_symbol.count(a + 6 + int64_t(disp))) plts.insert(a);
        }
    }
    return plts;
}

// Byte-wise table CRC64 (`crc = crc >> 8 ^ table[uint8_t(crc) ^ *p++]`, state stored after every byte), matched on its exact
// bytes; offset 11 holds the table's rip-relative displacement. Translated instruction by instruction its state goes
// through Context memory on every byte; bb::rt::crc64_bytes is the same loop natively. Returns the table vaddr or 0.
uint64_t crc64_idiom(const elf::Image& img, uint64_t a) {
    static constexpr uint8_t k[] = {0x48, 0x85, 0xd2, 0x74, 0x28, 0x48, 0x8b, 0x07, 0x4c, 0x8d, 0x05, 0, 0, 0, 0, 0x90,
                                    0x44, 0x0f, 0xb6, 0x0e, 0x0f, 0xb6, 0xc8, 0x4c, 0x31, 0xc9, 0x48, 0xc1, 0xe8, 0x08, 0x49, 0x33,
                                    0x04, 0xc8, 0x48, 0x89, 0x07, 0x48, 0xff, 0xc6, 0x48, 0xff, 0xca, 0x75, 0xe3, 0xc3};
    auto b = img.at_vaddr(a, sizeof k);
    if (b.size() != sizeof k) return 0;
    for (size_t i = 0; i < sizeof k; ++i)
        if ((i < 11 || i > 14) && b[i] != k[i]) return 0;
    int32_t disp;
    std::memcpy(&disp, b.data() + 11, 4);
    return a + 15 + int64_t(disp);
}

class FunctionEmitter {
public:
    FunctionEmitter(const elf::Image& img, ZydisDecoder& dec, const Function& fn,
                    const std::unordered_map<uint64_t, const JumpTable*>& tables,
                    const std::unordered_set<uint64_t>& known, const std::unordered_set<uint64_t>& setjmp_plts, EmitStats& stats, uint64_t bias)
        : img_(img), dec_(dec), fn_(fn), tables_(tables), known_(known), sjplt_(setjmp_plts), stats_(stats), bias_(bias) {}

    std::string emit() {
        out_ += "void " + fn_name(fn_.start) + "(bb::rt::Context& c) {\n    BB_TRACE_ENTER(" + hex(fn_.start + bias_) + "ull);\n";
        if (const uint64_t table = crc64_idiom(img_, fn_.start)) {
            out_ += "    bb::rt::crc64_bytes(c, " + hex(table + bias_) + "ull);\n    c.r[4] += 8; return;\n}\n\n";
            return std::move(out_);
        }
        discover();
        const size_t body_start = out_.size();
        uint64_t expect = 0;
        bool reachable_fallthrough = false;
        size_t run = 0, longest = 0;  // instructions without a label or branch in between
        for (auto& [ip, in] : insns_) {
            if (reachable_fallthrough && ip != expect) fallthrough(expect);
            if (labels_.count(ip)) {
                out_ += label(ip) + ":;\n";
                run = 0;
            }
            reachable_fallthrough = translate(in);
            expect = in.next();
            const auto cat = in.i.meta.category;
            run = cat == ZYDIS_CATEGORY_COND_BR || cat == ZYDIS_CATEGORY_UNCOND_BR ? 0 : run + 1;
            longest = std::max(longest, run);
        }
        if (reachable_fallthrough) fallthrough(expect);
        out_ += "}\n\n";
        if (!sj_sites_.empty()) {
            // A longjmp unwinds the host stack by throwing LongJmp; the function that called setjmp catches it and continues
            // right after its setjmp call (rax and the callee-saved registers were restored by longjmp).
            std::string body = out_.substr(body_start);
            body.resize(body.size() - 3);  // the closing "}\n\n"
            out_.resize(body_start);
            out_ += "    uint64_t resume_ = 0;\nagain_:\n    try {\n        if (resume_) switch (resume_) {\n";
            for (size_t k = 0; k < sj_sites_.size(); ++k) out_ += "        case " + hex(sj_sites_[k]) + "ull: goto sj_" + std::to_string(k) + ";\n";
            out_ += "        default: break;\n        }\n" + body + "    } catch (bb::rt::LongJmp& e_) {\n        switch (e_.ret_addr) {\n";
            for (size_t k = 0; k < sj_sites_.size(); ++k) out_ += "        case " + hex(sj_sites_[k]) + "ull:\n";
            out_ += "            resume_ = e_.ret_addr; goto again_;\n        default: throw;\n        }\n    }\n}\n\n";
        }
        // MSVC optimizes straight-line blocks of thousands of instructions superlinearly (with /d2OptimizeHugeFunctions: 20-30 min
        // for one function, hours per shard). Only cold one-off code has such blocks (7 functions; no profile samples): leave those unoptimized.
        if (longest > 2000) out_ = "#ifdef _MSC_VER\n#pragma optimize(\"\", off)\n#endif\n" + out_ + "#ifdef _MSC_VER\n#pragma optimize(\"\", on)\n#endif\n";
        return std::move(out_);
    }

private:
    bool inside(uint64_t a) const { return a >= fn_.start && a < fn_.start + fn_.size; }

    bool decode(uint64_t a, Insn& in) {
        auto bytes = img_.at_vaddr(a, std::min<uint64_t>(ZYDIS_MAX_INSTRUCTION_LENGTH, fn_.start + fn_.size - a));
        if (bytes.empty() || !ZYAN_SUCCESS(ZydisDecoderDecodeFull(&dec_, bytes.data(), bytes.size(), &in.i, in.ops)))
            return false;
        in.ip = a;
        for (size_t k = in.i.operand_count_visible; k < ZYDIS_MAX_OPERAND_COUNT; ++k)
            in.ops[k].type = ZYDIS_OPERAND_TYPE_UNUSED;
        return true;
    }

    static bool rel_target(const Insn& in, uint64_t& t) {
        ZyanU64 v;
        if (in.i.operand_count_visible < 1 || in.ops[0].type != ZYDIS_OPERAND_TYPE_IMMEDIATE ||
            !in.ops[0].imm.is_relative || !ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&in.i, &in.ops[0], in.ip, &v)))
            return false;
        t = v;
        return true;
    }

    // Recursive descent over [start, start+size) from the entry and every switch-case target of this function.
    void discover() {
        std::vector<uint64_t> todo{fn_.start};
        for (auto& [jmp, jt] : tables_)
            if (jt->function == fn_.start)
                for (uint64_t t : jt->targets) todo.push_back(t), labels_.insert(t);
        while (!todo.empty()) {
            uint64_t a = todo.back();
            todo.pop_back();
            while (inside(a) && !insns_.count(a)) {
                Insn in;
                if (!decode(a, in)) break;
                insns_.emplace(a, in);
                a = in.next();
                const auto m = in.i.mnemonic;
                const auto cat = in.i.meta.category;
                uint64_t t;
                if (cat == ZYDIS_CATEGORY_RET || m == ZYDIS_MNEMONIC_INT3 || m == ZYDIS_MNEMONIC_UD2 ||
                    m == ZYDIS_MNEMONIC_HLT)
                    break;
                if (cat == ZYDIS_CATEGORY_COND_BR && rel_target(in, t) && inside(t)) todo.push_back(t), labels_.insert(t);
                if (cat == ZYDIS_CATEGORY_UNCOND_BR) {
                    if (rel_target(in, t) && inside(t)) todo.push_back(t), labels_.insert(t);
                    break;
                }
            }
        }
        // Fallthrough into an instruction that is not the next emitted one needs a label to jump to.
        uint64_t prev_next = 0;
        bool prev_falls = false;
        for (auto& [ip, in] : insns_) {
            if (prev_falls && ip != prev_next && insns_.count(prev_next)) labels_.insert(prev_next);
            const auto cat = in.i.meta.category;
            prev_falls = !(cat == ZYDIS_CATEGORY_RET || cat == ZYDIS_CATEGORY_UNCOND_BR ||
                           in.i.mnemonic == ZYDIS_MNEMONIC_INT3 || in.i.mnemonic == ZYDIS_MNEMONIC_UD2 ||
                           in.i.mnemonic == ZYDIS_MNEMONIC_HLT);
            prev_next = in.next();
        }
    }

    void fallthrough(uint64_t a) {
        if (insns_.count(a)) out_ += "    goto " + label(a) + ";\n";
        else transfer(a);
    }

    // Control leaves this function for `a` without a call (tail jump / fallthrough into the next function).
    void transfer(uint64_t a) {
        leaf_ = false;
        if (known_.count(a)) {
            refs_.insert(a);
            out_ += "    " + fn_name(a) + "(c); return;\n";
        }
        else out_ += "    bb::rt::call_indirect(c, " + hex(a + bias_) + "); return;\n";
    }

    std::string reg_read(ZydisRegister r) {
        Gpr g;
        if (!gpr(r, g)) return {};
        std::string reg = "c.r[" + std::to_string(g.index) + "]";
        if (g.high8) return "uint8_t(" + reg + " >> 8)";
        if (g.bits == 64) return reg;
        return std::string(utype(g.bits)) + "(" + reg + ")";
    }

    bool reg_write(ZydisRegister r, const std::string& v) {
        Gpr g;
        if (!gpr(r, g)) return false;
        std::string reg = "c.r[" + std::to_string(g.index) + "]";
        if (g.high8)
            out_ += "    " + reg + " = (" + reg + " & ~0xFF00ull) | (uint64_t(uint8_t(" + v + ")) << 8);\n";
        else if (g.bits == 64) out_ += "    " + reg + " = uint64_t(" + v + ");\n";
        else if (g.bits == 32) out_ += "    " + reg + " = uint32_t(" + v + ");\n";  // zero-extends
        else
            out_ += "    " + reg + " = (" + reg + " & ~" + (g.bits == 16 ? "0xFFFFull" : "0xFFull") + ") | " +
                    utype(g.bits) + "(" + v + ");\n";
        return true;
    }

    std::string address(const Insn& in, const ZydisDecodedOperand& op) {
        const auto& m = op.mem;
        if (m.segment != ZYDIS_REGISTER_FS && m.segment != ZYDIS_REGISTER_DS && m.segment != ZYDIS_REGISTER_SS &&
            m.segment != ZYDIS_REGISTER_ES && m.segment != ZYDIS_REGISTER_CS && m.segment != ZYDIS_REGISTER_NONE)
            return {};
        std::string e;
        if (m.base == ZYDIS_REGISTER_RIP) {
            e = hex(in.next() + uint64_t(m.disp.value) + bias_);
        } else {
            if (m.base != ZYDIS_REGISTER_NONE) {
                std::string b = reg_read(m.base);
                if (b.empty()) return {};
                e = "uint64_t(" + b + ")";
            }
            if (m.index != ZYDIS_REGISTER_NONE) {
                std::string ix = reg_read(m.index);
                if (ix.empty()) return {};
                e += (e.empty() ? "" : " + ") + std::string("uint64_t(") + ix + ") * " + std::to_string(m.scale);
            }
            if (m.disp.has_displacement && m.disp.value)
                e += (e.empty() ? "" : " + ") + hex(uint64_t(m.disp.value)) + "ull";
            if (e.empty()) e = "0ull";
            if (in.i.address_width == 32) e = "uint64_t(uint32_t(" + e + "))";
        }
        if (m.segment == ZYDIS_REGISTER_FS) {
            leaf_ = false;
            e = "(c.fs_base + " + e + ")";
        }
        return e;
    }

    // Value of an operand as an expression of type uint<bits>_t (immediates are sign-extended then truncated).
    std::string read(const Insn& in, const ZydisDecodedOperand& op, unsigned bits) {
        switch (op.type) {
        case ZYDIS_OPERAND_TYPE_REGISTER: return reg_read(op.reg.value);
        case ZYDIS_OPERAND_TYPE_IMMEDIATE: {
            // Sign-extend to 64 bits, then narrow here so the literal already fits the destination type.
            uint64_t v = op.imm.is_signed ? uint64_t(op.imm.value.s) : op.imm.value.u;
            if (bits < 64) v &= (uint64_t(1) << bits) - 1;
            return std::string(utype(bits)) + "(" + hex(v) + "ull)";
        }
        case ZYDIS_OPERAND_TYPE_MEMORY: {
            std::string a = address(in, op);
            if (a.empty()) return {};
            return std::string("bb::rt::ld<") + utype(op.size) + ">(c, " + a + ")";
        }
        default: return {};
        }
    }

    bool write(const Insn& in, const ZydisDecodedOperand& op, const std::string& v) {
        if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) return reg_write(op.reg.value, v);
        if (op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
            std::string a = address(in, op);
            if (a.empty()) return false;
            out_ += std::string("    bb::rt::st<") + utype(op.size) + ">(c, " + a + ", " + utype(op.size) + "(" + v +
                    "));\n";
            return true;
        }
        return false;
    }

    void push(const std::string& v) {
        out_ += "    c.r[4] -= 8; bb::rt::st<uint64_t>(c, c.r[4], uint64_t(" + v + "));\n";
    }

    // Emits the translation; returns whether control can fall through to the next instruction.
    bool translate(const Insn& in) {
        ++stats_.instructions;
        const size_t mark = out_.size();
        int r = translate_known(in);
        if (r >= 0) return r != 0;
        out_.resize(mark);  // drop any partial output
        ++stats_.unsupported;
        leaf_ = false;
        const char* name = ZydisMnemonicGetString(in.i.mnemonic);
        ++stats_.unsupported_by_mnemonic[name ? name : "?"];
        out_ += "    bb::rt::unsupported(c, " + hex(in.ip) + ", \"" + (name ? name : "?") + "\");\n";
        return false;
    }

    static int xmm(ZydisRegister r) {
        return r >= ZYDIS_REGISTER_XMM0 && r <= ZYDIS_REGISTER_XMM15 ? int(r - ZYDIS_REGISTER_XMM0) : -1;
    }

    static bool isymm(const ZydisDecodedOperand& op) {
        return op.type == ZYDIS_OPERAND_TYPE_REGISTER && op.reg.value >= ZYDIS_REGISTER_YMM0 &&
               op.reg.value <= ZYDIS_REGISTER_YMM15;
    }

    // xmm register or 32/64/128-bit memory operand as an __m128i expression (scalar loads zero the other lanes).
    std::string vread(const Insn& in, const ZydisDecodedOperand& op) {
        if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            if (op.reg.value >= ZYDIS_REGISTER_MM0 && op.reg.value <= ZYDIS_REGISTER_MM7)
                return "bb::rt::mr(c, " + std::to_string(int(op.reg.value - ZYDIS_REGISTER_MM0)) + ")";
            int x = xmm(op.reg.value);
            return x < 0 ? std::string{} : "bb::rt::xr(c, " + std::to_string(x) + ")";
        }
        if (op.type != ZYDIS_OPERAND_TYPE_MEMORY) return {};
        std::string a = address(in, op);
        if (a.empty()) return {};
        switch (op.size) {
        case 128: return "bb::rt::ldx(c, " + a + ")";
        case 64: return "_mm_cvtsi64_si128(int64_t(bb::rt::ld<uint64_t>(c, " + a + ")))";
        case 32: return "_mm_cvtsi32_si128(int(bb::rt::ld<uint32_t>(c, " + a + ")))";
        case 16: return "_mm_cvtsi32_si128(int(bb::rt::ld<uint16_t>(c, " + a + ")))";
        default: return {};
        }
    }

    bool vwrite(const Insn& in, const ZydisDecodedOperand& op, const std::string& v) {
        if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            if (op.reg.value >= ZYDIS_REGISTER_MM0 && op.reg.value <= ZYDIS_REGISTER_MM7) {
                out_ += "    bb::rt::mw(c, " + std::to_string(int(op.reg.value - ZYDIS_REGISTER_MM0)) + ", " + v + ");\n";
                return true;
            }
            int x = xmm(op.reg.value);
            if (x < 0) return false;
            out_ += "    bb::rt::xw(c, " + std::to_string(x) + ", " + v + ");\n";
            return true;
        }
        if (op.type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
        std::string a = address(in, op);
        if (a.empty()) return false;
        switch (op.size) {
        case 128: out_ += "    bb::rt::stx(c, " + a + ", " + v + ");\n"; return true;
        case 64: out_ += "    bb::rt::st<uint64_t>(c, " + a + ", uint64_t(_mm_cvtsi128_si64(" + v + ")));\n"; return true;
        case 32: out_ += "    bb::rt::st<uint32_t>(c, " + a + ", uint32_t(_mm_cvtsi128_si32(" + v + ")));\n"; return true;
        default: return false;
        }
    }

    // ymm register or 256-bit memory operand as an __m256i expression.
    std::string yread(const Insn& in, const ZydisDecodedOperand& op) {
        if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            auto r = op.reg.value;
            if (r < ZYDIS_REGISTER_YMM0 || r > ZYDIS_REGISTER_YMM15) return {};
            return "bb::rt::yr(c, " + std::to_string(int(r - ZYDIS_REGISTER_YMM0)) + ")";
        }
        if (op.type != ZYDIS_OPERAND_TYPE_MEMORY || op.size != 256) return {};
        std::string a = address(in, op);
        return a.empty() ? std::string{} : "bb::rt::ldy(c, " + a + ")";
    }

    bool ywrite(const Insn& in, const ZydisDecodedOperand& op, const std::string& v) {
        if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            auto r = op.reg.value;
            if (r < ZYDIS_REGISTER_YMM0 || r > ZYDIS_REGISTER_YMM15) return false;
            out_ += "    bb::rt::yw(c, " + std::to_string(int(r - ZYDIS_REGISTER_YMM0)) + ", " + v + ");\n";
            return true;
        }
        if (op.type != ZYDIS_OPERAND_TYPE_MEMORY || op.size != 256) return false;
        std::string a = address(in, op);
        if (a.empty()) return false;
        out_ += "    bb::rt::sty(c, " + a + ", " + v + ");\n";
        return true;
    }

    // AVX/SSE subset (128-bit). Returns -2 if `in` is not handled here (not a vector instruction).
    int translate_vec(const Insn& in) {
        const auto& o = in.ops;
        const auto m = in.i.mnemonic;
        const int n = in.i.operand_count_visible;
        // String movsd/cmpsd share their mnemonic with the SSE scalar forms; those have a register operand.
        if ((m == ZYDIS_MNEMONIC_MOVSD || m == ZYDIS_MNEMONIC_CMPSD) &&
            !(o[0].type == ZYDIS_OPERAND_TYPE_REGISTER ||
              (o[0].type == ZYDIS_OPERAND_TYPE_MEMORY && o[1].type == ZYDIS_OPERAND_TYPE_REGISTER)))
            return -2;
        auto imm = [&](int k) { return std::to_string(o[k].imm.value.u & 0xFF); };
        // Y: the destination is a ymm register (VEX.256, float-only on AVX1). rd/fn/put follow the width.
        const bool Y = isymm(o[0]);
        auto rd = [&](int k) { return Y ? yread(in, o[k]) : vread(in, o[k]); };
        auto fn = [&](const char* f) { return Y ? "_mm256_" + std::string(f + 4) : std::string(f); };  // "_mm_x" -> "_mm256_x"
        auto put = [&](const std::string& v) { return (Y ? ywrite(in, o[0], v) : vwrite(in, o[0], v)) ? 1 : -1; };
        // dst = f(src1, src2): VEX 3-operand form, or legacy 2-operand form where src1 == dst.
        auto src2 = [&](std::string& a, std::string& b) {
            a = rd(n == 2 ? 0 : 1);
            b = rd(n == 2 ? 1 : 2);
            return !a.empty() && !b.empty();
        };
        auto fbin = [&](const char* f) -> int {  // packed/scalar single
            std::string a, b;
            if (!src2(a, b)) return -1;
            return put("bb::rt::pi(" + fn(f) + "(bb::rt::ps(" + a + "), bb::rt::ps(" + b + ")))");
        };
        auto dbin = [&](const char* f) -> int {  // packed/scalar double
            std::string a, b;
            if (!src2(a, b)) return -1;
            return put("bb::rt::pi(" + fn(f) + "(bb::rt::pd(" + a + "), bb::rt::pd(" + b + ")))");
        };
        auto ibin = [&](const char* f) -> int {  // integer; ymm forms (AVX2 encodings) run as two 128-bit halves
            std::string a, b;
            if (!src2(a, b)) return -1;
            if (Y)
                return put("bb::rt::ysplit(" + a + ", " + b + ", [](__m128i x, __m128i y) { return " + f + "(x, y); })");
            return put(std::string(f) + "(" + a + ", " + b + ")");
        };
        auto fimm = [&](const char* f, const char* cast = "ps") -> int {  // dst, s1, s2, imm8
            if (n != 4) return -1;
            std::string a = rd(1), b = rd(2);
            if (a.empty() || b.empty()) return -1;
            const std::string C = std::string("bb::rt::") + cast;
            return put("bb::rt::pi(" + fn(f) + "(" + C + "(" + a + "), " + C + "(" + b + "), " + imm(3) + "))");
        };
        auto blendv = [&](const char* f, const char* cast) -> int {  // dst, s1, s2, mask
            if (n != 4) return -1;
            std::string a = rd(1), b = rd(2), k = rd(3);
            if (a.empty() || b.empty() || k.empty()) return -1;
            const std::string C = std::string("bb::rt::") + cast;
            return put("bb::rt::pi(" + fn(f) + "(" + C + "(" + a + "), " + C + "(" + b + "), " + C + "(" + k + ")))");
        };
        auto iimm = [&](const char* f) -> int {  // dst, src, imm8 (128-bit integer)
            if (n != 3 || Y) return -1;
            std::string a = vread(in, o[1]);
            if (a.empty()) return -1;
            return put(std::string(f) + "(" + a + ", " + imm(2) + ")");
        };
        // Shift by immediate, or by the low qword of an xmm/m128 count.
        auto ishift = [&](const char* imm_f, const char* cnt_f) -> int {
            if (n != 3) return -1;
            if (o[2].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) return iimm(imm_f);
            std::string a = vread(in, o[1]), b = vread(in, o[2]);
            if (a.empty() || b.empty()) return -1;
            return put(std::string(cnt_f) + "(" + a + ", " + b + ")");
        };
        // dst = f(cast_in(src)) of o[1] (ymm/m256 if srcY, else xmm/mem); cast_out wraps the result to __m128i/__m256i.
        auto conv = [&](const std::string& f, const char* in_cast, const char* out_cast, bool srcY) -> int {
            if (n != 2) return -1;
            std::string a = srcY ? yread(in, o[1]) : vread(in, o[1]);
            if (a.empty()) return -1;
            if (*in_cast) a = std::string("bb::rt::") + in_cast + "(" + a + ")";
            std::string e = f + "(" + a + ")";
            if (*out_cast) e = std::string("bb::rt::") + out_cast + "(" + e + ")";
            return put(e);
        };
        auto funary = [&](const char* f, const char* cast = "ps") -> int {  // dst, src (packed)
            if (n != 2) return -1;
            std::string a = rd(1);
            if (a.empty()) return -1;
            return put("bb::rt::pi(" + fn(f) + "(bb::rt::" + cast + "(" + a + ")))");
        };
        auto fscalar_unary = [&](const char* f) -> int {  // VEX dst, s1, s2: low = f(s2), rest from s1
            if (n != 3) return -1;
            std::string a = vread(in, o[1]), b = vread(in, o[2]);
            if (a.empty() || b.empty()) return -1;
            return put("bb::rt::pi(_mm_move_ss(bb::rt::ps(" + a + "), " + f + "(bb::rt::ps(" + b + "))))");
        };

        switch (m) {
        case ZYDIS_MNEMONIC_VMOVAPS: case ZYDIS_MNEMONIC_VMOVUPS: case ZYDIS_MNEMONIC_VMOVDQA:
        case ZYDIS_MNEMONIC_VMOVDQU: case ZYDIS_MNEMONIC_VMOVAPD: case ZYDIS_MNEMONIC_VMOVUPD:
        case ZYDIS_MNEMONIC_VMOVNTPS: case ZYDIS_MNEMONIC_VMOVNTPD: case ZYDIS_MNEMONIC_VMOVNTDQ:
        case ZYDIS_MNEMONIC_VMOVNTDQA: case ZYDIS_MNEMONIC_MOVNTPS: case ZYDIS_MNEMONIC_MOVNTDQ:
        case ZYDIS_MNEMONIC_MOVAPS: case ZYDIS_MNEMONIC_MOVUPS: case ZYDIS_MNEMONIC_MOVDQA: case ZYDIS_MNEMONIC_MOVDQU: {
            if (o[0].size == 256 && o[1].size == 256) {
                std::string v = yread(in, o[1]);
                return v.empty() || !ywrite(in, o[0], v) ? -1 : 1;
            }
            if (o[0].size != 128 || o[1].size != 128) return -1;
            std::string v = vread(in, o[1]);
            return v.empty() ? -1 : put(v);
        }
        case ZYDIS_MNEMONIC_VMOVSS: case ZYDIS_MNEMONIC_MOVSS:
        case ZYDIS_MNEMONIC_VMOVSD: case ZYDIS_MNEMONIC_MOVSD: {
            const bool s = m == ZYDIS_MNEMONIC_VMOVSS || m == ZYDIS_MNEMONIC_MOVSS;
            if (n == 3) {
                std::string a = vread(in, o[1]), b = vread(in, o[2]);
                if (a.empty() || b.empty()) return -1;
                return put(s ? "bb::rt::pi(_mm_move_ss(bb::rt::ps(" + a + "), bb::rt::ps(" + b + ")))"
                             : "bb::rt::pi(_mm_move_sd(bb::rt::pd(" + a + "), bb::rt::pd(" + b + ")))");
            }
            std::string v = vread(in, o[1]);  // load zero-extends; store writes the low lane
            if (v.empty()) return -1;
            if (o[0].type == ZYDIS_OPERAND_TYPE_REGISTER && o[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
                std::string d = vread(in, o[0]);  // legacy reg-reg: low lane replaced, the rest kept
                return d.empty() ? -1
                       : put(s ? "bb::rt::pi(_mm_move_ss(bb::rt::ps(" + d + "), bb::rt::ps(" + v + ")))"
                               : "bb::rt::pi(_mm_move_sd(bb::rt::pd(" + d + "), bb::rt::pd(" + v + ")))");
            }
            return put(v);
        }
        case ZYDIS_MNEMONIC_VMOVD:
        case ZYDIS_MNEMONIC_VMOVQ: {
            const bool q = m == ZYDIS_MNEMONIC_VMOVQ;
            if (xmm(o[0].reg.value) >= 0 && o[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
                if (o[1].type == ZYDIS_OPERAND_TYPE_REGISTER && xmm(o[1].reg.value) >= 0)
                    return q ? put("_mm_move_epi64(" + vread(in, o[1]) + ")") : -1;
                if (o[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
                    std::string g = reg_read(o[1].reg.value);
                    if (g.empty()) return -1;
                    return put(q ? "_mm_cvtsi64_si128(int64_t(" + g + "))" : "_mm_cvtsi32_si128(int(" + g + "))");
                }
                std::string v = vread(in, o[1]);
                return v.empty() ? -1 : put(v);
            }
            std::string v = vread(in, o[1]);
            if (v.empty()) return -1;
            if (o[0].type == ZYDIS_OPERAND_TYPE_MEMORY) return put(v);
            return reg_write(o[0].reg.value, q ? "uint64_t(_mm_cvtsi128_si64(" + v + "))"
                                               : "uint32_t(_mm_cvtsi128_si32(" + v + "))")
                       ? 1
                       : -1;
        }
        case ZYDIS_MNEMONIC_VADDPS: return fbin("_mm_add_ps");
        case ZYDIS_MNEMONIC_VSUBPS: return fbin("_mm_sub_ps");
        case ZYDIS_MNEMONIC_VMULPS: return fbin("_mm_mul_ps");
        case ZYDIS_MNEMONIC_VDIVPS: return fbin("_mm_div_ps");
        case ZYDIS_MNEMONIC_VMINPS: return fbin("_mm_min_ps");
        case ZYDIS_MNEMONIC_VMAXPS: return fbin("_mm_max_ps");
        case ZYDIS_MNEMONIC_VANDPS: return fbin("_mm_and_ps");
        case ZYDIS_MNEMONIC_VORPS: return fbin("_mm_or_ps");
        case ZYDIS_MNEMONIC_VXORPS: return fbin("_mm_xor_ps");
        case ZYDIS_MNEMONIC_VANDNPS: return fbin("_mm_andnot_ps");
        case ZYDIS_MNEMONIC_VHADDPS: return fbin("_mm_hadd_ps");
        case ZYDIS_MNEMONIC_VUNPCKLPS: return fbin("_mm_unpacklo_ps");
        case ZYDIS_MNEMONIC_VUNPCKHPS: return fbin("_mm_unpackhi_ps");
        case ZYDIS_MNEMONIC_VMOVHLPS: return fbin("_mm_movehl_ps");
        case ZYDIS_MNEMONIC_VMOVLHPS: return fbin("_mm_movelh_ps");
        case ZYDIS_MNEMONIC_VADDSS: return fbin("_mm_add_ss");
        case ZYDIS_MNEMONIC_VSUBSS: return fbin("_mm_sub_ss");
        case ZYDIS_MNEMONIC_VMULSS: return fbin("_mm_mul_ss");
        case ZYDIS_MNEMONIC_VDIVSS: return fbin("_mm_div_ss");
        case ZYDIS_MNEMONIC_VMINSS: return fbin("_mm_min_ss");
        case ZYDIS_MNEMONIC_VMAXSS: return fbin("_mm_max_ss");
        case ZYDIS_MNEMONIC_VADDSD: return dbin("_mm_add_sd");
        case ZYDIS_MNEMONIC_VSUBSD: return dbin("_mm_sub_sd");
        case ZYDIS_MNEMONIC_VMULSD: return dbin("_mm_mul_sd");
        case ZYDIS_MNEMONIC_VDIVSD: return dbin("_mm_div_sd");
        case ZYDIS_MNEMONIC_VXORPD: return dbin("_mm_xor_pd");
        case ZYDIS_MNEMONIC_VPXOR: case ZYDIS_MNEMONIC_PXOR: return ibin("_mm_xor_si128");
        case ZYDIS_MNEMONIC_VPAND: return ibin("_mm_and_si128");
        case ZYDIS_MNEMONIC_VPOR: return ibin("_mm_or_si128");
        case ZYDIS_MNEMONIC_VPANDN: return ibin("_mm_andnot_si128");
        case ZYDIS_MNEMONIC_VPADDD: return ibin("_mm_add_epi32");
        case ZYDIS_MNEMONIC_VPSUBD: return ibin("_mm_sub_epi32");
        case ZYDIS_MNEMONIC_VPADDQ: return ibin("_mm_add_epi64");
        case ZYDIS_MNEMONIC_VPCMPEQD: return ibin("_mm_cmpeq_epi32");
        case ZYDIS_MNEMONIC_VPUNPCKLDQ: return ibin("_mm_unpacklo_epi32");
        case ZYDIS_MNEMONIC_VPUNPCKLQDQ: return ibin("_mm_unpacklo_epi64");
        case ZYDIS_MNEMONIC_VPSHUFD: return iimm("_mm_shuffle_epi32");
        case ZYDIS_MNEMONIC_VPSRLQ: return ishift("_mm_srli_epi64", "_mm_srl_epi64");
        case ZYDIS_MNEMONIC_VPSLLQ: return ishift("_mm_slli_epi64", "_mm_sll_epi64");
        case ZYDIS_MNEMONIC_VPSRLD: return ishift("_mm_srli_epi32", "_mm_srl_epi32");
        case ZYDIS_MNEMONIC_VPSLLD: return ishift("_mm_slli_epi32", "_mm_sll_epi32");
        case ZYDIS_MNEMONIC_VPSRAD: return ishift("_mm_srai_epi32", "_mm_sra_epi32");
        case ZYDIS_MNEMONIC_VPSRLW: return ishift("_mm_srli_epi16", "_mm_srl_epi16");
        case ZYDIS_MNEMONIC_VPSLLW: return ishift("_mm_slli_epi16", "_mm_sll_epi16");
        case ZYDIS_MNEMONIC_VPSRAW: return ishift("_mm_srai_epi16", "_mm_sra_epi16");
        case ZYDIS_MNEMONIC_VSHUFPS: return fimm("_mm_shuffle_ps");
        case ZYDIS_MNEMONIC_VBLENDPS: return fimm("_mm_blend_ps");
        case ZYDIS_MNEMONIC_VDPPS: return fimm("_mm_dp_ps");
        case ZYDIS_MNEMONIC_VINSERTPS: return fimm("_mm_insert_ps");
        case ZYDIS_MNEMONIC_VCMPPS: return fimm("_mm_cmp_ps");
        case ZYDIS_MNEMONIC_VCMPSS: return fimm("_mm_cmp_ss");
        case ZYDIS_MNEMONIC_VBLENDVPS: return blendv("_mm_blendv_ps", "ps");
        case ZYDIS_MNEMONIC_VSQRTPS: return funary("_mm_sqrt_ps");
        case ZYDIS_MNEMONIC_VRSQRTPS: return funary("_mm_rsqrt_ps");  // approximation: may differ AMD vs Intel
        case ZYDIS_MNEMONIC_VRCPPS: return funary("_mm_rcp_ps");
        case ZYDIS_MNEMONIC_VSQRTSS: return fscalar_unary("_mm_sqrt_ss");
        case ZYDIS_MNEMONIC_VRSQRTSS: return fscalar_unary("_mm_rsqrt_ss");
        case ZYDIS_MNEMONIC_VRCPSS: return fscalar_unary("_mm_rcp_ss");
        case ZYDIS_MNEMONIC_VCVTDQ2PS: return conv(fn("_mm_cvtepi32_ps"), "", "pi", Y);
        case ZYDIS_MNEMONIC_VCVTTPS2DQ: return conv(fn("_mm_cvttps_epi32"), "ps", "", Y);
        case ZYDIS_MNEMONIC_VCVTPS2DQ: return conv(fn("_mm_cvtps_epi32"), "ps", "", Y);
        case ZYDIS_MNEMONIC_VCVTPS2PD: return conv(Y ? "_mm256_cvtps_pd" : "_mm_cvtps_pd", "ps", "pi", false);
        case ZYDIS_MNEMONIC_VCVTDQ2PD: return conv(Y ? "_mm256_cvtepi32_pd" : "_mm_cvtepi32_pd", "", "pi", false);
        case ZYDIS_MNEMONIC_VCVTPD2PS: {  // xmm <- xmm/m128 or ymm/m256
            const bool sy = o[1].size == 256;
            return conv(sy ? "_mm256_cvtpd_ps" : "_mm_cvtpd_ps", "pd", "pi", sy);
        }
        case ZYDIS_MNEMONIC_VCVTTPD2DQ: {
            const bool sy = o[1].size == 256;
            return conv(sy ? "_mm256_cvttpd_epi32" : "_mm_cvttpd_epi32", "pd", "", sy);
        }
        case ZYDIS_MNEMONIC_VCVTSS2SD: {
            std::string a, b;
            return n == 3 && src2(a, b) ? put("bb::rt::pi(_mm_cvtss_sd(bb::rt::pd(" + a + "), bb::rt::ps(" + b + ")))")
                                        : -1;
        }
        case ZYDIS_MNEMONIC_VCVTSD2SS: {
            std::string a, b;
            return n == 3 && src2(a, b) ? put("bb::rt::pi(_mm_cvtsd_ss(bb::rt::ps(" + a + "), bb::rt::pd(" + b + ")))")
                                        : -1;
        }
        case ZYDIS_MNEMONIC_VCVTSI2SS: {
            if (n != 3) return -1;
            std::string a = vread(in, o[1]), g = read(in, o[2], o[2].size);
            if (a.empty() || g.empty()) return -1;
            return put(o[2].size == 64 ? "bb::rt::pi(_mm_cvtsi64_ss(bb::rt::ps(" + a + "), int64_t(" + g + ")))"
                                       : "bb::rt::pi(_mm_cvtsi32_ss(bb::rt::ps(" + a + "), int32_t(" + g + ")))");
        }
        case ZYDIS_MNEMONIC_VCVTTSS2SI: {
            std::string a = vread(in, o[1]);
            if (a.empty()) return -1;
            return reg_write(o[0].reg.value, o[0].size == 64 ? "_mm_cvttss_si64(bb::rt::ps(" + a + "))"
                                                             : "uint32_t(_mm_cvttss_si32(bb::rt::ps(" + a + ")))")
                       ? 1
                       : -1;
        }
        case ZYDIS_MNEMONIC_VUCOMISS: case ZYDIS_MNEMONIC_VCOMISS: case ZYDIS_MNEMONIC_UCOMISS: case ZYDIS_MNEMONIC_COMISS:
        case ZYDIS_MNEMONIC_VUCOMISD: case ZYDIS_MNEMONIC_VCOMISD: {
            std::string a = vread(in, o[0]), b = vread(in, o[1]);
            if (a.empty() || b.empty()) return -1;
            const bool d = m == ZYDIS_MNEMONIC_VUCOMISD || m == ZYDIS_MNEMONIC_VCOMISD;
            out_ += std::string("    bb::rt::") + (d ? "f_ucomisd" : "f_ucomiss") + "(c, " + a + ", " + b + ");\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_VMOVMSKPS: case ZYDIS_MNEMONIC_VMOVMSKPD: {
            const bool sy = isymm(o[1]), d = m == ZYDIS_MNEMONIC_VMOVMSKPD;
            std::string a = sy ? yread(in, o[1]) : vread(in, o[1]);
            if (a.empty()) return -1;
            return reg_write(o[0].reg.value, std::string("uint32_t(") + (sy ? "_mm256_" : "_mm_") + "movemask_" +
                                                 (d ? "pd(bb::rt::pd(" : "ps(bb::rt::ps(") + a + ")))")
                       ? 1
                       : -1;
        }
        case ZYDIS_MNEMONIC_VEXTRACTPS: {
            std::string a = vread(in, o[1]);
            if (a.empty() || n != 3) return -1;
            std::string v = "uint32_t(_mm_extract_ps(bb::rt::ps(" + a + "), " + imm(2) + "))";
            return write(in, o[0], v) ? 1 : -1;
        }
        case ZYDIS_MNEMONIC_VLDDQU: {
            if (o[0].size != 128) return -1;
            std::string v = vread(in, o[1]);
            return v.empty() ? -1 : put(v);
        }
        case ZYDIS_MNEMONIC_VPSHUFB: return ibin("_mm_shuffle_epi8");
        case ZYDIS_MNEMONIC_VPCMPGTD: return ibin("_mm_cmpgt_epi32");
        case ZYDIS_MNEMONIC_VPCMPEQB: return ibin("_mm_cmpeq_epi8");
        case ZYDIS_MNEMONIC_VPMAXSD: return ibin("_mm_max_epi32");
        case ZYDIS_MNEMONIC_VPMINSD: return ibin("_mm_min_epi32");
        case ZYDIS_MNEMONIC_VPUNPCKLWD: return ibin("_mm_unpacklo_epi16");
        case ZYDIS_MNEMONIC_VPUNPCKHWD: return ibin("_mm_unpackhi_epi16");
        case ZYDIS_MNEMONIC_VPUNPCKHDQ: return ibin("_mm_unpackhi_epi32");
        case ZYDIS_MNEMONIC_VPUNPCKHQDQ: return ibin("_mm_unpackhi_epi64");
        case ZYDIS_MNEMONIC_VPSLLDQ: return o[2].type == ZYDIS_OPERAND_TYPE_IMMEDIATE ? iimm("_mm_slli_si128") : -1;
        case ZYDIS_MNEMONIC_VPSRLDQ: return o[2].type == ZYDIS_OPERAND_TYPE_IMMEDIATE ? iimm("_mm_srli_si128") : -1;
        case ZYDIS_MNEMONIC_VPALIGNR: {  // dst = (s1:s2) >> imm*8
            if (n != 4) return -1;
            std::string a = vread(in, o[1]), b = vread(in, o[2]);
            if (a.empty() || b.empty()) return -1;
            return put("_mm_alignr_epi8(" + a + ", " + b + ", " + imm(3) + ")");
        }
        case ZYDIS_MNEMONIC_VPBLENDVB: {
            if (n != 4) return -1;
            std::string a = vread(in, o[1]), b = vread(in, o[2]), k = vread(in, o[3]);
            if (a.empty() || b.empty() || k.empty()) return -1;
            return put("_mm_blendv_epi8(" + a + ", " + b + ", " + k + ")");
        }
        case ZYDIS_MNEMONIC_VPINSRD:
        case ZYDIS_MNEMONIC_VPINSRQ: {  // dst, s1, r/m32|64, imm
            if (n != 4) return -1;
            const bool q = m == ZYDIS_MNEMONIC_VPINSRQ;
            std::string a = vread(in, o[1]), g = read(in, o[2], q ? 64 : 32);
            if (a.empty() || g.empty()) return -1;
            return put(q ? "_mm_insert_epi64(" + a + ", int64_t(" + g + "), " + imm(3) + ")"
                         : "_mm_insert_epi32(" + a + ", int(" + g + "), " + imm(3) + ")");
        }
        case ZYDIS_MNEMONIC_VPEXTRD:
        case ZYDIS_MNEMONIC_VPEXTRQ: {  // r/m32|64, x, imm
            if (n != 3) return -1;
            const bool q = m == ZYDIS_MNEMONIC_VPEXTRQ;
            std::string a = vread(in, o[1]);
            if (a.empty()) return -1;
            return write(in, o[0], q ? "uint64_t(_mm_extract_epi64(" + a + ", " + imm(2) + "))"
                                     : "uint32_t(_mm_extract_epi32(" + a + ", " + imm(2) + "))")
                       ? 1
                       : -1;
        }
        case ZYDIS_MNEMONIC_VPTEST: {  // ZF = (a & b) == 0, CF = (~a & b) == 0, others cleared
            if (o[0].size != 128) return -1;
            std::string a = vread(in, o[0]), b = vread(in, o[1]);
            if (a.empty() || b.empty()) return -1;
            out_ += "    c.zf = _mm_testz_si128(" + a + ", " + b + ") != 0; c.cf = _mm_testc_si128(" + a + ", " + b +
                    ") != 0; c.of = c.sf = c.pf = false;\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_VMOVLPS: case ZYDIS_MNEMONIC_VMOVLPD:
        case ZYDIS_MNEMONIC_VMOVHPS: case ZYDIS_MNEMONIC_VMOVHPD: {
            const bool high = m == ZYDIS_MNEMONIC_VMOVHPS || m == ZYDIS_MNEMONIC_VMOVHPD;
            if (n == 3) {  // dst, s1, m64: replace low/high qword
                std::string a = vread(in, o[1]), p = address(in, o[2]);
                if (a.empty() || p.empty()) return -1;
                return put(std::string("bb::rt::pi(") + (high ? "_mm_loadh_pd" : "_mm_loadl_pd") + "(bb::rt::pd(" +
                           a + "), reinterpret_cast<const double*>(bb::rt::host(c, " + p + "))))");
            }
            if (n != 2 || o[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return -1;
            std::string a = vread(in, o[1]), p = address(in, o[0]);
            if (a.empty() || p.empty()) return -1;
            out_ += "    bb::rt::st<uint64_t>(c, " + p + ", uint64_t(_mm_extract_epi64(" + a + ", " + (high ? "1" : "0") +
                    ")));\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_VADDPD: return dbin("_mm_add_pd");
        case ZYDIS_MNEMONIC_VSUBPD: return dbin("_mm_sub_pd");
        case ZYDIS_MNEMONIC_VMULPD: return dbin("_mm_mul_pd");
        case ZYDIS_MNEMONIC_VDIVPD: return dbin("_mm_div_pd");
        case ZYDIS_MNEMONIC_VUNPCKLPD: return dbin("_mm_unpacklo_pd");
        case ZYDIS_MNEMONIC_VUNPCKHPD: return dbin("_mm_unpackhi_pd");
        case ZYDIS_MNEMONIC_VSHUFPD: return fimm("_mm_shuffle_pd", "pd");
        case ZYDIS_MNEMONIC_VPERMILPD: {
            if (n != 3 || o[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE || o[0].size != 128) return -1;
            std::string a = vread(in, o[1]);
            return a.empty() ? -1 : put("bb::rt::pi(_mm_permute_pd(bb::rt::pd(" + a + "), " + imm(2) + "))");
        }
        case ZYDIS_MNEMONIC_VCVTSI2SD: {
            if (n != 3) return -1;
            std::string a = vread(in, o[1]), g = read(in, o[2], o[2].size);
            if (a.empty() || g.empty()) return -1;
            return put(o[2].size == 64 ? "bb::rt::pi(_mm_cvtsi64_sd(bb::rt::pd(" + a + "), int64_t(" + g + ")))"
                                       : "bb::rt::pi(_mm_cvtsi32_sd(bb::rt::pd(" + a + "), int32_t(" + g + ")))");
        }
        case ZYDIS_MNEMONIC_VCVTTSD2SI: {
            std::string a = vread(in, o[1]);
            if (a.empty()) return -1;
            return reg_write(o[0].reg.value, o[0].size == 64 ? "_mm_cvttsd_si64(bb::rt::pd(" + a + "))"
                                                             : "uint32_t(_mm_cvttsd_si32(bb::rt::pd(" + a + ")))")
                       ? 1
                       : -1;
        }
        case ZYDIS_MNEMONIC_VINSERTF128: {  // ymm, ymm, xmm/m128, imm
            if (n != 4) return -1;
            std::string a = yread(in, o[1]), b = vread(in, o[2]);
            if (a.empty() || b.empty()) return -1;
            return ywrite(in, o[0], "_mm256_insertf128_si256(" + a + ", " + b + ", " + imm(3) + ")") ? 1 : -1;
        }
        case ZYDIS_MNEMONIC_VEXTRACTF128: {  // xmm/m128, ymm, imm
            if (n != 3) return -1;
            std::string a = yread(in, o[1]);
            return a.empty() ? -1 : put("_mm256_extractf128_si256(" + a + ", " + imm(2) + ")");
        }
        case ZYDIS_MNEMONIC_VBROADCASTSS: {
            if (o[1].type != ZYDIS_OPERAND_TYPE_MEMORY) return -1;
            std::string a = address(in, o[1]);
            return a.empty() ? -1
                             : put("bb::rt::pi(" + fn("_mm_broadcast_ss") + "(reinterpret_cast<const float*>(bb::rt::host(c, " +
                                   a + "))))");
        }
        case ZYDIS_MNEMONIC_VANDPD: return dbin("_mm_and_pd");
        case ZYDIS_MNEMONIC_VORPD: return dbin("_mm_or_pd");
        case ZYDIS_MNEMONIC_VANDNPD: return dbin("_mm_andnot_pd");
        case ZYDIS_MNEMONIC_VMAXPD: return dbin("_mm_max_pd");
        case ZYDIS_MNEMONIC_VMINPD: return dbin("_mm_min_pd");
        case ZYDIS_MNEMONIC_VMAXSD: return dbin("_mm_max_sd");
        case ZYDIS_MNEMONIC_VMINSD: return dbin("_mm_min_sd");
        case ZYDIS_MNEMONIC_VSQRTSD: return dbin("_mm_sqrt_sd");  // low = sqrt(src2.low), rest from src1
        case ZYDIS_MNEMONIC_VHADDPD: return dbin("_mm_hadd_pd");
        case ZYDIS_MNEMONIC_VADDSUBPD: return dbin("_mm_addsub_pd");
        case ZYDIS_MNEMONIC_VSQRTPD: return funary("_mm_sqrt_pd", "pd");
        case ZYDIS_MNEMONIC_VDPPD: return fimm("_mm_dp_pd", "pd");
        case ZYDIS_MNEMONIC_VCMPPD: return fimm("_mm_cmp_pd", "pd");
        case ZYDIS_MNEMONIC_VCMPSD: return fimm("_mm_cmp_sd", "pd");
        case ZYDIS_MNEMONIC_VBLENDPD: return fimm("_mm_blend_pd", "pd");
        case ZYDIS_MNEMONIC_VBLENDVPD: return blendv("_mm_blendv_pd", "pd");
        case ZYDIS_MNEMONIC_VMOVDDUP: return conv(fn("_mm_movedup_pd"), "pd", "pi", Y);
        case ZYDIS_MNEMONIC_VMOVSHDUP: return conv(fn("_mm_movehdup_ps"), "ps", "pi", Y);
        case ZYDIS_MNEMONIC_VMOVSLDUP: return conv(fn("_mm_moveldup_ps"), "ps", "pi", Y);
        case ZYDIS_MNEMONIC_VROUNDPS:
        case ZYDIS_MNEMONIC_VROUNDPD: {  // dst, src, imm4
            if (n != 3) return -1;
            const bool d = m == ZYDIS_MNEMONIC_VROUNDPD;
            std::string a = rd(1);
            if (a.empty()) return -1;
            const std::string C = d ? "bb::rt::pd" : "bb::rt::ps";
            return put("bb::rt::pi(" + fn(d ? "_mm_round_pd" : "_mm_round_ps") + "(" + C + "(" + a + "), " +
                       std::to_string(o[2].imm.value.u & 0xF) + "))");
        }
        case ZYDIS_MNEMONIC_VROUNDSS:
        case ZYDIS_MNEMONIC_VROUNDSD: {  // dst, s1, s2, imm4: low = round(s2.low), rest from s1
            if (n != 4) return -1;
            const bool d = m == ZYDIS_MNEMONIC_VROUNDSD;
            std::string a = vread(in, o[1]), b = vread(in, o[2]);
            if (a.empty() || b.empty()) return -1;
            const std::string C = d ? "bb::rt::pd" : "bb::rt::ps";
            return put("bb::rt::pi(" + std::string(d ? "_mm_round_sd" : "_mm_round_ss") + "(" + C + "(" + a + "), " + C +
                       "(" + b + "), " + std::to_string(o[3].imm.value.u & 0xF) + "))");
        }
        case ZYDIS_MNEMONIC_VCVTSD2SI:
        case ZYDIS_MNEMONIC_VCVTSS2SI: {
            const bool d = m == ZYDIS_MNEMONIC_VCVTSD2SI;
            std::string a = vread(in, o[1]);
            if (a.empty()) return -1;
            const std::string f = std::string("_mm_cvt") + (d ? "sd" : "ss") + "_si" + (o[0].size == 64 ? "64" : "32");
            const std::string v = f + "(bb::rt::" + (d ? "pd(" : "ps(") + a + "))";
            return reg_write(o[0].reg.value, o[0].size == 64 ? v : "uint32_t(" + v + ")") ? 1 : -1;
        }
        case ZYDIS_MNEMONIC_VFNMADD132SD: {  // dst = -(dst * src3) + src2 (single rounding), upper lane kept
            if (n != 3) return -1;
            std::string a = vread(in, o[0]), b = vread(in, o[1]), k = vread(in, o[2]);
            if (a.empty() || b.empty() || k.empty()) return -1;
            return put("bb::rt::fnmadd_sd(" + a + ", " + a + ", " + k + ", " + b + ")");
        }
        case ZYDIS_MNEMONIC_VZEROUPPER: out_ += "    bb::rt::vzeroupper(c);\n"; return 1;
        case ZYDIS_MNEMONIC_VMASKMOVDQU:
        case ZYDIS_MNEMONIC_MASKMOVDQU: {
            std::string a = vread(in, o[0]), b = vread(in, o[1]);
            if (a.empty() || b.empty()) return -1;
            out_ += "    bb::rt::maskmovdqu(c, " + a + ", " + b + ");\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_VSTMXCSR: case ZYDIS_MNEMONIC_STMXCSR:
            return write(in, o[0], "0x1F80u") ? 1 : -1;  // power-on MXCSR
        case ZYDIS_MNEMONIC_VLDMXCSR: case ZYDIS_MNEMONIC_LDMXCSR:
            return 1;  // ponytail: guest rounding-mode/FTZ changes are not applied to the host MXCSR
        case ZYDIS_MNEMONIC_VPSUBQ: return ibin("_mm_sub_epi64");
        case ZYDIS_MNEMONIC_VPSUBB: return ibin("_mm_sub_epi8");
        case ZYDIS_MNEMONIC_VPSUBW: return ibin("_mm_sub_epi16");
        case ZYDIS_MNEMONIC_VPADDB: return ibin("_mm_add_epi8");
        case ZYDIS_MNEMONIC_VPADDW: case ZYDIS_MNEMONIC_PADDW: return ibin("_mm_add_epi16");
        case ZYDIS_MNEMONIC_VPADDUSW: return ibin("_mm_adds_epu16");
        case ZYDIS_MNEMONIC_VPSUBSB: return ibin("_mm_subs_epi8");
        case ZYDIS_MNEMONIC_VPMULLD: return ibin("_mm_mullo_epi32");
        case ZYDIS_MNEMONIC_VPMULUDQ: return ibin("_mm_mul_epu32");
        case ZYDIS_MNEMONIC_VPCMPEQQ: return ibin("_mm_cmpeq_epi64");
        case ZYDIS_MNEMONIC_VPCMPGTQ: return ibin("_mm_cmpgt_epi64");
        case ZYDIS_MNEMONIC_VPCMPEQW: return ibin("_mm_cmpeq_epi16");
        case ZYDIS_MNEMONIC_VPCMPGTW: return ibin("_mm_cmpgt_epi16");
        case ZYDIS_MNEMONIC_PCMPEQD: return ibin("_mm_cmpeq_epi32");
        case ZYDIS_MNEMONIC_POR: return ibin("_mm_or_si128");
        case ZYDIS_MNEMONIC_VPUNPCKLBW: return ibin("_mm_unpacklo_epi8");
        case ZYDIS_MNEMONIC_VPUNPCKHBW: return ibin("_mm_unpackhi_epi8");
        case ZYDIS_MNEMONIC_VPACKSSDW: return ibin("_mm_packs_epi32");
        case ZYDIS_MNEMONIC_VPACKUSWB: return ibin("_mm_packus_epi16");
        case ZYDIS_MNEMONIC_VPSADBW: return ibin("_mm_sad_epu8");
        case ZYDIS_MNEMONIC_VPMINUD: return ibin("_mm_min_epu32");
        case ZYDIS_MNEMONIC_VPMAXUD: return ibin("_mm_max_epu32");
        case ZYDIS_MNEMONIC_VPMINSW: return ibin("_mm_min_epi16");
        case ZYDIS_MNEMONIC_VPMAXSW: return ibin("_mm_max_epi16");
        case ZYDIS_MNEMONIC_VPHADDD: return ibin("_mm_hadd_epi32");
        case ZYDIS_MNEMONIC_VPSIGND: return ibin("_mm_sign_epi32");
        case ZYDIS_MNEMONIC_VPAVGB: return ibin("_mm_avg_epu8");
        case ZYDIS_MNEMONIC_VPBLENDW: {
            if (n != 4) return -1;
            std::string a = vread(in, o[1]), b = vread(in, o[2]);
            if (a.empty() || b.empty()) return -1;
            return put("_mm_blend_epi16(" + a + ", " + b + ", " + imm(3) + ")");
        }
        case ZYDIS_MNEMONIC_VPABSD: return conv("_mm_abs_epi32", "", "", false);
        case ZYDIS_MNEMONIC_VPSHUFLW: return iimm("_mm_shufflelo_epi16");
        case ZYDIS_MNEMONIC_VPSHUFHW: return iimm("_mm_shufflehi_epi16");
        case ZYDIS_MNEMONIC_VPMOVSXBW: return conv("_mm_cvtepi8_epi16", "", "", false);
        case ZYDIS_MNEMONIC_VPMOVSXBD: return conv("_mm_cvtepi8_epi32", "", "", false);
        case ZYDIS_MNEMONIC_VPMOVSXBQ: return conv("_mm_cvtepi8_epi64", "", "", false);
        case ZYDIS_MNEMONIC_VPMOVSXWD: return conv("_mm_cvtepi16_epi32", "", "", false);
        case ZYDIS_MNEMONIC_VPMOVSXWQ: return conv("_mm_cvtepi16_epi64", "", "", false);
        case ZYDIS_MNEMONIC_VPMOVSXDQ: return conv("_mm_cvtepi32_epi64", "", "", false);
        case ZYDIS_MNEMONIC_VPMOVZXBW: return conv("_mm_cvtepu8_epi16", "", "", false);
        case ZYDIS_MNEMONIC_VPMOVZXBD: return conv("_mm_cvtepu8_epi32", "", "", false);
        case ZYDIS_MNEMONIC_VPMOVZXBQ: return conv("_mm_cvtepu8_epi64", "", "", false);
        case ZYDIS_MNEMONIC_VPMOVZXWD: return conv("_mm_cvtepu16_epi32", "", "", false);
        case ZYDIS_MNEMONIC_VPMOVZXWQ: return conv("_mm_cvtepu16_epi64", "", "", false);
        case ZYDIS_MNEMONIC_VPMOVZXDQ: return conv("_mm_cvtepu32_epi64", "", "", false);
        case ZYDIS_MNEMONIC_VPEXTRB:
        case ZYDIS_MNEMONIC_VPEXTRW:
        case ZYDIS_MNEMONIC_PEXTRW: {  // r32/m | xmm, imm
            if (n != 3) return -1;
            const bool b = m == ZYDIS_MNEMONIC_VPEXTRB;
            std::string a = vread(in, o[1]);
            if (a.empty()) return -1;
            return write(in, o[0], std::string("uint32_t(_mm_extract_epi") + (b ? "8(" : "16(") + a + ", " +
                                       std::to_string(o[2].imm.value.u & (b ? 15 : 7)) + "))")
                       ? 1
                       : -1;
        }
        case ZYDIS_MNEMONIC_VPINSRB:
        case ZYDIS_MNEMONIC_VPINSRW:
        case ZYDIS_MNEMONIC_PINSRW: {  // [dst,] src1, r32/m, imm
            const int k = n - 3;
            if (k < 0 || k > 1) return -1;
            const bool b = m == ZYDIS_MNEMONIC_VPINSRB;
            std::string a = vread(in, o[k]), g = read(in, o[k + 1], o[k + 1].size);
            if (a.empty() || g.empty()) return -1;
            return put(std::string("_mm_insert_epi") + (b ? "8(" : "16(") + a + ", int(" + g + "), " +
                       std::to_string(o[k + 2].imm.value.u & (b ? 15 : 7)) + ")");
        }
        default: return -2;
        }
    }

    // One-operand mul/imul/div/idiv: implicit AX (8-bit) or rDX:rAX operands.
    int wide_muldiv(const Insn& in, ZydisMnemonic m) {
        const auto& op = in.ops[0];
        const unsigned w = op.size;
        const std::string T = utype(w);
        std::string s = read(in, op, w);
        if (s.empty()) return -1;
        if (w == 8) {
            if (m == ZYDIS_MNEMONIC_MUL)
                out_ += "    { uint16_t r = uint16_t(uint16_t(uint8_t(c.r[0])) * uint16_t(" + s +
                        ")); c.cf = c.of = (r >> 8) != 0;\n";
            else if (m == ZYDIS_MNEMONIC_IMUL)
                out_ += "    { int16_t r = int16_t(int16_t(int8_t(uint8_t(c.r[0]))) * int16_t(int8_t(" + s +
                        "))); c.cf = c.of = r != int8_t(r);\n";
            if (m == ZYDIS_MNEMONIC_MUL || m == ZYDIS_MNEMONIC_IMUL) {
                if (!reg_write(ZYDIS_REGISTER_AX, "uint16_t(r)")) return -1;
                out_ += "    }\n";
                return 1;
            }
            out_ += std::string("    { uint8_t q, rem; bb::rt::") + (m == ZYDIS_MNEMONIC_DIV ? "f_div" : "f_idiv") +
                    "<uint8_t>(c, uint8_t(c.r[0] >> 8), uint8_t(c.r[0]), " + s + ", q, rem);\n";
            if (!reg_write(ZYDIS_REGISTER_AL, "q") || !reg_write(ZYDIS_REGISTER_AH, "rem")) return -1;
            out_ += "    }\n";
            return 1;
        }
        const ZydisRegister acc = w == 16 ? ZYDIS_REGISTER_AX : w == 32 ? ZYDIS_REGISTER_EAX : ZYDIS_REGISTER_RAX;
        const ZydisRegister dx = w == 16 ? ZYDIS_REGISTER_DX : w == 32 ? ZYDIS_REGISTER_EDX : ZYDIS_REGISTER_RDX;
        if (m == ZYDIS_MNEMONIC_MUL || m == ZYDIS_MNEMONIC_IMUL) {
            out_ += "    { " + T + " hi; " + T + " lo = bb::rt::" + (m == ZYDIS_MNEMONIC_MUL ? "f_mul_wide" : "f_imul_wide") +
                    "<" + T + ">(c, " + reg_read(acc) + ", " + s + ", hi);\n";
            if (!reg_write(acc, "lo") || !reg_write(dx, "hi")) return -1;
        } else {
            out_ += "    { " + T + " q, rem; bb::rt::" + (m == ZYDIS_MNEMONIC_DIV ? "f_div" : "f_idiv") + "<" + T +
                    ">(c, " + reg_read(dx) + ", " + reg_read(acc) + ", " + s + ", q, rem);\n";
            if (!reg_write(acc, "q") || !reg_write(dx, "rem")) return -1;
        }
        out_ += "    }\n";
        return 1;
    }

    // bt/bts/btr/btc. Register/immediate bit offsets are taken modulo the width; a register offset with a memory
    // operand addresses a bit string (signed element index).
    // A lock prefix on a bts/btr/btc memory form becomes a compare-exchange loop (bit spinlocks in the engine's allocator).
    int bit_test(const Insn& in, ZydisMnemonic m, unsigned w) {
        const auto& o = in.ops;
        const std::string T = utype(w);
        std::string off = read(in, o[1], w);
        if (off.empty()) return -1;
        const std::string mask = std::to_string(w - 1);
        const char* modify = m == ZYDIS_MNEMONIC_BTS   ? " v |= " 
                             : m == ZYDIS_MNEMONIC_BTR ? " v &= ~"
                             : m == ZYDIS_MNEMONIC_BTC ? " v ^= "
                                                       : nullptr;
        const std::string bit = T + "(" + T + "(1) << b)";
        if (modify && (in.i.attributes & ZYDIS_ATTRIB_HAS_LOCK) && o[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            std::string a = address(in, o[0]);
            if (a.empty()) return -1;
            const std::string op = m == ZYDIS_MNEMONIC_BTS ? "old | " + bit : m == ZYDIS_MNEMONIC_BTR ? "old & " + T + "(~" + bit + ")" : "old ^ " + bit;
            std::string ba = a, bsel = "unsigned b = unsigned(" + off + ") & " + mask + ";";
            if (o[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
                const int log2w = w == 16 ? 4 : w == 32 ? 5 : 6;
                ba = a + " + uint64_t((int64_t(" + stype(w) + "(" + off + ")) >> " + std::to_string(log2w) + ") * " + std::to_string(w / 8) + ")";
            }
            out_ += "    { uint64_t ba = " + ba + "; " + bsel + " auto ref = bb::rt::atom<" + T + ">(c, ba); " + T + " old = ref.load(), nw;\n"
                    "      do { nw = " + T + "(" + op + "); } while (!ref.compare_exchange_weak(old, nw)); c.cf = (old >> b) & 1; }\n";
            return 1;
        }
        if (o[0].type == ZYDIS_OPERAND_TYPE_MEMORY && o[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            std::string a = address(in, o[0]);
            if (a.empty()) return -1;
            const int log2w = w == 16 ? 4 : w == 32 ? 5 : 6;
            out_ += "    { uint64_t ba = " + a + " + uint64_t((int64_t(" + stype(w) + "(" + off + ")) >> " +
                    std::to_string(log2w) + ") * " + std::to_string(w / 8) + "); " + T + " v = bb::rt::ld<" + T +
                    ">(c, ba); unsigned b = unsigned(" + off + ") & " + mask + "; c.cf = (v >> b) & 1;";
            if (modify) out_ += std::string(modify) + bit + "; bb::rt::st<" + T + ">(c, ba, v);";
            out_ += " }\n";
            return 1;
        }
        std::string v = read(in, o[0], w);
        if (v.empty()) return -1;
        out_ += "    { " + T + " v = " + v + "; unsigned b = unsigned(" + off + ") & " + mask + "; c.cf = (v >> b) & 1;";
        if (!modify) {
            out_ += " }\n";
            return 1;
        }
        out_ += std::string(modify) + bit + ";\n";
        if (!write(in, o[0], "v")) return -1;
        out_ += "    }\n";
        return 1;
    }

    static int st_index(const ZydisDecodedOperand& op) {
        return op.type == ZYDIS_OPERAND_TYPE_REGISTER && op.reg.value >= ZYDIS_REGISTER_ST0 &&
                       op.reg.value <= ZYDIS_REGISTER_ST7
                   ? int(op.reg.value - ZYDIS_REGISTER_ST0)
                   : -1;
    }
    static std::string sti(int i) { return "bb::rt::sti(c, " + std::to_string(i) + ")"; }

    // x87 subset on host doubles (see guest.h). Returns -2 if `in` is not an x87 instruction.
    int translate_x87(const Insn& in) {
        const auto& o = in.ops;
        const auto m = in.i.mnemonic;
        const int n = in.i.operand_count_visible;
        // Memory operand as a double expression: float/double, or int16/32/64 for the FI* forms.
        auto memval = [&](const ZydisDecodedOperand& op, bool isint) -> std::string {
            if (op.type != ZYDIS_OPERAND_TYPE_MEMORY) return {};
            std::string a = address(in, op);
            const char* t = isint ? (op.size == 16 ? "int16_t" : op.size == 32 ? "int32_t" : op.size == 64 ? "int64_t" : nullptr)
                                  : (op.size == 32 ? "float" : op.size == 64 ? "double" : nullptr);
            return a.empty() || !t ? std::string{} : std::string("bb::rt::ldf<") + t + ">(c, " + a + ")";
        };
        // dst = dst op src (rev: src op dst); register forms name dst/src explicitly, memory forms use st0 as dst.
        auto arith = [&](char opc, bool rev, bool isint, bool pop) -> int {
            std::string dst, src;
            if (n == 2 && st_index(o[0]) >= 0 && st_index(o[1]) >= 0) dst = sti(st_index(o[0])), src = sti(st_index(o[1]));
            else if (n == 1) dst = sti(0), src = memval(o[0], isint);
            if (dst.empty() || src.empty()) return -1;
            out_ += "    { double a = " + dst + ", b = " + src + "; " + dst + " = " + (rev ? "b " : "a ") + opc +
                    (rev ? " a" : " b") + "; }\n";
            if (pop) out_ += "    bb::rt::fpop(c);\n";
            return 1;
        };
        // Compare st0 with st(i) (default st1) or a memory operand; `pops` pops afterwards.
        auto compare = [&](bool isint, int pops) -> int {
            std::string src = n == 0 ? sti(1) : st_index(o[n - 1]) >= 0 ? sti(st_index(o[n - 1])) : memval(o[0], isint);
            if (src.empty()) return -1;
            out_ += "    bb::rt::fcmp(c, " + sti(0) + ", " + src + ");\n";
            for (int i = 0; i < pops; ++i) out_ += "    bb::rt::fpop(c);\n";
            return 1;
        };
        auto comi = [&](bool pop) -> int {
            if (n != 2 || st_index(o[0]) < 0 || st_index(o[1]) < 0) return -1;
            out_ += "    bb::rt::fcomi(c, " + sti(st_index(o[0])) + ", " + sti(st_index(o[1])) + ");\n";
            if (pop) out_ += "    bb::rt::fpop(c);\n";
            return 1;
        };
        auto cmov = [&](const char* cond) -> int {
            if (n != 2 || st_index(o[0]) < 0 || st_index(o[1]) < 0) return -1;
            out_ += std::string("    if (") + cond + ") " + sti(st_index(o[0])) + " = " + sti(st_index(o[1])) + ";\n";
            return 1;
        };
        // Store st0 as an integer (m16/32/64): `trunc` for fisttp, else per FCW rounding; `pop` pops afterwards.
        auto istore = [&](bool trunc, bool pop) -> int {
            std::string a = o[0].type == ZYDIS_OPERAND_TYPE_MEMORY ? address(in, o[0]) : std::string{};
            const char* t = o[0].size == 16 ? "int16_t" : o[0].size == 32 ? "int32_t" : o[0].size == 64 ? "int64_t" : nullptr;
            if (a.empty() || !t) return -1;
            out_ += std::string("    bb::rt::st<") + t + ">(c, " + a + ", bb::rt::fto_int<" + t + ">(" +
                    (trunc ? "std::trunc(" + sti(0) + ")" : "bb::rt::fround(c, " + sti(0) + ")") + "));\n";
            if (pop) out_ += "    bb::rt::fpop(c);\n";
            return 1;
        };
        auto fstore = [&](bool pop) -> int {
            if (st_index(o[0]) >= 0) out_ += "    " + sti(st_index(o[0])) + " = " + sti(0) + ";\n";
            else {
                std::string a = o[0].type == ZYDIS_OPERAND_TYPE_MEMORY ? address(in, o[0]) : std::string{};
                if (a.empty() || (o[0].size != 32 && o[0].size != 64)) return -1;
                const char* t = o[0].size == 32 ? "float" : "double";
                out_ += std::string("    bb::rt::st<") + t + ">(c, " + a + ", " + t + "(" + sti(0) + "));\n";
            }
            if (pop) out_ += "    bb::rt::fpop(c);\n";
            return 1;
        };
        switch (m) {
        case ZYDIS_MNEMONIC_FADD: return arith('+', false, false, false);
        case ZYDIS_MNEMONIC_FADDP: return arith('+', false, false, true);
        case ZYDIS_MNEMONIC_FIADD: return arith('+', false, true, false);
        case ZYDIS_MNEMONIC_FSUB: return arith('-', false, false, false);
        case ZYDIS_MNEMONIC_FSUBP: return arith('-', false, false, true);
        case ZYDIS_MNEMONIC_FISUB: return arith('-', false, true, false);
        case ZYDIS_MNEMONIC_FSUBR: return arith('-', true, false, false);
        case ZYDIS_MNEMONIC_FSUBRP: return arith('-', true, false, true);
        case ZYDIS_MNEMONIC_FISUBR: return arith('-', true, true, false);
        case ZYDIS_MNEMONIC_FMUL: return arith('*', false, false, false);
        case ZYDIS_MNEMONIC_FMULP: return arith('*', false, false, true);
        case ZYDIS_MNEMONIC_FIMUL: return arith('*', false, true, false);
        case ZYDIS_MNEMONIC_FDIV: return arith('/', false, false, false);
        case ZYDIS_MNEMONIC_FDIVP: return arith('/', false, false, true);
        case ZYDIS_MNEMONIC_FIDIV: return arith('/', false, true, false);
        case ZYDIS_MNEMONIC_FDIVR: return arith('/', true, false, false);
        case ZYDIS_MNEMONIC_FDIVRP: return arith('/', true, false, true);
        case ZYDIS_MNEMONIC_FIDIVR: return arith('/', true, true, false);
        case ZYDIS_MNEMONIC_FCOM: case ZYDIS_MNEMONIC_FUCOM: return compare(false, 0);
        case ZYDIS_MNEMONIC_FCOMP: case ZYDIS_MNEMONIC_FUCOMP: return compare(false, 1);
        case ZYDIS_MNEMONIC_FCOMPP: case ZYDIS_MNEMONIC_FUCOMPP: return n == 0 ? compare(false, 2) : -1;
        case ZYDIS_MNEMONIC_FICOM: return compare(true, 0);
        case ZYDIS_MNEMONIC_FICOMP: return compare(true, 1);
        case ZYDIS_MNEMONIC_FCOMI: case ZYDIS_MNEMONIC_FUCOMI: return comi(false);
        case ZYDIS_MNEMONIC_FCOMIP: case ZYDIS_MNEMONIC_FUCOMIP: return comi(true);
        case ZYDIS_MNEMONIC_FCMOVB: return cmov("c.cf");
        case ZYDIS_MNEMONIC_FCMOVE: return cmov("c.zf");
        case ZYDIS_MNEMONIC_FCMOVBE: return cmov("c.cf || c.zf");
        case ZYDIS_MNEMONIC_FCMOVU: return cmov("c.pf");
        case ZYDIS_MNEMONIC_FCMOVNB: return cmov("!c.cf");
        case ZYDIS_MNEMONIC_FCMOVNE: return cmov("!c.zf");
        case ZYDIS_MNEMONIC_FCMOVNBE: return cmov("!c.cf && !c.zf");
        case ZYDIS_MNEMONIC_FCMOVNU: return cmov("!c.pf");
        case ZYDIS_MNEMONIC_FLD: {
            if (n != 1) return -1;
            if (st_index(o[0]) >= 0) {
                out_ += "    { double v = " + sti(st_index(o[0])) + "; bb::rt::fpush(c, v); }\n";
                return 1;
            }
            std::string v = memval(o[0], false);
            if (v.empty()) return -1;
            out_ += "    bb::rt::fpush(c, " + v + ");\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_FILD: {
            std::string v = n == 1 ? memval(o[0], true) : std::string{};
            if (v.empty()) return -1;
            out_ += "    bb::rt::fpush(c, " + v + ");\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_FLD1: out_ += "    bb::rt::fpush(c, 1.0);\n"; return 1;
        case ZYDIS_MNEMONIC_FLDZ: out_ += "    bb::rt::fpush(c, 0.0);\n"; return 1;
        case ZYDIS_MNEMONIC_FST: return fstore(false);
        case ZYDIS_MNEMONIC_FSTP: return fstore(true);
        case ZYDIS_MNEMONIC_FIST: return istore(false, false);
        case ZYDIS_MNEMONIC_FISTP: return istore(false, true);
        case ZYDIS_MNEMONIC_FISTTP: return istore(true, true);
        case ZYDIS_MNEMONIC_FBSTP: {
            std::string a = o[0].type == ZYDIS_OPERAND_TYPE_MEMORY ? address(in, o[0]) : std::string{};
            if (a.empty()) return -1;
            out_ += "    bb::rt::fbstp(c, " + a + ", " + sti(0) + ");\n    bb::rt::fpop(c);\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_FSIN: out_ += "    bb::rt::fsin(c);\n"; return 1;
        case ZYDIS_MNEMONIC_FCOS: out_ += "    bb::rt::fcos(c);\n"; return 1;
        case ZYDIS_MNEMONIC_FSINCOS: out_ += "    bb::rt::fsincos(c);\n"; return 1;
        case ZYDIS_MNEMONIC_FSCALE: out_ += "    bb::rt::fscale(c);\n"; return 1;
        case ZYDIS_MNEMONIC_FNSTCW: {
            std::string a = o[0].type == ZYDIS_OPERAND_TYPE_MEMORY ? address(in, o[0]) : std::string{};
            if (a.empty()) return -1;
            out_ += "    bb::rt::st<uint16_t>(c, " + a + ", c.fcw);\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_FLDCW: {
            std::string a = o[0].type == ZYDIS_OPERAND_TYPE_MEMORY ? address(in, o[0]) : std::string{};
            if (a.empty()) return -1;
            out_ += "    c.fcw = bb::rt::ld<uint16_t>(c, " + a + ");\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_FNSTENV: {
            std::string a = o[0].type == ZYDIS_OPERAND_TYPE_MEMORY ? address(in, o[0]) : std::string{};
            if (a.empty()) return -1;
            out_ += "    bb::rt::fnstenv(c, " + a + ");\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_FNSTSW: return write(in, o[0], "bb::rt::fstatus(c)") ? 1 : -1;
        case ZYDIS_MNEMONIC_FFREE: return 1;  // tag word not modelled
        case ZYDIS_MNEMONIC_FFREEP: out_ += "    bb::rt::fpop(c);\n"; return 1;
        default: return -2;
        }
    }

    // 1 = translated, falls through; 0 = translated, no fallthrough; -1 = unsupported.
    int translate_known(const Insn& in) {
        if (int v = translate_vec(in); v != -2) return v;
        if (int v = translate_x87(in); v != -2) return v;
        const auto& o = in.ops;
        const auto m = in.i.mnemonic;
        const unsigned w = o[0].size ? o[0].size : 64;
        const std::string T = utype(w);
        auto bin_flags = [&](const char* helper, bool store) -> int {
            std::string a = read(in, o[0], w), b = read(in, o[1], w);
            if (a.empty() || b.empty()) return -1;
            std::string v = std::string("bb::rt::") + helper + "<" + T + ">(c, " + a + ", " + b + ")";
            if (!store) {
                out_ += "    " + v + ";\n";
                return 1;
            }
            return write(in, o[0], v) ? 1 : -1;
        };
        auto logic = [&](const char* op, bool store) -> int {
            std::string a = read(in, o[0], w), b = read(in, o[1], w);
            if (a.empty() || b.empty()) return -1;
            std::string v = "bb::rt::f_logic<" + T + ">(c, " + T + "(" + a + " " + op + " " + b + "))";
            if (!store) {
                out_ += "    " + v + ";\n";
                return 1;
            }
            return write(in, o[0], v) ? 1 : -1;
        };
        auto unary = [&](const char* helper) -> int {
            std::string a = read(in, o[0], w);
            if (a.empty()) return -1;
            return write(in, o[0], std::string("bb::rt::") + helper + "<" + T + ">(c, " + a + ")") ? 1 : -1;
        };
        auto shift = [&](const char* helper) -> int {
            std::string a = read(in, o[0], w), n = read(in, o[1], 8);
            if (a.empty() || n.empty()) return -1;
            std::string mask = w == 64 ? "63" : "31";
            return write(in, o[0], std::string("bb::rt::") + helper + "<" + T + ">(c, " + a + ", unsigned(" + n +
                                       ") & " + mask + ")")
                       ? 1
                       : -1;
        };

        auto jump_if = [&](const std::string& cond) -> int {  // relative branch taken when `cond`
            uint64_t t;
            if (!rel_target(in, t)) return -1;
            if (inside(t) && insns_.count(t)) out_ += "    if (" + cond + ") goto " + label(t) + ";\n";
            else if (inside(t))  // lands inside the function but not on a decoded instruction: data / undecodable
                out_ += "    if (" + cond + ") bb::rt::unsupported(c, " + hex(in.ip) + ", \"branch to undecodable code\");\n";
            else {
                out_ += "    if (" + cond + ") {\n";
                transfer(t);
                out_ += "    }\n";
            }
            return 1;
        };
        if (const char* cc = condition(m)) {
            const auto cat = in.i.meta.category;
            if (cat == ZYDIS_CATEGORY_COND_BR) return jump_if(cc);
            if (cat == ZYDIS_CATEGORY_SETCC) return write(in, o[0], std::string("uint8_t(") + cc + ")") ? 1 : -1;
            if (cat != ZYDIS_CATEGORY_CMOV) return -1;
            // cmovcc: the 32-bit form zero-extends the destination even when the condition is false.
            std::string a = read(in, o[0], w), b = read(in, o[1], w);
            if (a.empty() || b.empty()) return -1;
            return write(in, o[0], std::string("(") + cc + " ? " + b + " : " + a + ")") ? 1 : -1;
        }

        // LOCK-prefixed read-modify-write on memory must be atomic (spinlocks, refcounts): compare-exchange loop around the
        // same flag-computing helpers; the flags of the successful iteration are the instruction's flags.
        if ((in.i.attributes & ZYDIS_ATTRIB_HAS_LOCK) && o[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            std::string expr;
            const std::string B = in.i.operand_count_visible >= 2 ? read(in, o[1], w) : std::string();
            auto bin = [&](const char* h) { return B.empty() ? std::string() : std::string("bb::rt::") + h + "<" + T + ">(c, old, " + B + ")"; };
            switch (m) {
            case ZYDIS_MNEMONIC_ADD: expr = bin("f_add"); break;
            case ZYDIS_MNEMONIC_SUB: expr = bin("f_sub"); break;
            case ZYDIS_MNEMONIC_ADC: expr = bin("f_adc"); break;
            case ZYDIS_MNEMONIC_SBB: expr = bin("f_sbb"); break;
            case ZYDIS_MNEMONIC_AND: if (!B.empty()) expr = "bb::rt::f_logic<" + T + ">(c, " + T + "(old & " + B + "))"; break;
            case ZYDIS_MNEMONIC_OR: if (!B.empty()) expr = "bb::rt::f_logic<" + T + ">(c, " + T + "(old | " + B + "))"; break;
            case ZYDIS_MNEMONIC_XOR: if (!B.empty()) expr = "bb::rt::f_logic<" + T + ">(c, " + T + "(old ^ " + B + "))"; break;
            case ZYDIS_MNEMONIC_INC: expr = "bb::rt::f_inc<" + T + ">(c, old)"; break;
            case ZYDIS_MNEMONIC_DEC: expr = "bb::rt::f_dec<" + T + ">(c, old)"; break;
            case ZYDIS_MNEMONIC_NEG: expr = "bb::rt::f_neg<" + T + ">(c, old)"; break;
            case ZYDIS_MNEMONIC_NOT: expr = T + "(~old)"; break;
            default: break;
            }
            if (!expr.empty()) {
                const std::string a = address(in, o[0]);
                if (a.empty()) return -1;
                out_ += "    { auto ref = bb::rt::atom<" + T + ">(c, " + a + "); " + T + " old = ref.load(), nw; do { nw = " + T + "(" + expr +
                        "); } while (!ref.compare_exchange_weak(old, nw)); }\n";
                return 1;
            }
        }

        char sk = 0;
        if (const unsigned sb = string_op(m, sk)) {
            if (in.i.address_width != 64) return -1;
            const auto at = in.i.attributes;
            const bool repne = (at & ZYDIS_ATTRIB_HAS_REPNE) != 0;
            const bool repe = (at & (ZYDIS_ATTRIB_HAS_REP | ZYDIS_ATTRIB_HAS_REPE)) != 0;
            // rep mode: 0 none, 1 rep/repe, 2 repne (only cmps/scas distinguish; others treat F2 like F3)
            const std::string flag = (sk == 'c' || sk == 'a') ? (repne ? "2" : repe ? "1" : "0")
                                                              : (repe || repne ? "1" : "0");
            const std::string S = utype(sb);
            const ZydisRegister acc = sb == 8    ? ZYDIS_REGISTER_AL
                                      : sb == 16 ? ZYDIS_REGISTER_AX
                                      : sb == 32 ? ZYDIS_REGISTER_EAX
                                                 : ZYDIS_REGISTER_RAX;
            switch (sk) {
            case 'm': out_ += "    bb::rt::str_movs<" + S + ">(c, " + flag + ");\n"; break;
            case 's': out_ += "    bb::rt::str_stos<" + S + ">(c, " + flag + ", " + S + "(" + reg_read(acc) + "));\n"; break;
            case 'l':
                out_ += "    { " + S + " v; if (bb::rt::str_lods<" + S + ">(c, " + flag + ", v)) {\n";
                if (!reg_write(acc, "v")) return -1;
                out_ += "    } }\n";
                break;
            case 'c': out_ += "    bb::rt::str_cmps<" + S + ">(c, " + flag + ");\n"; break;
            default: out_ += "    bb::rt::str_scas<" + S + ">(c, " + flag + ", " + S + "(" + reg_read(acc) + "));\n"; break;
            }
            return 1;
        }

        switch (m) {
        case ZYDIS_MNEMONIC_CLC: out_ += "    c.cf = false;\n"; return 1;
        case ZYDIS_MNEMONIC_STC: out_ += "    c.cf = true;\n"; return 1;
        case ZYDIS_MNEMONIC_CMC: out_ += "    c.cf = !c.cf;\n"; return 1;
        case ZYDIS_MNEMONIC_CLD: out_ += "    c.df = false;\n"; return 1;
        case ZYDIS_MNEMONIC_STD: out_ += "    c.df = true;\n"; return 1;
        case ZYDIS_MNEMONIC_MFENCE: case ZYDIS_MNEMONIC_SFENCE: case ZYDIS_MNEMONIC_LFENCE:
            out_ += "    std::atomic_thread_fence(std::memory_order_seq_cst);\n";
            return 1;
        case ZYDIS_MNEMONIC_FWAIT: case ZYDIS_MNEMONIC_EMMS: return 1;
        case ZYDIS_MNEMONIC_CPUID: out_ += "    bb::rt::cpuid(c);\n"; return 1;
        case ZYDIS_MNEMONIC_RDTSCP:
            out_ += "    { uint64_t t = bb::rt::rdtsc();\n";
            if (!reg_write(ZYDIS_REGISTER_EAX, "uint32_t(t)") || !reg_write(ZYDIS_REGISTER_EDX, "uint32_t(t >> 32)") ||
                !reg_write(ZYDIS_REGISTER_ECX, "0u"))
                return -1;
            out_ += "    }\n";
            return 1;
        case ZYDIS_MNEMONIC_BSWAP: {
            std::string a = read(in, o[0], w);
            return !a.empty() && write(in, o[0], "bb::rt::bswap<" + T + ">(" + a + ")") ? 1 : -1;
        }
        case ZYDIS_MNEMONIC_LEAVE:
            out_ += "    c.r[4] = c.r[5]; c.r[5] = bb::rt::ld<uint64_t>(c, c.r[4]); c.r[4] += 8;\n";
            return 1;
        case ZYDIS_MNEMONIC_ENTER: {  // enter size, level
            const unsigned level = unsigned(o[1].imm.value.u & 31);
            out_ += "    { c.r[4] -= 8; bb::rt::st<uint64_t>(c, c.r[4], c.r[5]); uint64_t fp = c.r[4];\n";
            if (level) {
                out_ += "    for (unsigned i = 1; i < " + std::to_string(level) +
                        "; ++i) { c.r[5] -= 8; c.r[4] -= 8; bb::rt::st<uint64_t>(c, c.r[4], bb::rt::ld<uint64_t>(c, c.r[5])); }\n";
                out_ += "    c.r[4] -= 8; bb::rt::st<uint64_t>(c, c.r[4], fp);\n";
            }
            out_ += "    c.r[5] = fp; c.r[4] -= " + std::to_string(o[0].imm.value.u & 0xFFFF) + "; }\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_CBW: return reg_write(ZYDIS_REGISTER_AX, "int16_t(int8_t(c.r[0]))") ? 1 : -1;
        case ZYDIS_MNEMONIC_CWDE: return reg_write(ZYDIS_REGISTER_EAX, "int32_t(int16_t(c.r[0]))") ? 1 : -1;
        case ZYDIS_MNEMONIC_CWD:
            return reg_write(ZYDIS_REGISTER_DX, "(int16_t(c.r[0]) < 0 ? 0xFFFFu : 0u)") ? 1 : -1;
        case ZYDIS_MNEMONIC_LAHF:  // AF is not modelled (reads as 0)
            return reg_write(ZYDIS_REGISTER_AH, "uint8_t(c.sf << 7 | c.zf << 6 | c.pf << 2 | 2 | int(c.cf))") ? 1 : -1;
        case ZYDIS_MNEMONIC_SAHF:
            out_ += "    { uint8_t a = uint8_t(c.r[0] >> 8); c.cf = a & 1; c.pf = (a >> 2) & 1; c.zf = (a >> 6) & 1; "
                    "c.sf = (a >> 7) & 1; }\n";
            return 1;
        case ZYDIS_MNEMONIC_XLAT:
            return reg_write(ZYDIS_REGISTER_AL, "bb::rt::ld<uint8_t>(c, c.r[3] + uint8_t(c.r[0]))") ? 1 : -1;
        case ZYDIS_MNEMONIC_BLSI: case ZYDIS_MNEMONIC_BLSR: case ZYDIS_MNEMONIC_BLSMSK: {
            std::string a = read(in, o[1], w);
            const char* h = m == ZYDIS_MNEMONIC_BLSI ? "f_blsi<" : m == ZYDIS_MNEMONIC_BLSR ? "f_blsr<" : "f_blsmsk<";
            return !a.empty() && write(in, o[0], std::string("bb::rt::") + h + T + ">(c, " + a + ")") ? 1 : -1;
        }
        case ZYDIS_MNEMONIC_RCL: return shift("f_rcl");
        case ZYDIS_MNEMONIC_RCR: return shift("f_rcr");
        case ZYDIS_MNEMONIC_SHLD:
        case ZYDIS_MNEMONIC_SHRD: {
            std::string a = read(in, o[0], w), b = read(in, o[1], w), k = read(in, o[2], 8);
            if (a.empty() || b.empty() || k.empty()) return -1;
            return write(in, o[0], std::string("bb::rt::") + (m == ZYDIS_MNEMONIC_SHLD ? "f_shld<" : "f_shrd<") + T +
                                       ">(c, " + a + ", " + b + ", unsigned(" + k + ") & " + (w == 64 ? "63" : "31") + ")")
                       ? 1
                       : -1;
        }
        case ZYDIS_MNEMONIC_BSF:
        case ZYDIS_MNEMONIC_BSR: {  // zero source: ZF = 1, destination unchanged
            std::string a = read(in, o[1], w);
            if (a.empty()) return -1;
            out_ += "    { " + T + " s = " + a + "; c.zf = s == 0; if (s) {\n";
            if (!write(in, o[0], m == ZYDIS_MNEMONIC_BSF ? T + "(std::countr_zero(s))"
                                                         : T + "(" + std::to_string(w - 1) + " - std::countl_zero(s))"))
                return -1;
            out_ += "    } }\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_LOOP: case ZYDIS_MNEMONIC_LOOPE: case ZYDIS_MNEMONIC_LOOPNE:
        case ZYDIS_MNEMONIC_JRCXZ: case ZYDIS_MNEMONIC_JECXZ: case ZYDIS_MNEMONIC_JCXZ: {
            const unsigned aw = m == ZYDIS_MNEMONIC_JRCXZ ? 64 : m == ZYDIS_MNEMONIC_JECXZ ? 32
                                : m == ZYDIS_MNEMONIC_JCXZ ? 16 : in.i.address_width;
            const std::string cnt = aw == 64 ? "c.r[1]" : aw == 32 ? "uint32_t(c.r[1])" : "uint16_t(c.r[1])";
            if (m == ZYDIS_MNEMONIC_JRCXZ || m == ZYDIS_MNEMONIC_JECXZ || m == ZYDIS_MNEMONIC_JCXZ)
                return jump_if(cnt + " == 0");
            if (aw == 64) out_ += "    c.r[1] -= 1;\n";
            else if (aw == 32) out_ += "    c.r[1] = uint32_t(c.r[1] - 1);\n";
            else return -1;
            return jump_if(cnt + " != 0" + (m == ZYDIS_MNEMONIC_LOOPE ? " && c.zf" : m == ZYDIS_MNEMONIC_LOOPNE ? " && !c.zf" : ""));
        }
        case ZYDIS_MNEMONIC_NOP:
        case ZYDIS_MNEMONIC_PAUSE:
        case ZYDIS_MNEMONIC_PREFETCHT0:
        case ZYDIS_MNEMONIC_PREFETCHT1:
        case ZYDIS_MNEMONIC_PREFETCHT2:
        case ZYDIS_MNEMONIC_PREFETCHNTA:
            return 1;
        case ZYDIS_MNEMONIC_MOV: {
            // Segment-register moves: selectors are constants (FreeBSD amd64 user selectors); writes are ignored.
            auto sel = [](const ZydisDecodedOperand& op) -> const char* {
                if (op.type != ZYDIS_OPERAND_TYPE_REGISTER) return nullptr;
                switch (op.reg.value) {
                case ZYDIS_REGISTER_ES: case ZYDIS_REGISTER_SS: case ZYDIS_REGISTER_DS: return "0x3Bu";
                case ZYDIS_REGISTER_CS: return "0x43u";
                case ZYDIS_REGISTER_FS: return "0x13u";
                case ZYDIS_REGISTER_GS: return "0x1Bu";
                default: return nullptr;
                }
            };
            if (const char* s = sel(o[1])) return write(in, o[0], s) ? 1 : -1;
            if (sel(o[0])) return 1;
            std::string v = read(in, o[1], w);
            return !v.empty() && write(in, o[0], v) ? 1 : -1;
        }
        case ZYDIS_MNEMONIC_MOVZX: {
            std::string v = read(in, o[1], o[1].size);
            return !v.empty() && write(in, o[0], T + "(" + v + ")") ? 1 : -1;
        }
        case ZYDIS_MNEMONIC_MOVSX:
        case ZYDIS_MNEMONIC_MOVSXD: {
            std::string v = read(in, o[1], o[1].size);
            if (v.empty()) return -1;
            return write(in, o[0], T + "(" + stype(w) + "(" + stype(o[1].size) + "(" + v + ")))") ? 1 : -1;
        }
        case ZYDIS_MNEMONIC_LEA: {
            std::string a = address(in, o[1]);
            return !a.empty() && write(in, o[0], T + "(" + a + ")") ? 1 : -1;
        }
        case ZYDIS_MNEMONIC_ADD: return bin_flags("f_add", true);
        case ZYDIS_MNEMONIC_SUB: return bin_flags("f_sub", true);
        case ZYDIS_MNEMONIC_CMP: return bin_flags("f_sub", false);
        case ZYDIS_MNEMONIC_AND: return logic("&", true);
        case ZYDIS_MNEMONIC_OR: return logic("|", true);
        case ZYDIS_MNEMONIC_XOR: return logic("^", true);
        case ZYDIS_MNEMONIC_TEST: return logic("&", false);
        case ZYDIS_MNEMONIC_INC: return unary("f_inc");
        case ZYDIS_MNEMONIC_DEC: return unary("f_dec");
        case ZYDIS_MNEMONIC_NEG: return unary("f_neg");
        case ZYDIS_MNEMONIC_NOT: {
            std::string a = read(in, o[0], w);
            return !a.empty() && write(in, o[0], T + "(~" + a + ")") ? 1 : -1;
        }
        case ZYDIS_MNEMONIC_SHL: return shift("f_shl");
        case ZYDIS_MNEMONIC_SHR: return shift("f_shr");
        case ZYDIS_MNEMONIC_SAR: return shift("f_sar");
        case ZYDIS_MNEMONIC_ROL: return shift("f_rol");
        case ZYDIS_MNEMONIC_ROR: return shift("f_ror");
        case ZYDIS_MNEMONIC_ADC: return bin_flags("f_adc", true);
        case ZYDIS_MNEMONIC_SBB: return bin_flags("f_sbb", true);
        case ZYDIS_MNEMONIC_MUL:
        case ZYDIS_MNEMONIC_DIV:
        case ZYDIS_MNEMONIC_IDIV:
            if (in.i.operand_count_visible == 1) return wide_muldiv(in, m);
            return -1;
        case ZYDIS_MNEMONIC_BT:
        case ZYDIS_MNEMONIC_BTS:
        case ZYDIS_MNEMONIC_BTR:
        case ZYDIS_MNEMONIC_BTC: return bit_test(in, m, w);
        case ZYDIS_MNEMONIC_LZCNT:
        case ZYDIS_MNEMONIC_TZCNT:
        case ZYDIS_MNEMONIC_POPCNT: {
            std::string a = read(in, o[1], w);
            const char* h = m == ZYDIS_MNEMONIC_LZCNT ? "f_lzcnt" : m == ZYDIS_MNEMONIC_TZCNT ? "f_tzcnt" : "f_popcnt";
            return !a.empty() && write(in, o[0], std::string("bb::rt::") + h + "<" + T + ">(c, " + a + ")") ? 1 : -1;
        }
        case ZYDIS_MNEMONIC_BEXTR: {
            std::string a = read(in, o[1], w), b = read(in, o[2], w);
            if (a.empty() || b.empty()) return -1;
            return write(in, o[0], "bb::rt::f_bextr<" + T + ">(c, " + a + ", " + b + ")") ? 1 : -1;
        }
        case ZYDIS_MNEMONIC_ANDN: {  // dst = ~src1 & src2; SF/ZF from result, CF = OF = 0
            std::string a = read(in, o[1], w), b = read(in, o[2], w);
            if (a.empty() || b.empty()) return -1;
            return write(in, o[0], "bb::rt::f_logic<" + T + ">(c, " + T + "(~" + a + " & " + b + "))") ? 1 : -1;
        }
        case ZYDIS_MNEMONIC_MOVBE: {
            std::string a = read(in, o[1], w);
            return !a.empty() && write(in, o[0], "bb::rt::bswap<" + T + ">(" + a + ")") ? 1 : -1;
        }
        case ZYDIS_MNEMONIC_RDTSC:
            out_ += "    { uint64_t t = bb::rt::rdtsc();\n";
            if (!reg_write(ZYDIS_REGISTER_EAX, "uint32_t(t)") || !reg_write(ZYDIS_REGISTER_EDX, "uint32_t(t >> 32)"))
                return -1;
            out_ += "    }\n";
            return 1;
        case ZYDIS_MNEMONIC_IMUL: {
            if (in.i.operand_count_visible == 2 || in.i.operand_count_visible == 3) {
                const auto& x = in.i.operand_count_visible == 2 ? o[0] : o[1];
                const auto& y = in.i.operand_count_visible == 2 ? o[1] : o[2];
                std::string a = read(in, x, w), b = read(in, y, w);
                if (a.empty() || b.empty()) return -1;
                return write(in, o[0], "bb::rt::f_imul<" + T + ">(c, " + a + ", " + b + ")") ? 1 : -1;
            }
            return in.i.operand_count_visible == 1 ? wide_muldiv(in, m) : -1;
        }
        case ZYDIS_MNEMONIC_XADD: {
            std::string s = read(in, o[1], w);
            if (s.empty()) return -1;
            if (o[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {  // lock xadd: atomic fetch-add
                std::string a = address(in, o[0]);
                if (a.empty()) return -1;
                out_ += "    { " + T + " s = " + s + "; " + T + " old = bb::rt::atom<" + T + ">(c, " + a +
                        ").fetch_add(s); bb::rt::f_add<" + T + ">(c, old, s);\n";
                if (!write(in, o[1], "old")) return -1;
                out_ += "    }\n";
                return 1;
            }
            std::string d = read(in, o[0], w);
            if (d.empty()) return -1;
            out_ += "    { " + T + " a = " + d + ", s = " + s + "; " + T + " r = bb::rt::f_add<" + T + ">(c, a, s);\n";
            if (!write(in, o[1], "a") || !write(in, o[0], "r")) return -1;
            out_ += "    }\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_CMPXCHG: {
            if (o[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return -1;
            const ZydisRegister acc = w == 8    ? ZYDIS_REGISTER_AL
                                      : w == 16 ? ZYDIS_REGISTER_AX
                                      : w == 32 ? ZYDIS_REGISTER_EAX
                                                : ZYDIS_REGISTER_RAX;
            std::string a = address(in, o[0]), s = read(in, o[1], w);
            if (a.empty() || s.empty()) return -1;
            // compare_exchange_strong leaves the observed value in `cur`; flags = cmp(accumulator, observed).
            out_ += "    { " + T + " exp = " + reg_read(acc) + ", cur = exp; bool ok = bb::rt::atom<" + T + ">(c, " + a +
                    ").compare_exchange_strong(cur, " + T + "(" + s + ")); bb::rt::f_sub<" + T +
                    ">(c, exp, cur);\n    if (!ok) {\n";
            if (!reg_write(acc, "cur")) return -1;
            out_ += "    } }\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_XCHG: {
            if (o[0].type == ZYDIS_OPERAND_TYPE_MEMORY || o[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {  // implicitly locked on x86
                const ZydisDecodedOperand& mem = o[0].type == ZYDIS_OPERAND_TYPE_MEMORY ? o[0] : o[1];
                const ZydisDecodedOperand& reg = o[0].type == ZYDIS_OPERAND_TYPE_MEMORY ? o[1] : o[0];
                std::string a = address(in, mem), r = read(in, reg, w);
                if (a.empty() || r.empty()) return -1;
                out_ += "    { " + T + " old = bb::rt::atom<" + T + ">(c, " + a + ").exchange(" + T + "(" + r + "));\n";
                if (!write(in, reg, "old")) return -1;
                out_ += "    }\n";
                return 1;
            }
            std::string a = read(in, o[0], w), b = read(in, o[1], w);
            if (a.empty() || b.empty()) return -1;
            out_ += "    { " + T + " t0 = " + a + ", t1 = " + b + ";\n";
            if (!write(in, o[0], "t1") || !write(in, o[1], "t0")) return -1;
            out_ += "    }\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_CDQE: return reg_write(ZYDIS_REGISTER_RAX, "int64_t(int32_t(c.r[0]))") ? 1 : -1;
        case ZYDIS_MNEMONIC_CDQ:
            return reg_write(ZYDIS_REGISTER_EDX, "(int32_t(c.r[0]) < 0 ? 0xFFFFFFFFu : 0u)") ? 1 : -1;
        case ZYDIS_MNEMONIC_CQO: return reg_write(ZYDIS_REGISTER_RDX, "(int64_t(c.r[0]) < 0 ? ~0ull : 0ull)") ? 1 : -1;
        case ZYDIS_MNEMONIC_PUSH: {
            std::string v = read(in, o[0], 64);
            if (v.empty()) return -1;
            push(v);
            return 1;
        }
        case ZYDIS_MNEMONIC_PUSHFQ: push("bb::rt::rflags(c)"); return 1;
        case ZYDIS_MNEMONIC_POPFQ:
            out_ += "    bb::rt::set_rflags(c, bb::rt::ld<uint64_t>(c, c.r[4])); c.r[4] += 8;\n";
            return 1;
        case ZYDIS_MNEMONIC_POP: {
            out_ += "    { uint64_t v = bb::rt::ld<uint64_t>(c, c.r[4]); c.r[4] += 8;\n";
            if (!write(in, o[0], "v")) return -1;
            out_ += "    }\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_CALL: {
            leaf_ = false;
            uint64_t t;
            if (rel_target(in, t)) {
                if (sjplt_.count(t)) {  // setjmp: saves the callee-saved registers + rsp + resume address, returns 0
                    sj_sites_.push_back(in.next() + bias_);
                    out_ += "    bb::rt::setjmp_save(c, c.r[7], " + hex(in.next() + bias_) + "ull);\n    c.r[0] = 0;\n    sj_" + std::to_string(sj_sites_.size() - 1) + ":;\n";
                    return 1;
                }
                push(hex(in.next() + bias_) + "ull");
                if (known_.count(t)) refs_.insert(t), out_ += "    " + fn_name(t) + "(c);\n";
                else out_ += "    bb::rt::call_indirect(c, " + hex(t + bias_) + ");\n";
                return 1;
            }
            std::string v = read(in, o[0], 64);  // evaluate before the push changes rsp
            if (v.empty()) return -1;
            out_ += "    { uint64_t t = " + v + ";\n";
            push(hex(in.next() + bias_) + "ull");
            out_ += "    bb::rt::call_indirect(c, t); }\n";
            return 1;
        }
        case ZYDIS_MNEMONIC_RET: {
            uint64_t extra = in.i.operand_count_visible ? o[0].imm.value.u : 0;
            if (extra) leaf_ = false;
            out_ += "    c.r[4] += " + std::to_string(8 + extra) + "; return;\n";
            return 0;
        }
        case ZYDIS_MNEMONIC_JMP: {
            uint64_t t;
            if (rel_target(in, t)) {
                if (inside(t) && insns_.count(t)) out_ += "    goto " + label(t) + ";\n";
                else if (inside(t))
                    out_ += "    bb::rt::unsupported(c, " + hex(in.ip) + ", \"jump to undecodable code\");\n";
                else transfer(t);
                return 0;
            }
            std::string v = read(in, o[0], 64);
            if (v.empty()) return -1;
            auto jt = tables_.find(in.ip);
            if (jt != tables_.end()) {
                out_ += "    switch (" + v + ") {\n";
                std::set<uint64_t> seen;
                for (uint64_t c : jt->second->targets)
                    if (insns_.count(c) && seen.insert(c).second)
                        out_ += "    case " + hex(c + bias_) + "ull: goto " + label(c) + ";\n";
                out_ += "    }\n    bb::rt::unsupported(c, " + hex(in.ip) + ", \"switch: target outside table\");\n";
                return 0;
            }
            leaf_ = false;
            out_ += "    bb::rt::call_indirect(c, " + v + "); return;\n";  // indirect tail call
            return 0;
        }
        default: return -1;
        }
    }

    const elf::Image& img_;
    ZydisDecoder& dec_;
    const Function& fn_;
    const std::unordered_map<uint64_t, const JumpTable*>& tables_;
    const std::unordered_set<uint64_t>& known_;
    const std::unordered_set<uint64_t>& sjplt_;
    EmitStats& stats_;
    uint64_t bias_;  // guest address = vaddr + bias_ (all absolute addresses in generated code)
    std::map<uint64_t, Insn> insns_;
    std::set<uint64_t> labels_;
    std::set<uint64_t> refs_;  // other recompiled functions called or tail-called directly
    std::string out_;
    bool leaf_ = true;
    std::vector<uint64_t> sj_sites_;
    // (sj_sites_ declared above)

public:
    bool leaf() const { return leaf_; }
    const std::set<uint64_t>& refs() const { return refs_; }
};

} // namespace

void emit_translation_unit(const elf::Image& img, const std::vector<Function>& functions, const IndirectJumps& jumps,
                           uint64_t bias, std::string* out, EmitStats& stats) {
    ZydisDecoder dec;
    ZydisDecoderInit(&dec, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    std::unordered_map<uint64_t, const JumpTable*> tables;
    for (auto& t : jumps.tables) tables.emplace(t.jmp, &t);
    std::unordered_set<uint64_t> known;
    for (auto& f : functions) known.insert(f.start);
    const std::unordered_set<uint64_t> sjplts = find_setjmp_plts(img);

    if (out) {
        *out += "// Generated by bbrecomp from the user's own eboot.bin. Do not commit.\n"
                "#include \"runtime/guest.h\"\n\n";
        for (auto& f : functions) *out += "void " + fn_name(f.start) + "(bb::rt::Context& c);\n";
        *out += "\n";
    }
    for (auto& f : functions) {
        FunctionEmitter fe(img, dec, f, tables, known, sjplts, stats, bias);
        std::string body = fe.emit();
        if (fe.leaf()) stats.leaves.push_back(f.start);
        if (out) *out += body;
    }
    if (out) {
        *out += "const bb::rt::FunctionEntry bb::rt::kFunctions[] = {\n";
        for (auto& f : functions) *out += "    {" + hex(f.start + bias) + "ull, " + fn_name(f.start) + "},\n";
        *out += "};\nconst size_t bb::rt::kFunctionCount = " + std::to_string(functions.size()) + ";\n";
        *out += "const uint64_t bb::rt::kLoadBias = " + hex(bias) + "ull;\n";
    }
}

void emit_sharded(const elf::Image& img, const std::vector<Function>& functions, const IndirectJumps& jumps,
                  uint64_t bias, size_t per_shard, std::vector<std::pair<std::string, std::string>>& files,
                  EmitStats& stats) {
    ZydisDecoder dec;
    ZydisDecoderInit(&dec, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    std::unordered_map<uint64_t, const JumpTable*> tables;
    for (auto& t : jumps.tables) tables.emplace(t.jmp, &t);
    std::unordered_set<uint64_t> known;
    for (auto& f : functions) known.insert(f.start);
    const std::unordered_set<uint64_t> sjplts = find_setjmp_plts(img);
    const char* header = "// Generated by bbrecomp from the user's own eboot.bin. Do not commit.\n"
                         "#include \"runtime/guest.h\"\n\n";

    per_shard = std::max<size_t>(1, per_shard);
    for (size_t first = 0; first < functions.size(); first += per_shard) {
        const size_t last = std::min(functions.size(), first + per_shard);
        std::string bodies;
        std::set<uint64_t> decls;  // everything this shard defines or calls
        for (size_t i = first; i < last; ++i) {
            FunctionEmitter fe(img, dec, functions[i], tables, known, sjplts, stats, bias);
            bodies += fe.emit();
            if (fe.leaf()) stats.leaves.push_back(functions[i].start);
            decls.insert(functions[i].start);
            decls.insert(fe.refs().begin(), fe.refs().end());
        }
        std::string file = header;
        for (uint64_t a : decls) file += "void " + fn_name(a) + "(bb::rt::Context& c);\n";
        file += "\n" + bodies;
        char name[32];
        std::snprintf(name, sizeof name, "shard_%04zu.cpp", first / per_shard);
        files.emplace_back(name, std::move(file));
    }

    std::string table = header;
    for (auto& f : functions) table += "void " + fn_name(f.start) + "(bb::rt::Context& c);\n";
    table += "\nconst bb::rt::FunctionEntry bb::rt::kFunctions[] = {\n";
    for (auto& f : functions) table += "    {" + hex(f.start + bias) + "ull, " + fn_name(f.start) + "},\n";
    table += "};\nconst size_t bb::rt::kFunctionCount = " + std::to_string(functions.size()) + ";\n";
    table += "const uint64_t bb::rt::kLoadBias = " + hex(bias) + "ull;\n";
    files.emplace_back("table.cpp", std::move(table));
}


std::string emit_import_table(const elf::Image& img, uint64_t bias,
                              const std::unordered_map<std::string, std::string>& names) {
    // GOT slot (vaddr) -> symbol index, from JUMP_SLOT relocations.
    std::unordered_map<uint64_t, uint32_t> slot_symbol;
    for (auto& r : img.relocations)
        if (r.type == elf::R_X86_64_JUMP_SLOT && r.symbol) slot_symbol.emplace(r.offset, r.symbol);

    // PLT stubs: `jmp [rip+disp32]; push imm32; jmp rel32` (ff 25 .. 68 .. e9 ..) whose slot is a JUMP_SLOT. Stub
    // alignment is not assumed (the eboot's PLT is not 16-byte aligned relative to the image base).
    std::unordered_map<uint32_t, uint64_t> plt_of;
    for (auto& s : img.segments) {
        if (s.type != elf::PT_LOAD || !(s.flags & 1)) continue;
        for (uint64_t a = s.vaddr; a + 12 <= s.vaddr + s.filesz; ++a) {
            auto b = img.at_vaddr(a, 12);
            if (b.size() != 12 || b[0] != 0xFF || b[1] != 0x25 || b[6] != 0x68 || b[11] != 0xE9) continue;
            int32_t disp;
            std::memcpy(&disp, b.data() + 2, 4);
            auto it = slot_symbol.find(a + 6 + int64_t(disp));
            if (it != slot_symbol.end()) plt_of.emplace(it->second, a + bias);
        }
    }

    auto esc = [](const std::string& s) {
        std::string o;
        for (char ch : s) o += ch == '"' || ch == '\\' ? std::string("\\") + ch : std::string(1, ch);
        return o;
    };
    std::string out = "// Generated by bbrecomp from the user's own eboot.bin. Do not commit.\n"
                      "#include \"runtime/guest.h\"\n\nconst bb::rt::ImportEntry bb::rt::kImports[] = {\n";
    size_t count = 0;
    for (uint32_t i = 0; i < img.symbols.size(); ++i) {
        const auto& sym = img.symbols[i];
        if (sym.shndx != 0 || sym.nid.empty() || sym.library_id < 0) continue;  // imports only
        const elf::Library* lib = img.find_library(sym.library_id, false);
        auto n = names.find(sym.nid);
        auto p = plt_of.find(i);
        out += "    {" + std::to_string(i) + ", " + hex(p != plt_of.end() ? p->second : 0) + "ull, \"" + esc(sym.nid) +
               "\", \"" + esc(lib ? lib->name : "?") + "\", \"" + esc(n != names.end() ? n->second : "") + "\"},\n";
        ++count;
    }
    out += "};\nconst size_t bb::rt::kImportCount = " + std::to_string(count) + ";\n";
    return out;
}
} // namespace bb
