// GCN (GFX7 / Sea Islands, PS4 "Liverpool") instruction decoder + disassembler.
// Opcode names come from src/gpu/gcn_ops.inc (generated from Mesa ACO tables).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace bb::gcn {

enum class GcnFmt : uint8_t {
    SOP1, SOP2, SOPK, SOPC, SOPP, SMRD, VOP1, VOP2, VOPC, VOP3, VINTRP, DS, MUBUF, MTBUF, MIMG, EXP, Invalid
};

struct Insn {
    GcnFmt fmt = GcnFmt::Invalid;
    uint16_t op = 0;       // opcode in the format's own numbering (VOP3: 9-bit)
    uint8_t len = 1;       // dwords incl. literal
    uint32_t w[2] = {0, 0};
    uint32_t literal = 0;  // valid if has_literal
    bool has_literal = false;
};

// Decode one instruction at w[0..avail). Invalid -> fmt Invalid, len 1.
Insn decode(const uint32_t* w, size_t avail);
const char* mnemonic(GcnFmt fmt, unsigned op);  // nullptr if unknown
std::string disasm(const Insn& i);

// Sony shader binary trailer ("OrbShdr", psdevwiki / shadPS4 BinaryInfo): follows the code and the input-usage table.
// Layout: sig[7] ver(u8) | pssl:1 cached:1 type:4 source:2 length:24 (bytes of code) | ...
struct ShaderInfo {
    size_t trailer_dw = 0;   // dword index of the "OrbShdr" signature
    uint32_t code_bytes = 0; // exact code length (ends after the last s_endpgm)
    uint32_t type = 0;       // 0 PS, 1 VS, 2 GS, 3 CS, 4 HS, 5 LS, 6 ES (as seen: PS=0, VS=1)
    uint32_t crc = 0;
};
// Search the trailer within max_dw dwords. false if none.
bool shader_info(const uint32_t* w, size_t max_dw, ShaderInfo& out);

}  // namespace bb::gcn
