// SPDX-License-Identifier: GPL-3.0-or-later
// Orbis (PS4) ELF/SELF reader. Layout references:
//   https://www.psdevwiki.com/ps4/SELF_File_Format
//   https://github.com/shadps4-emu/shadPS4/blob/main/src/core/loader/elf.h (constants only, no code copied)
#pragma once
#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace bb::elf {

constexpr uint16_t ET_SCE_EXEC = 0xFE00;
constexpr uint16_t ET_SCE_DYNEXEC = 0xFE10;
constexpr uint16_t ET_SCE_DYNAMIC = 0xFE18;

constexpr uint32_t PT_LOAD = 1, PT_DYNAMIC = 2, PT_INTERP = 3, PT_TLS = 7;
constexpr uint32_t PT_SCE_RELA = 0x60000000;
constexpr uint32_t PT_SCE_DYNLIBDATA = 0x61000000;
constexpr uint32_t PT_SCE_PROCPARAM = 0x61000001;
constexpr uint32_t PT_SCE_MODULE_PARAM = 0x61000002;
constexpr uint32_t PT_SCE_RELRO = 0x61000010;
constexpr uint32_t PT_GNU_EH_FRAME = 0x6474E550;
constexpr uint32_t PT_SCE_COMMENT = 0x6FFFFF00;
constexpr uint32_t PT_SCE_LIBVERSION = 0x6FFFFF01;

constexpr int64_t DT_NULL = 0, DT_NEEDED = 1, DT_INIT = 12, DT_FINI = 13, DT_SONAME = 14;
constexpr int64_t DT_INIT_ARRAY = 25, DT_FINI_ARRAY = 26, DT_INIT_ARRAYSZ = 27, DT_FINI_ARRAYSZ = 28;
constexpr int64_t DT_SCE_FINGERPRINT = 0x61000007;
constexpr int64_t DT_SCE_ORIGINAL_FILENAME = 0x61000009;
constexpr int64_t DT_SCE_MODULE_INFO = 0x6100000D;
constexpr int64_t DT_SCE_NEEDED_MODULE = 0x6100000F;
constexpr int64_t DT_SCE_EXPORT_LIB = 0x61000013;
constexpr int64_t DT_SCE_IMPORT_LIB = 0x61000015;
constexpr int64_t DT_SCE_PLTGOT = 0x61000027;
constexpr int64_t DT_SCE_JMPREL = 0x61000029;
constexpr int64_t DT_SCE_PLTRELSZ = 0x6100002D;
constexpr int64_t DT_SCE_RELA = 0x6100002F;
constexpr int64_t DT_SCE_RELASZ = 0x61000031;
constexpr int64_t DT_SCE_STRTAB = 0x61000035;
constexpr int64_t DT_SCE_STRSZ = 0x61000037;
constexpr int64_t DT_SCE_SYMTAB = 0x61000039;
constexpr int64_t DT_SCE_SYMTABSZ = 0x6100003F;

constexpr uint32_t R_X86_64_64 = 1, R_X86_64_GLOB_DAT = 6, R_X86_64_JUMP_SLOT = 7, R_X86_64_RELATIVE = 8,
                   R_X86_64_DTPMOD64 = 16, R_X86_64_DTPOFF64 = 17, R_X86_64_TPOFF64 = 18;

struct Segment {
    uint32_t type, flags;
    uint64_t offset, vaddr, filesz, memsz, align;
};

struct Module {  // DT_SCE_NEEDED_MODULE / DT_SCE_MODULE_INFO
    uint16_t id;
    uint8_t major, minor;
    std::string name;
};

struct Library {  // DT_SCE_IMPORT_LIB / DT_SCE_EXPORT_LIB
    uint16_t id, version;
    std::string name;
    bool exported;
};

struct Symbol {
    std::string raw;      // e.g. "9BcDykPmo1I#A#B"
    std::string nid;      // "9BcDykPmo1I"
    int32_t library_id = -1, module_id = -1;
    uint8_t bind, type;
    uint16_t shndx;
    uint64_t value, size;
};

struct Relocation {
    uint64_t offset;  // vaddr patched
    uint32_t type;
    uint32_t symbol;  // index into symbols (0 = none)
    int64_t addend;
    bool plt;         // from DT_SCE_JMPREL
};

struct Image {
    bool from_self = false;
    uint16_t elf_type = 0;
    uint64_t entry = 0;
    std::vector<Segment> segments;
    std::vector<std::pair<int64_t, uint64_t>> dynamic;
    std::string original_filename;
    std::vector<uint8_t> fingerprint;
    std::vector<Module> self_module, needed_modules;  // self_module: DT_SCE_MODULE_INFO
    std::vector<Library> libraries;
    std::vector<Symbol> symbols;
    std::vector<Relocation> relocations;
    uint64_t sdk_version = 0;  // from PT_SCE_PROCPARAM, 0 if absent
    std::vector<std::string> warnings;
    std::vector<uint8_t> elf;  // plain ELF bytes (unwrapped if the input was a SELF)

    const Library* find_library(int32_t id, bool exported) const;
    const Module* find_module(int32_t id) const;
    // File-backed bytes at a virtual address (LOAD / GNU_EH_FRAME segments); empty if not fully backed.
    std::span<const uint8_t> at_vaddr(uint64_t vaddr, uint64_t size) const;
};

struct Error {
    std::string message;
};

// Accepts a plain ELF or an unencrypted, uncompressed (fake-signed) SELF.
// Encrypted SELFs are rejected: decryption is out of scope by design.
std::variant<Image, Error> parse(std::span<const uint8_t> file);

// Load bias for recompiled code: images linked at vaddr 0 (the eboot) go to 0x400000 - the address the PS4 uses and
// community patches assume - others stay at their vaddrs.
uint64_t default_load_bias(const Image& img);

const char* segment_type_name(uint32_t type);
const char* reloc_type_name(uint32_t type);
const char* elf_type_name(uint16_t type);

constexpr bool is_self(std::span<const uint8_t> d) {
    return d.size() >= 4 && d[0] == 0x4F && d[1] == 0x15 && d[2] == 0x3D && d[3] == 0x1D;
}
constexpr bool is_elf(std::span<const uint8_t> d) {
    return d.size() >= 4 && d[0] == 0x7F && d[1] == 'E' && d[2] == 'L' && d[3] == 'F';
}

} // namespace bb::elf
