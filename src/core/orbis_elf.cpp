// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/orbis_elf.h"

#include <algorithm>
#include <cstring>

#include "core/nid.h"
#include "core/util.h"

namespace bb::elf {
namespace {

constexpr uint64_t SELF_FLAG_ENCRYPTED = 0x2;
constexpr uint64_t SELF_FLAG_COMPRESSED = 0x8;
constexpr uint64_t SELF_FLAG_BLOCKED = 0x800;
constexpr uint32_t PROCPARAM_MAGIC = 0x4942524F;  // "ORBI"

struct Ehdr {
    uint8_t ident[16];
    uint16_t type, machine;
    uint32_t version;
    uint64_t entry, phoff, shoff;
    uint32_t flags;
    uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstrndx;
};
static_assert(sizeof(Ehdr) == 64);

struct Phdr {
    uint32_t type, flags;
    uint64_t offset, vaddr, paddr, filesz, memsz, align;
};
static_assert(sizeof(Phdr) == 56);

struct Sym {
    uint32_t name;
    uint8_t info, other;
    uint16_t shndx;
    uint64_t value, size;
};
static_assert(sizeof(Sym) == 24);

struct Rela {
    uint64_t offset, info;
    int64_t addend;
};
static_assert(sizeof(Rela) == 24);

struct SelfSegment {
    uint64_t flags, offset, filesz, memsz;
};
static_assert(sizeof(SelfSegment) == 32);

// Rebuilds the embedded ELF from a SELF whose segments are stored in plain form.
std::variant<std::vector<uint8_t>, Error> unwrap_self(std::span<const uint8_t> d) {
    uint16_t seg_count;
    if (!read_le(d, 0x18, seg_count)) return Error{"SELF header truncated"};
    size_t elf_off = 0x20 + size_t(seg_count) * sizeof(SelfSegment);
    Ehdr eh;
    if (!read_le(d, elf_off, eh) || eh.ident[0] != 0x7F || eh.ident[1] != 'E')
        return Error{"SELF does not contain a readable ELF header"};

    std::vector<Phdr> ph(eh.phnum);
    for (size_t i = 0; i < ph.size(); ++i)
        if (!read_le(d, elf_off + eh.phoff + i * sizeof(Phdr), ph[i])) return Error{"SELF program headers truncated"};

    uint64_t out_size = eh.phoff + ph.size() * sizeof(Phdr);
    for (auto& p : ph) out_size = std::max(out_size, p.offset + p.filesz);
    if (out_size > (uint64_t(4) << 30)) return Error{"SELF claims an implausible ELF size"};

    std::vector<uint8_t> out(out_size, 0);
    std::memcpy(out.data(), d.data() + elf_off, sizeof(Ehdr));
    std::memcpy(out.data() + eh.phoff, d.data() + elf_off + eh.phoff, ph.size() * sizeof(Phdr));

    for (uint16_t i = 0; i < seg_count; ++i) {
        SelfSegment s;
        if (!read_le(d, 0x20 + size_t(i) * sizeof(SelfSegment), s)) return Error{"SELF segment table truncated"};
        if (!(s.flags & SELF_FLAG_BLOCKED)) continue;
        if (s.flags & SELF_FLAG_ENCRYPTED)
            return Error{"eboot.bin is an encrypted SELF. This tool does not decrypt; provide a decrypted dump "
                         "of your own copy (plain ELF or fake-signed SELF)."};
        if (s.flags & SELF_FLAG_COMPRESSED)
            return Error{"SELF segment is compressed; compressed SELFs are not supported yet"};
        size_t id = size_t((s.flags >> 20) & 0xFFF);
        if (id >= ph.size()) return Error{"SELF segment references a missing program header"};
        if (s.offset > d.size() || s.filesz > d.size() - s.offset || ph[id].offset + s.filesz > out.size())
            return Error{"SELF segment data out of bounds"};
        std::memcpy(out.data() + ph[id].offset, d.data() + s.offset, s.filesz);
    }
    return out;
}

std::string read_cstr(std::span<const uint8_t> d, uint64_t off, uint64_t limit) {
    if (off >= d.size() || off >= limit) return {};
    const char* p = reinterpret_cast<const char*>(d.data() + off);
    return std::string(p, strnlen(p, size_t(std::min<uint64_t>(d.size(), limit) - off)));
}

Error parse_into(Image& img, std::span<const uint8_t> d) {
    Ehdr eh;
    if (!read_le(d, 0, eh) || !is_elf(d)) return {"not an ELF file"};
    if (eh.ident[4] != 2 || eh.ident[5] != 1 || eh.machine != 0x3E) return {"not a little-endian x86-64 ELF64"};
    img.elf_type = eh.type;
    img.entry = eh.entry;

    const Segment* dyn = nullptr;
    const Segment* dynlib = nullptr;
    for (uint16_t i = 0; i < eh.phnum; ++i) {
        Phdr p;
        if (!read_le(d, eh.phoff + size_t(i) * sizeof(Phdr), p)) return {"program header table truncated"};
        img.segments.push_back({p.type, p.flags, p.offset, p.vaddr, p.filesz, p.memsz, p.align});
    }
    for (auto& s : img.segments) {
        if (s.type == PT_DYNAMIC) dyn = &s;
        if (s.type == PT_SCE_DYNLIBDATA) dynlib = &s;
        if (s.type == PT_SCE_PROCPARAM) {
            uint32_t magic;
            if (read_le(d, s.offset + 8, magic) && magic == PROCPARAM_MAGIC) read_le(d, s.offset + 16, img.sdk_version);
        }
    }
    if (!dyn) {
        img.warnings.push_back("no PT_DYNAMIC segment (static image?)");
        return {};
    }
    if (!dynlib) return {"PT_DYNAMIC present but PT_SCE_DYNLIBDATA missing"};

    for (uint64_t off = dyn->offset; off + 16 <= dyn->offset + dyn->filesz; off += 16) {
        int64_t tag;
        uint64_t val;
        if (!read_le(d, off, tag) || !read_le(d, off + 8, val)) return {"dynamic section truncated"};
        if (tag == DT_NULL) break;
        img.dynamic.emplace_back(tag, val);
    }

    auto tag = [&](int64_t t) -> uint64_t {
        for (auto& [k, v] : img.dynamic)
            if (k == t) return v;
        return 0;
    };
    const uint64_t base = dynlib->offset, end = dynlib->offset + dynlib->filesz;
    for (int64_t required : {DT_SCE_STRTAB, DT_SCE_STRSZ, DT_SCE_SYMTAB, DT_SCE_SYMTABSZ})
        if (std::none_of(img.dynamic.begin(), img.dynamic.end(), [&](auto& kv) { return kv.first == required; }))
            return {"dynamic section lacks a required DT_SCE_STRTAB/STRSZ/SYMTAB/SYMTABSZ entry"};
    const uint64_t strtab = base + tag(DT_SCE_STRTAB), strend = strtab + tag(DT_SCE_STRSZ);
    if (strend > end) return {"DT_SCE_STRTAB outside PT_SCE_DYNLIBDATA"};
    auto str = [&](uint64_t off) { return read_cstr(d, strtab + off, strend); };

    for (auto& [t, v] : img.dynamic) {
        switch (t) {
        case DT_SCE_NEEDED_MODULE:
        case DT_SCE_MODULE_INFO: {
            Module m{uint16_t(v >> 48), uint8_t(v >> 40), uint8_t(v >> 32), str(uint32_t(v))};
            (t == DT_SCE_MODULE_INFO ? img.self_module : img.needed_modules).push_back(std::move(m));
            break;
        }
        case DT_SCE_IMPORT_LIB:
        case DT_SCE_EXPORT_LIB:
            img.libraries.push_back({uint16_t(v >> 48), uint16_t(v >> 32), str(uint32_t(v)), t == DT_SCE_EXPORT_LIB});
            break;
        case DT_SCE_ORIGINAL_FILENAME: img.original_filename = str(v); break;
        case DT_SCE_FINGERPRINT:
            if (base + v + 20 <= end) img.fingerprint.assign(d.begin() + (base + v), d.begin() + (base + v + 20));
            break;
        }
    }

    const uint64_t symtab = base + tag(DT_SCE_SYMTAB), symsz = tag(DT_SCE_SYMTABSZ);
    if (symtab + symsz > end) return {"DT_SCE_SYMTAB outside PT_SCE_DYNLIBDATA"};
    for (uint64_t off = 0; off + sizeof(Sym) <= symsz; off += sizeof(Sym)) {
        Sym s;
        read_le(d, symtab + off, s);
        Symbol sym{str(s.name), {}, -1, -1, uint8_t(s.info >> 4), uint8_t(s.info & 0xF), s.shndx, s.value, s.size};
        // Name format "NID#L#M": base64 NID, library id, module id.
        size_t h1 = sym.raw.find('#');
        sym.nid = sym.raw.substr(0, h1);
        if (h1 != std::string::npos) {
            size_t h2 = sym.raw.find('#', h1 + 1);
            if (auto l = decode_nid_id(std::string_view(sym.raw).substr(h1 + 1, h2 - h1 - 1))) sym.library_id = int32_t(*l);
            if (h2 != std::string::npos)
                if (auto m = decode_nid_id(std::string_view(sym.raw).substr(h2 + 1))) sym.module_id = int32_t(*m);
        }
        img.symbols.push_back(std::move(sym));
    }

    auto load_relocs = [&](uint64_t off, uint64_t size, bool plt) -> bool {
        if (base + off + size > end) return false;
        for (uint64_t o = 0; o + sizeof(Rela) <= size; o += sizeof(Rela)) {
            Rela r;
            read_le(d, base + off + o, r);
            img.relocations.push_back({r.offset, uint32_t(r.info), uint32_t(r.info >> 32), r.addend, plt});
        }
        return true;
    };
    if (!load_relocs(tag(DT_SCE_RELA), tag(DT_SCE_RELASZ), false)) return {"DT_SCE_RELA outside PT_SCE_DYNLIBDATA"};
    if (!load_relocs(tag(DT_SCE_JMPREL), tag(DT_SCE_PLTRELSZ), true)) return {"DT_SCE_JMPREL outside PT_SCE_DYNLIBDATA"};
    for (auto& r : img.relocations)
        if (r.symbol >= img.symbols.size() && r.symbol != 0) {
            img.warnings.push_back("relocation references out-of-range symbol index");
            break;
        }
    return {};
}

} // namespace

const Library* Image::find_library(int32_t id, bool exported) const {
    for (auto& l : libraries)
        if (l.id == id && l.exported == exported) return &l;
    return nullptr;
}

const Module* Image::find_module(int32_t id) const {
    for (auto& m : needed_modules)
        if (m.id == id) return &m;
    for (auto& m : self_module)
        if (m.id == id) return &m;
    return nullptr;
}

std::span<const uint8_t> Image::at_vaddr(uint64_t vaddr, uint64_t size) const {
    for (auto& s : segments) {
        if (s.type != PT_LOAD && s.type != PT_GNU_EH_FRAME) continue;
        if (vaddr < s.vaddr || vaddr - s.vaddr > s.filesz || size > s.filesz - (vaddr - s.vaddr)) continue;
        uint64_t off = s.offset + (vaddr - s.vaddr);
        if (off > elf.size() || size > elf.size() - off) return {};
        return {elf.data() + off, size_t(size)};
    }
    return {};
}

std::variant<Image, Error> parse(std::span<const uint8_t> file) {
    Image img;
    if (is_self(file)) {
        auto r = unwrap_self(file);
        if (auto* e = std::get_if<Error>(&r)) return *e;
        img.elf = std::move(std::get<std::vector<uint8_t>>(r));
        img.from_self = true;
    } else if (is_elf(file)) {
        img.elf.assign(file.begin(), file.end());
    } else {
        return Error{"neither ELF nor SELF (bad magic)"};
    }
    Error e = parse_into(img, img.elf);
    if (!e.message.empty()) return e;
    return img;
}

uint64_t default_load_bias(const Image& img) {
    uint64_t lowest = UINT64_MAX;
    for (auto& s : img.segments)
        if (s.type == PT_LOAD) lowest = std::min(lowest, s.vaddr);
    return lowest == 0 ? 0x400000 : 0;
}

const char* segment_type_name(uint32_t t) {
    switch (t) {
    case PT_LOAD: return "LOAD";
    case PT_DYNAMIC: return "DYNAMIC";
    case PT_INTERP: return "INTERP";
    case PT_TLS: return "TLS";
    case PT_SCE_RELA: return "SCE_RELA";
    case PT_SCE_DYNLIBDATA: return "SCE_DYNLIBDATA";
    case PT_SCE_PROCPARAM: return "SCE_PROCPARAM";
    case PT_SCE_MODULE_PARAM: return "SCE_MODULE_PARAM";
    case PT_SCE_RELRO: return "SCE_RELRO";
    case PT_GNU_EH_FRAME: return "GNU_EH_FRAME";
    case PT_SCE_COMMENT: return "SCE_COMMENT";
    case PT_SCE_LIBVERSION: return "SCE_LIBVERSION";
    default: return "?";
    }
}

const char* reloc_type_name(uint32_t t) {
    switch (t) {
    case R_X86_64_64: return "R_X86_64_64";
    case R_X86_64_GLOB_DAT: return "R_X86_64_GLOB_DAT";
    case R_X86_64_JUMP_SLOT: return "R_X86_64_JUMP_SLOT";
    case R_X86_64_RELATIVE: return "R_X86_64_RELATIVE";
    case R_X86_64_DTPMOD64: return "R_X86_64_DTPMOD64";
    case R_X86_64_DTPOFF64: return "R_X86_64_DTPOFF64";
    case R_X86_64_TPOFF64: return "R_X86_64_TPOFF64";
    default: return "R_X86_64_?";
    }
}

const char* elf_type_name(uint16_t t) {
    switch (t) {
    case ET_SCE_EXEC: return "ET_SCE_EXEC";
    case ET_SCE_DYNEXEC: return "ET_SCE_DYNEXEC";
    case ET_SCE_DYNAMIC: return "ET_SCE_DYNAMIC";
    case 2: return "ET_EXEC";
    case 3: return "ET_DYN";
    default: return "?";
    }
}

} // namespace bb::elf
