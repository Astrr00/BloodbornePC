// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/sfo.h"

#include <cstring>

#include "core/util.h"

namespace bb {

std::optional<std::string> Sfo::str(const std::string& key) const {
    auto it = entries.find(key);
    if (it == entries.end()) return std::nullopt;
    if (auto* s = std::get_if<std::string>(&it->second)) return *s;
    return std::nullopt;
}

std::optional<Sfo> parse_sfo(std::span<const uint8_t> d) {
    // Header: magic "\0PSF", version, key_table_start, data_table_start, entry count (all LE u32).
    uint32_t magic, key_table, data_table, count;
    if (!read_le(d, 0x00, magic) || magic != 0x46535000) return std::nullopt;
    if (!read_le(d, 0x08, key_table) || !read_le(d, 0x0C, data_table) || !read_le(d, 0x10, count))
        return std::nullopt;
    Sfo sfo;
    for (uint32_t i = 0; i < count; ++i) {
        size_t e = 0x14 + size_t(i) * 16;
        uint16_t key_off, fmt;
        uint32_t len, data_off;
        if (!read_le(d, e, key_off) || !read_le(d, e + 2, fmt) || !read_le(d, e + 4, len) ||
            !read_le(d, e + 12, data_off))
            return std::nullopt;
        size_t kpos = size_t(key_table) + key_off;
        size_t dpos = size_t(data_table) + data_off;
        if (kpos >= d.size() || dpos > d.size() || len > d.size() - dpos) return std::nullopt;
        const char* kp = reinterpret_cast<const char*>(d.data() + kpos);
        std::string key(kp, strnlen(kp, d.size() - kpos));
        if (fmt == 0x0404) {
            uint32_t v;
            if (len < 4 || !read_le(d, dpos, v)) return std::nullopt;
            sfo.entries[key] = v;
        } else {  // 0x0204 utf8 (NUL-terminated), 0x0004 utf8 special
            const char* vp = reinterpret_cast<const char*>(d.data() + dpos);
            sfo.entries[key] = std::string(vp, strnlen(vp, len));
        }
    }
    return sfo;
}

std::optional<Sfo> load_sfo(const std::filesystem::path& path) {
    auto data = read_file(path);
    if (!data) return std::nullopt;
    return parse_sfo(*data);
}

} // namespace bb
