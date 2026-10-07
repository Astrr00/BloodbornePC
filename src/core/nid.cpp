// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/nid.h"

#include <fstream>

#include "core/hash.h"

namespace bb {
namespace {
constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
constexpr uint8_t kSuffix[16] = {0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1,
                                 0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30};

int alphabet_index(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '-') return 63;
    return -1;
}
} // namespace

std::string generate_nid(std::string_view name) {
    Sha1 sha;
    sha.update({reinterpret_cast<const uint8_t*>(name.data()), name.size()});
    sha.update(kSuffix);
    auto d = sha.finish();
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | d[i];
    std::string out(11, '\0');
    for (int i = 0; i < 10; ++i) out[i] = kAlphabet[(v >> (58 - 6 * i)) & 63];
    out[10] = kAlphabet[(v & 0xF) << 2];
    return out;
}

std::optional<uint32_t> decode_nid_id(std::string_view s) {
    if (s.empty() || s.size() > 5) return std::nullopt;
    uint32_t v = 0;
    for (char c : s) {
        int i = alphabet_index(c);
        if (i < 0) return std::nullopt;
        v = (v << 6) | uint32_t(i);
    }
    return v;
}

std::unordered_map<std::string, std::string> load_nid_names(const std::filesystem::path& path) {
    std::unordered_map<std::string, std::string> map;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        map.emplace(generate_nid(line), line);
    }
    return map;
}

} // namespace bb
