// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace bb {

// Bounds-checked little-endian read (host is x86-64 / LE on all targets).
template <typename T>
    requires std::is_trivially_copyable_v<T>
bool read_le(std::span<const uint8_t> d, size_t off, T& out) {
    if (off > d.size() || sizeof(T) > d.size() - off) return false;
    std::memcpy(&out, d.data() + off, sizeof(T));
    return true;
}

inline std::optional<std::vector<uint8_t>> read_file(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return std::nullopt;
    auto size = f.tellg();
    if (size < 0) return std::nullopt;
    std::vector<uint8_t> data(static_cast<size_t>(size));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(data.data()), size)) return std::nullopt;
    return data;
}

} // namespace bb
