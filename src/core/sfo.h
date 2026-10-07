// SPDX-License-Identifier: GPL-3.0-or-later
// PARAM.SFO reader (format: https://www.psdevwiki.com/ps4/Param.sfo).
#pragma once
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <variant>

namespace bb {

using SfoValue = std::variant<std::string, uint32_t>;

struct Sfo {
    std::map<std::string, SfoValue> entries;
    std::optional<std::string> str(const std::string& key) const;
};

// Returns nullopt on malformed input.
std::optional<Sfo> parse_sfo(std::span<const uint8_t> data);
std::optional<Sfo> load_sfo(const std::filesystem::path& path);

} // namespace bb
