// SPDX-License-Identifier: GPL-3.0-or-later
// PS4 NIDs: 11-char custom base64 of the first 8 bytes (LE) of SHA-1(name || fixed suffix).
// Algorithm verified against known pairs (__error -> 9BcDykPmo1I, sceKernelUsleep -> 1jfXLRVzisc).
#pragma once
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace bb {

std::string generate_nid(std::string_view symbol_name);

// Decodes the library/module id suffix of "NID#L#M" symbol names (big-endian base64 digits).
std::optional<uint32_t> decode_nid_id(std::string_view encoded);

// One symbol name per line ('#' comments allowed). Returns NID -> name.
// The name list is user-supplied (e.g. from ps4libdoc); none is bundled.
std::unordered_map<std::string, std::string> load_nid_names(const std::filesystem::path& path);

} // namespace bb
