// SPDX-License-Identifier: GPL-3.0-or-later
// bbconfig.ini: KEY=VALUE lines with the BB_* option names (README "Runtime options"), read by bbgame at start and written by the launcher.
// Real environment variables always win over the file.
#pragma once
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bb {

using ConfigEntries = std::vector<std::pair<std::string, std::string>>;

// '#' comment lines, blank lines, optional spaces around '=' and a trailing '\r' are ignored; only keys starting with "BB_" are kept.
ConfigEntries parse_config(std::string_view text);
// Sets every entry that is not already in the environment; returns how many were set.
int apply_config(const ConfigEntries& entries);
// $BB_CONFIG, else <exe dir>/bbconfig.ini, else <config_dir()>/bbconfig.ini; empty if none exists.
std::filesystem::path find_config_file(const std::filesystem::path& exe_dir);
// Reads find_config_file() into the environment; returns the log line ("config: <path> (N keys)") or "" when there is no file.
std::string load_config_into_env(const std::filesystem::path& exe_dir);

// %APPDATA%/BloodbornePC on Windows, $XDG_CONFIG_HOME/BloodbornePC (default ~/.config) elsewhere.
std::filesystem::path config_dir();
// Directory of the running executable (empty if unknown).
std::filesystem::path exe_dir();

} // namespace bb
