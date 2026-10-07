// SPDX-License-Identifier: GPL-3.0-or-later
// Dump validation (structure + SHA-256 manifest) shared by CLI and future GUI.
#pragma once
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace bb::dump {

namespace fs = std::filesystem;

// Title IDs that ship app_ver 01.09, per
// https://github.com/illusion0001/console-game-patches/blob/main/_patch0/orbis/Bloodborne-Orbis.yml
inline constexpr const char* kKnownTitleIds[] = {"CUSA00900", "CUSA00207", "CUSA03173", "CUSA00208", "CUSA01363"};
inline constexpr const char* kRequiredAppVer = "01.09";

enum class Severity { Info, Warning, Error };

struct Finding {
    Severity severity;
    std::string message;
};

struct Report {
    std::vector<Finding> findings;
    std::string title_id, app_ver;
    bool ok() const;
    void add(Severity s, std::string m) { findings.push_back({s, std::move(m)}); }
};

struct ManifestEntry {
    std::string sha256;  // lowercase hex
    std::string path;    // relative, '/'-separated
};

// Parses `sha256sum` output format ("<hex>  <path>" or "<hex> *<path>"). Lines starting with '#' ignored.
std::vector<ManifestEntry> load_manifest(const fs::path& file, std::string& error);
bool write_manifest(const fs::path& file, const std::vector<ManifestEntry>& entries);

using Progress = std::function<void(size_t done, size_t total, const std::string& path)>;

// Hashes every regular file below root, sorted by path. Uses all hardware threads.
// On an unreadable directory or file, sets error and returns {} (a partial manifest would be misleading).
std::vector<ManifestEntry> hash_tree(const fs::path& root, const Progress& progress, std::string& error);

// Two accepted layouts: a console-style app folder (eboot.bin, sce_sys/ at root) or a PKG-extractor output
// (Image0/ = app files, Sc0/ = system files incl. param.sfo).
struct Layout {
    fs::path app, sce_sys;
};
Layout resolve_layout(const fs::path& root);

// Structural checks: param.sfo (title id, category, version), eboot.bin format, data dir.
void check_structure(const fs::path& game, Report& report);
void check_dlc(const fs::path& dlc, const std::string& game_title_id, Report& report);
// Every manifest entry must exist below root with a matching hash; extra files are allowed.
void check_manifest(const fs::path& root, const std::vector<ManifestEntry>& manifest, Report& report,
                    const Progress& progress);

// %APPDATA%/BloodbornePC on Windows, $XDG_DATA_HOME/BloodbornePC (default ~/.local/share) elsewhere.
fs::path default_install_dir();

} // namespace bb::dump
