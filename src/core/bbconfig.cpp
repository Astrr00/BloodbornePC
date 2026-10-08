// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/bbconfig.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace bb {
namespace fs = std::filesystem;

namespace {
std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
    return s;
}
} // namespace

ConfigEntries parse_config(std::string_view text) {
    ConfigEntries out;
    while (!text.empty()) {
        const size_t nl = text.find('\n');
        const std::string_view line = trim(text.substr(0, nl));
        text = nl == std::string_view::npos ? std::string_view{} : text.substr(nl + 1);
        const size_t eq = line.find('=');
        if (line.empty() || line[0] == '#' || eq == std::string_view::npos) continue;
        const std::string_view key = trim(line.substr(0, eq));
        if (key.rfind("BB_", 0) == 0) out.emplace_back(std::string(key), std::string(trim(line.substr(eq + 1))));
    }
    return out;
}

int apply_config(const ConfigEntries& entries) {
    int n = 0;
    for (const auto& [k, v] : entries) {
        if (std::getenv(k.c_str())) continue;
#ifdef _WIN32
        if (_putenv_s(k.c_str(), v.c_str()) == 0) ++n;
#else
        if (setenv(k.c_str(), v.c_str(), 0) == 0) ++n;
#endif
    }
    return n;
}

fs::path config_dir() {
#ifdef _WIN32
    if (const char* appdata = std::getenv("APPDATA")) return fs::path(appdata) / "BloodbornePC";
#else
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME"); xdg && *xdg) return fs::path(xdg) / "BloodbornePC";
    if (const char* home = std::getenv("HOME")) return fs::path(home) / ".config/BloodbornePC";
#endif
    return fs::current_path() / "BloodbornePC";
}

fs::path exe_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(nullptr, buf, DWORD(std::size(buf)));
    if (n == 0 || n >= std::size(buf)) return {};
    return fs::path(buf).parent_path();
#else
    std::error_code ec;
    return fs::read_symlink("/proc/self/exe", ec).parent_path();
#endif
}

fs::path find_config_file(const fs::path& dir) {
    std::error_code ec;
    if (const char* e = std::getenv("BB_CONFIG"); e && *e) return fs::is_regular_file(e, ec) ? fs::path(e) : fs::path();
    for (const fs::path& p : {dir / "bbconfig.ini", config_dir() / "bbconfig.ini"})
        if (fs::is_regular_file(p, ec)) return p;
    return {};
}

std::string load_config_into_env(const fs::path& dir) {
    const fs::path p = find_config_file(dir);
    if (p.empty()) return {};
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const int n = apply_config(parse_config(ss.str()));
    return "config: " + p.string() + " (" + std::to_string(n) + " keys)";
}

} // namespace bb
