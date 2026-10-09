// SPDX-License-Identifier: GPL-3.0-or-later
// Launcher logic without any UI: config file, build matching, launch plan, process start, log parsing.
#pragma once
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <istream>
namespace bbl {

namespace fs = std::filesystem;

// BB_* name -> value: exactly the lines of the bbconfig.ini that bbgame reads. A key that is absent means "off / game default".
using Settings = std::map<std::string, std::string>;
Settings default_settings();

struct Game {
    std::string id;     // folder name below profiles/, e.g. CUSA03173-01.00
    std::string name;   // shown in the list
    std::string path;   // dump folder (console layout, or PKG-extractor layout with Image0/ + Sc0/)
    std::string title_id, app_ver;
    std::string eboot_sha;                       // cached sha256 of eboot.bin (valid while size and mtime match)
    uint64_t eboot_size = 0;
    int64_t eboot_mtime = 0;
    Settings settings = default_settings();
    bool operator==(const Game&) const = default;
};

struct Config {
    int version = 1;
    std::string selected;  // Game::id
    std::vector<Game> games;
    std::vector<std::string> build_dirs;  // extra folders searched for bbgame builds (besides the launcher's own)
    // PKG import (external extractor, never decrypted by the launcher): command template, the last chosen PKG files, and the work folder of the last import
    std::string pkg_extractor, pkg_game, pkg_dlc, pkg_update;
    bool operator==(const Config&) const = default;
};

inline constexpr int kConfigVersion = 1;
// launcher.ini: "key=value" lines, "[game]" starts a game entry, '#' comments, unknown keys ignored.
Config parse_config(std::string_view text);
std::string write_config(const Config& c);
bool load_config(const fs::path& file, Config& out);  // false (and `out` untouched) if there is no readable file
bool save_config(const fs::path& file, const Config& c);

// Folder with eboot.bin for a dump folder.
fs::path eboot_path(const fs::path& dump);
// TITLE_ID / APP_VER / TITLE from param.sfo; false if unreadable.
bool read_sfo_info(const fs::path& dump, std::string& title_id, std::string& app_ver, std::string& title);
// Fills Game::eboot_sha (rehashes only when size/mtime changed); false if eboot.bin cannot be read.
bool ensure_eboot_hash(Game& g);
// "CUSA03173-01.00", made unique against `taken`.
std::string make_game_id(const std::string& title_id, const std::string& app_ver, const std::vector<Game>& taken);

struct Build {
    fs::path exe;
    std::string sha;     // sha256 of the eboot.bin this bbgame was recompiled from
    std::string source;  // where the hash came from: sidecar / games.ini / --eboot-hash
    std::string error;   // non-empty: the file is not a startable program (never launched, shown in the window)
};
// bbgame, bbgame_*(.exe) in each dir. Hash order: fresh `<exe>.eboot.sha256` sidecar (written by the CMake build), the dir's
// games.ini ("<exe name>=<sha256>" lines), then `<exe> --eboot-hash`. Builds without any hash are returned with an empty sha.
std::vector<Build> scan_builds(const std::vector<fs::path>& dirs, const fs::path& scratch_dir);
const Build* match_build(const std::vector<Build>& builds, const std::string& eboot_sha);

// Per-game folder: bbconfig.ini, game.log, default savedata.
fs::path profile_dir(const fs::path& config_dir, const Game& g);
// The bbconfig.ini the game will read (profile_dir/savedata fills an empty BB_SAVE_DIR).
std::string game_config_text(const Game& g, const fs::path& profile);

struct Plan {
    fs::path exe, eboot, cwd, config_file, log_file;
    std::vector<std::string> args;  // after the exe
    std::string config_text;
    bool skip_exe_check = false;  // third-party tool (PKG extractor): no PE/x64 gate, CreateProcess reports what is wrong
    bool kill_tree = false;  // terminate() also ends the child's own children (Windows job object / Linux process group)
};
Plan make_plan(const Game& g, const Build& b, const fs::path& config_dir);
// Human readable command line, environment and bbconfig.ini (--dry-run).
std::string describe(const Plan& p);
// Writes bbconfig.ini (creates folders); false + message on failure.
bool write_plan_files(const Plan& p, std::string& err);

// Is `exe` a program this platform can start (Windows: PE with the x64 machine type; elsewhere: ELF + execute bit)? spawn() checks this first
// so that a wrong file never reaches CreateProcess (which can pop up a system dialog); `err` explains a "no".
bool valid_executable(const fs::path& exe, std::string& err);

struct Proc {
    intptr_t handle = 0;  // HANDLE (Windows) or pid
    bool valid = false;
    intptr_t job = 0;  // Windows job object of a kill_tree process
};
// Finds a program for the extractor: a name with a folder part is taken as given (made absolute); a bare name is looked for in `dirs` in order
// (".exe" appended on Windows when it has no extension). The current directory is never searched.
fs::path resolve_program_in(const std::string& name, const std::vector<fs::path>& dirs, std::string& err);
// Same with the launcher's own folder first, then the absolute entries of PATH.
fs::path resolve_program(const std::string& name, std::string& err);
// Starts the game (stdout+stderr -> plan.log_file, BB_CONFIG=plan.config_file, cwd = plan.cwd).
bool spawn(const Plan& p, Proc& proc, std::string& err);
bool poll(Proc& proc, int& exit_code);  // true once the process ended (then proc.valid == false)
void terminate(Proc& proc);

std::vector<std::string> tail_lines(const fs::path& file, size_t n);
// Cheat verdicts from a game log: name -> true = "cheats: <name> unavailable on this build"; names in "cheats: <list> active" -> false.
std::map<std::string, bool> cheat_verdicts(std::istream& log);

} // namespace bbl
