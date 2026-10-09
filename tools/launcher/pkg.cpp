// SPDX-License-Identifier: GPL-3.0-or-later
#include "pkg.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <thread>

#include "core/install.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace bbl {

std::vector<std::string> split_command(std::string_view cmd, std::string& err) {
    err.clear();
    std::vector<std::string> out;
    std::string cur;
    bool in_quote = false, have = false;
    for (char c : cmd) {
        if (c == '"') {
            in_quote = !in_quote;
            have = true;
        } else if (c == ' ' && !in_quote) {
            if (have) out.push_back(cur), cur.clear(), have = false;
        } else {
            cur += c, have = true;
        }
    }
    if (in_quote) {
        err = "unbalanced quote in the extractor command";
        return {};
    }
    if (have) out.push_back(cur);
    return out;
}

std::vector<std::string> substitute_args(const std::vector<std::string>& tmpl, const std::string& pkg, const std::string& out) {
    std::vector<std::string> r;
    for (const std::string& a : tmpl) {
        std::string s;
        for (size_t i = 0; i < a.size();) {
            if (a.compare(i, 5, "{pkg}") == 0) s += pkg, i += 5;
            else if (a.compare(i, 5, "{out}") == 0) s += out, i += 5;
            else s += a[i++];
        }
        r.push_back(std::move(s));
    }
    return r;
}

std::string check_template(const std::vector<std::string>& tmpl) {
    if (tmpl.empty()) return "No extractor command is set.";
    if (tmpl[0].find("{pkg}") != std::string::npos || tmpl[0].find("{out}") != std::string::npos) return "The first word of the extractor command must be the program, not a placeholder.";
    bool pkg = false, out = false;
    for (const std::string& a : tmpl) pkg |= a.find("{pkg}") != std::string::npos, out |= a.find("{out}") != std::string::npos;
    if (!pkg || !out) return "The extractor command must contain both {pkg} and {out}.";
    return {};
}

std::string template_for_program(const std::string& path) {
    const std::string q = path.find(' ') != std::string::npos ? "\"" + path + "\"" : path;
    const bool py = path.size() > 3 && path.compare(path.size() - 3, 3, ".py") == 0;
    return (py ? "python " : "") + q + " {pkg} {out}";
}

bool read_pkg_header(const fs::path& file, PkgInfo& out, std::string& err) {
    std::error_code ec;
    if (!fs::is_regular_file(file, ec)) {
        err = "File not found: " + file.string();
        return false;
    }
    std::ifstream f(file, std::ios::binary);
    uint8_t h[0x64] = {};
    f.read(reinterpret_cast<char*>(h), sizeof h);
    if (f.gcount() < 4 || h[0] != 0x7F || h[1] != 'C' || h[2] != 'N' || h[3] != 'T') {
        err = "Not a PS4 PKG file (the header does not start with 7F 43 4E 54): " + file.filename().string();
        return false;
    }
    if (f.gcount() < std::streamsize(sizeof h)) {
        err = "The PKG file is truncated (shorter than its header): " + file.filename().string();
        return false;
    }
    out = {};
    out.size = fs::file_size(file, ec);
    for (int i = 0; i < 36 && h[0x40 + i] != 0; ++i) {
        if (h[0x40 + i] < 0x20 || h[0x40 + i] > 0x7E) {
            err = "The PKG header has no readable content id: " + file.filename().string();
            return false;
        }
        out.content_id += char(h[0x40 + i]);
    }
    if (out.content_id.size() >= 16 && out.content_id[6] == '-' && out.content_id[16] == '_') out.title_id = out.content_id.substr(7, 9);
    return true;
}

std::vector<bb::dump::Finding> check_pkg_set(const PkgInfo& game, const fs::path& game_path, const PkgInfo* dlc, const fs::path* dlc_path, const PkgInfo* update,
                                             const fs::path* update_path) {
    using bb::dump::Severity;
    std::vector<bb::dump::Finding> r;
    r.push_back({Severity::Info, "Game PKG: " + game.content_id + (game.title_id.empty() ? "" : " (title " + game.title_id + ")")});
    if (game.title_id.empty())
        r.push_back({Severity::Warning, "The game PKG's content id has an unusual shape; its title id cannot be checked."});
    else if (std::none_of(std::begin(bb::dump::kKnownTitleIds), std::end(bb::dump::kKnownTitleIds), [&](const char* id) { return game.title_id == id; }))
        r.push_back({Severity::Error, "The game PKG is not a known Bloodborne release (title " + game.title_id + ")."});
    auto one = [&](const char* what, const PkgInfo* p, const fs::path* path) {
        if (!p) return;
        std::error_code ec;
        r.push_back({Severity::Info, std::string(what) + " PKG: " + p->content_id + (p->title_id.empty() ? "" : " (title " + p->title_id + ")")});
        if (path && fs::equivalent(*path, game_path, ec))
            r.push_back({Severity::Error, std::string(what) + " PKG is the same file as the game PKG."});
        else if (!game.title_id.empty() && !p->title_id.empty() && p->title_id != game.title_id) {
            // ponytail: the Old Hunters DLC of the EU game ships under CUSA00900 (bbinstall warns for it too); another known Bloodborne title id is
            // therefore only a warning for a DLC PKG, while an update must match the game exactly.
            const bool known = std::any_of(std::begin(bb::dump::kKnownTitleIds), std::end(bb::dump::kKnownTitleIds), [&](const char* id) { return p->title_id == id; });
            if (std::string(what) == "DLC" && known)
                r.push_back({Severity::Warning, "DLC PKG has title " + p->title_id + ", the game has " + game.title_id + " (another release of the same game: allowed)."});
            else
                r.push_back({Severity::Error, std::string(what) + " PKG is not for this game (title " + p->title_id + " vs " + game.title_id + ")."});
        }
    };
    one("DLC", dlc, dlc_path);
    one("Update", update, update_path);
    if (dlc && update && dlc_path && update_path) {
        std::error_code ec;
        if (fs::equivalent(*dlc_path, *update_path, ec)) r.push_back({Severity::Error, "DLC PKG and update PKG are the same file."});
    }
    return r;
}

std::string new_run_id() {
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", std::localtime(&t));
#ifdef _WIN32
    const unsigned long pid = GetCurrentProcessId();
#else
    const unsigned long pid = static_cast<unsigned long>(getpid());
#endif
    return std::string(buf) + "-" + std::to_string(pid);
}

fs::path make_work_dir(const fs::path& data_dir, const std::string& run_id, std::string& err) {
    std::error_code ec;
    if (run_id.empty() || run_id.find_first_of("/\\.") != std::string::npos) {
        err = "invalid run id";
        return {};
    }
    const fs::path root = data_dir / "pkg_work", dir = root / run_id;
    fs::create_directories(root, ec);
    if (fs::exists(dir, ec)) {
        err = "The work folder already exists (" + dir.string() + "); refusing to reuse it.";
        return {};
    }
    if (!fs::create_directory(dir, ec)) {
        err = "Cannot create " + dir.string() + ": " + ec.message();
        return {};
    }
    std::ofstream(dir / kWorkMarker) << "created by bblauncher; safe to delete\n";
    return dir;
}

const char* kind_name(PkgKind kind) { return kind == PkgKind::Game ? "game" : kind == PkgKind::Dlc ? "dlc" : "update"; }

fs::path work_log(const fs::path& work_dir, PkgKind kind) {
    return work_dir.parent_path() / (work_dir.filename().string() + "-" + kind_name(kind) + ".log");
}

CheckAction check_request(AddPhase phase, bool install_registered) {
    if (phase == AddPhase::Checking || phase == AddPhase::Installing) return CheckAction::Defer;
    if (phase == AddPhase::Installed && !install_registered) return CheckAction::Defer;  // the installed copy must be added to the list first
    return CheckAction::Start;
}

std::vector<fs::path> list_work_dirs(const fs::path& data_dir) {
    std::vector<fs::path> out;
    std::error_code ec;
    for (fs::directory_iterator it(data_dir / "pkg_work", ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code e2;
        if (it->is_directory(e2) && !it->is_symlink(e2) && fs::is_regular_file(it->path() / kWorkMarker, e2)) out.push_back(it->path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool remove_work_dir(const fs::path& dir, const fs::path& data_dir, std::string& err, bool keep_logs) {
    std::error_code ec;
    const fs::path root = (data_dir / "pkg_work").lexically_normal(), d = dir.lexically_normal();
    if (dir.empty() || d.parent_path() != root || d.filename().empty()) {
        err = "Refusing to delete " + dir.string() + ": it is not a work folder of this launcher.";
        return false;
    }
    if (fs::is_symlink(d, ec) || !fs::is_regular_file(d / kWorkMarker, ec)) {
        err = "Refusing to delete " + dir.string() + ": the work folder marker is missing.";
        return false;
    }
    fs::remove_all(d, ec);
    if (ec) {
        err = "Cannot delete " + dir.string() + ": " + ec.message();
        return false;
    }
    if (!keep_logs)
        for (PkgKind k : {PkgKind::Game, PkgKind::Dlc, PkgKind::Update}) fs::remove(work_log(d, k), ec);
    return true;
}

std::string check_extract_space(uintmax_t pkg_bytes, const fs::path& data_dir) {
    // Work folder (data_dir/pkg_work) and installs (data_dir/games) share the volume of data_dir: the unpacked copy plus the installed copy.
    const auto avail = bb::dump::free_bytes(data_dir);
    if (!avail || *avail >= 2 * pkg_bytes) return {};
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "Not enough free space: extraction needs about %.1f GiB (the unpacked size, estimated from the PKG size) and the install needs the same again; %.1f GiB are free at %s.",
                  double(pkg_bytes) / double(1u << 30), double(*avail) / double(1u << 30), data_dir.string().c_str());
    return buf;
}

ExtractResult run_extractor(const std::vector<std::string>& tmpl, const fs::path& pkg, const fs::path& out, PkgKind kind, const fs::path& log_file,
                            const std::atomic<bool>* cancel) {
    ExtractResult r;
    if (const std::string bad = check_template(tmpl); !bad.empty()) {
        r.error = bad;
        return r;
    }
    std::error_code ec;
    if (fs::exists(out, ec)) {
        r.error = "The output folder already exists: " + out.string();
        return r;
    }
    if (!fs::create_directory(out, ec)) {
        r.error = "Cannot create " + out.string() + ": " + ec.message();
        return r;
    }
    // The extractor runs in the work folder: it must get absolute paths.
    std::vector<std::string> argv = substitute_args(tmpl, fs::absolute(pkg).string(), fs::absolute(out).string());
    std::string err;
    Plan p;
    p.exe = resolve_program(argv[0], err);
    if (p.exe.empty()) {
        r.error = "Extractor: " + err;
        return r;
    }
    p.args.assign(argv.begin() + 1, argv.end());
    p.cwd = fs::absolute(out).parent_path();
    p.log_file = log_file;
    p.kill_tree = true;
    p.skip_exe_check = true;  // a third-party tool: 32-bit programs, .NET AnyCPU, Store aliases are fine; CreateProcess reports real problems
    Proc proc;
    if (!spawn(p, proc, err)) {
        r.error = "Extractor: " + err;
        return r;
    }
    int code = 0;
    while (!poll(proc, code)) {
        if (cancel && cancel->load()) {
            terminate(proc);
            r.error = "Cancelled.";
            r.exit_code = 1;
            return r;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    r.exit_code = code;
    if (code != 0) {
        r.error = "The extractor exited with code " + std::to_string(code) + ". Last output:";
        for (const std::string& l : tail_lines(log_file, 6)) r.error += "\n  " + l;
        r.error += "\nFull log: " + log_file.string();
        return r;
    }
    const bool sc0 = fs::is_regular_file(out / "Sc0" / "param.sfo", ec), image0 = fs::is_directory(out / "Image0", ec);
    if (!sc0 || (kind != PkgKind::Dlc && !image0)) {
        r.error = std::string("The extractor finished but ") + (kind == PkgKind::Dlc ? "Sc0/param.sfo" : "Image0/ and Sc0/param.sfo") + " are missing in " + out.string() +
                  ". The command has to write the PKG-extractor layout (Image0/ + Sc0/) directly below {out}.\nFull log: " + log_file.string();
        return r;
    }
    r.ok = true;
    return r;
}

} // namespace bbl
