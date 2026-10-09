// SPDX-License-Identifier: GPL-3.0-or-later
#include "core.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "core/bbconfig.h"
#include "core/hash.h"
#include "core/install.h"
#include "core/sfo.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace bbl {
namespace {

std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
    return s;
}

std::string read_text(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Keeps ids usable as folder names.
std::string sanitize(std::string s) {
    for (char& c : s)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.')) c = '_';
    return s.empty() ? "game" : s;
}

int64_t mtime_secs(const fs::path& p) {
    std::error_code ec;
    const auto t = fs::last_write_time(p, ec);
    return ec ? 0 : std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count();
}

bool is_hex64(std::string_view s) {
    return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string_view::npos;
}

// First 64-hex token of a text, lowercase; "" if none.
std::string first_hash(std::string_view text) {
    text = trim(text);
    std::string h(text.substr(0, 64));
    std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return is_hex64(h) ? h : std::string();
}

// games.ini: "<exe name>=<sha256 of the eboot it was built from>" lines, '#' comments.
std::map<std::string, std::string> parse_games_ini(const std::string& text) {
    std::map<std::string, std::string> out;
    std::istringstream in(text);
    for (std::string l; std::getline(in, l);) {
        const std::string_view t = trim(l);
        const size_t eq = t.find('=');
        if (t.empty() || t[0] == '#' || eq == std::string_view::npos) continue;
        if (const std::string h = first_hash(t.substr(eq + 1)); !h.empty()) out[std::string(trim(t.substr(0, eq)))] = h;
    }
    return out;
}

} // namespace

Settings default_settings() {
    return {{"BB_FPS", "refresh"}, {"BB_WINDOW", "1920x1080"}, {"BB_RES_SCALE", "1"}, {"BB_UPSCALE", "fsr"}, {"BB_FSR_SHARP", "0.2"}, {"BB_INTERP", "1"}};
}

Config parse_config(std::string_view text) {
    Config c;
    Game* g = nullptr;
    auto finish = [&] {  // keys that a game entry lacks fall back to the defaults (an absent optional key stays off)
        if (!g) return;
        for (const auto& [k, v] : default_settings()) g->settings.emplace(k, v);
    };
    while (!text.empty()) {
        const size_t nl = text.find('\n');
        const std::string_view line = trim(text.substr(0, nl));
        text = nl == std::string_view::npos ? std::string_view{} : text.substr(nl + 1);
        if (line.empty() || line[0] == '#') continue;
        if (line == "[game]") {
            finish();
            c.games.emplace_back();
            g = &c.games.back();
            g->settings.clear();
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string_view::npos) continue;
        const std::string key(trim(line.substr(0, eq))), val(trim(line.substr(eq + 1)));
        if (!g) {
            if (key == "version") c.version = std::atoi(val.c_str());
            else if (key == "selected") c.selected = val;
            else if (key == "build_dir") c.build_dirs.push_back(val);
            else if (key == "pkg_extractor") c.pkg_extractor = val;
            else if (key == "pkg_game") c.pkg_game = val;
            else if (key == "pkg_dlc") c.pkg_dlc = val;
            else if (key == "pkg_update") c.pkg_update = val;
        } else if (key == "id") g->id = val;
        else if (key == "name") g->name = val;
        else if (key == "path") g->path = val;
        else if (key == "title_id") g->title_id = val;
        else if (key == "app_ver") g->app_ver = val;
        else if (key == "eboot_sha") g->eboot_sha = val;
        else if (key == "eboot_size") g->eboot_size = std::strtoull(val.c_str(), nullptr, 10);
        else if (key == "eboot_mtime") g->eboot_mtime = std::strtoll(val.c_str(), nullptr, 10);
        else if (key.rfind("BB_", 0) == 0) g->settings[key] = val;
        // anything else: written by a newer launcher, ignored
    }
    finish();
    std::erase_if(c.games, [](const Game& x) { return x.id.empty() || x.path.empty(); });
    return c;
}

std::string write_config(const Config& c) {
    std::ostringstream o;
    o << "# BloodbornePC launcher state. Edited by the launcher; unknown keys are ignored.\n";
    o << "version=" << kConfigVersion << "\nselected=" << c.selected << "\n";
    for (const std::string& d : c.build_dirs) o << "build_dir=" << d << "\n";
    for (const auto& [k, v] : {std::pair<const char*, const std::string&>{"pkg_extractor", c.pkg_extractor}, {"pkg_game", c.pkg_game}, {"pkg_dlc", c.pkg_dlc}, {"pkg_update", c.pkg_update}})
        if (!v.empty()) o << k << "=" << v << "\n";
    for (const Game& g : c.games) {
        o << "\n[game]\nid=" << g.id << "\nname=" << g.name << "\npath=" << g.path << "\ntitle_id=" << g.title_id << "\napp_ver=" << g.app_ver
          << "\neboot_sha=" << g.eboot_sha << "\neboot_size=" << g.eboot_size << "\neboot_mtime=" << g.eboot_mtime << "\n";
        for (const auto& [k, v] : g.settings) o << k << "=" << v << "\n";
    }
    return o.str();
}

bool load_config(const fs::path& file, Config& out) {
    std::ifstream f(file, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = parse_config(ss.str());
    return true;
}

bool save_config(const fs::path& file, const Config& c) {
    std::error_code ec;
    fs::create_directories(file.parent_path(), ec);
    const fs::path tmp = fs::path(file).concat(".tmp");
    {
        std::ofstream f(tmp, std::ios::binary);
        f << write_config(c);
        if (!f) return false;
    }
    fs::rename(tmp, file, ec);  // replaces the old file; a crash mid-write never leaves a truncated launcher.ini
    return !ec;
}

fs::path eboot_path(const fs::path& dump) { return bb::dump::resolve_layout(dump).app / "eboot.bin"; }

bool read_sfo_info(const fs::path& dump, std::string& title_id, std::string& app_ver, std::string& title) {
    const auto sfo = bb::load_sfo(bb::dump::resolve_layout(dump).sce_sys / "param.sfo");
    if (!sfo) return false;
    title_id = sfo->str("TITLE_ID").value_or("");
    app_ver = sfo->str("APP_VER").value_or("");
    title = sfo->str("TITLE").value_or("");
    return true;
}

bool ensure_eboot_hash(Game& g) {
    const fs::path p = eboot_path(g.path);
    std::error_code ec;
    const uint64_t size = fs::file_size(p, ec);
    if (ec) return false;
    const int64_t mt = mtime_secs(p);
    if (!g.eboot_sha.empty() && g.eboot_size == size && g.eboot_mtime == mt) return true;
    auto h = bb::sha256_file_hex(p);
    if (!h) return false;
    g.eboot_sha = *h, g.eboot_size = size, g.eboot_mtime = mt;
    return true;
}

std::string make_game_id(const std::string& title_id, const std::string& app_ver, const std::vector<Game>& taken) {
    const std::string base = sanitize(title_id.empty() ? "game" : title_id + (app_ver.empty() ? "" : "-" + app_ver));
    std::string id = base;
    for (int n = 2; std::any_of(taken.begin(), taken.end(), [&](const Game& g) { return g.id == id; }); ++n) id = base + "-" + std::to_string(n);
    return id;
}

std::vector<Build> scan_builds(const std::vector<fs::path>& dirs, const fs::path& scratch_dir) {
    std::vector<Build> out;
    for (const fs::path& dir : dirs) {
        std::error_code ec;
        const auto ini = parse_games_ini(read_text(dir / "games.ini"));
        std::vector<fs::path> exes;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string n = it->path().filename().string();
#ifdef _WIN32
            const bool is_exe = it->path().extension() == ".exe" && (n == "bbgame.exe" || n.rfind("bbgame_", 0) == 0);
#else
            const bool is_exe = n == "bbgame" || n.rfind("bbgame_", 0) == 0;
#endif
            std::error_code e2;
            if (is_exe && it->is_regular_file(e2) && n.find(".eboot.") == std::string::npos) exes.push_back(it->path());
        }
        std::sort(exes.begin(), exes.end());
        for (const fs::path& exe : exes) {
            Build b{exe, {}, {}, {}};
            if (valid_executable(exe, b.error)) b.error.clear();
            const fs::path side = fs::path(exe).concat(".eboot.sha256");
            // A sidecar older than the exe belongs to an earlier build (the CMake step rewrites it after every link).
            if (fs::is_regular_file(side, ec) && mtime_secs(side) >= mtime_secs(exe) - 2) {
                b.sha = first_hash(read_text(side)), b.source = "sidecar";
            }
            if (b.sha.empty())
                if (auto it = ini.find(exe.filename().string()); it != ini.end()) b.sha = it->second, b.source = "games.ini";
            if (b.sha.empty()) {
                const fs::path tmp = scratch_dir / "eboot-hash.txt";
                Plan p;
                p.exe = exe, p.args = {"--eboot-hash"}, p.cwd = dir, p.log_file = tmp;
                Proc pr;
                std::string err;
                std::error_code e3;
                fs::create_directories(scratch_dir, e3);
                if (spawn(p, pr, err)) {
                    int code = 1;
                    for (int i = 0; i < 100 && !poll(pr, code); ++i) {  // up to 10 s
#ifdef _WIN32
                        Sleep(100);
#else
                        usleep(100'000);
#endif
                    }
                    if (pr.valid) terminate(pr);
                    else if (code == 0) b.sha = first_hash(read_text(tmp)), b.source = "--eboot-hash";
                }
                fs::remove(tmp, e3);
            }
            out.push_back(std::move(b));
        }
    }
    return out;
}

const Build* match_build(const std::vector<Build>& builds, const std::string& eboot_sha) {
    if (eboot_sha.empty()) return nullptr;
    for (const Build& b : builds)
        if (b.sha == eboot_sha) return &b;
    return nullptr;
}

fs::path profile_dir(const fs::path& config_dir, const Game& g) { return config_dir / "profiles" / g.id; }

std::string game_config_text(const Game& g, const fs::path& profile) {
    Settings s = g.settings;
    std::erase_if(s, [](const auto& kv) { return kv.second.empty(); });
    if (!s.count("BB_SAVE_DIR")) s["BB_SAVE_DIR"] = (profile / "savedata").string();
    std::ostringstream o;
    o << "# Written by bblauncher for " << g.name << "; environment variables override these lines.\n";
    for (const auto& [k, v] : s) o << k << "=" << v << "\n";
    return o.str();
}

Plan make_plan(const Game& g, const Build& b, const fs::path& config_dir) {
    Plan p;
    const fs::path prof = fs::absolute(profile_dir(config_dir, g));  // absolute: the game runs with the profile folder as cwd
    p.exe = fs::absolute(b.exe);
    p.eboot = fs::absolute(eboot_path(g.path));
    p.cwd = prof;
    p.config_file = prof / "bbconfig.ini";
    p.log_file = prof / "game.log";
    p.args = {p.eboot.string()};
    p.config_text = game_config_text(g, prof);
    return p;
}

std::string describe(const Plan& p) {
    std::ostringstream o;
    o << "exe:    " << p.exe.string() << "\nargs:  ";
    for (const std::string& a : p.args) o << " \"" << a << "\"";
    o << "\ncwd:    " << p.cwd.string() << "\nenv:    BB_CONFIG=" << p.config_file.string() << "\nlog:    " << p.log_file.string() << "\n";
    o << "--- " << p.config_file.string() << " ---\n" << p.config_text;
    return o.str();
}

bool write_plan_files(const Plan& p, std::string& err) {
    std::error_code ec;
    for (const fs::path& d : {p.cwd, p.config_file.parent_path(), p.log_file.parent_path()})
        if (!d.empty()) fs::create_directories(d, ec);
    std::ofstream f(p.config_file, std::ios::binary);
    f << p.config_text;
    if (!f) {
        err = "cannot write " + p.config_file.string();
        return false;
    }
    return true;
}

fs::path resolve_program_in(const std::string& name, const std::vector<fs::path>& dirs, std::string& err) {
    if (name.empty()) {
        err = "no program given";
        return {};
    }
    std::error_code ec;
    if (name.find_first_of("/\\") != std::string::npos) return fs::absolute(name, ec);  // the user gave a path: take it, CreateProcess reports problems
    std::vector<std::string> names{name};
#ifdef _WIN32
    if (!fs::path(name).has_extension()) names.push_back(name + ".exe");
#endif
    for (const fs::path& d : dirs) {
        if (d.empty() || !d.is_absolute()) continue;  // an empty or relative entry means "the current directory": skipped on purpose
        for (const std::string& n : names) {
            const fs::path c = d / n;
            if (fs::exists(c, ec) && !fs::is_directory(c, ec)) return c;  // exists(): app-execution aliases (Store python) are reparse points
        }
    }
    err = "program '" + name + "' not found next to the launcher or on PATH";
    return {};
}

fs::path resolve_program(const std::string& name, std::string& err) {
    std::vector<fs::path> dirs{bb::exe_dir()};
    if (const char* path = std::getenv("PATH")) {
#ifdef _WIN32
        const char sep = ';';
#else
        const char sep = ':';
#endif
        std::istringstream in(path);
        for (std::string d; std::getline(in, d, sep);) dirs.emplace_back(d);
    }
    return resolve_program_in(name, dirs, err);
}

bool valid_executable(const fs::path& exe, std::string& err) {
    err = exe.string() + " is not a valid 64-bit program";
    std::ifstream f(exe, std::ios::binary);
    if (!f) {
        err = "cannot open " + exe.string();
        return false;
    }
    uint8_t h[64] = {};
    f.read(reinterpret_cast<char*>(h), sizeof h);
#ifdef _WIN32
    if (f.gcount() < 64 || h[0] != 'M' || h[1] != 'Z') return false;
    const uint32_t pe = h[60] | h[61] << 8 | h[62] << 16 | uint32_t(h[63]) << 24;
    uint8_t sig[6] = {};
    f.clear();
    f.seekg(pe);
    f.read(reinterpret_cast<char*>(sig), sizeof sig);
    return f.gcount() == 6 && sig[0] == 'P' && sig[1] == 'E' && sig[2] == 0 && sig[3] == 0 && (sig[4] | sig[5] << 8) == 0x8664;  // IMAGE_FILE_MACHINE_AMD64
#else
    return f.gcount() >= 4 && h[0] == 0x7f && h[1] == 'E' && h[2] == 'L' && h[3] == 'F' && access(exe.c_str(), X_OK) == 0;
#endif
}

#ifdef _WIN32

bool spawn(const Plan& p, Proc& proc, std::string& err) {
    if (!p.skip_exe_check && !valid_executable(p.exe, err)) return false;
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);  // never a system dialog for a bad image (the game gets the default mode back below)
    if (!p.config_file.empty()) SetEnvironmentVariableW(L"BB_CONFIG", p.config_file.c_str());
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE log = CreateFileW(p.log_file.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
    if (log == INVALID_HANDLE_VALUE) {
        err = "cannot open log file " + p.log_file.string();
        if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
        return false;
    }
    // CommandLineToArgvW rules: quote every argument, double the backslashes before a quote and at the end, escape embedded quotes.
    auto quote = [](const std::wstring& a) {
        std::wstring o = L"\"";
        for (size_t i = 0, bs = 0; i <= a.size(); ++i) {
            if (i < a.size() && a[i] == L'\\') { ++bs; continue; }
            if (i == a.size() || a[i] == L'"') o.append(bs * 2 + (i < a.size() ? 1 : 0), L'\\');
            else o.append(bs, L'\\');
            bs = 0;
            if (i < a.size()) o += a[i];
        }
        return o + L"\"";
    };
    std::wstring cmd = quote(p.exe.wstring());
    for (const std::string& a : p.args) cmd += L" " + quote(fs::path(a).wstring());
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nul, si.hStdOutput = log, si.hStdError = log;
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                                   CREATE_NO_WINDOW | CREATE_DEFAULT_ERROR_MODE | (p.kill_tree ? CREATE_SUSPENDED : 0), nullptr,
                                   p.cwd.empty() ? nullptr : p.cwd.c_str(), &si, &pi);
    const DWORD last = GetLastError();
    CloseHandle(log);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) {
        err = "cannot start " + p.exe.string() + " (error " + std::to_string(last) + ")";
        return false;
    }
    HANDLE job = nullptr;
    if (p.kill_tree) {  // the child (suspended until now) joins a job that dies with terminate() or with this process
        job = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION lim{};
        lim.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (job) SetInformationJobObject(job, JobObjectExtendedLimitInformation, &lim, sizeof lim);
        if (job) AssignProcessToJobObject(job, pi.hProcess);
        ResumeThread(pi.hThread);
    }
    CloseHandle(pi.hThread);
    proc = {reinterpret_cast<intptr_t>(pi.hProcess), true, reinterpret_cast<intptr_t>(job)};
    return true;
}

bool poll(Proc& proc, int& exit_code) {
    if (!proc.valid) return true;
    HANDLE h = reinterpret_cast<HANDLE>(proc.handle);
    DWORD code = 0;
    if (WaitForSingleObject(h, 0) != WAIT_OBJECT_0) return false;
    GetExitCodeProcess(h, &code);
    CloseHandle(h);
    if (proc.job) CloseHandle(reinterpret_cast<HANDLE>(proc.job)), proc.job = 0;  // kill-on-close: stray grandchildren end with the job
    proc.valid = false;
    exit_code = int(code);
    return true;
}

void terminate(Proc& proc) {
    if (!proc.valid) return;
    HANDLE h = reinterpret_cast<HANDLE>(proc.handle);
    if (proc.job) TerminateJobObject(reinterpret_cast<HANDLE>(proc.job), 1);
    TerminateProcess(h, 1);
    WaitForSingleObject(h, 5000);
    CloseHandle(h);
    if (proc.job) CloseHandle(reinterpret_cast<HANDLE>(proc.job)), proc.job = 0;
    proc.valid = false;
}

#else

bool spawn(const Plan& p, Proc& proc, std::string& err) {
    if (!p.skip_exe_check && !valid_executable(p.exe, err)) return false;
    if (!p.config_file.empty()) setenv("BB_CONFIG", p.config_file.c_str(), 1);
    std::vector<std::string> argv_s{p.exe.string()};
    argv_s.insert(argv_s.end(), p.args.begin(), p.args.end());
    std::vector<char*> argv;
    for (std::string& a : argv_s) argv.push_back(a.data());
    argv.push_back(nullptr);
    const int log = open(p.log_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (log < 0) {
        err = "cannot open log file " + p.log_file.string();
        return false;
    }
    const pid_t pid = fork();
    if (pid == 0) {
        if (p.kill_tree) setpgid(0, 0);
        if (!p.cwd.empty() && chdir(p.cwd.c_str()) != 0) _exit(127);
        dup2(log, 1), dup2(log, 2);
        close(log);
        execv(argv[0], argv.data());
        _exit(127);
    }
    close(log);
    if (pid < 0) {
        err = "fork failed";
        return false;
    }
    proc = {intptr_t(pid), true};
    return true;
}

bool poll(Proc& proc, int& exit_code) {
    if (!proc.valid) return true;
    int st = 0;
    if (waitpid(pid_t(proc.handle), &st, WNOHANG) != pid_t(proc.handle)) return false;
    proc.valid = false;
    exit_code = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
    return true;
}

void terminate(Proc& proc) {
    if (!proc.valid) return;
    kill(pid_t(proc.handle), SIGKILL);
    kill(-pid_t(proc.handle), SIGKILL);  // the process group of a kill_tree child (harmless ESRCH otherwise)
    waitpid(pid_t(proc.handle), nullptr, 0);
    proc.valid = false;
}

#endif

std::vector<std::string> tail_lines(const fs::path& file, size_t n) {
    std::ifstream f(file, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamoff size = f.tellg(), chunk = std::min<std::streamoff>(size, 64 * 1024);  // ponytail: lines longer than 64 KiB get cut
    std::string buf(size_t(chunk), '\0');
    f.seekg(size - chunk);
    f.read(buf.data(), chunk);
    std::vector<std::string> lines;
    std::istringstream in(buf);
    for (std::string l; std::getline(in, l);) {
        if (!l.empty() && l.back() == '\r') l.pop_back();
        lines.push_back(l);
    }
    if (lines.size() > n) lines.erase(lines.begin(), lines.end() - std::ptrdiff_t(n));
    return lines;
}

std::map<std::string, bool> cheat_verdicts(std::istream& log) {
    std::map<std::string, bool> out;
    constexpr std::string_view kTag = "cheats: ";
    for (std::string line; std::getline(log, line);) {
        const size_t at = line.find(kTag);
        if (at == std::string::npos) continue;
        std::string rest = line.substr(at + kTag.size());
        while (!rest.empty() && (rest.back() == '\r' || rest.back() == ' ')) rest.pop_back();
        constexpr std::string_view kUnavail = " unavailable on this build", kActive = " active";
        if (rest.size() > kUnavail.size() && rest.compare(rest.size() - kUnavail.size(), kUnavail.size(), kUnavail) == 0) {
            out[rest.substr(0, rest.size() - kUnavail.size())] = true;
        } else if (rest.size() > kActive.size() && rest.compare(rest.size() - kActive.size(), kActive.size(), kActive) == 0) {
            std::string list = rest.substr(0, rest.size() - kActive.size());
            std::replace(list.begin(), list.end(), ',', ' ');  // the game writes "god nohit" or "god,nohit"
            std::istringstream names(list);
            for (std::string n; names >> n;) out.emplace(n, false);
        }
    }
    return out;
}

} // namespace bbl
