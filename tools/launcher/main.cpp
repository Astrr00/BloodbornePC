// SPDX-License-Identifier: GPL-3.0-or-later
// bblauncher: pick a dump, choose settings and cheats, start the matching bbgame build. SDL3 + Dear ImGui (SDL_Renderer).
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>

#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_sdlrenderer3.h"
#include "core.h"
#include "core/bbconfig.h"
#include "core/install.h"
#include "imgui.h"
#include "pkg.h"
#include "strings.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace bbl {
namespace {

using bb::dump::Severity;

constexpr ImVec4 kGreen{0.45f, 0.85f, 0.45f, 1}, kYellow{0.95f, 0.8f, 0.35f, 1}, kRed{1.0f, 0.45f, 0.45f, 1}, kGrey{0.62f, 0.62f, 0.66f, 1};

// ---- cheats: one table; every entry is proven by effect in-game (ammo dropped: no effect found) -----------------------------------
struct Cheat {
    const char* name;  // token in BB_CHEATS
    S label, desc;
};
constexpr Cheat kCheats[] = {
    {"god", S::cheat_god, S::cheat_god_desc},
    {"nohit", S::cheat_nohit, S::cheat_nohit_desc},
    {"stamina", S::cheat_stamina, S::cheat_stamina_desc},
};

struct Choice {
    const char* label;
    const char* value;
};

// ---- PNG (--shot): uncompressed deflate blocks, no zlib dependency ---------------------------------------------------------------------
uint32_t crc32(const uint8_t* p, size_t n, uint32_t c = 0) {
    c = ~c;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1)));
    }
    return ~c;
}

bool write_png(const char* path, const uint8_t* rgba, int w, int h) {
    std::vector<uint8_t> raw;
    raw.reserve(size_t(h) * (size_t(w) * 4 + 1));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgba + size_t(y) * w * 4, rgba + size_t(y + 1) * w * 4);
    }
    std::vector<uint8_t> z{0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t v : raw) a = (a + v) % 65521, b = (b + a) % 65521;
    for (size_t at = 0; at < raw.size() || at == 0;) {
        const size_t n = std::min<size_t>(65535, raw.size() - at);
        const bool last = at + n >= raw.size();
        z.insert(z.end(), {uint8_t(last), uint8_t(n), uint8_t(n >> 8), uint8_t(~n), uint8_t(~n >> 8)});
        z.insert(z.end(), raw.begin() + at, raw.begin() + at + n);
        at += n;
        if (last) break;
    }
    const uint32_t ad = (b << 16) | a;
    z.insert(z.end(), {uint8_t(ad >> 24), uint8_t(ad >> 16), uint8_t(ad >> 8), uint8_t(ad)});
    std::ofstream f(path, std::ios::binary);
    auto be32 = [](uint32_t v) { return std::vector<uint8_t>{uint8_t(v >> 24), uint8_t(v >> 16), uint8_t(v >> 8), uint8_t(v)}; };
    auto chunk = [&](const char* type, const std::vector<uint8_t>& data) {
        std::vector<uint8_t> c(type, type + 4);
        c.insert(c.end(), data.begin(), data.end());
        const auto len = be32(uint32_t(data.size())), crc = be32(crc32(c.data(), c.size()));
        f.write(reinterpret_cast<const char*>(len.data()), 4);
        f.write(reinterpret_cast<const char*>(c.data()), std::streamsize(c.size()));
        f.write(reinterpret_cast<const char*>(crc.data()), 4);
    };
    f.write("\x89PNG\r\n\x1a\n", 8);
    auto ihdr = be32(uint32_t(w));
    const auto hh = be32(uint32_t(h));
    ihdr.insert(ihdr.end(), hh.begin(), hh.end());
    ihdr.insert(ihdr.end(), {8, 6, 0, 0, 0});
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
    return bool(f);
}

// ---- add / install flow (runs on a worker thread, polled by the UI) --------------------------------------------------------------------
struct AddState {
    using enum AddPhase;
    std::atomic<AddPhase> phase{Checking};
    bool registered = false;  // UI thread: the finished install has been added to the game list
    fs::path dump, dlc, update, manifest;  // dlc / update: optional add-on and 1.09 update folders
    std::string title_id, app_ver, upd_ver, title, hint, current;
    bb::dump::Report report;
    std::vector<bb::dump::Finding> log;  // install messages
    uintmax_t need = 0;
    std::optional<uintmax_t> avail;
    fs::path dest;
    std::atomic<size_t> done{0}, total{0};
    std::atomic<bool> cancel{false};
    std::mutex m;  // guards current, log, report while a job runs
    std::future<void> job;
    bool has_eboot = false;
};

fs::path find_manifest(const std::string& title_id, const std::string& app_ver) {
    if (title_id.empty() || app_ver.empty()) return {};
    const fs::path root = bb::exe_dir();
    std::error_code ec;
    for (const fs::path& d : {root / "manifests", root.parent_path() / "manifests", root.parent_path().parent_path() / "manifests"}) {
        const fs::path f = d / (title_id + "-" + app_ver + ".sha256");
        if (fs::is_regular_file(f, ec)) return f;
    }
    return {};
}

void run_check(AddState& st, const fs::path& install_root) {
    bb::dump::Report pre;
    bb::dump::check_structure(st.dump, pre);
    st.title_id = pre.title_id, st.app_ver = pre.app_ver;
    std::string t, v, title;
    if (read_sfo_info(st.dump, t, v, title)) st.title = title;
    if (!st.update.empty()) read_sfo_info(st.update, t, st.upd_ver, title);
    st.manifest = find_manifest(st.title_id, st.update.empty() ? st.app_ver : bb::dump::kRequiredAppVer);  // with an update the result is 01.09
    bb::dump::Source src{st.dump, std::nullopt, std::nullopt, std::nullopt};
    if (!st.manifest.empty()) src.manifest = st.manifest;
    if (!st.dlc.empty()) src.dlc = st.dlc;
    if (!st.update.empty()) src.update = st.update;
    auto rep = bb::dump::validate(src, [&](size_t d, size_t n, const std::string& p) {
        st.done = d, st.total = n;
        std::lock_guard l(st.m);
        st.current = p;
    });
    std::error_code ec;
    st.has_eboot = fs::is_regular_file(eboot_path(st.dump), ec);
    // eboot.bin is there but param.sfo is not: most likely Image0 itself was chosen instead of the folder that also holds Sc0
    if (fs::is_regular_file(st.dump / "eboot.bin", ec) && !fs::is_regular_file(bb::dump::resolve_layout(st.dump).sce_sys / "param.sfo", ec) &&
        fs::is_regular_file(st.dump.parent_path() / "Sc0" / "param.sfo", ec))
        st.hint = T(S::parent_hint);
    st.need = bb::dump::install_bytes(src);
    st.avail = bb::dump::free_bytes(install_root);
    std::lock_guard l(st.m);
    st.report = std::move(rep);
    st.phase = AddState::Checked;
}

void run_install(AddState& st) {
    bb::dump::Source src{st.dump, st.manifest.empty() ? std::nullopt : std::optional<fs::path>(st.manifest),
                         st.dlc.empty() ? std::nullopt : std::optional<fs::path>(st.dlc),
                         st.update.empty() ? std::nullopt : std::optional<fs::path>(st.update)};
    const bool ok = bb::dump::install(
        src, st.dest,
        [&](size_t d, size_t n, const std::string& p) {
            st.done = d, st.total = n;
            std::lock_guard l(st.m);
            st.current = p;
        },
        [&](Severity s, const std::string& msg) {
            std::lock_guard l(st.m);
            st.log.push_back({s, msg});
        },
        &st.report, &st.cancel);
    st.phase = ok ? AddState::Installed : AddState::Failed;
}

// ---- the app ---------------------------------------------------------------------------------------------------------------------------
struct Picked {
    std::mutex m;
    std::string path;
    int kind = 0;  // 1 dump, 2 build folder, 3 save folder, 4 DLC folder, 5 update folder, 6-8 game/DLC/update PKG, 9 extractor program
};

// One PKG file chosen in the dialog; the plain header is read whenever the path text changes.
struct PkgSlot {
    char path[1024] = {};
    std::string shown, err;
    PkgInfo info;
    bool ok = false;
    void refresh() {
        if (shown == path) return;
        shown = path, err.clear(), info = {}, ok = false;
        if (path[0]) ok = read_pkg_header(path, info, err);
    }
};

// Extraction of the chosen PKGs by the external extractor (worker thread; the UI polls).
struct PkgJob {
    std::atomic<bool> cancel{false}, finished{false}, ok{false};
    bool handled = false;  // UI thread only
    std::mutex m;          // guards status, error, log, work
    std::string status, error;
    fs::path log, work;
    std::future<void> job;
};

void run_extract(PkgJob& j, std::vector<std::string> tmpl, std::array<fs::path, 3> pkgs, fs::path work, fs::path data_dir) {
    static const char* const kName[] = {"Game", "DLC", "Update"};
    static const PkgKind kKind[] = {PkgKind::Game, PkgKind::Dlc, PkgKind::Update};
    for (int i = 0; i < 3; ++i) {
        if (pkgs[i].empty()) continue;
        const fs::path log = work_log(work, kKind[i]);  // outside the work folder: survives the cleanup below
        {
            std::lock_guard l(j.m);
            j.status = std::string("Extracting the ") + kName[i] + " PKG ...";
            j.log = log;
        }
        const ExtractResult r = run_extractor(tmpl, pkgs[i], work / kind_name(kKind[i]), kKind[i], log, &j.cancel);
        if (!r.ok) {
            std::string e;
            const bool cleaned = remove_work_dir(work, data_dir, e, true);  // only the folder this run created (marker + parent checked); logs stay
            std::lock_guard l(j.m);
            j.error = r.error == "Cancelled." ? r.error : std::string(kName[i]) + " PKG: " + r.error;
            if (!cleaned) j.error += "\nThe partial output could not be removed: " + e + " (see 'Extracted PKG files' on the Games tab).";
            j.finished = true;
            return;
        }
    }
    j.ok = true;
    j.finished = true;
}

bool is_under(const fs::path& p, const fs::path& root) {
    const fs::path rel = p.lexically_normal().lexically_relative(root.lexically_normal());
    return !rel.empty() && *rel.begin() != ".." && !rel.is_absolute();
}

struct App {
    fs::path cfg_dir, data_dir, cfg_file;
    Config cfg;
    std::vector<fs::path> extra_build_dirs;  // --builds
    std::vector<Build> builds;
    std::future<std::vector<Build>> scan_f;
    std::future<std::vector<Game>> hash_f;
    std::shared_ptr<AddState> add;
    char add_path[1024] = {}, add_dlc[1024] = {}, add_update[1024] = {};
    PkgSlot pkg[3];  // game, DLC, update
    char extractor[1024] = {};
    std::shared_ptr<PkgJob> pkgjob;
    std::string pkg_msg, pkg_tail_text;
    uint64_t pkg_tail_at = 0;
    bool auto_extract = false, pending_check = false, show_delete = false;
    fs::path delete_target;  // work folder the confirmation popup is about (empty = popup closed)
    Picked picked;
    SDL_Window* window = nullptr;

    int want_tab = -1, shown_tab = 0;
    bool force_custom_res = false;
    bool open_add = false, auto_install = false;  // auto_install: --auto-install (tests)
    Proc proc;
    bool has_exit = false;
    int last_exit = 0;
    std::string status;
    std::vector<std::string> tail;
    std::map<std::string, bool> verdicts;
    bool has_log = false;
    std::string log_game;  // game whose log `verdicts`/`tail` describe

    Game* sel() {
        for (Game& g : cfg.games)
            if (g.id == cfg.selected) return &g;
        return nullptr;
    }
    const Build* build_for(const Game& g) const { return match_build(builds, g.eboot_sha); }
    bool busy() const { return scan_f.valid() || hash_f.valid(); }
    void save() {
        cfg.pkg_extractor = extractor, cfg.pkg_game = pkg[0].path, cfg.pkg_dlc = pkg[1].path, cfg.pkg_update = pkg[2].path;
        save_config(cfg_file, cfg);
    }
    void load_pkg_fields() {
        std::snprintf(extractor, sizeof extractor, "%s", cfg.pkg_extractor.c_str());
        std::snprintf(pkg[0].path, sizeof pkg[0].path, "%s", cfg.pkg_game.c_str());
        std::snprintf(pkg[1].path, sizeof pkg[1].path, "%s", cfg.pkg_dlc.c_str());
        std::snprintf(pkg[2].path, sizeof pkg[2].path, "%s", cfg.pkg_update.c_str());
    }
    void start_extract() {
        pkg_msg.clear();
        if (pkgjob && pkgjob->job.valid()) return;
        for (PkgSlot& p : pkg) p.refresh();
        if (!pkg[0].path[0]) return void(pkg_msg = T(S::pkg_need_game));
        for (int i = 0; i < 3; ++i)
            if (pkg[i].path[0] && !pkg[i].ok) return void(pkg_msg = pkg[i].err);
        const fs::path gp = pkg[0].path, dp = pkg[1].path, up = pkg[2].path;
        for (const auto& f : check_pkg_set(pkg[0].info, gp, pkg[1].path[0] ? &pkg[1].info : nullptr, &dp, pkg[2].path[0] ? &pkg[2].info : nullptr, &up))
            if (f.severity == Severity::Error) return void(pkg_msg = f.message);
        std::string e;
        const auto tmpl = split_command(extractor, e);
        if (e.empty()) e = check_template(tmpl);
        if (!e.empty()) return void(pkg_msg = e);
        uintmax_t bytes = 0;
        for (const PkgSlot& p : pkg)
            if (p.path[0]) bytes += p.info.size;
        if (const std::string sp = check_extract_space(bytes, data_dir); !sp.empty()) return void(pkg_msg = sp);
        const fs::path work = make_work_dir(data_dir, new_run_id(), e);
        if (work.empty()) return void(pkg_msg = e);
        save();
        pkgjob = std::make_shared<PkgJob>();
        pkgjob->work = work;
        pkgjob->job = std::async(std::launch::async, [j = pkgjob, tmpl, pkgs = std::array<fs::path, 3>{gp, pkg[1].path[0] ? dp : fs::path(), pkg[2].path[0] ? up : fs::path()},
                                                      work, dd = data_dir] { run_extract(*j, tmpl, pkgs, work, dd); });
    }
    void poll_extract() {
        if (!pkgjob || !pkgjob->finished || pkgjob->handled) return;
        pkgjob->handled = true;
        if (pkgjob->job.valid()) pkgjob->job.get();
        if (!pkgjob->ok) return;
        std::snprintf(add_path, sizeof add_path, "%s", (pkgjob->work / "game").string().c_str());
        std::snprintf(add_dlc, sizeof add_dlc, "%s", pkg[1].path[0] ? (pkgjob->work / "dlc").string().c_str() : "");
        std::snprintf(add_update, sizeof add_update, "%s", pkg[2].path[0] ? (pkgjob->work / "update").string().c_str() : "");
        save();
        start_check();
    }
    void browse_file(int kind, bool pkg_filter) {
        static const SDL_DialogFileFilter kPkg[] = {{"PS4 packages (*.pkg)", "pkg"}, {"All files", "*"}};
        picked.kind = kind;
        SDL_ShowOpenFileDialog(
            +[](void* ud, const char* const* list, int) {
                auto* p = static_cast<Picked*>(ud);
                if (!list || !list[0]) return;
                std::lock_guard l(p->m);
                p->path = list[0];
            },
            &picked, window, pkg_filter ? kPkg : nullptr, pkg_filter ? 2 : 0, nullptr, false);
    }

    void start_scan() {
        if (scan_f.valid()) return;
        std::vector<fs::path> dirs{bb::exe_dir()};
        for (const std::string& d : cfg.build_dirs) dirs.emplace_back(d);
        dirs.insert(dirs.end(), extra_build_dirs.begin(), extra_build_dirs.end());
        scan_f = std::async(std::launch::async, [dirs, scratch = cfg_dir] { return scan_builds(dirs, scratch); });
    }
    void start_hash() {
        if (hash_f.valid()) return;
        bool need = false;
        for (const Game& g : cfg.games) {
            Game probe = g;
            std::error_code ec;
            need |= probe.eboot_sha.empty() || probe.eboot_size != fs::file_size(eboot_path(g.path), ec);
        }
        if (!need) return;
        hash_f = std::async(std::launch::async, [games = cfg.games] {
            auto out = games;
            for (Game& g : out) ensure_eboot_hash(g);
            return out;
        });
    }
    void reload_log() {
        verdicts.clear();
        tail.clear();
        has_log = false;
        Game* g = sel();
        log_game = g ? g->id : "";
        if (!g) return;
        const fs::path log = profile_dir(cfg_dir, *g) / "game.log";
        std::ifstream f(log);
        if (!f) return;
        has_log = true;
        verdicts = cheat_verdicts(f);
        tail = tail_lines(log, 14);
    }
    void poll_jobs() {
        using namespace std::chrono_literals;
        if (scan_f.valid() && scan_f.wait_for(0s) == std::future_status::ready) builds = scan_f.get();
        if (hash_f.valid() && hash_f.wait_for(0s) == std::future_status::ready) {
            for (const Game& h : hash_f.get())
                for (Game& g : cfg.games)
                    if (g.id == h.id && g.path == h.path) g.eboot_sha = h.eboot_sha, g.eboot_size = h.eboot_size, g.eboot_mtime = h.eboot_mtime;
            save();
        }
        poll_extract();
        if (add && add->phase == AddPhase::Installed && !add->registered) {  // never orphan a finished install, whatever the dialog is doing
            if (add->job.valid()) add->job.get();
            add->registered = true;
            add_game(add->dest / "game");
        }
        if (pending_check && !(add && check_request(add->phase, add->registered) == CheckAction::Defer)) start_check();
        int code = 0;
        if (proc.valid && poll(proc, code)) {
            has_exit = true, last_exit = code;
            reload_log();
        }
        std::lock_guard l(picked.m);
        if (!picked.path.empty()) {
            const std::string p = std::move(picked.path);
            picked.path.clear();
            if (picked.kind == 1) {
                std::snprintf(add_path, sizeof add_path, "%s", p.c_str());
                start_check();
            } else if (picked.kind == 2) {
                cfg.build_dirs.push_back(p);
                save();
                start_scan();
            } else if (picked.kind == 3) {
                if (Game* g = sel()) g->settings["BB_SAVE_DIR"] = p, save();
            } else if (picked.kind == 4) {
                std::snprintf(add_dlc, sizeof add_dlc, "%s", p.c_str());
                if (add_path[0]) start_check();
            } else if (picked.kind == 5) {
                std::snprintf(add_update, sizeof add_update, "%s", p.c_str());
                if (add_path[0]) start_check();
            } else if (picked.kind >= 6 && picked.kind <= 8) {
                std::snprintf(pkg[picked.kind - 6].path, sizeof pkg[0].path, "%s", p.c_str());
                save();
            } else if (picked.kind == 9) {
                std::snprintf(extractor, sizeof extractor, "%s", template_for_program(p).c_str());
                save();
            }
        }
    }
    // Replaces the dialog state with a fresh check. While a check/install runs (or a finished install is not yet registered) the request
    // waits (pending_check) instead of orphaning that job.
    void start_check() {
        if (add && check_request(add->phase, add->registered) == CheckAction::Defer) {
            pending_check = true;
            return;
        }
        pending_check = false;
        if (add && add->job.valid()) add->job.wait();
        add = std::make_shared<AddState>();
        add->dump = fs::path(add_path);
        add->dlc = fs::path(add_dlc);
        add->update = fs::path(add_update);
        open_add = true;
        add->job = std::async(std::launch::async, [st = add, root = data_dir / "games"] { run_check(*st, root); });
    }
    void start_install() {
        AddState* st = add.get();
        st->dest = data_dir / "games" / make_game_id(st->title_id, st->update.empty() ? st->app_ver : bb::dump::kRequiredAppVer, cfg.games);
        st->phase = AddState::Installing;
        st->done = 0, st->total = 0;
        st->cancel = false;
        st->job = std::async(std::launch::async, [st] { run_install(*st); });
    }
    void browse(int kind) {
        picked.kind = kind;
        SDL_ShowOpenFolderDialog(
            +[](void* ud, const char* const* list, int) {
                auto* p = static_cast<Picked*>(ud);
                if (!list || !list[0]) return;
                std::lock_guard l(p->m);
                p->path = list[0];
            },
            &picked, window, nullptr, false);
    }
    // `dlc` (in-place use only; an install keeps it at <game>/../dlc, where the game looks by default): handed to the game as BB_DLC_DIR.
    void add_game(const fs::path& path, const fs::path& dlc = {}) {
        Game g;
        std::string title;
        read_sfo_info(path, g.title_id, g.app_ver, title);
        g.id = make_game_id(g.title_id, g.app_ver, cfg.games);
        g.name = !title.empty() ? title : !g.title_id.empty() ? g.title_id : path.filename().string();
        g.path = path.string();
        if (!dlc.empty()) g.settings["BB_DLC_DIR"] = dlc.string();
        cfg.games.push_back(g);
        cfg.selected = g.id;
        save();
        start_hash();
        reload_log();
    }
    void play() {
        Game* g = sel();
        const Build* b = g ? build_for(*g) : nullptr;
        if (!b || !b->error.empty() || proc.valid) return;
        const Plan p = make_plan(*g, *b, cfg_dir);
        std::string err;
        has_exit = false;
        status.clear();
        if (!write_plan_files(p, err) || !spawn(p, proc, err)) status = T(S::status_error) + err;
    }
};

// ---- widgets -------------------------------------------------------------------------------------------------------------------------
void hint(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, kGrey);
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
    ImGui::Spacing();
}

bool choice_combo(const char* label, Settings& s, const char* key, const Choice* items, int n) {
    const std::string cur = s.count(key) ? s[key] : "";
    int idx = -1;
    for (int i = 0; i < n; ++i)
        if (cur == items[i].value) idx = i;
    bool changed = false;
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 18);
    if (ImGui::BeginCombo(label, idx >= 0 ? items[idx].label : cur.c_str())) {
        for (int i = 0; i < n; ++i)
            if (ImGui::Selectable(items[i].label, i == idx)) s[key] = items[i].value, changed = true;
        ImGui::EndCombo();
    }
    return changed;
}

bool edit_text(const char* label, std::string& s) {
    char buf[1024];
    std::snprintf(buf, sizeof buf, "%s", s.c_str());
    ImGui::SetNextItemWidth(-ImGui::GetFontSize() * 9);
    if (!ImGui::InputText(label, buf, sizeof buf)) return false;
    s = buf;
    return true;
}

void sev_text(Severity s, const std::string& msg) {
    const ImVec4 c = s == Severity::Error ? kRed : s == Severity::Warning ? kYellow : kGrey;
    ImGui::PushStyleColor(ImGuiCol_Text, c);
    ImGui::TextWrapped("%s %s", s == Severity::Error ? "ERROR" : s == Severity::Warning ? "WARN " : "info ", msg.c_str());
    ImGui::PopStyleColor();
}

// ---- tabs ----------------------------------------------------------------------------------------------------------------------------
void games_tab(App& a) {
    Game* sel = a.sel();
    if (a.cfg.games.empty()) hint(T(S::no_games));
    const float line = ImGui::GetTextLineHeightWithSpacing();
    for (Game& g : a.cfg.games) {
        ImGui::PushID(g.id.c_str());
        const bool selected = g.id == a.cfg.selected;
        if (ImGui::Selectable("##row", selected, 0, ImVec2(0, line * 3.2f))) {
            a.cfg.selected = g.id;
            a.save();
            a.reload_log();
            sel = &g;
        }
        const ImVec2 p = ImGui::GetItemRectMin();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const Build* b = a.build_for(g);
        const std::string sub = (g.title_id.empty() ? "" : g.title_id + "  v" + g.app_ver + "   ") + g.path;
        const char* state = g.eboot_sha.empty() ? T(S::hashing) : b ? T(S::build_found) : T(S::build_none);
        const bool bad = b && !b->error.empty();
        const std::string st = bad ? b->error : b ? std::string(state) + b->exe.filename().string() : state;
        dl->AddText(ImVec2(p.x + 8, p.y + 2), ImGui::GetColorU32(ImGuiCol_Text), g.name.c_str());
        dl->AddText(ImVec2(p.x + 8, p.y + 2 + line), ImGui::GetColorU32(kGrey), sub.c_str());
        dl->AddText(ImVec2(p.x + 8, p.y + 2 + 2 * line), ImGui::GetColorU32(g.eboot_sha.empty() ? kGrey : bad ? kRed : b ? kGreen : kYellow), st.c_str());
        ImGui::PopID();
    }
    ImGui::Spacing();
    if (ImGui::Button(T(S::add_game))) {
        a.add_path[0] = a.add_dlc[0] = a.add_update[0] = 0;
        a.add.reset();
        a.open_add = true;
    }
    ImGui::SameLine();
    if (ImGui::Button(T(S::remove_game)) && sel) {
        std::erase_if(a.cfg.games, [&](const Game& g) { return g.id == a.cfg.selected; });
        a.cfg.selected = a.cfg.games.empty() ? "" : a.cfg.games.front().id;
        a.save();
        a.reload_log();
        sel = a.sel();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", T(S::remove_hint));

    if (const auto work = list_work_dirs(a.data_dir); !work.empty()) {
        ImGui::Separator();
        ImGui::TextUnformatted(T(S::pkg_work_note));
        const bool locked = (a.pkgjob && a.pkgjob->job.valid() && !a.pkgjob->finished) || (a.add && check_request(a.add->phase, true) == CheckAction::Defer);
        for (const fs::path& w : work) {
            ImGui::PushID(w.string().c_str());
            ImGui::BeginDisabled(locked);
            if (ImGui::Button(T(S::pkg_delete))) a.delete_target = w, a.show_delete = true;
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextUnformatted(w.string().c_str());
            ImGui::PopID();
        }
        if (!a.pkg_msg.empty()) ImGui::TextColored(kRed, "%s", a.pkg_msg.c_str());
    }
    if (sel && !a.build_for(*sel) && !sel->eboot_sha.empty() && !a.busy()) {
        ImGui::Separator();
        ImGui::PushStyleColor(ImGuiCol_Text, kYellow);
        ImGui::TextWrapped("%s", T(S::build_none));
        ImGui::PopStyleColor();
        hint(T(S::build_none_help));
        char steps[1024];
        std::snprintf(steps, sizeof steps, T(S::build_none_steps), eboot_path(sel->path).string().c_str());
        ImGui::InputTextMultiline("##steps", steps, sizeof steps, ImVec2(-1, line * 6.0f), ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_WordWrap);
        hint(T(S::build_none_after));
    }
    ImGui::Separator();
    ImGui::TextUnformatted(T(S::build_dirs));
    ImGui::TextDisabled("%s", bb::exe_dir().string().c_str());
    std::string rm;
    for (const std::string& d : a.cfg.build_dirs) {
        ImGui::PushID(d.c_str());
        ImGui::TextUnformatted(d.c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("x")) rm = d;
        ImGui::PopID();
    }
    if (!rm.empty()) {
        std::erase(a.cfg.build_dirs, rm);
        a.save();
        a.start_scan();
    }
    if (ImGui::Button(T(S::add_build_dir))) a.browse(2);
    ImGui::SameLine();
    if (ImGui::Button(T(S::rescan))) a.start_scan();
    if (a.scan_f.valid()) ImGui::SameLine(), ImGui::TextDisabled("%s", T(S::scanning));

    if (sel && a.has_log) {
        ImGui::Separator();
        const bool failed = a.has_exit && a.last_exit != 0;
        if (a.has_exit) {
            ImGui::TextColored(failed ? kRed : kGreen, "%s: %s %d", T(S::last_run), T(S::exit_code), a.last_exit);
        } else if (a.proc.valid) {
            ImGui::TextUnformatted(T(S::running));
        }
        if (ImGui::CollapsingHeader(T(S::log_tail), failed ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
            hint((std::string(T(S::log_file)) + (profile_dir(a.cfg_dir, *sel) / "game.log").string()).c_str());
            for (const std::string& l : a.tail) ImGui::TextUnformatted(l.c_str());
        }
    }
}

void settings_tab(App& a) {
    Game* g = a.sel();
    if (!g) {
        hint(T(S::no_game_selected));
        return;
    }
    Settings& s = g->settings;
    bool changed = false;
    ImGui::Text("%s%s", T(S::settings_for), g->name.c_str());
    ImGui::Spacing();

    const Choice fps[] = {{"30", "30"}, {"60", "60"}, {"120", "120"}, {"144", "144"}, {"165", "165"},
                          {T(S::fps_refresh), "refresh"}, {T(S::fps_unlimited), "unlimited"}, {T(S::fps_native), "native"}};
    changed |= choice_combo(T(S::fps), s, "BB_FPS", fps, int(std::size(fps)));
    hint(T(S::fps_hint));

    const Choice mode[] = {{T(S::windowed), ""}, {T(S::fullscreen), "1"}};
    {
        const std::string cur = s.count("BB_FULLSCREEN") ? s["BB_FULLSCREEN"] : "";
        int idx = cur == "1" ? 1 : 0;
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 18);
        if (ImGui::BeginCombo(T(S::window_mode), mode[idx].label)) {
            for (int i = 0; i < 2; ++i)
                if (ImGui::Selectable(mode[i].label, i == idx)) {
                    changed = true;
                    if (i) s["BB_FULLSCREEN"] = "1";
                    else s.erase("BB_FULLSCREEN");
                }
            ImGui::EndCombo();
        }
    }
    hint(T(S::window_mode_hint));

    static const char* const kRes[] = {"1280x720", "1920x1080", "2560x1440", "3440x1440", "3840x2160"};
    {
        const std::string cur = s["BB_WINDOW"];
        int idx = -1;
        for (int i = 0; i < int(std::size(kRes)); ++i)
            if (cur == kRes[i]) idx = i;
        const bool custom = idx < 0 || a.force_custom_res;
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 18);
        if (ImGui::BeginCombo(T(S::resolution), custom ? T(S::custom) : kRes[idx])) {
            for (int i = 0; i < int(std::size(kRes)); ++i)
                if (ImGui::Selectable(kRes[i], !custom && i == idx)) s["BB_WINDOW"] = kRes[i], a.force_custom_res = false, changed = true;
            if (ImGui::Selectable(T(S::custom), custom)) a.force_custom_res = true;
            ImGui::EndCombo();
        }
        if (custom) {
            int w = 1920, h = 1080;
            std::sscanf(cur.c_str(), "%dx%d", &w, &h);
            ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8);
            bool c = ImGui::InputInt("##w", &w, 0, 0);
            ImGui::SameLine();
            ImGui::TextUnformatted("x");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8);
            c |= ImGui::InputInt("##h", &h, 0, 0);
            if (c) {
                w = std::clamp(w, 320, 16384), h = std::clamp(h, 200, 16384);
                s["BB_WINDOW"] = std::to_string(w) + "x" + std::to_string(h);
                changed = true;
            }
        }
    }
    hint(T(S::resolution_hint));

    const Choice scale[] = {{"1x", "1"}, {"2x", "2"}, {"3x", "3"}};
    changed |= choice_combo(T(S::res_scale), s, "BB_RES_SCALE", scale, 3);
    hint(T(S::res_scale_hint));

    const Choice up[] = {{T(S::upscaler_fsr), "fsr"}, {T(S::upscaler_linear), "linear"}};
    changed |= choice_combo(T(S::upscaler), s, "BB_UPSCALE", up, 2);
    hint(T(S::upscaler_hint));

    {
        float sharp = float(std::atof(s["BB_FSR_SHARP"].c_str()));
        ImGui::BeginDisabled(s["BB_UPSCALE"] == "linear");
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 18);
        if (ImGui::SliderFloat(T(S::sharpness), &sharp, 0.0f, 2.0f, "%.2f")) {
            char buf[16];
            std::snprintf(buf, sizeof buf, "%.2f", sharp);
            s["BB_FSR_SHARP"] = buf;
            changed = true;
        }
        ImGui::EndDisabled();
    }
    hint(T(S::sharpness_hint));

    {
        bool on = s["BB_INTERP"] != "0";
        if (ImGui::Checkbox(T(S::interp), &on)) s["BB_INTERP"] = on ? "1" : "0", changed = true;
    }
    hint(T(S::interp_hint));

    {
        std::string dir = s.count("BB_SAVE_DIR") ? s["BB_SAVE_DIR"] : "";
        if (edit_text("##save", dir)) {
            if (dir.empty()) s.erase("BB_SAVE_DIR");
            else s["BB_SAVE_DIR"] = dir;
            changed = true;
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(T(S::save_dir));
        ImGui::SameLine();
        if (ImGui::Button(T(S::browse))) a.browse(3);
    }
    hint(T(S::save_dir_hint));

    {
        bool on = s.count("BB_STUB_IMPORTS") != 0;
        if (ImGui::Checkbox(T(S::stub_imports), &on)) {
            if (on) s["BB_STUB_IMPORTS"] = "1";
            else s.erase("BB_STUB_IMPORTS");
            changed = true;
        }
    }
    hint(T(S::stub_imports_hint));

    ImGui::Separator();
    if (ImGui::Button(T(S::reset_defaults))) {
        const std::string dlc = s.count("BB_DLC_DIR") ? s["BB_DLC_DIR"] : "";  // where the add-on lives is not a preference
        s = default_settings();
        if (!dlc.empty()) s["BB_DLC_DIR"] = dlc;
        a.force_custom_res = false;
        changed = true;
    }
    if (changed) a.save();
}

void cheats_tab(App& a) {
    Game* g = a.sel();
    if (!g) {
        hint(T(S::no_game_selected));
        return;
    }
    Settings& s = g->settings;
    ImGui::Text("%s%s", T(S::settings_for), g->name.c_str());
    hint(T(S::cheats_note));
    std::string list = s.count("BB_CHEATS") ? s["BB_CHEATS"] : "";
    bool changed = false;
    std::string out;
    for (const Cheat& c : kCheats) {
        bool on = ("," + list + ",").find(std::string(",") + c.name + ",") != std::string::npos;
        if (ImGui::Checkbox(T(c.label), &on)) changed = true;
        if (on) out += (out.empty() ? "" : ",") + std::string(c.name);
        if (auto it = a.verdicts.find(c.name); it != a.verdicts.end()) {
            ImGui::SameLine();
            ImGui::TextColored(it->second ? kYellow : kGreen, "(%s)", it->second ? T(S::cheat_unsupported) : T(S::cheat_active));
        }
        hint(T(c.desc));
    }
    if (a.verdicts.empty()) hint(T(S::cheats_before_run));
    if (changed) {
        if (out.empty()) s.erase("BB_CHEATS");
        else s["BB_CHEATS"] = out;
        a.save();
    }
}

// "From PKG files": three pickers, the external extractor command, and the extraction progress.
void pkg_section(App& a) {
    ImGui::TextUnformatted(T(S::pkg_section));
    hint(T(S::pkg_note));
    static const S kHint[] = {S::pkg_game, S::pkg_dlc, S::pkg_update};
    const bool running = a.pkgjob && a.pkgjob->job.valid() && !a.pkgjob->finished;
    ImGui::BeginDisabled(running);
    for (int i = 0; i < 3; ++i) {
        ImGui::PushID(i);
        PkgSlot& p = a.pkg[i];
        ImGui::SetNextItemWidth(-ImGui::GetFontSize() * 9);
        ImGui::InputTextWithHint("##pkg", T(kHint[i]), p.path, sizeof p.path);
        if (ImGui::IsItemDeactivatedAfterEdit()) a.save();
        ImGui::SameLine();
        if (ImGui::Button(T(S::browse))) a.browse_file(6 + i, true);
        p.refresh();
        if (p.path[0]) {
            if (!p.ok) ImGui::TextColored(kRed, "%s", p.err.c_str());
            else ImGui::TextDisabled("%s%s%s", p.info.content_id.c_str(), p.info.title_id.empty() ? "" : "  (title ", p.info.title_id.empty() ? "" : (p.info.title_id + ")").c_str());
        }
        ImGui::PopID();
    }
    const fs::path dlc_path = a.pkg[1].path, upd_path = a.pkg[2].path;
    for (const auto& f : [&] {
             std::vector<bb::dump::Finding> v;
             if (a.pkg[0].path[0] && a.pkg[0].ok)
                 v = check_pkg_set(a.pkg[0].info, a.pkg[0].path, a.pkg[1].path[0] && a.pkg[1].ok ? &a.pkg[1].info : nullptr, &dlc_path,
                                   a.pkg[2].path[0] && a.pkg[2].ok ? &a.pkg[2].info : nullptr, &upd_path);
             return v;
         }())
        if (f.severity == Severity::Error) ImGui::TextColored(kRed, "%s", f.message.c_str());
    ImGui::TextUnformatted(T(S::pkg_extractor));
    ImGui::SetNextItemWidth(-ImGui::GetFontSize() * 9);
    ImGui::InputTextWithHint("##extractor", "python C:\\path\\tool.py {pkg} {out}", a.extractor, sizeof a.extractor);
    if (ImGui::IsItemDeactivatedAfterEdit()) a.save();
    ImGui::SameLine();
    if (ImGui::Button((std::string(T(S::browse)) + "##ext").c_str())) a.browse_file(9, false);
    ImGui::EndDisabled();
    hint(T(S::pkg_extractor_hint));
    if (!running) {
        if (ImGui::Button(T(S::pkg_extract)) || a.auto_extract) a.start_extract(), a.auto_extract = false;
        ImGui::SameLine();
        ImGui::TextDisabled("%s", T(S::pkg_space_note));
    } else {
        if (ImGui::Button(T(S::cancel))) a.pkgjob->cancel = true;
        std::string status;
        fs::path log;
        {
            std::lock_guard l(a.pkgjob->m);
            status = a.pkgjob->status, log = a.pkgjob->log;
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(status.c_str());
        if (SDL_GetTicks() > a.pkg_tail_at) {  // the live tail of the extractor output, twice a second
            a.pkg_tail_at = SDL_GetTicks() + 500;
            a.pkg_tail_text.clear();
            for (const std::string& l : tail_lines(log, 4)) a.pkg_tail_text += l.substr(0, 160) + "\n";
        }
        ImGui::TextDisabled("%s", a.pkg_tail_text.c_str());
    }
    if (!a.pkg_msg.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, kRed);
        ImGui::TextWrapped("%s", a.pkg_msg.c_str());
        ImGui::PopStyleColor();
    }
    if (a.pkgjob && a.pkgjob->finished) {
        std::lock_guard l(a.pkgjob->m);
        if (!a.pkgjob->ok) {
            ImGui::PushStyleColor(ImGuiCol_Text, kRed);
            ImGui::TextWrapped("%s", a.pkgjob->error.c_str());
            ImGui::PopStyleColor();
        }
        else ImGui::TextColored(kGreen, "%s", T(S::pkg_done));
    }
}

void add_popup(App& a) {
    if (a.open_add) ImGui::OpenPopup(T(S::add_title)), a.open_add = false;
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(ds.x * 0.5f, ds.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(ds.x * 0.9f, ds.y * 0.88f));
    if (!ImGui::BeginPopupModal(T(S::add_title), nullptr, ImGuiWindowFlags_NoSavedSettings)) return;
    pkg_section(a);
    ImGui::Separator();
    // Folder fields stay locked while an extraction runs (its result is about to fill them) and while a check/install is in progress.
    const bool extracting = a.pkgjob && a.pkgjob->job.valid() && !a.pkgjob->finished;
    ImGui::BeginDisabled(extracting);
    ImGui::TextUnformatted(T(S::folders_section));
    hint(T(S::add_hint));
    ImGui::SetNextItemWidth(-ImGui::GetFontSize() * 16);
    ImGui::InputText("##path", a.add_path, sizeof a.add_path);
    ImGui::SameLine();
    if (ImGui::Button(T(S::browse))) a.browse(1);
    ImGui::SameLine();
    if (ImGui::Button(T(S::check)) && a.add_path[0]) a.start_check();
    ImGui::SetNextItemWidth(-ImGui::GetFontSize() * 16);
    ImGui::InputTextWithHint("##dlc", T(S::dlc_folder), a.add_dlc, sizeof a.add_dlc);
    ImGui::SameLine();
    if (ImGui::Button((std::string(T(S::browse)) + "##dlc").c_str())) a.browse(4);
    ImGui::TextDisabled("%s", T(S::dlc_hint));
    ImGui::SetNextItemWidth(-ImGui::GetFontSize() * 16);
    ImGui::InputTextWithHint("##upd", T(S::update_folder), a.add_update, sizeof a.add_update);
    ImGui::SameLine();
    if (ImGui::Button((std::string(T(S::browse)) + "##upd").c_str())) a.browse(5);
    ImGui::TextDisabled("%s", T(S::update_hint));
    ImGui::EndDisabled();
    ImGui::Separator();
    AddState* st = a.add.get();
    bool close = false;
    if (st) {
        if (st->job.valid() && st->job.wait_for(std::chrono::seconds(0)) == std::future_status::ready) st->job.get();
        const bool working = st->job.valid();
        const bool pending_work = st->phase == AddState::Checking || st->phase == AddState::Installing;
        ImGui::BeginChild("report", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 2.6f));
        if (pending_work || working) {
            std::string cur;
            {
                std::lock_guard l(st->m);
                cur = st->current;
            }
            ImGui::TextUnformatted(st->phase == AddState::Installing ? T(S::installing) : T(S::checking));
            const size_t tot = st->total.load();
            ImGui::ProgressBar(tot ? float(st->done.load()) / float(tot) : 0.f, ImVec2(-1, 0));
            ImGui::TextDisabled("%s", cur.c_str());
        }
        if (st->phase != AddState::Checking) {
            std::lock_guard l(st->m);
            if (!st->title_id.empty() && !st->update.empty()) {
                ImGui::TextColored(st->report.ok() ? kGreen : kYellow, "%s%s %s v%s + update v%s -> v%s after merge", T(S::version_recognised), st->title.c_str(),
                                   st->title_id.c_str(), st->app_ver.c_str(), st->upd_ver.c_str(), bb::dump::kRequiredAppVer);
            } else if (!st->title_id.empty()) {
                ImGui::TextColored(st->app_ver == bb::dump::kRequiredAppVer ? kGreen : kYellow, "%s%s %s v%s", T(S::version_recognised), st->title.c_str(),
                                   st->title_id.c_str(), st->app_ver.c_str());
            }
            ImGui::TextDisabled("%s%s", T(S::manifest_used), st->manifest.empty() ? T(S::manifest_none) : st->manifest.string().c_str());
            const double gib = double(1u << 30);
            if (st->need) {
                char buf[512];
                std::snprintf(buf, sizeof buf, T(S::need_space), double(st->need) / gib, st->avail ? double(*st->avail) / gib : 0.0, (a.data_dir / "games").string().c_str());
                ImGui::TextDisabled("%s", buf);
            }
            if (!st->hint.empty()) ImGui::TextColored(kYellow, "%s", st->hint.c_str());
            ImGui::Spacing();
            size_t shown = 0;
            for (const auto& f : st->report.findings) {
                if (++shown > 60) {
                    ImGui::TextDisabled("... %zu more", st->report.findings.size() - 60);
                    break;
                }
                sev_text(f.severity, f.message);
            }
            for (const auto& f : st->log) sev_text(f.severity, f.message);
            if (st->phase == AddState::Installed) ImGui::TextColored(kGreen, "%s", T(S::install_done));
        }
        ImGui::EndChild();
        const auto extracted = [&](const AddState* x) { return is_under(x->dump, a.data_dir / "pkg_work") || (!x->dlc.empty() && is_under(x->dlc, a.data_dir / "pkg_work")); };
        const bool checked = st->phase == AddState::Checked, failed_install = st->phase == AddState::Failed;
        const bool can_install = (checked || failed_install) && st->report.ok() && !working && (!st->avail || *st->avail >= st->need);
        ImGui::BeginDisabled(!can_install);
        if (ImGui::Button(T(S::install)) || (a.auto_install && can_install)) a.start_install();
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", T(S::install_hint));
        ImGui::SameLine();
        const bool can_place = (checked || failed_install) && st->has_eboot && !working && st->update.empty() && !extracted(st);  // an update can only be merged by Install; extracted files are temporary
        ImGui::BeginDisabled(!can_place);
        if (ImGui::Button(st->report.ok() ? T(S::use_in_place) : T(S::use_in_place_unverified))) {
            a.add_game(st->dump, st->dlc);
            close = true;
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", !st->update.empty() ? T(S::use_in_place_update) : extracted(st) ? T(S::use_in_place_pkg) : T(S::use_in_place_hint));
        ImGui::SameLine();
        if (st->phase == AddState::Installed && st->registered) close = true;  // registered by poll_jobs
        if (st->phase == AddState::Installing && ImGui::Button(T(S::cancel))) st->cancel = true;
    }
    ImGui::SameLine();
    if (!(st && st->phase == AddState::Installing) && ImGui::Button(T(S::close))) close = true;
    if (close) {
        if (st) {
            st->cancel = true;
            if (st->job.valid()) st->job.wait();
        }
        a.add.reset();
        a.pending_check = false;
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

// Asks before removing the extraction folder of the last PKG import (only that folder: remove_work_dir checks its marker and parent).
void delete_popup(App& a) {
    if (a.show_delete) ImGui::OpenPopup(T(S::pkg_delete)), a.show_delete = false;
    if (!ImGui::BeginPopupModal(T(S::pkg_delete), nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings)) return;
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 34);
    ImGui::TextWrapped("%s", T(S::pkg_delete_confirm));
    ImGui::TextWrapped("%s", a.delete_target.string().c_str());
    ImGui::PopTextWrapPos();
    if (ImGui::Button(T(S::yes))) {
        std::string err;
        a.pkg_msg.clear();
        if (!remove_work_dir(a.delete_target, a.data_dir, err)) a.pkg_msg = err;
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button(T(S::no))) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

void frame(App& a, int tab_override) {
    a.poll_jobs();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("##main", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
    // L1/R1 (gamepad shoulder buttons) switch tabs.
    if (ImGui::IsKeyPressed(ImGuiKey_GamepadL1, false)) a.want_tab = (a.shown_tab + 2) % 3;
    if (ImGui::IsKeyPressed(ImGuiKey_GamepadR1, false)) a.want_tab = (a.shown_tab + 1) % 3;
    if (tab_override >= 0) a.want_tab = tab_override;
    const float footer = ImGui::GetFrameHeightWithSpacing() * 3.0f;
    if (ImGui::BeginTabBar("tabs")) {
        const S names[] = {S::tab_games, S::tab_settings, S::tab_cheats};
        for (int i = 0; i < 3; ++i) {
            if (ImGui::BeginTabItem(T(names[i]), nullptr, a.want_tab == i ? ImGuiTabItemFlags_SetSelected : 0)) {
                a.shown_tab = i;
                ImGui::BeginChild("content", ImVec2(0, -footer), ImGuiChildFlags_None);
                if (i == 0) games_tab(a);
                else if (i == 1) settings_tab(a);
                else cheats_tab(a);
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
        }
        a.want_tab = -1;
        ImGui::EndTabBar();
    }
    // footer: status on the left, Play on the right
    ImGui::Separator();
    Game* g = a.sel();
    const Build* b = g ? a.build_for(*g) : nullptr;
    const float bw = ImGui::GetFontSize() * 9;
    ImGui::BeginGroup();
    if (!a.status.empty()) ImGui::TextColored(kRed, "%s", a.status.c_str());
    else if (a.proc.valid) ImGui::TextUnformatted(T(S::running));
    else if (!g) ImGui::TextUnformatted(T(S::no_game_selected));
    else if (!b) ImGui::TextColored(kYellow, "%s", g->eboot_sha.empty() ? T(S::hashing) : T(S::status_no_build));
    else if (!b->error.empty()) ImGui::TextColored(kRed, "%s", b->error.c_str());
    else ImGui::TextColored(kGreen, "%s - %s", g->name.c_str(), T(S::status_ready));
    ImGui::EndGroup();
    ImGui::SameLine(ImGui::GetWindowWidth() - bw - ImGui::GetStyle().WindowPadding.x);
    if (a.proc.valid) {
        if (ImGui::Button(T(S::stop), ImVec2(bw, ImGui::GetFrameHeight() * 1.6f))) terminate(a.proc), a.has_exit = true, a.last_exit = 1, a.reload_log();
    } else {
        ImGui::BeginDisabled(!b || !b->error.empty());
        if (ImGui::Button(T(S::play), ImVec2(bw, ImGui::GetFrameHeight() * 1.6f))) a.play();
        ImGui::EndDisabled();
    }
    add_popup(a);
    delete_popup(a);
    ImGui::End();
}

// ---- command line --------------------------------------------------------------------------------------------------------------------
struct Cli {
    bool dry_run = false, run = false, validate = false, auto_install = false, auto_extract = false, open_add = false, show_delete = false;
    double auto_play = 0;  // --auto-play <secs>: press Play, stop after secs (tests)
    double secs = 60;
    std::string shot, game, add, add_dlc, add_update, add_pkg, add_dlc_pkg, add_update_pkg, extractor, config_dir, manifest, validate_dir;
    int tab = -1, w = 1100, h = 700;
    std::vector<fs::path> builds;
};

Cli parse_cli(int argc, char** argv) {
    Cli c;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--dry-run") c.dry_run = true;
        else if (a == "--run") c.run = true, c.secs = std::atof(next().c_str());
        else if (a == "--validate") c.validate = true, c.validate_dir = next();
        else if (a == "--manifest") c.manifest = next();
        else if (a == "--shot") c.shot = next();
        else if (a == "--tab") {
            const std::string t = next();
            c.tab = t == "games" ? 0 : t == "settings" ? 1 : t == "cheats" ? 2 : std::atoi(t.c_str());
        } else if (a == "--size") std::sscanf(next().c_str(), "%dx%d", &c.w, &c.h);
        else if (a == "--game") c.game = next();
        else if (a == "--add") c.add = next();
        else if (a == "--add-dlc") c.add_dlc = next();
        else if (a == "--add-update") c.add_update = next();
        else if (a == "--add-pkg") c.add_pkg = next();
        else if (a == "--add-dlc-pkg") c.add_dlc_pkg = next();
        else if (a == "--add-update-pkg") c.add_update_pkg = next();
        else if (a == "--extractor") c.extractor = next();
        else if (a == "--auto-extract") c.auto_extract = true;
        else if (a == "--open-add") c.open_add = true;
        else if (a == "--show-delete") c.show_delete = true;
        else if (a == "--auto-install") c.auto_install = true;
        else if (a == "--auto-play") c.auto_play = std::atof(next().c_str());
        else if (a == "--config-dir") c.config_dir = next();
        else if (a == "--builds") c.builds.emplace_back(next());
    }
    return c;
}

void attach_console() {
#ifdef _WIN32
    // GUI-subsystem exe: CLI output goes to the parent console unless stdout is already redirected.
    if (GetStdHandle(STD_OUTPUT_HANDLE) == nullptr && AttachConsole(ATTACH_PARENT_PROCESS)) {
        std::freopen("CONOUT$", "w", stdout);
        std::freopen("CONOUT$", "w", stderr);
    }
#endif
}

void print_progress(size_t done, size_t total, const std::string& path) {
    if (done == total || done % 500 == 0) std::fprintf(stderr, "\r[%zu/%zu] %-50.50s", done, total, path.c_str());
}

int run_cli(App& a, const Cli& cli) {
    attach_console();
    if (cli.validate) {
        bb::dump::Source src{cli.validate_dir, std::nullopt, std::nullopt, std::nullopt};
        bb::dump::Report pre;
        bb::dump::check_structure(src.dump, pre);
        if (!cli.manifest.empty()) src.manifest = fs::path(cli.manifest);
        else if (auto m = find_manifest(pre.title_id, pre.app_ver); !m.empty()) src.manifest = m;
        const auto rep = bb::dump::validate(src, print_progress);
        std::fputc('\n', stderr);
        for (const auto& f : rep.findings)
            std::printf("%s %s\n", f.severity == Severity::Error ? "ERROR" : f.severity == Severity::Warning ? "WARN " : "info ", f.message.c_str());
        std::printf("%s\n", rep.ok() ? "Dump OK." : "Dump INVALID.");
        return rep.ok() ? 0 : 1;
    }
    Game* g = nullptr;
    for (Game& x : a.cfg.games)
        if (cli.game.empty() ? x.id == a.cfg.selected || (a.cfg.selected.empty() && &x == &a.cfg.games.front()) : x.id == cli.game) g = &x;
    if (!g) {
        std::fprintf(stderr, "no such game in %s\n", a.cfg_file.string().c_str());
        return 2;
    }
    ensure_eboot_hash(*g);
    std::vector<fs::path> dirs{bb::exe_dir()};
    for (const std::string& d : a.cfg.build_dirs) dirs.emplace_back(d);
    dirs.insert(dirs.end(), cli.builds.begin(), cli.builds.end());
    const auto builds = scan_builds(dirs, a.cfg_dir);
    const Build* b = match_build(builds, g->eboot_sha);
    if (!b) {
        std::printf("%s (%s): %s\nbuilds found: %zu\n", g->name.c_str(), g->eboot_sha.c_str(), T(S::build_none), builds.size());
        return 3;
    }
    const Plan p = make_plan(*g, *b, a.cfg_dir);
    std::printf("game:   %s (%s), eboot sha256 %s\nbuild:  %s (hash from %s)\n%s", g->id.c_str(), g->name.c_str(), g->eboot_sha.c_str(), b->exe.string().c_str(),
                b->source.c_str(), describe(p).c_str());
    if (cli.dry_run) return 0;
    std::string err;
    Proc pr;
    if (!write_plan_files(p, err) || !spawn(p, pr, err)) {
        std::fprintf(stderr, "%s%s\n", T(S::status_error), err.c_str());
        return 1;
    }
    int code = -1;
    const auto t0 = std::chrono::steady_clock::now();
    while (!poll(pr, code) && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < cli.secs) SDL_Delay(500);
    if (pr.valid) terminate(pr), std::printf("stopped after %.0f s\n", cli.secs);
    else std::printf("%s %d\n", T(S::exit_code), code);
    std::printf("--- last lines of %s ---\n", p.log_file.string().c_str());
    for (const std::string& l : tail_lines(p.log_file, 12)) std::printf("%s\n", l.c_str());
    std::ifstream f(p.log_file);
    for (const auto& [n, un] : cheat_verdicts(f)) std::printf("cheat %s: %s\n", n.c_str(), un ? T(S::cheat_unsupported) : T(S::cheat_active));
    return 0;
}

} // namespace
} // namespace bbl

int main(int argc, char** argv) {
    using namespace bbl;
    const Cli cli = parse_cli(argc, argv);
    App app;
    app.cfg_dir = cli.config_dir.empty() ? bb::config_dir() : fs::path(cli.config_dir);
    app.data_dir = cli.config_dir.empty() ? bb::dump::default_install_dir() : fs::path(cli.config_dir);
    app.cfg_file = app.cfg_dir / "launcher.ini";
    load_config(app.cfg_file, app.cfg);  // no file = empty config, created on the first change
    if (cli.dry_run || cli.run || cli.validate) return run_cli(app, cli);
    app.extra_build_dirs = cli.builds;
    app.auto_install = cli.auto_install;
    app.load_pkg_fields();
    app.auto_extract = cli.auto_extract;
    if (cli.show_delete)
        if (const auto w = list_work_dirs(app.data_dir); !w.empty()) app.delete_target = w.front(), app.show_delete = true;
    for (const auto& [slot, v] : {std::pair<int, const std::string&>{0, cli.add_pkg}, {1, cli.add_dlc_pkg}, {2, cli.add_update_pkg}})
        if (!v.empty()) std::snprintf(app.pkg[slot].path, sizeof app.pkg[0].path, "%s", v.c_str());
    if (!cli.extractor.empty()) std::snprintf(app.extractor, sizeof app.extractor, "%s", cli.extractor.c_str());
    if (cli.open_add || !cli.add_pkg.empty()) app.open_add = true;
    if (!cli.game.empty()) app.cfg.selected = cli.game;
    if (app.cfg.selected.empty() && !app.cfg.games.empty()) app.cfg.selected = app.cfg.games.front().id;

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    if (!SDL_CreateWindowAndRenderer(T(S::window_title), cli.w, cli.h, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY, &window, &renderer)) {
        std::fprintf(stderr, "cannot create a window: %s\n", SDL_GetError());
        return 1;
    }
    SDL_SetWindowMinimumSize(window, 640, 400);
    SDL_SetRenderVSync(renderer, 1);
    app.window = window;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = style.FrameRounding = style.TabRounding = 4;
    style.FontSizeBase = 17;
    for (const char* ttf : {"C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"})
        if (fs::exists(ttf) && io.Fonts->AddFontFromFileTTF(ttf)) break;
    if (io.Fonts->Fonts.empty()) io.Fonts->AddFontDefault();
    ImGui_ImplSDL3_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer3_Init(renderer);

    app.start_scan();
    app.start_hash();
    app.reload_log();
    if (!cli.add.empty()) {
        std::snprintf(app.add_path, sizeof app.add_path, "%s", cli.add.c_str());
        std::snprintf(app.add_dlc, sizeof app.add_dlc, "%s", cli.add_dlc.c_str());
        std::snprintf(app.add_update, sizeof app.add_update, "%s", cli.add_update.c_str());
        app.start_check();
    }

    bool quit = false;
    int frames = 0;
    double play_at = -1;
    int settled_frames = 0;
    while (!quit) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL3_ProcessEvent(&e);
            if (e.type == SDL_EVENT_QUIT) quit = true;
        }
        // text and widgets scale with the window (relative to the 1100x700 design size) and with the display scale
        int w = 0, h = 0;
        SDL_GetWindowSize(window, &w, &h);
        style.FontScaleMain = std::clamp(std::min(float(w) / 1100.f, float(h) / 700.f), 0.8f, 3.0f);
        style.FontScaleDpi = SDL_GetWindowDisplayScale(window);
        if (cli.auto_play > 0) {
            const double now = double(SDL_GetTicks()) / 1000;
            if (play_at < 0 && app.sel() && app.build_for(*app.sel()) && !app.busy()) app.play(), play_at = now;
            if (play_at >= 0 && app.proc.valid && now - play_at > cli.auto_play) terminate(app.proc), app.has_exit = true, app.last_exit = 1, app.reload_log();
        }
        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        frame(app, frames < 2 || !cli.shot.empty() ? cli.tab : -1);
        ImGui::Render();
        SDL_SetRenderDrawColor(renderer, 24, 24, 28, 255);
        SDL_RenderClear(renderer);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer);
        ++frames;
        // Screenshot once everything has settled for a few frames (the read-back can lag one presented frame behind).
        if (!cli.shot.empty() && frames >= 8) {
            const bool settled = !app.busy() && (!app.add || !app.add->job.valid()) && !(app.pkgjob && app.pkgjob->job.valid() && !app.pkgjob->handled) && !app.auto_extract && !app.proc.valid && (cli.auto_play <= 0 || play_at >= 0);
            settled_frames = settled ? settled_frames + 1 : 0;
            if (settled_frames >= 4 || frames > 3000) {
                SDL_Surface* s = SDL_RenderReadPixels(renderer, nullptr);
                if (!s) {
                    std::fprintf(stderr, "screenshot failed: %s\n", SDL_GetError());
                    return 1;
                }
                SDL_Surface* rgba = SDL_ConvertSurface(s, SDL_PIXELFORMAT_RGBA32);
                const bool ok = rgba && write_png(cli.shot.c_str(), static_cast<const uint8_t*>(rgba->pixels), rgba->w, rgba->h);
                SDL_DestroySurface(rgba);
                SDL_DestroySurface(s);
                std::printf("%s %s\n", ok ? "wrote" : "FAILED", cli.shot.c_str());
                quit = true;
            }
        }
        SDL_RenderPresent(renderer);
    }
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
