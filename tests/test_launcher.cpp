// SPDX-License-Identifier: GPL-3.0-or-later
// Launcher logic: launcher.ini round trip, bbconfig.ini parser/env override, build matching, dry-run plan, process start + log capture.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <thread>

#include "core.h"
#include "core/bbconfig.h"
#include "core/install.h"
#include "pkg.h"
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

using namespace bbl;

static int failures = 0;
#define CHECK(c)                                                     \
    do {                                                             \
        if (!(c)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                              \
        }                                                            \
    } while (0)

static void write(const fs::path& p, const std::string& s) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    std::ofstream(p, std::ios::binary) << s;
}

static const std::string kShaA(64, 'a'), kShaB = std::string(63, 'b') + "c";

int main(int argc, char** argv) {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);  // no system dialog from this test or its children (spawn() adds CREATE_NO_WINDOW)
#endif
    if (argc > 3 && std::string(argv[1]).rfind("--extract-", 0) == 0) {  // stands in for a PKG extractor: <mode> <pkg> <out>
        const std::string mode = argv[1], pkg = argv[argc - 2];
        const fs::path out = argv[argc - 1];
        if (mode == "--extract-sleep") {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            return 0;
        }
        if (mode == "--extract-fail") {
            std::fprintf(stderr, "boom: cannot decode this package\n");
            return 3;
        }
        if (mode == "--extract-fixture") {
            write(out / "Image0" / "eboot.bin", "elf");
            write(out / "Sc0" / "param.sfo", "sfo");
            write(out / "pkg_seen.txt", pkg);  // the {pkg} argument exactly as received
        }
        std::printf("extracted %s\n", pkg.c_str());  // --extract-empty: output but no files
        return 0;
    }
    if (argc > 1 && (std::string(argv[1]) == "--child" || std::getenv("BB_TEST_CHILD"))) {  // stands in for bbgame: echoes what the launcher gave it
        const char* cfg = std::getenv("BB_CONFIG");
        std::printf("child config=%s\nchild arg=%s\ncheats: god unavailable on this build\ncheats: nohit stamina active\ncheats: ammo,god active\n", cfg ? cfg : "(none)", argv[argc - 1]);
        std::fprintf(stderr, "child stderr line\nfatal: guest memory could not be reserved (demo failure)\n");
        return 7;
    }
    const fs::path tmp = fs::temp_directory_path() / "bbl_test";
    std::error_code ec;
    fs::remove_all(tmp, ec);
    fs::create_directories(tmp);

    // --- bbconfig.ini parser (the game side) ---
    {
        const auto e = bb::parse_config("# comment\r\nBB_FPS = 144 \nnot_bb=1\n\nBB_WINDOW=1920x1080\r\nBB_CHEATS=god,nohit\nno equals here\n=x\n");
        CHECK(e.size() == 3);
        CHECK(e.size() == 3 && e[0].first == "BB_FPS" && e[0].second == "144");
        CHECK(e.size() == 3 && e[1].second == "1920x1080");
        CHECK(e.size() == 3 && e[2].second == "god,nohit");
        // the environment wins over the file; unset keys are filled in
#ifdef _WIN32
        _putenv_s("BB_TEST_PRESET", "env");
        _putenv_s("BB_TEST_FRESH", "");
#else
        setenv("BB_TEST_PRESET", "env", 1);
        unsetenv("BB_TEST_FRESH");
#endif
        const int n = bb::apply_config({{"BB_TEST_PRESET", "file"}, {"BB_TEST_FRESH", "file"}});
        CHECK(std::string(std::getenv("BB_TEST_PRESET")) == "env");
        CHECK(n == 1 || n == 0);  // Windows cannot unset via _putenv_s(""): the empty value counts as set
        // BB_CONFIG selects the file
        write(tmp / "cfg" / "x.ini", "BB_TEST_VIA_FILE=42\n");
        const std::string p = (tmp / "cfg" / "x.ini").string();
#ifdef _WIN32
        _putenv_s("BB_CONFIG", p.c_str());
#else
        setenv("BB_CONFIG", p.c_str(), 1);
#endif
        const std::string line = bb::load_config_into_env(tmp);
        CHECK(line.find("(1 keys)") != std::string::npos);
        CHECK(std::getenv("BB_TEST_VIA_FILE") && std::string(std::getenv("BB_TEST_VIA_FILE")) == "42");
    }

    // --- launcher.ini round trip, unknown keys, defaults for missing keys ---
    Config c;
    c.selected = "CUSA03173-01.00";
    c.build_dirs = {"C:/some/where", "/other"};
    c.pkg_extractor = "python \"C:/my tools/x.py\" {pkg} {out}", c.pkg_game = "D:/pkgs/game.pkg", c.pkg_dlc = "D:/pkgs/dlc.pkg", c.pkg_update = "D:/pkgs/upd.pkg";
    Game g;
    g.id = "CUSA03173-01.00", g.name = "Bloodborne", g.path = "D:/dump/Image0", g.title_id = "CUSA03173", g.app_ver = "01.00";
    g.eboot_sha = kShaA, g.eboot_size = 123456789012ull, g.eboot_mtime = 1700000000;
    g.settings["BB_FPS"] = "165";
    g.settings["BB_CHEATS"] = "god,nohit";
    g.settings["BB_STUB_IMPORTS"] = "1";
    g.settings["BB_SAVE_DIR"] = "E:/saves with space";
    c.games.push_back(g);
    Game g2 = g;
    g2.id = "CUSA03173-01.09", g2.name = "Bloodborne 1.09", g2.settings = default_settings();
    c.games.push_back(g2);
    const std::string text = write_config(c);
    CHECK(parse_config(text) == c);
    const fs::path ini = tmp / "launcher.ini";
    CHECK(save_config(ini, c));
    Config back;
    CHECK(load_config(ini, back) && back == c);
    Config none;
    CHECK(!load_config(tmp / "missing.ini", none) && none.games.empty());
    Config tol = parse_config("version=99\nfuture_key=1\n[game]\nid=x\npath=/p\nunknown=1\nBB_FPS=30\n[game]\nname=no id, dropped\n");
    CHECK(tol.games.size() == 1 && tol.version == 99);
    CHECK(tol.games.size() == 1 && tol.games[0].settings["BB_FPS"] == "30" && tol.games[0].settings["BB_WINDOW"] == "1920x1080" && !tol.games[0].settings.count("BB_CHEATS"));
    CHECK(make_game_id("CUSA03173", "01.00", c.games) == "CUSA03173-01.00-2");
    CHECK(make_game_id("CUSA03173", "01.09", {}) == "CUSA03173-01.09");

    // --- executable validator: junk must never reach CreateProcess ---
    {
        std::string e;
        write(tmp / "v" / "text.exe", "this is not a program");
        write(tmp / "v" / "empty.exe", "");
        CHECK(!valid_executable(tmp / "v" / "text.exe", e) && !e.empty());
        CHECK(!valid_executable(tmp / "v" / "empty.exe", e));
        CHECK(!valid_executable(tmp / "v" / "missing.exe", e));
        CHECK(valid_executable(argv[0], e));
        Plan junk;
        junk.exe = tmp / "v" / "text.exe", junk.log_file = tmp / "v" / "junk.log";
        Proc jp;
        CHECK(!spawn(junk, jp, e) && !jp.valid && e.find("not a valid 64-bit program") != std::string::npos);
    }

    // --- update merge size: base 3 files (a 1, b 2, c 4 bytes), update 2 files (b 20 replaces, d 8 new) -> 1 + 20 + 4 + 8 = 33 bytes, 2 base-only, 2 from update, 1 replaced ---
    {
        write(tmp / "mb" / "Image0" / "a", "a");
        write(tmp / "mb" / "Image0" / "b", "bb");
        write(tmp / "mb" / "Image0" / "c", "cccc");
        write(tmp / "mu" / "Image0" / "b", std::string(20, 'u'));
        write(tmp / "mu" / "Image0" / "d", std::string(8, 'd'));
        write(tmp / "mb" / "Sc0" / "param.sfo", "x");  // PKG-extractor layout markers (resolve_layout looks for Sc0/param.sfo)
        write(tmp / "mu" / "Sc0" / "param.sfo", "y");
        const auto m = bb::dump::merge_stats(tmp / "mb", tmp / "mu");
        CHECK(m.base_only == 2 && m.from_update == 2 && m.overridden == 1 && m.bytes == 33);
        bb::dump::Source ms{tmp / "mb", std::nullopt, std::nullopt, tmp / "mu"};
        CHECK(bb::dump::install_bytes(ms) == 33 + 28);  // peak = max(base 7, merged 33) + the update itself (20 + 8)
    }

    // --- build matching: fresh sidecar, games.ini, no hash ---
    const fs::path bdir = tmp / "builds";
#ifdef _WIN32
    const std::string ext = ".exe";
#else
    const std::string ext;
#endif
    write(bdir / ("bbgame" + ext), "x");
    write(bdir / ("bbgame" + ext + ".eboot.sha256"), kShaA + "\n");
    write(bdir / ("bbgame_other" + ext), "x");
    write(bdir / "games.ini", "# demo\nbbgame_other" + ext + " = " + kShaB + "\nbbgame" + ext + " = " + std::string(64, '0') + "\n");
    write(bdir / ("bbgame_none" + ext), "x");
    write(bdir / ("notabuild" + ext), "x");
    const auto builds = scan_builds({bdir}, tmp);
    CHECK(builds.size() == 3);
    const Build* a = match_build(builds, kShaA);
    CHECK(a && a->exe.filename() == "bbgame" + ext && a->source == "sidecar");  // sidecar wins over games.ini
    const Build* b = match_build(builds, kShaB);
    CHECK(b && b->exe.filename() == "bbgame_other" + ext && b->source == "games.ini");
    CHECK(!match_build(builds, std::string(64, 'f')) && !match_build(builds, ""));

    // --- dry-run plan for a sample profile ---
    const Build fake{"C:/b/bbgame.exe", kShaA, "sidecar", {}};
    const Plan plan = make_plan(g, fake, tmp / "cfgdir");
    CHECK(plan.args.size() == 1 && plan.args[0] == (fs::absolute("D:/dump/Image0") / "eboot.bin").string());
    CHECK(plan.config_file == tmp / "cfgdir" / "profiles" / g.id / "bbconfig.ini");
    CHECK(plan.cwd == tmp / "cfgdir" / "profiles" / g.id);
    const std::string want = "BB_CHEATS=god,nohit\nBB_FPS=165\nBB_FSR_SHARP=0.2\nBB_INTERP=1\nBB_RES_SCALE=1\nBB_SAVE_DIR=E:/saves with space\n"
                             "BB_STUB_IMPORTS=1\nBB_UPSCALE=fsr\nBB_WINDOW=1920x1080\n";
    CHECK(plan.config_text.substr(plan.config_text.find('\n') + 1) == want);
    const Plan def = make_plan(g2, fake, tmp / "cfgdir");
    CHECK(def.config_text.find("BB_SAVE_DIR=" + (def.cwd / "savedata").string() + "\n") != std::string::npos);  // empty save dir = inside the profile
    CHECK(def.config_text.find("BB_CHEATS") == std::string::npos);                                                // off = not written at all
    Game gd = g;
    gd.settings["BB_DLC_DIR"] = "E:/dlc/The Old Hunters";
    const Plan pd = make_plan(gd, fake, tmp / "cfgdir");
    CHECK(pd.config_text.find("BB_DLC_DIR=E:/dlc/The Old Hunters\n") != std::string::npos);  // in-place game with a DLC folder
    CHECK(plan.config_text.find("BB_DLC_DIR") == std::string::npos);                          // none = not written (the game looks at <app0>/../dlc)
    CHECK(parse_config(write_config([&] { Config cc; cc.selected = gd.id, cc.games = {gd}; return cc; }())).games[0].settings["BB_DLC_DIR"] == "E:/dlc/The Old Hunters");
    CHECK(describe(plan).find("BB_CONFIG=" + plan.config_file.string()) != std::string::npos);
    // the file the game reads parses back to the same settings
    const auto entries = bb::parse_config(plan.config_text);
    CHECK(entries.size() == 9);
    CHECK(bb::parse_config(pd.config_text).size() == 10);  // BB_DLC_DIR is a plain BB_ key: the game applies it via bbconfig like the others

    // --- real process: this test executable as a stand-in for bbgame ---
    Plan run;
    run.exe = argv[0];
    run.args = {"--child", "arg with space"};
    run.cwd = tmp;
    run.config_file = tmp / "run" / "bbconfig.ini";
    run.log_file = tmp / "run" / "game.log";
    run.config_text = "BB_FPS=60\n";
    std::string err;
    CHECK(write_plan_files(run, err));
    Proc pr;
    CHECK(spawn(run, pr, err));
    int code = -1;
    for (int i = 0; i < 200 && !poll(pr, code); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(!pr.valid && code == 7);
    const auto lines = tail_lines(run.log_file, 10);
    CHECK(lines.size() == 7);
    auto has = [&](const std::string& l) { return std::find(lines.begin(), lines.end(), l) != lines.end(); };  // stderr and stdout interleave freely
    CHECK(has("child config=" + run.config_file.string()));
    CHECK(has("child arg=arg with space") && has("child stderr line"));
    CHECK(tail_lines(run.log_file, 2).size() == 2);
    std::map<std::string, bool> verdicts;
    {
        std::ifstream lf(run.log_file);
        verdicts = cheat_verdicts(lf);
    }  // closed before the cleanup below (an open file blocks remove_all on Windows)
    CHECK(verdicts.size() == 4 && verdicts.at("god") && !verdicts.at("nohit") && !verdicts.at("stamina") && !verdicts.at("ammo"));  // first verdict per name wins
    Plan bad = run;
    bad.exe = tmp / "does-not-exist";
    CHECK(!spawn(bad, pr, err) && !err.empty());

    // --- PKG import: command template, plain header, work folder, extractor runs (real helper = this executable) ---
    {
        std::string e;
        auto sp = split_command("python \"C:\\my tools\\x.py\" {pkg} {out}", e);
        CHECK(e.empty() && sp.size() == 4 && sp[0] == "python" && sp[1] == "C:\\my tools\\x.py" && sp[3] == "{out}");
        sp = split_command("tool & echo %PATH% $HOME; rm -rf *", e);  // no shell: metacharacters are plain argument text
        CHECK(e.empty() && sp.size() == 8 && sp[1] == "&" && sp[3] == "%PATH%" && sp[4] == "$HOME;");
        CHECK(split_command("a \"b", e).empty() && !e.empty());
        CHECK(split_command("  a   \"\"  b ", e).size() == 3);  // an empty quoted argument survives
        const auto sub = substitute_args(split_command("x --out={out} \"{pkg}\" {pkg}{pkg}", e), "C:/a b/p&k g.pkg", "D:/o u;t");
        CHECK(sub.size() == 4 && sub[1] == "--out=D:/o u;t" && sub[2] == "C:/a b/p&k g.pkg" && sub[3] == "C:/a b/p&k g.pkgC:/a b/p&k g.pkg");
        CHECK(check_template(split_command("tool {pkg} {out}", e)).empty());
        CHECK(!check_template(split_command("tool {pkg}", e)).empty() && !check_template({}).empty() && !check_template(split_command("{pkg} {out}", e)).empty());
        CHECK(template_for_program("C:/t/x.py") == "python C:/t/x.py {pkg} {out}" && template_for_program("C:/my t/x.exe") == "\"C:/my t/x.exe\" {pkg} {out}");

        auto header = [&](const char* name, size_t n, const std::string& magic, const std::string& cid) {
            std::string b(n, '\0');
            b.replace(0, std::min(magic.size(), n), magic.substr(0, n));
            if (n >= 0x40) b.replace(0x40, std::min<size_t>(cid.size(), n - 0x40), cid.substr(0, n - 0x40));
            write(tmp / "pkg" / name, b);
            return tmp / "pkg" / name;
        };
        const std::string magic = "\x7f" "CNT";
        PkgInfo pi;
        CHECK(read_pkg_header(header("game.pkg", 0x100, magic, "EP9000-CUSA03173_00-BLOODBORNE0000EU"), pi, e) && pi.content_id == "EP9000-CUSA03173_00-BLOODBORNE0000EU" &&
              pi.title_id == "CUSA03173" && pi.size == 0x100);
        PkgInfo dlc, other;
        CHECK(read_pkg_header(header("dlc.pkg", 0x100, magic, "EP9000-CUSA03173_00-BLOODBORNEDLC0001"), dlc, e) && dlc.title_id == "CUSA03173");
        CHECK(read_pkg_header(header("other.pkg", 0x100, magic, "EP0000-CUSA11111_00-OTHERGAME0000000"), other, e) && other.title_id == "CUSA11111");
        CHECK(!read_pkg_header(header("short.pkg", 0x50, magic, "EP9000-CUSA03173_00-BLOODBORNE0000EU"), pi, e) && e.find("truncated") != std::string::npos);
        CHECK(!read_pkg_header(header("tiny.pkg", 2, "\x7f" "C", ""), pi, e) && e.find("Not a PS4 PKG") != std::string::npos);
        CHECK(!read_pkg_header(header("bad.pkg", 0x100, "MZ\x90\0", "EP9000-CUSA03173_00-BLOODBORNE0000EU"), pi, e) && e.find("Not a PS4 PKG") != std::string::npos);
        CHECK(!read_pkg_header(tmp / "pkg" / "missing.pkg", pi, e) && e.find("not found") != std::string::npos);
        CHECK(read_pkg_header(header("odd.pkg", 0x100, magic, "SOMETHING-ELSE"), pi, e) && pi.title_id.empty());  // no claim about unusual ids
        PkgInfo gi;
        read_pkg_header(tmp / "pkg" / "game.pkg", gi, e);
        const fs::path gp = tmp / "pkg" / "game.pkg", dp = tmp / "pkg" / "dlc.pkg", op = tmp / "pkg" / "other.pkg";
        auto errors = [](const std::vector<bb::dump::Finding>& f) { return std::count_if(f.begin(), f.end(), [](auto& x) { return x.severity == bb::dump::Severity::Error; }); };
        CHECK(errors(check_pkg_set(gi, gp, &dlc, &dp, nullptr, nullptr)) == 0);
        CHECK(errors(check_pkg_set(gi, gp, &other, &op, nullptr, nullptr)) == 1);
        CHECK(errors(check_pkg_set(gi, gp, &dlc, &gp, nullptr, nullptr)) == 1);   // the game PKG picked as DLC
        CHECK(errors(check_pkg_set(gi, gp, &dlc, &dp, &dlc, &dp)) == 1);          // same file as DLC and update
        CHECK(errors(check_pkg_set(other, op, nullptr, nullptr, nullptr, nullptr)) == 1);  // fix 6: an unknown game title is refused at header time
        CHECK(errors(check_pkg_set(pi, gp, nullptr, nullptr, nullptr, nullptr)) == 0);     // unusual content id: only a warning, no claim
        // The Old Hunters DLC PKG of the EU game carries title CUSA00900 (a known Bloodborne id): a warning, not an error; an update must match exactly.
        PkgInfo dlc900, upd900;
        CHECK(read_pkg_header(header("dlc900.pkg", 0x100, magic, "UP9000-CUSA00900_00-SPEXPANSIONDLC03"), dlc900, e) && dlc900.title_id == "CUSA00900");
        CHECK(read_pkg_header(header("upd900.pkg", 0x100, magic, "UP9000-CUSA00900_00-BLOODBORNE0000UP"), upd900, e));
        const fs::path d9 = tmp / "pkg" / "dlc900.pkg", u9 = tmp / "pkg" / "upd900.pkg";
        {
            const auto f = check_pkg_set(gi, gp, &dlc900, &d9, nullptr, nullptr);
            CHECK(errors(f) == 0 && std::count_if(f.begin(), f.end(), [](auto& x) { return x.severity == bb::dump::Severity::Warning; }) == 1);
        }
        CHECK(errors(check_pkg_set(gi, gp, nullptr, nullptr, &upd900, &u9)) == 1);

        // fix 1: the dialog state rules (a running check/install is never replaced; a finished install is registered first)
        CHECK(check_request(AddPhase::Checking, false) == CheckAction::Defer && check_request(AddPhase::Installing, false) == CheckAction::Defer);
        CHECK(check_request(AddPhase::Installed, false) == CheckAction::Defer && check_request(AddPhase::Installed, true) == CheckAction::Start);
        CHECK(check_request(AddPhase::Checked, false) == CheckAction::Start && check_request(AddPhase::Failed, false) == CheckAction::Start);

        // fixes 2/3: extractor lookup never uses the current directory; explicit paths are taken as given (made absolute)
        {
            const fs::path cwd0 = fs::current_path(ec);
            write(tmp / "cwdprobe" / "bbl_cwd_probe.exe", "x");
            write(tmp / "bindir" / "bbl_bin_probe.exe", "x");
            fs::current_path(tmp / "cwdprobe", ec);
            std::string er;
            CHECK(resolve_program_in("bbl_cwd_probe", {tmp / "bindir", fs::path(), fs::path("."), fs::path("relative")}, er).empty() && er.find("not found") != std::string::npos);
            CHECK(resolve_program("bbl_cwd_probe.exe", er).empty());                       // really in the cwd, still not found
#ifdef _WIN32
            CHECK(resolve_program_in("bbl_bin_probe", {fs::path(), tmp / "bindir"}, er) == tmp / "bindir" / "bbl_bin_probe.exe");  // ".exe" is appended
#else
            CHECK(resolve_program_in("bbl_bin_probe.exe", {fs::path(), tmp / "bindir"}, er) == tmp / "bindir" / "bbl_bin_probe.exe");
#endif
            CHECK(resolve_program_in("bbl_bin_probe.exe", {tmp / "bindir"}, er) == tmp / "bindir" / "bbl_bin_probe.exe");
            CHECK(resolve_program_in("sub/tool", {}, er) == fs::absolute("sub/tool", ec) && fs::absolute("sub/tool", ec).is_absolute());
            CHECK(resolve_program_in("", {tmp}, er).empty());
            fs::current_path(cwd0, ec);
        }
        {  // the extractor is not held to the bbgame PE gate: a non-PE file is started (and fails in CreateProcess / exec), not refused with "not a valid 64-bit program"
            std::string er;
            write(tmp / "tools" / "junk-tool.exe", "not a pe");
            const ExtractResult jr = [&] {
                const fs::path wd = make_work_dir(tmp / "data", "junk", er);
                return run_extractor(split_command("\"" + (tmp / "tools" / "junk-tool.exe").string() + "\" {pkg} {out}", er), tmp / "pkg" / "game.pkg", wd / "game", PkgKind::Game,
                                     wd.parent_path() / "junk.log", nullptr);
            }();
            CHECK(!jr.ok && jr.error.find("not a valid 64-bit program") == std::string::npos);
        }

        // work folder: fresh only, marker, removal only of its own
        const fs::path data = tmp / "data";
        const fs::path w1 = make_work_dir(data, "run1", e);
        CHECK(!w1.empty() && fs::is_regular_file(w1 / kWorkMarker) && w1 == data / "pkg_work" / "run1");
        CHECK(make_work_dir(data, "run1", e).empty() && !e.empty());                  // refuses an existing one
        CHECK(make_work_dir(data, "../evil", e).empty());
        write(data / "pkg_work" / "foreign" / "keep.txt", "user data");               // no marker: not ours
        write(tmp / "precious" / "keep.txt", "user data");
        write(w1 / "game" / "Image0" / "x", "x");
        CHECK(!remove_work_dir(data / "pkg_work" / "foreign", data, e) && fs::exists(data / "pkg_work" / "foreign" / "keep.txt"));
        CHECK(!remove_work_dir(tmp / "precious", data, e) && fs::exists(tmp / "precious" / "keep.txt"));
        CHECK(!remove_work_dir(data / "pkg_work", data, e) && !remove_work_dir(data, data, e) && !remove_work_dir({}, data, e));
        CHECK(!remove_work_dir(w1 / "game", data, e) && fs::exists(w1 / "game"));     // a subfolder is not a work folder
        CHECK(remove_work_dir(w1, data, e) && !fs::exists(w1) && fs::exists(data / "pkg_work" / "foreign" / "keep.txt") && fs::exists(tmp / "precious" / "keep.txt"));
        CHECK(check_extract_space(1, data).empty() && !check_extract_space(uintmax_t(1) << 62, data).empty());

        // fix 4: several work folders (also ones left behind by an earlier launcher run) are all listed; unmarked folders are not
        {
            const fs::path d2 = tmp / "data2";
            std::string er;
            const fs::path a1 = make_work_dir(d2, "20260101-000000-1", er), a2 = make_work_dir(d2, "20260102-000000-2", er);
            write(d2 / "pkg_work" / "unmarked" / "x", "x");
            write(d2 / "pkg_work" / "20260101-000000-1-game.log", "log");
            const auto l = list_work_dirs(d2);
            CHECK(l.size() == 2 && l[0] == a1 && l[1] == a2);
            CHECK(list_work_dirs(tmp / "nothing-here").empty());
            CHECK(remove_work_dir(a1, d2, er) && !fs::exists(d2 / "pkg_work" / "20260101-000000-1-game.log") && list_work_dirs(d2).size() == 1);  // explicit delete takes its logs along
        }

        // extractor runs: this executable in helper modes; the paths contain spaces and shell metacharacters
        const std::string self = std::string("\"") + argv[0] + "\"";
        auto extract = [&](const char* mode, const char* name, PkgKind kind, const std::atomic<bool>* cancel = nullptr) {
            std::string er;
            const fs::path wd = make_work_dir(data, name, er);
            const fs::path pkg = tmp / "pkg & x %PATH%" / "my game.pkg";
            write(pkg, "pkg");
            return std::make_pair(run_extractor(split_command(self + " " + mode + " {pkg} {out}", er), pkg, wd / "game", kind, work_log(wd, kind), cancel), wd);
        };
        auto [ok, wd_ok] = extract("--extract-fixture", "ok", PkgKind::Game);
        CHECK(ok.ok && fs::is_regular_file(wd_ok / "game" / "Image0" / "eboot.bin") && fs::is_regular_file(wd_ok / "game" / "Sc0" / "param.sfo"));
        {
            std::ifstream seen(wd_ok / "game" / "pkg_seen.txt");
            std::string line;
            std::getline(seen, line);
            CHECK(line == (tmp / "pkg & x %PATH%" / "my game.pkg").string());  // one intact argument
        }
        CHECK(!tail_lines(work_log(wd_ok, PkgKind::Game), 3).empty());
        auto [fail, wd_fail] = extract("--extract-fail", "fail", PkgKind::Game);
        CHECK(!fail.ok && fail.error.find("Full log: ") != std::string::npos);
        CHECK(!fail.ok && fail.exit_code == 3 && fail.error.find("code 3") != std::string::npos && fail.error.find("boom: cannot decode") != std::string::npos);
        {  // fix 5: the failure cleanup removes the work folder but keeps the extractor log (outside the folder), and the error names it
            const fs::path lg = wd_fail.parent_path() / "fail-game.log";
            std::string er;
            CHECK(work_log(wd_fail, PkgKind::Game) == lg);
            CHECK(remove_work_dir(wd_fail, data, er, true) && !fs::exists(wd_fail));
            const auto lg_lines = tail_lines(lg, 5);
            CHECK(!lg_lines.empty() && std::any_of(lg_lines.begin(), lg_lines.end(), [](const std::string& x) { return x.find("boom") != std::string::npos; }));  // the log survives
        }
        auto [empty, wd_empty] = extract("--extract-empty", "empty", PkgKind::Game);
        CHECK(!empty.ok && empty.exit_code == 0 && empty.error.find("Image0/ and Sc0/param.sfo are missing") != std::string::npos);
        auto [dlcrun, wd_dlc] = extract("--extract-fixture", "dlc", PkgKind::Dlc);
        CHECK(dlcrun.ok);
        {  // relative {out}: the extractor runs in the work folder, so it must receive absolute paths
            std::string er;
            const fs::path wd = make_work_dir(data, "rel", er), rel = fs::relative(wd / "game", fs::current_path(), ec);
            if (!rel.empty() && !rel.is_absolute())
                CHECK(run_extractor(split_command(self + " --extract-fixture {pkg} {out}", er), fs::relative(tmp / "pkg" / "game.pkg", fs::current_path(), ec), rel, PkgKind::Game, wd / "l.log", nullptr).ok);
        }
        {
            std::string er;
            const fs::path wd = make_work_dir(data, "exists", er);
            fs::create_directories(wd / "game");
            CHECK(!run_extractor(split_command(self + " --extract-fixture {pkg} {out}", er), tmp / "pkg" / "game.pkg", wd / "game", PkgKind::Game, wd / "l.log", nullptr).ok);
            CHECK(!run_extractor(split_command("no-such-extractor-xyz {pkg} {out}", er), tmp / "pkg" / "game.pkg", wd / "g2", PkgKind::Game, wd / "l.log", nullptr).ok &&
                  run_extractor(split_command("no-such-extractor-xyz {pkg} {out}", er), tmp / "pkg" / "game.pkg", wd / "g3", PkgKind::Game, wd / "l.log", nullptr).error.find("not found next to the launcher") != std::string::npos);
            CHECK(run_extractor(split_command("tool {pkg}", er), tmp / "pkg" / "game.pkg", wd / "g4", PkgKind::Game, wd / "l.log", nullptr).error.find("{out}") != std::string::npos);
        }
        std::atomic<bool> cancel{false};
        std::thread killer([&] { std::this_thread::sleep_for(std::chrono::milliseconds(600)), cancel = true; });
        const auto t0 = std::chrono::steady_clock::now();
        auto [cancelled, wd_cancel] = extract("--extract-sleep", "cancel", PkgKind::Game, &cancel);
        killer.join();
        CHECK(!cancelled.ok && cancelled.error == "Cancelled." && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(15));
        CHECK(remove_work_dir(wd_cancel, data, e) && !fs::exists(wd_cancel));         // the child is gone: the folder can be removed
    }

    fs::remove_all(tmp, ec);
    std::printf(failures ? "%d FAILED\n" : "launcher: all ok\n", failures);
    return failures ? 1 : 0;
}
