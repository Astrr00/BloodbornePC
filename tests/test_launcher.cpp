// SPDX-License-Identifier: GPL-3.0-or-later
// Launcher logic: launcher.ini round trip, bbconfig.ini parser/env override, build matching, dry-run plan, process start + log capture.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>

#include "core.h"
#include "core/bbconfig.h"
#ifdef _WIN32
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
    const Build fake{"C:/b/bbgame.exe", kShaA, "sidecar"};
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
    CHECK(parse_config(write_config(Config{kConfigVersion, gd.id, {gd}, {}})).games[0].settings["BB_DLC_DIR"] == "E:/dlc/The Old Hunters");
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

    fs::remove_all(tmp, ec);
    std::printf(failures ? "%d FAILED\n" : "launcher: all ok\n", failures);
    return failures ? 1 : 0;
}
