// SPDX-License-Identifier: GPL-3.0-or-later
// Closed-loop navigation for scripted play (debug tooling; everything stays off unless one of the env vars below is set).
// BB_TELEMETRY=<file>: ~4 lines/s "T t= f= st= cam= yaw= pitch= pos= cmd=" plus event lines "E t= <id> done|fail|stuck ...".
// Commands (BB_PAD_LIVE lines, BB_ROUTE files; run in order): goto <x> <z> [tol] [sprint] | face <x> <z> | wait_load | wait <s> |
// press <btn> | tp <x> <y> <z>.  BB_ROUTE_REC=<file>[,<metres>] appends "goto x z" every few metres walked.
#include "gpu/pad.h"

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <deque>
#include <mutex>
#include <sstream>
#include <string>

namespace bb::gpu {
namespace {

const char* env(const char* k) { const char* v = std::getenv(k); return v && *v ? v : nullptr; }
#if BB_DEV_TOOLS  // dev cheats for testing (CMake BB_DEV_TOOLS): BB_DEV_GODMODE=1 keeps HP at max, BB_DEV_TELEPORT=x,y,z once after a load
const bool g_god = env("BB_DEV_GODMODE") && env("BB_DEV_GODMODE")[0] == '1';
const char* const g_tp_once = env("BB_DEV_TELEPORT");
#else
constexpr bool g_god = false;
constexpr const char* g_tp_once = nullptr;
#endif
const bool g_on = env("BB_TELEMETRY") || env("BB_PAD_LIVE") || env("BB_ROUTE") || env("BB_ROUTE_REC") || g_god || g_tp_once;

bool peek(uint64_t a, void* out, size_t n) {  // guest read that tolerates unmapped addresses (host address = guest address)
    if (a < 0x10000) return false;
#ifdef _WIN32
    SIZE_T got = 0;
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(a), out, n, &got) && got == n;
#else
    std::memcpy(out, reinterpret_cast<const void*>(a), n);
    return true;
#endif
}
#if BB_DEV_TOOLS
bool poke(uint64_t a, const void* in, size_t n) {
#ifdef _WIN32
    SIZE_T put = 0;
    return WriteProcessMemory(GetCurrentProcess(), reinterpret_cast<void*>(a), in, n, &put) && put == n;
#else
    std::memcpy(reinterpret_cast<void*>(a), in, n);
    return true;
#endif
}
#endif

struct V3 { float x = 0, y = 0, z = 0; };

std::mutex g_m;  // everything below: backend thread (camera votes), flip thread (telemetry), pad poll thread (commands)

// Main camera: the 864-byte constant block (ROADMAP 68d) with far plane >= 500 and a 16:9 render size; per frame the view seen most often.
struct Vote { float v[12]; int n; };
Vote g_votes[8];
int g_nvotes = 0;
float g_view[12] = {};  // view 3x4, row-major: row 0 = screen right, row 2 = forward (world), column 3 = translation
uint64_t g_flip = 0, g_cam_flip = 0;
bool g_cam = false;

bool cam_pos(V3& c, V3& fwd, V3& right) {
    if (!g_cam || g_flip - g_cam_flip > 10) return false;
    const float* v = g_view;
    c = {-(v[0] * v[3] + v[4] * v[7] + v[8] * v[11]), -(v[1] * v[3] + v[5] * v[7] + v[9] * v[11]), -(v[2] * v[3] + v[6] * v[7] + v[10] * v[11])};
    right = {v[0], v[1], v[2]}, fwd = {v[8], v[9], v[10]};
    return true;
}

// The player (EU eboot; found by memory scan + write tests): X = [[0x593e848] (vtable 0x57301f0)
// + 0x60] = the player character (vtable 0x578f310; re-created on respawn). Position: [[[X+0x20]+0x1a0]+0xf8] = its physics proxy
// (vtable 0x57a7940): rotation 3x4 at +0x40, position float3 at +0x70 (feet; writing it teleports, the game copies it into every
// other position). HP: an object with vtable 0x5735810 and owner X at +8, inside or shortly after X (seen at X+0xdc0, +0x1080,
// +0x2a800): int32 HP at +0xf8, max HP +0xfc (the game's source: the HUD and other copies follow writes). The vtable checks
// reject another eboot / a half-built world.
uint64_t ld64(uint64_t a) { uint64_t v = 0; return peek(a, &v, 8) ? v : 0; }
uint64_t player_chr() {
    const uint64_t m = ld64(0x593e848), x = ld64(m) == 0x57301f0 ? ld64(m + 0x60) : 0;
    return x && ld64(x) == 0x578f310 ? x : 0;
}
// Three pointer paths from X reach the proxy; one alone failed for a whole run once (no position until 226 s), so take the value
// two of them agree on, else the first valid one. ponytail: paths seen in two sessions only; add more if all three fail.
uint64_t player_proxy() {
    const uint64_t x = player_chr();
    if (!x) return 0;
    auto ok = [](uint64_t p) { return p && ld64(p) == 0x57a7940 ? p : 0; };
    const uint64_t a = ok(ld64(ld64(ld64(x + 0x20) + 0x1a0) + 0xf8)), b = ok(ld64(ld64(ld64(x + 0x58) + 0x558) + 0x770)),
                   c = ok(ld64(ld64(ld64(x + 0x2c0) + 0x110) + 0x38));
    if (a && (a == b || a == c)) return a;
    if (b && b == c) return b;
    return a ? a : b ? b : c;
}
// the main camera follows the player (3-5 m); the title screen has a camera 18 m away while the player already exists
bool in_world(const V3& cam, const V3& p) { return std::hypot(cam.x - p.x, cam.y - p.y, cam.z - p.z) < 8.f; }
bool player_pos(V3& p) {
    const uint64_t proxy = player_proxy();
    float f[3];
    if (!proxy || !peek(proxy + 0x70, f, 12) || !std::isfinite(f[0]) || !std::isfinite(f[1]) || !std::isfinite(f[2])) return false;
    p = {f[0], f[1], f[2]};
    return true;
}
uint64_t hp_block() {
    static uint64_t cached = 0, scanned = 0;  // the block found, the X last searched in vain
    static int retry = 0;
    const uint64_t x = player_chr();
    if (!x) return 0;
    if (cached && ld64(cached) == 0x5735810 && ld64(cached + 8) == x) return cached;
    cached = 0;
    if (scanned == x && ++retry < 60) return 0;  // in vain for this X: look again every 60 calls (~2 s), the block may come later
    scanned = x, retry = 0;
    static uint64_t buf[0x2000];  // 64 KB chunks over [X, X + 1 MB)
    for (uint64_t a = x; a < x + (1u << 20) && !cached; a += sizeof buf)
        if (peek(a, buf, sizeof buf))
            for (size_t i = 0; i + 1 < std::size(buf); ++i)
                if (buf[i] == 0x5735810 && buf[i + 1] == x) { cached = a + i * 8; break; }
    if (cached) scanned = 0;
    return cached;
}
#if BB_DEV_TOOLS
bool teleport(float x, float y, float z) {
    const uint64_t proxy = player_proxy();
    const float v[3] = {x, y, z};
    return proxy && poke(proxy + 0x70, v, 12);
}
#endif

FILE* g_tel = nullptr;
void ev(double t, int id, const char* what, const std::string& cmd) {
    std::fprintf(stderr, "nav @%.2f: %d %s %s\n", t, id, what, cmd.c_str());
    if (!g_tel) return;
    std::fprintf(g_tel, "E t=%.2f %d %s %s\n", t, id, what, cmd.c_str());
    std::fflush(g_tel);
}

uint32_t button(const std::string& n) {
    static const struct { const char* n; uint32_t b; } names[] = {{"up", 0x10}, {"down", 0x40}, {"left", 0x80}, {"right", 0x20}, {"cross", 0x4000}, {"circle", 0x2000},
                                                                  {"square", 0x8000}, {"triangle", 0x1000}, {"l1", 0x400}, {"r1", 0x800}, {"l2", 0x100}, {"r2", 0x200},
                                                                  {"l3", 0x2}, {"r3", 0x4}, {"options", 0x8}};
    for (const auto& b : names) if (n == b.n) return b.b;
    return 0;
}

enum class K { Goto, Face, WaitLoad, Wait, Press, Tp };
struct Cmd { K k; float x = 0, y = 0, z = 0, tol = 0.8f; bool sprint = false; uint32_t btn = 0; int id = 0; std::string text; };
std::deque<Cmd> g_q;
int g_next_id = 1;
bool g_active = false;
Cmd g_cur;
double g_t0 = 0;            // command start
double g_win_t = 0;         // goto: stuck window start, distance to target then
float g_win_d = 0;
int g_stuck = 0;            // goto: stuck events so far
double g_rec_until = 0, g_rec_side = 0;  // goto: recovery manoeuvre until then, its side-step from g_rec_side on
float g_rec_x = 0, g_rec_z = 0;          // its back-off direction (world)
double g_play_since = -1;   // wait_load: play state continuously since; a load seen since the command started
bool g_load_seen = false;

bool parse(const std::string& line, Cmd& c) {
    std::istringstream in(line);
    std::string w;
    if (!(in >> w)) return false;
    c.text = line;
    if (w == "goto") {
        c.k = K::Goto;
        if (!(in >> c.x >> c.z)) return false;
        for (std::string o; in >> o;) o == "sprint" ? void(c.sprint = true) : void(c.tol = std::strtof(o.c_str(), nullptr));
    } else if (w == "face") { c.k = K::Face; if (!(in >> c.x >> c.z)) return false; }
    else if (w == "wait_load") c.k = K::WaitLoad;
    else if (w == "wait") { c.k = K::Wait; if (!(in >> c.x)) return false; }
    else if (w == "press") { c.k = K::Press; if (!(in >> w) || !(c.btn = button(w))) return false; }
#if BB_DEV_TOOLS
    else if (w == "tp") { c.k = K::Tp; if (!(in >> c.x >> c.y >> c.z)) return false; }
#endif
    else return false;
    return true;
}

void load_route() {  // BB_ROUTE: commands, one per line, '#' comments
    const char* path = env("BB_ROUTE");
    if (!path) return;
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "nav: cannot open BB_ROUTE %s\n", path); return; }
    char buf[256];
    while (std::fgets(buf, sizeof buf, f)) {
        std::string l = buf;
        if (const size_t h = l.find('#'); h != std::string::npos) l.erase(h);
        Cmd c;
        if (parse(l, c)) { while (!c.text.empty() && (c.text.back() == '\n' || c.text.back() == '\r' || c.text.back() == ' ')) c.text.pop_back(); c.id = g_next_id++; g_q.push_back(c); }
    }
    std::fclose(f);
}

// world direction (dx, dz) -> left stick, relative to the camera; scale 0..1
void stick_to(float dx, float dz, float scale, const V3& fwd, const V3& right, uint8_t& lx, uint8_t& ly) {
    float fx = fwd.x, fz = fwd.z, rx = right.x, rz = right.z;
    const float fl = std::hypot(fx, fz), rl = std::hypot(rx, rz), dl = std::hypot(dx, dz);
    if (fl < 1e-4f || rl < 1e-4f || dl < 1e-4f) return;
    const float sx = (dx * rx + dz * rz) / (rl * dl), sy = (dx * fx + dz * fz) / (fl * dl);
    const float m = std::max(std::fabs(sx), std::fabs(sy));  // push the stick to the square's edge (full run)
    lx = uint8_t(std::clamp(128.f + 127.f * scale * sx / m, 0.f, 255.f));
    ly = uint8_t(std::clamp(128.f - 127.f * scale * sy / m, 0.f, 255.f));
}

void finish(double now, const char* what) {
    ev(now, g_cur.id, what, g_cur.text);
    g_active = false;
}

}  // namespace

bool nav_command(const char* line) {
    if (!g_on) return false;
    Cmd c;
    if (!parse(line, c)) return false;
    std::lock_guard lk(g_m);
    c.id = g_next_id++;
    g_q.push_back(c);
    return true;
}

void nav_note_camera(const uint32_t* blk) {
    float zfar, w, h;  // dword 0 = far plane: 3000 for the main view, ~100-150 for shadow cascades, < 20 for point-light cubes
    std::memcpy(&zfar, &blk[0], 4), std::memcpy(&w, &blk[4], 4), std::memcpy(&h, &blk[5], 4);
    if (!(zfar >= 500 && w >= 320 && h >= 180 && std::fabs(w * 9 - h * 16) < 32)) return;
    float v[12];
    std::memcpy(v, &blk[8], sizeof v);
    std::lock_guard lk(g_m);
    for (int i = 0; i < g_nvotes; ++i)
        if (!std::memcmp(g_votes[i].v, v, sizeof v)) { ++g_votes[i].n; return; }
    if (g_nvotes < int(std::size(g_votes))) { std::memcpy(g_votes[g_nvotes].v, v, sizeof v); g_votes[g_nvotes++].n = 1; }
}

bool nav_camera_wanted() { return g_on; }

void nav_flip(double now) {
    if (!g_on) return;
    static const char* const rec_arg = env("BB_ROUTE_REC");
    static FILE* rec = rec_arg ? std::fopen(std::string(rec_arg).substr(0, std::string(rec_arg).find(',')).c_str(), "a") : nullptr;
    static const float rec_step = rec_arg && std::strchr(rec_arg, ',') ? std::strtof(std::strchr(rec_arg, ',') + 1, nullptr) : 3.f;
    static V3 rec_last;
    static bool rec_any = false;
    static double tel_t = -1;
    std::lock_guard lk(g_m);
    if (!g_tel && env("BB_TELEMETRY")) g_tel = std::fopen(env("BB_TELEMETRY"), "w");
    if (g_flip++ == 0 && (g_god || g_tp_once))
        std::fprintf(stderr, "nav: DEV CHEAT ACTIVE:%s%s%s\n", g_god ? " godmode" : "", g_tp_once ? " teleport " : "", g_tp_once ? g_tp_once : "");
    if (g_nvotes) {
        int best = 0;
        for (int i = 1; i < g_nvotes; ++i) if (g_votes[i].n > g_votes[best].n) best = i;
        std::memcpy(g_view, g_votes[best].v, sizeof g_view);
        g_cam = true, g_cam_flip = g_flip, g_nvotes = 0;
    }
    V3 c, f, r, p;
    const bool cam = cam_pos(c, f, r), pos = player_pos(p), world = cam && pos && in_world(c, p);
    const uint64_t hb = pos ? hp_block() : 0;
    int32_t hp[2] = {};
    const bool hp_ok = hb && peek(hb + 0xf8, hp, 8);
#if BB_DEV_TOOLS
    static int settled = 0;  // BB_DEV_TELEPORT: once the player has existed for 2 s
    if (g_tp_once && settled >= 0 && (pos && cam ? ++settled : (settled = 0)) > 60) {
        float t[3] = {};
        std::sscanf(g_tp_once, "%f,%f,%f", &t[0], &t[1], &t[2]);
        std::fprintf(stderr, "nav: dev teleport to %g,%g,%g: %s\n", t[0], t[1], t[2], teleport(t[0], t[1], t[2]) ? "ok" : "failed");
        settled = -1;
    }
    if (g_god && hp_ok && hp[0] > 0 && hp[0] < hp[1]) poke(hb + 0xf8, &hp[1], 4);  // ponytail: per flip; one hit above max HP still kills
#endif
    if (rec && pos && cam) {
        if (!rec_any || std::hypot(p.x - rec_last.x, p.z - rec_last.z) >= rec_step) {
            std::fprintf(rec, "goto %.2f %.2f\n", p.x, p.z);
            std::fflush(rec);
            rec_last = p, rec_any = true;
        }
    }
    if (!g_tel || now - tel_t < 0.25) return;
    tel_t = now;
    std::fprintf(g_tel, "T t=%.2f f=%llu st=%s", now, (unsigned long long)g_flip, world ? "play" : cam ? "menu" : "load");
    if (cam)
        std::fprintf(g_tel, " cam=%.2f,%.2f,%.2f yaw=%.1f pitch=%.1f", c.x, c.y, c.z, std::atan2(f.x, f.z) * 57.29578f,
                     std::asin(std::clamp(f.y, -1.f, 1.f)) * 57.29578f);
    if (pos) std::fprintf(g_tel, " pos=%.2f,%.2f,%.2f", p.x, p.y, p.z);
    if (hp_ok) std::fprintf(g_tel, " hp=%d/%d", hp[0], hp[1]);
    if (g_active) std::fprintf(g_tel, " cmd=%d", g_cur.id);
    std::fputc('\n', g_tel);
    std::fflush(g_tel);
}

void nav_apply(PadSnapshot& s, double now) {
    if (!g_on) return;
    static bool routed = (load_route(), true);
    (void)routed;
    std::lock_guard lk(g_m);
    if (!g_active && !g_q.empty()) {
        g_cur = g_q.front(), g_q.pop_front(), g_active = true;
        g_t0 = g_win_t = now, g_win_d = -1, g_stuck = 0, g_rec_until = 0, g_play_since = -1, g_load_seen = false;
        ev(now, g_cur.id, "start", g_cur.text);
    }
    if (!g_active) return;
    V3 c, f, r, p;
    const bool cam = cam_pos(c, f, r), pos = player_pos(p), world = cam && pos && in_world(c, p);
    switch (g_cur.k) {
        case K::Press: s.buttons |= g_cur.btn; finish(now, "done"); break;
        case K::Wait: if (now - g_t0 >= g_cur.x) finish(now, "done"); break;
        case K::WaitLoad:
            if (!world) { g_load_seen = true, g_play_since = -1; break; }
            if (g_play_since < 0) g_play_since = now;
            if (now - g_play_since >= 2 && (g_load_seen || now - g_t0 >= 5)) finish(now, "done");
            else if (now - g_t0 > 300) finish(now, "fail timeout");
            break;
#if BB_DEV_TOOLS
        case K::Tp: finish(now, teleport(g_cur.x, g_cur.y, g_cur.z) ? "done" : "fail no-player"); break;
#else
        case K::Tp: break;
#endif
        case K::Face: {
            if (!cam) break;
            const float dx = g_cur.x - c.x, dz = g_cur.z - c.z;
            const float ang = std::atan2(dx * r.x + dz * r.z, dx * f.x + dz * f.z);  // + = target to the right
            if (std::fabs(ang) < 0.06f) { finish(now, "done"); break; }
            if (now - g_t0 > 8) { finish(now, "fail timeout"); break; }
            s.rx = uint8_t(std::clamp(128.f + 127.f * std::clamp(ang * 1.5f, -1.f, 1.f), 0.f, 255.f));
            if (std::fabs(ang) < 0.3f) s.rx = uint8_t(ang > 0 ? 128 + 40 : 128 - 40);  // slow near the goal (the camera lags the stick)
            break;
        }
        case K::Goto: {
            if (!world) { g_win_t = now, g_win_d = -1; break; }  // loading / menu / no player: hold, the window restarts
            const float dx = g_cur.x - p.x, dz = g_cur.z - p.z, d = std::hypot(dx, dz);
            char dd[32];
            std::snprintf(dd, sizeof dd, " d=%.2f", d);
            if (d < g_cur.tol) { finish(now, (std::string("done") + dd).c_str()); break; }
            if (now < g_rec_until) {  // recovery: back off 0.5 s, then side-step (sides L L R R ..., longer each time)
                const float side = ((g_stuck - 1) >> 1) & 1 ? 1.f : -1.f;
                if (now < g_rec_side) stick_to(g_rec_x, g_rec_z, 1, f, r, s.lx, s.ly);
                else stick_to(-g_rec_z * side, g_rec_x * side, 1, f, r, s.lx, s.ly);
                break;
            }
            if (g_win_d < 0) g_win_d = d, g_win_t = now;
            if (now - g_win_t >= 1.0) {  // stuck: < 0.3 m progress toward the target in 1 s
                if (g_win_d - d < 0.3f) {
                    ev(now, g_cur.id, (std::string("stuck") + dd).c_str(), g_cur.text);
                    if (++g_stuck > 5) { finish(now, (std::string("fail stuck") + dd).c_str()); break; }
                    g_rec_x = -dx / d, g_rec_z = -dz / d;
                    g_rec_side = now + 0.5;
                    g_rec_until = g_rec_side + 0.4f + 0.3f * float(g_stuck);
                    g_win_d = -1;
                    break;
                }
                g_win_d = d, g_win_t = now;
            }
            stick_to(dx, dz, std::clamp(d / 1.5f, 0.45f, 1.f), f, r, s.lx, s.ly);
            if (g_cur.sprint && d > 4) s.buttons |= 0x2000;  // circle held = sprint
            break;
        }
    }
}

}  // namespace bb::gpu
