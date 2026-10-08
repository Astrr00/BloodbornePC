// SPDX-License-Identifier: GPL-3.0-or-later
// Closed-loop navigation for scripted play (debug tooling; everything stays off unless one of the env vars below is set).
// BB_TELEMETRY=<file>: ~4 lines/s "T t= f= st= cam= yaw= pitch= pos= hp= cmd= stk= [sprint]" plus event lines "E t= <id> start|done|fail|stuck ...".
// Commands (BB_PAD_LIVE lines, BB_ROUTE files; run in order; after "failfast 1" a failed one drops the rest of the queue):
// goto <x> <z> [tol] [t=<s>] | ladder <x> <z> [<fx> <fz>] [down] | face <x> <z> | wait_load | wait <s> | press <btn> | tp <x> <y> <z> | hwwatch <hex>... |
// failfast 0|1; live "abort" (or "clear") drops the running and pending commands at once. BB_ROUTE_REC=<file>[,<metres>] records
// the walk as a goto route (loops cut, simplified).
#include "gpu/gpu_hooks.h"
#include "gpu/pad.h"
#include "gpu/cheats.h"

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <deque>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#if BB_DEV_TOOLS  // the whole file: telemetry, navigation, tp, hwwatch, dev cheat switches (stubs at the end for release builds)

namespace bb::gpu {
namespace {

const char* env(const char* k) { const char* v = std::getenv(k); return v && *v ? v : nullptr; }
// dev tools for testing (CMake BB_DEV_TOOLS): BB_DEV_GODMODE=1 / BB_DEV_NOHIT=1 switch on the BB_CHEATS god / nohit (cheats.cpp),
// BB_DEV_TELEPORT=x,y,z once after a load
bool dev_cheat(const char* k, uint32_t bit) {
    const char* v = env(k);
    if (!v || v[0] != '1') return false;
    cheats_enable(bit);
    return true;
}
const bool g_god = dev_cheat("BB_DEV_GODMODE", kCheatGod);
const bool g_nohit = dev_cheat("BB_DEV_NOHIT", kCheatNoHit);
const char* const g_tp_once = env("BB_DEV_TELEPORT");
const bool g_on = env("BB_TELEMETRY") || env("BB_PAD_LIVE") || env("BB_ROUTE") || env("BB_ROUTE_REC") || g_god || g_nohit || g_tp_once;

struct V3 { float x = 0, y = 0, z = 0; };
float dist(const V3& a, const V3& b) { return std::hypot(a.x - b.x, a.y - b.y, a.z - b.z); }
V3 lerp(const V3& a, const V3& b, float t) { return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t}; }
float seg_t(const V3& p, const V3& a, const V3& b) {  // the closest point of segment ab to p, as 0..1
    const float ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z, uu = ux * ux + uy * uy + uz * uz;
    return uu > 1e-6f ? std::clamp(((p.x - a.x) * ux + (p.y - a.y) * uy + (p.z - a.z) * uz) / uu, 0.f, 1.f) : 0.f;
}
float seg_dist(const V3& p, const V3& a, const V3& b) { return dist(p, lerp(a, b, seg_t(p, a, b))); }
float wrap(float a) { return std::remainder(a, 6.2831853f); }  // angle -> -pi..pi
float heading(const V3& a, const V3& b) { return std::atan2(b.x - a.x, b.z - a.z); }  // world heading of a->b (x, z)
float turn(const V3& a, const V3& b, const V3& c) {  // direction change at b, 0..pi
    return dist(a, b) < 0.1f || dist(b, c) < 0.1f ? 0.f : std::fabs(wrap(heading(b, c) - heading(a, b)));
}

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

// The player: X = player_chr() (cheats.cpp) = the player character (re-created on respawn). Position: [[[X+0x20]+0x1a0]+0xf8] = its
// physics proxy (vtable 0x57a7940): rotation 3x4 at +0x40, position float3 at +0x70 (feet; writing it teleports, the game copies it
// into every other position).
// Three pointer paths from X reach the proxy; one alone failed for a whole run once (no position until 226 s), so take the value
// two of them agree on, else the first valid one. All three break now and then after a load (seen after lamp warps into the
// Hunter's Dream: [X+0x20]+0x1a0, [X+0x58]+0x558, [X+0x2c0]+0x110 then hold other data, i.e. those objects differ per load), and
// no proxy is reachable from X within 3 levels then. Fallback: nav_flip scans the heap for proxies (scan_proxy) and keeps the one in
// front of the camera for this X.
bool is_proxy_vt(uint64_t v) { return v == 0x57a7940 || v == 0x57a7960; }  // EU 1.00 / EU 1.09
struct Path { uint32_t o[3]; };
const Path g_paths[3] = {{{0x20, 0x1a0, 0xf8}}, {{0x58, 0x558, 0x770}}, {{0x2c0, 0x110, 0x38}}};
uint64_t g_fb_x = 0, g_fb_proxy = 0;  // scan_proxy's proxy, valid while X is g_fb_x
uint64_t follow(uint64_t x, const Path& p) {
    const uint64_t v = ld64(ld64(ld64(x + p.o[0]) + p.o[1]) + p.o[2]);
    return v && is_proxy_vt(ld64(v)) ? v : 0;
}
uint64_t player_proxy() {
    static const bool no_paths = env("BB_NAV_NOPATHS");  // test the fallback: ignore the three paths
    const uint64_t x = player_chr();
    if (!x) return 0;
    const uint64_t a = no_paths ? 0 : follow(x, g_paths[0]), b = no_paths ? 0 : follow(x, g_paths[1]), c = no_paths ? 0 : follow(x, g_paths[2]);
    if (a && (a == b || a == c)) return a;
    if (b && b == c) return b;
    if (a || b || c) return a ? a : b ? b : c;
    return g_fb_x == x && is_proxy_vt(ld64(g_fb_proxy)) ? g_fb_proxy : 0;
}
std::string lookup_state() {  // where the player lookup breaks: every pointer of every path
    const uint64_t m = player_mgr(), x = ld64(m + 0x60);
    std::string s;
    char b[128];
    std::snprintf(b, sizeof b, "m=%llx vt=%llx x=%llx vt=%llx", (unsigned long long)m, (unsigned long long)ld64(m), (unsigned long long)x,
                  (unsigned long long)ld64(x));
    s = b;
    for (const Path& p : g_paths) {
        uint64_t a = x;
        s += " [";
        for (const uint32_t o : p.o) std::snprintf(b, sizeof b, " %llx", (unsigned long long)(a = ld64(a + o))), s += b;
        std::snprintf(b, sizeof b, " vt=%llx]", (unsigned long long)ld64(a)), s += b;
    }
    return s;
}
// Heap scan for physics proxies 1.5-8 m from the camera; the player is the one nearest the view axis (the camera looks at it).
// ponytail: Windows only, guest heap [64 GB, 128 GB) only, ~1-2 s per scan; an enemy right on the view axis could win.
uint64_t scan_proxy(const V3& cam, const V3& fwd) {
    uint64_t best = 0;
#ifdef _WIN32
    float best_off = 2.f;  // metres from the view axis
    static uint64_t buf[1 << 17];
    MEMORY_BASIC_INFORMATION mi;
    for (uint64_t a = 0x1000000000; a < 0x2000000000 && VirtualQuery(reinterpret_cast<void*>(a), &mi, sizeof mi); a = uint64_t(mi.BaseAddress) + mi.RegionSize) {
        if (mi.State != MEM_COMMIT || (mi.Protect & (PAGE_GUARD | PAGE_NOACCESS)) || !(mi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READWRITE)))
            continue;
        const uint64_t end = uint64_t(mi.BaseAddress) + mi.RegionSize;
        for (uint64_t c = uint64_t(mi.BaseAddress); c < end; c += sizeof buf) {
            const size_t n = size_t(std::min<uint64_t>(sizeof buf, end - c));
            if (!peek(c, buf, n)) continue;
            for (size_t i = 0; i < n / 8; ++i) {
                float f[3];
                if (!is_proxy_vt(buf[i]) || !peek(c + i * 8 + 0x70, f, 12)) continue;
                const V3 q{f[0], f[1], f[2]}, v{q.x - cam.x, q.y - cam.y, q.z - cam.z};
                const float d = dist(q, cam), along = v.x * fwd.x + v.y * fwd.y + v.z * fwd.z;
                const float off = std::sqrt(std::max(0.f, d * d - along * along));
                if (d > 1.5f && d < 8.f && along > 0 && off < best_off) best_off = off, best = c + i * 8;
            }
        }
    }
#else
    (void)cam, (void)fwd;
#endif
    return best;
}
// the main camera follows the player (3-5 m); the title screen has a camera 18 m away while the player already exists
bool in_world(const V3& cam, const V3& p) { return dist(cam, p) < 8.f; }
bool player_pos(V3& p) {
    const uint64_t proxy = player_proxy();
    float f[3];
    if (!proxy || !peek(proxy + 0x70, f, 12) || !std::isfinite(f[0]) || !std::isfinite(f[1]) || !std::isfinite(f[2])) return false;
    p = {f[0], f[1], f[2]};
    return true;
}
bool teleport(float x, float y, float z) {
    const uint64_t proxy = player_proxy();
    const float v[3] = {x, y, z};
    return proxy && poke(proxy + 0x70, v, 12);
}

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
constexpr uint32_t kCross = 0x4000, kCircle = 0x2000, kR3 = 0x4;

// goto tuning (metres, seconds, radians)
constexpr float kLook = 2.f, kLookPerMps = 0.3f;  // pure-pursuit look-ahead along the route: kLook + kLookPerMps * speed
constexpr float kPass = 1.f;                      // a pass-through waypoint counts as reached this close at the latest
constexpr float kSmooth = 0.1f;                   // stick heading low-pass time constant
constexpr float kSprintTurn = 0.9f, kSprintSee = 5.f, kSprintEnd = 6.f;  // no sprint into a turn > ~50 deg within 5 m or the last 6 m
constexpr double kSprintMin = 1.0;                // hold circle at least this long: released earlier it is a dodge
constexpr int kChain = 16;                        // route points looked at ahead

enum class K { Goto, Face, WaitLoad, Wait, Press, Tp, Ladder, FailFast };
struct Cmd {
    K k = K::Wait;
    float x = 0, y = 0, z = 0, tol = 0.8f, limit = 0, fx = NAN, fz = NAN;  // limit: goto t=<s>; fx fz: ladder facing point
    bool exact = false, down = false;  // goto: explicit tol = stop there; ladder: climb down
    uint32_t btn = 0;
    int id = 0;
    std::string text;
};
std::deque<Cmd> g_q;
int g_next_id = 1;
bool g_active = false, g_failfast = false;  // "failfast 1": a failed command drops the queue
Cmd g_cur;
double g_t0 = 0;            // command start
double g_win_t = 0;         // goto: stuck window start, distance to target then
float g_win_d = 0;
int g_stuck = 0;            // goto: stuck events so far (since the last > 1 m gain on the best distance g_stuck_d)
float g_stuck_d = 1e9f, g_side = -1;  // ...; the side recovery steps to (+ = right)
double g_rec_until = 0, g_rec_side = 0;  // goto: recovery manoeuvre until then, its second leg from g_rec_side on
float g_rec_h1 = 0, g_rec_h2 = 0;        // its two world headings
V3 g_a;                     // goto: start of the current route segment
bool g_a_ok = false;
float g_head = NAN;         // goto: smoothed stick heading (world)
double g_head_t = 0;
V3 g_sp;                    // goto: speed estimate (horizontal m/s) from the position g_sp at g_sp_t
double g_sp_t = -1;
float g_speed = 0;
bool g_sprint = false;      // goto: circle held since g_sprint_t
double g_sprint_t = 0;
float g_stk = 0;            // stick magnitude nav applied at the last poll (telemetry)
double g_play_since = -1;   // wait_load: play state continuously since; a load seen since the command started
bool g_load_seen = false;
float g_face_ang = NAN;     // face: |angle| at g_face_t; R3 pressed (lock-on release)
double g_face_t = 0;
bool g_face_r3 = false;
int g_phase = 0;            // ladder / tp
V3 g_mark;                  // ladder: position when y last moved, at g_mark_t; tp: last position (still since g_mark_t), then the write time
double g_mark_t = 0;
float g_lad_y0 = 0;         // ladder: start height

// 1 = ok, 0 = a nav verb with bad arguments, -1 = not a nav command
int parse(const std::string& line, Cmd& c) {
    std::istringstream in(line);
    std::string w;
    if (!(in >> w)) return -1;
    c.text = line;
    if (w == "goto") {
        c.k = K::Goto;
        if (!(in >> c.x >> c.z)) return 0;
        for (std::string o; in >> o;) {  // "sprint": older routes; sprint is automatic now
            char* e = nullptr;
            if (o == "sprint") continue;
            if (!o.compare(0, 2, "t=")) c.limit = std::strtof(o.c_str() + 2, &e);
            else c.tol = std::strtof(o.c_str(), &e), c.exact = true;
            if (*e || !(c.tol > 0)) return 0;
        }
    } else if (w == "ladder") {
        c.k = K::Ladder;
        if (!(in >> c.x >> c.z)) return 0;
        for (std::string o; in >> o;)
            if (o == "down") c.down = true;
            else if (std::isnan(c.fx)) { c.fx = std::strtof(o.c_str(), nullptr); if (!(in >> c.fz)) return 0; }
            else return 0;
    } else if (w == "face") { c.k = K::Face; if (!(in >> c.x >> c.z)) return 0; }
    else if (w == "wait_load") c.k = K::WaitLoad;
    else if (w == "wait") { c.k = K::Wait; if (!(in >> c.x)) return 0; }
    else if (w == "failfast") { c.k = K::FailFast; if (!(in >> c.x)) return 0; }
    else if (w == "press") { c.k = K::Press; if (!(in >> w) || !(c.btn = button(w))) return 0; }
    else if (w == "tp") { c.k = K::Tp; if (!(in >> c.x >> c.y >> c.z)) return 0; }
    else return -1;
    return 1;
}

void push(Cmd c) {  // ladder = an exact goto to the ladder's foot (or top), then the climb
    if (c.k == K::Ladder) {
        Cmd g = c;
        char t[64];
        std::snprintf(t, sizeof t, "goto %g %g 0.35", double(c.x), double(c.z));
        g.k = K::Goto, g.tol = 0.35f, g.exact = true, g.text = t, g.id = g_next_id++;
        g_q.push_back(g);
    }
    c.id = g_next_id++;
    g_q.push_back(c);
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
        while (!l.empty() && (l.back() == '\n' || l.back() == '\r' || l.back() == ' ')) l.pop_back();
        Cmd c;
        if (const int ok = parse(l, c); ok > 0) push(c);
        else if (!ok) std::fprintf(stderr, "nav: bad route line: %s\n", l.c_str());
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

void finish(double now, const std::string& what) {
    ev(now, g_cur.id, what.c_str(), g_cur.text);
    g_active = false;
    if (!g_failfast || what.compare(0, 4, "fail") || g_q.empty()) return;
    ev(now, g_cur.id, ("dropped " + std::to_string(g_q.size()) + " queued").c_str(), g_cur.text);  // fail-fast: they assume this one worked
    g_q.clear();
}

// The goto route: segment start, the current goto, then the queued gotos that follow (passed through without stopping) up to the
// first stop: a goto with an explicit tolerance or the last one before another command. stop: the last point is that stop.
int route(V3* pts, bool& stop) {
    pts[0] = g_a, pts[1] = {g_cur.x, 0, g_cur.z};
    int n = 2;
    stop = true;
    if (g_cur.exact) return n;
    for (const Cmd& q : g_q) {
        if (q.k != K::Goto) break;
        if (n == kChain) { stop = false; break; }
        pts[n++] = {q.x, 0, q.z};
        if (q.exact) break;
    }
    return n;
}

void step_goto(PadSnapshot& s, double now, const V3& p, const V3& f, const V3& r) {
    const V3 me{p.x, 0, p.z};
    if (!g_a_ok) g_a = me, g_a_ok = true;
    if (g_sp_t < 0) g_sp = me, g_sp_t = now;
    else if (now - g_sp_t >= 0.2) g_speed = dist(me, g_sp) / float(now - g_sp_t), g_sp = me, g_sp_t = now;
    V3 pts[kChain];
    bool stop;
    int n = route(pts, stop);
    // pass-through: switch to the next waypoint once its segment is the nearer one (or the waypoint is close); no stop, no reset.
    // Stuck twice within 3 m of a pass-through waypoint (an obstacle on it): skip it.
    for (char b[48]; n >= 3; n = route(pts, stop)) {
        const float db = dist(me, pts[1]), tbc = seg_t(me, pts[1], pts[2]);
        const bool skip = g_stuck >= 2 && db < 3;
        if (!(skip || db < kPass || (tbc > 0 && dist(me, lerp(pts[1], pts[2], tbc)) < seg_dist(me, pts[0], pts[1]) && db < kLook + kLookPerMps * g_speed + 1)))
            break;
        std::snprintf(b, sizeof b, "done d=%.2f %s", double(db), skip ? "skip" : "pass");
        ev(now, g_cur.id, b, g_cur.text);
        g_a = skip ? me : pts[1], g_cur = g_q.front(), g_q.pop_front();
        g_t0 = now, g_win_d = -1, g_stuck = 0, g_stuck_d = 1e9f, g_rec_until = 0;
        ev(now, g_cur.id, "start", g_cur.text);
    }
    const float d = dist(me, pts[1]);
    char dd[32];
    std::snprintf(dd, sizeof dd, " d=%.2f", double(d));
    if (n == 2 && d < g_cur.tol) { finish(now, std::string("done") + dd); return; }
    if (g_cur.limit > 0 && now - g_t0 > g_cur.limit) { finish(now, std::string("fail timeout") + dd); return; }
    // look-ahead carrot on the route (shorter before a sharp turn, so corners are cut less), remaining length, sharpest turn ahead
    float look = kLook + kLookPerMps * g_speed;
    if (n >= 3) look = std::max(1.2f, look * (1 - turn(pts[0], pts[1], pts[2]) / 3.1416f));
    V3 q = lerp(pts[0], pts[1], seg_t(me, pts[0], pts[1]));
    for (int i = 1; i < n; ++i) {
        const float l = dist(q, pts[i]);
        if (l > look) { q = lerp(q, pts[i], look / l); break; }
        look -= l, q = pts[i];
    }
    float rem = d, sharp = 0;
    for (int i = 1; i + 1 < n; ++i) {
        if (rem < kSprintSee) sharp = std::max(sharp, turn(pts[i - 1], pts[i], pts[i + 1]));
        rem += dist(pts[i], pts[i + 1]);
    }
    float tgt = heading(me, q), mag = stop ? std::clamp(rem / 1.5f, 0.45f, 1.f) : 1.f;
    bool want = !(stop && rem < kSprintEnd) && sharp < kSprintTurn && std::fabs(wrap(tgt - g_head)) < 0.5f;
    if (g_rec_until > 0 && now >= g_rec_until) g_a = me, g_rec_until = 0;  // after a recovery the route leg starts here (keeps the offset)
    if (g_rec_until == 0) {  // stuck: < 0.3 m progress toward the waypoint in 1 s
        if (g_win_d < 0) g_win_d = d, g_win_t = now;
        if (now - g_win_t >= 1.0) {
            if (g_win_d - d < 0.3f) {
                ev(now, g_cur.id, (std::string("stuck") + dd).c_str(), g_cur.text);
                // the last side-step got us > 1 m closer than ever: same side again, the count starts over; else the other side
                if (g_stuck && d < g_stuck_d - 1) g_stuck = 0;
                else if (g_stuck) g_side = -g_side;
                g_stuck_d = std::min(g_stuck_d, d);
                if (++g_stuck > 5) { finish(now, std::string("fail stuck") + dd); return; }
                // recovery: odd tries a diagonal side-step, even ones back off 0.4 s first and step sideways, longer each time; the
                // stick stays full and the heading filter blends into the new leg afterwards
                const float h = heading(me, pts[1]);
                const bool back = !(g_stuck & 1);
                g_rec_h1 = h + 3.1416f, g_rec_h2 = h + g_side * (back ? 1.5708f : 1.1f);
                g_rec_side = back ? now + 0.4 : now;
                g_rec_until = g_rec_side + 0.5 + 0.2 * g_stuck;
            }
            g_win_d = -1;
        }
    }
    if (g_rec_until > 0) tgt = now < g_rec_side ? g_rec_h1 : g_rec_h2, mag = 1, want = false;
    const float dt = float(std::clamp(now - g_head_t, 0.0, 0.2));
    g_head = std::isnan(g_head) ? tgt : g_head + wrap(tgt - g_head) * (1 - std::exp(-dt / kSmooth));
    g_head_t = now;
    stick_to(std::sin(g_head), std::cos(g_head), mag, f, r, s.lx, s.ly);
    g_stk = mag;
    // sprint by curvature: on straight stretches once moving (from standstill circle is a backstep), never released early (a dodge)
    if (want && !g_sprint && g_speed > 2.5f) g_sprint = true, g_sprint_t = now;
    else if (!want && g_sprint && now - g_sprint_t >= kSprintMin) g_sprint = false;
    if (g_sprint) s.buttons |= kCircle;
}

// Ladder at the foot (or top) position: turn toward fx fz, cross, then hold the stick up (down) until the player stops moving along
// the ladder or steps off it; settle 1 s.
void step_ladder(PadSnapshot& s, double now, const V3& p, const V3& f, const V3& r) {
    const double t = now - g_t0;
    char b[48];
    if (g_phase == 0) {
        if (t < 0.5) { if (!std::isnan(g_cur.fx)) stick_to(g_cur.fx - p.x, g_cur.fz - p.z, 0.25f, f, r, s.lx, s.ly); return; }
        s.buttons |= kCross, g_phase = 1, g_mark = p, g_mark_t = now, g_lad_y0 = p.y;
        return;
    }
    const float dy = p.y - g_lad_y0;
    std::snprintf(b, sizeof b, " dy=%.2f", double(dy));
    if (g_phase == 2) { if (now - g_mark_t > 1.0) finish(now, std::string("done") + b); return; }
    if (t < 1.8) return;  // mounting
    s.ly = g_cur.down ? 255 : 0, g_stk = 1;
    if (std::fabs(p.y - g_mark.y) > 0.15f) g_mark = p, g_mark_t = now;
    const bool off = std::hypot(p.x - g_mark.x, p.z - g_mark.z) > 0.8f;  // walked off at the end (the stick still pushes)
    if (std::fabs(dy) > 1 && (off || now - g_mark_t > 1.0)) { s.ly = 128, g_phase = 2, g_mark_t = now; return; }
    if (t > 4.5 && std::fabs(dy) < 0.5f) finish(now, std::string("fail no-ladder") + b);
    else if (t > 90) finish(now, std::string("fail timeout") + b);
}

}  // namespace

bool nav_command(const char* line) {
    if (!g_on) return false;
    std::lock_guard lk(g_m);
    if (!std::strcmp(line, "abort") || !std::strcmp(line, "clear")) {
        const double now = pad_script_clock();
        if (g_active) ev(now, g_cur.id, "fail aborted", g_cur.text);
        std::fprintf(stderr, "nav @%.2f: abort, %zu queued dropped\n", now, g_q.size());
        g_q.clear(), g_active = false;
        return true;
    }
    // hwwatch <hex addr>... (up to 4): arm the 8-byte hardware write watchpoints (crash.cpp; BB_HWWATCH_ALL=1 prints every write with the guest call chain)
    if (!std::strncmp(line, "hwwatch ", 8)) {
        uint64_t a[4];
        int n = 0;
        for (const char* p = line + 8; *p && n < 4; ) {
            char* e = nullptr;
            a[n] = std::strtoull(p, &e, 16);
            if (e == p) break;
            ++n, p = e;
        }
        if (n && hooks().hw_watch) hooks().hw_watch(a, n);
        return true;
    }
    Cmd c;
    const int ok = parse(line, c);
    if (ok > 0) push(c);
    else if (!ok) std::fprintf(stderr, "nav: bad command: %s\n", line);
    return ok >= 0;
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

// BB_ROUTE_REC: a sample every <step> m; a sample back within <step> of an earlier kept one cuts the loop since then (fights, searching,
// backtracking); the file is rewritten with the kept path, simplified (Ramer-Douglas-Peucker, 0.5 m), after every change.
void rdp(const std::vector<V3>& v, size_t a, size_t b, std::vector<char>& keep) {
    float worst = 0.5f;
    size_t w = 0;
    for (size_t i = a + 1; i < b; ++i) if (const float e = seg_dist(v[i], v[a], v[b]); e > worst) worst = e, w = i;
    if (!w) return;
    keep[w] = 1;
    rdp(v, a, w, keep), rdp(v, w, b, keep);
}
void record(const V3& p) {
    static const char* const arg = env("BB_ROUTE_REC");
    static const std::string path = arg ? std::string(arg).substr(0, std::string(arg).find(',')) : "";
    static const float step = arg && std::strchr(arg, ',') ? std::strtof(std::strchr(arg, ',') + 1, nullptr) : 3.f;
    static std::vector<V3> kept;
    if (!arg || (!kept.empty() && dist(p, kept.back()) < step)) return;
    for (size_t i = 0; i + 1 < kept.size(); ++i)
        if (dist(p, kept[i]) < step) { kept.resize(i + 1); break; }
    kept.push_back(p);
    std::vector<char> keep(kept.size(), 0);
    keep.front() = keep.back() = 1;
    rdp(kept, 0, kept.size() - 1, keep);
    if (FILE* fl = std::fopen(path.c_str(), "w")) {
        for (size_t i = 0; i < kept.size(); ++i) if (keep[i]) std::fprintf(fl, "goto %.2f %.2f\n", double(kept[i].x), double(kept[i].z));
        std::fclose(fl);
    }
}

void nav_flip(double now) {
    if (!g_on) return;
    static double tel_t = -1, lost_t = -1, lost_log = -100, scan_t = 0;
    static uint64_t scanned = 0, seen_ok = 0;  // the X last scanned for; the X whose lookup was logged as working
    std::lock_guard lk(g_m);
    if (!g_tel && env("BB_TELEMETRY")) g_tel = std::fopen(env("BB_TELEMETRY"), "w");
    if (g_flip++ == 0 && (g_god || g_nohit || g_tp_once))
        std::fprintf(stderr, "nav: DEV CHEAT ACTIVE:%s%s%s%s\n", g_god ? " godmode" : "", g_nohit ? " nohit" : "", g_tp_once ? " teleport " : "",
                     g_tp_once ? g_tp_once : "");
    if (g_nvotes) {
        int best = 0;
        for (int i = 1; i < g_nvotes; ++i) if (g_votes[i].n > g_votes[best].n) best = i;
        std::memcpy(g_view, g_votes[best].v, sizeof g_view);
        g_cam = true, g_cam_flip = g_flip, g_nvotes = 0;
    }
    V3 c, f, r, p;
    const bool cam = cam_pos(c, f, r), pos = player_pos(p), world = cam && pos && in_world(c, p);
    if (pos || !cam) {
        if (const uint64_t x = pos ? player_chr() : 0; x && x != seen_ok)
            seen_ok = x, std::fprintf(stderr, "nav: player found%s: %s\n", g_fb_x == x ? " (heap scan)" : "", lookup_state().c_str());
        lost_t = -1;
    } else if (const uint64_t x = player_chr(); lost_t < 0) lost_t = now;
    else {  // a camera but no player position: log where the lookup breaks; after 3 s scan the heap for the proxy
        if (now - lost_log >= 10)
            lost_log = now, std::fprintf(stderr, "nav @%.2f: no player position for %.0f s: %s\n", now, now - lost_t, lookup_state().c_str());
        if (x && now - lost_t > 3 && (x != scanned || now - scan_t > 10)) {  // ponytail: the scan stalls this thread ~1 s, under g_m
            scanned = x, scan_t = now;
            const auto t0 = std::chrono::steady_clock::now();
            if (const uint64_t q = scan_proxy(c, f)) g_fb_x = x, g_fb_proxy = q;
            std::fprintf(stderr, "nav @%.2f: heap scan for the player's proxy (%.2f s): %llx\n", now,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), (unsigned long long)(g_fb_x == x ? g_fb_proxy : 0));
        }
    }
    const uint64_t hb = pos ? hp_block() : 0;
    int32_t hp[2] = {};
    const bool hp_ok = hb && peek(hb + 0xf8, hp, 8);
    static int settled = 0;  // BB_DEV_TELEPORT: once the player has existed for 2 s
    if (g_tp_once && settled >= 0 && (pos && cam ? ++settled : (settled = 0)) > 60) {
        float t[3] = {};
        std::sscanf(g_tp_once, "%f,%f,%f", &t[0], &t[1], &t[2]);
        std::fprintf(stderr, "nav: dev teleport to %g,%g,%g: %s\n", t[0], t[1], t[2], teleport(t[0], t[1], t[2]) ? "ok" : "failed");
        settled = -1;
    }
    if (world) record(p);
    if (!g_tel || now - tel_t < 0.25) return;
    tel_t = now;
    std::fprintf(g_tel, "T t=%.2f f=%llu st=%s", now, (unsigned long long)g_flip, world ? "play" : cam ? "menu" : "load");
    if (cam)
        std::fprintf(g_tel, " cam=%.2f,%.2f,%.2f yaw=%.1f pitch=%.1f", c.x, c.y, c.z, std::atan2(f.x, f.z) * 57.29578f,
                     std::asin(std::clamp(f.y, -1.f, 1.f)) * 57.29578f);
    if (pos) std::fprintf(g_tel, " pos=%.2f,%.2f,%.2f", p.x, p.y, p.z);
    if (hp_ok) std::fprintf(g_tel, " hp=%d/%d", hp[0], hp[1]);
    if (uint16_t fl = 0; hb && peek(hb + 0x200, &fl, 2)) std::fprintf(g_tel, " fl=%x", fl);
    if (int32_t st = 0; hb && peek(hb + 0x134, &st, 4)) std::fprintf(g_tel, " stam=%d", st);  // EU 1.00: int32 stamina in the HP block
    if (cheats_god_fixes()) std::fprintf(g_tel, " godfix=%u", cheats_god_fixes());
    if (g_active) std::fprintf(g_tel, " cmd=%d stk=%.2f%s", g_cur.id, double(g_stk), g_sprint ? " sprint" : "");
    std::fputc('\n', g_tel);
    std::fflush(g_tel);
}

void nav_apply(PadSnapshot& s, double now) {
    if (!g_on) return;
    static bool routed = (load_route(), true);
    (void)routed;
    std::lock_guard lk(g_m);
    g_stk = 0;
    if (!g_active && !g_q.empty()) {
        g_cur = g_q.front(), g_q.pop_front(), g_active = true;
        g_t0 = g_win_t = now, g_win_d = -1, g_stuck = 0, g_stuck_d = 1e9f, g_rec_until = 0, g_play_since = -1, g_load_seen = false;
        g_a_ok = false, g_head = NAN, g_sp_t = -1, g_speed = 0, g_sprint = false, g_face_ang = NAN, g_face_r3 = false, g_phase = 0, g_mark_t = 0;
        ev(now, g_cur.id, "start", g_cur.text);
    }
    if (!g_active) { g_sprint = false; return; }
    V3 c, f, r, p;
    const bool cam = cam_pos(c, f, r), pos = player_pos(p), world = cam && pos && in_world(c, p);
    switch (g_cur.k) {
        case K::Press: s.buttons |= g_cur.btn; finish(now, "done"); break;
        case K::Wait: if (now - g_t0 >= g_cur.x) finish(now, "done"); break;
        case K::FailFast: g_failfast = g_cur.x != 0; finish(now, "done"); break;
        case K::WaitLoad:
            if (!world) { g_load_seen = true, g_play_since = -1; break; }
            if (g_play_since < 0) g_play_since = now;
            if (now - g_play_since >= 2 && (g_load_seen || now - g_t0 >= 5)) finish(now, "done");
            else if (now - g_t0 > 300) finish(now, "fail timeout");
            break;
        case K::Tp: {  // teleport once the player stands still (a write while airborne / landing does not stick), then check for 1.5 s
            // the player stays there: a wall / fence pushes back, the void lets fall (no way back: the write fails mid-air)
            char b[64];
            if (!pos) { if (now - g_t0 > 5) finish(now, "fail no-player"); break; }
            if (g_phase == 0) {
                if (g_mark_t == 0 || dist(p, g_mark) > 0.05f) g_mark = p, g_mark_t = now;
                if (now - g_mark_t < 0.3 && now - g_t0 < 3) break;
                if (!teleport(g_cur.x, g_cur.y, g_cur.z)) { finish(now, "fail no-player"); break; }
                g_phase = 1, g_mark_t = now;
                break;
            }
            const float dy = p.y - g_cur.y, moved = std::hypot(p.x - g_cur.x, p.z - g_cur.z);
            if (dy < -2 && dist(p, g_mark) > 0.3f) {  // (still at the origin: the write has not landed yet)
                std::snprintf(b, sizeof b, "fail fell dy=%.1f", double(dy));
                finish(now, b);
            } else if (moved > 1 && dist(p, g_mark) < 0.3f && now - g_mark_t > 0.3 && now - g_t0 < 4) {  // write ignored (an animation): again
                teleport(g_cur.x, g_cur.y, g_cur.z), g_mark_t = now;
            } else if (now - g_mark_t < 1.5) break;
            else if (moved > 1 || dy > 2) {
                std::snprintf(b, sizeof b, "fail moved=%.1f dy=%.1f", double(moved), double(dy));
                finish(now, b);
            } else finish(now, "done");
            break;
        }
        case K::Face: {  // turn the camera (right stick) toward x z
            if (!cam) break;
            const float dx = g_cur.x - c.x, dz = g_cur.z - c.z;
            const float ang = std::atan2(dx * r.x + dz * r.z, dx * f.x + dz * f.z);  // + = target to the right
            char b[32];
            std::snprintf(b, sizeof b, " ang=%.2f", double(ang));
            if (std::fabs(ang) < 0.06f) { finish(now, "done"); break; }
            // no progress for 0.7 s: lock-on holds the camera (the stick only switches targets): R3 once releases it
            if (std::isnan(g_face_ang)) g_face_ang = std::fabs(ang), g_face_t = now;
            else if (now - g_face_t > 0.7) {
                if (std::fabs(ang) > g_face_ang - 0.03f) {
                    if (g_face_r3) { finish(now, std::string("fail no-turn") + b); break; }
                    s.buttons |= kR3, g_face_r3 = true;
                    ev(now, g_cur.id, "lock-on? r3", g_cur.text);
                }
                g_face_ang = std::fabs(ang), g_face_t = now;
            }
            if (now - g_t0 > 8) { finish(now, std::string("fail timeout") + b); break; }
            // proportional, at least 0.45 deflection (below that the game's dead zone may swallow it; the camera lags the stick)
            s.rx = uint8_t(128.f + 127.f * std::copysign(std::clamp(std::fabs(ang) * 1.5f, 0.45f, 1.f), ang));
            break;
        }
        case K::Goto:
            if (world) step_goto(s, now, p, f, r);
            else g_win_t = now, g_win_d = -1;  // loading / menu / no player: hold, the window restarts
            break;
        case K::Ladder:
            if (world) step_ladder(s, now, p, f, r);
            break;
    }
    if (!g_active) g_sprint = false;
}

}  // namespace bb::gpu
#else  // release build: the scripted-play tools (telemetry, navigation, tp, hwwatch) are compiled out
namespace bb::gpu {
bool nav_camera_wanted() { return false; }
void nav_note_camera(const uint32_t*) {}
void nav_flip(double) {}
bool nav_command(const char*) { return false; }
void nav_apply(PadSnapshot&, double) {}
}  // namespace bb::gpu
#endif
