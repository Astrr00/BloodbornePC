// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpu/pad.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace bb::gpu {
namespace {

// ORBIS_PAD_BUTTON_* (ScePadData::buttons)
enum : uint32_t {
    kL3 = 0x2, kR3 = 0x4, kOptions = 0x8, kUp = 0x10, kRight = 0x20, kDown = 0x40, kLeft = 0x80, kL2 = 0x100, kR2 = 0x200,
    kL1 = 0x400, kR1 = 0x800, kTriangle = 0x1000, kCircle = 0x2000, kCross = 0x4000, kSquare = 0x8000, kTouchPad = 0x100000
};

std::atomic<uint32_t> g_pad_buttons{0}, g_key_buttons{0};
// The game samples the pad once per frame (a slow scene may run at a few fps): a tap shorter than a frame would be lost, so
// every press stays visible until one snapshot has delivered it.
std::atomic<uint32_t> g_pending{0};
void note_press(uint32_t bits) { g_pending |= bits; }
std::atomic<int> g_pad_axes[6] = {};   // SDL axis values: lx, ly, rx, ry, l2, r2
std::atomic<uint32_t> g_key_axes{0};   // bit mask: W A S D (left stick) I J K L (right stick)
SDL_Gamepad* g_pad = nullptr;

uint32_t map_button(SDL_GamepadButton b) {
    switch (b) {
        case SDL_GAMEPAD_BUTTON_SOUTH: return kCross;
        case SDL_GAMEPAD_BUTTON_EAST: return kCircle;
        case SDL_GAMEPAD_BUTTON_WEST: return kSquare;
        case SDL_GAMEPAD_BUTTON_NORTH: return kTriangle;
        case SDL_GAMEPAD_BUTTON_BACK: return kTouchPad;
        case SDL_GAMEPAD_BUTTON_START: return kOptions;
        case SDL_GAMEPAD_BUTTON_LEFT_STICK: return kL3;
        case SDL_GAMEPAD_BUTTON_RIGHT_STICK: return kR3;
        case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: return kL1;
        case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: return kR1;
        case SDL_GAMEPAD_BUTTON_DPAD_UP: return kUp;
        case SDL_GAMEPAD_BUTTON_DPAD_DOWN: return kDown;
        case SDL_GAMEPAD_BUTTON_DPAD_LEFT: return kLeft;
        case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: return kRight;
        default: return 0;
    }
}

// Keyboard layout: arrows = D-pad, WASD = left stick, IJKL = right stick, Enter/Space = Cross, Esc/Backspace = Circle,
// F = Square, R = Triangle, Q/E = L1/R1, Z/X = L2/R2, Tab = Options.
uint32_t key_button(SDL_Scancode s) {
    switch (s) {
        case SDL_SCANCODE_UP: return kUp;
        case SDL_SCANCODE_DOWN: return kDown;
        case SDL_SCANCODE_LEFT: return kLeft;
        case SDL_SCANCODE_RIGHT: return kRight;
        case SDL_SCANCODE_RETURN: case SDL_SCANCODE_SPACE: return kCross;
        case SDL_SCANCODE_ESCAPE: case SDL_SCANCODE_BACKSPACE: return kCircle;
        case SDL_SCANCODE_F: return kSquare;
        case SDL_SCANCODE_R: return kTriangle;
        case SDL_SCANCODE_Q: return kL1;
        case SDL_SCANCODE_E: return kR1;
        case SDL_SCANCODE_Z: return kL2;
        case SDL_SCANCODE_X: return kR2;
        case SDL_SCANCODE_TAB: return kOptions;
        default: return 0;
    }
}
int key_axis_bit(SDL_Scancode s) {
    switch (s) {
        case SDL_SCANCODE_W: return 0; case SDL_SCANCODE_A: return 1; case SDL_SCANCODE_S: return 2; case SDL_SCANCODE_D: return 3;
        case SDL_SCANCODE_I: return 4; case SDL_SCANCODE_J: return 5; case SDL_SCANCODE_K: return 6; case SDL_SCANCODE_L: return 7;
        default: return -1;
    }
}

uint8_t stick(int sdl) { return uint8_t(std::clamp((sdl + 32768) >> 8, 0, 255)); }

}  // namespace

void pad_handle_event(const SDL_Event& e) {
    switch (e.type) {
        case SDL_EVENT_GAMEPAD_ADDED:
            if (!g_pad) g_pad = SDL_OpenGamepad(e.gdevice.which);
            break;
        case SDL_EVENT_GAMEPAD_REMOVED:
            if (g_pad && SDL_GetGamepadID(g_pad) == e.gdevice.which) { SDL_CloseGamepad(g_pad); g_pad = nullptr; g_pad_buttons = 0; for (auto& a : g_pad_axes) a = 0; }
            break;
        case SDL_EVENT_GAMEPAD_BUTTON_DOWN: g_pad_buttons |= map_button(SDL_GamepadButton(e.gbutton.button)); note_press(map_button(SDL_GamepadButton(e.gbutton.button))); break;
        case SDL_EVENT_GAMEPAD_BUTTON_UP: g_pad_buttons &= ~map_button(SDL_GamepadButton(e.gbutton.button)); break;
        case SDL_EVENT_GAMEPAD_AXIS_MOTION:
            switch (e.gaxis.axis) {
                case SDL_GAMEPAD_AXIS_LEFTX: g_pad_axes[0] = e.gaxis.value; break;
                case SDL_GAMEPAD_AXIS_LEFTY: g_pad_axes[1] = e.gaxis.value; break;
                case SDL_GAMEPAD_AXIS_RIGHTX: g_pad_axes[2] = e.gaxis.value; break;
                case SDL_GAMEPAD_AXIS_RIGHTY: g_pad_axes[3] = e.gaxis.value; break;
                case SDL_GAMEPAD_AXIS_LEFT_TRIGGER: g_pad_axes[4] = e.gaxis.value; break;
                case SDL_GAMEPAD_AXIS_RIGHT_TRIGGER: g_pad_axes[5] = e.gaxis.value; break;
                default: break;
            }
            break;
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP: {
            const bool down = e.type == SDL_EVENT_KEY_DOWN;
            if (const uint32_t b = key_button(e.key.scancode)) { if (down) { g_key_buttons |= b; note_press(b); } else g_key_buttons &= ~b; }
            if (const int a = key_axis_bit(e.key.scancode); a >= 0) { if (down) g_key_axes |= 1u << a; else g_key_axes &= ~(1u << a); }
            break;
        }
        case SDL_EVENT_WINDOW_FOCUS_LOST: g_key_buttons = 0; g_key_axes = 0; break;
        default: break;
    }
}

// BB_PAD_SCRIPT="45:circle,53:down,..." : reproducible test input; each entry presses the button once, `seconds` after the
// first pad poll of the game (the title screen needs ~45 s). Names: up down left right cross circle square triangle l1 r1 l2 r2 options.
// "110:rx=255,112:rx=128": from then on that stick axis (lx / ly / rx / ry) holds the value (0..255, 128 = centre; ly 0 = forward).
// BB_PAD_LIVE=<file>: live steering of a running game; every complete line appended to the file is parsed like BB_PAD_SCRIPT with
// the times relative to the moment the line is read (checked every poll; "0:ly=0,2:ly=128,3:cross" walks 2 s, then presses cross).
// BB_PAD_CLOCK=flips: count the seconds as game flips / 30 instead of wall time, so a slowed-down run (low priority, busy PC) presses at the
// same game state. ponytail: assumes the reference timing was taken at 30 flips/s (menus and loading screens run at 30 here).
std::atomic<uint64_t> g_flips{0};
void pad_note_flip() { ++g_flips; }
struct ScriptedPress { double at; uint32_t bits; bool done; int axis = -1, value = 128; };  // axis 0 lx, 1 ly, 2 rx, 3 ry
int g_script_axes[4] = {-1, -1, -1, -1};  // held script values (pad_snapshot only)
std::atomic<double> g_script_now{0};       // the script clock at the last poll
double pad_script_clock() { return g_script_now; }
void parse_script(std::string s, double base, std::vector<ScriptedPress>& out) {
    static const struct { const char* n; uint32_t b; } names[] = {{"up", kUp}, {"down", kDown}, {"left", kLeft}, {"right", kRight}, {"cross", kCross}, {"circle", kCircle},
                                                                  {"square", kSquare}, {"triangle", kTriangle}, {"l1", kL1}, {"r1", kR1}, {"l2", kL2}, {"r2", kR2}, {"options", kOptions}};
    while (!s.empty()) {
        const size_t comma = s.find(',');
        const std::string item = s.substr(0, comma);
        s = comma == std::string::npos ? "" : s.substr(comma + 1);
        const size_t colon = item.find(':');
        if (colon == std::string::npos) continue;
        const double at = base + std::atof(item.substr(0, colon).c_str());
        for (const auto& n : names)
            if (item.substr(colon + 1) == n.n) out.push_back({at, n.b, false});
        static const char* const axes[4] = {"lx=", "ly=", "rx=", "ry="};
        for (int a = 0; a < 4; ++a)
            if (item.compare(colon + 1, 3, axes[a]) == 0) out.push_back({at, 0, false, a, std::atoi(item.c_str() + colon + 4)});
    }
}
std::vector<ScriptedPress>& pad_script() {
    static std::vector<ScriptedPress> v = [] {
        std::vector<ScriptedPress> out;
        const char* e = std::getenv("BB_PAD_SCRIPT");
        parse_script(e ? e : "", 0, out);
        return out;
    }();
    return v;
}
void poll_live(double now) {  // BB_PAD_LIVE (pad_snapshot only)
    static const char* const path = std::getenv("BB_PAD_LIVE");
    static long consumed = 0;  // bytes of complete lines already parsed
    if (!path) return;
    FILE* f = std::fopen(path, "rb");
    if (!f) return;
    std::string rest;
    char buf[4096];
    std::fseek(f, consumed, SEEK_SET);
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) rest.append(buf, n);
    std::fclose(f);
    for (size_t nl; (nl = rest.find('\n')) != std::string::npos; rest.erase(0, nl + 1)) {
        consumed += long(nl + 1);
        std::string line = rest.substr(0, nl);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        parse_script(line, now, pad_script());
        std::fprintf(stderr, "pad-live @%.2f: %s\n", now, line.c_str());
    }
}

PadSnapshot pad_snapshot() {
    PadSnapshot s;
    static const uint64_t t0 = SDL_GetTicks();
    static const bool by_flips = std::getenv("BB_PAD_CLOCK") && std::string(std::getenv("BB_PAD_CLOCK")) == "flips";
    static const uint64_t f0 = g_flips.load();
    const double now = by_flips ? double(g_flips.load() - f0) / 30.0 : double(SDL_GetTicks() - t0) / 1000.0;
    g_script_now = now;
    static std::mutex script_mutex;  // pad_snapshot may run on any thread; the live input appends to the script
    std::lock_guard<std::mutex> lk(script_mutex);
    poll_live(now);
    for (auto& p : pad_script())
        if (!p.done && now >= p.at && p.axis >= 0) { p.done = true; g_script_axes[p.axis] = std::clamp(p.value, 0, 255); }
    for (auto& p : pad_script())
        if (!p.done && now >= p.at && p.axis < 0) { p.done = true; g_pending |= p.bits; break; }  // one press per poll
    s.buttons = g_pad_buttons | g_key_buttons | g_pending.exchange(0);
    const uint32_t k = g_key_axes;
    auto axis = [&](int pad_axis, int neg_bit, int pos_bit) {
        const int v = g_pad_axes[pad_axis];
        if (std::abs(v) > 4096) return stick(v);  // gamepad outside the dead zone wins
        const int dir = ((k >> pos_bit) & 1) - ((k >> neg_bit) & 1);
        return uint8_t(dir < 0 ? 0 : dir > 0 ? 255 : 128);
    };
    s.lx = axis(0, 1, 3);  // A / D
    s.ly = axis(1, 0, 2);  // W / S (up = 0)
    s.rx = axis(2, 5, 7);  // J / L
    s.ry = axis(3, 4, 6);  // I / K
    uint8_t* const st[4] = {&s.lx, &s.ly, &s.rx, &s.ry};
    for (int a = 0; a < 4; ++a)
        if (g_script_axes[a] >= 0 && *st[a] == 128) *st[a] = uint8_t(g_script_axes[a]);  // a real stick wins
    s.l2 = uint8_t(std::max(0, g_pad_axes[4].load()) >> 7);
    s.r2 = uint8_t(std::max(0, g_pad_axes[5].load()) >> 7);
    if (s.l2 > 30) s.buttons |= kL2;
    if (s.r2 > 30) s.buttons |= kR2;
    if (s.buttons & kL2 && !s.l2) s.l2 = 255;
    if (s.buttons & kR2 && !s.r2) s.r2 = 255;
    return s;
}

}  // namespace bb::gpu
