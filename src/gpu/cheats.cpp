// SPDX-License-Identifier: GPL-3.0-or-later
// Player-facing cheats: BB_CHEATS=god,nohit,stamina (comma list). Applied once per guest flip, only after the player's data
// has been verified; nothing is read or written without BB_CHEATS.
//
// The constants below are for the EU 1.00 and EU 1.09 ebootS (CUSA03173). The recompiled binary carries no eboot hash, so the
// identity is the pointer-path signature itself: manager vtable 0x57301f0 at [0x593e848 (1.00) / 0x593e878 (1.09)], player vtable
// 0x578f310 at X, HP block vtable 0x5735810 with owner X (field offsets proven on 1.00 only). Another eboot fails these checks.
#include "gpu/cheats.h"

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>

namespace bb::gpu {

bool peek(uint64_t a, void* out, size_t n) {
    if (a < 0x10000) return false;
#ifdef _WIN32
    SIZE_T got = 0;
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(a), out, n, &got) && got == n;
#else
    std::memcpy(out, reinterpret_cast<const void*>(a), n);  // ponytail: no fault guard off Windows; the vtable checks read mapped guest memory only
    return true;
#endif
}
bool poke(uint64_t a, const void* in, size_t n) {
#ifdef _WIN32
    SIZE_T put = 0;
    return WriteProcessMemory(GetCurrentProcess(), reinterpret_cast<void*>(a), in, n, &put) && put == n;
#else
    std::memcpy(reinterpret_cast<void*>(a), in, n);
    return true;
#endif
}
uint64_t ld64(uint64_t a) { uint64_t v = 0; return peek(a, &v, 8) ? v : 0; }

// X = [[kGlobal] + 0x60] = the player character (re-created on respawn / load). kGlobal differs per eboot: EU 1.00 / EU 1.09 (found
// by matching the manager's constructors in both recompilations; the vtables of manager, player and HP block are the same).
constexpr uint64_t kGlobals[] = {0x593e848, 0x593e878}, kManagerVt = 0x57301f0, kPlayerVt = 0x578f310, kHpVt = 0x5735810;
uint64_t player_mgr() {
    for (const uint64_t g : kGlobals) {
        const uint64_t m = ld64(g);
        if (m && ld64(m) == kManagerVt) return m;
    }
    return 0;
}
uint64_t player_chr() {
    const uint64_t m = player_mgr(), x = m ? ld64(m + 0x60) : 0;
    return x && ld64(x) == kPlayerVt ? x : 0;
}
// HP block: an object with vtable kHpVt and owner X at +8, inside or shortly after X (seen at X+0xdc0, +0x1080, +0x2a800).
// int32 HP at +0xf8, max HP +0xfc; uint16 debug flags at +0x200 (4 NoDead, 8 NoDamage, 0x10 NoHit, 0x80 NoStaminaConsume,
// 0x100 NoMpConsume; the engine's hit/damage/death/consume code tests them).
uint64_t hp_block() {
    static uint64_t cached = 0, scanned = 0;  // the block found, the X last searched in vain
    static int retry = 0;
    const uint64_t x = player_chr();
    if (!x) return 0;
    if (cached && ld64(cached) == kHpVt && ld64(cached + 8) == x) return cached;
    cached = 0;
    if (scanned == x && ++retry < 60) return 0;  // in vain for this X: look again every 60 calls (~2 s), the block may come later
    scanned = x, retry = 0;
    static uint64_t buf[0x2000];  // 64 KB chunks over [X, X + 1 MB)
    for (uint64_t a = x; a < x + (1u << 20) && !cached; a += sizeof buf)
        if (peek(a, buf, sizeof buf))
            for (size_t i = 0; i + 1 < std::size(buf); ++i)
                if (buf[i] == kHpVt && buf[i + 1] == x) { cached = a + i * 8; break; }
    if (cached) scanned = 0;
    return cached;
}

namespace {

struct Def { const char* name; uint32_t bit; uint16_t flags; };  // flags: bits of the HP block's flag word the cheat sets
const Def kDefs[] = {{"god", kCheatGod, 0}, {"nohit", kCheatNoHit, 0x1c}, {"stamina", kCheatStamina, 0x80}};

std::atomic<uint32_t>& mask() {
    static std::atomic<uint32_t> m = [] {
        uint32_t bits = 0;
        const char* s = std::getenv("BB_CHEATS");
        for (std::string w; s && *s;) {
            w.clear();
            while (*s && *s != ',' && *s != ';' && *s != ' ') w += char(std::tolower((unsigned char)*s++));
            while (*s == ',' || *s == ';' || *s == ' ') ++s;
            uint32_t b = 0;
            for (const Def& d : kDefs) if (w == d.name) b = d.bit;
            if (b) bits |= b;
            else if (!w.empty()) std::fprintf(stderr, "cheats: unknown cheat '%s' ignored\n", w.c_str());
        }
        return bits;
    }();
    return m;
}

uint32_t g_god_fixes = 0;
}  // namespace

void cheats_enable(uint32_t bits) { mask() |= bits; }
uint32_t cheats_god_fixes() { return g_god_fixes; }

void cheats_flip() {
    static const uint32_t want = mask();
    if (!want) return;
    static uint64_t last_x = 0;
    static int settled = 0, no_block = 0;  // flips X has been the same player; flips it stayed without an HP block
    static bool warned = false;
    static std::chrono::steady_clock::time_point t0;
    static bool seen = false;  // a player has existed
    if (!t0.time_since_epoch().count()) {
        t0 = std::chrono::steady_clock::now();
        std::string l;
        for (const Def& d : kDefs) if (want & d.bit) l += std::string(l.empty() ? "" : " ") + d.name;
        std::fprintf(stderr, "cheats: %s active\n", l.c_str());
    }
    auto unavailable = [&] {
        if (warned) return;
        warned = true;
        for (const Def& d : kDefs) if (want & d.bit) std::fprintf(stderr, "cheats: %s unavailable on this build\n", d.name);
    };
    const uint64_t m = player_mgr(), x = player_chr();
    if (!m && ld64(kGlobals[0]) && ld64(kGlobals[1])) unavailable();  // both hold something else: another eboot (one is junk on each known build)
    if (!x) {
        last_x = 0, settled = 0;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(600) && !seen) unavailable();  // loading the save takes minutes
        return;
    }
    seen = true;
    if (x != last_x) last_x = x, settled = 0, no_block = 0;
    // No writes while the player is (re)built after a respawn, a lamp warp or a loading screen: X must have been stable for 2 s.
    if (settled < 60) { ++settled; return; }
    const uint64_t hb = hp_block();
    if (!hb) {
        if (++no_block > 300) unavailable();  // ~10 s, several scans in vain
        return;
    }
    no_block = 0;
    int32_t hp[2] = {};
    uint16_t fl = 0;
    if (!peek(hb + 0xf8, hp, 8) || !peek(hb + 0x200, &fl, 2)) return;
    if (hp[1] <= 0 || hp[1] > 100000 || hp[0] <= 0 || hp[0] > hp[1]) return;  // not a sane live player (dead, half-built, loading)
    if ((want & kCheatGod) && hp[0] < hp[1]) poke(hb + 0xf8, &hp[1], 4), ++g_god_fixes;  // ponytail: per flip; one hit above max HP still kills
    uint16_t need = 0;
    for (const Def& d : kDefs) if (want & d.bit) need |= d.flags;
    if ((fl & need) != need) fl |= need, poke(hb + 0x200, &fl, 2);  // ponytail: read-modify-write races the game thread, retried next flip
}

}  // namespace bb::gpu
