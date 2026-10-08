// SPDX-License-Identifier: GPL-3.0-or-later
// Player-facing cheats (BB_CHEATS) and the guest-memory helpers shared with the dev tools in nav.cpp.
#pragma once
#include <cstddef>
#include <cstdint>

namespace bb::gpu {

// Guest reads/writes that tolerate unmapped addresses (host address = guest address).
bool peek(uint64_t a, void* out, size_t n);
bool poke(uint64_t a, const void* in, size_t n);
uint64_t ld64(uint64_t a);

// The player manager, the player character X and its HP block, only when every vtable / owner check holds (else 0). The signature
// matches EU 1.00 and EU 1.09 (the pointer path to X is the same); the HP block's field offsets are proven on EU 1.00 only.
uint64_t player_mgr();
uint64_t player_chr();
uint64_t hp_block();

enum : uint32_t { kCheatGod = 1, kCheatNoHit = 2, kCheatStamina = 4 };
void cheats_enable(uint32_t bits);  // adds to BB_CHEATS; only before the first flip (nav.cpp's BB_DEV_GODMODE / BB_DEV_NOHIT)
void cheats_flip();                 // guest flip: applies the cheats; does nothing (and reads nothing) unless one is enabled
uint32_t cheats_god_fixes();        // how often god had to refill HP (dev telemetry)

}  // namespace bb::gpu
