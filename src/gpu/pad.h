// SPDX-License-Identifier: GPL-3.0-or-later
// Host input (SDL3 gamepad + keyboard fallback) exposed as a PS4 pad state; events are fed from the window loop.
#pragma once
#include <cstdint>

union SDL_Event;

namespace bb::gpu {

struct PadSnapshot {
    uint32_t buttons = 0;        // ORBIS_PAD_BUTTON_* bits
    uint8_t lx = 128, ly = 128, rx = 128, ry = 128, l2 = 0, r2 = 0;
    bool connected = true;       // the keyboard always counts as a pad
};

void pad_handle_event(const SDL_Event& e);  // main thread
PadSnapshot pad_snapshot();                  // any thread
void pad_note_flip();                        // guest flip (scripted-input clock, BB_PAD_CLOCK=flips)
double pad_script_clock();                   // BB_PAD_SCRIPT's clock (seconds) at the game's last pad poll

// nav.cpp: closed-loop navigation for scripted play (debug; all no-ops unless BB_TELEMETRY / BB_PAD_LIVE / BB_ROUTE / BB_ROUTE_REC is set)
bool nav_camera_wanted();                    // the backend should pass 864-byte constant blocks to nav_note_camera
void nav_note_camera(const uint32_t* blk);   // backend: a camera-block candidate (216 dwords)
void nav_flip(double now);                  // guest flip: picks the frame's main camera, writes telemetry
bool nav_command(const char* line);         // a BB_PAD_LIVE line that is a nav command (queued) -> true
void nav_apply(PadSnapshot& s, double now);  // pad poll: the running command steers

}  // namespace bb::gpu
