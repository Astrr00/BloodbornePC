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

}  // namespace bb::gpu
