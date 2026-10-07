// SPDX-License-Identifier: GPL-3.0-or-later
// Host audio device (SDL3): one stream per AudioOut port; the HLE layer pushes the PCM the game outputs.
#pragma once
#include <cstdint>

namespace bb::gpu {

// Returns a stream id, or -1 if the device cannot be opened (the caller then just paces without output).
int audio_open(uint32_t rate, uint32_t channels, bool is_float);
// Queues interleaved PCM; drops it when the stream already holds more than ~250 ms (the game runs ahead of the device).
void audio_write(int id, const void* data, uint32_t bytes);
void audio_close(int id);

}  // namespace bb::gpu
