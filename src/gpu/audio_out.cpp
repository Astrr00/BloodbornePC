// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpu/audio_out.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

namespace bb::gpu {
namespace {

struct Stream { SDL_AudioStream* s = nullptr; uint32_t bytes_per_second = 0; bool is_float = false; FILE* dump = nullptr; };
std::mutex g_m;
std::map<int, Stream> g_streams;
int g_next = 0;

}  // namespace

int audio_open(uint32_t rate, uint32_t channels, bool is_float) {
    // Background test runs never play (BB_BACKGROUND); with BB_AUDIO_DUMP they still record what would be played.
    const bool play = !std::getenv("BB_BACKGROUND");
    SDL_AudioStream* s = nullptr;
    if (play) {
        if (!SDL_WasInit(SDL_INIT_AUDIO) && !SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            std::fprintf(stderr, "audio: SDL audio init failed: %s\n", SDL_GetError());
            return -1;
        }
        SDL_AudioSpec spec{};
        spec.format = is_float ? SDL_AUDIO_F32 : SDL_AUDIO_S16;
        spec.channels = int(channels);
        spec.freq = int(rate);
        s = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
        if (!s) {
            std::fprintf(stderr, "audio: cannot open a playback stream (%u Hz, %u ch): %s\n", rate, channels, SDL_GetError());
            return -1;
        }
        SDL_ResumeAudioStreamDevice(s);
    }
    std::lock_guard lk(g_m);
    const int id = g_next++;
    FILE* dump = nullptr;
    if (const char* dir = std::getenv("BB_AUDIO_DUMP")) {  // debug: raw PCM of every stream, <dir>/stream_<id>_<rate>_<ch>_<f32|s16>.raw
        char path[512];
        std::snprintf(path, sizeof path, "%s/stream_%d_%u_%u_%s.raw", dir, id, rate, channels, is_float ? "f32" : "s16");
        dump = std::fopen(path, "wb");
    }
    g_streams[id] = {s, rate * channels * (is_float ? 4u : 2u), is_float, dump};
    return id;
}

void audio_write(int id, const void* data, uint32_t bytes) {
    // the lock covers the SDL calls: audio_close must not destroy the stream under them (SDL's own calls are thread-safe)
    std::lock_guard lk(g_m);
    auto it = g_streams.find(id);
    if (it == g_streams.end()) return;
    const Stream st = it->second;
    static const bool log = std::getenv("BB_AUDIO_LOG") != nullptr;
    static std::atomic<uint64_t> written{0}, dropped{0};
    static std::atomic<uint32_t> n{0};
    static std::atomic<float> peak{0};  // peak |sample| over all streams (diagnosis only)
    if (st.s && SDL_GetAudioStreamQueued(st.s) > int(st.bytes_per_second / 4)) {
        dropped += bytes;
        return;
    }
    if (log) {
        written += bytes;
        float pk = peak;
        if (st.is_float) for (uint32_t i = 0; i < bytes / 4; ++i) pk = std::max(pk, std::abs(static_cast<const float*>(data)[i]));
        else for (uint32_t i = 0; i < bytes / 2; ++i) pk = std::max(pk, std::abs(static_cast<const int16_t*>(data)[i]) / 32768.f);
        peak = pk;
        if (n++ % 500 == 0) std::fprintf(stderr, "audio: written %llu KB, dropped %llu KB, peak %.3f\n", (unsigned long long)(written / 1024), (unsigned long long)(dropped / 1024), double(pk));
    }
    if (st.dump) std::fwrite(data, 1, bytes, st.dump);
    if (st.s) SDL_PutAudioStreamData(st.s, data, int(bytes));
}

void audio_close(int id) {
    std::lock_guard lk(g_m);
    if (auto it = g_streams.find(id); it != g_streams.end()) {
        if (it->second.s) SDL_DestroyAudioStream(it->second.s);
        if (it->second.dump) std::fclose(it->second.dump);
        g_streams.erase(it);
    }
}

}  // namespace bb::gpu
