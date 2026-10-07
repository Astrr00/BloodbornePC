// SPDX-License-Identifier: GPL-3.0-or-later
// AudioOut. Output blocks for one buffer period like a real device; the samples go to the host sink (SDL3 in the game, see
// gpu/audio_out.cpp) when one is installed. Ajm (decoding) lives in ajm.cpp.
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "hle/abi.h"
#include "hle/hle.h"

namespace bb::hle {
namespace {

struct Port {
    uint32_t type = 0, samples = 256, rate = 48000, channels = 2, bytes_per_sample = 2;
    bool is_float = false;
    int sink = -1;
};
std::mutex g_m;
std::map<uint64_t, Port> g_ports;
AudioSink g_sink;

void audio_open(Context& c) {  // (userId, type, index, len, freq, param)
    // param low byte: 0 S16 mono, 1 S16 stereo, 2 S16 8ch, 3 float mono, 4 float stereo, 5 float 8ch, 6 S16 8ch std, 7 float 8ch std
    static std::atomic<uint64_t> next{0};
    Port p;
    p.type = arg32(c, 1);
    p.samples = arg32(c, 3) ? arg32(c, 3) : 256;
    p.rate = arg32(c, 4) ? arg32(c, 4) : 48000;
    const uint32_t fmt = arg32(c, 5) & 0xFF;
    p.is_float = fmt >= 3 && fmt != 6;
    p.bytes_per_sample = p.is_float ? 4 : 2;
    p.channels = (fmt == 0 || fmt == 3) ? 1 : (fmt == 1 || fmt == 4) ? 2 : 8;
    // only the ports a player hears: main, background music, personal (headphones); the pad speaker (4) and aux ports stay silent
    if (g_sink.open && (p.type == 0 || p.type == 1 || p.type == 3)) p.sink = g_sink.open(p.rate, p.channels, p.is_float);
    const uint64_t handle = ++next;
    {
        std::lock_guard lk(g_m);
        g_ports[handle] = p;
    }
    ret(c, handle);
}

void audio_output(Context& c) {  // (handle, ptr): ptr null = just wait
    Port p;
    {
        std::lock_guard lk(g_m);
        auto it = g_ports.find(arg(c, 0));
        if (it != g_ports.end()) p = it->second;
    }
    // absolute per-thread deadline: no drift from sleep overshoot, a long stall (loading) does not cause a burst afterwards
    thread_local std::chrono::steady_clock::time_point next{};
    const auto period = std::chrono::microseconds(uint64_t(p.samples) * 1000000 / p.rate);
    const auto now = std::chrono::steady_clock::now();
    if (next < now - std::chrono::milliseconds(100)) next = now;
    next += period;
    if (arg(c, 1) && p.sink >= 0) g_sink.write(p.sink, ptr<const void>(c, arg(c, 1)), p.samples * p.channels * p.bytes_per_sample);
    if (next > now) precise_sleep_us(uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(next - now).count()));
    ret(c, p.samples);
}

} // namespace

void set_audio_sink(const AudioSink& sink) { g_sink = sink; }

void register_audio() {
    reg("sceAudioOutInit", [](Context& c) { ret(c, 0); });
    reg("sceAudioOutOpen", audio_open);
    reg("sceAudioOutSetVolume", [](Context& c) { ret(c, 0); });
    reg("sceAudioOutOutput", audio_output);
    reg("sceAudioOutGetPortState", [](Context& c) {  // (handle, SceAudioOutPortState*): output, channel, volume, ...
        std::memset(ptr(c, arg(c, 1)), 0, 32);
        rt::st<uint16_t>(c, arg(c, 1), 1);  // output: connected
        rt::st<uint8_t>(c, arg(c, 1) + 2, 2);  // channels
        ret(c, 0);
    });
    reg("sceAudioOutClose", [](Context& c) {
        Port p;
        {
            std::lock_guard lk(g_m);
            auto it = g_ports.find(arg(c, 0));
            if (it != g_ports.end()) { p = it->second; g_ports.erase(it); }
        }
        if (p.sink >= 0) g_sink.close(p.sink);
        ret(c, 0);
    });
}

} // namespace bb::hle
