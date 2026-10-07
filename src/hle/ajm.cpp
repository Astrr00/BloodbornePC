// SPDX-License-Identifier: GPL-3.0-or-later
// libSceAjm (audio decode jobs). The engine places consecutive jobs at the SDK's own record sizes (observed: control job
// 0x40 bytes, run job 0x50), so the jobs are kept in a side table keyed by their batch address and the calls return the
// SDK sizes; sceAjmBatchStartBuffer runs every job whose address lies inside [batch, batch + size). Decoding is not implemented yet (ATRAC9/MP3
// via LibAtrac9 arrive with the audio milestone): a run job reports its whole input as consumed and fills the output
// with silence, so the engine's streaming logic keeps advancing instead of waiting for a batch that never completes.
// Sideband layouts follow the SDK as documented by shadPS4 (src/core/libraries/ajm/ajm.h, ajm_instance.cpp).
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <mutex>

#include "hle/abi.h"

namespace bb::hle {
namespace {

struct Rec {
    uint32_t kind, instance;  // 1 control, 2 run, 3 inline data (payload follows), 4 run with split buffers
    uint64_t flags, in, in_size, out, out_size, sb_in, sb_in_size, sb_out, sb_out_size, pad[2];
};
static_assert(sizeof(Rec) == 96);

std::mutex g_m;
std::map<uint64_t, Rec> g_jobs;  // batch address -> job
struct At9Info { uint32_t superframe_size = 0, frames_in_superframe = 1, frame_samples = 0, channels = 2, sample_rate = 48000; uint8_t config[4] = {}; };
std::map<uint32_t, uint64_t> g_total_samples;  // instance id -> samples reported so far (AjmSidebandStream.total_decoded_samples)
std::map<uint32_t, uint64_t> g_instance_flags;  // instance id -> AjmInstanceFlags
std::map<uint32_t, uint32_t> g_codec;            // instance id -> AjmCodecType (0 MP3, 1 ATRAC9, 2 AAC)
std::map<uint32_t, At9Info> g_at9;               // instance id -> parsed ATRAC9 config (set by the initialise control job)

// ATRAC9 config data (4 bytes, big endian): sync 0xFE | sample rate idx 4 | channel cfg 3 | valid 1 | frame bytes-1 11 |
// superframe idx 2 | unused 3 (layout and the frame-samples table as in LibAtrac9, MIT).
At9Info parse_at9(const uint8_t* cfg) {
    static const uint8_t kFramePower[16] = {6, 6, 7, 7, 7, 8, 8, 8, 6, 6, 7, 7, 7, 8, 8, 8};
    static const uint8_t kChannels[8] = {1, 2, 2, 6, 8, 4, 0, 0};  // channel config index -> channel count
    static const uint32_t kRates[16] = {11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 44100, 48000, 64000, 88200, 96000, 128000, 176400, 192000};
    const uint32_t rate_idx = cfg[1] >> 4, frame_bytes = ((uint32_t(cfg[2]) << 3) | (cfg[3] >> 5)) + 1, super_idx = (cfg[3] >> 3) & 3;
    At9Info i;
    i.frames_in_superframe = 1u << super_idx;
    i.superframe_size = frame_bytes * i.frames_in_superframe;
    i.frame_samples = 1u << kFramePower[rate_idx];
    i.channels = kChannels[(cfg[1] >> 1) & 7] ? kChannels[(cfg[1] >> 1) & 7] : 2;
    i.sample_rate = kRates[rate_idx];
    std::memcpy(i.config, cfg, 4);
    return i;
}
std::atomic<uint32_t> g_next_instance{0}, g_next_batch{0};

uint64_t align16(uint64_t v) { return (v + 15) & ~uint64_t(15); }

constexpr uint64_t kControlStride = 0x40, kRunStride = 0x50, kInlineHeader = 0x10;

uint64_t append(uint64_t at, const Rec& r) {
    std::lock_guard lk(g_m);
    g_jobs[at] = r;
    return at + (r.kind == 1 ? kControlStride : kRunStride);
}
uint32_t codec_of(uint32_t instance) {
    std::lock_guard lk(g_m);
    auto it = g_codec.find(instance);
    return it == g_codec.end() ? ~0u : it->second;
}
// LibAtrac9 (MIT): one decoder handle per Ajm instance; frames decode to interleaved S16.
extern "C" {
void* Atrac9GetHandle(void);
void Atrac9ReleaseHandle(void* handle);
int Atrac9InitDecoder(void* handle, unsigned char* config_data);
int Atrac9Decode(void* handle, const unsigned char* at9, short* pcm, int* bytes_used);
}
std::map<uint32_t, void*> g_decoders;  // guarded by g_m

void release_decoder(uint32_t instance) {
    std::lock_guard lk(g_m);
    if (auto it = g_decoders.find(instance); it != g_decoders.end()) { Atrac9ReleaseHandle(it->second); g_decoders.erase(it); }
}

// Decodes `frames` whole superframes (input starts at `in`) and appends the PCM in the instance's encoding (0 S16, 1 S32, 2 float).
void decode_at9(uint32_t instance, const At9Info& a9, const uint8_t* in, uint64_t frames, uint32_t enc, std::vector<uint8_t>& out) {
    void* h;
    {
        std::lock_guard lk(g_m);
        void*& slot = g_decoders[instance];
        if (!slot) {
            slot = Atrac9GetHandle();
            unsigned char cfg[4] = {};
            std::memcpy(cfg, a9.config, 4);
            if (!slot || Atrac9InitDecoder(slot, cfg) != 0) { if (slot) Atrac9ReleaseHandle(slot); slot = nullptr; }
        }
        h = slot;
    }
    const size_t per_frame = size_t(a9.frame_samples) * a9.channels, bps = enc == 0 ? 2 : 4;
    std::vector<short> pcm(per_frame);
    std::vector<uint8_t> padded(a9.superframe_size + 8);  // LibAtrac9's bit reader peeks up to 4 bytes past the current byte without a length
    for (uint64_t f = 0; f < frames; ++f) {
        std::memcpy(padded.data(), in + f * a9.superframe_size, a9.superframe_size);
        const uint8_t* sf = padded.data();
        uint32_t off = 0;
        for (uint32_t k = 0; k < a9.frames_in_superframe; ++k) {
            int used = 0;
            if (!h || off >= a9.superframe_size || Atrac9Decode(h, sf + off, pcm.data(), &used) != 0) { std::fill(pcm.begin(), pcm.end(), short(0)); used = 0; }
            off += uint32_t(used);
            const size_t at = out.size();
            out.resize(at + per_frame * bps);
            for (size_t i = 0; i < per_frame; ++i) {
                if (enc == 0) std::memcpy(&out[at + 2 * i], &pcm[i], 2);
                else if (enc == 1) { const int32_t v = int32_t(pcm[i]) << 16; std::memcpy(&out[at + 4 * i], &v, 4); }
                else { const float v = float(pcm[i]) / 32768.f; std::memcpy(&out[at + 4 * i], &v, 4); }
            }
        }
    }
}

void run_job(Context& c, const Rec& r) {
    uint64_t inst_flags = 0;
    {
        std::lock_guard lk(g_m);
        auto it = g_instance_flags.find(r.instance);
        if (it != g_instance_flags.end()) inst_flags = it->second;
    }
    if (r.kind == 1) {  // control: result {0,0}
        const uint32_t control_flags = uint32_t(r.flags >> 13) & 7;  // 1 reset, 2 initialize, 4 resample
        if ((control_flags & 2) && r.sb_in && r.sb_in_size >= 8 && codec_of(r.instance) == 1) {  // ATRAC9
            release_decoder(r.instance);  // a new stream configuration
            std::lock_guard lk(g_m);
            g_at9[r.instance] = parse_at9(ptr<const uint8_t>(c, r.sb_in));
            if (std::getenv("BB_AJM_LOG")) {
                const At9Info& i = g_at9[r.instance];
                const uint8_t* p = ptr<const uint8_t>(c, r.sb_in);
                std::fprintf(stderr, "ajm: at9 inst=%u cfg=%02x%02x%02x%02x superframe=%u frames=%u frame_samples=%u flags=%llx\n", r.instance, p[0], p[1], p[2], p[3],
                             i.superframe_size, i.frames_in_superframe, i.frame_samples, (unsigned long long)g_instance_flags[r.instance]);
            }
        }
        if (control_flags & 1) release_decoder(r.instance);  // reset: decoder state restarts with the next stream
        if (r.sb_out && r.sb_out_size >= 8) std::memset(ptr<void>(c, r.sb_out), 0, 8);
        return;
    }
    At9Info a9;
    bool have9 = false;
    {
        std::lock_guard lk(g_m);
        auto it = g_at9.find(r.instance);
        if (it != g_at9.end() && it->second.superframe_size) { a9 = it->second; have9 = true; }
    }
    // ATRAC9: the stream's own channel layout wins over the instance flags (the flags carry a maximum, e.g. 4 for a mono stream)
    const uint32_t flag_channels = uint32_t(inst_flags >> 3) & 15 ? uint32_t(inst_flags >> 3) & 15 : 2;
    const uint32_t channels = have9 ? a9.channels : flag_channels;
    const uint32_t enc = uint32_t(inst_flags >> 7) & 7;  // 0 S16, 1 S32, 2 float
    const uint32_t bytes_per_sample = enc == 0 ? 2 : 4;
    uint64_t out_size = r.out_size;
    uint64_t in_size = r.in_size;
    // gather the input (kind 4: arrays of AjmBuffer {u8* p; u64 size}; r.in_size / r.out_size are then buffer counts)
    std::vector<uint8_t> gathered;
    const uint8_t* in_bytes = r.in ? ptr<const uint8_t>(c, r.in) : nullptr;
    if (r.kind == 4) {
        in_size = 0;
        for (uint64_t i = 0; i < r.in_size; ++i) {
            const uint64_t p = rt::ld<uint64_t>(c, r.in + 16 * i), n = rt::ld<uint64_t>(c, r.in + 16 * i + 8);
            if (p && n) gathered.insert(gathered.end(), ptr<const uint8_t>(c, p), ptr<const uint8_t>(c, p) + n);
            in_size += n;
        }
        in_bytes = gathered.data();
        out_size = 0;
        for (uint64_t i = 0; i < r.out_size; ++i) out_size += rt::ld<uint64_t>(c, r.out + 16 * i + 8);
    }
    std::vector<uint8_t> pcm;  // decoded output, in the instance's sample encoding
    if (have9 && in_bytes) {
        // whole superframes only: as many as the input holds and the output has room for
        const uint64_t sf_samples = uint64_t(a9.frame_samples) * a9.frames_in_superframe, sf_out = sf_samples * channels * bytes_per_sample;
        const uint64_t frames = std::min(in_size / a9.superframe_size, sf_out ? out_size / sf_out : 0);
        decode_at9(r.instance, a9, in_bytes, frames, enc, pcm);
        in_size = frames * a9.superframe_size;
        out_size = frames * sf_out;
        if (static const bool log = std::getenv("BB_AJM_LOG") != nullptr; log) {  // diagnosis: is real audio coming out?
            static std::atomic<uint32_t> n{0};
            double peak = 0;
            for (size_t i = 0; i + bytes_per_sample <= pcm.size(); i += bytes_per_sample) {
                double v;
                if (enc == 0) { int16_t s; std::memcpy(&s, &pcm[i], 2); v = s / 32768.0; }
                else if (enc == 1) { int32_t s; std::memcpy(&s, &pcm[i], 4); v = s / 2147483648.0; }
                else { float s; std::memcpy(&s, &pcm[i], 4); v = s; }
                peak = std::max(peak, std::abs(v));
            }
            const uint32_t job = n++;
            if (job % 100 == 0) std::fprintf(stderr, "ajm: decode inst=%u superframes=%llu in=%llu out=%llu peak=%.3f (job #%u)\n", r.instance, (unsigned long long)frames, (unsigned long long)in_size, (unsigned long long)out_size, peak, job);
        }
    }
    pcm.resize(size_t(out_size), 0);  // no decoder (MP3/AAC/unknown stream): silence
    if (r.kind == 4) {
        uint64_t done = 0;
        for (uint64_t i = 0; i < r.out_size; ++i) {
            const uint64_t p = rt::ld<uint64_t>(c, r.out + 16 * i), n = rt::ld<uint64_t>(c, r.out + 16 * i + 8);
            if (!p || !n) continue;
            const uint64_t take = std::min<uint64_t>(n, pcm.size() - std::min<uint64_t>(done, pcm.size()));
            if (take) std::memcpy(ptr<void>(c, p), pcm.data() + done, size_t(take));
            if (take < n) std::memset(ptr<uint8_t>(c, p) + take, 0, size_t(n - take));
            done += n;
        }
    } else if (r.out && r.out_size) {
        std::memcpy(ptr<void>(c, r.out), pcm.data(), pcm.size());
        if (pcm.size() < r.out_size) std::memset(ptr<uint8_t>(c, r.out) + pcm.size(), 0, size_t(r.out_size - pcm.size()));
    }
    uint64_t total_samples;
    {
        std::lock_guard lk(g_m);
        total_samples = g_total_samples[r.instance] += out_size / (uint64_t(channels) * bytes_per_sample);
    }
    // sideband output in SDK order: result, [stream], [format], [gapless], [mframe], [codec info]
    if (!r.sb_out) return;
    const bool multiple_frames = (r.flags >> 11) & 2, codec_info = (r.flags >> 11) & 1;  // run flags at bits 11..12
    const uint32_t sideband_flags = uint32_t(r.flags >> 45) & 7;  // 1 gapless, 2 format, 4 stream
    uint64_t at = r.sb_out;
    const uint64_t end = r.sb_out + r.sb_out_size;
    auto room = [&](uint64_t n) { return at + n <= end; };
    if (room(8)) { rt::st<int32_t>(c, at, 0); rt::st<int32_t>(c, at + 4, 0); at += 8; }
    if ((sideband_flags & 4) && room(16)) {
        rt::st<int32_t>(c, at, int32_t(in_size));
        rt::st<int32_t>(c, at + 4, int32_t(out_size));
        rt::st<uint64_t>(c, at + 8, total_samples);  // cumulative per instance
        at += 16;
    }
    if ((sideband_flags & 2) && room(24)) {
        rt::st<uint32_t>(c, at, channels);
        rt::st<uint32_t>(c, at + 4, channels == 1 ? 4 : channels == 2 ? 3 : 0x63F);  // front mask for 1/2 channels
        rt::st<uint32_t>(c, at + 8, have9 ? a9.sample_rate : 48000);
        rt::st<uint32_t>(c, at + 12, enc);
        rt::st<uint32_t>(c, at + 16, 0);
        rt::st<uint32_t>(c, at + 20, 0);
        at += 24;
    }
    if ((sideband_flags & 1) && room(8)) { std::memset(ptr<void>(c, at), 0, 8); at += 8; }
    if (multiple_frames && room(8)) { rt::st<uint32_t>(c, at, 0); rt::st<uint32_t>(c, at + 4, 0); at += 8; }
    if (codec_info && room(16)) {  // AjmSidebandDecAt9CodecInfo {superframe size, frames in superframe, next frame size, frame samples}
        At9Info i;
        {
            std::lock_guard lk(g_m);
            auto it = g_at9.find(r.instance);
            if (it != g_at9.end()) i = it->second;
        }
        rt::st<uint32_t>(c, at, i.superframe_size);
        rt::st<uint32_t>(c, at + 4, i.frames_in_superframe);
        rt::st<uint32_t>(c, at + 8, i.superframe_size);
        rt::st<uint32_t>(c, at + 12, i.frame_samples);
    }
}

} // namespace

void register_ajm() {
    reg("sceAjmInitialize", [](Context& c) { rt::st<uint32_t>(c, arg(c, 1), 1); ret(c, 0); });  // (reserved, uint32_t* ctx)
    reg("sceAjmFinalize", [](Context& c) { ret(c, 0); });
    reg("sceAjmModuleRegister", [](Context& c) { ret(c, 0); });
    reg("sceAjmModuleUnregister", [](Context& c) { ret(c, 0); });
    reg("sceAjmMemoryRegister", [](Context& c) { ret(c, 0); });
    reg("sceAjmMemoryUnregister", [](Context& c) { ret(c, 0); });
    reg("sceAjmInstanceCreate", [](Context& c) {  // (ctx, codec, flags, uint32_t* instance)
        const uint32_t id = ++g_next_instance;
        {
            std::lock_guard lk(g_m);
            g_instance_flags[id] = arg(c, 2);
            g_codec[id] = arg32(c, 1);
        }
        rt::st<uint32_t>(c, arg(c, 3), id);
        ret(c, 0);
    });
    reg("sceAjmInstanceDestroy", [](Context& c) {
        std::lock_guard lk(g_m);
        g_instance_flags.erase(arg32(c, 1));
        ret(c, 0);
    });
    // (void* buf, u32 instance, u64 flags, void* sbIn, size sbInSize, void* sbOut, size sbOutSize, void* returnAddress)
    reg("sceAjmBatchJobControlBufferRa", [](Context& c) {
        Rec r{};
        if (std::getenv("BB_AJM_LOG")) {
            std::fprintf(stderr, "ajm: control args:");
            for (int i = 0; i < 9; ++i) std::fprintf(stderr, " %llx", (unsigned long long)arg(c, i));
            std::fputc('\n', stderr);
        }
        r.kind = 1; r.instance = arg32(c, 1); r.flags = arg(c, 2);
        r.sb_in = arg(c, 3); r.sb_in_size = arg(c, 4); r.sb_out = arg(c, 5); r.sb_out_size = arg(c, 6);
        ret(c, append(arg(c, 0), r));
    });
    // (buf, instance, flags, in, inSize, out, outSize, sbOut, sbOutSize, returnAddress)
    reg("sceAjmBatchJobRunBufferRa", [](Context& c) {
        Rec r{};
        r.kind = 2; r.instance = arg32(c, 1); r.flags = arg(c, 2);
        r.in = arg(c, 3); r.in_size = arg(c, 4); r.out = arg(c, 5); r.out_size = arg(c, 6);
        r.sb_out = arg(c, 7); r.sb_out_size = arg(c, 8);
        ret(c, append(arg(c, 0), r));
    });
    // (buf, instance, flags, const AjmBuffer* in, nIn, const AjmBuffer* out, nOut, sbOut, sbOutSize, returnAddress)
    reg("sceAjmBatchJobRunSplitBufferRa", [](Context& c) {
        Rec r{};
        r.kind = 4; r.instance = arg32(c, 1); r.flags = arg(c, 2);
        r.in = arg(c, 3); r.in_size = arg(c, 4); r.out = arg(c, 5); r.out_size = arg(c, 6);
        r.sb_out = arg(c, 7); r.sb_out_size = arg(c, 8);
        ret(c, append(arg(c, 0), r));
    });
    // (buf, const void* data, size, const void** batchAddress): data is copied into the batch behind a 16-byte header
    reg("sceAjmBatchJobInlineBuffer", [](Context& c) {
        const uint64_t at = arg(c, 0), n = arg(c, 2);
        if (n) std::memcpy(ptr<char>(c, at + kInlineHeader), ptr<const char>(c, arg(c, 1)), size_t(n));
        if (arg(c, 3)) rt::st<uint64_t>(c, arg(c, 3), at + kInlineHeader);
        ret(c, at + kInlineHeader + align16(n));
    });
    // (u32 context, u8* batch, u32 size, int priority, AjmBatchError* error, u32* batchId): jobs run synchronously
    reg("sceAjmBatchStartBuffer", [](Context& c) {
        const uint64_t batch = arg(c, 1), size = arg32(c, 2);
        std::vector<Rec> jobs;
        {
            std::lock_guard lk(g_m);
            for (auto it = g_jobs.lower_bound(batch); it != g_jobs.end() && it->first < batch + size;) {
                jobs.push_back(it->second);
                it = g_jobs.erase(it);
            }
        }
        for (const Rec& r : jobs) {
            if (std::getenv("BB_AJM_LOG"))
                std::fprintf(stderr, "ajm: job kind=%u inst=%u flags=%llx in=%llx/%llu out=%llx/%llu sbin=%llx/%llu sbout=%llx/%llu\n", r.kind, r.instance,
                             (unsigned long long)r.flags, (unsigned long long)r.in, (unsigned long long)r.in_size, (unsigned long long)r.out,
                             (unsigned long long)r.out_size, (unsigned long long)r.sb_in, (unsigned long long)r.sb_in_size,
                             (unsigned long long)r.sb_out, (unsigned long long)r.sb_out_size);
            run_job(c, r);
        }
        if (arg(c, 4)) std::memset(ptr<void>(c, arg(c, 4)), 0, 32);  // AjmBatchError: no error
        if (arg(c, 5)) rt::st<uint32_t>(c, arg(c, 5), ++g_next_batch);
        ret(c, 0);
    });
    reg("sceAjmBatchWait", [](Context& c) {  // (context, batchId, timeout, AjmBatchError*)
        if (arg(c, 3)) std::memset(ptr<void>(c, arg(c, 3)), 0, 32);
        ret(c, 0);
    });
    reg("sceAjmBatchCancel", [](Context& c) { ret(c, 0); });
}

} // namespace bb::hle
