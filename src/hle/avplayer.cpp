// SPDX-License-Identifier: GPL-3.0-or-later
// libSceAvPlayer: the pre-rendered movies (dvdroot_ps4/movie: title attract sprj_advertise, sprj_opening, staff rolls; H.264 High
// 1080p + AAC-LC 5.1 in MP4). Decoding sits behind the dec_* functions: Media Foundation on Windows (the OS's H.264/AAC decoders);
// elsewhere there is no backend, sceAvPlayerInit returns NULL and the game skips the movie.
// Layouts (SceAvPlayerInitData, SceAvPlayerStreamInfo, SceAvPlayerFrameInfo / FrameInfoEx, event ids) follow shadPS4's avplayer.h and
// match the game's accesses: f_290e0e0 (Init / AddSource), f_290f9c0 (stream selection on READY), f_290eae0 (GetVideoDataEx),
// f_2910710 (audio thread: GetAudioData), f_290e5b0 (event callback -> queue polled by f_290e940).
// ponytail: no seeking, no timed text, wall-clock pacing per player (video frames late by > kLateMs are dropped); audio is downmixed
// to 16-bit stereo. Add sceAvPlayerJumpToTime etc. if a caller appears.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

#include "hle/hle.h"
#include "hle/vfs.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>
#endif

namespace bb::hle {
namespace {

// ---- decoder backend --------------------------------------------------------------------------------------------------------------
struct StreamDesc {
    uint32_t type;           // SceAvPlayerStreamType: 0 video, 1 audio, 2 timed text / other
    uint32_t width, height;  // video
    uint32_t rate;           // audio (decoded to 16-bit stereo)
};
struct Decoder;
// Opens a movie and lists its streams in container order; nullptr if it cannot be decoded.
Decoder* dec_open(const std::filesystem::path& file, std::vector<StreamDesc>& streams, uint64_t& duration_ms);
bool dec_start(Decoder* d, int video, int audio);  // selects the streams (-1: none) and the output formats
bool dec_next_video(Decoder* d, uint64_t& pts_ms);  // decodes the next frame (held for dec_copy_video); false at the end
void dec_copy_video(Decoder* d, uint8_t* nv12, uint32_t width, uint32_t height);  // NV12, pitch = width, chroma right after luma
bool dec_next_audio(Decoder* d, std::vector<int16_t>& stereo, uint64_t& pts_ms);  // appends the next chunk; false at the end
void dec_rewind(Decoder* d);
void dec_close(Decoder* d);

#ifdef _WIN32
constexpr bool kHaveDecoder = true;
using Microsoft::WRL::ComPtr;

struct Decoder {
    std::wstring url;
    ComPtr<IMFSourceReader> v, a;  // one reader per stream: video (game thread) and audio (the game's audio thread) never share one
    DWORD vs = 0, as = 0;         // Media Foundation stream indices
    std::vector<DWORD> mf_index;  // player stream index (container track order) -> Media Foundation stream index
    UINT32 pitch = 0, rows = 0;  // decoded NV12 buffer: bytes per row, rows per plane (coded height, e.g. 1088)
    UINT32 channels = 2;
    ComPtr<IMFSample> held;
};

bool mf_ready() {
    thread_local const bool com = [] { CoInitializeEx(nullptr, COINIT_MULTITHREADED); return true; }();  // RPC_E_CHANGED_MODE: STA works too
    static const bool mf = SUCCEEDED(MFStartup(MF_VERSION, MFSTARTUP_LITE));
    return com && mf;
}

ComPtr<IMFSourceReader> make_reader(const std::wstring& url) {
    ComPtr<IMFSourceReader> r;
    if (FAILED(MFCreateSourceReaderFromURL(url.c_str(), nullptr, &r))) return nullptr;
    r->SetStreamSelection(DWORD(MF_SOURCE_READER_ALL_STREAMS), FALSE);
    return r;
}

ComPtr<IMFMediaType> partial_type(const GUID& major, const GUID& sub) {
    ComPtr<IMFMediaType> t;
    if (FAILED(MFCreateMediaType(&t))) return nullptr;
    t->SetGUID(MF_MT_MAJOR_TYPE, major);
    t->SetGUID(MF_MT_SUBTYPE, sub);
    return t;
}

void video_format(Decoder* d) {
    ComPtr<IMFMediaType> t;
    if (FAILED(d->v->GetCurrentMediaType(d->vs, &t))) return;
    UINT32 w = 0;
    MFGetAttributeSize(t.Get(), MF_MT_FRAME_SIZE, &w, &d->rows);
    d->pitch = MFGetAttributeUINT32(t.Get(), MF_MT_DEFAULT_STRIDE, w);  // fallback when the buffer has no IMF2DBuffer
}

// Next sample of a reader; false at the end of the stream or on an error.
bool read(IMFSourceReader* r, DWORD stream, ComPtr<IMFSample>& s, uint64_t& pts_ms, Decoder* d) {
    for (;;) {
        DWORD flags = 0;
        LONGLONG ts = 0;
        s.Reset();
        if (FAILED(r->ReadSample(stream, 0, nullptr, &flags, &ts, &s)) || (flags & (MF_SOURCE_READERF_ENDOFSTREAM | MF_SOURCE_READERF_ERROR)))
            return false;
        if ((flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) && d && r == d->v.Get()) video_format(d);
        if (s) {
            pts_ms = uint64_t(std::max<LONGLONG>(ts, 0)) / 10000;  // 100 ns units
            return true;
        }
    }
}

// MP4 track table in file order (moov/trak): handler type and the first sample sizes (stsz).
struct Mp4Track {
    uint32_t type;  // 0 video, 1 audio, 2 other
    std::vector<uint32_t> sizes;
};
constexpr uint32_t fourcc(const char (&s)[5]) { return uint32_t(uint8_t(s[0])) << 24 | uint32_t(uint8_t(s[1])) << 16 | uint32_t(uint8_t(s[2])) << 8 | uint8_t(s[3]); }
uint32_t be32(std::ifstream& f, uint64_t at) {
    uint8_t b[4] = {};
    f.seekg(std::streamoff(at));
    f.read(reinterpret_cast<char*>(b), 4);
    return uint32_t(b[0]) << 24 | uint32_t(b[1]) << 16 | uint32_t(b[2]) << 8 | b[3];
}
// First box `type` in [at, end): its payload [start, stop).
bool mp4_box(std::ifstream& f, uint64_t at, uint64_t end, uint32_t type, uint64_t& start, uint64_t& stop) {
    while (at + 8 <= end) {
        uint64_t size = be32(f, at), hdr = 8;
        if (size == 1) size = uint64_t(be32(f, at + 8)) << 32 | be32(f, at + 12), hdr = 16;
        else if (size == 0) size = end - at;
        if (!f || size < hdr) return false;
        if (be32(f, at + 4) == type) {
            start = at + hdr;
            stop = at + size;
            return true;
        }
        at += size;
    }
    return false;
}
std::vector<Mp4Track> mp4_tracks(const std::filesystem::path& file, size_t n) {
    std::vector<Mp4Track> out;
    std::error_code ec;
    const uint64_t size = std::filesystem::file_size(file, ec);
    std::ifstream f(file, std::ios::binary);
    uint64_t ms = 0, me = 0;
    if (ec || !f || !mp4_box(f, 0, size, fourcc("moov"), ms, me)) return out;
    for (uint64_t at = ms, ts, te; mp4_box(f, at, me, fourcc("trak"), ts, te); at = te) {
        Mp4Track t{2, {}};
        uint64_t a, b, c, e;
        if (mp4_box(f, ts, te, fourcc("mdia"), a, b)) {
            if (mp4_box(f, a, b, fourcc("hdlr"), c, e)) {
                const uint32_t h = be32(f, c + 8);
                t.type = h == fourcc("vide") ? 0 : h == fourcc("soun") ? 1 : 2;
            }
            if (mp4_box(f, a, b, fourcc("minf"), c, e) && mp4_box(f, c, e, fourcc("stbl"), a, b) && mp4_box(f, a, b, fourcc("stsz"), c, e)) {
                const uint32_t fixed = be32(f, c + 4), count = be32(f, c + 8);
                for (uint32_t i = 0; i < count && i < n; ++i) t.sizes.push_back(fixed ? fixed : be32(f, c + 12 + 4ull * i));
            }
        }
        out.push_back(std::move(t));
    }
    return out;
}
// Sizes of the first n compressed samples of a Media Foundation stream.
std::vector<uint32_t> mf_sizes(const std::wstring& url, DWORD stream, size_t n) {
    std::vector<uint32_t> out;
    auto r = make_reader(url);
    if (!r || FAILED(r->SetStreamSelection(stream, TRUE))) return out;
    ComPtr<IMFSample> s;
    uint64_t pts = 0;
    for (DWORD len = 0; out.size() < n && read(r.Get(), stream, s, pts, nullptr) && SUCCEEDED(s->GetTotalLength(&len));) out.push_back(len);
    return out;
}

Decoder* dec_open(const std::filesystem::path& file, std::vector<StreamDesc>& streams, uint64_t& duration_ms) {
    if (!mf_ready()) return nullptr;
    auto d = std::make_unique<Decoder>();
    d->url = file.wstring();
    if (!(d->v = make_reader(d->url))) return nullptr;
    std::vector<StreamDesc> mf;  // in Media Foundation's order
    for (DWORD i = 0;; ++i) {
        ComPtr<IMFMediaType> t;
        GUID major{};
        if (FAILED(d->v->GetNativeMediaType(i, 0, &t)) || FAILED(t->GetMajorType(&major))) break;
        StreamDesc s{major == MFMediaType_Video ? 0u : major == MFMediaType_Audio ? 1u : 2u, 0, 0, 0};
        if (s.type == 0) MFGetAttributeSize(t.Get(), MF_MT_FRAME_SIZE, &s.width, &s.height);
        if (s.type == 1) s.rate = MFGetAttributeUINT32(t.Get(), MF_MT_AUDIO_SAMPLES_PER_SECOND, 48000);
        mf.push_back(s);
    }
    // The console numbers streams in track order, and the game counts them per type to pick its audio track (sprj_opening: video,
    // then 8 languages). Media Foundation's order differs (sprj_opening: the audio tracks reversed, video last; its stream ids are
    // just 1..n), so each stream is matched to its MP4 track: by type, and where a type repeats by the first 512 sample sizes.
    const auto tracks = mp4_tracks(file, 512);
    std::vector<size_t> track_of(mf.size(), SIZE_MAX);
    bool mapped = tracks.size() == mf.size();
    for (size_t i = 0; mapped && i < mf.size(); ++i) {
        const auto same_type = std::count_if(tracks.begin(), tracks.end(), [&](const Mp4Track& t) { return t.type == mf[i].type; });
        const auto sizes = same_type > 1 ? mf_sizes(d->url, DWORD(i), 512) : std::vector<uint32_t>{};
        for (size_t j = 0; j < tracks.size(); ++j)
            if (tracks[j].type == mf[i].type && (same_type == 1 || tracks[j].sizes == sizes)) {
                mapped &= track_of[i] == SIZE_MAX && std::find(track_of.begin(), track_of.end(), j) == track_of.end();
                track_of[i] = j;
            }
        mapped &= track_of[i] != SIZE_MAX;
    }
    for (DWORD i = 0; i < mf.size(); ++i) d->mf_index.push_back(i);
    if (mapped) std::sort(d->mf_index.begin(), d->mf_index.end(), [&](DWORD x, DWORD y) { return track_of[x] < track_of[y]; });
    else std::fprintf(stderr, "avplayer: MP4 tracks not matched to Media Foundation streams; using its order\n");
    std::string order;
    for (const DWORD i : d->mf_index) {
        streams.push_back(mf[i]);
        order += " " + std::string(mf[i].type == 0 ? "v" : "a") + "=mf" + std::to_string(i);
    }
    std::fprintf(stderr, "avplayer: streams in track order:%s\n", order.c_str());
    PROPVARIANT var;
    PropVariantInit(&var);
    duration_ms = SUCCEEDED(d->v->GetPresentationAttribute(DWORD(MF_SOURCE_READER_MEDIASOURCE), MF_PD_DURATION, &var)) ? var.uhVal.QuadPart / 10000 : 0;
    PropVariantClear(&var);
    return d.release();
}

bool dec_start(Decoder* d, int video, int audio) {
    if (!mf_ready()) return false;
    if (video >= 0) {
        d->vs = d->mf_index[size_t(video)];
        auto t = partial_type(MFMediaType_Video, MFVideoFormat_NV12);
        if (!t || FAILED(d->v->SetStreamSelection(d->vs, TRUE)) || FAILED(d->v->SetCurrentMediaType(d->vs, nullptr, t.Get()))) return false;
        video_format(d);
    }
    if (audio >= 0) {
        d->as = d->mf_index[size_t(audio)];
        auto t = partial_type(MFMediaType_Audio, MFAudioFormat_PCM);
        ComPtr<IMFMediaType> cur;
        if (!t || FAILED(t->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16)) || !(d->a = make_reader(d->url)) || FAILED(d->a->SetStreamSelection(d->as, TRUE)) ||
            FAILED(d->a->SetCurrentMediaType(d->as, nullptr, t.Get())) || FAILED(d->a->GetCurrentMediaType(d->as, &cur)))
            return false;
        d->channels = std::max<UINT32>(1, MFGetAttributeUINT32(cur.Get(), MF_MT_AUDIO_NUM_CHANNELS, 2));
    }
    return true;
}

bool dec_next_video(Decoder* d, uint64_t& pts_ms) { return read(d->v.Get(), d->vs, d->held, pts_ms, d); }

void dec_copy_video(Decoder* d, uint8_t* dst, uint32_t w, uint32_t h) {
    ComPtr<IMFMediaBuffer> b;
    if (!d->held || FAILED(d->held->ConvertToContiguousBuffer(&b))) return;
    ComPtr<IMF2DBuffer> b2;
    BYTE* p = nullptr;
    LONG pitch = LONG(d->pitch);
    DWORD len = 0;
    const bool two_d = SUCCEEDED(b.As(&b2)) && SUCCEEDED(b2->Lock2D(&p, &pitch));
    if (!two_d && FAILED(b->Lock(&p, nullptr, &len))) return;
    if (pitch >= LONG(w) && d->rows >= h && (two_d || len >= uint64_t(pitch) * d->rows * 3 / 2)) {
        const uint8_t* uv = p + size_t(pitch) * d->rows;
        for (uint32_t y = 0; y < h; ++y) std::memcpy(dst + size_t(y) * w, p + size_t(y) * pitch, w);
        for (uint32_t y = 0; y < h / 2; ++y) std::memcpy(dst + size_t(w) * h + size_t(y) * w, uv + size_t(y) * pitch, w);
    }
    if (two_d) b2->Unlock2D();
    else b->Unlock();
}

bool dec_next_audio(Decoder* d, std::vector<int16_t>& out, uint64_t& pts_ms) {
    ComPtr<IMFSample> s;
    ComPtr<IMFMediaBuffer> b;
    BYTE* p = nullptr;
    DWORD len = 0;
    if (!read(d->a.Get(), d->as, s, pts_ms, d) || FAILED(s->ConvertToContiguousBuffer(&b)) || FAILED(b->Lock(&p, nullptr, &len))) return false;
    const auto clamp16 = [](float v) { return int16_t(std::clamp(v, -32768.0f, 32767.0f)); };
    const UINT32 ch = d->channels;
    const size_t n = len / (2 * ch);
    for (size_t i = 0; i < n; ++i) {
        const int16_t* f = reinterpret_cast<const int16_t*>(p) + i * ch;
        if (ch >= 6) {  // FL FR FC LFE BL BR: ITU-R BS.775 downmix without the LFE
            out.push_back(clamp16(f[0] + 0.7071f * (f[2] + f[4])));
            out.push_back(clamp16(f[1] + 0.7071f * (f[2] + f[5])));
        } else {  // ponytail: 3-5 channel layouts keep front L/R only
            out.push_back(f[0]);
            out.push_back(f[ch > 1 ? 1 : 0]);
        }
    }
    b->Unlock();
    return true;
}

void dec_rewind(Decoder* d) {
    PROPVARIANT var;
    PropVariantInit(&var);
    var.vt = VT_I8;
    var.hVal.QuadPart = 0;
    for (IMFSourceReader* r : {d->v.Get(), d->a.Get()})
        if (r) r->SetCurrentPosition(GUID_NULL, var);
    d->held.Reset();
}

void dec_close(Decoder* d) { delete d; }
#else
constexpr bool kHaveDecoder = false;
struct Decoder {};
Decoder* dec_open(const std::filesystem::path&, std::vector<StreamDesc>&, uint64_t&) { return nullptr; }
bool dec_start(Decoder*, int, int) { return false; }
bool dec_next_video(Decoder*, uint64_t&) { return false; }
void dec_copy_video(Decoder*, uint8_t*, uint32_t, uint32_t) {}
bool dec_next_audio(Decoder*, std::vector<int16_t>&, uint64_t&) { return false; }
void dec_rewind(Decoder*) {}
void dec_close(Decoder*) {}
#endif

// ---- libSceAvPlayer ---------------------------------------------------------------------------------------------------------------
// SCE_AVPLAYER_ERROR_* (int32, sign-extended like a 64-bit return of int)
constexpr uint64_t kErrInvalid = 0xFFFFFFFF806A0001, kErrFailed = 0xFFFFFFFF806A0002, kErrAvNoMem = 0xFFFFFFFF806A0003;
constexpr int kEvStop = 1, kEvReady = 2, kEvPlay = 3, kEvPause = 4;  // SceAvPlayerEvents
constexpr uint64_t kAudioLeadMs = 100;  // audio is handed out this far ahead of the clock (the game queues it in its own mixer)
constexpr uint64_t kLateMs = 50;        // a video frame this late is dropped for the next one
constexpr uint32_t kAudioBufBytes = 32768;
constexpr auto kPrepare = std::chrono::milliseconds(100);  // AddSource -> READY ("source parsed and buffered")
using Clock = std::chrono::steady_clock;

struct Player {
    std::mutex m;
    // SceAvPlayerInitData: memory replacement (+0x00 object, +0x08 allocate, +0x10 deallocate, +0x18 allocateTexture,
    // +0x20 deallocateTexture; allocate(object, alignment, size)), file replacement (+0x28..+0x50), event replacement (+0x50 object,
    // +0x58 callback(object, event, sourceId, data)), +0x68 numOutputVideoFramebuffers.
    uint64_t mem_obj = 0, alloc = 0, dealloc = 0, tex_alloc = 0, tex_dealloc = 0, ev_obj = 0, ev_cb = 0;
    uint32_t nfb = 2;
    Decoder* dec = nullptr;
    std::vector<StreamDesc> streams;
    uint64_t duration = 0;
    int vs = -1, as = -1;
    bool looping = false, started = false, active = false, paused = false;
    Clock::time_point t0, paused_at;
    // video: a ring of NV12 frames in the game's texture memory (GPU-visible; host writes are seen by the write watch)
    std::vector<uint64_t> frames;
    uint32_t fw = 0, fh = 0;
    size_t next = 0;
    bool v_have = false, v_end = false;
    uint64_t v_pts = 0, v_last = 0;
    // audio: two buffers from the game's allocator alternate (a frame stays valid until the next call)
    uint64_t abuf[2] = {};
    int ai = 0;
    std::vector<int16_t> pcm;
    size_t pcm_off = 0;
    uint64_t a_pts = 0;
    bool a_end = false;
    uint64_t n_video = 0, n_drop = 0, n_audio = 0, n_samples = 0, decode_us = 0;

    uint64_t now_ms() const { return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>((paused ? paused_at : Clock::now()) - t0).count()); }
    void reset_clock() {
        t0 = Clock::now();
        paused = false;
        v_have = v_end = a_end = false;
        pcm.clear();
        pcm_off = 0;
    }
    void stats(const char* when) {
        std::fprintf(stderr, "avplayer: %s t=%llu ms video %llu frames (%llu dropped, decode avg %llu us) audio %llu chunks %llu samples\n", when,
                     (unsigned long long)now_ms(), (unsigned long long)n_video, (unsigned long long)n_drop,
                     (unsigned long long)(n_video + n_drop ? decode_us / (n_video + n_drop) : 0), (unsigned long long)n_audio, (unsigned long long)n_samples);
    }
};

Player* player(Context& c) { return reinterpret_cast<Player*>(arg(c, 0)); }  // the handle is the host pointer (identity mapping)
// State changes the game requests (Start/Stop/Pause/Resume, the end seen by IsActive) report on the calling thread.
void emit(Context& c, const Player& p, int ev) {
    if (p.ev_cb) call_guest(c, p.ev_cb, p.ev_obj, uint64_t(ev), 0, 0);
}

// READY comes later from a library thread, as on the console: after AddSource the game starts its movie audio thread, and its READY
// handler (f_290f9c0) needs that thread running (state 2), else the player stays half-open and the scene closes it. Delivered
// synchronously, READY lost that race under load. ponytail: one delayed event at a time, a fixed kPrepare delay.
std::mutex g_ready_m;
std::condition_variable g_ready_cv;
const Player* g_ready = nullptr;  // pending READY
bool g_ready_busy = false;        // its callback runs right now
Clock::time_point g_ready_due;
void post_ready(const Player& p) {
    if (!p.ev_cb) return;
    std::lock_guard lk(g_ready_m);
    static const bool started = start_guest_thread(*new HostThread, 64 << 10, "AvPlayer events", 0, [](Context& tc) {  // runs for good
        std::unique_lock lk(g_ready_m);
        for (;;) {
            g_ready_cv.wait(lk, [] { return g_ready != nullptr; });
            if (g_ready_cv.wait_until(lk, g_ready_due, [] { return g_ready == nullptr; })) continue;  // closed meanwhile
            const uint64_t cb = g_ready->ev_cb, obj = g_ready->ev_obj;
            g_ready = nullptr;
            g_ready_busy = true;
            lk.unlock();
            call_guest(tc, cb, obj, uint64_t(kEvReady), 0, 0);
            lk.lock();
            g_ready_busy = false;
            g_ready_cv.notify_all();
        }
    });
    if (!started) return;
    g_ready = &p;
    g_ready_due = Clock::now() + kPrepare;
    g_ready_cv.notify_all();
}
void cancel_ready(const Player& p) {  // Close: no READY into a game object that is going away
    std::unique_lock lk(g_ready_m);
    if (g_ready == &p) g_ready = nullptr;
    g_ready_cv.notify_all();
    g_ready_cv.wait(lk, [] { return !g_ready_busy; });
}

bool video_frame(Context& c, Player& p, uint64_t info) {
    if (!p.active || p.paused || p.vs < 0 || p.frames.empty()) return false;
    const uint64_t t = p.now_ms();
    for (int drops = 0;; ++drops) {
        if (!p.v_have && !p.v_end) {
            const auto d0 = Clock::now();
            p.v_have = dec_next_video(p.dec, p.v_pts);
            p.v_end = !p.v_have;
            p.decode_us += uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - d0).count());
        }
        if (!p.v_have || p.v_pts > t) return false;
        if (p.v_pts + kLateMs >= t || drops == 4) break;  // bounded catch-up per call
        p.v_have = false;
        ++p.n_drop;
    }
    const uint64_t f = p.frames[p.next++ % p.frames.size()];
    dec_copy_video(p.dec, ptr<uint8_t>(c, f), p.fw, p.fh);
    p.v_have = false;
    p.v_last = p.v_pts;
    // SceAvPlayerFrameInfoEx: pData, reserved[4], u64 timeStamp (ms) +0x10, SceAvPlayerVideoEx +0x18: width, height, f32 aspectRatio,
    // languageCode[4], framerate +0x28, cropLeft/Right/Top/Bottom +0x2c..+0x38, pitch +0x3c, lumaBitDepth, chromaBitDepth,
    // videoFullRangeFlag +0x40..+0x42 (0x68 bytes). The game puts chroma at pData + align64(width) * height.
    std::memset(ptr<void>(c, info), 0, 0x68);
    rt::st<uint64_t>(c, info, f);
    rt::st<uint64_t>(c, info + 0x10, p.v_pts);
    rt::st<uint32_t>(c, info + 0x18, p.fw);
    rt::st<uint32_t>(c, info + 0x1c, p.fh);
    rt::st<float>(c, info + 0x20, float(p.fw) / float(p.fh));
    rt::st<uint32_t>(c, info + 0x3c, p.fw);
    rt::st<uint8_t>(c, info + 0x40, 8);
    rt::st<uint8_t>(c, info + 0x41, 8);
    if (++p.n_video % 300 == 0) p.stats("playing");
    return true;
}

bool audio_frame(Context& c, Player& p, uint64_t info) {
    if (!p.active || p.paused || p.as < 0 || p.a_end || !p.abuf[0]) return false;
    if (p.pcm_off >= p.pcm.size()) {
        p.pcm.clear();
        p.pcm_off = 0;
        if (!dec_next_audio(p.dec, p.pcm, p.a_pts)) {
            p.a_end = true;
            return false;
        }
    }
    if (p.a_pts > p.now_ms() + kAudioLeadMs) return false;
    const uint32_t rate = p.streams[size_t(p.as)].rate;
    const size_t n = std::min((p.pcm.size() - p.pcm_off) / 2, size_t(kAudioBufBytes / 4));
    const uint64_t buf = p.abuf[p.ai];
    p.ai ^= 1;
    std::memcpy(ptr<void>(c, buf), p.pcm.data() + p.pcm_off, n * 4);
    // SceAvPlayerFrameInfo: pData, reserved[4], u64 timeStamp (ms) +0x10, SceAvPlayerAudio +0x18: u16 channelCount, reserved[2],
    // u32 sampleRate, u32 size (bytes), languageCode[4] (0x28 bytes)
    std::memset(ptr<void>(c, info), 0, 0x28);
    rt::st<uint64_t>(c, info, buf);
    rt::st<uint64_t>(c, info + 0x10, p.a_pts);
    rt::st<uint16_t>(c, info + 0x18, 2);
    rt::st<uint32_t>(c, info + 0x1c, rate);
    rt::st<uint32_t>(c, info + 0x20, uint32_t(n * 4));
    p.pcm_off += n * 2;
    p.a_pts += n * 1000 / rate;
    ++p.n_audio;
    p.n_samples += n;
    return true;
}

// Movie over (called from IsActive on the game thread): restart if looping, else stop. True if it just stopped.
bool check_end(Player& p) {
    if (!p.active || p.paused) return false;
    const bool vdone = p.vs < 0 || p.v_end;
    const bool adone = p.as < 0 || p.a_end || (p.vs >= 0 && p.v_end && p.now_ms() > p.v_last + 1000);
    if (!vdone || !adone) return false;
    if (p.looping) {
        dec_rewind(p.dec);
        p.reset_clock();
        return false;
    }
    p.active = false;
    p.stats("ended");
    return true;
}

} // namespace

void register_avplayer() {
    reg("sceAvPlayerInit", [](Context& c) {  // (SceAvPlayerInitData*) -> handle
        const uint64_t d = arg(c, 0);
        if (!kHaveDecoder || !d) return ret(c, 0);
        if (rt::ld<uint64_t>(c, d + 0x30)) {  // file replacement: not used by this game
            std::fputs("avplayer: file replacement callbacks unsupported\n", stderr);
            return ret(c, 0);
        }
        auto* p = new Player;
        p->mem_obj = rt::ld<uint64_t>(c, d);
        p->alloc = rt::ld<uint64_t>(c, d + 0x08);
        p->dealloc = rt::ld<uint64_t>(c, d + 0x10);
        p->tex_alloc = rt::ld<uint64_t>(c, d + 0x18);
        p->tex_dealloc = rt::ld<uint64_t>(c, d + 0x20);
        p->ev_obj = rt::ld<uint64_t>(c, d + 0x50);
        p->ev_cb = rt::ld<uint64_t>(c, d + 0x58);
        p->nfb = uint32_t(std::clamp(rt::ld<int32_t>(c, d + 0x68), 2, 8));
        ret(c, reinterpret_cast<uint64_t>(p));
    });
    reg("sceAvPlayerPostInit", [](Context& c) { ret(c, player(c) ? 0 : kErrInvalid); });
    reg("sceAvPlayerAddSource", [](Context& c) {  // (handle, path)
        Player* p = player(c);
        if (!p || !arg(c, 1) || p->dec) return ret(c, kErrInvalid);
        const std::string path = ptr<const char>(c, arg(c, 1));
        const auto host = vfs_resolve(path);
        p->dec = host ? dec_open(*host, p->streams, p->duration) : nullptr;
        std::fprintf(stderr, "avplayer: AddSource %s -> %s, %zu streams, %llu ms\n", path.c_str(), p->dec ? "open" : "FAILED", p->streams.size(), (unsigned long long)p->duration);
        if (!p->dec) return ret(c, kErrFailed);
        for (const StreamDesc& s : p->streams)
            if (s.type == 0 && !p->fw) p->fw = s.width, p->fh = s.height;
        // frames from the game's texture allocator (GPU-visible memory), audio buffers from its general allocator
        for (uint32_t i = 0; p->fw && i < p->nfb; ++i)
            if (const uint64_t f = call_guest(c, p->tex_alloc, p->mem_obj, 256, uint64_t(p->fw) * p->fh * 3 / 2)) p->frames.push_back(f);
        for (uint64_t& b : p->abuf) b = call_guest(c, p->alloc, p->mem_obj, 64, kAudioBufBytes);
        if ((p->fw && p->frames.size() < 2) || !p->abuf[0] || !p->abuf[1]) {
            std::fprintf(stderr, "avplayer: out of guest memory (%zu frames)\n", p->frames.size());
            return ret(c, kErrAvNoMem);
        }
        post_ready(*p);
        ret(c, 0);
    });
    reg("sceAvPlayerStreamCount", [](Context& c) { Player* p = player(c); ret(c, p ? p->streams.size() : kErrInvalid); });
    reg("sceAvPlayerGetStreamInfo", [](Context& c) {  // (handle, index, SceAvPlayerStreamInfo*)
        Player* p = player(c);
        const uint32_t i = arg32(c, 1);
        const uint64_t o = arg(c, 2);
        if (!p || i >= p->streams.size() || !o) return ret(c, kErrInvalid);
        // SceAvPlayerStreamInfo: u32 type, reserved[4], details +0x08 (video: u32 width, height, f32 aspectRatio, languageCode[4];
        // audio: u16 channelCount, reserved[2], u32 sampleRate, u32 size, languageCode[4]), u64 duration (ms) +0x18, startTime +0x20
        const StreamDesc& s = p->streams[i];
        std::memset(ptr<void>(c, o), 0, 0x28);
        rt::st<uint32_t>(c, o, s.type);
        if (s.type == 0) {
            rt::st<uint32_t>(c, o + 0x08, s.width);
            rt::st<uint32_t>(c, o + 0x0c, s.height);
            rt::st<float>(c, o + 0x10, s.height ? float(s.width) / float(s.height) : 0.0f);
        } else if (s.type == 1) {
            rt::st<uint16_t>(c, o + 0x08, 2);
            rt::st<uint32_t>(c, o + 0x0c, s.rate);
        }
        rt::st<uint64_t>(c, o + 0x18, p->duration);
        ret(c, 0);
    });
    reg("sceAvPlayerEnableStream", [](Context& c) {  // (handle, index)
        Player* p = player(c);
        const uint32_t i = arg32(c, 1);
        if (!p || i >= p->streams.size()) return ret(c, kErrInvalid);
        if (p->streams[i].type == 0) p->vs = int(i);
        if (p->streams[i].type == 1) p->as = int(i);
        ret(c, 0);
    });
    reg("sceAvPlayerSetLooping", [](Context& c) { if (Player* p = player(c)) p->looping = arg8(c, 1) != 0; ret(c, 0); });
    reg("sceAvPlayerStart", [](Context& c) {
        Player* p = player(c);
        if (!p || !p->dec) return ret(c, kErrInvalid);
        {
            std::lock_guard lk(p->m);
            if (p->started) dec_rewind(p->dec);
            else if (!dec_start(p->dec, p->vs, p->as)) return ret(c, kErrFailed);
            p->started = p->active = true;
            p->reset_clock();
        }
        std::fprintf(stderr, "avplayer: Start (video stream %d, audio stream %d, %u frame buffers)\n", p->vs, p->as, unsigned(p->frames.size()));
        emit(c, *p, kEvPlay);
        ret(c, 0);
    });
    reg("sceAvPlayerStop", [](Context& c) {
        Player* p = player(c);
        if (!p) return ret(c, kErrInvalid);
        {
            std::lock_guard lk(p->m);
            if (p->active) p->stats("stopped");
            p->active = false;
        }
        emit(c, *p, kEvStop);
        ret(c, 0);
    });
    reg("sceAvPlayerPause", [](Context& c) {
        Player* p = player(c);
        if (!p) return ret(c, kErrInvalid);
        {
            std::lock_guard lk(p->m);
            if (!p->active || p->paused) return ret(c, 0);
            p->paused_at = Clock::now();
            p->paused = true;
        }
        emit(c, *p, kEvPause);
        ret(c, 0);
    });
    reg("sceAvPlayerResume", [](Context& c) {
        Player* p = player(c);
        if (!p) return ret(c, kErrInvalid);
        {
            std::lock_guard lk(p->m);
            if (!p->paused) return ret(c, 0);
            p->t0 += Clock::now() - p->paused_at;
            p->paused = false;
        }
        emit(c, *p, kEvPlay);
        ret(c, 0);
    });
    reg("sceAvPlayerIsActive", [](Context& c) {
        Player* p = player(c);
        if (!p) return ret(c, 0);
        bool stopped, active;
        {
            std::lock_guard lk(p->m);
            stopped = check_end(*p);
            active = p->active;
        }
        if (stopped) emit(c, *p, kEvStop);
        ret(c, active);
    });
    reg("sceAvPlayerGetVideoDataEx", [](Context& c) {  // (handle, SceAvPlayerFrameInfoEx*) -> bool: a new frame
        Player* p = player(c);
        if (!p || !arg(c, 1)) return ret(c, 0);
        std::lock_guard lk(p->m);
        ret(c, video_frame(c, *p, arg(c, 1)));
    });
    reg("sceAvPlayerGetAudioData", [](Context& c) {  // (handle, SceAvPlayerFrameInfo*) -> bool, from the game's audio thread
        Player* p = player(c);
        if (!p || !arg(c, 1)) return ret(c, 0);
        bool ok;
        {
            std::lock_guard lk(p->m);
            ok = audio_frame(c, *p, arg(c, 1));
        }
        if (!ok) precise_sleep_us(5000);  // the audio thread polls in a tight loop
        ret(c, ok);
    });
    reg("sceAvPlayerCurrentTime", [](Context& c) {  // ms
        Player* p = player(c);
        if (!p) return ret(c, 0);
        std::lock_guard lk(p->m);
        ret(c, p->active ? p->now_ms() : 0);
    });
    reg("sceAvPlayerClose", [](Context& c) {
        Player* p = player(c);
        if (!p) return ret(c, kErrInvalid);
        cancel_ready(*p);
        p->stats("close");
        for (const uint64_t f : p->frames) call_guest(c, p->tex_dealloc, p->mem_obj, f);
        for (const uint64_t b : p->abuf)
            if (b) call_guest(c, p->dealloc, p->mem_obj, b);
        if (p->dec) dec_close(p->dec);
        delete p;
        ret(c, 0);
    });
}

} // namespace bb::hle
