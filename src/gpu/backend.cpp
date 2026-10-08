// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpu/backend.h"

#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <vk_mem_alloc.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <chrono>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <smmintrin.h>  // SSE4.1 (any CPU that runs the AVX guest code has it)
#if defined(__GNUC__)
#define BB_SSE41 __attribute__((target("sse4.1")))
#else
#define BB_SSE41
#endif

#include "gpu/gcn.h"
#include "gpu/tiling.h"
#include "gpu/gcn_spirv.h"
#include "gpu/gpu_hooks.h"
#include "gpu/pad.h"
#include "gpu/spirv_builder.h"
#include "gpu/vk_context.h"
#include "gpu/write_watch.h"

namespace bb::gpu {
namespace {
bool g_log_tex = false;  // BB_GPU_DUMP_AT window: log sampled T# of the draw being recorded
bool g_replay_req = false;  // backend_request_replay (before backend_init)

constexpr uint32_t CTX(uint32_t byte) { return (byte - 0x28000) / 4; }
constexpr uint32_t SH(uint32_t byte) { return (byte - 0xB000) / 4; }
constexpr uint32_t UC(uint32_t byte) { return (byte - 0x30000) / 4; }

// context registers (byte addresses from Mesa's register database)
constexpr uint32_t kCbColor0Base = CTX(0x28c60), kCbColorStride = 15;  // BASE,PITCH,SLICE,VIEW,INFO,ATTRIB,... per target
constexpr uint32_t kCbTargetMask = CTX(0x28238), kCbBlend0 = CTX(0x28780), kCbBlendRed = CTX(0x28414);
constexpr uint32_t kPaScVportScissor0Tl = CTX(0x28250), kPaClVportXscale = CTX(0x2843c);
constexpr uint32_t kPaSuScModeCntl = CTX(0x28814), kSpiShaderColFormat = CTX(0x28714);
constexpr uint32_t kSpiPsInputAddr = CTX(0x286d0), kDbDepthControl = CTX(0x28800), kDbZInfo = CTX(0x28040);
constexpr uint32_t kSpiPsInputCntl0 = CTX(0x28644), kSpiPsInControl = CTX(0x286d8), kPaClVsOutCntl = CTX(0x28818);
constexpr uint32_t kVgtPrimType = UC(0x30908);

constexpr uint32_t kShPsLo = SH(0xb020), kShVsLo = SH(0xb120), kShCsLo = SH(0xb830);
constexpr uint32_t kShPsRsrc2 = SH(0xb02c), kShVsRsrc2 = SH(0xb12c), kShCsRsrc2 = SH(0xb84c);
constexpr uint32_t kShPsUser = SH(0xb030), kShVsUser = SH(0xb130), kShCsUser = SH(0xb900), kShCsThreadX = SH(0xb81c);
constexpr uint32_t kShLsLo = SH(0xb520), kShLsRsrc2 = SH(0xb52c), kShLsUser = SH(0xb530);  // LS stage (tessellation); HS registers are only read through kShTessHs below
constexpr uint32_t kShTessHs = 0x3C0;  // virtual SH slot written by gnm.cpp (sceGnmSetHsShader's register block): [5]/[6] HOS max/min tess level, [7] input control points

bool read_guest(uint64_t addr, uint32_t* dst, uint32_t dwords) {
    if (addr < 0x10000 || (addr & 3)) return false;
    if (hooks().mem_valid && !hooks().mem_valid(addr, uint64_t(dwords) * 4)) return false;
    std::memcpy(dst, reinterpret_cast<const void*>(addr), size_t(dwords) * 4);
    return true;
}

uint32_t f32_to_f16(float f) {  // flush denormals, truncate the mantissa; enough for clear colours
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000;
    const int32_t e = int32_t((u >> 23) & 0xFF) - 127 + 15;
    if (((u >> 23) & 0xFF) == 0xFF) return sign | 0x7C00 | ((u & 0x7FFFFF) ? 0x200 : 0);
    if (e >= 31) return sign | 0x7C00;
    if (e <= 0) return sign;
    return sign | (uint32_t(e) << 10) | ((u >> 13) & 0x3FF);  // ponytail: truncating mantissa
}

// Index range of a draw (min, max): 8 / 4 indices per step instead of a scalar cmp/cmov chain (~3 us per draw before)
BB_SSE41 void index_minmax16(const uint16_t* p, uint32_t n, uint32_t& mn, uint32_t& mx) {
    __m128i vmx = _mm_setzero_si128(), vmn = _mm_set1_epi16(-1);
    uint32_t k = 0;
    for (; k + 8 <= n; k += 8) {
        const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + k));
        vmx = _mm_max_epu16(vmx, v);
        vmn = _mm_min_epu16(vmn, v);
    }
    alignas(16) uint16_t a[8], b[8];
    _mm_store_si128(reinterpret_cast<__m128i*>(a), vmx);
    _mm_store_si128(reinterpret_cast<__m128i*>(b), vmn);
    for (int i = 0; i < 8; ++i) { mx = std::max<uint32_t>(mx, a[i]); if (k) mn = std::min<uint32_t>(mn, b[i]); }
    for (; k < n; ++k) { mx = std::max<uint32_t>(mx, p[k]); mn = std::min<uint32_t>(mn, p[k]); }
}
BB_SSE41 void index_minmax32(const uint32_t* p, uint32_t n, uint32_t& mn, uint32_t& mx) {
    __m128i vmx = _mm_setzero_si128(), vmn = _mm_set1_epi32(-1);
    uint32_t k = 0;
    for (; k + 4 <= n; k += 4) {
        const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + k));
        vmx = _mm_max_epu32(vmx, v);
        vmn = _mm_min_epu32(vmn, v);
    }
    alignas(16) uint32_t a[4], b[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(a), vmx);
    _mm_store_si128(reinterpret_cast<__m128i*>(b), vmn);
    for (int i = 0; i < 4; ++i) { mx = std::max(mx, a[i]); if (k) mn = std::min(mn, b[i]); }
    for (; k < n; ++k) { mx = std::max(mx, p[k]); mn = std::min(mn, p[k]); }
}

// 4 independent lanes (the serial hash_bytes below is latency bound); for change detection only, never for the shader-code constants
uint64_t hash_fast(const void* p, size_t n) {
    const uint64_t* w = static_cast<const uint64_t*>(p);
    uint64_t h0 = 0x243F6A8885A308D3ull, h1 = 0x13198A2E03707344ull, h2 = 0xA4093822299F31D0ull, h3 = 0x082EFA98EC4E6C89ull;
    size_t i = 0;
    const size_t words = n / 8;
    for (; i + 4 <= words; i += 4) {
        h0 = (h0 ^ w[i]) * 0x9E3779B97F4A7C15ull; h0 ^= h0 >> 32;
        h1 = (h1 ^ w[i + 1]) * 0x9E3779B97F4A7C15ull; h1 ^= h1 >> 32;
        h2 = (h2 ^ w[i + 2]) * 0x9E3779B97F4A7C15ull; h2 ^= h2 >> 32;
        h3 = (h3 ^ w[i + 3]) * 0x9E3779B97F4A7C15ull; h3 ^= h3 >> 32;
    }
    for (; i < words; ++i) { h0 = (h0 ^ w[i]) * 0x9E3779B97F4A7C15ull; h0 ^= h0 >> 32; }
    return h0 ^ (h1 * 3) ^ (h2 * 5) ^ (h3 * 7);
}

uint64_t hash_bytes(const void* p, size_t n) {  // FNV-style 64-bit over 8-byte words
    const uint64_t* w = static_cast<const uint64_t*>(p);
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n / 8; ++i) h = (h ^ w[i]) * 0x100000001b3ull, h ^= h >> 29;
    return h;
}

// Texture change detection runs once per epoch: bumped by every guest submit (the game writes what a submit samples before it submits)
// and frame. Writes by the port itself re-check only the textures they overlap (tex_touch).
std::atomic<uint64_t> g_tex_epoch{1};
std::atomic<uint64_t> g_frame_counter{0};  // frames only: schedules the full texture hashes

// Statistics clock: section timers (printed with BB_GPU_LOG, per frame with BB_FRAME_LOG) read the clock only when statistics are on
// (BB_GPU_LOG / BB_GPU_PROF / BB_FRAME_LOG); otherwise every interval is 0. ~16 reads per draw were ~3 % of the render thread.
const bool g_timing = std::getenv("BB_GPU_LOG") != nullptr || std::getenv("BB_GPU_PROF") != nullptr || std::getenv("BB_FRAME_LOG") != nullptr;
inline std::chrono::steady_clock::time_point tnow() { return g_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}; }
// Debug: adds the scope's wall time in nanoseconds to a counter (printed in ms with the BB_GPU_LOG statistics). Nanoseconds: most timed
// calls take well under a microsecond, and per-call truncation to whole microseconds dropped most of their time.
struct ScopeNs {
    uint64_t& acc;
    std::chrono::steady_clock::time_point t0 = tnow();
    ~ScopeNs() { acc += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(tnow() - t0).count()); }
};
constexpr uint64_t kVtxBudget = uint64_t(1) << 30;  // vertex-stream cache (see Backend::get_vertex_buffer)
uint64_t g_ns_shader = 0, g_ns_desc = 0, g_ns_rec = 0, g_ns_pipe = 0, g_ns_idx = 0, g_ns_early = 0;  // draw() sections: shader lookup, targets/depth setup, pipeline, index scan, descriptors, recording
template <size_t N>
struct ArrayHash {  // unordered_map keys (std::map lookups with these keys were ~5 % of the render thread)
    size_t operator()(const std::array<uint64_t, N>& k) const {
        uint64_t h = 0;
        for (uint64_t v : k) h = (h ^ v) * 0x9E3779B97F4A7C15ull;
        return size_t(h ^ h >> 29);
    }
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t w = 0, h = 0, layers = 1;  // Vulkan (physical) size: guest size * scale
    uint32_t scale = 1;                 // BB_RES_SCALE for screen-sized render / depth targets (res_scale), else 1
    uint32_t gw() const { return w / scale; }
    uint32_t gh() const { return h / scale; }
    uint32_t levels = 1;  // Vulkan mip levels (a mipped texture holds its full chain, see get_texture)
    uint32_t mips = 1;    // levels uploaded from guest memory (0 .. the highest LAST_LEVEL seen)
    VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D;
    bool initialised = false;  // has been transitioned to GENERAL
    bool bgra = false;         // memory component order is B,G,R,A (vs R,G,B,A)
    bool ds = false;           // depth/stencil image (DB_* targets)
    uint64_t sbase = 0;        // guest address of the stencil plane (the depth plane's address is `base`)
    bool snapshot = false;     // created by a copy of another target into plain memory (see image_cs_op)
    uint32_t version = 0;      // bumped by every such copy; `crop_of` = the source version a cropped texture copy was taken from
    uint32_t crop_of = ~0u;
    uint64_t hash = 0, qhash = 0;
    uint64_t hash_epoch = 0, full_epoch = 0;  // texture change detection, see get_texture / tex_touch
    uint64_t bytes = 0;                          // guest bytes a texture reads from `base`
    bool dynamic = false;                        // its bytes changed after the first upload (untracked: hashed in full every epoch)
    int8_t ww = 0;                               // texture change detection by the write watch: 1 tracked, -1 untracked (get_texture)
    uint64_t ww_epoch = 0;                       // (write-watch epoch of the last check)
    uint32_t touched = 0;                        // port writes into the texture (tex_touch): its tex_memo entries are stale
    uint64_t base = 0;
    bool cube_ok = false;                  // created CUBE_COMPATIBLE (layered render targets with a multiple of 6 slices)
    bool srgb_storage = false;             // sRGB format with storage use: MUTABLE + EXTENDED_USAGE, storage views in the UNORM twin
    bool mutable_fmt = false;              // texture of one memory for every number format: views take the T#'s format (get_texture)
    std::vector<VkImageView> layer_views;  // single-layer 2D views of a layered render target (one attachment per CB slice), lazily created
    // Replay carried state (the interpolation hybrid, see Backend::note_read): tick of the first access and its kind (1 read, 2 write); `carry` once a
    // tick read it before writing it (its contents come from earlier ticks); copies before the tick's first write / after the tick.
    uint64_t acc_tick = 0;
    uint8_t acc = 0;
    bool carry = false;          // read first in at least 2 of the ticks with gaps <= 16 (one-off reads, e.g. while loading, are not state)
    uint64_t rf_tick = 0;        // last tick it was read first
    uint32_t rf_count = 0;
    uint64_t twin_tick = 0;      // last tick its twins were copied (unused twins are freed, see replay_tick)
    Image* pre = nullptr;
    Image* post = nullptr;
};

// The UNORM twin of an sRGB format (storage images cannot be sRGB; the store writes the raw bits either way), else the format itself.
VkFormat unorm_twin(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
        default: return f;
    }
}
bool block_compressed(VkFormat f) { return f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; }

struct Buf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    uint8_t* mapped = nullptr;
    size_t size = 0;
};

union DescInfo { VkDescriptorBufferInfo b; VkDescriptorImageInfo i; };  // one per resource: the data of a descriptor update template

struct ShaderEntry {
    Translation tr;
    ShStage stage = ShStage::VS;
    VkShaderModule module = VK_NULL_HANDLE;
    bool has_uav = false;  // writes a storage image or buffer (draws with side effects break the rendering instance; set in translate_new)
    std::vector<uint32_t> fetch_code;
    std::string log_name;
    uint64_t code_hash = 0;
    int kind = 0;  // 1: Gnm library "fill typed buffer with a constant" CS, 2: "copy typed buffer" CS (handled on images, see image_cs_op)
    uint32_t fail_uses = 0;  // lookups that hit this failed translation (see get_shader)
    uint64_t fetch_seen = 0;  // last fetch-shader address other than tr.fetch_addr a draw used (logged once per address, see get_shader)
};

struct BoundShader {
    ShaderEntry* e = nullptr;
    std::vector<std::array<uint32_t, 8>> words;
};

struct Writeback { uint8_t* dst; size_t ring_offset, size; };
struct HostBuf { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; uint64_t start = 0, end = 0; };

struct Backend {
    // ---- device-level objects --------------------------------------------------------------------------------------
    VkCommandPool pool = VK_NULL_HANDLE;
    // Submission slots (command buffer, fence, descriptor pool, upload-ring part), used round robin: the GPU runs up to nslots - 1 of
    // them while the CPU records the next. BB_SLOTS=2..8, default 6: in the clinic (paired runs) 4 slots waited 3.7 ms per frame for
    // the slot to reuse and 1.8 ms for the backend lock, 8 slots 0 / 0.45 ms (23.7 -> 25.3 fps); 6 were as fast as 8. Each slot owns
    // kSlotBytes of the host-visible upload ring.
    static constexpr uint32_t kMaxSlots = 8;
    uint32_t nslots = 6;
    VkCommandBuffer cb = VK_NULL_HANDLE;  // the recording slot's
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkCommandBuffer cmd_bufs[kMaxSlots] = {};
    // Per slot, a command buffer submitted ahead of the main one: uploads into freshly created cache buffers (nothing recorded earlier
    // reads them), so they need neither inline barriers nor a break of the rendering instance. One barrier at its end orders them.
    VkCommandBuffer pre_bufs[kMaxSlots] = {};
    bool pre_open = false;
    VkCommandBuffer pre();
    struct VtxRange { uint32_t pool = 0; VmaVirtualAllocation va = VK_NULL_HANDLE; };  // a sub-range of a vertex-cache pool (see VtxEntry)
    std::vector<VtxRange> retired_s[kMaxSlots];  // cache ranges replaced or evicted while recording the slot: freed once its work is done (settle)
    // Long submissions can be cut into chunks that are submitted without waiting, so the GPU runs them while the rest is recorded.
    // BB_CHUNK=<draws+dispatches per chunk> (default off: +0-3 % frames, within noise, ROADMAP 50; async submits exposed races before).
    uint32_t ops_since_flush = 0;
    void chunk_tick() {
        static const uint32_t k = std::getenv("BB_CHUNK") ? uint32_t(std::atoi(std::getenv("BB_CHUNK"))) : 0;
        if (k && ++ops_since_flush >= k) flush(false);
    }
    // Interpolation hybrid (present.cpp, BB_FPS above 30): each guest flip ("tick") is shown n times; n-1 images re-submit the tick's command buffers with the camera
    // constants interpolated towards the previous tick (replay_tick). A slot then records several segment command buffers: ops that write
    // guest memory (written buffers, GDS, GPU memory copies) sit in "unsafe" segments the replay skips. The tick's slots (segments, ring
    // part, pool sets, retired vertex-cache ranges) stay held until its replay ran.
    bool replay_on = false, replay_lerp = true, replay_log = false;
    struct Seg { VkCommandBuffer cb; bool unsafe; bool patched = false; };  // patched: holds a camera/object patch site (statistics)
    std::vector<VkCommandBuffer> seg_pool[kMaxSlots];
    uint32_t seg_used[kMaxSlots] = {};
    std::vector<Seg> tick_segs, rp_segs;  // the open tick's segments in recording order (rp_segs: replay_tick scratch)
    std::vector<VkCommandBuffer> submit_cbs;
    uint32_t tick_slots = 0;              // slots holding the open tick's work
    bool tick_dropped = false;            // a resource of the open tick was freed or its first slot reused: no replay
    bool cur_unsafe = false, op_unsafe = false;
    VkFence hold[kMaxSlots] = {};         // the replay that still reads slot s
    void cut(bool unsafe);
    bool hold_done(uint32_t s);
    void drop_tick() { tick_segs.clear(); tick_slots = 0; tick_dropped = true; carry_pre.clear(); carry_post.clear(); }
    // Carried state (BB_INTERP_RESTORE, default on): images whose contents come from earlier ticks (TAA history, adaptation) must look to
    // every replay as before the tick, and to the next tick as after the original. Images read before written this tick get a copy just
    // before that write (pre, in the stream); every image written this tick that ever carried state gets a copy at the tick end (post).
    // A replay starts from pre; after the last one, post is copied back.
    bool replay_carry = false;
    uint64_t tick_no = 1;
    std::vector<Image*> carry_pre, carry_post, op_writes;  // op_writes: storage images the op being recorded writes (write_descriptors)
    uint64_t twin_bytes = 0;
    void note_read(Image* im) {
        if (!replay_carry || im->acc_tick == tick_no) return;
        im->acc_tick = tick_no, im->acc = 1;
        im->rf_count = tick_no - im->rf_tick <= 16 ? im->rf_count + 1 : 1;
        im->rf_tick = tick_no;
        im->carry = im->rf_count >= 2;
    }
    void note_write(Image* im);
    static uint64_t twin_size(const Image* im) { return uint64_t(im->w) * im->h * im->layers * (im->format == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : im->format == VK_FORMAT_D32_SFLOAT_S8_UINT ? 5 : 4); }
    Image* twin(Image*& tw, const Image* im, VkCommandBuffer cmd);
    void copy_image(VkCommandBuffer cmd, const Image* src, const Image* dst);
    static constexpr uint32_t kReplaySets = 3, kMaxReplayN = 8, kMaxBlocks = 256;
    static constexpr size_t kStagingBytes = size_t(32) << 20;  // per replay set: lerped constants of all replays of a tick
    struct ReplaySet { VkFence fence = VK_NULL_HANDLE; VkCommandBuffer glue[kMaxReplayN] = {}; VkQueryPool q = VK_NULL_HANDLE; Buf staging; bool used = false; };
    ReplaySet rsets[kReplaySets];
    uint32_t rset_next = 0;
    void wait_replay(ReplaySet& rs);
    bool replay_tick(uint64_t addr, uint32_t width, uint32_t height, const VkImage* dst, VkExtent2D ext, uint32_t n, const float* alpha, ReplayWarp* warp, uint64_t* ready, ReplayResult* res);
    VkSemaphore ready_sem = VK_NULL_HANDLE;  // (timeline; replay_tick's `ready`)
    uint64_t ready_val = 0;                  // its last signalled value
    // Reprojection support (BB_INTERP_WARP, present.cpp): the main camera = the display-sized camera block most draws use; its depth buffer
    // and viewport come from those draws. The display pass = the non-blended draw into the display target; its largest sampled image is its
    // source (disp_src; the HUD is blended into it, then the pass maps it to the display, e.g. through a colour LUT). The scene before the
    // HUD = that source just before its first blended draw of the tick (scene_img, copied in the stream: replays take it too).
    bool replay_warp = false, warp_diag = false;
    uint64_t disp_addr[4] = {};
    uint32_t disp_w = 0, disp_h = 0;
    uint64_t disp_src = 0;           // (learned from the last display pass)
    const Image* disp_src_img = nullptr;  // this tick's display pass source
    Image* scene_img = nullptr;
    bool scene_taken = false;
    struct DsUse { Image* ds; uint32_t n; };
    std::vector<DsUse> main_ds;
    float main_vport[6] = {};
    std::vector<std::string> diag_lines;  // (BB_INTERP_LOG, every 300 ticks: the draws without the main camera = post/HUD candidates)
    uint32_t diag_draw = 0;                // draws of the tick so far
    bool disp_block(const std::array<uint32_t, 216>& b) const {  // dwords 4-5: the render size (float or integer)
        float w, h;
        std::memcpy(&w, &b[4], 4), std::memcpy(&h, &b[5], 4);
        return disp_w && ((w == float(disp_w) && h == float(disp_h)) || (b[4] == disp_w && b[5] == disp_h));
    }
    void warp_draw(const RegView& r, Image* rt0, uint32_t rt0_slot, Image* dsimg, bool use_z, size_t sites0, const char* who);
    static constexpr uint32_t kCamAt = 8;  // the camera block's dwords 8-71 go to the presenter (view 3x4 at 8, projection at 36-67; ROADMAP 68d)
    std::vector<uint32_t> site_n;          // (replay_tick scratch: sites per block)
    // camera interpolation: every 864-byte constant block copied into the ring (the camera block, ROADMAP 68d) is a patch site
    struct Site { size_t off; uint32_t block; };
    std::vector<Site> sites;
    std::vector<std::array<uint32_t, 216>> blocks, prev_blocks;  // distinct contents of this / the previous tick
    std::unordered_map<uint64_t, uint32_t> block_ids;
    void note_site(const uint8_t* p, size_t off);
    std::vector<VkBufferCopy> rp_regions[kMaxReplayN];  // replay_tick scratch: patch copies of replay i
    // Object interpolation (BB_INTERP_OBJ, default on): per draw, the small constant blocks (<= 4 KiB, not the camera block) and, for
    // single-instance draws, the 16-byte-stride VS buffers not fetched by vertex id (bone palettes; routed through the ring instead of the
    // vertex cache so they can be patched). Draws pair up across ticks by (VS, PS, count, instances): identical bytes first, else the
    // nearest constants of that key (ROADMAP 68a).
    bool replay_obj = true, obj_capture = false;
    uint32_t cur_instances = 0;
    struct ObjSite { size_t off; uint32_t res, dw, at; bool pal; };  // ring offset, stage << 16 | resource, dwords, arena index, palette
    struct ObjDraw { uint64_t key, hash; uint32_t first, n; };
    std::vector<ObjSite> osites, prev_osites;
    std::vector<ObjDraw> odraws, prev_odraws;
    std::vector<uint32_t> oarena, prev_oarena;  // the sites' bytes at recording time
    void note_obj(const uint8_t* p, size_t off, size_t size, uint32_t res, bool pal);
    void end_obj_draw();
    std::vector<size_t> cam_off;                                       // replay_tick scratch: staging offset per (replay, camera block)
    std::unordered_map<uint64_t, std::vector<uint32_t>> obj_pmap;      // (previous tick's draws by key)
    // Per-tick counters by 64-bit key without allocations: open addressing, cleared by bumping `gen` (ponytail: 16 Ki slots, a tick has
    // < 2 k draws; past 12 k keys everything counts into one spare slot)
    struct KeyTab {
        struct E { uint64_t key = 0, gen = 0; uint32_t n = 0; };
        std::vector<E> e = std::vector<E>(16384);
        uint64_t gen = 1;
        uint32_t used = 0;
        E spare;
        void clear() { ++gen, used = 0; }
        uint32_t& at(uint64_t k) {
            for (size_t i = size_t((k * 0x9E3779B97F4A7C15ull) >> 50);; i = (i + 1) & 16383) {
                E& x = e[i];
                if (x.gen == gen && x.key == k) return x.n;
                if (x.gen != gen) { if (++used > 12288) return spare.n; x = {k, gen, 0}; return x.n; }
            }
        }
    };
    KeyTab obj_phash, obj_occ;  // (previous tick's key ^ bytes; this tick's draws per key so far)
    uint64_t tick_imp_reads = 0, hold_ns = 0, rp_cpu_ns = 0, rp_ticks = 0, rp_fallbacks = 0;
    std::atomic<double> replay_gpu_ms{0};  // last measured GPU time of one replay (read by the present thread)
    std::atomic<double> replay_hold_ms{0};  // the last tick's waits for replays (backend_replay_hold_ms)
    VkFence fences[kMaxSlots] = {};
    VkDescriptorPool dpools[kMaxSlots] = {};
    bool in_flight[kMaxSlots] = {};
    uint32_t slot = 0;
    bool imp_written = false;  // the recording writes guest memory in place through an imported buffer
    uint64_t sync_imp = 0;
    bool recording = false;
    Buf ring;  // upload/readback ring (host visible): [zero head][slot 0]..[slot nslots-1][GDS]
    static constexpr size_t kZeroBytes = size_t(64) << 20;  // ring head: never written, bound for null/unmapped read-only buffers
    static constexpr size_t kSlotBytes = size_t(224) << 20;
    static constexpr size_t kGdsBytes = size_t(64) << 10;   // ring tail: the global data share, alive for the whole run (Resource::GdsBuffer)
    size_t gds_off() const { return kZeroBytes + nslots * kSlotBytes; }
    uint32_t older(uint32_t k) const { return (slot + k) % nslots; }  // k = 1: the oldest submitted slot .. nslots - 1: the newest
    size_t slot_begin() const { return kZeroBytes + slot * kSlotBytes; }
    size_t slot_end() const { return slot_begin() + kSlotBytes; }
    size_t ring_used = kZeroBytes;
    uint64_t sync_n = 0, sync_wb = 0, sync_dirty = 0, sync_grave = 0, sync_host = 0, sync_wait = 0, vtx_ns = 0;  // flush(): how often and why it had to wait; time in the vertex cache
    uint64_t sync_overlap = 0, sync_cur = 0, ft_submit = 0, ft_settle = 0, ft_syncwait = 0;  // reads of GPU-written memory that had to wait (in flight / still recording); time (ns) in vkQueueSubmit / waiting for the slot being reused / waiting in synchronous flushes
    size_t dirty_own = SIZE_MAX;  // while writing descriptors: dirty_s[slot] entries from this index on belong to the op itself
    bool sync_next = false;       // the next flush waits for its work (a recorded draw reads ring data of the other slot)
    bool flush_pending = false;  // a coalesced end-of-submit flush is due at flush_due (hook_end_submit, fence watcher)
    std::chrono::steady_clock::time_point flush_due{};
    uint64_t tess_cross = 0, tess_ring_flush = 0;  // patch draws whose LS ring data ended up in the other slot (see draw); flush count at tess_begin's ring allocation
    std::vector<Image*> graveyard_pending;
    std::map<uint64_t, std::unique_ptr<Image>> rts;
    std::map<uint64_t, std::unique_ptr<Image>> dss;  // depth/stencil targets by DB_Z/STENCIL_WRITE_BASE
    std::map<std::array<uint64_t, 3>, std::unique_ptr<Image>> texs;  // by base (bit 63: crop copy), size, format (get_texture); base order: tex_touch
    std::unordered_map<std::array<uint64_t, 3>, Image*, ArrayHash<3>> tex_lut;  // the same, for lookups (images are never freed)
    std::unordered_map<std::array<uint64_t, 4>, std::vector<std::unique_ptr<ShaderEntry>>, ArrayHash<4>> shaders;  // by translation environment (get_shader)
    std::unordered_map<uint64_t, VkPipeline> gpipelines;          // graphics pipelines by state hash (see draw())
    // Descriptors: one set 0 per (VS, PS) pair or CS (bindings: GcnEnv::binding_base), pushed into the command buffer (VK_KHR_push_descriptor:
    // no per-draw allocate/update/bind) when its resource count fits maxPushDescriptors, else allocated from the slot's pool and bound.
    struct DescLayout {
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorUpdateTemplate tmpl = VK_NULL_HANDLE;  // resource i of the first shader, then of the second <- DescInfo i (null: no resources)
        bool push = false;
    };
    std::unordered_map<uint64_t, DescLayout> dlayouts;  // by (vs, ps) / (cs, compute); node-based: references stay valid
    std::map<std::string, VkPipeline> pipelines;                  // compute pipelines
    std::map<uint64_t, VkSampler> samplers;
    std::map<uint64_t, VkShaderModule> rect_gs;
    Image dummy_tex;
    std::set<std::string> logged;
    uint64_t draws = 0, dispatches = 0, skipped = 0, rendered = 0, dispatched = 0, ds_clears = 0, fast_clears = 0, buf_bytes = 0, tex_bytes = 0, flushes = 0, image_ops = 0, cpu_ops = 0, cpu_bytes = 0, wb_bytes = 0, buf_ns = 0, buf_copies = 0, tex_ns = 0, tex_calls = 0, tex_uploads = 0, wd_ns = 0, wd_buf_ns = 0, host_ns = 0, host_imports = 0, host_import_bytes = 0, zero_binds = 0, tex_unsup = 0;
    uint64_t cb_bytes = 0, vb_bytes = 0, ob_bytes = 0;  // buffer bytes copied per draw: constant buffers (s_buffer_load), vertex streams, everything else
    uint64_t ds_alloc_ns = 0, ds_update_ns = 0, ds_pool_sets = 0;  // pool path vkAllocateDescriptorSets / template update or push (draws + dispatches); sets allocated (BB_GPU_LOG statistics)
    uint64_t view_ns = 0, smp_ns = 0, idx_up_ns = 0;    // texture_view, get_sampler, index upload into the ring (BB_GPU_LOG statistics)
    uint64_t crop_copies = 0, crop_bytes = 0;            // padded render targets copied to their sampled size (get_texture)
    struct Label { uint64_t at, value; uint32_t bytes; };
    std::vector<Label> labels_s[kMaxSlots];  // completion labels of the work recorded in each slot: written by settle() once it ran
    std::map<std::string, uint64_t> skip_by;
    std::map<std::string, uint64_t> tex_mix;  // texture images created, by tile index / formats (diagnosis)
    std::map<uint64_t, uint64_t> cmask_rt;  // CMASK address -> colour target base, learned from draws (the game fast-clears a target by zeroing its CMASK)
    std::set<uint64_t> fast_clear_pending;  // targets whose CMASK was zeroed: the next draw into them first clears the image to the clear colour
    std::map<uint64_t, uint64_t> htile_ds;  // HTILE address -> depth buffer base, learned from draws (DB_HTILE_DATA_BASE)
    std::set<uint64_t> htile_clear_pending;  // depth buffers whose HTILE was filled with ZMASK 0 (= cleared): the next draw clears them
    std::mutex mtx;
    std::thread watcher;  // fence watcher (backend_init)
    std::atomic<bool> stop_watch{false};
    bool verbose = std::getenv("BB_GPU_LOG") != nullptr;

    void log_once(const std::string& key, const std::string& msg) {
        if (logged.insert(key).second) std::fprintf(stderr, "gpu: %s\n", msg.c_str());
    }

    bool init();
    void shutdown();
    void begin();
    void flush(bool wait = true);
    uint8_t* ring_alloc(size_t n, size_t align, size_t& offset);
    Image* make_image(uint32_t w, uint32_t h, uint32_t layers, VkFormat fmt, VkImageViewType vt, VkImageUsageFlags usage, bool cube = false, uint32_t levels = 1, bool mutable_fmt = false);
    void destroy_image(Image* im);
    void ensure_init(Image* im);
    void barrier(VkPipelineStageFlags src = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VkAccessFlags src_access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    // BB_BARRIER_OPT=0: every barrier() is recorded (A/B). fenced: nothing recorded since the last barrier; other_since: a command other
    // than draws into the open pass's attachments (copies, clears, dispatches, a new command buffer) since the last barrier.
    const bool bar_opt = !(std::getenv("BB_BARRIER_OPT") && std::getenv("BB_BARRIER_OPT")[0] == '0');
    bool fenced = false, other_since = true, crop_last = false;  // crop_last: the last command recorded is a crop copy (no barrier yet)
    uint64_t barriers_skipped = 0, pass_barriers_skipped = 0, ls_barriers_skipped = 0;
    VkCommandBuffer rcb() { fenced = false, other_since = true, crop_last = false; return cb; }  // records an ordered command (see barrier)

    // Shader lookup (translation + live descriptor words) into `out`: callers pass one of the bs_* scratch results, whose word vectors keep
    // their capacity (a fresh vector per lookup was a heap allocation and free per draw stage). A memo of whole lookups by (key, user
    // SGPRs) within a submit was tried: 7 % hits in the clinic, and its upkeep made lookups ~30 % slower.
    void get_shader(ShStage st, const RegView& r, BoundShader& out);
    BoundShader bs_vs, bs_ps, bs_cs, bs_ls;
    const std::function<bool(uint64_t, uint32_t*, uint32_t)> rd_guest = read_guest;  // (built once, not per eval_resources call)
    ShaderEntry* translate_new(ShStage st, const RegView& r, uint64_t code_addr);
    VkSampler get_sampler(const uint32_t* w);
    Image* get_texture(const uint32_t* w, bool storage, uint32_t pad_axes = 0);  // pad_axes: see the padded-target case
    uint64_t pad_direct_binds = 0;
    Image* get_rt(uint64_t base, uint32_t w, uint32_t h, VkFormat fmt, uint32_t layers = 1, uint32_t min_scale = 0);
    VkImageView attach_view(Image* im, uint32_t layer);
    Image* get_ds(uint64_t base, uint32_t w, uint32_t h, VkFormat fmt, uint32_t min_scale = 0);
    // Scale of a new target of guest size w x h at `base` (res_scale): screen-sized (scale_class 'S') single-layer targets and targets once
    // promoted (bound together with a scaled attachment, see draw) are scaled, everything else (shadow maps, cubes, glare pyramid) not.
    std::set<uint64_t> promoted;
    uint32_t target_scale(uint64_t base, uint32_t w, uint32_t h, uint32_t layers) const {
        const uint32_t s = res_scale();
        return s > 1 && layers == 1 && (scale_class(w, h) == 'S' || promoted.count(base)) ? s : 1;
    }
    VkImageView texture_view(Image* im, const uint32_t* w, bool swizzle);
    const DescLayout& desc_layout(const ShaderEntry* a, const ShaderEntry* b, bool compute);
    bool write_descriptors(const BoundShader& bs, DescInfo* info);  // resource i -> info[i]
    bool prepare_descriptors(const DescLayout& dl, const BoundShader& a, const BoundShader* b, VkDescriptorSet& set);  // may flush
    void bind_descriptors(VkPipelineBindPoint bp, const DescLayout& dl, VkDescriptorSet set);
    std::vector<DescInfo> wd_info;  // prepare_descriptors scratch (capacity reused): the descriptor update template's data
    bool get_host_buf(uint64_t base, uint64_t size, VkBuffer& out, VkDeviceSize& offset, bool any_size = false);  // any_size: also < 64 KiB
    void drop_host_bufs();
    // Read-only buffers are cached in device-local buffers: the engine redraws the same meshes in several passes and frames, and copying
    // them into the upload ring each time (hundreds of GB per minute) dominated the frame. An entry is validated on every use: by the
    // write watch (untouched pages) or, for written pages / without tracking, by a full hash of the guest bytes (see get_vertex_buffer).
    // Entries are sub-ranges of a few large buffers (VMA virtual blocks): a buffer per entry cost a create and a destroy per upload
    // (~230 uploads per frame in the clinic), and the budget eviction destroyed thousands at once (~80 ms under the backend lock).
    struct VtxPool { VkBuffer buf = VK_NULL_HANDLE; VmaAllocation alloc = nullptr; VmaVirtualBlock block = VK_NULL_HANDLE; };
    static constexpr VkDeviceSize kVtxPoolBytes = VkDeviceSize(256) << 20;
    std::vector<VtxPool> vtx_pools;
    bool vtx_alloc(VkDeviceSize size, VtxRange& r, VkDeviceSize& off);
    void vtx_free(const VtxRange& r) { vmaVirtualFree(vtx_pools[r.pool].block, r.va); }
    struct VtxEntry {
        VkBuffer buf = VK_NULL_HANDLE; VkDeviceSize off = 0; VtxRange range; uint64_t base = 0, size = 0, hash = 0, last_frame = 0, ww_epoch = 0;
        uint32_t idx_mn = 0, idx_mx = 0, idx_bytes = 0;  // index entries: range of the cached indices (computed once per upload), index width
    };
    struct VtxKeyHash { size_t operator()(uint64_t k) const { return size_t(k ^ k >> 32); } };  // keys are mixed already (their high bits best)
    std::unordered_map<uint64_t, VtxEntry, VtxKeyHash> vtx_cache;
    // vtx_cache[key] through a direct-mapped memo of element addresses (node-based map: stable until erased; erasing clears the memo
    // entry or, on eviction, all of it). The map lookups were ~2 % of the render thread. Indexed by the key's top bits (its low bits
    // repeat for aligned bases and sizes); 16 Ki slots for the ~4 k uses per frame.
    struct VtxMemo { uint64_t key = 0; VtxEntry* e = nullptr; };
    std::vector<VtxMemo> vtx_memo = std::vector<VtxMemo>(16384);
    static size_t vtx_slot(uint64_t key) { return size_t(key >> 50); }
    VtxEntry& vtx_entry(uint64_t key) {
        VtxMemo& m = vtx_memo[vtx_slot(key)];
        if (m.e && m.key == key) return *m.e;
        VtxEntry& e = vtx_cache[key];
        m = {key, &e};
        return e;
    }
    uint64_t vtx_cache_bytes = 0, vtx_hits = 0, vtx_uploads = 0, idx_hits = 0, idx_uploads = 0;
    uint64_t vtx_up_ns = 0, vtx_up_bytes = 0;  // time in / bytes of (re)uploads (part of vtx_ns; statistics)
    uint64_t vtx_evicted = 0, vtx_swept = 0, vtx_alloc_fails = 0;  // entries dropped over the budget / unused (sweep); pool allocations that failed
    // T# -> (image, view), valid within one texture epoch (guest submit): 86 % of the image bindings in the clinic repeat a T# already
    // resolved in the same epoch. Plain guest-memory textures, render/depth targets sampled in place and padded-target crops (`src`: the
    // target, whose version must still be the one the crop was taken from). Invalidated by tex_gen (render/depth target created or
    // replaced) or the image's `touched` count (a port write into the texture) and re-checked when GPU work records a new dirty range
    // (settle_pending of the slow path). A write-watch tracked texture carries over into a new epoch when its pages were not written.
    struct TexMemo { uint32_t w[8]; uint32_t kind = ~0u, ver = 0, touched = 0; Image* im = nullptr; Image* src = nullptr; VkImageView view = VK_NULL_HANDLE; uint64_t epoch = 0, gen = 0, dgen = 0, base = 0, bytes = 0; };
    std::vector<TexMemo> tex_memo = std::vector<TexMemo>(4096);
    uint64_t tex_gen = 1, dirty_gen = 0, tex_memo_hits = 0;
    bool tex_memo_ok = false;        // get_texture: the result may be memoized (tex_memo) ...
    Image* tex_memo_src = nullptr;   // ... while this target's version is unchanged (crops)
    uint64_t tex_verify_tick = 0, tex_verified = 0, tex_stale_epoch = 0, tex_stale_sample = 0, tex_stale_ww = 0;  // BB_WW_VERIFY for textures (get_texture)
    uint64_t tex_dynamic = 0, tex_ww_checks = 0, tex_ww_renewals = 0;  // images changed after the first upload; write-watch texture checks (of them: memo renewals, write_descriptors)
    uint64_t tex_mipped = 0, tex_texture_images = 0;  // texture images created with a mip chain / in total (get_texture, BB_GPU_LOG)
    // The same cache holds index buffers (ix != nullptr: `count` indices of `bytes` each; returns their min/max): a hit skips both the
    // index scan and the copy into the ring (the clinic draws ~5 M indices per frame, nearly all from static meshes).
    struct IndexInfo { uint32_t bytes = 2, count = 0, mn = ~0u, mx = 0; };
    bool get_vertex_buffer(uint64_t base, uint64_t size, VkBuffer& out, VkDeviceSize& out_off, IndexInfo* ix = nullptr);
    void evict_vertex_cache();
    uint64_t vtx_evict_frame = 0;
    bool vtx_full = false;  // a pool allocation failed: the next flush evicts at once (no 30-frame pause), also below the budget
    void sweep_vertex_cache();
    size_t vtx_sweep = 0;            // next hash bucket of the incremental sweep
    std::vector<uint64_t> vtx_dead;  // (sweep scratch)
    std::map<uint64_t, HostBuf> hostbufs;  // guest memory regions imported as buffers, by start address
    std::vector<HostBuf> retired_hostbufs;
    // BB_GPU_PROF: GPU time per pipeline from timestamp queries (accumulated, printed with the BB_GPU_LOG statistics)
    static constexpr uint32_t kProfQueries = 2048;
    bool prof = std::getenv("BB_GPU_PROF") != nullptr;
    VkQueryPool qpool = VK_NULL_HANDLE, tpool = VK_NULL_HANDLE;  // tpool: per-slot command buffer timing (BB_GPU_LOG)
    uint64_t gpu_cb_us = 0, gpu_cbs = 0, barriers = 0;
    uint32_t qn = 0;
    bool q_open = false;
    std::vector<std::string> qname;
    std::map<std::string, std::pair<double, uint64_t>> gpu_ms;  // name -> (ms, count)
    void prof_begin(const std::string& name) {
        if (!prof || qn + 2 > kProfQueries) return;
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, qpool, qn);
        qname.push_back(name);
        q_open = true;
    }
    void prof_end() {
        if (!q_open) return;
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, qpool, qn + 1);
        qn += 2;
        q_open = false;
    }
    // BB_GPU_TS=<frames>: timestamps (bottom of pipe, no extra waits) around every rendering instance, dispatch, barrier and copy; the time
    // between two marks belongs to the label current before the second. Read when the slot settles, printed per frame every <frames>.
    // Queries 0/1 of a slot: its pre buffer (uploads). Off: one predictable branch per mark.
    static constexpr uint32_t kTsQueries = 8192;
    const uint32_t ts_every = std::getenv("BB_GPU_TS") ? std::max(1, std::atoi(std::getenv("BB_GPU_TS"))) : 0;
    struct TsLab { std::string name; uint32_t draws; };
    VkQueryPool ts_pool[kMaxSlots] = {};
    std::vector<TsLab> ts_lab[kMaxSlots];
    bool ts_pre[kMaxSlots] = {};
    std::string ts_cur;
    uint32_t ts_draws = 0;
    uint64_t ts_prev_end = 0, ts_frame0 = 0;
    std::chrono::steady_clock::time_point ts_wall0;
    std::map<std::string, std::pair<double, uint64_t>> ts_ms, ts_dr;  // label -> (ms, intervals), (draws, -)
    void ts_mark(std::string next) {
        if (!ts_every || ts_lab[slot].size() + 3 > kTsQueries) return;
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool[slot], 2 + uint32_t(ts_lab[slot].size()));
        ts_lab[slot].push_back({std::move(ts_cur), ts_draws});
        ts_cur = std::move(next);
        ts_draws = 0;
    }
    void ts_label(const char* s) { if (ts_every) ts_mark(s); }  // closes the work before it: ops that need no barrier first
    void ts_read(uint32_t s);
    // Open dynamic-rendering instance (draw() batches consecutive draws into the same attachments, see there)
    struct OpenPass { bool open = false; uint32_t n = 0; VkImageView colors[8] = {}; VkImageView ds = VK_NULL_HANDLE; bool use_z = false, use_s = false; uint32_t w = 0, h = 0; } pass;
    std::vector<const Image*> pass_written;  // images rendered into since the last barrier
    std::vector<const Image*> pass_read;     // attachments and sampled images of the draws since the last barrier
    std::vector<const Image*> sampled;       // images the draw being recorded binds (filled by write_descriptors)
    void end_pass() { if (pass.open) { vkCmdEndRendering(cb); pass.open = false; } }
    bool px_on = std::getenv("BB_PIXEL") != nullptr;  // debug: see pixel_probe
    void pixel_probe(const std::string& who);
    // Debug BB_SCALE_LOG=<s>: one frame from second s (counted from the first flip), groundwork for resolution scaling: a line per draw /
    // dispatch with its targets and images (base:guest size:format, class S = screen-sized, i.e. 16:9 within 5 % and >= 960 wide, else F;
    // images: T# size / image size, use L = loaded only, S = sampled, W = storage write; rt / ds / tex = render target, depth, other), then a
    // summary (CS / PS-UAV writes of screen-sized images, attachments of mixed class, FragCoord + load of screen-sized images).
    // BB_RES_SCALE > 1: per stage (ShStage) bit i = resource i of the op being recorded binds a scaled image; bit 31 (draws) = its targets
    // are scaled. Pushed after the user data (gcn_spirv GcnEnv::res_scale), so the shader maps guest texel coordinates to physical ones.
    uint32_t scale_bits[3] = {};
    // Push constants per stage: the 16 user-data dwords, with BB_RES_SCALE > 1 then scale_bits; VS/CS at 0, PS at push_fs().
    static uint32_t push_bytes() { return res_scale() > 1 ? 68 : 64; }
    static uint32_t push_fs() { return res_scale() > 1 ? 80 : 64; }
    void push_user(VkPipelineLayout layout, VkShaderStageFlags st, uint32_t offset, const uint32_t* user, ShStage s) {
        if (push_bytes() == 64) { vkCmdPushConstants(cb, layout, st, offset, 64, user); return; }
        uint32_t v[17];
        std::memcpy(v, user, 64);
        v[16] = scale_bits[uint32_t(s)];
        vkCmdPushConstants(cb, layout, st, offset, 68, v);
    }
    bool census = false;
    std::string census_res;                      // the op being recorded: its images (write_descriptors)
    std::map<std::string, uint64_t> census_sum;  // summary line -> ops
    static char scale_class(uint32_t w, uint32_t h) { return w >= 960 && h && std::fabs(double(w) / h * 9 / 16 - 1) <= 0.05 ? 'S' : 'F'; }
    std::string census_name(const Image* im) const;
    void census_img(const BoundShader& bs, const Resource& r, const uint32_t* w, const Image* im);
    void census_frame();
    // Tessellation (patch draws): the LS runs as a compute pass over every control point into `tess_lds` (the emulated LDS), the DS then draws a
    // generated (N+1)^2 grid per patch (instance = patch) and reads it back. get_shader/translate_new pick the LS or DS variant of the VS slot
    // through cur_tess_ls / cur_tess_ds (0 = plain VS); the extra descriptor resources bind tess_lds / tess_idx.
    uint32_t cur_tess_ls = 0, cur_tess_ds = 0;
    struct TessBind { VkBuffer buf = VK_NULL_HANDLE; VkDeviceSize off = 0, size = 0; } tess_lds, tess_idx;
    VkBuffer tess_dev[kMaxSlots] = {};  // per slot: the device-local LDS of large patch draws (tess_begin), 16 MB, created on first use
    VmaAllocation tess_dev_alloc[kMaxSlots] = {};
    VkBuffer tess_dev_lds();
    std::map<uint32_t, std::vector<uint16_t>> tess_grid;  // synthetic DS index buffers by level
    bool tess_begin(const RegView& r, const DrawCmd& in, DrawCmd& out, uint32_t& level, uint32_t& patches);
    void record_compute(const BoundShader& cs, const uint32_t* user, uint32_t x, uint32_t y, uint32_t z, bool ls = false);
    void const_dump(uint32_t kind, uint64_t sh0, uint64_t sh1, const BoundShader* a, const BoundShader* b, const uint32_t* ua, const uint32_t* ub,
                    uint64_t rt0, uint64_t zb, uint32_t mask, uint32_t count, uint32_t inst);
    const bool cdump_on = std::getenv("BB_CONST_DUMP") != nullptr;
    uint64_t cur_vtx_records = ~0ull;      // vertex records the draw being recorded can fetch by vertex id (window of the vertex_use 1 streams)
    uint64_t cur_vtx_first = 0;            // first vertex record of that window (see draw(): the vertex buffers are bound from there)
    std::vector<std::pair<uint64_t, uint64_t>> dirty_s[kMaxSlots];  // guest ranges written by GPU work recorded in slot s and not yet retired (dirty_add)
    std::vector<Writeback> wb_s[kMaxSlots];                          // results to copy back to guest memory when slot s completes (ranges also in dirty_s)
    // Per MiB of guest address space (aliased modulo 1 TiB): how many dirty_s ranges touch it. Most reads touch no dirty MiB and skip the
    // range scans, which grow with the work in flight (8 slots instead of 4 cost ~3 ms per frame in scans before this).
    std::vector<uint32_t> dirty_mib = std::vector<uint32_t>(size_t(1) << 20);
    void dirty_count(uint64_t lo, uint64_t hi, int d) {
        if (hi > lo) for (uint64_t m = lo >> 20; m <= (hi - 1) >> 20; ++m) dirty_mib[m & 0xFFFFF] += uint32_t(d);  // d = -1 wraps
    }
    void dirty_add(uint64_t lo, uint64_t hi) { dirty_s[slot].push_back({lo, hi}); dirty_count(lo, hi, 1); ++dirty_gen; }
    bool maybe_dirty(uint64_t lo, uint64_t hi) const {
        if (hi > lo) for (uint64_t m = lo >> 20; m <= (hi - 1) >> 20; ++m) if (dirty_mib[m & 0xFFFFF]) return true;
        return false;
    }
    // true if GPU work (recorded in the current slot or still in flight) writes guest memory in [lo, hi)
    bool overlaps_dirty(uint64_t lo, uint64_t hi) const {
        if (!maybe_dirty(lo, hi)) return false;
        for (const auto& s : dirty_s) for (const auto& d : s) if (d.first < hi && lo < d.second) return true;
        return false;
    }
    void settle(uint32_t s);
    void settle_pending(uint64_t lo, uint64_t hi, int why = 10);  // why: 0 texture, 1 buffer, 2 indices, 3 tess indices, 4-9 PM4 ops (gpu_hooks.h sync_read), 10 other (statistics)
    void tex_touch(uint64_t lo, uint64_t hi);  // the port wrote guest memory [lo, hi): overlapping textures are checked again at their next use
    uint64_t tex_max_bytes = 0;                 // largest Image::bytes (bounds the texs scan of tex_touch)
    uint64_t sync_cur_by[11] = {}, sync_cur_wb = 0;  // current-slot read flushes by caller; how many lay inside one write-back range
    uint64_t gpu_dmas = 0, import_barriers = 0, gpu_indirect = 0;  // DMA_DATA copies recorded on the GPU (hook_mem_copy); imports ordered by a barrier, not a wait; GPU-side indirect dispatches
    struct { VkBuffer buf = VK_NULL_HANDLE; VkDeviceSize off = 0; } cur_indirect;  // set while recording a GPU-side DISPATCH_INDIRECT
    // Graphics state last recorded into `cb` (begin() resets it): unchanged binds/dynamic state are not recorded again. Every graphics
    // pipeline declares the same dynamic states, so a pipeline bind keeps them.
    struct RecState {
        VkPipeline pipe = VK_NULL_HANDLE;
        bool dyn = false;  // vp, sc, bc set
        VkViewport vp{};
        VkRect2D sc{};
        float bc[4] = {};
        uint32_t stencil[6] = {~0u, ~0u, ~0u, ~0u, ~0u, ~0u};  // reference, compare mask, write mask per face
        VkBuffer ib = VK_NULL_HANDLE;
        VkDeviceSize ib_off = 0;
        VkIndexType ib_type = VK_INDEX_TYPE_MAX_ENUM;
    } rec;
    uint64_t vtx_hash_bytes = 0, tex_hash_bytes = 0, tex_checks = 0, tex_full_hashes = 0, ww_checked = 0, ww_mismatch = 0, ww_mismatch_late = 0;  // bytes the vertex cache / texture checks hashed; BB_WW_VERIFY: clean verdicts checked / wrong (never seen by the watch / written after this submit's poll)
    VkShaderModule get_rect_gs(uint32_t param_mask);
    void draw(const RegView& r, const DrawCmd& d);
    void dispatch(const RegView& r, uint32_t x, uint32_t y, uint32_t z);
    bool image_cs_op(const BoundShader& cs, uint32_t x, uint32_t local_x);

    std::unordered_map<std::array<uint64_t, 3>, VkImageView, ArrayHash<3>> views;  // by image, swizzle, layer/level range, view type, view format (texture_view)
    std::unordered_map<uint64_t, std::array<float, 4>> mem_fill;  // colour of the last CPU fill helper at an address (a target created there starts with it)
};

Backend* g_be = nullptr;

// ---- format tables ------------------------------------------------------------------------------------------------
VkFormat rt_format(uint32_t info, bool& bgra) {
    const uint32_t fmt = (info >> 2) & 31, num = (info >> 8) & 7, swap = (info >> 11) & 3;
    bgra = swap == 1;
    auto pick = [&](VkFormat unorm, VkFormat snorm, VkFormat uint_, VkFormat sint, VkFormat flt, VkFormat srgb) {
        switch (num) {
            case 0: return unorm;
            case 1: return snorm;
            case 4: return uint_;
            case 5: return sint;
            case 6: return srgb;
            case 7: return flt;
            default: return unorm;
        }
    };
    const VkFormat U = VK_FORMAT_UNDEFINED;
    switch (fmt) {
        case 1: return pick(VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT, U, VK_FORMAT_R8_SRGB);
        case 2: return pick(VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT, VK_FORMAT_R16_SFLOAT, U);
        case 3: return pick(VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, U, VK_FORMAT_R8G8_SRGB);
        case 4: return pick(U, U, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, VK_FORMAT_R32_SFLOAT, U);
        case 5: return pick(VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R16G16_SFLOAT, U);
        case 9: return pick(VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_SNORM_PACK32, VK_FORMAT_A2B10G10R10_UINT_PACK32, U, U, U);
        case 10:
            if (bgra) return pick(VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SNORM, VK_FORMAT_B8G8R8A8_UINT, VK_FORMAT_B8G8R8A8_SINT, U, VK_FORMAT_B8G8R8A8_SRGB);
            return pick(VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT, U, VK_FORMAT_R8G8B8A8_SRGB);
        case 11: return pick(U, U, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32_SFLOAT, U);
        case 12: return pick(VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R16G16B16A16_SFLOAT, U);
        case 14: return pick(U, U, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT, VK_FORMAT_R32G32B32A32_SFLOAT, U);
        case 6: case 7: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        default: return U;
    }
}

struct TexFmt { VkFormat vk = VK_FORMAT_UNDEFINED; uint32_t bytes = 0; uint32_t block = 1; bool bgra = false; };
TexFmt tex_format(uint32_t dfmt, uint32_t nfmt) {
    auto pick = [&](VkFormat unorm, VkFormat snorm, VkFormat uint_, VkFormat sint, VkFormat flt, VkFormat srgb, uint32_t bytes) {
        VkFormat f = unorm;
        switch (nfmt) { case 1: case 6: f = snorm; break; case 4: f = uint_; break; case 5: f = sint; break; case 7: f = flt; break; case 9: f = srgb; break; default: break; }
        if (f == VK_FORMAT_UNDEFINED) f = unorm;
        return TexFmt{f, bytes, 1, false};
    };
    const VkFormat U = VK_FORMAT_UNDEFINED;
    switch (dfmt) {
        case 1: return pick(VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT, U, VK_FORMAT_R8_SRGB, 1);
        case 2: return pick(VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT, VK_FORMAT_R16_SFLOAT, U, 2);
        case 6: case 7: return {VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4, 1, false};  // 10_11_11 / 11_11_10 float (same mapping as render targets)
        case 3: return pick(VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, U, VK_FORMAT_R8G8_SRGB, 2);
        case 4: return pick(U, U, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, VK_FORMAT_R32_SFLOAT, U, 4);
        case 5: return pick(VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R16G16_SFLOAT, U, 4);
        case 10: return pick(VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT, U, VK_FORMAT_R8G8B8A8_SRGB, 4);
        case 11: return pick(U, U, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32_SFLOAT, U, 8);
        case 12: return pick(VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R16G16B16A16_SFLOAT, U, 8);
        case 14: return pick(U, U, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT, VK_FORMAT_R32G32B32A32_SFLOAT, U, 16);
        case 35: return {nfmt == 9 ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 8, 4, false};
        case 36: return {nfmt == 9 ? VK_FORMAT_BC2_SRGB_BLOCK : VK_FORMAT_BC2_UNORM_BLOCK, 16, 4, false};
        case 37: return {nfmt == 9 ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK, 16, 4, false};
        case 38: return {nfmt == 1 ? VK_FORMAT_BC4_SNORM_BLOCK : VK_FORMAT_BC4_UNORM_BLOCK, 8, 4, false};
        case 39: return {nfmt == 1 ? VK_FORMAT_BC5_SNORM_BLOCK : VK_FORMAT_BC5_UNORM_BLOCK, 16, 4, false};
        case 40: return {nfmt == 1 ? VK_FORMAT_BC6H_SFLOAT_BLOCK : VK_FORMAT_BC6H_UFLOAT_BLOCK, 16, 4, false};
        case 41: return {nfmt == 9 ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK, 16, 4, false};
        default: return {};
    }
}

// ---- image / buffer helpers -----------------------------------------------------------------------------------------
bool Backend::init() {
    VkCtx& c = vk();
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = c.family;
    if (!vk_check(vkCreateCommandPool(c.device, &pci, nullptr, &pool), "backend command pool")) return false;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    replay_on = g_replay_req && !prof && !ts_every;  // (BB_GPU_PROF/BB_GPU_TS timestamps live in the command buffers a replay would run again)
    if (replay_on) {
        nslots = kMaxSlots;  // a tick spans ~2-5 flushes; its slots stay held until the replay ran
        replay_lerp = !(std::getenv("BB_INTERP_LERP") && std::getenv("BB_INTERP_LERP")[0] == '0');
        replay_log = std::getenv("BB_INTERP_LOG") != nullptr;
        replay_obj = replay_lerp && !(std::getenv("BB_INTERP_OBJ") && std::getenv("BB_INTERP_OBJ")[0] == '0');
        replay_carry = !(std::getenv("BB_INTERP_RESTORE") && std::getenv("BB_INTERP_RESTORE")[0] == '0');
        replay_warp = !(std::getenv("BB_INTERP_WARP") && std::getenv("BB_INTERP_WARP")[0] == '0');
        warp_diag = replay_log;
    }
    if (const char* e = std::getenv("BB_SLOTS")) nslots = std::clamp<uint32_t>(uint32_t(std::atoi(e)), 2, kMaxSlots);
    ai.commandBufferCount = nslots;
    if (!vk_check(vkAllocateCommandBuffers(c.device, &ai, cmd_bufs), "backend command buffers")) return false;
    if (!vk_check(vkAllocateCommandBuffers(c.device, &ai, pre_bufs), "backend command buffers")) return false;
    cb = cmd_bufs[0];
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (VkFence& f : fences)
        if (!vk_check(vkCreateFence(c.device, &fci, nullptr, &f), "backend fence")) return false;

    const VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16384}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8192},
                                          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLER, 8192}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 4096;
    dpi.poolSizeCount = 4;
    dpi.pPoolSizes = sizes;
    for (VkDescriptorPool& dp : dpools)
        if (!vk_check(vkCreateDescriptorPool(c.device, &dpi, nullptr, &dp), "descriptor pool")) return false;
    dpool = dpools[0];

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    VmaAllocationCreateInfo aci{};
    aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    VmaAllocationInfo info;
    for (;;) {  // kSlotBytes per slot in one host-visible buffer: devices with less such memory get fewer slots
        ring.size = gds_off() + kGdsBytes;
        bci.size = ring.size;
        if (vmaCreateBuffer(c.vma, &bci, &aci, &ring.buffer, &ring.alloc, &info) == VK_SUCCESS) break;
        if (nslots == 2) { std::fprintf(stderr, "gpu: the upload ring (%llu MB) could not be allocated\n", (unsigned long long)(ring.size >> 20)); return false; }
        --nslots;
        std::fprintf(stderr, "gpu: upload ring of %llu MB not available, retrying with %u slots\n", (unsigned long long)(ring.size >> 20), nslots);
    }
    ring.mapped = static_cast<uint8_t*>(info.pMappedData);
    std::memset(ring.mapped, 0, kZeroBytes);
    std::memset(ring.mapped + gds_off(), 0, kGdsBytes);

    dummy_tex = *make_image(1, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    if (prof) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kProfQueries;
        if (!vk_check(vkCreateQueryPool(c.device, &qi, nullptr, &qpool), "query pool")) prof = false;
    }
    if (verbose && !replay_on) {  // statistics: GPU time from the start to the end of each submitted command buffer (2 timestamps per slot)
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = 2 * kMaxSlots;
        if (!vk_check(vkCreateQueryPool(c.device, &qi, nullptr, &tpool), "query pool")) tpool = VK_NULL_HANDLE;
    }
    for (uint32_t s = 0; s < kMaxSlots && ts_every; ++s) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kTsQueries;
        if (!vk_check(vkCreateQueryPool(c.device, &qi, nullptr, &ts_pool[s]), "timestamp pool")) return false;
    }
    for (ReplaySet& rs : rsets) {
        if (!replay_on) break;
        VkCommandBufferAllocateInfo gi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, kMaxReplayN};
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = 2;
        VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sci.size = rs.staging.size = kStagingBytes;
        sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo sa{};
        sa.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        sa.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        VmaAllocationInfo si;
        if (!vk_check(vkCreateFence(c.device, &fci, nullptr, &rs.fence), "replay fence") || !vk_check(vkAllocateCommandBuffers(c.device, &gi, rs.glue), "replay command buffers") ||
            !vk_check(vkCreateQueryPool(c.device, &qi, nullptr, &rs.q), "replay query pool") || !vk_check(vmaCreateBuffer(c.vma, &sci, &sa, &rs.staging.buffer, &rs.staging.alloc, &si), "replay staging"))
            return false;
        rs.staging.mapped = static_cast<uint8_t*>(si.pMappedData);
    }
    begin();
    return true;
}

void Backend::begin() {
    if (recording) return;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = replay_on ? VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT : VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;  // (a replay re-submits it)
    if (replay_on) {
        std::vector<VkCommandBuffer>& p = seg_pool[slot];
        if (seg_used[slot] == p.size()) {
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
            p.emplace_back();
            vkAllocateCommandBuffers(vk().device, &ai, &p.back());
        }
        cb = p[seg_used[slot]++];
        tick_segs.push_back({cb, cur_unsafe});
        tick_slots |= 1u << slot;
    }
    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &bi);
    if (prof) { vkCmdResetQueryPool(cb, qpool, 0, kProfQueries); qn = 0; qname.clear(); }
    if (tpool) { vkCmdResetQueryPool(cb, tpool, slot * 2, 2); vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, tpool, slot * 2); }
    if (ts_every) {
        vkCmdResetQueryPool(cb, ts_pool[slot], 2, kTsQueries - 2);
        ts_lab[slot].clear();
        ts_cur.clear();
        ts_pre[slot] = false;
        ts_mark("other");  // query 2: start of the command buffer
    }
    recording = true;
    fenced = false, other_since = true, crop_last = false;  // a new command buffer / replay segment starts with a real barrier
    rec = {};
}

// Waits for slot `s` and makes its results visible: write-backs land in guest memory, the slot's dirty ranges are retired.
void Backend::settle(uint32_t s) {
    if (in_flight[s]) {
        vkWaitForFences(vk().device, 1, &fences[s], VK_TRUE, UINT64_MAX);
        in_flight[s] = false;
        uint64_t ts[2];
        if (tpool && vkGetQueryPoolResults(vk().device, tpool, s * 2, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && ts[1] >= ts[0]) {
            gpu_cb_us += uint64_t(double(ts[1] - ts[0]) * vk().props.limits.timestampPeriod * 1e-3);
            ++gpu_cbs;
        }
        if (ts_every) ts_read(s);
    }
    for (const Writeback& w : wb_s[s]) { std::memcpy(w.dst, ring.mapped + w.ring_offset, w.size); wb_bytes += w.size; }
    wb_s[s].clear();
    for (const auto& d : dirty_s[s]) {  // GPU stores into imported memory bypass the write watch; textures there may be sampled next
        ww_mark(d.first, d.second);
        tex_touch(d.first, d.second);
        dirty_count(d.first, d.second, -1);
    }
    dirty_s[s].clear();
    for (const Label& l : labels_s[s]) std::memcpy(reinterpret_cast<void*>(l.at), &l.value, l.bytes);  // after the write-backs: data before its label
    labels_s[s].clear();
    // the recording slot's retired ranges may still be read by its unsubmitted commands; an open or replaying tick's by the replay
    if (!(s == slot && recording) && !(tick_slots >> s & 1) && hold_done(s)) {
        for (const VtxRange& r : retired_s[s]) vtx_free(r);
        retired_s[s].clear();
    }
}

// BB_GPU_TS: slot `s` finished; its intervals go into ts_ms, printed per frame every ts_every frames.
void Backend::ts_read(uint32_t s) {
    const size_t n = ts_lab[s].size();
    if (!n) return;
    static std::vector<uint64_t> ts;
    ts.resize(n);
    VkDevice dev = vk().device;
    const double ms = vk().props.limits.timestampPeriod * 1e-6;
    if (vkGetQueryPoolResults(dev, ts_pool[s], 2, uint32_t(n), n * 8, ts.data(), 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return;
    uint64_t first = ts[0], pre[2];
    if (ts_pre[s] && vkGetQueryPoolResults(dev, ts_pool[s], 0, 2, sizeof pre, pre, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && pre[0] <= pre[1] && pre[1] <= ts[0]) {
        auto& e = ts_ms["upload (pre buffer)"]; e.first += double(pre[1] - pre[0]) * ms; ++e.second;
        first = pre[0];
    }
    if (ts_prev_end && first >= ts_prev_end) { auto& e = ts_ms["gap (GPU idle / presenter)"]; e.first += double(first - ts_prev_end) * ms; ++e.second; }
    for (size_t k = 1; k < n; ++k) {
        if (ts[k] < ts[k - 1]) continue;
        std::string& nm = ts_lab[s][k].name;
        if (nm == "barrier") nm += " after " + ts_lab[s][k - 1].name.substr(0, ts_lab[s][k - 1].name.find(' '));  // what it waited for
        auto& e = ts_ms[nm]; e.first += double(ts[k] - ts[k - 1]) * ms; ++e.second;
        ts_dr[nm].first += ts_lab[s][k].draws;
    }
    ts_prev_end = ts[n - 1];
    ts_lab[s].clear();
    const uint64_t fr = g_frame_counter.load(std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    if (!ts_frame0) { ts_frame0 = fr, ts_wall0 = now, ts_ms.clear(), ts_dr.clear(); return; }
    if (fr - ts_frame0 < ts_every) return;
    const double f = double(fr - ts_frame0), wall = std::chrono::duration<double, std::milli>(now - ts_wall0).count() / f;
    std::vector<std::pair<double, const std::string*>> top;
    double busy = 0;
    for (const auto& [k, v] : ts_ms) { top.push_back({v.first, &k}); if (k.compare(0, 3, "gap")) busy += v.first; }
    std::sort(top.rbegin(), top.rend());
    std::fprintf(stderr, "gpu-ts: frames %.0f wall %.3f ms/frame busy %.3f ms/frame labels %zu\n", f, wall, busy / f, top.size());
    for (size_t i = 0; i < top.size() && i < 40; ++i) {
        const auto& v = ts_ms[*top[i].second];
        std::fprintf(stderr, "gpu-ts: %8.3f ms %5.1f%% n %7.1f draws %7.1f | %s\n", top[i].first / f, 100 * top[i].first / f / wall, double(v.second) / f, ts_dr[*top[i].second].first / f, top[i].second->c_str());
    }
    ts_frame0 = fr, ts_wall0 = now;
    ts_ms.clear(), ts_dr.clear();
}

VkCommandBuffer Backend::pre() {
    if (!pre_open) {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(pre_bufs[slot], 0);
        vkBeginCommandBuffer(pre_bufs[slot], &bi);
        if (ts_every) {
            vkCmdResetQueryPool(pre_bufs[slot], ts_pool[slot], 0, 2);
            vkCmdWriteTimestamp(pre_bufs[slot], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ts_pool[slot], 0);
        }
        ts_pre[slot] = ts_every != 0;
        pre_open = true;
    }
    return pre_bufs[slot];
}

// Reading guest memory for the GPU or the CPU (buffer copies, textures, indices, indirect arguments): GPU work that writes the range must
// have run and its write-backs landed. Recorded but unsubmitted work is submitted and waited for (callers inside draw/dispatch redo their
// descriptor writes after such a flush); work in flight on the other slot is waited for.
void Backend::settle_pending(uint64_t lo, uint64_t hi, int why) {
    if (!maybe_dirty(lo, hi)) return;
    const size_t n = std::min(dirty_s[slot].size(), dirty_own);  // not the ranges the draw/dispatch being recorded writes itself
    for (size_t k = 0; k < n; ++k)
        if (dirty_s[slot][k].first < hi && lo < dirty_s[slot][k].second) {
            ++sync_cur, ++sync_cur_by[why];
            for (const Writeback& w : wb_s[slot]) if (uint64_t(reinterpret_cast<uintptr_t>(w.dst)) <= lo && hi <= uint64_t(reinterpret_cast<uintptr_t>(w.dst)) + w.size) { ++sync_cur_wb; break; }
            flush(true);
            return;
        }
    for (uint32_t k = 1; k < nslots; ++k)  // submitted slots, oldest first: settling one settles everything older first (write-back order)
        for (const auto& dr : dirty_s[older(k)])
            if (dr.first < hi && lo < dr.second) { ++sync_overlap; for (uint32_t j = 1; j <= k; ++j) settle(older(j)); return; }
}

// A write by the port itself (helper ops on the CPU, GPU write-backs and stores) re-checks the textures it overlaps at their next use.
// (It used to start a new epoch for every texture: ~100 per frame, ~1650 sampled hashes of 12 KB per frame, ~8 % of the render thread.)
void Backend::tex_touch(uint64_t lo, uint64_t hi) {
    for (auto it = texs.lower_bound({lo > tex_max_bytes ? lo - tex_max_bytes : 0, 0, 0}); it != texs.end() && it->first[0] < hi; ++it)
        if (Image* im = it->second.get(); im->base + im->bytes > lo) im->hash_epoch = 0, ++im->touched;  // (memo entries of it: stale)
}

// Submits the recorded work. Resources to free, profiling or `wait` make the flush synchronous; otherwise the GPU runs the work while the
// CPU records into the other slot, and a slot is waited for (and its write-backs applied) when it is reused or when something reads
// memory it writes (settle_pending). ponytail: CPU code that reads GPU-written memory without such a read through the backend (guest
// threads polling a result buffer) sees it up to two flushes late; the engine's completion labels are written when the PM4 stream is scanned.
void Backend::flush(bool wait) {
    if (!recording) return;
    ++flushes;
    ops_since_flush = 0;
    flush_pending = false;  // whatever triggered this flush also submits a coalesced end-of-submit
    VkCtx& c = vk();
    end_pass();
    if (tpool) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tpool, slot * 2 + 1);
    if (ts_every) ts_mark("other");  // closes the last interval
    vkEndCommandBuffer(cb);
    recording = false;
    VkCommandBuffer cbs[2] = {pre_bufs[slot], cb};
    if (pre_open) {  // its uploads complete before anything in the main command buffer of this submission
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(pre_bufs[slot], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        if (ts_every) vkCmdWriteTimestamp(pre_bufs[slot], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool[slot], 1);
        vkEndCommandBuffer(pre_bufs[slot]);
    }
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = pre_open ? 2 : 1;
    si.pCommandBuffers = pre_open ? cbs : &cb;
    if (replay_on) {  // the slot's segments in order
        submit_cbs.clear();
        if (pre_open) submit_cbs.push_back(pre_bufs[slot]);
        submit_cbs.insert(submit_cbs.end(), seg_pool[slot].begin(), seg_pool[slot].begin() + seg_used[slot]);
        si.commandBufferCount = uint32_t(submit_cbs.size());
        si.pCommandBuffers = submit_cbs.data();
    }
    pre_open = false;
    const auto ft0 = tnow();
    {
        std::lock_guard<std::mutex> lk(c.queue_mutex);
        vkResetFences(c.device, 1, &fences[slot]);
        vkQueueSubmit(c.queue, 1, &si, fences[slot]);
    }
    ft_submit += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(tnow() - ft0).count());
    in_flight[slot] = true;
    // GPU stores into imported guest memory land whenever the GPU runs them (as on the console). Guest reads that must see them go
    // through sync_read (dirty_s) and completion labels wait for the fence (settle / fence watcher), so no wait here. BB_IMP_SYNC=1: wait.
    static const bool imp_sync = std::getenv("BB_IMP_SYNC") && std::getenv("BB_IMP_SYNC")[0] == '1';
    const bool sync = wait || sync_next || (imp_sync && imp_written) || !graveyard_pending.empty() || hostbufs.size() > 64 || (prof && qn);
    sync_imp += imp_written;
    imp_written = false;
    sync_next = false;
    if (sync) {
        ++sync_n; sync_grave += !graveyard_pending.empty(); sync_host += hostbufs.size() > 64; sync_wait += wait;
        { ScopeNs sw{ft_syncwait};
          for (uint32_t k = 1; k <= nslots; ++k) settle(older(k)); }  // oldest first, ending with the one just submitted
        if (prof && qn) {
            std::vector<uint64_t> ts(qn);
            if (vkGetQueryPoolResults(c.device, qpool, 0, qn, ts.size() * 8, ts.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
                for (uint32_t k = 0; k + 1 < qn && k / 2 < qname.size(); k += 2) {
                    auto& e = gpu_ms[qname[k / 2]];
                    e.first += double(ts[k + 1] - ts[k]) * c.props.limits.timestampPeriod * 1e-6;
                    ++e.second;
                }
            qn = 0;
        }
        if (replay_on && (hostbufs.size() > 64 || !graveyard_pending.empty())) {  // replays may still use them; the open tick's segments do
            for (ReplaySet& rs : rsets) wait_replay(rs);
            drop_tick();
        }
        if (hostbufs.size() > 64) drop_host_bufs();  // the GPU is idle: nothing references the imports any more
        for (Image* im : graveyard_pending) destroy_image(im);
        graveyard_pending.clear();
    }
    slot = (slot + 1) % nslots;
    cb = cmd_bufs[slot];
    dpool = dpools[slot];
    {
        const auto t1 = tnow();
        if (tick_slots >> slot & 1) drop_tick();  // the open tick has used every slot: its oldest is reused now
        if (hold[slot]) {
            const auto h0 = std::chrono::steady_clock::now();
            vkWaitForFences(c.device, 1, &hold[slot], VK_TRUE, UINT64_MAX);
            hold_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - h0).count());
        }
        settle(slot);  // the oldest submitted slot must be finished before its command buffer, pool and ring part are reused
        ft_settle += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(tnow() - t1).count());
    }
    seg_used[slot] = 0;
    vkResetDescriptorPool(c.device, dpool, 0);
    ring_used = slot_begin();
    if (dirty_own != SIZE_MAX) dirty_own = 0;  // an op being recorded: everything the new slot collects from here on is its own
    begin();
    sweep_vertex_cache();  // both retire into the new slot: freed once it (and all older work) completed
    if (vtx_cache_bytes > kVtxBudget || vtx_full) evict_vertex_cache();
}

// Replay mode: ends the recording segment and starts the next one (same slot, no submit). A safe segment after an unsafe one opens with a
// full barrier: the replay skips the unsafe segment and with it the barriers recorded there.
void Backend::cut(bool unsafe) {
    if (!replay_on || unsafe == cur_unsafe) return;
    end_pass();
    vkEndCommandBuffer(cb);
    recording = false;
    cur_unsafe = unsafe;
    begin();
    if (!unsafe) barrier();
}

bool Backend::hold_done(uint32_t s) {
    if (hold[s] && vkGetFenceStatus(vk().device, hold[s]) != VK_SUCCESS) return false;
    hold[s] = VK_NULL_HANDLE;
    return true;
}

void Backend::wait_replay(ReplaySet& rs) {
    if (!rs.used) return;
    VkDevice dev = vk().device;
    vkWaitForFences(dev, 1, &rs.fence, VK_TRUE, UINT64_MAX);
    rs.used = false;
    uint64_t ts[2];
    if (vkGetQueryPoolResults(dev, rs.q, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && ts[1] >= ts[0])
        replay_gpu_ms = double(ts[1] - ts[0]) * vk().props.limits.timestampPeriod * 1e-6;
    for (VkFence& h : hold) if (h == rs.fence) h = VK_NULL_HANDLE;
}

void Backend::note_site(const uint8_t* p, size_t off) {
    const auto [it, fresh] = block_ids.try_emplace(hash_fast(p, 864), uint32_t(blocks.size()));
    if (fresh) {
        if (blocks.size() == kMaxBlocks) { block_ids.erase(it); return; }
        blocks.emplace_back();
        std::memcpy(blocks.back().data(), p, 864);
    }
    sites.push_back({off, it->second});
    if (!tick_segs.empty()) tick_segs.back().patched = true;
}

void Backend::note_obj(const uint8_t* p, size_t off, size_t size, uint32_t res, bool pal) {
    const uint32_t at = uint32_t(oarena.size()), dw = uint32_t(size / 4);
    oarena.resize(oarena.size() + dw);
    std::memcpy(&oarena[at], p, size_t(dw) * 4);
    osites.push_back({off, res, dw, at, pal});
    if (!tick_segs.empty()) tick_segs.back().patched = true;
}

// Reprojection support (see replay_tick): which depth buffer and viewport the main camera's draws use; the display pass's source; before
// the first blended draw into that source (the HUD) a copy of it = the scene.
void Backend::warp_draw(const RegView& r, Image* rt0, uint32_t rt0_slot, Image* dsimg, bool use_z, size_t sites0, const char* who) {
    bool main = false;
    for (size_t s = sites0; s < sites.size() && !main; ++s) main = disp_block(blocks[sites[s].block]);
    if (main && dsimg && use_z) {
        auto it = std::find_if(main_ds.begin(), main_ds.end(), [&](const DsUse& u) { return u.ds == dsimg; });
        if (it == main_ds.end()) main_ds.push_back({dsimg, 1});
        else ++it->n;
        std::memcpy(main_vport, &r.context[kPaClVportXscale], sizeof main_vport);
        for (int k = 0; k < 4; ++k) main_vport[k] *= float(dsimg->scale);  // x/y scale and offset in target pixels (BB_RES_SCALE; the presenter's snapshots are that size)
    }
    const bool disp = rt0 && std::find(std::begin(disp_addr), std::end(disp_addr), rt0->base) != std::end(disp_addr);
    const bool blend = rt0 && (r.context[kCbBlend0 + rt0_slot] >> 30) & 1;
    ++diag_draw;
    if (warp_diag && tick_no % 300 == 150 && diag_lines.size() < 160 && rt0 && !main && rt0->w >= 1280 &&
        (rt0->format == VK_FORMAT_B8G8R8A8_UNORM || rt0->format == VK_FORMAT_R8G8B8A8_UNORM)) {  // (where the HUD goes)
        char b[600];
        int n = std::snprintf(b, sizeof b, "replay diag: draw %u: rt %llx %ux%u fmt %d blend %d (ctl %08x) ps %s samples", diag_draw, (unsigned long long)rt0->base,
                              rt0->w, rt0->h, int(rt0->format), int(blend), r.context[kCbBlend0 + rt0_slot], who);
        for (const Image* s : sampled)
            if (n > 0 && n < int(sizeof b) - 60) n += std::snprintf(b + n, sizeof b - size_t(n), " %llx %ux%u fmt %d", (unsigned long long)s->base, s->w, s->h, int(s->format));
        diag_lines.push_back(std::string(disp ? "[display] " : "") + b);
    }
    if (disp && !blend) {  // the display pass: its largest sampled image is the source
        disp_src_img = nullptr;
        for (const Image* s : sampled)
            if (!disp_src_img || uint64_t(s->w) * s->h > uint64_t(disp_src_img->w) * disp_src_img->h) disp_src_img = s;
        disp_src = disp_src_img ? disp_src_img->base : 0;
        return;
    }
    if (!rt0 || !blend || scene_taken || !disp_src || rt0->base != disp_src) return;  // the scene = the source before the first HUD draw:
    for (const Image* s : sampled)  // a blended draw that samples a render target is still the scene's (bloom); the HUD samples textures
        if (auto it = rts.find(s->base); it != rts.end() && it->second.get() == s) return;
    if (scene_img && (scene_img->w != rt0->w || scene_img->h != rt0->h || scene_img->format != rt0->format)) graveyard_pending.push_back(scene_img), scene_img = nullptr;
    if (!scene_img) {
        scene_img = make_image(rt0->w, rt0->h, 1, rt0->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        if (!scene_img) return;
        ensure_init(scene_img);
    }
    end_pass();
    barrier();
    copy_image(rcb(), rt0, scene_img);  // (rcb: the barrier after it is recorded, the HUD draws write rt0 next)
    barrier();
    scene_taken = true;
}

void Backend::end_obj_draw() {  // the draw's sites are complete: count and content hash (unchanged draws need no patch)
    if (!obj_capture) return;
    obj_capture = false;
    ObjDraw& od = odraws.back();
    od.n = uint32_t(osites.size()) - od.first;
    od.hash = od.n ? hash_fast(&oarena[osites[od.first].at], (oarena.size() - osites[od.first].at) * 4) : 0;
}

// The op being recorded writes `im` (before its first command): a first write after a read this tick saves the pre-tick contents.
void Backend::note_write(Image* im) {
    if (!replay_carry || (im->acc_tick == tick_no && im->acc == 2)) return;
    const bool read_first = im->acc_tick == tick_no;
    im->acc_tick = tick_no, im->acc = 2;
    if (!im->carry || tick_no - im->rf_tick > 16) return;
    im->twin_tick = tick_no;
    carry_post.push_back(im);
    if (!read_first || !im->initialised) return;  // (never written yet: nothing to carry, and its first-use clear comes later)
    end_pass();  // (twin's layout barrier and the copy are not allowed inside a rendering instance)
    if (!twin(im->pre, im, cb)) return;
    barrier();
    copy_image(rcb(), im, im->pre);  // (rcb: the barrier after it is recorded, the op then writes im)
    barrier();
    carry_pre.push_back(im);
}

// A transfer-only copy of `im` (same format, size, layers), made GENERAL in `cmd` when created. ponytail: kept while `im` lives.
Image* Backend::twin(Image*& tw, const Image* im, VkCommandBuffer cmd) {
    if (tw) return tw;
    auto* t = new Image;
    t->w = im->w, t->h = im->h, t->layers = im->layers, t->levels = im->levels, t->format = im->format, t->view_type = im->view_type, t->ds = im->ds;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    const bool v3 = im->view_type == VK_IMAGE_VIEW_TYPE_3D, v1 = im->view_type == VK_IMAGE_VIEW_TYPE_1D || im->view_type == VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    ici.imageType = v3 ? VK_IMAGE_TYPE_3D : v1 ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D;
    ici.format = im->format;
    ici.extent = {im->w, v1 ? 1u : im->h, v3 ? im->layers : 1u};
    ici.mipLevels = im->levels;
    ici.arrayLayers = v3 ? 1 : im->layers;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(vk().vma, &ici, &aci, &t->image, &t->alloc, nullptr) != VK_SUCCESS) { delete t; return nullptr; }
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = t->image;
    b.subresourceRange = {VkImageAspectFlags(im->format == VK_FORMAT_D32_SFLOAT_S8_UINT ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                             : im->format == VK_FORMAT_D32_SFLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT : im->format == VK_FORMAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
                          0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    twin_bytes += twin_size(im);
    return tw = t;
}

void Backend::copy_image(VkCommandBuffer cmd, const Image* src, const Image* dst) {  // whole image, all levels, layers and aspects (both GENERAL)
    const bool v3 = src->view_type == VK_IMAGE_VIEW_TYPE_3D;
    VkImageCopy cp[16]{};  // (depth + stencil, or one per colour level: at most 15)
    uint32_t n = 0;
    for (const VkImageAspectFlags a : {VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT), VkImageAspectFlags(VK_IMAGE_ASPECT_STENCIL_BIT), VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT)}) {
        const bool has = a == VK_IMAGE_ASPECT_COLOR_BIT ? !src->ds : a == VK_IMAGE_ASPECT_DEPTH_BIT ? src->ds && src->format != VK_FORMAT_S8_UINT : src->ds && src->format != VK_FORMAT_D32_SFLOAT;
        if (!has) continue;
        for (uint32_t l = 0; l < src->levels; ++l, ++n) {
            cp[n].srcSubresource = cp[n].dstSubresource = {a, l, 0, v3 ? 1u : src->layers};
            cp[n].extent = {std::max(1u, src->w >> l), std::max(1u, src->h >> l), v3 ? src->layers : 1u};
        }
    }
    vkCmdCopyImage(cmd, src->image, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, n, cp);
}

// Tick end (guest flip, render thread): dst[n-1] = the tick's picture; dst[i-1] (i = 1..n-1) = the tick's safe segments run again with the
// camera and object constants at alpha[i-1] between the previous tick and this one. One submit: glue command buffers between the runs copy the display
// target out and patch the constants in the ring (queue order, no CPU wait). Without a usable replay every dst gets the tick's picture.
bool Backend::replay_tick(uint64_t addr, uint32_t width, uint32_t height, const VkImage* dst, VkExtent2D ext, uint32_t n, const float* alpha, ReplayWarp* warp, uint64_t* ready,
                          ReplayResult* res) {
    const auto t0 = std::chrono::steady_clock::now();
    VkCtx& c = vk();
    flush(false);  // the tick's last work reaches the queue; the new slot's first segment belongs to the next tick
    const auto t_flush = std::chrono::steady_clock::now();
    auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    rp_segs.swap(tick_segs);
    tick_segs.clear();
    if (!rp_segs.empty() && rp_segs.back().cb == cb) { tick_segs.push_back(rp_segs.back()); rp_segs.pop_back(); }
    const uint32_t held = tick_slots & ~(1u << slot);
    tick_slots = 1u << slot;
    const bool dropped = tick_dropped;
    tick_dropped = false;
    struct Finish {
        Backend& b;
        ~Finish() {
            b.replay_hold_ms = double(b.hold_ns) / 1e6;
            b.prev_blocks.swap(b.blocks); b.blocks.clear(); b.block_ids.clear(); b.sites.clear(); b.tick_imp_reads = 0; b.hold_ns = 0;
            b.prev_osites.swap(b.osites); b.osites.clear(); b.prev_odraws.swap(b.odraws); b.odraws.clear(); b.prev_oarena.swap(b.oarena); b.oarena.clear();
            b.carry_pre.clear(); b.carry_post.clear(); ++b.tick_no;
            b.scene_taken = false; b.disp_src_img = nullptr; b.main_ds.clear(); b.diag_lines.clear(); b.diag_draw = 0;
        }
    } fin{*this};
    const auto it = rts.find(addr);
    if (it == rts.end()) return false;
    const Image* im = it->second.get();
    n = std::clamp(n, 1u, kMaxReplayN);
    size_t safe = 0;
    size_t unpatched = 0;  // safe segments without a patch site (statistics: what culling could skip at most)
    for (const Seg& s : rp_segs) safe += !s.unsafe, unpatched += !s.unsafe && !s.patched;
    ReplaySet& rs = rsets[rset_next];
    rset_next = (rset_next + 1) % kReplaySets;
    wait_replay(rs);
    vkResetFences(c.device, 1, &rs.fence);
    if (std::find(std::begin(disp_addr), std::end(disp_addr), addr) == std::end(disp_addr)) std::rotate(std::begin(disp_addr), std::begin(disp_addr) + 3, std::end(disp_addr)), disp_addr[0] = addr;
    disp_w = width, disp_h = height;

    // The main camera = the display-sized block with the most sites, its partner = the previous tick's block of the same class nearest in
    // dwords 8-35 (as for the interpolation below). A camera cut (no partner, or the camera moved more than 8 units or turned more than
    // 45 degrees between the ticks; walking and turning in the clinic: at most 1.7 units, the camera's collision snaps 2-3 units, both
    // interpolated): no replays and no reprojection motion (the presenter shows this tick's picture for the whole window).
    int main_b = -1;
    const std::array<uint32_t, 216>* cam_prev = nullptr;
    site_n.assign(blocks.size(), 0);
    for (const Site& s : sites) ++site_n[s.block];
    for (size_t b = 0; b < blocks.size(); ++b)
        if (disp_block(blocks[b]) && (main_b < 0 || site_n[b] > site_n[size_t(main_b)])) main_b = int(b);
    bool cam_cut = main_b >= 0;
    double cam_dr = -1, cam_move = -1, cam_turn = -2;  // (BB_INTERP_LOG: partner distance / norm, camera move, cos of the turn)
    if (main_b >= 0) {
        const auto& cur = blocks[size_t(main_b)];
        double best_d = 0, norm = 0;
        for (int j = 8; j < 36; ++j) { float v; std::memcpy(&v, &cur[j], 4); norm += std::fabs(double(v)); }
        for (const auto& pb : prev_blocks) {
            if (pb[0] != cur[0] || pb[4] != cur[4] || pb[5] != cur[5]) continue;
            double d = 0;
            for (int j = 8; j < 36; ++j) { float a, v; std::memcpy(&a, &pb[j], 4); std::memcpy(&v, &cur[j], 4); d += std::fabs(double(a) - double(v)); }
            if (!cam_prev || d < best_d) cam_prev = &pb, best_d = d;
        }
        if (cam_prev) cam_dr = best_d / std::max(norm, 1e-30);
        if (cam_prev && best_d <= 0.25 * norm) {
            auto V = [](const std::array<uint32_t, 216>& b, int i) { float v; std::memcpy(&v, &b[8 + i], 4); return double(v); };  // view 3x4, row-major
            auto pos = [&](const std::array<uint32_t, 216>& b, int k) { return -(V(b, k) * V(b, 3) + V(b, 4 + k) * V(b, 7) + V(b, 8 + k) * V(b, 11)); };
            double move = 0, turn = 0;
            for (int k = 0; k < 3; ++k) move += (pos(cur, k) - pos(*cam_prev, k)) * (pos(cur, k) - pos(*cam_prev, k)), turn += V(cur, 8 + k) * V(*cam_prev, 8 + k);
            cam_cut = !(move <= 64.0 && turn >= 0.7071);  // (forward rows: unit vectors)
            cam_move = std::sqrt(move), cam_turn = turn;
        }
    }
    const bool replay = !dropped && safe && n > 1 && !cam_cut;
    rp_fallbacks += !replay;
    if (res) res->cut = cam_cut, res->replayed = replay;

    // Reprojection: the main camera (above, its motion from the partner unless a cut) and the depth buffer most of its draws used.
    const Image* wds = nullptr;
    if (warp) {
        warp->ok = false;
        uint32_t best_n = 0;
        for (const DsUse& u : main_ds) if (u.n > best_n) wds = u.ds, best_n = u.n;
        if (main_b >= 0 && wds && wds->gw() >= width && wds->gh() >= height && wds->format != VK_FORMAT_S8_UINT) {
            const auto& cur = blocks[size_t(main_b)];
            std::memcpy(warp->cam0, &(cam_cut ? cur : *cam_prev)[kCamAt], sizeof warp->cam0);
            std::memcpy(warp->cam1, &cur[kCamAt], sizeof warp->cam1);
            std::memcpy(warp->vport, main_vport, sizeof main_vport);
            warp->ok = true;
        }
    }

    // Lerp per dword (camera and objects): when both values are normal floats (or 0) and the step is at most half their magnitude;
    // anything else (ints, flags, pointers, jumps) takes this tick's value.
    auto f = [](uint32_t u) { float v; std::memcpy(&v, &u, 4); return v; };
    auto ok = [](uint32_t u) { const uint32_t e = u >> 23 & 0xFF; return (e && e != 0xFF) || !(u & 0x7FFFFFFF); };  // normal float or +-0
    auto lerp = [&](uint32_t a, uint32_t b, float alpha, uint32_t& out) {
        out = b;
        if (a == b || !ok(a) || !ok(b)) return false;
        const float x = f(a), y = f(b);
        if (std::fabs(y - x) > 0.5f * std::max({1.0f, std::fabs(x), std::fabs(y)})) return false;
        const float o = x + (y - x) * alpha;
        std::memcpy(&out, &o, 4);
        return true;
    };
    size_t st_used = 0;
    auto st_alloc = [&](size_t bytes) -> uint32_t* {  // staging space for one patch; nullptr when full (that patch is skipped)
        if (st_used + bytes > rs.staging.size) return nullptr;
        uint32_t* p = reinterpret_cast<uint32_t*>(rs.staging.mapped + st_used);
        st_used += bytes;
        return p;
    };
    for (auto& v : rp_regions) v.clear();

    // Camera blocks: partner = the previous tick's block of the same class (far plane, resolution) with the nearest view matrices
    // (dwords 8-35); farther than 0.25 of their norm = a camera cut (not interpolated).
    uint32_t matched = 0, cuts = 0, lerped = 0, lerped_max = 0;
    if (replay && replay_lerp && !blocks.empty()) {
        const size_t nb = blocks.size();
        cam_off.assign(nb * (n - 1), SIZE_MAX);
        for (size_t b = 0; b < nb; ++b) {
            const auto& cur = blocks[b];
            const std::array<uint32_t, 216>* best = nullptr;
            double best_d = 0, norm = 0;
            for (int j = 8; j < 36; ++j) norm += std::fabs(double(f(cur[j])));
            for (const auto& pb : prev_blocks) {
                if (pb[0] != cur[0] || pb[4] != cur[4] || pb[5] != cur[5]) continue;
                double d = 0;
                for (int j = 8; j < 36; ++j) d += std::fabs(double(f(pb[j])) - double(f(cur[j])));
                if (!best || d < best_d) best = &pb, best_d = d;
            }
            if (!best) continue;
            ++matched;
            if (!(best_d <= 0.25 * norm)) { ++cuts; continue; }  // (also NaN)
            uint32_t nl = 0;
            for (uint32_t i = 1; i < n; ++i) {
                uint32_t* out = st_alloc(864);
                if (!out) break;
                cam_off[(i - 1) * nb + b] = size_t(reinterpret_cast<uint8_t*>(out) - rs.staging.mapped);
                for (int j = 0; j < 216; ++j) nl += lerp((*best)[j], cur[j], alpha[i - 1], out[j]) && i == 1;
            }
            if (!nl) for (uint32_t i = 1; i < n; ++i) cam_off[(i - 1) * nb + b] = SIZE_MAX;
            lerped += nl;
            lerped_max = std::max(lerped_max, nl);
        }
        for (uint32_t i = 1; i < n; ++i)
            for (const Site& s : sites)
                if (const size_t so = cam_off[(i - 1) * nb + s.block]; so != SIZE_MAX) rp_regions[i].push_back({VkDeviceSize(so), VkDeviceSize(s.off), 864});
    }

    const auto t_cam = std::chrono::steady_clock::now();
    // Objects: draws whose constants are unchanged (same key and bytes as a draw of the previous tick) need nothing; the others take the
    // previous tick's draw of the same key and site layout with the nearest scalar constants (cut: farther than 0.25 of their norm).
    uint32_t o_changed = 0, o_pairs = 0, o_cuts = 0, o_dw = 0, o_pal = 0;
    if (replay && replay_obj && !odraws.empty() && !prev_odraws.empty()) {
        for (auto& kv : obj_pmap) kv.second.clear();  // (keeps the vectors' capacity)
        if (obj_pmap.size() > 4 * prev_odraws.size() + 1024) obj_pmap.clear();  // keys no longer drawn
        obj_phash.clear();
        obj_occ.clear();
        for (uint32_t k = 0; k < prev_odraws.size(); ++k) {
            if (!prev_odraws[k].n) continue;
            obj_pmap[prev_odraws[k].key].push_back(k);
            obj_phash.at(prev_odraws[k].key ^ prev_odraws[k].hash * 0x9E3779B97F4A7C15ull) = 1;
        }
        auto dist = [&](const ObjDraw& pd, const ObjDraw& od, float& dd, float& nn) {  // same site layout? then L1 distance and norm of the scalars
            dd = nn = 0;
            if (pd.n != od.n) return false;
            for (uint32_t s = 0; s < od.n; ++s) {
                const ObjSite& a = prev_osites[pd.first + s];
                const ObjSite& b = osites[od.first + s];
                if (a.res != b.res || a.dw != b.dw) return false;
                if (!b.pal)
                    for (uint32_t j = 0; j < std::min(b.dw, 64u); ++j)  // (the head of a block: transforms; the whole block cost ~1 ms per tick)
                        if (const uint32_t x = prev_oarena[a.at + j], y = oarena[b.at + j]; ok(x) && ok(y)) dd += std::fabs(f(x) - f(y)), nn += std::fabs(f(y));
            }
            return true;
        };
        for (const ObjDraw& od : odraws) {
            if (!od.n) continue;
            const uint32_t occ = obj_occ.at(od.key)++;  // position among the tick's draws of this key
            if (obj_phash.at(od.key ^ od.hash * 0x9E3779B97F4A7C15ull)) continue;
            ++o_changed;
            const auto pit = obj_pmap.find(od.key);
            if (pit == obj_pmap.end() || pit->second.empty()) continue;
            float norm = 0, best_d = 0, dd = 0, nn = 0;
            // the draw at the same position among its key first (the order holds while nothing appears or disappears), else the nearest
            const ObjDraw* best = nullptr;
            if (occ < pit->second.size() && dist(prev_odraws[pit->second[occ]], od, dd, nn) && dd <= 0.25f * nn) best = &prev_odraws[pit->second[occ]], best_d = dd, norm = nn;
            else
                for (const uint32_t k : pit->second)
                    if (dist(prev_odraws[k], od, dd, nn) && (!best || dd < best_d)) best = &prev_odraws[k], best_d = dd, norm = nn;
            if (!best) continue;
            ++o_pairs;
            if (!(best_d <= 0.25f * norm)) { ++o_cuts; continue; }
            for (uint32_t s = 0; s < od.n; ++s) {
                const ObjSite& a = prev_osites[best->first + s];
                const ObjSite& b = osites[od.first + s];
                const uint32_t* pa = &prev_oarena[a.at];
                const uint32_t* pb = &oarena[b.at];
                if (std::memcmp(pa, pb, size_t(b.dw) * 4) == 0) continue;
                for (uint32_t i = 1; i < n; ++i) {
                    uint32_t* out = st_alloc(size_t(b.dw) * 4);
                    if (!out) break;
                    uint32_t nl = 0;
                    for (uint32_t j = 0; j < b.dw; ++j) nl += lerp(pa[j], pb[j], alpha[i - 1], out[j]);
                    if (!nl) { st_used -= size_t(b.dw) * 4; break; }  // nothing continuous: leave this tick's bytes
                    rp_regions[i].push_back({VkDeviceSize(reinterpret_cast<uint8_t*>(out) - rs.staging.mapped), VkDeviceSize(b.off), VkDeviceSize(b.dw) * 4});
                    if (i == 1) o_dw += nl, o_pal += b.pal;
                }
            }
        }
    }
    if (st_used) vmaFlushAllocation(c.vma, rs.staging.alloc, 0, st_used);
    const auto t_obj = std::chrono::steady_clock::now();

    VkMemoryBarrier full{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    auto glue = [&](uint32_t i) {
        VkCommandBuffer g = rs.glue[i];
        const VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        vkResetCommandBuffer(g, 0);
        vkBeginCommandBuffer(g, &bi);
        return g;
    };
    auto blit = [&](VkCommandBuffer g, const Image* src, VkImage d) {  // src (GENERAL) -> d, then d readable by the presenter
        const VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;  // overwritten completely
        ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image = d;
        ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(g, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 1, &ib);
        VkImageBlit bl{};
        bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        bl.srcOffsets[1] = {int32_t(std::min(src->w, width * src->scale)), int32_t(std::min(src->h, height * src->scale)), 1};  // (guest size -> its pixels)
        bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        bl.dstOffsets[1] = {int32_t(ext.width), int32_t(ext.height), 1};
        vkCmdBlitImage(g, src->image, VK_IMAGE_LAYOUT_GENERAL, d, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
        ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ib.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        ib.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(g, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
    };
    auto blit_to = [&](VkCommandBuffer g, uint32_t i) {  // image i: the display target, with `warp` also the HUD inputs and the main depth
        blit(g, im, dst[i]);
        if (!warp || !warp->ok) return;
        const bool hud = scene_taken && disp_src_img;  // else no HUD known: scene = pre (nothing masked)
        blit(g, hud ? scene_img : im, warp->scene[i]);
        blit(g, hud ? disp_src_img : im, warp->pre[i]);
        const VkBufferImageCopy cp{0, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {width * wds->scale, height * wds->scale, 1}};
        vkCmdCopyImageToBuffer(g, wds->image, VK_IMAGE_LAYOUT_GENERAL, warp->depth[i], 1, &cp);
    };
    auto patch = [&](VkCommandBuffer g, uint32_t i) {  // replay i's constants into the ring (after blit_to's barrier: the last run read them)
        if (!rp_regions[i].empty()) vkCmdCopyBuffer(g, rs.staging.buffer, ring.buffer, uint32_t(rp_regions[i].size()), rp_regions[i].data());
    };
    auto end = [&](VkCommandBuffer g) {
        vkCmdPipelineBarrier(g, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &full, 0, nullptr, 0, nullptr);
    };
    auto save_post = [&](VkCommandBuffer g) {  // carried state after the original run
        for (Image* im : carry_post) if (twin(im->post, im, g)) copy_image(g, im, im->post);
    };
    auto restore = [&](VkCommandBuffer g, bool post) {  // carried state before the tick (another replay follows) or after the original (the last)
        for (Image* im : post ? carry_post : carry_pre) if (const Image* t = post ? im->post : im->pre) copy_image(g, t, im);
    };
    submit_cbs.clear();
    VkCommandBuffer g0 = glue(0);
    vkCmdResetQueryPool(g0, rs.q, 0, 2);
    if (!replay) {
        for (uint32_t i = 0; i < n; ++i) blit_to(g0, i);
        end(g0);
    } else {
        blit_to(g0, n - 1);
        save_post(g0);
        end(g0);
        restore(g0, false);
        patch(g0, 1);
        end(g0);
        vkCmdWriteTimestamp(g0, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, rs.q, 0);
    }
    vkEndCommandBuffer(g0);
    submit_cbs.push_back(g0);
    uint32_t batch_end[kMaxReplayN];  // (with `ready`) submit_cbs index past each image's batch: g0 = the own picture, then replay i
    uint32_t nb = 0;
    batch_end[nb++] = uint32_t(submit_cbs.size());
    for (uint32_t i = 1; replay && i < n; ++i) {
        for (const Seg& s : rp_segs) if (!s.unsafe) submit_cbs.push_back(s.cb);
        VkCommandBuffer g = glue(i);
        if (i == 1) vkCmdWriteTimestamp(g, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, rs.q, 1);
        blit_to(g, i - 1);
        end(g);
        restore(g, i + 1 == n);
        if (i + 1 < n) patch(g, i + 1);
        end(g);
        vkEndCommandBuffer(g);
        submit_cbs.push_back(g);
        batch_end[nb++] = uint32_t(submit_cbs.size());
    }
    if (ready && !ready_sem) {
        VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, nullptr, VK_SEMAPHORE_TYPE_TIMELINE, 0};
        const VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti, 0};
        if (vkCreateSemaphore(c.device, &sci, nullptr, &ready_sem) != VK_SUCCESS) ready_sem = VK_NULL_HANDLE;
    }
    if (ready && ready_sem) {  // one batch per image, each signalling the next value: own picture = +1, replay i (image i - 1) = +1 + i
        VkSubmitInfo si[kMaxReplayN];
        VkTimelineSemaphoreSubmitInfo ti[kMaxReplayN];
        uint64_t val[kMaxReplayN];
        for (uint32_t b = 0; b < nb; ++b) {
            val[b] = ready_val + 1 + b;
            ti[b] = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, nullptr, 0, nullptr, 1, &val[b]};
            si[b] = {VK_STRUCTURE_TYPE_SUBMIT_INFO, &ti[b]};
            si[b].commandBufferCount = batch_end[b] - (b ? batch_end[b - 1] : 0);
            si[b].pCommandBuffers = submit_cbs.data() + (b ? batch_end[b - 1] : 0);
            si[b].signalSemaphoreCount = 1;
            si[b].pSignalSemaphores = &ready_sem;
        }
        for (uint32_t i = 0; i < n; ++i) ready[i] = replay ? (i + 1 == n ? val[0] : val[i + 1]) : val[0];
        ready_val += nb;
        std::lock_guard<std::mutex> lk(c.queue_mutex);
        vkQueueSubmit(c.queue, nb, si, rs.fence);
    } else {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = uint32_t(submit_cbs.size());
        si.pCommandBuffers = submit_cbs.data();
        std::lock_guard<std::mutex> lk(c.queue_mutex);
        vkQueueSubmit(c.queue, 1, &si, rs.fence);
    }
    rs.used = true;
    if (replay)
        for (uint32_t s = 0; s < nslots; ++s) if (held >> s & 1) hold[s] = rs.fence;
    ++rp_ticks;
    if (rp_ticks % 300 == 0) {  // twins unused for 300 ticks (10 s): long out of every command buffer and replay in flight
        auto sweep = [&](Image* x) {
            if ((!x->pre && !x->post) || tick_no - x->twin_tick <= 300) return;
            for (Image** t : {&x->pre, &x->post}) if (*t) destroy_image(*t), *t = nullptr, twin_bytes -= twin_size(x);
        };
        for (auto& kv : rts) sweep(kv.second.get());
        for (auto& kv : dss) sweep(kv.second.get());
        for (auto& kv : texs) sweep(kv.second.get());
    }
    const uint64_t cpu_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    rp_cpu_ns += cpu_ns;
    if (replay_log)
        std::fprintf(stderr, "replay tick=%llu n=%u replayed=%d cam_cut=%d (d %.3f move %.2f turn %.3f) segs=%zu unsafe=%zu unpatched=%zu sites=%zu blocks=%zu matched=%u cut=%u lerped_dw=%u max_dw=%u "
                             "draws=%zu changed=%u paired=%u obj_cut=%u obj_dw=%u pal=%u staging_kb=%zu carry_pre=%zu carry_post=%zu twin_mb=%llu imp_reads=%llu hold_ms=%.2f gpu_ms=%.2f cpu_ms=%.3f "
                             "(flush %.3f cam %.3f obj %.3f glue %.3f)\n",
                     (unsigned long long)rp_ticks, n, int(replay), int(cam_cut), cam_dr, cam_move, cam_turn, rp_segs.size(), rp_segs.size() - safe, unpatched, sites.size(), blocks.size(), matched, cuts, lerped, lerped_max,
                     odraws.size(), o_changed, o_pairs, o_cuts, o_dw, o_pal, st_used >> 10, carry_pre.size(), carry_post.size(), (unsigned long long)(twin_bytes >> 20),
                     (unsigned long long)tick_imp_reads, double(hold_ns) / 1e6, replay_gpu_ms.load(), double(cpu_ns) / 1e6, ms(t0, t_flush), ms(t_flush, t_cam), ms(t_cam, t_obj),
                     ms(t_obj, std::chrono::steady_clock::now()));
    if (warp_diag && tick_no % 300 == 150) {  // reprojection inputs (every 10 s): display-sized camera blocks, the main one, its depth and viewport
        std::fprintf(stderr, "replay diag: tick %llu display %ux%u scene %s main block %d, depth candidates:", (unsigned long long)tick_no, width, height, scene_taken ? "taken" : "not taken", main_b);
        for (const DsUse& u : main_ds) std::fprintf(stderr, " %ux%u fmt %d base %llx x%u", u.ds->w, u.ds->h, int(u.ds->format), (unsigned long long)u.ds->base, u.n);
        std::fprintf(stderr, "; viewport %g %g %g %g %g %g\n", double(main_vport[0]), double(main_vport[1]), double(main_vport[2]), double(main_vport[3]), double(main_vport[4]), double(main_vport[5]));
        std::vector<uint32_t> cnt(blocks.size(), 0);
        for (const Site& s : sites) ++cnt[s.block];
        for (size_t b = 0; b < blocks.size(); ++b) {
            if (cnt[b] < 5 && !disp_block(blocks[b])) continue;
            std::fprintf(stderr, "replay diag: block %zu sites %u display %d dw0-7:", b, cnt[b], int(disp_block(blocks[b])));
            for (int j = 0; j < 8; ++j) std::fprintf(stderr, " %08x", blocks[b][j]);
            if (int(b) == main_b) {
                std::fprintf(stderr, " | dw8-71 as floats:");
                for (int j = 8; j < 72; ++j) { float v; std::memcpy(&v, &blocks[b][j], 4); std::fprintf(stderr, " %g", double(v)); }
            }
            std::fputc('\n', stderr);
        }
        for (const std::string& l : diag_lines) std::fprintf(stderr, "%s\n", l.c_str());
    }
    if (replay_log && rp_ticks % 300 == 0)  // which images carry state (every 10 s)
        for (const Image* ci : carry_post)
            std::fprintf(stderr, "replay carry: %ux%u x%u fmt %d ds %d base 0x%llx%s\n", ci->w, ci->h, ci->layers, int(ci->format), int(ci->ds), (unsigned long long)ci->base,
                         std::find(carry_pre.begin(), carry_pre.end(), ci) != carry_pre.end() ? " (read first)" : "");
    return true;
}

// Guest memory is host memory (identity mapped): big, suitably aligned buffers are imported with VK_EXT_external_memory_host so
// the GPU reads and writes the guest bytes in place. Regions are whole 32 MB blocks when mapped, so repeated sub-ranges of one
// pool share a single import.
bool Backend::get_host_buf(uint64_t base, uint64_t size, VkBuffer& out, VkDeviceSize& offset, bool any_size) {
    VkCtx& c = vk();
    ScopeNs hbt{host_ns};
    if (!c.host_import || (size < (64u << 10) && !any_size) || (base % std::max<VkDeviceSize>(c.props.limits.minStorageBufferOffsetAlignment, 4)) || !hooks().mem_valid) return false;
    const uint64_t end = base + size, A = c.host_import_align;
    auto it = hostbufs.upper_bound(base);
    if (it != hostbufs.begin()) {
        --it;
        if (it->second.start <= base && it->second.end >= end) { out = it->second.buf; offset = base - it->second.start; return true; }
    }
    constexpr uint64_t kBlock = 32u << 20;
    uint64_t rs = base & ~(kBlock - 1), re = (end + kBlock - 1) & ~(kBlock - 1);
    if (!hooks().mem_valid(rs, re - rs)) {
        rs = base & ~(A - 1);
        re = (end + A - 1) & ~(A - 1);
        if (!hooks().mem_valid(rs, re - rs)) return false;
    }
    VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (vkGetMemoryHostPointerPropertiesEXT(c.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, reinterpret_cast<void*>(rs), &hp) != VK_SUCCESS) return false;
    VkExternalMemoryBufferCreateInfo eb{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    eb.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.pNext = &eb;
    bi.size = re - rs;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;  // GDS/DMA copies, indirect args
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    HostBuf hbuf;
    hbuf.start = rs; hbuf.end = re;
    if (vkCreateBuffer(c.device, &bi, nullptr, &hbuf.buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(c.device, hbuf.buf, &mr);
    const uint32_t bits = mr.memoryTypeBits & hp.memoryTypeBits;
    if (!bits) { vkDestroyBuffer(c.device, hbuf.buf, nullptr); log_once("hostbits", "host import: no compatible memory type"); return false; }
    VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = reinterpret_cast<void*>(rs);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &imp;
    ai.allocationSize = re - rs;
    uint32_t ti = 0;
    while (!((bits >> ti) & 1)) ++ti;
    ai.memoryTypeIndex = ti;
    if (vkAllocateMemory(c.device, &ai, nullptr, &hbuf.mem) != VK_SUCCESS || vkBindBufferMemory(c.device, hbuf.buf, hbuf.mem, 0) != VK_SUCCESS) {
        if (hbuf.mem) vkFreeMemory(c.device, hbuf.mem, nullptr);
        vkDestroyBuffer(c.device, hbuf.buf, nullptr);
        log_once("hostalloc", "host import: vkAllocateMemory failed; falling back to copies");
        return false;
    }
    if (auto f = hostbufs.find(rs); f != hostbufs.end()) retired_hostbufs.push_back(f->second);  // still referenced by the recording
    hostbufs[rs] = hbuf;
    ++host_imports;
    host_import_bytes += re - rs;
    out = hbuf.buf;
    offset = base - rs;
    return true;
}

void Backend::drop_host_bufs() {
    VkCtx& c = vk();
    for (auto& [k, h] : hostbufs) {
        vkDestroyBuffer(c.device, h.buf, nullptr);
        vkFreeMemory(c.device, h.mem, nullptr);
    }
    hostbufs.clear();
    for (const HostBuf& h : retired_hostbufs) { vkDestroyBuffer(c.device, h.buf, nullptr); vkFreeMemory(c.device, h.mem, nullptr); }
    retired_hostbufs.clear();
}


bool Backend::vtx_alloc(VkDeviceSize size, VtxRange& r, VkDeviceSize& off) {
    VmaVirtualAllocationCreateInfo ai{};
    ai.size = size;
    ai.alignment = std::max<VkDeviceSize>(256, vk().props.limits.minStorageBufferOffsetAlignment);
    for (uint32_t p = 0; p < vtx_pools.size(); ++p)
        if (vmaVirtualAllocate(vtx_pools[p].block, &ai, &r.va, &off) == VK_SUCCESS) { r.pool = p; return true; }
    // a new pool, up to the budget plus one pool (evicted ranges are freed only once their slot completed)
    if (size > kVtxPoolBytes || (vtx_pools.size() + 1) * kVtxPoolBytes > kVtxBudget + kVtxPoolBytes) return false;
    VtxPool np;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = kVtxPoolBytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateBuffer(vk().vma, &bci, &aci, &np.buf, &np.alloc, nullptr) != VK_SUCCESS) return false;
    VmaVirtualBlockCreateInfo vbi{};
    vbi.size = kVtxPoolBytes;
    if (vmaCreateVirtualBlock(&vbi, &np.block) != VK_SUCCESS) { vmaDestroyBuffer(vk().vma, np.buf, np.alloc); return false; }
    vtx_pools.push_back(np);
    r.pool = uint32_t(vtx_pools.size() - 1);
    return vmaVirtualAllocate(np.block, &ai, &r.va, &off) == VK_SUCCESS;
}

bool Backend::get_vertex_buffer(uint64_t base, uint64_t size, VkBuffer& out, VkDeviceSize& out_off, IndexInfo* ix) {
    // BB_VTX_CACHE=0 turns it off (every draw then copies its streams). The flicker it once caused (ROADMAP 41) came from reads of memory
    // the unsubmitted slot still writes (see settle_pending), which the frequent ring-wrap flushes of the uncached path happened to hide.
    static const bool enabled = !(std::getenv("BB_VTX_CACHE") && std::getenv("BB_VTX_CACHE")[0] == '0');
    if (!enabled || size < 1024 || size > (uint64_t(64) << 20) || overlaps_dirty(base, base + size)) return false;
    ScopeNs vt{vtx_ns};
    const uint64_t frame = g_frame_counter.load(std::memory_order_relaxed);
    const uint64_t key = (base * 0x9E3779B97F4A7C15ull) ^ (size * 0xC2B2AE3D27D4EB4Full) ^ (ix ? 0x5BD1E9955BD1E995ull : 0);  // index use: own entry
    VtxEntry& e = vtx_entry(key);
    const uint8_t* src = reinterpret_cast<const uint8_t*>(base);
    const size_t n8 = size_t(size) & ~size_t(7);
    e.last_frame = frame;
    // Validated on every use: unchanged guest pages (write watch, see write_watch.h) need no hash; written ones (or no tracking: Linux)
    // are hashed, so a rewrite with the same bytes still hits. (Skipping the check for entries already validated in the same submit,
    // until a port write touched their MiBs, was tried: 28 % of the uses, no measurable gain; the write-watch query is not the cost.)
    uint64_t ww_now = 0;
    const int ww = ww_clean_since(base, base + size, e.ww_epoch, ww_now);
    if (static const bool verify = std::getenv("BB_WW_VERIFY") != nullptr; verify && ww == 1 && e.buf && e.size == size && e.base == base) {
        // debug: a "clean" verdict must agree with the hash; a mismatch means a write source the watch does not see. Polled again at
        // once: "late" = the watch sees the write now (it came after this submit's poll of the range), "blind" = it never sees it.
        ++ww_checked;
        if (hash_fast(src, n8) != e.hash) {
            const bool late = ww_repoll(base, base + size, e.ww_epoch) == 0;
            ++(late ? ww_mismatch_late : ww_mismatch);
            if (ww_mismatch + ww_mismatch_late <= 16)
                std::fprintf(stderr, "ww-verify: MISMATCH (%s) base=0x%llx size=%llu (imported region: %d) frame=%llu epoch=%llu\n", late ? "late write" : "blind", (unsigned long long)base, (unsigned long long)size,
                             int(hostbufs.upper_bound(base) != hostbufs.begin() && std::prev(hostbufs.upper_bound(base))->second.end > base), (unsigned long long)g_frame_counter.load(), (unsigned long long)g_tex_epoch.load());
        }
    }
    if (e.buf && e.size == size && e.base == base && (!ix || e.idx_bytes == ix->bytes) && (ww == 1 || (vtx_hash_bytes += n8, hash_fast(src, n8) == e.hash))) {
        e.ww_epoch = ww_now;
        if (ix) { ix->mn = e.idx_mn; ix->mx = e.idx_mx; ++idx_hits; }
        else ++vtx_hits;
        out = e.buf;
        out_off = e.off;
        return true;
    }
    // (re)upload through the ring
    ScopeNs upt{vtx_up_ns};
    vtx_up_bytes += size;
    size_t off = 0;
    uint8_t* p = ring_alloc(size_t(size), 16, off);  // may flush: before any command of this upload is recorded
    if (!p) return false;
    VtxEntry& e2 = vtx_entry(key);  // (ring_alloc's flush may evict; re-find)
    if (e2.buf && (e2.size != size || e2.base != base)) return false;  // key collision: the range may still be in use, so no resize
    if (e2.buf) {  // changed content: earlier commands may still read the old copy, so the new bytes go to a new range (copy on write)
        retired_s[slot].push_back(e2.range);
        vtx_cache_bytes -= e2.size;
        e2.buf = VK_NULL_HANDLE;
    }
    VkDeviceSize doff = 0;
    if (!vtx_alloc(size, e2.range, doff)) { ++vtx_alloc_fails; vtx_full = true; vtx_cache.erase(key); vtx_memo[vtx_slot(key)] = {}; return false; }
    e2.buf = vtx_pools[e2.range.pool].buf;
    e2.off = doff;
    e2.size = size;
    e2.base = base;
    vtx_cache_bytes += size;
    std::memcpy(p, src, size_t(size));
    const VkBufferCopy bc{off, doff, size};
    vkCmdCopyBuffer(pre(), ring.buffer, e2.buf, 1, &bc);  // a fresh range: the copy runs ahead of this slot's commands (see pre_bufs)
    e2.hash = hash_fast(p, n8);  // the bytes the GPU actually gets: guest threads may rewrite `src` concurrently
    e2.ww_epoch = ww_now;        // writes after the poll above show up as written pages next time
    if (ix) {  // the range of the copy the GPU reads
        uint32_t mn = ~0u, mx = 0;
        if (ix->bytes == 4) index_minmax32(reinterpret_cast<const uint32_t*>(p), ix->count, mn, mx);
        else index_minmax16(reinterpret_cast<const uint16_t*>(p), ix->count, mn, mx);
        e2.idx_mn = ix->mn = mn;
        e2.idx_mx = ix->mx = mx;
        e2.idx_bytes = ix->bytes;
        ++idx_uploads;
    } else ++vtx_uploads;
    out = e2.buf;
    out_off = e2.off;
    return true;
}

// Over the budget (end of a flush), or after a pool allocation failed (vtx_full): the least recently used entries are dropped. Their ranges
// are retired into the new slot and freed once it completed (and with it all older work), so nothing waits for the GPU.
void Backend::evict_vertex_cache() {
    const uint64_t frame = g_frame_counter.load(std::memory_order_relaxed);
    if (frame - vtx_evict_frame < 30 && !vtx_full) return;  // the working set may exceed the budget: not a map scan per flush
    vtx_evict_frame = frame;
    const uint64_t target = vtx_full ? std::min(kVtxBudget, vtx_cache_bytes) * 3 / 4 : kVtxBudget * 3 / 4;  // full pools below the budget: a quarter
    vtx_full = false;
    std::fill(vtx_memo.begin(), vtx_memo.end(), VtxMemo{});
    for (uint64_t age = 1000; vtx_cache_bytes > target && age > 0; age /= 4)
        for (auto it = vtx_cache.begin(); it != vtx_cache.end();) {
            if (frame - it->second.last_frame >= age) {
                if (it->second.buf) { retired_s[slot].push_back(it->second.range); vtx_cache_bytes -= it->second.size; }
                ++vtx_evicted;
                it = vtx_cache.erase(it);
            } else ++it;
        }
}

// Entries unused for kVtxMaxAge frames are dropped by an incremental sweep over the hash buckets, a few hundred per flush: dead dynamic
// streams otherwise piled up until the budget eviction above, which then took ~35 ms under the backend lock (every ~16 s in the clinic).
void Backend::sweep_vertex_cache() {
    constexpr uint64_t kVtxMaxAge = 300;  // frames
    const uint64_t frame = g_frame_counter.load(std::memory_order_relaxed);
    const size_t nb = vtx_cache.bucket_count();
    if (vtx_sweep >= nb) vtx_sweep = 0;
    vtx_dead.clear();
    for (size_t k = 0; k < 512 && vtx_sweep < nb; ++k, ++vtx_sweep)
        for (auto it = vtx_cache.begin(vtx_sweep); it != vtx_cache.end(vtx_sweep); ++it)
            if (frame - it->second.last_frame >= kVtxMaxAge) vtx_dead.push_back(it->first);
    vtx_swept += vtx_dead.size();
    for (uint64_t key : vtx_dead) {
        const auto it = vtx_cache.find(key);
        if (it->second.buf) { retired_s[slot].push_back(it->second.range); vtx_cache_bytes -= it->second.size; }
        if (vtx_memo[vtx_slot(key)].key == key) vtx_memo[vtx_slot(key)] = {};
        vtx_cache.erase(it);
    }
}

uint8_t* Backend::ring_alloc(size_t n, size_t align, size_t& offset) {
    size_t at = (ring_used + align - 1) & ~(align - 1);
    if (at + n > slot_end()) {
        flush();
        at = slot_begin();
        if (n > kSlotBytes) return nullptr;
    }
    ring_used = at + n;
    offset = at;
    return ring.mapped + at;
}

Image* Backend::make_image(uint32_t w, uint32_t h, uint32_t layers, VkFormat fmt, VkImageViewType vt, VkImageUsageFlags usage, bool cube, uint32_t levels, bool mutable_fmt) {
    VkCtx& c = vk();
    auto* im = new Image;
    im->w = w; im->h = h; im->layers = layers; im->format = fmt; im->view_type = vt; im->cube_ok = cube; im->levels = levels; im->mutable_fmt = mutable_fmt;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = vt == VK_IMAGE_VIEW_TYPE_3D ? VK_IMAGE_TYPE_3D : (vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_1D_ARRAY) ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = {w, ici.imageType == VK_IMAGE_TYPE_1D ? 1u : h, 1};
    if (vt == VK_IMAGE_VIEW_TYPE_3D) ici.extent.depth = layers;
    ici.mipLevels = levels;
    ici.arrayLayers = vt == VK_IMAGE_VIEW_TYPE_3D ? 1 : layers;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.flags = (cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0) | (mutable_fmt ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0);
    // sRGB + storage is not a supported combination (validation: VUID-VkImageCreateInfo-imageCreateMaxMipLevels-02251): the storage
    // usage is granted through the UNORM twin view
    im->srgb_storage = (usage & VK_IMAGE_USAGE_STORAGE_BIT) && unorm_twin(fmt) != fmt;
    if (im->srgb_storage) { ici.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT; im->mutable_fmt = true; }
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (!vk_check(vmaCreateImage(c.vma, &ici, &aci, &im->image, &im->alloc, nullptr), "vmaCreateImage")) { delete im; return nullptr; }
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    VkImageViewUsageCreateInfo vu{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    vu.usage = usage & ~VkImageUsageFlags(VK_IMAGE_USAGE_STORAGE_BIT);
    if (im->srgb_storage) vci.pNext = &vu;
    vci.image = im->image;
    vci.viewType = vt;
    vci.format = fmt;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, vt == VK_IMAGE_VIEW_TYPE_3D ? 1u : vt == VK_IMAGE_VIEW_TYPE_CUBE ? 6u : layers};  // (a cube array's first cube, VUID 02960)
    if (!vk_check(vkCreateImageView(c.device, &vci, nullptr, &im->view), "image view")) { delete im; return nullptr; }
    return im;
}

void Backend::destroy_image(Image* im) {
    VkCtx& c = vk();
    if (im->view) vkDestroyImageView(c.device, im->view, nullptr);
    for (VkImageView v : im->layer_views) if (v) vkDestroyImageView(c.device, v, nullptr);
    // its texture_view cache entries (keyed by the VkImage handle: a later image reusing the handle would get these stale views)
    // ponytail: linear scan of all views per destroyed image; images die rarely (graveyard, replay twins)
    const uint64_t h = reinterpret_cast<uintptr_t>(im->image);
    std::erase_if(views, [&](const auto& kv) { if (kv.first[0] != h) return false; vkDestroyImageView(c.device, kv.second, nullptr); return true; });
    if (im->image) vmaDestroyImage(c.vma, im->image, im->alloc);
    for (Image* t : {im->pre, im->post}) if (t) destroy_image(t);
    delete im;
}

void Backend::ensure_init(Image* im) {
    if (im->initialised) return;
    im->initialised = true;
    end_pass();  // the initial clear below is recorded outside any rendering instance
    if (im->ds) {
        const VkImageAspectFlags asp = (im->format == VK_FORMAT_S8_UINT ? 0u : VK_IMAGE_ASPECT_DEPTH_BIT) | (im->format == VK_FORMAT_D32_SFLOAT ? 0u : VK_IMAGE_ASPECT_STENCIL_BIT);
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = im->image;
        b.subresourceRange = {asp, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkClearDepthStencilValue cv{1.0f, 0};
        ts_label("clear: first use");
        vkCmdClearDepthStencilImage(rcb(), im->image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &b.subresourceRange);
        return;
    }
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im->image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkClearColorValue zero{};
    ts_label("clear: first use");
    if (!block_compressed(im->format)) vkCmdClearColorImage(rcb(), im->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &b.subresourceRange);  // BC images are filled by their upload
}

// Full memory barrier; `src`/`src_access` narrow the first scope (after a dispatch). With BB_BARRIER_OPT (default on) a barrier with nothing
// recorded since the last one is skipped: commands that need ordering are recorded through rcb() (or mark `fenced` themselves).
void Backend::barrier(VkPipelineStageFlags src, VkAccessFlags src_access) {
    end_pass();  // a barrier cannot sit inside a rendering instance
    if (bar_opt && fenced) { ++barriers_skipped; return; }
    pass_written.clear();
    pass_read.clear();
    fenced = true, other_since = false, crop_last = false;
    ++barriers;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = src_access;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    if (ts_every) ts_mark("barrier");  // closes the work before it
    vkCmdPipelineBarrier(cb, src, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    if (ts_every) ts_mark("other");
}

// ---- shaders ------------------------------------------------------------------------------------------------------------
ShaderEntry* Backend::translate_new(ShStage st, const RegView& r, uint64_t code_addr) {
    const bool ls = cur_tess_ls && st == ShStage::VS;  // the VS slot's LS variant reads the LS registers instead
    const uint32_t rsrc2 = r.sh[st == ShStage::PS ? kShPsRsrc2 : st == ShStage::VS ? (ls ? kShLsRsrc2 : kShVsRsrc2) : kShCsRsrc2];
    const uint32_t ud = st == ShStage::PS ? kShPsUser : st == ShStage::VS ? (ls ? kShLsUser : kShVsUser) : kShCsUser;
    GcnEnv env;
    env.stage = st;
    if (st == ShStage::VS) { env.tess_ls = cur_tess_ls; env.tess_ds_level = cur_tess_ds; }
    env.binding_base = st == ShStage::PS ? 1 : 0;
    env.push_offset = st == ShStage::PS ? push_fs() : 0;
    env.res_scale = res_scale();
    env.user_sgprs = (rsrc2 >> 1) & 0x1F;
    for (int i = 0; i < 16; ++i) env.user[i] = r.sh[ud + i];
    env.ps_input_addr = r.context[kSpiPsInputAddr];
    if (st == ShStage::PS) {
        env.ps_input_map = true;
        env.ps_bary = vk().bary;
        for (int k = 0; k < 32; ++k) env.ps_input_cntl[k] = r.context[kSpiPsInputCntl0 + k];
    }
    if (st == ShStage::VS && !ls) env.vs_out_cntl = r.context[kPaClVsOutCntl];
    env.cs_tgid_en = (rsrc2 >> 7) & 7;
    env.cs_tidig_comps = (rsrc2 >> 11) & 3;
    if (st == ShStage::CS) env.lds_bytes = ((rsrc2 >> 15) & 0x1FF) * 512;  // COMPUTE_PGM_RSRC2.LDS_SIZE, in units of 128 dwords
    for (int i = 0; i < 3; ++i) env.cs_local[i] = r.sh[kShCsThreadX + i];
    env.read_mem = read_guest;

    // shader length from the OrbShdr trailer
    std::vector<uint32_t> code(16384);
    if (!read_guest(code_addr, code.data(), 64)) return nullptr;
    // read progressively so we never touch unmapped memory
    uint32_t have = 64;
    gcn::ShaderInfo si;
    while (!gcn::shader_info(code.data(), have, si) && have < code.size()) {
        const uint32_t more = std::min<uint32_t>(256, uint32_t(code.size()) - have);
        if (!read_guest(code_addr + uint64_t(have) * 4, code.data() + have, more)) break;
        have += more;
    }
    auto* e = new ShaderEntry;
    e->stage = st;
    char nm[64];
    std::snprintf(nm, sizeof nm, "%s@%llx", st == ShStage::VS ? "vs" : st == ShStage::PS ? "ps" : "cs", (unsigned long long)code_addr);
    e->log_name = nm;
    if (!si.code_bytes) { e->tr.error = "no OrbShdr trailer"; return e; }
    if (si.code_bytes > uint64_t(have) * 4) { e->tr.error = "shader longer than the readable window"; return e; }
    e->code_hash = hash_bytes(code.data(), si.code_bytes & ~size_t(7));
    // Gnm's embedded helper compute shaders (identified by their code hash): their buffers alias render targets / depth
    // buffers that live in VkImages here, so they are executed on the images instead of on guest memory.
    if (e->code_hash == 0x32b79ba3d47cccb3ull || e->code_hash == 0xe236b42aa5cb87eeull) e->kind = 1;
    else if (e->code_hash == 0x158ef127fa456d83ull) e->kind = 2;
    else if (e->code_hash == 0xca710881d448644aull) e->kind = 3;
    e->tr = translate(code.data(), si.code_bytes / 4, env);
    if (!e->tr.error.empty()) return e;
    if (verbose && e->tr.ps_kill) log_once("kill" + e->log_name, "PS with pixel kill: " + e->log_name);
    e->fetch_code = e->tr.fetch_code;
    for (const Resource& rr : e->tr.resources) e->has_uav = e->has_uav || rr.type == Resource::StorageImage || (rr.type == Resource::Buffer && rr.written);
    if (const char* dir = std::getenv("BB_SPV_DUMP")) {  // debug: every translated module as <dir>/<name>[_ls|_dsN].spv, validate with bbspv
        char path[400];
        std::snprintf(path, sizeof path, "%s/%s%s%s.spv", dir, nm, cur_tess_ls && st == ShStage::VS ? "_ls" : "", cur_tess_ds && st == ShStage::VS ? ("_ds" + std::to_string(cur_tess_ds)).c_str() : "");
        if (FILE* f = std::fopen(path, "wb")) { std::fwrite(e->tr.spirv.data(), 4, e->tr.spirv.size(), f); std::fclose(f); }
    }

    VkCtx& c = vk();
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = e->tr.spirv.size() * 4;
    smi.pCode = e->tr.spirv.data();
    if (!vk_check(vkCreateShaderModule(c.device, &smi, nullptr, &e->module), "vkCreateShaderModule")) { e->tr.error = "vkCreateShaderModule failed"; return e; }
    if (verbose) std::fprintf(stderr, "gpu: translated %s (%zu SPIR-V words, %zu resources)\n", nm, e->tr.spirv.size(), e->tr.resources.size());
    return e;
}

void Backend::get_shader(ShStage st, const RegView& r, BoundShader& out) {
    const bool ls = cur_tess_ls && st == ShStage::VS;
    const uint32_t lo = st == ShStage::PS ? kShPsLo : st == ShStage::VS ? (ls ? kShLsLo : kShVsLo) : kShCsLo;
    const uint64_t addr = (uint64_t(r.sh[lo + 1]) << 40) | (uint64_t(r.sh[lo]) << 8);
    const uint32_t rsrc2 = r.sh[st == ShStage::PS ? kShPsRsrc2 : st == ShStage::VS ? (ls ? kShLsRsrc2 : kShVsRsrc2) : kShCsRsrc2];
    const uint32_t ud = st == ShStage::PS ? kShPsUser : st == ShStage::VS ? (ls ? kShLsUser : kShVsUser) : kShCsUser;
    // translation environment: stage, code, RSRC2, PS input enables, CS group size, LS / DS(level) variant of the VS slot, PS input routing
    const uint64_t tess = st == ShStage::VS && (cur_tess_ls || cur_tess_ds) ? (ls ? 1 : 2 + uint64_t(cur_tess_ds)) : 0;
    std::array<uint64_t, 4> key{addr | uint64_t(st) << 56, rsrc2 | uint64_t(st == ShStage::PS ? r.context[kSpiPsInputAddr] : 0) << 32,
                                st == ShStage::CS ? r.sh[kShCsThreadX] | uint64_t(r.sh[kShCsThreadX + 1]) << 32 : 0,
                                (st == ShStage::CS ? r.sh[kShCsThreadX + 2] : 0) | tess << 32};
    if (st == ShStage::VS && !ls) key[2] = r.context[kPaClVsOutCntl];  // what the POS1-3 exports mean (CS-only field above)
    if (st == ShStage::PS) {  // SPI_PS_INPUT_CNTL_n route PS inputs to VS params: part of the translation
        uint64_t h = 1469598103934665603ull;
        for (uint32_t k = 0, n = r.context[kSpiPsInControl] & 0x3F; k < n && k < 32; ++k) h = (h ^ r.context[kSpiPsInputCntl0 + k]) * 1099511628211ull;
        key[2] = h;  // CS-only field above
    }
    uint32_t user[16];
    for (int i = 0; i < 16; ++i) user[i] = r.sh[ud + i];
    out.e = nullptr;

    auto& list = shaders[key];
    for (auto& e : list) {
        if (!e->tr.error.empty()) {  // failed translations are sticky per key (descriptor shape retries cost too much), except:
            // a failed guest read (tr.read_failed; it is the last of tr.loads) retranslates on the first draw that reads it; an image slot
            // still empty on first use or a buffer format not yet set up retries every 128th use
            const std::string& er = e->tr.error;
            if (!e->kind && (e->tr.read_failed ? eval_resources(e->tr, user, rd_guest, out.words)
                                               : (er.rfind("image type", 0) == 0 || er.rfind("buffer format", 0) == 0) && ++e->fail_uses % 128 == 0)) {
                if (std::unique_ptr<ShaderEntry> n(translate_new(st, r, addr)); n) {
                    if (n->tr.error.empty()) log_once(n->log_name + "retry", std::string("shader ") + n->log_name + " now translates (it failed earlier: " + er + ")");
                    e = std::move(n);  // in place; the failed entry owns no GPU objects. Still failing: the new error (e.g. a later read) decides
                    if (e->tr.error.empty()) {
                        if (eval_resources(e->tr, user, rd_guest, out.words)) out.e = e.get();
                        return;
                    }
                }
            }
            out.e = e.get();
            out.words.clear();  // (may hold an earlier candidate's words)
            if (e->kind) eval_resources(e->tr, user, rd_guest, out.words);  // helper CS: still executable as an image op
            return;
        }
        if (!eval_resources(e->tr, user, rd_guest, out.words)) continue;
        bool match = true;
        for (size_t i = 0; i < out.words.size() && match; ++i)
            for (uint32_t k = 0; k < e->tr.resources[i].dwords; ++k)
                if ((out.words[i][k] ^ e->tr.resources[i].words[k]) & shape_mask(e->tr.resources[i].type, k)) { match = false; break; }
        if (match && e->tr.fetch_addr) {  // the draw's own fetch shader (from its user data) must be the inlined code (compared in place, as read_guest checks)
            const uint64_t fa = fetch_address(e->tr, user);
            const size_t fb = e->fetch_code.size() * 4;
            if (fa < 0x10000 || (fa & 3) || (hooks().mem_valid && !hooks().mem_valid(fa, fb)) || std::memcmp(reinterpret_cast<const void*>(fa), e->fetch_code.data(), fb) != 0) match = false;
            if (fa != e->tr.fetch_addr && fa != e->fetch_seen) {  // (log key built only when the address changes)
                e->fetch_seen = fa;
                char msg[160];
                std::snprintf(msg, sizeof msg, "fetch shader of %s at 0x%llx (translated with 0x%llx): %s", e->log_name.c_str(), (unsigned long long)fa,
                              (unsigned long long)e->tr.fetch_addr, match ? "same code" : "different code");
                log_once(std::string("fetch") + msg, msg);
            }
        }
        if (!match) continue;
        out.e = e.get();
        return;
    }
    ShaderEntry* e = translate_new(st, r, addr);
    if (!e) return;
    list.emplace_back(e);
    if (!e->tr.error.empty()) {
        log_once(e->log_name + "fail", std::string("shader ") + e->log_name + " not translated: " + e->tr.error);
        out.e = e;
        out.words.clear();
        return;
    }
    if (eval_resources(e->tr, user, rd_guest, out.words)) out.e = e;
}

// ---- resources ------------------------------------------------------------------------------------------------------------
VkSampler Backend::get_sampler(const uint32_t* w) {
    // (S# word 1: MIN_LOD [11:0], MAX_LOD [23:12] are part of the key: with mipped textures they decide the levels sampled; word 0
    // [11:9]: MAX_ANISO_RATIO)
    const uint64_t key = uint64_t((w[0] & 0x1FF) ^ ((w[2] >> 20) & 0xFF) << 9 ^ ((w[2] & 0x3FFF) << 17)) | uint64_t((w[0] >> 12) & 7) << 32 | uint64_t(w[1] & 0xFFFFFF) << 35 |
                         uint64_t((w[0] >> 9) & 7) << 59;
    auto it = samplers.find(key);
    if (it != samplers.end()) return it->second;
    auto addr = [](uint32_t m) {
        switch (m) {
            case 0: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            case 1: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            case 2: case 4: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            case 3: case 5: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
            default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        }
    };
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = ((w[2] >> 20) & 3) & 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.minFilter = ((w[2] >> 22) & 3) & 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    const uint32_t mip_filter = (w[2] >> 26) & 3;  // 0 none (base level only), 1 point, 2 linear
    si.mipmapMode = mip_filter == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = addr(w[0] & 7);
    si.addressModeV = addr((w[0] >> 3) & 7);
    si.addressModeW = addr((w[0] >> 6) & 7);
    si.minLod = mip_filter ? float(w[1] & 0xFFF) / 256.0f : 0.0f;
    si.maxLod = mip_filter ? float((w[1] >> 12) & 0xFFF) / 256.0f : 0.25f;  // (none: NEAREST rounds to the base level, minification stays)
    int bias = int(w[2] & 0x3FFF);
    if (bias & 0x2000) bias -= 0x4000;
    si.mipLodBias = float(bias) / 256.0f;
    si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    if (const uint32_t cmp = (w[0] >> 12) & 7) { si.compareEnable = VK_TRUE; si.compareOp = VkCompareOp(cmp); }  // S# depth compare function = Vulkan order; only image_sample_c* may use such a sampler
    // Anisotropy: MAX_ANISO_RATIO (1, 2, 4, 8, 16) applies with an aniso XY filter (2 point, 3 linear). BB_ANISO=<1..16>: that ratio for
    // every linearly minifying sampler instead (quality override). Capped by the device limit.
    static const uint32_t aniso_env = std::getenv("BB_ANISO") ? std::clamp(std::atoi(std::getenv("BB_ANISO")), 1, 16) : 0;
    float ratio = std::max((w[2] >> 20) & 3, (w[2] >> 22) & 3) >= 2 ? float(1u << std::min((w[0] >> 9) & 7, 4u)) : 1.0f;
    if (aniso_env && si.minFilter == VK_FILTER_LINEAR) ratio = float(aniso_env);
    if (vk().aniso && ratio > 1.0f) si.anisotropyEnable = VK_TRUE, si.maxAnisotropy = std::min(ratio, vk().props.limits.maxSamplerAnisotropy);
    VkSampler s = VK_NULL_HANDLE;
    vkCreateSampler(vk().device, &si, nullptr, &s);
    samplers[key] = s;
    return s;
}

// Colour target; layers > 1 = an array / cube target drawn one slice at a time (CB_COLOR_VIEW), see attach_view. w, h: guest size.
// Screen-sized single-layer targets are created at res_scale() times that size; `min_scale` (promotion, see draw) forces a scale.
Image* Backend::get_rt(uint64_t base, uint32_t w, uint32_t h, VkFormat fmt, uint32_t layers, uint32_t min_scale) {
    const uint32_t want = std::max(min_scale, target_scale(base, w, h, layers));
    auto it = rts.find(base);
    if (it != rts.end() && it->second->gw() == w && it->second->gh() == h && it->second->format == fmt && it->second->layers >= layers && it->second->scale >= want) return it->second.get();
    if (it != rts.end()) {
        graveyard_pending.push_back(it->second.release());
        rts.erase(it);
    }
    // (MUTABLE: a T# may read the target in another number format, e.g. the blood maps' mip copy loads an sRGB target as UINT)
    Image* im = make_image(w * want, h * want, layers, fmt, layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           layers >= 6 && layers % 6 == 0 && w == h, 1, true);
    if (!im) return nullptr;
    im->scale = want;
    im->base = base;
    if (verbose) { char b2[128]; std::snprintf(b2, sizeof b2, "render target 0x%llx %ux%u x%u format %d (cube %d, scale %u)", (unsigned long long)base, w, h, layers, int(fmt), int(im->cube_ok), want); log_once(b2, b2); }
    if (std::getenv("BB_TINY_LOG") && w <= 8 && h <= 8) std::fprintf(stderr, "tiny: RT created base=%llx %ux%u fmt=%d draw=%llu\n", (unsigned long long)base, w, h, int(fmt), (unsigned long long)draws);
    rts[base].reset(im);
    ++tex_gen;  // texture lookups at this address now resolve to the target (tex_memo)
    return im;
}

// Attachment view of one slice of a colour target (the whole image when it has a single layer).
VkImageView Backend::attach_view(Image* im, uint32_t layer) {
    if (im->layers <= 1) return im->view;
    layer = std::min(layer, im->layers - 1);
    if (im->layer_views.size() < im->layers) im->layer_views.resize(im->layers, VK_NULL_HANDLE);
    VkImageView& v = im->layer_views[layer];
    if (!v) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        VkImageViewUsageCreateInfo vu{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
        vu.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;  // an attachment view of an sRGB storage image (see make_image)
        if (im->srgb_storage) vci.pNext = &vu;
        vci.image = im->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = im->format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, layer, 1};
        vkCreateImageView(vk().device, &vci, nullptr, &v);
    }
    return v;
}

// Depth/stencil target at a DB_Z_WRITE_BASE / DB_STENCIL_WRITE_BASE address (stencil-only: the stencil base). Tiling is internal to the GPU: never read back.
// w, h: guest size; scaled like get_rt.
Image* Backend::get_ds(uint64_t base, uint32_t w, uint32_t h, VkFormat fmt, uint32_t min_scale) {
    const uint32_t want = std::max(min_scale, target_scale(base, w, h, 1));
    auto it = dss.find(base);
    if (it != dss.end() && it->second->gw() == w && it->second->gh() == h && it->second->format == fmt && it->second->scale >= want) return it->second.get();
    if (it != dss.end()) {
        if (verbose) {
            static uint64_t n = 0;
            if (++n <= 40) std::fprintf(stderr, "gpu: depth buffer 0x%llx RECREATED %ux%u fmt %d -> %ux%u fmt %d scale %u (contents lost)\n", (unsigned long long)base, it->second->gw(), it->second->gh(), int(it->second->format), w, h, int(fmt), want);
        }
        graveyard_pending.push_back(it->second.release());
        dss.erase(it);
    }
    VkCtx& c = vk();
    auto* im = new Image;
    im->w = w * want; im->h = h * want; im->scale = want; im->format = fmt; im->ds = true; im->base = base;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = {im->w, im->h, 1};
    ici.mipLevels = ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (!vk_check(vmaCreateImage(c.vma, &ici, &aci, &im->image, &im->alloc, nullptr), "depth image")) { delete im; return nullptr; }
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = im->image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = {(fmt == VK_FORMAT_S8_UINT ? 0u : VK_IMAGE_ASPECT_DEPTH_BIT) | (fmt == VK_FORMAT_D32_SFLOAT ? 0u : VK_IMAGE_ASPECT_STENCIL_BIT), 0, 1, 0, 1};
    if (!vk_check(vkCreateImageView(c.device, &vci, nullptr, &im->view), "depth view")) { vmaDestroyImage(c.vma, im->image, im->alloc); delete im; return nullptr; }
    dss[base].reset(im);
    ++tex_gen;
    if (verbose) { char b2[160]; std::snprintf(b2, sizeof b2, "depth buffer created: base=0x%llx %ux%u fmt=%d scale %u", (unsigned long long)base, w, h, int(fmt), want); log_once(b2, b2); }
    return im;
}

// Linear, 1D/2D thin, display and 1D thick surfaces with their mip levels. Other tiling modes are not implemented yet: the texture stays empty.
Image* Backend::get_texture(const uint32_t* w, bool storage, uint32_t pad_axes) {
    tex_memo_ok = false, tex_memo_src = nullptr;
    const uint64_t base = (uint64_t(w[0]) | (uint64_t(w[1] & 0x3F) << 32)) << 8;  // BASE_ADDRESS is 38 bits; the game sets word 1 bit 6 on some T#s
    const uint32_t dfmt = (w[1] >> 20) & 0x3F, nfmt = (w[1] >> 26) & 0xF;
    const uint32_t width = (w[2] & 0x3FFF) + 1, height = ((w[2] >> 14) & 0x3FFF) + 1;
    const uint32_t type = w[3] >> 28, tile = (w[3] >> 20) & 31;
    const uint32_t depth = (w[4] & 0x1FFF) + 1;
    if (static const uint64_t tl = std::getenv("BB_TINY_LOG") ? std::strtoull(std::getenv("BB_TINY_LOG"), nullptr, 10) + 1 : 0; tl && draws + 1 >= tl && width <= 4 && height <= 4) {  // debug: BB_TINY_LOG=<from draw>: where 1x1..4x4 lookups (exposure etc.) resolve
        static std::set<uint64_t> seen;
        if (seen.size() < 300 && seen.insert(base ^ (uint64_t(storage) << 63)).second) {
            auto ri = rts.find(base);
            uint32_t first = 0;
            if (hooks().mem_valid && hooks().mem_valid(base, 4)) std::memcpy(&first, reinterpret_cast<const void*>(base), 4);
            std::fprintf(stderr, "tiny: %s base=%llx %ux%u dfmt=%u nfmt=%u type=%u tile=%u rt=%s mem[0]=%08x draw=%llu\n", storage ? "STORE" : "sample", (unsigned long long)base, width, height, dfmt, nfmt, type, tile,
                         ri == rts.end() ? "no" : "yes", first, (unsigned long long)draws);
        }
    }
    // Alias render targets that live at this address
    auto rt = rts.find(base);
    if (rt != rts.end() && rt->second->gw() >= width && rt->second->gh() >= height) {
        Image* src = rt->second.get();
        if (storage || (src->gw() == width && src->gh() == height)) { tex_memo_ok = true; return src; }
        // A padded target (a 1920x1088 colour target holding a 1920x1080 picture, 1024x576 / 960x540): the shader scales normalised
        // coordinates by T# size / image size and clamps to the last T# texel (gcn_spirv `pad`), exact for clamp-to-edge axes
        // (`pad_axes` bit 0 = x, bit 1 = y, from the S# the image is sampled with). Otherwise (wrap/mirror/border, several samplers,
        // layered or cube T#s, BB_PAD_DIRECT=0): a cropped copy of exactly the T# size, re-taken when the target changed. (Shrinking
        // the target to the sampled size broke the image: walls replaced by sky.) A layered target (e.g. the 6 faces of a 64x64 cube
        // map rendered into a padded 128x64 target, which cannot be CUBE_COMPATIBLE because it is not square) keeps all its layers in
        // the crop, and a cube T# gets a cube-compatible crop (validation VUID 07752: a 2D crop of layer 0 lost five faces).
        static const bool pad_direct = !(std::getenv("BB_PAD_DIRECT") && std::getenv("BB_PAD_DIRECT")[0] == '0');
        if (pad_direct && type == 9 && src->layers == 1 && (src->gw() == width || (pad_axes & 1)) && (src->gh() == height || (pad_axes & 2))) {
            tex_memo_ok = true;
            ++pad_direct_binds;
            return src;
        }
        const uint32_t cl = src->layers, sc = src->scale;  // (a crop of a scaled target is scaled alike)
        const bool cube_crop = type == 11 && cl % 6 == 0 && width == height;
        const std::array<uint64_t, 3> ck{base | 1ull << 63, width | uint64_t(height) << 16, uint64_t(cl) | uint64_t(cube_crop) << 32 | uint64_t(sc) << 40};
        auto cit = tex_lut.find(ck);
        Image* ci = cit != tex_lut.end() ? cit->second : nullptr;
        if (!ci) {
            ci = make_image(width * sc, height * sc, cl, src->format, cl > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, cube_crop);
            if (!ci) return src;
            ci->bgra = src->bgra;
            ci->scale = sc;
            texs[ck].reset(ci);
            tex_lut[ck] = ci;
        }
        if (ci->crop_of != src->version) {
            ensure_init(ci);
            ensure_init(src);
            note_read(src);
            note_write(ci);
            end_pass();
            if (!crop_last) barrier();  // crops back to back read targets the last barrier already ordered
            VkImageCopy cp{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, cl}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, cl}, {0, 0, 0}, {width * sc, height * sc, 1}};
            if (ts_every) ts_mark("copy: padded-target crop " + std::to_string(src->w) + "x" + std::to_string(src->h) + "->" + std::to_string(width) + "x" + std::to_string(height) + " f" + std::to_string(int(src->format)));
            vkCmdCopyImage(rcb(), src->image, VK_IMAGE_LAYOUT_GENERAL, ci->image, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
            // Its barrier waits for the user (a new rendering instance after other commands, or the barrier before a dispatch): the crops
            // of one draw share one barrier.
            if (bar_opt) crop_last = true;
            else barrier();
            ci->crop_of = src->version;
            ++crop_copies;
            crop_bytes += uint64_t(ci->w) * ci->h * cl * (src->format == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 4);
        }
        tex_memo_ok = true, tex_memo_src = src;  // (while crop_of == src->version)
        return ci;
    }
    auto dsit = dss.find(base);  // a depth buffer sampled as a texture (shadow maps etc.)
    if (dsit != dss.end() && dsit->second->format != VK_FORMAT_S8_UINT && dsit->second->gw() >= width && dsit->second->gh() >= height) { ensure_init(dsit->second.get()); tex_memo_ok = true; return dsit->second.get(); }

    // Sampled and storage uses of the same memory must be one VkImage (a compute pass writes what a later pixel shader samples), whatever
    // number format each T# names: the blood maps of the characters are written through a UINT storage T# per mip level and sampled
    // as sRGB with the full chain; separate images left the sampled one with the stale guest bytes (black or rainbow fur, ROADMAP 97).
    // Uncompressed textures are therefore keyed without the number format, created MUTABLE, and viewed in the T#'s format (texture_view).
    // Slices: T# DEPTH or LAST_ARRAY (word 5 [25:13]) + 1, whichever is larger; cube maps often leave DEPTH 0 and name the faces only
    // through LAST_ARRAY, and a cube view needs whole sets of 6 faces. The slice count is part of the image identity (key): a 36-layer
    // cube array at the address of a 6-layer cube is another image (validation VUID 07968: upload past the existing image's layers).
    const bool arrayed = type == 12 || type == 13 || type == 11 || type == 10;
    const uint32_t last_array = (w[5] >> 13) & 0x1FFF;
    const uint32_t layers = !arrayed ? 1 : type == 10 ? depth : type == 11 ? (std::max(depth, last_array + 1) + 5) / 6 * 6 : std::max(depth, last_array + 1);
    TexFmt tf = tex_format(dfmt, nfmt);
    const std::array<uint64_t, 3> key{base, width | uint64_t(height) << 16 | uint64_t(layers) << 32, dfmt | (tf.block == 1 ? 0 : nfmt) << 8 | type << 16 | tile << 24};
    auto kstr = [&] { char b[96]; std::snprintf(b, sizeof b, "%llx:%u:%u:%u:%u:%u:%u:%u", (unsigned long long)base, width, height, layers, dfmt, nfmt, type, tile); return std::string(b); };  // log keys
    VkImageViewType vt = type == 8 ? VK_IMAGE_VIEW_TYPE_1D : type == 12 ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : type == 13 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                       : type == 10 ? VK_IMAGE_VIEW_TYPE_3D : type == 11 ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
    if (tf.vk == VK_FORMAT_UNDEFINED || type < 8 || type > 13) {
        log_once("tf" + kstr(), "texture format dfmt=" + std::to_string(dfmt) + " nfmt=" + std::to_string(nfmt) + " type=" + std::to_string(type) + " not supported yet");
        return nullptr;
    }
    auto it = tex_lut.find(key);
    Image* im = it != tex_lut.end() ? it->second : nullptr;
    if (!im) {
        // a mipped T# (BASE_LEVEL [15:12] or LAST_LEVEL [19:16] of word 3 above 0) gets the full chain: storage writes to level n (mip
        // generation) and sampling of the chain must meet in one image. A storage T# gets it too: the mip generation's level-0 store
        // (BASE = LAST = 0) comes first and creates the image. ponytail: an image first created by a sampled level-0-only T# stays
        // one level (recreate on growth if a game writes mips into such a texture); volume textures keep level 0 only (the clinic's two
        // 16x16x16 volumes have no mips; logged).
        const uint32_t ih = vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_1D_ARRAY ? 1 : height;
        const uint32_t levels = type != 10 && ((w[3] >> 12 & 0xFF) || (storage && tf.block == 1)) ? uint32_t(std::bit_width(std::max(width, ih))) : 1;
        im = make_image(width, ih, layers, tf.vk, vt,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | (tf.block == 1 ? VK_IMAGE_USAGE_STORAGE_BIT : 0), type == 11, levels, tf.block == 1);
        if (!im) return nullptr;
        im->base = base;
        ++tex_texture_images, tex_mipped += levels > 1;
        texs[key].reset(im);
        tex_lut[key] = im;
        { char mk[96]; std::snprintf(mk, sizeof mk, "tile%u dfmt%u nfmt%u type%u", tile, dfmt, nfmt, type); ++tex_mix[mk]; }  // BB_GPU_LOG: which tilings/formats the game really uses
    }
    // supported tilings: linear (aligned/general) and 1D thin (8x8 micro tiles, raster order of tiles)
    // tile index -> array mode per PS4 table: 8 LinearAligned, 9 Display1DThin, 13 Thin1DThin, 31 LinearGeneral
    const bool linear = tile == 8 || tile == 31, thin1d = tile == 9 || tile == 13;
    const uint32_t bpp = tf.bytes * 8;
    tiling::Macro mt;
    const bool thick = tile == 19;  // Thick1DThick: volume textures
    const bool macro = !linear && !thin1d && !thick && tiling::macro_mode(tile, bpp, mt);
    if (!linear && !thin1d && !thick && !macro) {
        {
            std::string extra = " base=0x" + [&] { char b2[32]; std::snprintf(b2, sizeof b2, "%llx", (unsigned long long)base); return std::string(b2); }() + " " + std::to_string(width) + "x" + std::to_string(height) +
                                " dfmt=" + std::to_string(dfmt) + " nfmt=" + std::to_string(nfmt) + " known depth buffers:";
            for (const auto& [a, d2] : dss) { char b2[64]; std::snprintf(b2, sizeof b2, " 0x%llx(%ux%u)", (unsigned long long)a, d2->w, d2->h); extra += b2; }
            log_once("tmx" + kstr(), "texture tiling index " + std::to_string(tile) + extra);
        }
        ++tex_unsup;
        log_once("tm" + kstr(), "texture tiling index " + std::to_string(tile) + " not implemented; texture left empty");
        ensure_init(im);
        return im;
    }
    // Mip levels (layout: tiling::mip_chain) from the T# pitch (word 4 [26:13], texels) and POW2_PAD (word 3 bit 25)
    const uint32_t eb = tf.bytes;
    const uint32_t pitch = std::max<uint32_t>(((w[4] >> 13) & 0x3FFF) + 1, width);
    const bool v1d = vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    const uint32_t base_level = (w[3] >> 12) & 15, want = std::min(std::max((w[3] >> 16) & 15, base_level) + 1, im->levels);
    const uint32_t old_mips = im->mips, mips = std::max(old_mips, want);  // levels read from guest memory
    tiling::Level lv[16];
    size_t src_bytes = tiling::mip_chain(lv, mips, width, v1d ? 1 : height, pitch, layers, tf.block, eb,
                                         linear ? tiling::kLinear : thin1d ? tiling::k1D : macro ? tiling::k2D : tiling::kThick,
                                         tile == 9 ? tiling::kDisplay : tiling::kThin, tile == 8, (w[3] >> 25) & 1, &mt, type == 11);
    if (lv[mips - 1].mode == tiling::kLinear) src_bytes -= size_t(lv[mips - 1].pe - lv[mips - 1].ew) * eb;  // the last row needs only its used part
    if (hooks().mem_valid && !hooks().mem_valid(base, src_bytes)) {
        char t3[200];
        std::snprintf(t3, sizeof t3, "texture memory not mapped: base=0x%llx %ux%u x%u levels %u dfmt=%u tile=%u bytes=0x%zx (first byte mapped: %d)", (unsigned long long)base, width, height, layers, mips, dfmt, tile, src_bytes, int(hooks().mem_valid(base, 1)));
        log_once("tv" + kstr(), t3);
        return im;
    }
    const uint8_t* src = reinterpret_cast<const uint8_t*>(base);
    settle_pending(base, base + src_bytes, 0);
    tex_memo_ok = true;  // (memoizable from here on within the epoch, see tex_memo)
    // more levels than read so far: the new ones are uploaded at once (the old ones too if their bytes changed)
    const bool grow = im->initialised && mips > old_mips;
    const size_t old_bytes = im->bytes;
    im->mips = mips;
    im->bytes = src_bytes;
    tex_max_bytes = std::max<uint64_t>(tex_max_bytes, src_bytes);
    // Change detection, once per epoch (g_tex_epoch, tex_touch): where the write watch tracks the bytes (Windows, write-watched memory;
    // BB_TEX_WW=0 off) unwritten pages mean an unchanged texture, written ones are hashed in full (a rewrite with the same bytes keeps
    // the image). Elsewhere a cheap sample of three 4 KB spans; the full hash runs when the sample changes and every kFullEvery frames
    // (ponytail: an update confined to the unsampled bytes shows up after <= kFullEvery frames). The polls (~0.2 GB of texture pages per
    // frame in the clinic) cost less than the sample hashes they replace, and a clean texture carries its tex_memo entry over.
    // Debug BB_WW_VERIFY: for every 64th "unchanged" verdict below, do guest bytes still hash to what was uploaded (im->hash)? Stale = a
    // write the epochs / tex_touch / the write watch miss, or the sample's blind spot; the first few are logged.
    static const bool tverify = std::getenv("BB_WW_VERIFY") != nullptr;
    auto stale = [&](uint64_t& count, const char* why) {
        if (!tverify || (++tex_verify_tick & 63) != 0) return;
        ++tex_verified;
        if (hash_fast(src, src_bytes & ~size_t(7)) == im->hash) return;
        const char* when = im->ww == 1 ? (ww_repoll(base, base + src_bytes, im->ww_epoch) == 0 ? ", write watch: late write" : ", write watch: blind") : "";
        if (count++ < 8) std::fprintf(stderr, "tex-verify: stale (%s%s) base=0x%llx bytes=%zu %ux%u x%u dfmt=%u nfmt=%u tile=%u frame=%llu\n", why, when, (unsigned long long)base, src_bytes, width, height, layers, dfmt, nfmt, tile, (unsigned long long)g_frame_counter.load());
    };
    const uint64_t epoch = g_tex_epoch.load(std::memory_order_relaxed);
    if (im->initialised && !grow && im->hash_epoch == epoch) { stale(tex_stale_epoch, "same epoch"); return im; }
    im->hash_epoch = epoch;
    constexpr uint64_t kFullEvery = 64;  // frames
    const size_t n8 = src_bytes & ~size_t(7), span = std::min<size_t>(4096, n8 / 3 & ~size_t(7));
    const uint64_t frame = g_frame_counter.load(std::memory_order_relaxed);
    static const bool tex_ww = !(std::getenv("BB_TEX_WW") && std::getenv("BB_TEX_WW")[0] == '0');
    uint64_t ww_now = 0;
    int ww = -1;  // 1 clean since im->ww_epoch, 0 written, -1 untracked
    if (tex_ww && im->ww >= 0) {  // (polled before the bytes are read; a first upload or a grown chain only arms it)
        ww = ww_clean_since(base, base + src_bytes, im->initialised && !grow ? im->ww_epoch : 0, ww_now);
        im->ww = int8_t(ww < 0 ? -1 : 1);
        ++tex_ww_checks;
        if (ww == 1 && im->initialised && !grow) { im->ww_epoch = ww_now; stale(tex_stale_ww, "write watch clean"); return im; }
    }
    // (without the write watch, a texture whose bytes changed after its first upload is hashed in full every epoch: the sample missed
    // the updates of a CPU-written 1024x1024 R8 texture for up to kFullEvery frames, BB_WW_VERIFY)
    if (ww < 0 && !im->dynamic) {
        const uint64_t q = hash_fast(src, span) ^ hash_fast(src + (n8 - span) / 2 / 8 * 8, span) * 3 ^ hash_fast(src + n8 - span, span) * 5;
        tex_hash_bytes += 3 * span, ++tex_checks;
        // each image at its own phase of the period: images first hashed in the same frame stayed in step (564 full hashes in one frame
        // every 64 frames, +10-20 ms each time)
        const uint64_t phase = (base >> 8) * 0x9E3779B97F4A7C15ull >> 58;
        const bool full_due = (frame + phase) / kFullEvery != (im->full_epoch + phase) / kFullEvery;
        if (im->initialised && !grow && q == im->qhash && !full_due) { stale(tex_stale_sample, "sample matched"); return im; }
        im->qhash = q;
    }
    const uint64_t h = hash_fast(src, n8);
    tex_hash_bytes += n8, ++tex_full_hashes;
    im->full_epoch = frame;
    if (ww >= 0) im->ww_epoch = ww_now;
    if (im->initialised && !grow && h == im->hash) return im;
    const uint32_t lo = grow && hash_fast(src, old_bytes & ~size_t(7)) == im->hash ? old_mips : 0;  // (levels below `lo` are current)
    if (im->initialised && !im->dynamic && lo == 0) im->dynamic = true, ++tex_dynamic;  // (changed after the first upload: see the sample above)
    im->hash = h;
    if (static int up = 0; gpu_watched(base, src_bytes) && up++ < 40)  // BB_GPU_WATCH: uploads of textures over the range
        std::fprintf(stderr, "gpu-watch: upload %s levels %u..%u (base level %u) bytes 0x%zx frame=%llu draw=%llu\n", kstr().c_str(), lo, mips - 1, base_level, src_bytes, (unsigned long long)frame, (unsigned long long)draws);

    size_t out_off[16], out_bytes = 0;
    for (uint32_t n = lo; n < mips; ++n) out_off[n] = out_bytes, out_bytes += (size_t(lv[n].ew) * lv[n].eh * eb * layers + 15) & ~size_t(15);
    size_t off = 0;
    uint8_t* stage_ptr = ring_alloc(out_bytes, 16, off);
    tex_bytes += out_bytes;
    ++tex_uploads;
    if (!stage_ptr) { tex_memo_ok = false; return im; }
    // de-tile each level into tightly packed rows: linear rows at the pitch (ponytail: slices of linear arrays assumed packed at
    // pitch*height), 2D per element through the pipe/bank interleave, 1D thin/display and thick micro tiles in raster order of tiles
    // (thick: 8x8x4 elements, slabs of 4 slices)
    for (uint32_t n = lo; n < mips; ++n) {
        const tiling::Level& v = lv[n];
        const uint8_t* s = src + v.off;
        const size_t row = size_t(v.ew) * eb, slice_dst = row * v.eh;
        for (uint32_t l = 0; l < layers; ++l)
            for (uint32_t y = 0; y < v.eh; ++y) {
                uint8_t* d = stage_ptr + out_off[n] + l * slice_dst + y * row;
                if (v.mode == tiling::kLinear) { std::memcpy(d, s + l * v.slice + size_t(y) * v.pe * eb, row); continue; }
                for (uint32_t x = 0; x < v.ew; ++x) {
                    const size_t so = v.mode == tiling::k2D ? size_t(tiling::macro_addr(mt, bpp, x, y, l, v.pe, v.he))
                                    : v.mode == tiling::kThick ? (l >> 2) * v.slice * 4 + (size_t(y >> 3) * (v.pe / 8) + (x >> 3)) * 256 * eb + size_t(tiling::micro_pixel_index_thick(bpp, x & 7, y & 7, l & 3)) * eb
                                                  : l * v.slice + (size_t(y >> 3) * (v.pe / 8) + (x >> 3)) * 64 * eb + size_t(tiling::micro_pixel_index(v.micro, bpp, x & 7, y & 7)) * eb;
                    std::memcpy(d + size_t(x) * eb, s + so, eb);
                }
            }
    }
    // BB_TEX_DUMP=<dir>: write every uploaded (de-tiled) texture as <dir>/t_<base>_<w>x<h>_d<dfmt>n<nfmt>_t<tile>_<layers>.bin (raw texels/blocks,
    // the uploaded levels one after the other, each padded to 16 bytes) and the guest bytes behind it as .raw (twice the computed size
    // where mapped, to check mip layouts). With BB_GPU_WATCH only the textures over the watched range (latest upload wins).
    static const char* tex_dump = std::getenv("BB_TEX_DUMP");
    if (tex_dump && (!std::getenv("BB_GPU_WATCH") || gpu_watched(base, src_bytes))) {
        char path[400];
        std::snprintf(path, sizeof path, "%s/t_%llx_%ux%u_d%un%u_t%u_%u.bin", tex_dump, (unsigned long long)base, width, height, dfmt, nfmt, tile, layers);
        if (FILE* f = std::fopen(path, "wb")) { std::fwrite(stage_ptr, 1, out_bytes, f); std::fclose(f); }
        std::strcpy(path + std::strlen(path) - 3, "raw");
        const size_t raw = hooks().mem_valid && hooks().mem_valid(base, src_bytes * 2) ? src_bytes * 2 : src_bytes;
        if (FILE* f = std::fopen(path, "wb")) { std::fwrite(src, 1, raw, f); std::fclose(f); }
    }
    ensure_init(im);
    VkBufferImageCopy copy[16]{};
    for (uint32_t n = lo; n < mips; ++n) {
        VkBufferImageCopy& c = copy[n - lo];
        c.bufferOffset = off + out_off[n];
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, n, 0, vt == VK_IMAGE_VIEW_TYPE_3D ? 1u : layers};
        c.imageExtent = {lv[n].w, lv[n].h, vt == VK_IMAGE_VIEW_TYPE_3D ? layers : 1u};
    }
    barrier();
    vkCmdCopyBufferToImage(rcb(), ring.buffer, im->image, VK_IMAGE_LAYOUT_GENERAL, mips - lo, copy);
    barrier();
    return im;
}

VkImageView Backend::texture_view(Image* im, const uint32_t* w, bool swizzle) {
    // component mapping from dst_sel (0 zero, 1 one, 4..7 = memory component X..W) onto the Vulkan format's channels;
    // storage views must keep the identity mapping
    const uint32_t sel[4] = {w[3] & 7, (w[3] >> 3) & 7, (w[3] >> 6) & 7, (w[3] >> 9) & 7};
    VkComponentSwizzle map[4] = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    for (int k = 0; k < 4 && swizzle; ++k) {
        switch (sel[k]) {
            case 0: map[k] = VK_COMPONENT_SWIZZLE_ZERO; break;
            case 1: map[k] = VK_COMPONENT_SWIZZLE_ONE; break;
            default: {
                const int m = int(sel[k]) - 4;  // memory component 0..3
                static const VkComponentSwizzle rgba[4] = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
                static const VkComponentSwizzle bgra[4] = {VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_A};
                map[k] = (m >= 0 && m < 4) ? (im->format == VK_FORMAT_B8G8R8A8_UNORM || im->format == VK_FORMAT_B8G8R8A8_SRGB ? bgra[m] : rgba[m]) : VK_COMPONENT_SWIZZLE_IDENTITY;
            }
        }
    }
    // layered render targets are 2D arrays; the descriptor decides how they are sampled (cube for the shadow cube, plain 2D for one slice)
    VkImageViewType vt = im->view_type;
    if (!im->ds && im->layers > 1 && im->view_type == VK_IMAGE_VIEW_TYPE_2D_ARRAY) {
        const uint32_t tt = w[3] >> 28;
        if (tt == 11 && im->cube_ok) vt = VK_IMAGE_VIEW_TYPE_CUBE;
        else if (tt == 9) vt = VK_IMAGE_VIEW_TYPE_2D;
    }
    {  // the shader's image type comes from the T# (translation): array-ness must match (validation VUID 07752)
        const uint32_t tt = w[3] >> 28;
        if (tt == 13 && vt == VK_IMAGE_VIEW_TYPE_2D) vt = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        else if (tt == 9 && vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY) vt = VK_IMAGE_VIEW_TYPE_2D;
        else if (tt == 12 && vt == VK_IMAGE_VIEW_TYPE_1D) vt = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
        else if (tt == 8 && vt == VK_IMAGE_VIEW_TYPE_1D_ARRAY) vt = VK_IMAGE_VIEW_TYPE_1D;
    }
    // array range of the descriptor (T# word 5: base_array [12:0], last_array [25:13]); non-array images use layer 0
    const bool arrayed = vt == VK_IMAGE_VIEW_TYPE_1D_ARRAY || vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY || vt == VK_IMAGE_VIEW_TYPE_CUBE || (vt == VK_IMAGE_VIEW_TYPE_2D && im->layers > 1);
    uint32_t base_layer = 0, layer_count = vt == VK_IMAGE_VIEW_TYPE_3D ? 1 : im->layers;
    if (arrayed) {
        base_layer = std::min(w[5] & 0x1FFF, im->layers - 1);
        const uint32_t last = std::min((w[5] >> 13) & 0x1FFF, im->layers - 1);
        layer_count = last >= base_layer ? last - base_layer + 1 : 1;
        // one cube (the shader declares a non-arrayed cube; VUID 02960): the 6 faces holding BASE_ARRAY, of a cube array too
        if (vt == VK_IMAGE_VIEW_TYPE_CUBE) { base_layer = std::min(base_layer / 6 * 6, im->layers - 6); layer_count = 6; }
        if (vt == VK_IMAGE_VIEW_TYPE_2D) layer_count = 1;
    }
    // mip range BASE_LEVEL..LAST_LEVEL (word 3 [15:12] / [19:16]) within the image's levels; a storage view holds one level
    const uint32_t base_level = std::min((w[3] >> 12) & 15, im->levels - 1);
    const uint32_t level_count = swizzle ? std::min(std::max((w[3] >> 16) & 15, base_level), im->levels - 1) - base_level + 1 : 1;
    // view format: on a mutable image (one texture for every number format, see get_texture; render targets) the T#'s number format
    // within the image format's family (same dfmt: e.g. an sRGB target loaded as UINT), else the image's; storage views cannot be sRGB
    // (the UNORM twin: the store writes the raw bits either way)
    VkFormat vf = im->format;
    if (im->mutable_fmt) {
        const uint32_t dfmt = (w[1] >> 20) & 0x3F;
        const VkFormat tv = tex_format(dfmt, (w[1] >> 26) & 0xF).vk;
        for (const uint32_t n : {0u, 1u, 4u, 5u, 7u, 9u})
            if (tv != VK_FORMAT_UNDEFINED && tex_format(dfmt, n).vk == im->format) { vf = tv; break; }
        if (!swizzle) vf = unorm_twin(vf);
    }
    const std::array<uint64_t, 3> key{reinterpret_cast<uintptr_t>(im->image),
                                      uint64_t(map[0]) | uint64_t(map[1]) << 4 | uint64_t(map[2]) << 8 | uint64_t(map[3]) << 12 | uint64_t(base_layer) << 16 | uint64_t(layer_count) << 32 | uint64_t(vt) << 48 |
                                      uint64_t(base_level) << 52 | uint64_t(level_count) << 56, uint64_t(vf)};
    auto it = views.find(key);
    if (it != views.end()) return it->second;
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    VkImageViewUsageCreateInfo vu{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    vu.usage = swizzle ? VK_IMAGE_USAGE_SAMPLED_BIT : VK_IMAGE_USAGE_STORAGE_BIT;
    if (im->srgb_storage || vf != im->format) vci.pNext = &vu;  // e.g. an sRGB view must not carry the image's storage usage
    vci.image = im->image;
    vci.viewType = vt;
    vci.format = vf;
    vci.components = {map[0], map[1], map[2], map[3]};
    vci.subresourceRange = {im->ds ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT) : VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT), base_level, level_count, base_layer, layer_count};
    VkImageView v = VK_NULL_HANDLE;
    vkCreateImageView(vk().device, &vci, nullptr, &v);
    views[key] = v;
    return v;
}

bool Backend::write_descriptors(const BoundShader& bs, DescInfo* info) {
    ScopeNs wdt{wd_ns};
    const auto& res = bs.e->tr.resources;
    for (size_t i = 0; i < res.size(); ++i) {
        const uint32_t* w = bs.words[i].data();
        switch (res[i].type) {
            case Resource::Buffer: {
                ScopeNs bt{wd_buf_ns};
                uint64_t base = uint64_t(w[0]) | (uint64_t(w[1] & 0xFFF) << 32);  // 44-bit base address
                const uint32_t stride = (w[1] >> 16) & 0x3FFF;
                uint64_t size = stride ? uint64_t(w[2]) * stride : w[2];
                size = std::min<uint64_t>(size, uint64_t(64) << 20);
                // Vertex streams fetched by vertex id alone: only the records this draw can fetch (the V# often spans the whole vertex pool).
                // Everything else is bound whole: instance data and bone palettes are addressed by computed indices (e.g. instance * 13 + k,
                // (bone + base) * 3 + k), beyond the vertex window. Constant buffers are read whole.
                if (bs.e->stage == ShStage::VS && stride && !res[i].scalar && res[i].vertex_use == 1 && cur_vtx_records != ~0ull) {
                    const uint64_t skip = cur_vtx_first * stride;  // the window starts at the draw's lowest index (draw() adds vertexOffset = -first)
                    if (skip) { base += skip; size = size > skip ? size - skip : 0; }
                    size = std::min<uint64_t>(size, (cur_vtx_records - cur_vtx_first) * stride);
                }
                size = (size + 15) & ~uint64_t(15);
                if (size == 0) size = 16;
                // CPU-side uses of the bytes (copies below) need pending GPU writes to have landed: settle_pending further down. The
                // vertex cache refuses ranges with pending GPU writes, and an in-place import is ordered against them on the GPU.
                if (static const bool vdbg = std::getenv("BB_VTX_DBG") != nullptr; vdbg && bs.e->stage == ShStage::VS && stride && !res[i].scalar && size > (256u << 10)) {
                    static int nlog = 0;
                    if (nlog++ < 60) std::fprintf(stderr, "vtxdbg: %s stride=%u num_records=%u copy=%llu window [%llu,%llu) vertex_use=%u\n", bs.e->log_name.c_str(), stride, w[2], (unsigned long long)size, (unsigned long long)cur_vtx_first, (unsigned long long)cur_vtx_records, unsigned(res[i].vertex_use));
                }
                if (size > (1u << 20) && verbose) {
                    char t[300];
                    std::snprintf(t, sizeof t, "large buffer V#: %08x %08x %08x %08x size=%llu stride=%u (%s) vtx window [%llu, %llu)", w[0], w[1], w[2], w[3], (unsigned long long)size, stride, bs.e->log_name.c_str(), (unsigned long long)cur_vtx_first, (unsigned long long)cur_vtx_records);
                    log_once(t, t);
                }
                const auto mv = hooks().mem_valid;
                const bool mapped = base >= 0x10000 && (!mv || mv(base, size));  // (one mapping lookup for every use below)
                VkBuffer vtxb = VK_NULL_HANDLE;
                VkDeviceSize vtxo = 0;
                // (replay object interpolation: a bone palette goes through the ring, where a replay can patch it)
                const bool pal = obj_capture && cur_instances == 1 && stride == 16 && bs.e->stage == ShStage::VS && !res[i].scalar && !res[i].written && !res[i].load_data &&
                                 res[i].vertex_use == 2 && size < (64u << 10);
                if (!pal && bs.e->stage == ShStage::VS && stride && !res[i].scalar && !res[i].written && !res[i].load_data && mv && mapped && get_vertex_buffer(base, size, vtxb, vtxo)) {
                    info[i].b = {vtxb, vtxo, size};  // cached device-local copy of the stream
                    break;
                }
                VkBuffer hb = VK_NULL_HANDLE;
                VkDeviceSize hoff = 0;
                const bool maybe = base >= 0x10000 && maybe_dirty(base, base + size);  // (most buffers: no GPU writes in flight near them)
                auto in_writeback = [&] {  // the GPU result for these bytes still sits in the ring (copied back at settle; also a dirty range)
                    if (!maybe) return false;
                    for (uint32_t s = 0; s < nslots; ++s)
                        for (const Writeback& wb : wb_s[s]) { const uint64_t wl = reinterpret_cast<uintptr_t>(wb.dst); if (wl < base + size && base < wl + wb.size) return true; }
                    return false;
                };
                // small buffers are copied, unless pending GPU writes would force a full flush for the copy: then imported too
                const bool pending = maybe && overlaps_dirty(base, base + size);
                if (mv && mapped && (size >= (64u << 10) || pending) && !in_writeback() && get_host_buf(base, size, hb, hoff, pending)) {
                    // the GPU reads/writes guest memory directly: no copy, no write-back. Earlier GPU writes to the range (this slot or an
                    // in-flight one, same queue) are ordered by a full barrier instead of a CPU wait.
                    if (pending) { barrier(); ++import_barriers; }
                    if (res[i].written) { dirty_add(base, base + size); imp_written = true; op_unsafe = true; gpu_watch("shader (imported buffer)", base, size, bs.e->log_name.c_str()); }
                    else ++tick_imp_reads;  // a replay reads these guest bytes again, later
                    info[i].b = {hb, hoff, std::min<VkDeviceSize>(size, vk().props.limits.maxStorageBufferRange)};
                    break;
                }
                if (maybe) settle_pending(base, base + size, 1);
                if (!res[i].written && !mapped) {
                    // null/unmapped read-only buffer: reads yield 0 (the head of the ring is a permanent zero region)
                    log_once("vb" + std::to_string(base), "buffer resource points at unmapped memory");
                    ++zero_binds;
                    info[i].b = {ring.buffer, 0, std::min<uint64_t>(size, kZeroBytes)};
                    break;
                }
                size_t off = 0;
                uint8_t* p = ring_alloc(size, 256, off);
                if (!p) return false;
                if (mapped) {
                    { const auto t0 = tnow();
                      std::memcpy(p, reinterpret_cast<const void*>(base), size);
                      buf_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(tnow() - t0).count()); }
                    buf_bytes += size;
                    ++buf_copies;
                    (res[i].scalar ? cb_bytes : (bs.e->stage == ShStage::VS && stride) ? vb_bytes : ob_bytes) += size;
                    if (res[i].written) { wb_s[slot].push_back({reinterpret_cast<uint8_t*>(base), off, size}); dirty_add(base, base + size); op_unsafe = true; gpu_watch("shader (write-back)", base, size, bs.e->log_name.c_str()); }
                    else if (replay_on && replay_lerp && res[i].scalar && size == 864) note_site(p, off);
                    else if (obj_capture && (pal || (res[i].scalar && size <= 4096))) note_obj(p, off, size, uint32_t(bs.e->stage) << 16 | uint32_t(i), pal);
                    static const bool nav_cam = nav_camera_wanted();
                    if (nav_cam && res[i].scalar && size == 864 && !res[i].written) nav_note_camera(reinterpret_cast<const uint32_t*>(p));
                } else {
                    std::memset(p, 0, size);
                    log_once("vb" + std::to_string(base), "buffer resource points at unmapped memory");
                }
                if (g_log_tex) {
                    std::fprintf(stderr, "gpu:   V# b%u: %08x %08x %08x %08x  data:", res[i].binding, w[0], w[1], w[2], w[3]);
                    static const size_t vlog_off = std::getenv("BB_VLOG_OFF") ? std::strtoull(std::getenv("BB_VLOG_OFF"), nullptr, 16) * 4 : 0;  // debug: start the data dump at this dword offset (hex)
                    if (vlog_off) std::fprintf(stderr, " [from dword 0x%zx]", vlog_off / 4);
                    for (size_t k = vlog_off / 4; k < std::min<size_t>(size, vlog_off + 512) / 4; ++k) { float f; std::memcpy(&f, p + k * 4, 4); uint32_t u; std::memcpy(&u, p + k * 4, 4); std::fprintf(stderr, " %08x(%g)", u, f); }
                    {
                        size_t ff = 0, nz = 0;
                        for (size_t k = 0; k < size / 4; ++k) { uint32_t u; std::memcpy(&u, p + k * 4, 4); ff += u == 0xFFFFFFFFu; nz += u != 0 && u != 0xFFFFFFFFu; }
                        std::fprintf(stderr, " | size %zu: %zu dwords ffffffff, %zu other non-zero; written:", size, ff, nz);
                        for (size_t k = 0, s = ~size_t(0); k <= size / 4; ++k) {
                            uint32_t u = 0xFFFFFFFFu;
                            if (k < size / 4) std::memcpy(&u, p + k * 4, 4);
                            if (u != 0xFFFFFFFFu && s == ~size_t(0)) s = k;
                            if (u == 0xFFFFFFFFu && s != ~size_t(0)) { std::fprintf(stderr, " [%zx-%zx]", s, k - 1); s = ~size_t(0); }
                        }
                    }
                    std::fprintf(stderr, "\n");
                }
                info[i].b = {ring.buffer, off, size};
                break;
            }
            case Resource::Image:
            case Resource::StorageImage: {
                if (g_log_tex && res[i].type == Resource::Image) {
                    std::fprintf(stderr, "gpu:   T# b%u: %08x %08x %08x %08x %08x %08x %08x %08x", res[i].binding, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
                    const uint64_t tb = (uint64_t(w[0]) | (uint64_t(w[1] & 0x3F) << 32)) << 8;
                    if ((w[2] & 0x3FFF) < 4 && ((w[2] >> 14) & 0x3FFF) < 4 && tb >= 0x10000 && hooks().mem_valid && hooks().mem_valid(tb, 16)) {  // tiny texture (exposure etc.): show its first texel(s)
                        std::fprintf(stderr, "  texels:");
                        for (int k = 0; k < 4; ++k) { uint32_t u; std::memcpy(&u, reinterpret_cast<const uint8_t*>(tb) + 4 * k, 4); float f; std::memcpy(&f, &u, 4); std::fprintf(stderr, " %08x(%g)", u, f); }
                    }
                    std::fprintf(stderr, "\n");
                }
                if (verbose && res[i].type == Resource::StorageImage) {
                    char t[200];
                    std::snprintf(t, sizeof t, "storage image T#: %08x %08x %08x %08x %08x %08x %08x %08x", w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
                    log_once(t, t);
                }
                const auto tt0 = tnow();
                // padded targets in place: axes the image's sampler clamps to the edge (only loaded: any axis; several samplers: none)
                auto edge = [](uint32_t m) { return m == 2 || m == 4; };
                const int16_t si = res[i].type == Resource::Image ? res[i].sampler : -2;
                const uint32_t pad_axes = si == -1 ? 3u : si < 0 ? 0u : (edge(bs.words[si][0] & 7) ? 1u : 0u) | (edge((bs.words[si][0] >> 3) & 7) ? 2u : 0u);
                const uint32_t kind = uint32_t(res[i].type) | pad_axes << 4;
                uint64_t mh = kind;
                for (int k = 0; k < 8; ++k) mh = (mh ^ w[k]) * 0x9E3779B97F4A7C15ull;
                TexMemo& m = tex_memo[(mh >> 40) & 4095];
                const uint64_t epoch = g_tex_epoch.load(std::memory_order_relaxed);
                Image* im = nullptr;
                VkImageView view = VK_NULL_HANDLE;
                bool hit = m.im && m.kind == kind && m.gen == tex_gen && std::memcmp(m.w, w, sizeof m.w) == 0 && m.im->touched == m.touched &&
                           (m.dgen == dirty_gen || !maybe_dirty(m.base, m.base + m.bytes)) && (!m.src || m.src->version == m.ver);
                if (hit && m.epoch != epoch) {
                    // a new epoch: a write-watch tracked texture is still current when its pages were not written (get_texture's check
                    // without the lookups; written pages: get_texture hashes them). The whole image range: ww_epoch covers all of it.
                    Image* t = m.im;
                    uint64_t now = 0;
                    hit = t->ww == 1 && t->hash_epoch && m.bytes == t->bytes && ww_clean_since(t->base, t->base + t->bytes, t->ww_epoch, now) == 1;
                    if (hit) {
                        const uint64_t since = t->ww_epoch;
                        t->ww_epoch = now, t->hash_epoch = epoch, m.epoch = epoch;
                        ++tex_ww_checks, ++tex_ww_renewals;
                        if (static const bool tverify = std::getenv("BB_WW_VERIFY") != nullptr; tverify && (++tex_verify_tick & 63) == 0) {
                            ++tex_verified;
                            if (hash_fast(reinterpret_cast<const void*>(t->base), t->bytes & ~uint64_t(7)) != t->hash && tex_stale_ww++ < 8)
                                std::fprintf(stderr, "tex-verify: stale (write watch clean, memo, write watch: %s) base=0x%llx bytes=%llu %ux%u frame=%llu\n", ww_repoll(t->base, t->base + t->bytes, since) == 0 ? "late write" : "blind",
                                             (unsigned long long)t->base, (unsigned long long)t->bytes, t->w, t->h, (unsigned long long)g_frame_counter.load());
                        }
                    }
                }
                if (hit) {
                    m.dgen = dirty_gen;
                    im = m.im, view = m.view;
                    ++tex_memo_hits;
                    tex_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(tnow() - tt0).count());
                } else {
                    im = get_texture(w, res[i].type == Resource::StorageImage, pad_axes);
                    const bool memo_ok = tex_memo_ok;
                    Image* const memo_src = tex_memo_src;
                    tex_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(tnow() - tt0).count());
                    if (!im) { im = &dummy_tex; ensure_init(im); }
                    { ScopeNs vt{view_ns}; view = im == &dummy_tex ? im->view : texture_view(im, w, res[i].type == Resource::Image); }
                    if (memo_ok && im != &dummy_tex) {  // (get_texture may have flushed: gens are read after it; the epoch is the one read
                        std::memcpy(m.w, w, sizeof m.w);  // before it, as the present thread bumps g_tex_epoch without the backend lock)
                        m.kind = kind, m.im = im, m.view = view, m.epoch = epoch, m.gen = tex_gen, m.dgen = dirty_gen;
                        m.base = im->base, m.bytes = im->bytes;
                        m.src = memo_src, m.ver = memo_src ? memo_src->version : 0, m.touched = im->touched;
                    }
                }
                ++tex_calls;
                if (im != &dummy_tex) sampled.push_back(im), note_read(im);
                if (census && im != &dummy_tex) census_img(bs, res[i], w, im);  // (BB_SCALE_LOG)
                if (im->scale > 1 && i < 31) scale_bits[uint32_t(bs.e->stage)] |= 1u << i;
                if (res[i].type == Resource::StorageImage && res[i].written) ++im->version, op_writes.push_back(im);  // padded-target crops must be re-taken
                if (const uint64_t tb = (uint64_t(w[0]) | (uint64_t(w[1] & 0x3F) << 32)) << 8; gpu_watched(tb, uint64_t((w[2] & 0x3FFF) + 1) * (((w[2] >> 14) & 0x3FFF) + 1) * 16 * 8)) {  // BB_GPU_WATCH: T#s over the range, once each
                    static std::set<std::string> seen;
                    char t[300];
                    std::snprintf(t, sizeof t, "%s T# %08x %08x %08x %08x %08x %08x %08x %08x %s", res[i].type == Resource::Image ? "sampled" : res[i].written ? "storage-W" : "storage-R", w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], bs.e->log_name.c_str());
                    if (seen.size() < 200 && seen.insert(t).second) std::fprintf(stderr, "gpu-watch: %s draw=%llu frame=%llu\n", t, (unsigned long long)draws, (unsigned long long)g_frame_counter.load());
                }
                info[i].i = {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
                break;
            }
            case Resource::LdsBuffer:
            case Resource::TessIndices: {  // emulated LDS / control-point ids of the current patch draw (see tess_begin)
                const TessBind& tb = res[i].type == Resource::LdsBuffer ? tess_lds : tess_idx;
                info[i].b = {tb.buf, tb.off, tb.size};
                break;
            }
            case Resource::GdsBuffer:
                info[i].b = {ring.buffer, gds_off(), kGdsBytes};
                op_unsafe = true;  // (a replay would count again)
                break;
            case Resource::Sampler:
                { ScopeNs st{smp_ns}; info[i].i = {get_sampler(w), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED}; }
                break;
        }
    }
    return true;
}

const Backend::DescLayout& Backend::desc_layout(const ShaderEntry* a, const ShaderEntry* b, bool compute) {
    const uint64_t key = reinterpret_cast<uintptr_t>(a) * 0x9E3779B97F4A7C15ull ^ reinterpret_cast<uintptr_t>(b) ^ uint64_t(compute);
    auto it = dlayouts.find(key);
    if (it != dlayouts.end()) return it->second;
    VkCtx& c = vk();
    DescLayout& dl = dlayouts[key];
    std::vector<VkDescriptorSetLayoutBinding> lbs;
    std::vector<VkDescriptorUpdateTemplateEntry> te;  // resource i of a, then of b <- DescInfo i (prepare_descriptors)
    for (const ShaderEntry* e : {a, b}) {
        if (!e) continue;
        const VkShaderStageFlags stage = compute ? VK_SHADER_STAGE_COMPUTE_BIT : e == a ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        for (const Resource& res : e->tr.resources) {
            const VkDescriptorType t = res.type == Resource::Buffer || res.type == Resource::LdsBuffer || res.type == Resource::TessIndices || res.type == Resource::GdsBuffer ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                     : res.type == Resource::Image ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                     : res.type == Resource::StorageImage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER;
            te.push_back({res.binding, 0, 1, t, lbs.size() * sizeof(DescInfo), sizeof(DescInfo)});
            lbs.push_back({res.binding, t, 1, stage, nullptr});
        }
    }
    dl.push = !lbs.empty() && lbs.size() <= c.max_push;
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.flags = dl.push ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0;
    dli.bindingCount = uint32_t(lbs.size());
    dli.pBindings = lbs.data();
    vkCreateDescriptorSetLayout(c.device, &dli, nullptr, &dl.dsl);
    const VkPushConstantRange pcs[2] = {{compute ? VkShaderStageFlags(VK_SHADER_STAGE_COMPUTE_BIT) : VkShaderStageFlags(VK_SHADER_STAGE_VERTEX_BIT), 0, push_bytes()}, {VK_SHADER_STAGE_FRAGMENT_BIT, push_fs(), push_bytes()}};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dl.dsl;
    pli.pushConstantRangeCount = b ? 2 : 1;
    pli.pPushConstantRanges = pcs;
    vkCreatePipelineLayout(c.device, &pli, nullptr, &dl.layout);
    if (!te.empty()) {
        VkDescriptorUpdateTemplateCreateInfo tci{VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO};
        tci.descriptorUpdateEntryCount = uint32_t(te.size());
        tci.pDescriptorUpdateEntries = te.data();
        tci.templateType = dl.push ? VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR : VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
        tci.descriptorSetLayout = dl.dsl;
        tci.pipelineBindPoint = compute ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS;
        tci.pipelineLayout = dl.layout;
        vkCreateDescriptorUpdateTemplate(c.device, &tci, nullptr, &dl.tmpl);
    }
    return dl;
}

// Fills wd_info with a's then b's descriptors; pool path (dl.push false): also allocates and updates `set`. May flush (see draw).
bool Backend::prepare_descriptors(const DescLayout& dl, const BoundShader& a, const BoundShader* b, VkDescriptorSet& set) {
    VkCtx& c = vk();
    set = VK_NULL_HANDLE;
    if (!dl.push && dl.tmpl) {
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = dpool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &dl.dsl;
        VkResult ar;
        { ScopeNs at{ds_alloc_ns}; ar = vkAllocateDescriptorSets(c.device, &dai, &set); ++ds_pool_sets; }
        if (ar != VK_SUCCESS) { flush(); dai.descriptorPool = dpool; if (vkAllocateDescriptorSets(c.device, &dai, &set) != VK_SUCCESS) return false; }
    }
    const size_t na = a.e->tr.resources.size();
    wd_info.resize(na + (b && b->e ? b->e->tr.resources.size() : 0));
    if (!write_descriptors(a, wd_info.data()) || (b && b->e && !write_descriptors(*b, wd_info.data() + na))) return false;
    // one template update (cheaper in the driver than a VkWriteDescriptorSet per resource, which also no longer have to be built)
    if (set) { ScopeNs ut{ds_update_ns}; vkUpdateDescriptorSetWithTemplate(c.device, set, dl.tmpl, wd_info.data()); }
    return true;
}

// Records the descriptors prepare_descriptors filled in (push: wd_info must be unchanged since).
void Backend::bind_descriptors(VkPipelineBindPoint bp, const DescLayout& dl, VkDescriptorSet set) {
    if (set) vkCmdBindDescriptorSets(cb, bp, dl.layout, 0, 1, &set, 0, nullptr);
    else if (dl.tmpl) { ScopeNs ut{ds_update_ns}; vkCmdPushDescriptorSetWithTemplateKHR(cb, dl.tmpl, dl.layout, 0, wd_info.data()); }
}

// Geometry shader turning a 3-vertex RECTLIST primitive into a 4-vertex strip (4th = v1 + v2 - v0 for every output).
VkShaderModule Backend::get_rect_gs(uint32_t param_mask) {
    auto it = rect_gs.find(param_mask);
    if (it != rect_gs.end()) return it->second;
    SpvBuilder b;
    b.capability(spv::Capability::Shader);
    b.capability(spv::Capability::Geometry);
    using Id = SpvBuilder::Id;
    Id F = b.t_f32(), V4 = b.t_vec(F, 4), U = b.t_u32();
    Id arr3 = b.t_array(V4, b.c_u32(3));
    std::vector<Id> iface;
    auto var = [&](spv::StorageClass sc, Id t) { Id v = b.global_var(b.t_ptr(sc, t), sc); iface.push_back(v); return v; };
    Id pos_in = var(spv::StorageClass::Input, arr3), pos_out = var(spv::StorageClass::Output, V4);
    b.decorate(pos_in, spv::Decoration::BuiltIn, {uint32_t(spv::BuiltIn::Position)});
    b.decorate(pos_out, spv::Decoration::BuiltIn, {uint32_t(spv::BuiltIn::Position)});
    std::vector<std::pair<Id, Id>> params;  // (in array, out)
    for (uint32_t n = 0; n < 32; ++n)
        if ((param_mask >> n) & 1) {
            Id in = var(spv::StorageClass::Input, arr3), out = var(spv::StorageClass::Output, V4);
            b.decorate(in, spv::Decoration::Location, {n});
            b.decorate(out, spv::Decoration::Location, {n});
            params.push_back({in, out});
        }
    Id fn = b.begin_function(b.t_void(), b.t_fn(b.t_void()));
    auto ld3 = [&](Id arr, uint32_t i) { Id p = b.op(spv::Op::OpAccessChain, b.t_ptr(spv::StorageClass::Input, V4), {arr, b.c_u32(i)}); return b.op(spv::Op::OpLoad, V4, {p}); };
    auto emit = [&](Id pos_val, const std::vector<Id>& pvals) {
        b.op0(spv::Op::OpStore, {pos_out, pos_val});
        for (size_t k = 0; k < params.size(); ++k) b.op0(spv::Op::OpStore, {params[k].second, pvals[k]});
        b.op0(spv::Op::OpEmitVertex, {});
    };
    auto corner = [&](int i) { std::vector<Id> pv; for (auto& p : params) pv.push_back(ld3(p.first, i)); return std::make_pair(ld3(pos_in, i), pv); };
    auto c0 = corner(0), c1 = corner(1), c2 = corner(2);
    std::vector<Id> p3;
    for (size_t k = 0; k < params.size(); ++k) p3.push_back(b.op(spv::Op::OpFSub, V4, {b.op(spv::Op::OpFAdd, V4, {c1.second[k], c2.second[k]}), c0.second[k]}));
    Id pos3 = b.op(spv::Op::OpFSub, V4, {b.op(spv::Op::OpFAdd, V4, {c1.first, c2.first}), c0.first});
    emit(c0.first, c0.second);
    emit(c1.first, c1.second);
    emit(c2.first, c2.second);
    emit(pos3, p3);
    b.op0(spv::Op::OpEndPrimitive, {});
    b.op0(spv::Op::OpReturn, {});
    b.end_function();
    b.entry_point(spv::ExecutionModel::Geometry, fn, "main", iface);
    b.exec_mode(fn, spv::ExecutionMode::Triangles);
    b.exec_mode(fn, spv::ExecutionMode::OutputTriangleStrip);
    b.exec_mode(fn, spv::ExecutionMode::OutputVertices, {4});
    b.exec_mode(fn, spv::ExecutionMode::Invocations, {1});
    (void)U;
    std::vector<uint32_t> code = b.finish();
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size() * 4;
    smi.pCode = code.data();
    VkShaderModule m = VK_NULL_HANDLE;
    vk_check(vkCreateShaderModule(vk().device, &smi, nullptr, &m), "rect geometry shader");
    rect_gs[param_mask] = m;
    return m;
}

// ---- draw / dispatch ---------------------------------------------------------------------------------------------------------
VkBlendFactor blend_factor(uint32_t f) {
    switch (f) {
        case 0: return VK_BLEND_FACTOR_ZERO;
        case 1: return VK_BLEND_FACTOR_ONE;
        case 2: return VK_BLEND_FACTOR_SRC_COLOR;
        case 3: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 4: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 5: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 6: return VK_BLEND_FACTOR_DST_ALPHA;
        case 7: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 8: return VK_BLEND_FACTOR_DST_COLOR;
        case 9: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 10: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 13: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case 14: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case 19: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case 20: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        default: return VK_BLEND_FACTOR_ONE;
    }
}
VkBlendOp blend_op(uint32_t f) {
    switch (f) { case 1: return VK_BLEND_OP_SUBTRACT; case 2: return VK_BLEND_OP_MIN; case 3: return VK_BLEND_OP_MAX; case 4: return VK_BLEND_OP_REVERSE_SUBTRACT; default: return VK_BLEND_OP_ADD; }
}

// Debug BB_CONST_DUMP=<from s>[,<frames>[,<file>]]: <frames> (default 8) whole frames from flip <from s>*30 on (seconds as with
// BB_PAD_CLOCK=flips, plus the flips before the first pad poll), one binary record per recorded draw/dispatch into <file>
// (default const_dump.bin; read by build/const_diff.py). Record: u32 'CDRW', frame, seq, kind (0 draw, 1 dispatch); u64 shader0 (VS/CS),
// shader1 (PS), rt0, zbase; u32 target mask, count, instances, resources; u32 user0[16], user1[16]. Per resource: u8 stage (0 VS/CS,
// 1 PS), type, flags (1 scalar, 2 written, 4 VS strided buffer, 8 load_data, 16 mapped, 32/64 vertex_use bit 0/1), 0; u32 words[8];
// u64 base, size, hash (pure vertex streams: the draw's record window, as bound); u32 n, then n bytes (every buffer but pure vertex
// streams (vertex_use 1): up to 64 KiB, written ones up to 4 KiB).
// ponytail: reads guest memory as is (pending GPU writes not settled); ranges capped at 16 MiB. BB_CONST_DUMP_VS=<hex VS address>: only draws with that VS.
void Backend::const_dump(uint32_t kind, uint64_t sh0, uint64_t sh1, const BoundShader* a, const BoundShader* b, const uint32_t* ua, const uint32_t* ub,
                         uint64_t rt0, uint64_t zb, uint32_t mask, uint32_t count, uint32_t inst) {
    static const char* e = std::getenv("BB_CONST_DUMP");
    static const char* c1 = std::strchr(e, ',');
    static const char* c2 = c1 ? std::strchr(c1 + 1, ',') : nullptr;
    static const uint64_t frames = c1 ? std::strtoull(c1 + 1, nullptr, 10) : 8;
    static FILE* f = nullptr;
    static bool opened = false;
    static uint64_t first = ~0ull, cur = ~0ull;
    static uint32_t seq = 0;
    static const uint64_t only_vs = std::getenv("BB_CONST_DUMP_VS") ? std::strtoull(std::getenv("BB_CONST_DUMP_VS"), nullptr, 16) : 0;  // debug: only draws with this VS
    if (only_vs && sh0 != only_vs) return;
    const uint64_t fr = g_frame_counter.load(std::memory_order_relaxed);
    if (first == ~0ull) {
        if (double(fr) / 30 < std::atof(e)) return;
        first = fr + 1;  // start at a frame boundary
    }
    if (fr < first) return;
    if (fr >= first + frames) {
        if (f) { std::fclose(f); f = nullptr; std::fprintf(stderr, "gpu: const dump of frames %llu..%llu done\n", (unsigned long long)first, (unsigned long long)(first + frames - 1)); }
        return;
    }
    if (!opened) { opened = true; f = std::fopen(c2 ? c2 + 1 : "const_dump.bin", "wb"); }
    if (!f) return;
    if (fr != cur) { cur = fr; seq = 0; }
    auto put = [&](const void* p, size_t n) { std::fwrite(p, 1, n, f); };
    auto u32 = [&](uint32_t v) { put(&v, 4); };
    auto u64 = [&](uint64_t v) { put(&v, 8); };
    static const uint32_t zero[16] = {};
    const size_t na = a && a->e ? std::min(a->words.size(), a->e->tr.resources.size()) : 0, nb = b && b->e ? std::min(b->words.size(), b->e->tr.resources.size()) : 0;
    u32(0x57524443); u32(uint32_t(fr)); u32(seq++); u32(kind);
    u64(sh0); u64(sh1); u64(rt0); u64(zb);
    u32(mask); u32(count); u32(inst); u32(uint32_t(na + nb));
    put(ua ? ua : zero, 64); put(ub ? ub : zero, 64);
    const auto mv = hooks().mem_valid;
    for (const BoundShader* bs : {a, b}) {
        const size_t n_res = bs == a ? na : nb;
        for (size_t i = 0; i < n_res; ++i) {
            const Resource& rs = bs->e->tr.resources[i];
            const uint32_t* w = bs->words[i].data();
            uint64_t base = 0, size = 0, h = 0;
            uint32_t n = 0;
            uint8_t flags = uint8_t((rs.scalar ? 1 : 0) | (rs.written ? 2 : 0) | (rs.load_data ? 8 : 0) | ((rs.vertex_use & 3) << 5));
            if (rs.type == Resource::Buffer) {
                base = uint64_t(w[0]) | (uint64_t(w[1] & 0xFFF) << 32);
                const uint32_t stride = (w[1] >> 16) & 0x3FFF;
                size = stride ? uint64_t(w[2]) * stride : w[2];
                const bool vtx = bs->e->stage == ShStage::VS && stride && !rs.scalar;
                if (vtx && rs.vertex_use == 1 && cur_vtx_records != ~0ull) {  // as write_descriptors binds it
                    const uint64_t skip = cur_vtx_first * stride;
                    base += skip;
                    size = std::min(size > skip ? size - skip : 0, (cur_vtx_records - cur_vtx_first) * stride);
                }
                size = std::min<uint64_t>(size, uint64_t(16) << 20);
                flags |= vtx ? 4 : 0;
                if (base >= 0x10000 && size && mv && mv(base, size)) {
                    flags |= 16;
                    h = hash_fast(reinterpret_cast<const void*>(base), size_t(size));
                    n = vtx && rs.vertex_use == 1 ? 0 : uint32_t(std::min<uint64_t>(size, rs.written ? 4096 : 65536));
                }
            }
            const uint8_t hdr[4] = {uint8_t(bs == b), uint8_t(rs.type), flags, 0};
            put(hdr, 4); put(w, 32); u64(base); u64(size); u64(h); u32(n);
            if (n) put(reinterpret_cast<const void*>(base), n);
        }
    }
}

void Backend::draw(const RegView& r, const DrawCmd& d_in) {
    ++draws;
    DrawCmd d = d_in;  // patch draws replace count/instances/indices below
    // BB_LOG_RT=<hex base>[,<first draw>|,t<seconds>]: the full state log (incl. sampled T#/V#) for the 24 draws into that target from the
    // <first draw>-th draw (or that many wall seconds) on
    static const uint64_t log_rt = std::getenv("BB_LOG_RT") ? std::strtoull(std::getenv("BB_LOG_RT"), nullptr, 16) : 0;
    static const char* log_rt_c = [] { const char* e = std::getenv("BB_LOG_RT"); return e ? std::strchr(e, ',') : nullptr; }();
    static const bool log_rt_secs = log_rt_c && log_rt_c[1] == 't';
    static const uint64_t log_rt_from = log_rt_c ? std::strtoull(log_rt_c + 1 + log_rt_secs, nullptr, 10) : 0ull;
    static const auto log_rt_t0 = std::chrono::steady_clock::now();
    static unsigned log_rt_n = 0;
    static const unsigned log_rt_max = std::getenv("BB_LOG_RT_N") ? unsigned(std::atoi(std::getenv("BB_LOG_RT_N"))) : 24;
    static const uint64_t log_ps = std::getenv("BB_LOG_PS") ? std::strtoull(std::getenv("BB_LOG_PS"), nullptr, 16) : 0;  // debug: same log for draws with this PS (honours the ,t<seconds> start of BB_LOG_RT)
    const uint64_t ps_addr = (uint64_t(r.sh[kShPsLo + 1]) << 40) | (uint64_t(r.sh[kShPsLo]) << 8);
    const bool rt_hit = ((log_rt && (uint64_t(r.context[kCbColor0Base]) << 8) == log_rt) || (log_ps && ps_addr == log_ps))
                     && (log_rt_secs ? std::chrono::steady_clock::now() - log_rt_t0 >= std::chrono::seconds(log_rt_from) : draws >= log_rt_from) && log_rt_n++ < log_rt_max;
    {
        static unsigned wlo = 0, whi = 0;
        static const bool wat = [] { const char* a = std::getenv("BB_GPU_DUMP_AT"); return a && std::sscanf(a, "%u-%u", &wlo, &whi) == 2; }();
        g_log_tex = rt_hit || (wat && rendered + 1 >= wlo && rendered + 1 <= whi);
    }
    if (verbose) {  // one line per distinct (vs, ps, primitive, colour target, mask, index count, indexed) signature
        char sig[200];
        const uint64_t ps = (uint64_t(r.sh[kShPsLo + 1]) << 40) | (uint64_t(r.sh[kShPsLo]) << 8), vs = (uint64_t(r.sh[kShVsLo + 1]) << 40) | (uint64_t(r.sh[kShVsLo]) << 8);
        std::snprintf(sig, sizeof sig, "draw sig: vs=%llx ps=%llx prim=%u n=%u%s rt0=%llx mask=%x zctl=%x", (unsigned long long)vs, (unsigned long long)ps,
                      r.uconfig[kVgtPrimType] & 0x3F, d.count, d.indexed ? "i" : "", (unsigned long long)r.context[kCbColor0Base] << 8,
                      r.context[kCbTargetMask], r.context[kDbDepthControl]);
        log_once(sig, sig);
    }
    const uint32_t prim = r.uconfig[kVgtPrimType] & 0x3F;
    const bool rect = prim == 17;
    VkPrimitiveTopology topo;
    bool tess = false;
    switch (prim) {
        case 1: topo = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
        case 2: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case 3: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
        case 4: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
        case 5: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
        case 6: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
        case 17: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;  // + geometry shader expansion
        case 9: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; tess = true; break;  // patch list: LS compute pass + DS over a generated grid (tess_begin)
        default:
            { char b[200]; std::snprintf(b, sizeof b, "primitive type %u not implemented; draw skipped (zwrite=0x%x dctl=0x%x zinfo=0x%x target_mask=0x%x)", unsigned(prim), r.context[CTX(0x28050)], r.context[kDbDepthControl], r.context[kDbZInfo], r.context[kCbTargetMask]); log_once(b, b); }
            if (static bool tess_dumped = false; prim == 9 && verbose && !tess_dumped) {  // diagnosis for the future tessellation support: every register that configures LS/HS/DS and the tess factors
                tess_dumped = true;
                std::fprintf(stderr, "gpu: tess regs: sh LS=%08x:%08x HS=%08x:%08x ES=%08x:%08x GS=%08x:%08x VS=%08x:%08x | rsrc LS %08x %08x HS %08x %08x VS %08x %08x\n", r.sh[0x148], r.sh[0x149], r.sh[0x108], r.sh[0x109], r.sh[0xC8], r.sh[0xC9], r.sh[0x88], r.sh[0x89], r.sh[0x48], r.sh[0x49], r.sh[0x14A], r.sh[0x14B], r.sh[0x10A], r.sh[0x10B], r.sh[0x4A], r.sh[0x4B]);
                std::fprintf(stderr, "gpu: tess regs: user LS:"); for (int k = 0; k < 16; ++k) std::fprintf(stderr, " %08x", r.sh[0x14C + k]);
                std::fprintf(stderr, "\ngpu: tess regs: user HS:"); for (int k = 0; k < 16; ++k) std::fprintf(stderr, " %08x", r.sh[0x10C + k]);
                std::fprintf(stderr, "\ngpu: tess regs: user VS:"); for (int k = 0; k < 16; ++k) std::fprintf(stderr, " %08x", r.sh[0x4C + k]);
                std::fprintf(stderr, "\ngpu: tess regs: ctx 2d4..2e0:"); for (int k = 0x2d4; k <= 0x2e0; ++k) std::fprintf(stderr, " [%x]=%08x", k, r.context[k]);
                std::fprintf(stderr, "\ngpu: tess regs: uconfig tf ring 0x240..0x24a:"); for (int k = 0x240; k <= 0x24a; ++k) std::fprintf(stderr, " [%x]=%08x", k, r.uconfig[k]);
                std::fprintf(stderr, "\ngpu: tess regs: prim=%u count=%u instances=%u indexed=%d index_addr=%llx\n", unsigned(prim), d.count, d.instances, int(d.indexed), (unsigned long long)d.index_addr);
            }
            ++skipped, ++skip_by["prim"];
            return;
    }
    auto sec_t = tnow();
    uint32_t tess_level = 0;
    uint64_t tess_f0 = 0;
    if (tess) {
        uint32_t patches = 0;
        static const bool tess_on = !std::getenv("BB_TESS") || std::atoi(std::getenv("BB_TESS")) != 0;  // BB_TESS=0 disables patch draws (diagnosis)
        if (!tess_on || !tess_begin(r, d_in, d, tess_level, patches)) { ++skipped, ++skip_by["tess"]; return; }
        cur_tess_ds = tess_level;
        tess_f0 = tess_ring_flush;
    }
    BoundShader& vs = bs_vs;
    get_shader(ShStage::VS, r, vs);
    cur_tess_ds = 0;
    BoundShader& ps = bs_ps;
    get_shader(ShStage::PS, r, ps);
    static const std::vector<uint64_t> skip_ps = [] {  // debug BB_SKIP_PS=<hex>[,<hex>...]: A/B pixel shaders (their draws are dropped)
        std::vector<uint64_t> v;
        if (const char* e = std::getenv("BB_SKIP_PS")) for (const char* c = e; *c;) { char* end = nullptr; v.push_back(std::strtoull(c, &end, 16)); if (end == c) break; c = *end == ',' ? end + 1 : end; }
        return v;
    }();
    if (!skip_ps.empty() && std::find(skip_ps.begin(), skip_ps.end(), (uint64_t(r.sh[kShPsLo + 1]) << 40) | (uint64_t(r.sh[kShPsLo]) << 8)) != skip_ps.end()) { ++skipped, ++skip_by["skip_ps"]; return; }
    // Debug BB_RES_LOG=<hex vs address>[,<seconds>]: every 10 s (from <seconds>) for one frame, the inputs of each draw with that VS: user
    // registers, descriptor words, and per buffer a hash of the guest bytes (no readback, no flush). Time counts from the first draw.
    static const auto res_t0 = std::chrono::steady_clock::now();
    if (static const char* rl = std::getenv("BB_RES_LOG"); rl && vs.e && ((uint64_t(r.sh[kShVsLo + 1]) << 40) | (uint64_t(r.sh[kShVsLo]) << 8)) == std::strtoull(rl, nullptr, 16)) {
        static const double from = std::strchr(rl, ',') ? std::atof(std::strchr(rl, ',') + 1) : 0;
        const auto t0 = res_t0;
        static double next = from;
        static uint64_t armed = ~0ull, first_armed = ~0ull;
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (t >= next) { next = t + 10; armed = g_frame_counter + 1; if (first_armed == ~0ull) first_armed = armed; }
        static const char* dump_dir = std::getenv("BB_RES_DUMP");  // with BB_RES_LOG: the buffers and indices of the first logged frame as files d<draw>_c<count>_*
        const bool dump = dump_dir && g_frame_counter == first_armed;
        auto dump_file = [&](const char* what, const void* p, uint64_t n) {
            char path[512];
            std::snprintf(path, sizeof path, "%s/d%llu_c%u_%s.bin", dump_dir, (unsigned long long)draws, d.count, what);
            if (FILE* f = std::fopen(path, "wb")) { std::fwrite(p, 1, size_t(n), f); std::fclose(f); }
        };
        if (g_frame_counter == armed) {
            uint64_t uh = 0;
            for (int k = 0; k < 16; ++k) uh = uh * 31 + r.sh[kShVsUser + k];
            std::fprintf(stderr, "reslog t=%.0f frame %llu draw %llu count=%u inst=%u idx=%llx user=%016llx\n", t, (unsigned long long)g_frame_counter.load(), (unsigned long long)draws, d.count, d.instances, (unsigned long long)d.index_addr, (unsigned long long)uh);
            std::fprintf(stderr, "reslog   ps_in_control %08x input_cntl", r.context[kSpiPsInControl]);
            for (uint32_t k = 0; k < 8; ++k) std::fprintf(stderr, " %08x", r.context[kSpiPsInputCntl0 + k]);
            std::fprintf(stderr, " vs %s ps %s\n", vs.e->log_name.c_str(), ps.e ? ps.e->log_name.c_str() : "-");
            if (dump && d.indexed && hooks().mem_valid && hooks().mem_valid(d.index_addr, uint64_t(d.count) * d.index_bytes))
                dump_file("indices", reinterpret_cast<const void*>(d.index_addr), uint64_t(d.count) * d.index_bytes);
            for (const BoundShader* b : {&vs, &ps}) {
                if (!b->e) continue;
                for (size_t i = 0; i < b->words.size() && i < b->e->tr.resources.size(); ++i) {
                    const auto& w = b->words[i];
                    const auto ty = b->e->tr.resources[i].type;
                    std::fprintf(stderr, "reslog   %s r%zu type %d: %08x %08x %08x %08x", b == &vs ? "vs" : "ps", i, int(ty), w[0], w[1], w[2], w[3]);
                    if (ty == Resource::Buffer) {
                        const uint64_t base = uint64_t(w[0]) | (uint64_t(w[1] & 0xFFF) << 32), stride = (w[1] >> 16) & 0x3FFF;
                        const uint64_t size = std::min<uint64_t>(stride ? uint64_t(w[2]) * stride : w[2], uint64_t(64) << 20) & ~uint64_t(7);
                        if (base >= 0x10000 && size && hooks().mem_valid && hooks().mem_valid(base, size)) {
                            const uint8_t* p = reinterpret_cast<const uint8_t*>(base);
                            size_t nz = 0;
                            for (size_t k = 0; k < std::min<uint64_t>(size, 4096); ++k) nz += p[k] != 0;
                            std::fprintf(stderr, " size %llu hash %016llx nz4k %zu", (unsigned long long)size, (unsigned long long)hash_fast(p, size_t(size)), nz);
                            if (dump) { char what[32]; std::snprintf(what, sizeof what, "%s_r%zu", b == &vs ? "vs" : "ps", i); dump_file(what, p, std::min<uint64_t>(size, 65536)); }  // (a draw's vertices start at the V# base)
                        } else std::fprintf(stderr, " size %llu (unmapped)", (unsigned long long)size);
                    }
                    std::fputc('\n', stderr);
                }
            }
        }
    }
    g_ns_shader += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(tnow() - sec_t).count());
    sec_t = tnow();
    // depth/stencil target (DB_* registers): the format registers say what exists, DB_DEPTH_CONTROL what this draw uses
    const uint32_t zfmt = r.context[kDbZInfo] & 3, sfmt = r.context[CTX(0x28044)] & 1, zsize = r.context[CTX(0x28058)];
    const uint32_t dctl = r.context[kDbDepthControl], sctl = r.context[CTX(0x2842c)];
    if (static const char* lz = std::getenv("BB_LOG_Z"); lz && (uint64_t(r.context[CTX(0x28050)]) << 8) == std::strtoull(lz, nullptr, 16)) {  // debug: every draw into one depth buffer, by state
        char b[400];
        std::snprintf(b, sizeof b, "logz: dctl=0x%x zinfo=0x%x rctl=0x%x target_mask=0x%x vs=%s ps=%s topo=%u vserr=%d pserr=%d", dctl, r.context[kDbZInfo], r.context[CTX(0x28000)], r.context[kCbTargetMask], vs.e ? vs.e->log_name.c_str() : "?", ps.e ? ps.e->log_name.c_str() : "none", unsigned(topo), vs.e ? int(!vs.e->tr.error.empty()) : -1, ps.e ? int(!ps.e->tr.error.empty()) : -1);
        log_once(b, b);
    }
    const VkFormat dsfmt = zfmt && sfmt ? VK_FORMAT_D32_SFLOAT_S8_UINT : zfmt ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_S8_UINT;  // ponytail: 16-bit depth stored as 32-bit float
    auto ds_image = [&]() -> Image* {
        if (!zfmt && !sfmt) return nullptr;
        const uint64_t dsbase = uint64_t(zfmt ? r.context[CTX(0x28050)] : r.context[CTX(0x28054)]) << 8;
        Image* dsi = get_ds(dsbase, ((zsize & 0x7FF) + 1) * 8, (((zsize >> 11) & 0x7FF) + 1) * 8, dsfmt);
        if (dsi && sfmt) dsi->sbase = uint64_t(r.context[CTX(0x28054)]) << 8;
        if (dsi && ((r.context[kDbZInfo] >> 29) & 1) && r.context[CTX(0x28014)]) htile_ds[uint64_t(r.context[CTX(0x28014)]) << 8] = dsbase;  // DB_HTILE_DATA_BASE
        if (dsi && htile_clear_pending.erase(dsbase)) {  // HTILE fast clear: every tile reads as DB_DEPTH_CLEAR / DB_STENCIL_CLEAR until written
            ensure_init(dsi);
            note_write(dsi);
            barrier();
            VkClearDepthStencilValue v{};
            std::memcpy(&v.depth, &r.context[CTX(0x2802c)], 4);
            v.stencil = r.context[CTX(0x28028)] & 0xFF;
            const VkImageAspectFlags asp = (dsi->format == VK_FORMAT_S8_UINT ? 0u : VK_IMAGE_ASPECT_DEPTH_BIT) | (dsi->format == VK_FORMAT_D32_SFLOAT ? 0u : VK_IMAGE_ASPECT_STENCIL_BIT);
            const VkImageSubresourceRange range{asp, 0, 1, 0, dsi->layers};
            ts_label("clear: HTILE depth");
            vkCmdClearDepthStencilImage(rcb(), dsi->image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &range);
            barrier();
            ++ds_clears;
            if (static const bool ops = std::getenv("BB_OPS_LOG") != nullptr; ops) std::fprintf(stderr, "op: HTILE clear of depth %llx to %08x applied\n", (unsigned long long)dsbase, r.context[CTX(0x2802c)]);
        }
        return dsi;
    };
    // effective scissor = intersection of vport, generic, window and screen scissors (TL/BR pairs; unprogrammed registers are ignored)
    auto eff_scissor = [&](uint32_t w, uint32_t h) {
        int32_t sx = 0, sy = 0, ex = int32_t(w), ey = int32_t(h);
        for (const uint32_t reg : {kPaScVportScissor0Tl, CTX(0x28240), CTX(0x28204), CTX(0x28030)}) {
            if (reg == kPaScVportScissor0Tl && !(r.context[CTX(0x28a48)] & 2)) continue;  // PA_SC_MODE_CNTL_0.VPORT_SCISSOR_ENABLE
            const uint32_t tl = r.context[reg], br = r.context[reg + 1];
            const int32_t x0 = int32_t(tl & 0x7FFF), y0 = int32_t((tl >> 16) & 0x7FFF), x1 = int32_t(br & 0x7FFF), y1 = int32_t((br >> 16) & 0x7FFF);
            if (x1 <= x0 || y1 <= y0) continue;
            sx = std::max(sx, x0); sy = std::max(sy, y0); ex = std::min(ex, x1); ey = std::min(ey, y1);
        }
        return VkRect2D{{sx, sy}, {uint32_t(std::max(0, ex - sx)), uint32_t(std::max(0, ey - sy))}};
    };
    auto phys_rect = [](VkRect2D r, uint32_t s) {  // a guest-pixel rectangle on a target of scale s (integer: exact, no gaps)
        return VkRect2D{{r.offset.x * int32_t(s), r.offset.y * int32_t(s)}, {r.extent.width * s, r.extent.height * s}};
    };
    const uint32_t rctl = r.context[CTX(0x28000)];
    if (rctl & 3) {  // DB_RENDER_CONTROL depth/stencil clear: the draw is only the clear rectangle, values come from DB_DEPTH_CLEAR / DB_STENCIL_CLEAR
        if (Image* dsimg = ds_image()) {
            ensure_init(dsimg);
            note_write(dsimg);
            barrier();
            VkRenderingAttachmentInfo a{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            a.imageView = dsimg->view; a.imageLayout = VK_IMAGE_LAYOUT_GENERAL; a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0, 0}, {dsimg->w, dsimg->h}};
            ri.layerCount = 1;
            ri.pDepthAttachment = zfmt ? &a : nullptr;
            ri.pStencilAttachment = sfmt ? &a : nullptr;
            VkClearAttachment ca{};
            ca.aspectMask = ((rctl & 1) && zfmt ? VK_IMAGE_ASPECT_DEPTH_BIT : 0u) | ((rctl & 2) && sfmt ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
            std::memcpy(&ca.clearValue.depthStencil.depth, &r.context[CTX(0x2802c)], 4);
            ca.clearValue.depthStencil.stencil = r.context[CTX(0x28028)] & 0xFF;
            const VkClearRect cr{phys_rect(eff_scissor(dsimg->gw(), dsimg->gh()), dsimg->scale), 0, 1};
            ts_label("clear: depth attachments");
            vkCmdBeginRendering(rcb(), &ri);
            if (ca.aspectMask) vkCmdClearAttachments(cb, 1, &ca, 1, &cr);
            vkCmdEndRendering(cb);
            barrier();
            ++ds_clears;
        }
        return;
    }
    if (!vs.e || !vs.e->tr.error.empty() || (ps.e && !ps.e->tr.error.empty())) {
        ++skipped, ++skip_by["shader"];
        if (verbose) log_once(std::string("skipvsps") + (vs.e ? vs.e->log_name : "?") + (ps.e ? ps.e->log_name : "?"), std::string("draw skipped: ") + (vs.e ? vs.e->log_name : "vs unavailable") + " / " + (ps.e ? ps.e->log_name : "ps absent"));
        return;
    }

    // colour targets exported by the pixel shader and enabled in CB_TARGET_MASK
    struct Rt { Image* im; uint32_t slot; uint32_t layer; };
    static thread_local std::vector<Rt> targets;  // capacity reused across draws (draw() is not re-entered)
    targets.clear();
    uint32_t rt_w = ~0u, rt_h = ~0u;
    for (uint32_t k = 0; k < 8; ++k) {
        if (!ps.e || !((ps.e->tr.ps_mrts >> k) & 1) || !((r.context[kCbTargetMask] >> (4 * k)) & 0xF)) continue;
        const uint32_t* cbr = &r.context[kCbColor0Base + k * kCbColorStride];
        const uint64_t base = uint64_t(cbr[0]) << 8;
        bool bgra;
        const VkFormat fmt = rt_format(cbr[4], bgra);
        const uint32_t pitch_px = ((cbr[1] & 0x7FF) + 1) * 8, slice_tiles = (cbr[2] & 0x3FFFFF) + 1;
        const uint32_t height = std::max<uint32_t>(1, slice_tiles * 64 / pitch_px);
        if (!base || fmt == VK_FORMAT_UNDEFINED) { log_once("rtf" + std::to_string(cbr[4]), "colour target format info=" + std::to_string(cbr[4]) + " not implemented; draw skipped"); ++skipped, ++skip_by["rt_fmt"]; return; }
        // CB_COLOR_VIEW: SLICE_START [10:0], SLICE_MAX [23:13]. A small SLICE_MAX = an array/cube target (e.g. the point-light shadow cube,
        // drawn face by face); ponytail: larger values are treated as plain 2D targets (no game target seen with more than 6 slices)
        const uint32_t slice = cbr[3] & 0x7FF, slice_max = (cbr[3] >> 13) & 0x7FF;
        const uint32_t need = std::max(slice, slice_max) + 1;  // per-face draws set SLICE_MAX = face index: size the array up front (multiple of 6)
        const uint32_t layers = need > 1 && need <= 16 ? (need + 5) / 6 * 6 : 1;
        Image* im = get_rt(base, pitch_px, height, fmt, layers);
        if (static int rtw = 0; gpu_watched(base, uint64_t(pitch_px) * height * 16 * layers) && rtw++ < 200) std::fprintf(stderr, "gpu-watch: render target 0x%llx %ux%u x%u fmt %d ps %s draw=%llu\n", (unsigned long long)base, pitch_px, height, layers, int(fmt), ps.e->log_name.c_str(), (unsigned long long)draws);
        if (!im) { ++skipped, ++skip_by["rt_alloc"]; return; }
        im->bgra = bgra;
        if (cbr[7]) cmask_rt[uint64_t(cbr[7]) << 8] = base;
        // A fill helper that ran on this memory before it became a target (CPU fill of plain memory, image_cs_op) set its first contents:
        // a new target starts with that colour, not zero (the characters' blood maps: zero alpha turned bloodied fur black, ROADMAP 97).
        const auto mf = !im->initialised ? mem_fill.find(base) : mem_fill.end();
        const bool fast = fast_clear_pending.erase(base) > 0;
        if (mf != mem_fill.end() || fast) {  // CMASK fast clear: every tile reads as the clear colour (CB_COLOR*_CLEAR_WORD) until a draw covers it
            VkClearColorValue cc{};
            if (mf != mem_fill.end() && !fast) {
                const std::array<float, 4>& c = mf->second;  // (memory order; sRGB targets: ponytail, 0 and 1 only are exact)
                const float rgba[4] = {bgra ? c[2] : c[0], c[1], bgra ? c[0] : c[2], c[3]};
                std::memcpy(cc.float32, rgba, 16);
                mem_fill.erase(mf);
            } else if ((cbr[11] | cbr[12]) && (fmt == VK_FORMAT_R8G8B8A8_UNORM || fmt == VK_FORMAT_R8G8B8A8_SRGB))  // ponytail: only 8-bit RGBA clear words are decoded, other formats clear to 0
                for (unsigned ch = 0; ch < 4; ++ch) cc.float32[ch] = float((cbr[11] >> (8 * ch)) & 0xFF) / 255.f;
            ensure_init(im);
            note_write(im);
            end_pass();
            barrier();
            const VkImageSubresourceRange rg{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            ts_label("clear: fast colour");
            vkCmdClearColorImage(rcb(), im->image, VK_IMAGE_LAYOUT_GENERAL, &cc, 1, &rg);
            barrier();
            ++fast_clears;
        }
        targets.push_back({im, k, layers > 1 ? slice : 0});
        ++im->version;
        rt_w = std::min(rt_w, pitch_px);
        rt_h = std::min(rt_h, height);
    }
    const bool use_z = zfmt && (dctl & 2), use_s = sfmt && (dctl & 1);
    Image* dsimg = nullptr;
    if (use_z || use_s) {
        dsimg = ds_image();
        if (!dsimg) { ++skipped, ++skip_by["ds_alloc"]; return; }
        rt_w = std::min(rt_w, dsimg->gw());
        rt_h = std::min(rt_h, dsimg->gh());
    }
    // BB_RES_SCALE: all attachments of a draw share one scale. A native one bound with a scaled one is recreated scaled once (contents
    // lost, ROADMAP 88: 1024x576 colour + 1024x640 depth) and stays scaled (`promoted`).
    uint32_t tsc = dsimg ? dsimg->scale : 1;
    for (const Rt& t : targets) tsc = std::max(tsc, t.im->scale);
    if (tsc > 1) {
        for (Rt& t : targets)
            if (t.im->scale < tsc) {
                Image* o = t.im;
                promoted.insert(o->base);
                const bool bgra = o->bgra;
                if (!(t.im = get_rt(o->base, o->gw(), o->gh(), o->format, o->layers, tsc))) { ++skipped, ++skip_by["rt_alloc"]; return; }
                t.im->bgra = bgra, ++t.im->version;
                if (verbose) log_once("promo" + std::to_string(t.im->base), "render target promoted to scale " + std::to_string(tsc));
            }
        if (dsimg && dsimg->scale < tsc) {
            const uint64_t sb = dsimg->sbase;
            promoted.insert(dsimg->base);
            if (verbose) log_once("promo" + std::to_string(dsimg->base), "depth buffer promoted to scale " + std::to_string(tsc));
            if (!(dsimg = get_ds(dsimg->base, dsimg->gw(), dsimg->gh(), dsimg->format, tsc))) { ++skipped, ++skip_by["ds_alloc"]; return; }
            dsimg->sbase = sb;
        }
    }
    const uint32_t guest_w = rt_w, guest_h = rt_h;  // (guest pixels: viewport and scissor registers)
    if (rt_w != ~0u) rt_w *= tsc, rt_h *= tsc;
    g_ns_early += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(tnow() - sec_t).count());
    sec_t = tnow();
    if (targets.empty() && !dsimg) {  // neither colour nor depth/stencil output
        ++skipped, ++skip_by["no_target"];
        char b[600];
        std::snprintf(b, sizeof b, "draw without colour/depth target skipped (target_mask=0x%x dctl=0x%x zinfo=0x%x sinfo=0x%x zwrite=0x%x zsize=0x%x ps=%s vs=%s rctl=0x%x dovr=0x%x dshader=0x%x topo=%u cbctl=0x%x cb0: base=0x%x info=0x%x dcc=0x%x cmask=0x%x fmask=0x%x clear=%08x:%08x)", r.context[kCbTargetMask], dctl, r.context[kDbZInfo], r.context[CTX(0x28044)], r.context[CTX(0x28050)], zsize, ps.e ? ps.e->log_name.c_str() : "none", vs.e->log_name.c_str(), rctl, r.context[CTX(0x2800c)], r.context[CTX(0x2880c)], unsigned(topo), r.context[CTX(0x28808)], r.context[kCbColor0Base], r.context[kCbColor0Base + 4], r.context[kCbColor0Base + 6], r.context[kCbColor0Base + 7], r.context[kCbColor0Base + 9], r.context[kCbColor0Base + 11], r.context[kCbColor0Base + 12]);
        if (verbose) log_once(b, b);
        return;
    }

    VkCtx& c = vk();
    // pipeline
    // pipeline key: a 64-bit hash of everything that goes into the pipeline (shaders, topology, face state, depth/stencil, per-target format/blend/mask)
    uint64_t key = 0x9E3779B97F4A7C15ull;
    auto mix = [&key](uint64_t v) { key = (key ^ v) * 0xFF51AFD7ED558CCDull; key ^= key >> 32; };
    mix(reinterpret_cast<uintptr_t>(vs.e));
    mix(reinterpret_cast<uintptr_t>(ps.e));
    mix(uint64_t(topo) | uint64_t(rect) << 8 | uint64_t(r.context[kPaSuScModeCntl] & 7) << 9);
    if (dsimg) { mix(uint64_t(use_z) | uint64_t(use_s) << 1 | uint64_t(dsfmt) << 2); mix(dctl); mix(sctl); }
    for (const Rt& t : targets) {
        mix(uint64_t(t.im->format));
        mix(uint64_t(r.context[kCbBlend0 + t.slot]) | uint64_t((r.context[kCbTargetMask] >> (4 * t.slot)) & 0xF) << 32);
    }
    const DescLayout& dl = desc_layout(vs.e, ps.e, false);
    const VkPipelineLayout layout = dl.layout;
    VkPipeline pipe = VK_NULL_HANDLE;
    auto pit = gpipelines.find(key);
    if (pit != gpipelines.end()) pipe = pit->second;
    else {
        std::vector<VkFormat> fmts;
        std::vector<VkPipelineColorBlendAttachmentState> blend;
        for (const Rt& t : targets) {
            const uint32_t bc = r.context[kCbBlend0 + t.slot], mask = (r.context[kCbTargetMask] >> (4 * t.slot)) & 0xF;
            fmts.push_back(t.im->format);
            VkPipelineColorBlendAttachmentState a{};
            a.colorWriteMask = mask;
            if (bc & (1u << 30)) {
                a.blendEnable = VK_TRUE;
                a.srcColorBlendFactor = blend_factor(bc & 31);
                a.colorBlendOp = blend_op((bc >> 5) & 7);
                a.dstColorBlendFactor = blend_factor((bc >> 8) & 31);
                if (bc & (1u << 29)) {
                    a.srcAlphaBlendFactor = blend_factor((bc >> 16) & 31);
                    a.alphaBlendOp = blend_op((bc >> 21) & 7);
                    a.dstAlphaBlendFactor = blend_factor((bc >> 24) & 31);
                } else {
                    a.srcAlphaBlendFactor = a.srcColorBlendFactor;
                    a.alphaBlendOp = a.colorBlendOp;
                    a.dstAlphaBlendFactor = a.dstColorBlendFactor;
                }
            }
            blend.push_back(a);
        }
        std::vector<VkPipelineShaderStageCreateInfo> stages;
        VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        s.pName = "main";
        s.stage = VK_SHADER_STAGE_VERTEX_BIT; s.module = vs.e->module; stages.push_back(s);
        if (rect) { s.stage = VK_SHADER_STAGE_GEOMETRY_BIT; s.module = get_rect_gs(vs.e->tr.vs_params); stages.push_back(s); }
        if (ps.e) { s.stage = VK_SHADER_STAGE_FRAGMENT_BIT; s.module = ps.e->module; stages.push_back(s); }
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = topo;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        const uint32_t mode = r.context[kPaSuScModeCntl];
        rs.cullMode = rect ? VK_CULL_MODE_NONE : (mode & 1 ? VK_CULL_MODE_FRONT_BIT : 0) | (mode & 2 ? VK_CULL_MODE_BACK_BIT : 0);
        rs.frontFace = (mode & 4) ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        if (dsimg) {
            // GCN compare functions (0 never .. 7 always) and VkCompareOp share numbering
            auto sop = [](uint32_t v) {
                switch (v) {
                    case 1: return VK_STENCIL_OP_ZERO;
                    case 2: case 3: case 4: return VK_STENCIL_OP_REPLACE;  // ponytail: ONES approximated as REPLACE, REPLACE_OP uses the op value as reference
                    case 5: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
                    case 6: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
                    case 7: return VK_STENCIL_OP_INVERT;
                    case 8: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
                    case 9: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
                    default: return VK_STENCIL_OP_KEEP;  // keep and the bitwise ops (10..15, unused so far)
                }
            };
            ds.depthTestEnable = use_z;
            ds.depthWriteEnable = use_z && (dctl & 4);
            ds.depthCompareOp = VkCompareOp((dctl >> 4) & 7);
            if (static const bool z_off = std::getenv("BB_ZTEST_OFF") != nullptr; z_off && !(dctl & 4)) ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;  // debug: read-only depth tests always pass
            ds.stencilTestEnable = use_s;
            VkStencilOpState f{};
            f.compareOp = VkCompareOp((dctl >> 8) & 7);
            f.failOp = sop(sctl & 15); f.passOp = sop((sctl >> 4) & 15); f.depthFailOp = sop((sctl >> 8) & 15);
            VkStencilOpState bk = f;
            if (dctl & 0x80) {
                bk.compareOp = VkCompareOp((dctl >> 20) & 7);
                bk.failOp = sop((sctl >> 12) & 15); bk.passOp = sop((sctl >> 16) & 15); bk.depthFailOp = sop((sctl >> 20) & 15);
            }
            ds.front = f; ds.back = bk;
        }
        VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cbs.attachmentCount = uint32_t(blend.size());
        cbs.pAttachments = blend.data();
        const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                                      VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE};
        VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dsi.dynamicStateCount = 6;
        dsi.pDynamicStates = dyn;
        VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rci.colorAttachmentCount = uint32_t(fmts.size());
        rci.pColorAttachmentFormats = fmts.data();
        rci.depthAttachmentFormat = use_z ? dsfmt : VK_FORMAT_UNDEFINED;
        rci.stencilAttachmentFormat = use_s ? dsfmt : VK_FORMAT_UNDEFINED;
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.pNext = &rci;
        gp.stageCount = uint32_t(stages.size());
        gp.pStages = stages.data();
        gp.pVertexInputState = &vi; gp.pInputAssemblyState = &ia; gp.pViewportState = &vp; gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms; gp.pDepthStencilState = &ds; gp.pColorBlendState = &cbs; gp.pDynamicState = &dsi;
        gp.layout = layout;
        if (!vk_check(vkCreateGraphicsPipelines(c.device, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe), "graphics pipeline")) { ++skipped, ++skip_by["other"]; return; }
        gpipelines[key] = pipe;
    }

    // A ring wrap flushes (ends the command buffer, resets the descriptor pool): never let it happen between this draw's descriptor
    // writes and its draw call. 96 MB covers the buffer copies (<= 64 MB each, usually none) and the index copy of one draw.
    if (slot_end() - ring_used < (size_t(96) << 20)) flush();
    g_ns_pipe += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(tnow() - sec_t).count());
    sec_t = tnow();
    // descriptors
    // vertex records this draw can reach by vertex id: highest index + 1 (indexed) or the vertex count
    cur_vtx_records = d.count;
    cur_vtx_first = 0;
    VkBuffer idx_buf = VK_NULL_HANDLE;  // cached device-local copy of this draw's indices (get_vertex_buffer); else they go through the ring
    VkDeviceSize idx_buf_off = 0;
    if (d.indexed) {
        const size_t n = size_t(d.count) * d.index_bytes;
        if (d.index_addr >= 0x10000 && (d.host_indices || (hooks().mem_valid && hooks().mem_valid(d.index_addr, n)))) {
            uint32_t mx = 0, mn = ~0u;
            // cache entries are whole 16-byte units (hashed in 8-byte words); ranges with pending GPU writes are refused by the cache
            static const bool idx_cache = !(std::getenv("BB_IDX_CACHE") && std::getenv("BB_IDX_CACHE")[0] == '0');
            const uint64_t n16 = (uint64_t(n) + 15) & ~uint64_t(15);
            IndexInfo ii;
            ii.bytes = d.index_bytes;
            ii.count = d.count;
            if (idx_cache && !d.host_indices && hooks().mem_valid(d.index_addr, n16) && get_vertex_buffer(d.index_addr, n16, idx_buf, idx_buf_off, &ii)) {
                mn = ii.mn;
                mx = ii.mx;
            } else {
                idx_buf = VK_NULL_HANDLE;
                if (!d.host_indices) settle_pending(d.index_addr, d.index_addr + n, 2);
                if (d.index_bytes == 4) index_minmax32(reinterpret_cast<const uint32_t*>(d.index_addr), d.count, mn, mx);
                else index_minmax16(reinterpret_cast<const uint16_t*>(d.index_addr), d.count, mn, mx);
            }
            cur_vtx_records = uint64_t(mx) + 1;
            // A mesh inside a big vertex pool: copy only [min, max] and draw with vertexOffset = -min (VertexIndex then runs from 0). Valid
            // when every vertex stream of the VS is addressed by the vertex id alone (not by instance id or a computed offset) and
            // there is a single instance.
            bool all_by_vertex_id = d.instances == 1 && mn != ~0u;
            for (const Resource& vr : vs.e->tr.resources)
                if (vr.type == Resource::Buffer && !vr.scalar && !vr.load_data && vr.vertex_use != 1) all_by_vertex_id = false;
            if (all_by_vertex_id) cur_vtx_first = mn;
        } else cur_vtx_records = ~0ull;
    }
    g_ns_idx += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(tnow() - sec_t).count());
    sec_t = tnow();
    // Everything that may flush (descriptor sets, ring space for streams and indices) happens before the first command of the draw: a flush
    // starts a new command buffer, so a pipeline/pass/set bound before it would be missing for the draw (objects dropped out of frames
    // whenever the ring wrapped at the index upload). A flush in the middle also leaves earlier ring data in the other slot: redo once in
    // the fresh slot. ponytail: a tessellation LS pass recorded before this point is not redone.
    if (cdump_on)
        const_dump(0, (uint64_t(r.sh[kShVsLo + 1]) << 40) | (uint64_t(r.sh[kShVsLo]) << 8), ps_addr, &vs, &ps, &r.sh[kShVsUser], &r.sh[kShPsUser],
                   uint64_t(r.context[kCbColor0Base]) << 8, uint64_t(r.context[CTX(0x28050)]) << 8, r.context[kCbTargetMask], d.count, d.instances);
    {  // debug BB_HWWATCH_VS=<vs hex>,<VS resource index>,<hex byte offset>,<hex key byte offset>,<key float>: for draws with that VS whose resource holds the
        // float <key> at the key offset, arm hardware write watchpoints (every thread, see crash.cpp) on <offset>; each such draw re-arms the last 4
        // distinct addresses, so a pool that rewrites the block at one of them within 4 draws is caught. Finds who writes e.g. one object's per-instance constants.
        static unsigned long long hv = 0, hi = 0, ho = 0, hko = 0;
        static float hkf = 0;
        static const bool hw_on = [] { const char* e = std::getenv("BB_HWWATCH_VS"); return e && std::sscanf(e, "%llx,%llu,%llx,%llx,%f", &hv, &hi, &ho, &hko, &hkf) == 5; }();
        if (hw_on && hooks().hw_watch && vs.e && ((uint64_t(r.sh[kShVsLo + 1]) << 40) | (uint64_t(r.sh[kShVsLo]) << 8)) == hv && hi < vs.words.size()) {
            static uint64_t hw_list[4] = {};
            static int hw_n = 0;
            const uint32_t* w = vs.words[hi].data();
            const uint64_t base = uint64_t(w[0]) | (uint64_t(w[1] & 0xFFF) << 32);
            float kv = 0;
            if (hooks().mem_valid && hooks().mem_valid(base + hko, 4)) std::memcpy(&kv, reinterpret_cast<const void*>(base + hko), 4);
            if (std::fabs(kv - hkf) < 1e-3f && std::find(hw_list, hw_list + hw_n, base + ho) == hw_list + hw_n) {
                if (hw_n == 4) { std::memmove(hw_list, hw_list + 1, 3 * sizeof hw_list[0]); --hw_n; }
                hw_list[hw_n++] = base + ho;
                hooks().hw_watch(hw_list, hw_n);
                std::fprintf(stderr, "gpu: hwwatch_vs frame %llu: watching %d address(es), newest 0x%llx\n", (unsigned long long)g_frame_counter.load(std::memory_order_relaxed), hw_n, (unsigned long long)(base + ho));
            }
        }
    }
    VkDescriptorSet set = VK_NULL_HANDLE;
    size_t idx_off = 0;
    bool idx_ok = false;
    struct OwnReset { size_t& r; ~OwnReset() { r = SIZE_MAX; } } own_reset{dirty_own};
    dirty_own = dirty_s[slot].size();  // a flush sets it to 0: the aborted attempt's ranges in the new slot count as own too
    op_unsafe = false;
    op_writes.clear();
    if (replay_on && replay_obj) {  // replay object interpolation: this draw's constant blocks (write_descriptors -> note_obj)
        const uint64_t okey = uint64_t(reinterpret_cast<uintptr_t>(vs.e)) * 0x9E3779B97F4A7C15ull ^ uint64_t(reinterpret_cast<uintptr_t>(ps.e)) * 0xC2B2AE3D27D4EB4Full ^
                              (uint64_t(d.count) << 20) ^ d.instances;
        odraws.push_back({okey, 0, uint32_t(osites.size()), 0});
        obj_capture = true;
        cur_instances = d.instances;
    }
    struct ObjEnd { Backend& b; ~ObjEnd() { b.end_obj_draw(); } } obj_end{*this};
    const size_t sites0 = sites.size();
    for (int attempt = 0;; ++attempt) {
        const uint64_t f0 = flushes;
        sampled.clear();
        if (census) census_res.clear();
        scale_bits[0] = scale_bits[1] = scale_bits[2] = 0;
        if (obj_capture && osites.size() > odraws.back().first) {  // a retry: drop the aborted attempt's sites
            oarena.resize(osites[odraws.back().first].at);
            osites.resize(odraws.back().first);
        }
        if (!prepare_descriptors(dl, vs, &ps, set)) { ++skipped, ++skip_by["other"]; return; }
        if (d.indexed && idx_buf) idx_ok = true;  // the cached copy (a device-local buffer, unaffected by flushes)
        else if (d.indexed) {
            ScopeNs it{idx_up_ns};
            const size_t n = size_t(d.count) * d.index_bytes;
            uint8_t* p = ring_alloc(n, 16, idx_off);
            idx_ok = p && d.index_addr >= 0x10000 && (d.host_indices || !hooks().mem_valid || hooks().mem_valid(d.index_addr, n));
            if (idx_ok) std::memcpy(p, reinterpret_cast<const void*>(d.index_addr), n);
        }
        if (flushes == f0) break;
        if (attempt) { ++skipped, ++skip_by["flush in draw"]; return; }
    }
    end_obj_draw();
    // The LS pass and its ring data (tess_idx/tess_lds) stay in the slot they were recorded in. After a flush in between, this draw reads
    // them from the other slot's ring half, which the next flush would reuse while this draw may still run: make that flush wait. After two
    // flushes that half has already been reused.
    if (tess && flushes - tess_f0 >= 2) { ++skipped, ++skip_by["tess flushed twice"]; return; }
    if (tess && flushes != tess_f0) { sync_next = true; ++tess_cross; }
    if (op_unsafe) cut(true);  // writes guest memory: a replay skips it (replay_tick)
    if (replay_carry) {  // carried state: what this draw reads (blending, depth/stencil tests) and writes
        for (const Rt& t : targets) {
            if ((r.context[kCbBlend0 + t.slot] >> 30) & 1) note_read(t.im);  // CB_BLEND*_CONTROL.ENABLE
            note_write(t.im);
        }
        if (dsimg) {
            if (use_s || (use_z && (dctl & 4))) note_read(dsimg), note_write(dsimg);
            else note_read(dsimg);
        }
        for (Image* w : op_writes) note_write(w);
    }
    g_ns_desc += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(tnow() - sec_t).count());
    sec_t = tnow();
    for (const Rt& t : targets) ensure_init(t.im);
    if (dsimg) ensure_init(dsimg);
    if (replay_warp) warp_draw(r, targets.empty() ? nullptr : targets[0].im, targets.empty() ? 0 : targets[0].slot, dsimg, use_z, sites0, ps.e ? ps.e->log_name.c_str() : "-");
    // Consecutive draws into the same attachments share one rendering instance (no per-draw barrier). A new instance (after a barrier)
    // starts when the attachments change, when a draw samples an image rendered into since the last barrier, or around a draw with UAV
    // side effects (storage images / written buffers).
    const bool uav = (vs.e && vs.e->has_uav) || (ps.e && ps.e->has_uav);
    bool same = pass.open && !uav && pass.n == targets.size() && pass.ds == (dsimg ? dsimg->view : VK_NULL_HANDLE) && pass.use_z == use_z && pass.use_s == use_s && pass.w == rt_w && pass.h == rt_h;
    for (size_t k = 0; same && k < targets.size(); ++k) same = pass.colors[k] == attach_view(targets[k].im, targets[k].layer);
    if (same)
        for (const Image* im : sampled)
            if (std::find(pass_written.begin(), pass_written.end(), im) != pass_written.end()) { same = false; break; }
    if (!same) {
        end_pass();
        // The barrier orders this pass after earlier work it depends on: anything but draws since the last barrier, attachments the
        // draws since then read or wrote (WAR/WAW), images they wrote that this draw samples (RAW). Otherwise the passes are independent.
        auto hit = [](const std::vector<const Image*>& v, const Image* im) { return std::find(v.begin(), v.end(), im) != v.end(); };
        bool dep = !bar_opt || other_since || uav;
        for (size_t k = 0; !dep && k < targets.size(); ++k) dep = hit(pass_read, targets[k].im);
        if (!dep && dsimg) dep = hit(pass_read, dsimg);
        for (size_t k = 0; !dep && k < sampled.size(); ++k) dep = hit(pass_written, sampled[k]);
        if (dep) barrier();
        else ++pass_barriers_skipped;
        for (const Rt& t : targets) if (!hit(pass_read, t.im)) pass_read.push_back(t.im);
        if (dsimg && !hit(pass_read, dsimg)) pass_read.push_back(dsimg);
        std::vector<VkRenderingAttachmentInfo> atts;
        for (const Rt& t : targets) {
            VkRenderingAttachmentInfo a{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            a.imageView = attach_view(t.im, t.layer);
            a.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            atts.push_back(a);
        }
        VkRenderingAttachmentInfo dsa{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        if (dsimg) { dsa.imageView = dsimg->view; dsa.imageLayout = VK_IMAGE_LAYOUT_GENERAL; dsa.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; dsa.storeOp = VK_ATTACHMENT_STORE_OP_STORE; }
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {rt_w, rt_h}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = uint32_t(atts.size());
        ri.pColorAttachments = atts.data();
        ri.pDepthAttachment = use_z ? &dsa : nullptr;
        ri.pStencilAttachment = use_s ? &dsa : nullptr;
        if (ts_every) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "pass %ux%u c%zu f%d%s ps=", rt_w, rt_h, targets.size(), targets.empty() ? 0 : int(targets[0].im->format), use_z ? (use_s ? " zs" : " z") : use_s ? " s" : "");
            ts_mark(buf + (ps.e ? ps.e->log_name : std::string("nops")));
        }
        vkCmdBeginRendering(cb, &ri);
        pass.open = true;
        pass.n = uint32_t(targets.size());
        for (size_t k = 0; k < targets.size() && k < 8; ++k) pass.colors[k] = attach_view(targets[k].im, targets[k].layer);
        pass.ds = dsimg ? dsimg->view : VK_NULL_HANDLE;
        pass.use_z = use_z; pass.use_s = use_s; pass.w = rt_w; pass.h = rt_h;
    }
    if (prof) prof_begin(vs.e->log_name + "/" + (ps.e ? ps.e->log_name : "nops"));  // the name is a heap string: only when profiling
    if (pipe != rec.pipe) { vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); rec.pipe = pipe; }
    // viewport: window = scale * ndc + offset (y may be flipped by a negative scale)
    auto fl = [&](uint32_t i) { float f; std::memcpy(&f, &r.context[i], 4); return f; };
    const float xs = fl(kPaClVportXscale), xo = fl(kPaClVportXscale + 1), ys = fl(kPaClVportXscale + 2), yo = fl(kPaClVportXscale + 3), zs = fl(kPaClVportXscale + 4), zo = fl(kPaClVportXscale + 5);
    const float fs = float(tsc);  // guest pixels -> target pixels (BB_RES_SCALE)
    VkViewport vpt{(xo - xs) * fs, (yo - ys) * fs, 2 * xs * fs, 2 * ys * fs, zo, zo + zs};
    if (!(xs != 0 && ys != 0)) vpt = {0, 0, float(rt_w), float(rt_h), 0, 1};
    vpt.minDepth = std::clamp(std::min(zo, zo + zs), 0.0f, 1.0f);
    vpt.maxDepth = std::clamp(std::max(zo, zo + zs), 0.0f, 1.0f);
    const VkRect2D sc = phys_rect(eff_scissor(guest_w, guest_h), tsc);
    float bc[4];
    for (int i = 0; i < 4; ++i) std::memcpy(&bc[i], &r.context[kCbBlendRed + i], 4);
    if (!rec.dyn || std::memcmp(&vpt, &rec.vp, sizeof vpt)) { vkCmdSetViewport(cb, 0, 1, &vpt); rec.vp = vpt; }
    if (!rec.dyn || std::memcmp(&sc, &rec.sc, sizeof sc)) { vkCmdSetScissor(cb, 0, 1, &sc); rec.sc = sc; }
    if (!rec.dyn || std::memcmp(bc, rec.bc, sizeof bc)) { vkCmdSetBlendConstants(cb, bc); std::memcpy(rec.bc, bc, sizeof bc); }
    rec.dyn = true;
    if (dsimg) {  // TESTVAL [7:0], MASK [15:8], WRITEMASK [23:16], OPVAL [31:24]; ponytail: a REPLACE_OP stencil op takes the op value as reference
        const uint32_t sr[2] = {r.context[CTX(0x28430)], (dctl & 0x80) ? r.context[CTX(0x28434)] : r.context[CTX(0x28430)]};
        const VkStencilFaceFlags faces[2] = {VK_STENCIL_FACE_FRONT_BIT, VK_STENCIL_FACE_BACK_BIT};
        for (int i = 0; i < 2; ++i) {
            const uint32_t ops = i == 0 ? sctl : ((dctl & 0x80) ? sctl >> 12 : sctl);
            const bool op_val = ((ops & 15) == 4) || (((ops >> 4) & 15) == 4) || (((ops >> 8) & 15) == 4);
            const uint32_t ref = op_val ? sr[i] >> 24 : sr[i] & 0xFF, cmp = (sr[i] >> 8) & 0xFF, wr = (sr[i] >> 16) & 0xFF;
            uint32_t* st = &rec.stencil[3 * i];
            if (st[0] != ref) vkCmdSetStencilReference(cb, faces[i], st[0] = ref);
            if (st[1] != cmp) vkCmdSetStencilCompareMask(cb, faces[i], st[1] = cmp);
            if (st[2] != wr) vkCmdSetStencilWriteMask(cb, faces[i], st[2] = wr);
        }
    }
    bind_descriptors(VK_PIPELINE_BIND_POINT_GRAPHICS, dl, set);
    if (census) {  // BB_SCALE_LOG: this draw's targets and images
        std::string a;
        char cls = 0;
        bool mixed = false;
        auto att = [&](const Image* im) { a += ' ' + census_name(im); const char c = scale_class(im->gw(), im->gh()); mixed |= cls && c != cls; cls = c; };
        for (const Rt& t : targets) att(t.im);
        if (dsimg) att(dsimg);
        const bool fc = ps.e && (r.context[kSpiPsInputAddr] & 0x300);  // POS_X/Y_FLOAT: the PS reads FragCoord
        std::fprintf(stderr, "scale: draw %llu vs=%s ps=%s fc=%d targets%s | images%s\n", (unsigned long long)draws, vs.e->log_name.c_str(), ps.e ? ps.e->log_name.c_str() : "none", int(fc), a.c_str(), census_res.c_str());
        ++census_sum[std::string("draws into class ") + cls + " targets"];
        if (mixed) ++census_sum["attachments of mixed class:" + a];
        if (fc) ++census_sum["PS reading FragCoord: " + ps.e->log_name + " into class " + cls];
    }
    if (tsc > 1) scale_bits[uint32_t(ShStage::PS)] |= 1u << 31;  // (the PS maps FragCoord and its texel accesses to guest pixels)
    push_user(layout, VK_SHADER_STAGE_VERTEX_BIT, 0, &r.sh[kShVsUser], ShStage::VS);
    if (ps.e) push_user(layout, VK_SHADER_STAGE_FRAGMENT_BIT, push_fs(), &r.sh[kShPsUser], ShStage::PS);
    if (d.indexed) {
        if (idx_ok) {
            const VkBuffer ib = idx_buf ? idx_buf : ring.buffer;
            const VkDeviceSize ib_off = idx_buf ? idx_buf_off : idx_off;
            const VkIndexType ib_type = d.index_bytes == 4 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
            if (ib != rec.ib || ib_off != rec.ib_off || ib_type != rec.ib_type) { vkCmdBindIndexBuffer(cb, ib, ib_off, ib_type); rec.ib = ib, rec.ib_off = ib_off, rec.ib_type = ib_type; }
            vkCmdDrawIndexed(cb, d.count, d.instances, 0, -int32_t(cur_vtx_first), 0);
        }
    } else {
        vkCmdDraw(cb, d.count, d.instances, 0, 0);
    }
    fenced = false, crop_last = false;
    for (const Image* im : sampled) if (std::find(pass_read.begin(), pass_read.end(), im) == pass_read.end()) pass_read.push_back(im);
    ++ts_draws;
    prof_end();
    ++rendered;
    if (uav) {  // side effects other than the attachments: no batching around this draw
        end_pass();
        barrier();
    } else {
        for (const Rt& t : targets) if (std::find(pass_written.begin(), pass_written.end(), t.im) == pass_written.end()) pass_written.push_back(t.im);
        if (dsimg && ((use_z && (dctl & 4)) || use_s) && std::find(pass_written.begin(), pass_written.end(), dsimg) == pass_written.end()) pass_written.push_back(dsimg);
    }
    if (op_unsafe) cut(false);
    if (px_on) { char hb[160]; std::snprintf(hb, sizeof hb, " count=%u inst=%u blend0=%08x mask=%08x cb0=%llx pitch=%x slice=%x info=%x", d.count, d.instances, r.context[kCbBlend0], r.context[kCbTargetMask], (unsigned long long)(uint64_t(r.context[kCbColor0Base]) << 8), r.context[kCbColor0Base + 1], r.context[kCbColor0Base + 2], r.context[kCbColor0Base + 4]); pixel_probe(std::string("draw vs=") + vs.e->log_name + " ps=" + (ps.e ? ps.e->log_name : "none") + hb); }
    g_ns_rec += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(tnow() - sec_t).count());
    // BB_GPU_DUMP_AT="a-b": dump the first colour target after recorded draws a..b (raw BGRA/RGBA, see BB_GPU_DUMP dir)
    static const char* at = std::getenv("BB_GPU_DUMP_AT");
    static const char* ddir = std::getenv("BB_GPU_DUMP");
    static unsigned lo = 0, hi = 0;
    static bool parsed = false;
    if (!parsed) { parsed = true; if (at) std::sscanf(at, "%u-%u", &lo, &hi); }
    const bool dbg_draw = (at && rendered >= lo && rendered <= hi) || rt_hit;
    if (dbg_draw) {
        std::fprintf(stderr, "gpu:   ps user:");
        for (int k = 0; k < 16; ++k) std::fprintf(stderr, " %08x", r.sh[0x0C + k]);
        std::fprintf(stderr, "\n");
        const uint64_t tab = (uint64_t(r.sh[0x0C + 3] & 0xFFFF) << 32) | r.sh[0x0C + 2];
        if (tab > 0x10000 && hooks().mem_valid && hooks().mem_valid(tab, 128)) {
            std::fprintf(stderr, "gpu:   table@%llx:", (unsigned long long)tab);
            for (int k = 0; k < 32; ++k) std::fprintf(stderr, " %08x", reinterpret_cast<const uint32_t*>(tab)[k]);
            std::fprintf(stderr, "\n");
        }
    }
    if (dbg_draw && d.indexed)
        std::fprintf(stderr, "gpu:   index addr %llx bytes %u count %u valid=%d first: %04x %04x %04x\n", (unsigned long long)d.index_addr, d.index_bytes, d.count,
                     int(d.index_addr >= 0x10000 && hooks().mem_valid && hooks().mem_valid(d.index_addr, size_t(d.count) * d.index_bytes)),
                     d.index_addr >= 0x10000 && hooks().mem_valid && hooks().mem_valid(d.index_addr, 6) ? reinterpret_cast<const uint16_t*>(d.index_addr)[0] : 0xFFFF,
                     d.index_addr >= 0x10000 && hooks().mem_valid && hooks().mem_valid(d.index_addr, 6) ? reinterpret_cast<const uint16_t*>(d.index_addr)[1] : 0xFFFF,
                     d.index_addr >= 0x10000 && hooks().mem_valid && hooks().mem_valid(d.index_addr, 6) ? reinterpret_cast<const uint16_t*>(d.index_addr)[2] : 0xFFFF);
    if (dbg_draw)
        std::fprintf(stderr, "gpu:   depth: dctl %08x sctl %08x sref %08x/%08x zinfo %08x sinfo %08x zread %08x sread %08x zwrite %08x swrite %08x size %08x rctl %08x view %08x dinfo %08x clear z %08x s %08x\n",
                     r.context[kDbDepthControl], r.context[CTX(0x2842c)], r.context[CTX(0x28430)], r.context[CTX(0x28434)], r.context[kDbZInfo], r.context[CTX(0x28044)], r.context[CTX(0x28048)],
                     r.context[CTX(0x2804c)], r.context[CTX(0x28050)], r.context[CTX(0x28054)], r.context[CTX(0x28058)], r.context[CTX(0x28000)], r.context[CTX(0x28008)], r.context[CTX(0x2803c)],
                     r.context[CTX(0x2802c)], r.context[CTX(0x28028)]);
    (void)0;
    g_log_tex = false;
    if (dbg_draw)
        std::fprintf(stderr, "gpu:   scissors: screen %08x/%08x winoff %08x win %08x/%08x gen %08x/%08x vp0 %08x/%08x cliprule %08x rect0 %08x/%08x mode0 %08x vpt xs=%08x ys=%08x zs=%08x zo=%08x\n",
                     r.context[CTX(0x28030)], r.context[CTX(0x28034)], r.context[CTX(0x28200)], r.context[CTX(0x28204)], r.context[CTX(0x28208)], r.context[CTX(0x28240)], r.context[CTX(0x28244)],
                     r.context[CTX(0x28250)], r.context[CTX(0x28254)], r.context[CTX(0x2820c)], r.context[CTX(0x28210)], r.context[CTX(0x28214)], r.context[CTX(0x28a48)],
                     r.context[kPaClVportXscale], r.context[kPaClVportXscale + 2], r.context[kPaClVportXscale + 4], r.context[kPaClVportXscale + 5]);
    if (dbg_draw)
        std::fprintf(stderr, "gpu:   targets=%zu blend0=%08x mask=%08x scmode=%08x cbcolor0 %08x/%08x/%08x\n", targets.size(), r.context[kCbBlend0], r.context[kCbTargetMask], r.context[kPaSuScModeCntl],
                     r.context[kCbColor0Base], r.context[kCbColor0Base + 4], r.context[kCbColor0Base + 5]);
    if (dbg_draw)
        std::fprintf(stderr, "gpu: draw %llu: vs=%llx ps=%llx n=%u inst=%u idx=%d rt=%llx\n", (unsigned long long)rendered,
                     (unsigned long long)((uint64_t(r.sh[kShVsLo + 1]) << 40) | (uint64_t(r.sh[kShVsLo]) << 8)), (unsigned long long)((uint64_t(r.sh[kShPsLo + 1]) << 40) | (uint64_t(r.sh[kShPsLo]) << 8)), d.count, d.instances, int(d.indexed),
                     (unsigned long long)(targets.empty() ? 0 : targets[0].im->base));
    // BB_DUMP_IMG=<hex base>[,<hex base>...]: with BB_LOG_RT/BB_LOG_PS and BB_GPU_DUMP, also dump those render targets after every logged draw
    static const std::vector<uint64_t> dump_imgs = [] {
        std::vector<uint64_t> v;
        if (const char* e = std::getenv("BB_DUMP_IMG")) for (const char* c = e; *c;) { char* end = nullptr; v.push_back(std::strtoull(c, &end, 16)); if (end == c) break; c = *end == ',' ? end + 1 : end; }
        return v;
    }();
    if (rt_hit && ddir) for (const uint64_t dump_img : dump_imgs) {
        if (auto it = rts.find(dump_img); it != rts.end()) {
            Image* im = it->second.get();
            const size_t bpp4 = im->format == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 4, n = size_t(im->w) * im->h * bpp4;
            size_t off = 0;
            uint8_t* p = ring_alloc(n, 16, off);
            if (p) {
                VkBufferImageCopy copy{off, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {im->w, im->h, 1}};
                barrier();
                vkCmdCopyImageToBuffer(rcb(), im->image, VK_IMAGE_LAYOUT_GENERAL, ring.buffer, 1, &copy);
                barrier();
                const size_t roff = off;
                flush();
                char path[400];
                std::snprintf(path, sizeof path, "%s/img_%llu_%llx_%ux%u_%d.raw", ddir, (unsigned long long)rendered, (unsigned long long)dump_img, im->w, im->h, int(im->format));
                if (FILE* f = std::fopen(path, "wb")) { std::fwrite(ring.mapped + roff, 1, n, f); std::fclose(f); }
                std::fprintf(stderr, "gpu: dumped %llx: same image as target0 = %d (targets=%zu, target0 base %llx %ux%u fmt %d)\n", (unsigned long long)dump_img, int(!targets.empty() && targets[0].im == im), targets.size(),
                             (unsigned long long)(targets.empty() ? 0 : targets[0].im->base), targets.empty() ? 0 : targets[0].im->w, targets.empty() ? 0 : targets[0].im->h, targets.empty() ? 0 : int(targets[0].im->format));
            }
        }
    }
    if (at && ddir && !targets.empty() && rendered >= lo && rendered <= hi) {
        Image* im = targets[0].im;
        const size_t n = size_t(im->w) * im->h * 4;
        size_t off = 0;
        uint8_t* p = ring_alloc(n, 16, off);
        if (p && im->format != VK_FORMAT_R16G16B16A16_SFLOAT) {
            VkBufferImageCopy copy{};
            copy.bufferOffset = off;
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {im->w, im->h, 1};
            vkCmdCopyImageToBuffer(rcb(), im->image, VK_IMAGE_LAYOUT_GENERAL, ring.buffer, 1, &copy);
            barrier();
            const size_t roff = off;
            flush();
            char path[400];
            std::snprintf(path, sizeof path, "%s/draw_%llu_%llx_%ux%u.raw", ddir, (unsigned long long)rendered, (unsigned long long)im->base, im->w, im->h);
            if (FILE* f = std::fopen(path, "wb")) { std::fwrite(ring.mapped + roff, 1, n, f); std::fclose(f); }
        }
    }
}

// Fill/copy helper shaders on render-target / depth memory. Returns false when the buffers are plain memory (normal compute runs).
bool Backend::image_cs_op(const BoundShader& cs, uint32_t x, uint32_t local_x) {
    const uint32_t* a = cs.words[0].data();
    const uint32_t* b = cs.words[1].data();
    auto vbase = [](const uint32_t* w) { return uint64_t(w[0]) | (uint64_t(w[1] & 0xFFF) << 32); };
    auto vstride = [](const uint32_t* w) { return (w[1] >> 16) & 0x3FFF; };
    auto bpp = [](VkFormat f) -> uint32_t {
        switch (f) {
            case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_R32_SFLOAT: return 4;
            case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
            case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return 4;
            default: return 0;
        }
    };
    if (cs.e->kind == 3) {
        // Gnm "copy dwords by job list" CS (cs@10d5e00100): workgroup g reads job g = {dst element, src element, last} from words[0]
        // and copies elements 0..last (v_cmp_ge_u32 last, tid: last + 1 of them, one per thread) from words[1] to words[2]. Executed here,
        // at record time, like the other memory helpers: on the GPU it read its source live, much later than the console would (per-draw
        // constants came out partly zero: rooms vanished, ROADMAP 48), and its 64 MB destination V# marked that whole range as GPU-written
        // (a read flush for every buffer in it). Returns false (GPU path) for anything unexpected.
        if (cs.words.size() < 3) return false;
        const uint32_t* jw = cs.words[0].data();
        const uint32_t* sw = cs.words[1].data();
        const uint32_t* dw = cs.words[2].data();
        if (vstride(jw) != 12 || vstride(sw) != 4 || vstride(dw) != 4 || !hooks().mem_valid) return false;
        const uint64_t jb = vbase(jw), sb = vbase(sw), db = vbase(dw);
        if (rts.count(db) || dss.count(db) || rts.count(sb) || dss.count(sb) || !hooks().mem_valid(jb, uint64_t(x) * 12)) return false;
        settle_pending(jb, jb + uint64_t(x) * 12);
        for (uint32_t g = 0; g < x && g < jw[2]; ++g) {
            uint32_t job[3];
            std::memcpy(job, reinterpret_cast<const void*>(jb + uint64_t(g) * 12), 12);
            const uint64_t n = std::min<uint64_t>(uint64_t(job[2]) + 1, local_x);
            const uint64_t s = sb + uint64_t(job[1]) * 4, d = db + uint64_t(job[0]) * 4;
            const uint64_t nd = job[0] >= dw[2] ? 0 : std::min<uint64_t>(n, dw[2] - job[0]);  // the shader's bound: dst element < num_records
            if (!nd || !hooks().mem_valid(s, nd * 4) || !hooks().mem_valid(d, nd * 4)) continue;
            settle_pending(s, s + nd * 4);
            settle_pending(d, d + nd * 4);
            gpu_watch("job copy helper CS", d, nd * 4);
            std::memmove(reinterpret_cast<void*>(d), reinterpret_cast<const void*>(s), size_t(nd * 4));
            cpu_bytes += nd * 4;
        }
        ++cpu_ops;
        tex_touch(db, db + uint64_t(dw[2]) * 4);
        return true;
    }
    if (cs.e->kind == 1) {  // fill: words[0] = constant buffer (colour), words[1] = destination typed buffer
        const uint64_t dst = vbase(b), cbase = vbase(a);
        const uint32_t stride = vstride(b), dnfmt = (b[3] >> 12) & 7, per_thread = cs.e->code_hash == 0x32b79ba3d47cccb3ull ? 2 : 1;
        float col[4];
        if (!stride || !hooks().mem_valid || !hooks().mem_valid(cbase, 16)) return false;
        std::memcpy(col, reinterpret_cast<const void*>(cbase), 16);
        const uint64_t covered = uint64_t(x) * 64 * per_thread * stride;
        if (auto it = rts.find(dst); it != rts.end()) {
            Image* im = it->second.get();
            const uint32_t px = bpp(im->format);
            if (!px || (dnfmt != 0 && dnfmt != 7)) return false;
            const uint64_t full = uint64_t(im->gw()) * im->gh() * px;  // (guest bytes: `covered` counts guest elements)
            const bool zero = col[0] == 0.f && col[1] == 0.f && col[2] == 0.f && col[3] == 0.f;
            if (covered < full && !zero) return false;
            ensure_init(im);
            note_write(im);
            barrier();
            if (covered >= full) {
                VkClearColorValue cv{};
                const float rgba[4] = {im->bgra ? col[2] : col[0], col[1], im->bgra ? col[0] : col[2], col[3]};
                std::memcpy(cv.float32, rgba, 16);
                const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(rcb(), im->image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &range);
            } else {  // the dispatch only covers the first `covered` bytes (row-major): zeros copied from the ring's zero head
                const uint64_t pixels = covered / px;
                const uint32_t s = im->scale, rows = uint32_t(pixels / im->gw()) * s, rem = uint32_t(pixels % im->gw()) * s;  // (physical)
                std::vector<VkBufferImageCopy> regions;
                const uint32_t chunk = uint32_t(std::max<uint64_t>(1, kZeroBytes / (uint64_t(im->w) * px)));  // rows the zero head holds
                for (uint32_t y = 0; y < rows; y += chunk) regions.push_back({0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, int32_t(y), 0}, {im->w, std::min(chunk, rows - y), 1}});
                if (rem) regions.push_back({0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, int32_t(rows), 0}, {rem, s, 1}});
                if (!regions.empty()) vkCmdCopyBufferToImage(rcb(), ring.buffer, im->image, VK_IMAGE_LAYOUT_GENERAL, uint32_t(regions.size()), regions.data());
            }
            barrier();
            ++image_ops;
            ++im->version;
            return true;
        }
        for (auto& [k, up] : dss) {  // stencil plane fill (byte value in the first component's raw bits)
            Image* im = up.get();
            if (!im->sbase || im->sbase != dst || im->format == VK_FORMAT_D32_SFLOAT || covered < uint64_t(im->gw()) * im->gh()) continue;
            uint32_t raw;
            std::memcpy(&raw, &col[0], 4);
            ensure_init(im);
            note_write(im);
            barrier();
            VkClearDepthStencilValue sv{1.0f, raw & 0xFF};
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
            vkCmdClearDepthStencilImage(rcb(), im->image, VK_IMAGE_LAYOUT_GENERAL, &sv, 1, &range);
            barrier();
            ++image_ops;
            return true;
        }
        if (auto it = dss.find(dst); it != dss.end()) {  // depth fill (value in the first component)
            Image* im = it->second.get();
            if (covered < uint64_t(im->gw()) * im->gh() * 4) return false;
            ensure_init(im);
            note_write(im);
            barrier();
            VkClearDepthStencilValue dv{col[0], 0};
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            vkCmdClearDepthStencilImage(rcb(), im->image, VK_IMAGE_LAYOUT_GENERAL, &dv, 1, &range);
            barrier();
            ++image_ops;
            return true;
        }
        // plain memory: fill on the CPU (a GPU store to imported host memory crosses PCIe, far slower than a memset)
        {
            const uint32_t dfmt = (b[3] >> 15) & 15;
            static const uint8_t kComps[16] = {0, 1, 1, 2, 1, 2, 0, 0, 0, 0, 4, 2, 4, 3, 4, 0}, kWidth[16] = {0, 8, 16, 8, 32, 16, 0, 0, 0, 0, 8, 32, 16, 32, 32, 0};
            const uint32_t comps = kComps[dfmt], width = kWidth[dfmt], records = b[2];
            const uint32_t esz = comps * width / 8;
            if (!comps || (dnfmt != 0 && dnfmt != 7 && dnfmt != 4) || esz != stride || !hooks().mem_valid) {
                if (verbose) { char t3[200]; std::snprintf(t3, sizeof t3, "fill helper fallback: dfmt=%u nfmt=%u stride=%u esz=%u comps=%u V# %08x %08x %08x %08x", dfmt, dnfmt, stride, esz, comps, b[0], b[1], b[2], b[3]); log_once(t3, t3); }
                static uint64_t nfb = 0;
                if (verbose && ++nfb % 200 == 0) std::fprintf(stderr, "gpu: %llu fill-helper fallbacks so far (last V# %08x %08x, rts has it: %d, covered %llu bytes)\n", (unsigned long long)nfb, b[0], b[1], int(rts.count(dst)), (unsigned long long)covered);
                return false;
            }
            const uint64_t count = std::min<uint64_t>(records, uint64_t(x) * 64 * per_thread);
            if (!count || !hooks().mem_valid(dst, count * stride)) return false;
            uint8_t elem[16] = {};
            for (uint32_t k = 0; k < comps; ++k) {
                uint32_t raw;
                std::memcpy(&raw, &col[k], 4);  // uint formats take the raw integer bits
                uint32_t v;
                if (dnfmt == 7) v = width == 32 ? raw : f32_to_f16(col[k]);
                else if (dnfmt == 0) v = uint32_t(std::lrintf(std::clamp(col[k], 0.f, 1.f) * float(width == 32 ? 0xFFFFFFFFu : (1u << width) - 1)));
                else v = width == 32 ? raw : std::min<uint32_t>(raw, (1u << width) - 1);
                std::memcpy(elem + k * width / 8, &v, width / 8);
            }
            if (overlaps_dirty(dst, dst + count * stride)) flush();
            uint8_t* out = reinterpret_cast<uint8_t*>(dst);
            gpu_watch("fill helper CS", dst, count * stride);
            if (comps == 4 && (dnfmt == 0 || dnfmt == 7)) mem_fill[dst] = {col[0], col[1], col[2], col[3]};  // (a render target created here later)
            {  // one byte value: memset; else the first element, then doubling copies (a memcpy per element was ~8 % of the render thread)
                const uint64_t total = count * stride;
                bool same = true;
                for (uint32_t k = 1; k < stride && same; ++k) same = elem[k] == elem[0];
                if (same) std::memset(out, elem[0], size_t(total));
                else {
                    std::memcpy(out, elem, stride);
                    for (uint64_t done = stride; done < total;) { const uint64_t n = std::min(done, total - done); std::memcpy(out + done, out, size_t(n)); done += n; }
                }
            }
            if (!cmask_rt.empty())  // a zeroed CMASK = fast clear of its target, anything else (0xCC = expanded) cancels a pending one
                for (const auto& [cm, rtb] : cmask_rt)
                    if (cm >= dst && cm < dst + count * stride) {
                        bool zero = true;
                        for (uint32_t k = 0; k < stride && k < 16; ++k) zero = zero && elem[k] == 0;
                        if (zero) fast_clear_pending.insert(rtb); else fast_clear_pending.erase(rtb);
                    }
            for (const auto& [ht, dsb] : htile_ds)  // HTILE words with ZMASK (bits 3:0) 0 = tile cleared; anything else (0xF = expanded) cancels
                if (ht >= dst && ht < dst + count * stride) {
                    uint32_t word = 0;
                    std::memcpy(&word, elem, std::min<uint32_t>(stride, 4));
                    if (verbose) { char t3[96]; std::snprintf(t3, sizeof t3, "HTILE fill of depth 0x%llx with 0x%08x", (unsigned long long)dsb, word); log_once(t3, t3); }
                    if ((word & 0xF) == 0) htile_clear_pending.insert(dsb); else htile_clear_pending.erase(dsb);
                    if (static const bool ops = std::getenv("BB_OPS_LOG") != nullptr; ops) std::fprintf(stderr, "op: HTILE fill of depth %llx with %08x\n", (unsigned long long)dsb, word);
                }
            ++cpu_ops;
            tex_touch(dst, dst + count * stride);
            return true;
        }
    }
    if (cs.e->kind == 2) {  // copy: words[0] = source, words[1] = destination (same typed format)
        if (!rts.count(vbase(a)) && !rts.count(vbase(b)) && !dss.count(vbase(a)) && !dss.count(vbase(b)) && a[3] == b[3] && vstride(a) == vstride(b) && vstride(a)) {
            // plain memory to plain memory with identical typed formats: a byte copy on the CPU
            const uint64_t s = vbase(a), d = vbase(b), count = std::min<uint64_t>(std::min(a[2], b[2]), uint64_t(x) * 64), bytes = count * vstride(a);
            if (count && hooks().mem_valid && hooks().mem_valid(s, bytes) && hooks().mem_valid(d, bytes)) {
                if (overlaps_dirty(s, s + bytes) || overlaps_dirty(d, d + bytes)) flush();
                gpu_watch("copy helper CS", d, bytes);
                std::memmove(reinterpret_cast<void*>(d), reinterpret_cast<const void*>(s), bytes);
                ++cpu_ops;
                tex_touch(d, d + bytes);
                cpu_bytes += bytes;
                return true;
            }
        }
        auto s = rts.find(vbase(a)), d = rts.find(vbase(b));
        if (!dss.count(vbase(b)) && (d == rts.end() || d->second->snapshot)) {  // new snapshot, or a refresh of one taken earlier
            // The game snapshots a render/depth target into plain memory that later passes sample as a texture: keep the snapshot as an
            // image registered at the destination address (texture lookups alias render targets by address).
            if (s != rts.end() && d == rts.end()) {
                Image* si = s->second.get();
                if (Image* di = get_rt(vbase(b), si->gw(), si->gh(), si->format, 1, si->scale)) { di->bgra = si->bgra; di->snapshot = true; d = rts.find(vbase(b)); s = rts.find(vbase(a)); }
            } else if (auto ds = dss.find(vbase(a)); ds != dss.end() && ds->second->format != VK_FORMAT_S8_UINT) {
                Image* si = ds->second.get();
                size_t off = 0;
                const size_t bytes = size_t(si->w) * si->h * 4;
                uint8_t* p = ring_alloc(bytes, 16, off);  // may flush: before any command of this copy is recorded
                Image* di = p ? get_rt(vbase(b), si->gw(), si->gh(), VK_FORMAT_R32_SFLOAT, 1, si->scale) : nullptr;
                if (di && (di->w != si->w || di->h != si->h)) di = nullptr;  // (a refresh after the depth buffer was promoted: the next one fits)
                if (di) {
                    ensure_init(si);
                    ensure_init(di);
                    note_read(si);
                    note_write(di);
                    barrier();
                    VkBufferImageCopy dc{off, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {si->w, si->h, 1}};
                    vkCmdCopyImageToBuffer(rcb(), si->image, VK_IMAGE_LAYOUT_GENERAL, ring.buffer, 1, &dc);
                    barrier();
                    VkBufferImageCopy cc = dc;
                    cc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    vkCmdCopyBufferToImage(rcb(), ring.buffer, di->image, VK_IMAGE_LAYOUT_GENERAL, 1, &cc);
                    barrier();
                    di->snapshot = true;
                    ++di->version;
                    if (static const bool snap_log = std::getenv("BB_SNAP_LOG") != nullptr; snap_log) {  // debug: statistics of every depth snapshot (flushes)
                        const size_t roff = off;
                        flush();
                        const float* f = reinterpret_cast<const float*>(ring.mapped + roff);
                        float mn = 1e30f, mx = -1e30f;
                        size_t ones = 0;
                        for (size_t i = 0; i < bytes / 4; ++i) { mn = std::min(mn, f[i]); mx = std::max(mx, f[i]); ones += f[i] == 1.0f; }
                        std::fprintf(stderr, "gpu: depth snapshot 0x%llx -> 0x%llx: min %g max %g, %.1f%% at 1.0 (draws so far %llu)\n", (unsigned long long)si->base, (unsigned long long)vbase(b), mn, mx, 100.0 * double(ones) / double(bytes / 4), (unsigned long long)draws);
                    }
                    ++image_ops;
                    return true;
                }
            }
        }
        if (s == rts.end() || d == rts.end() || s->second->format != d->second->format || s->second->gw() != d->second->gw() || s->second->gh() != d->second->gh()) {
            if (verbose) {
                char t2[400];
                std::snprintf(t2, sizeof t2, "image copy not handled: src=0x%llx (%s) dst=0x%llx (%s) V#a %08x %08x %08x %08x V#b %08x %08x %08x %08x", (unsigned long long)vbase(a), rts.count(vbase(a)) ? "rt" : dss.count(vbase(a)) ? "ds" : "mem",
                              (unsigned long long)vbase(b), rts.count(vbase(b)) ? "rt" : dss.count(vbase(b)) ? "ds" : "mem", a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]);
                log_once(t2, t2);
            }
            return false;
        }
        ensure_init(s->second.get());
        ensure_init(d->second.get());
        note_read(s->second.get());
        note_write(d->second.get());
        barrier();
        ts_label("copy: CS helper image copy");
        if (s->second->w == d->second->w && s->second->h == d->second->h) {
            VkImageCopy cp{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {s->second->w, s->second->h, 1}};
            vkCmdCopyImage(rcb(), s->second->image, VK_IMAGE_LAYOUT_GENERAL, d->second->image, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
        } else {  // same guest size, other scale (BB_RES_SCALE: one side promoted)
            VkImageBlit bl{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {int32_t(s->second->w), int32_t(s->second->h), 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {int32_t(d->second->w), int32_t(d->second->h), 1}}};
            vkCmdBlitImage(rcb(), s->second->image, VK_IMAGE_LAYOUT_GENERAL, d->second->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
        }
        barrier();
        ++d->second->version;  // the copy changed the destination (padded-target crops / snapshots must be re-taken)
        ++image_ops;
        return true;
    }
    return false;
}

void Backend::dispatch(const RegView& r, uint32_t x, uint32_t y, uint32_t z) {
    ++dispatches;
    static const bool skip_cs = std::getenv("BB_SKIP_CS") != nullptr;  // diagnosis: where does GPU time go?
    if (skip_cs) return;
    std::string px_who = "dispatch";
    struct PxGuard { Backend* b; const std::string* w; ~PxGuard() { if (b->px_on) b->pixel_probe(*w); } } pxg{this, &px_who};  // BB_PIXEL: also look after compute work
    BoundShader& cs = bs_cs;
    get_shader(ShStage::CS, r, cs);
    if (px_on && cs.e) px_who = "dispatch " + cs.e->log_name + " " + std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z) + (cs.e->kind ? " (helper)" : "");
    if (!cs.e) { ++skipped, ++skip_by["other"]; return; }
    if (cs.e->kind && cs.words.size() >= 2 && image_cs_op(cs, x, r.sh[kShCsThreadX] & 0xFFFF)) {  // COMPUTE_NUM_THREAD_X[15:0] = group size
        if (census) ++census_sum["Gnm helper CS kind " + std::to_string(cs.e->kind) + " (1 fill, 2 copy, 3 job copy) done as image op / on the CPU"];
        ++dispatched;
        return;
    }
    if (!cs.e->tr.error.empty()) { ++skipped, ++skip_by["other"]; return; }
    static const bool cs_log = std::getenv("BB_CS_LOG") != nullptr;
    if (cs_log) std::fprintf(stderr, "gpu: dispatch %s %ux%ux%u local %u,%u,%u\n", cs.e->log_name.c_str(), x, y, z, r.sh[kShCsThreadX], r.sh[kShCsThreadX + 1], r.sh[kShCsThreadX + 2]);
    if (cdump_on) const_dump(1, (uint64_t(r.sh[kShCsLo + 1]) << 40) | (uint64_t(r.sh[kShCsLo]) << 8), 0, &cs, nullptr, &r.sh[kShCsUser], nullptr, 0, 0, 0, x * y * z, 1);
    record_compute(cs, &r.sh[kShCsUser], x, y, z);
}

void Backend::record_compute(const BoundShader& cs, const uint32_t* user, uint32_t x, uint32_t y, uint32_t z, bool ls) {
    VkCtx& c = vk();
    const DescLayout& dl = desc_layout(cs.e, nullptr, true);
    const VkPipelineLayout layout = dl.layout;
    const std::string key = "c" + std::to_string(reinterpret_cast<uintptr_t>(cs.e));
    VkPipeline pipe = VK_NULL_HANDLE;
    auto pit = pipelines.find(key);
    if (pit != pipelines.end()) pipe = pit->second;
    else {
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, cs.e->module, "main", nullptr};
        ci.layout = layout;
        if (!vk_check(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe), "compute pipeline")) { ++skipped, ++skip_by["other"]; return; }
        pipelines[key] = pipe;
    }
    VkDescriptorSet set{};
    struct OwnReset { size_t& r; ~OwnReset() { r = SIZE_MAX; } } own_reset{dirty_own};
    dirty_own = dirty_s[slot].size();
    op_unsafe = false;
    op_writes.clear();
    sampled.clear();
    for (int attempt = 0;; ++attempt) {  // a flush in the middle leaves earlier ring data in the other slot: redo once (see draw)
        if (census) census_res.clear();
        scale_bits[0] = scale_bits[1] = scale_bits[2] = 0;
        const uint64_t f0 = flushes;
        if (!prepare_descriptors(dl, cs, nullptr, set)) { ++skipped, ++skip_by["other"]; return; }
        if (flushes == f0) break;
        if (attempt) { ++skipped, ++skip_by["flush in draw"]; return; }
    }
    if (op_unsafe) cut(true);  // writes guest memory or GDS: a replay skips it (replay_tick)
    for (Image* w : op_writes) note_write(w);
    // An LS pass only reads (vertex streams, constants, images) and writes its fresh, CPU-zeroed LDS range: after nothing but draws
    // into attachments it does not sample, it needs no barrier before it (the one after it orders the DS draws).
    bool dep = !(ls && bar_opt && !other_since);
    for (size_t k = 0; !dep && k < sampled.size(); ++k) dep = std::find(pass_written.begin(), pass_written.end(), sampled[k]) != pass_written.end();
    if (dep) barrier();
    else end_pass(), ++ls_barriers_skipped;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    bind_descriptors(VK_PIPELINE_BIND_POINT_COMPUTE, dl, set);
    if (census) {  // BB_SCALE_LOG
        std::fprintf(stderr, "scale: %s %s grid %ux%ux%u%s | images%s\n", ls ? "ls" : "cs", cs.e->log_name.c_str(), x, y, z, cur_indirect.buf ? " (indirect)" : "", census_res.c_str());
        ++census_sum[ls ? "LS passes (tessellation)" : "dispatches"];
    }
    push_user(layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, user, cs.e->stage);  // (an LS pass is a VS-stage shader)
    prof_begin(cs.e->log_name);
    if (ts_every) ts_mark("cs " + cs.e->log_name);
    if (cur_indirect.buf) vkCmdDispatchIndirect(rcb(), cur_indirect.buf, cur_indirect.off);  // DISPATCH_INDIRECT with GPU-written arguments
    else vkCmdDispatch(rcb(), x, y, z);
    prof_end();
    ++dispatched;
    if (bar_opt) barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);  // chains with the full barrier before the dispatch
    else barrier();
    if (op_unsafe) cut(false);
}

VkBuffer Backend::tess_dev_lds() {
    VkBuffer& b = tess_dev[slot];
    if (b) return b;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size_t(16) << 20;  // tess_begin's largest LDS
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (!vk_check(vmaCreateBuffer(vk().vma, &bci, &aci, &b, &tess_dev_alloc[slot], nullptr), "tessellation LDS buffer")) b = VK_NULL_HANDLE;
    return b;
}

// Patch draw: runs the LS over all control points into the emulated LDS (compute pass) and turns the draw into an indexed triangle draw of the DS grid
// (instance = patch). ponytail: the HS itself is not run. The game's HS only stores constant tess factors, equal to the HS register block's max/min
// levels, so edge factors = max level; quad domain, ONE uniform integer level N and one control point per patch is all the corpus shows.
bool Backend::tess_begin(const RegView& r, const DrawCmd& in, DrawCmd& out, uint32_t& level, uint32_t& patches) {
    const uint32_t* hs = &r.sh[kShTessHs];
    float maxl;
    std::memcpy(&maxl, &hs[5], 4);
    level = uint32_t(std::clamp<long>(std::lround(maxl), 1, 16));
    const uint32_t count = in.count;
    patches = count / std::max<uint32_t>(hs[7], 1);
    if (!patches || !(r.sh[kShLsLo] | r.sh[kShLsLo + 1]) || count > (1u << 20)) return false;
    constexpr size_t kCpuZero = 256u << 10;
    const size_t idx_bytes = (4 * (size_t(count) + 1) + 255) & ~size_t(255);
    const size_t lds_bytes = std::clamp<size_t>(size_t(count) * 256, size_t(64) << 10, size_t(16) << 20);
    if (in.indexed) {
        const size_t n = size_t(count) * in.index_bytes;
        if (in.index_addr < 0x10000 || (hooks().mem_valid && !hooks().mem_valid(in.index_addr, n))) return false;
        settle_pending(in.index_addr, in.index_addr + n, 3);  // may flush: before the ring allocation
    }
    size_t off = 0;
    // Large LDS (> kCpuZero) lives in a device-local buffer of the slot, not in the host-visible ring: the LS writes it, the DS reads it back
    // (over PCIe the LS passes were ~5 ms each, GPU 80+ ms per frame in the Forbidden Woods). Small ones stay in the ring (CPU-zeroed).
    const bool dev_lds = lds_bytes > kCpuZero;
    uint8_t* p = ring_alloc(idx_bytes + (dev_lds ? 0 : lds_bytes), 256, off);  // one allocation: a second one could flush and move the first to the other slot
    if (!p) return false;
    tess_ring_flush = flushes;  // a flush from here on (e.g. in the LS pass's descriptor writes) leaves tess_idx/tess_lds in the other slot
    uint32_t* ids = reinterpret_cast<uint32_t*>(p);
    ids[0] = count;
    if (in.indexed) {
        for (uint32_t k = 0; k < count; ++k) ids[1 + k] = in.index_bytes == 4 ? reinterpret_cast<const uint32_t*>(in.index_addr)[k] : reinterpret_cast<const uint16_t*>(in.index_addr)[k];
    } else {
        for (uint32_t k = 0; k < count; ++k) ids[1 + k] = k;
    }
    // The emulated LDS starts zeroed. Small ones on the CPU (the ring is host-visible): then the LS pass needs no barrier before it
    // (record_compute `ls`). Large ones on the GPU (a CPU memset of up to 16 MB per patch draw was ~1 % of the render thread), ordered
    // by record_compute's barrier.
    if (!dev_lds) std::memset(p + idx_bytes, 0, lds_bytes);
    else {
        VkBuffer lb = tess_dev_lds();
        if (!lb) return false;
        barrier();  // the buffer is shared by the slot's patch draws: earlier DS reads and LS writes are done (also ends the pass: no transfer command inside a rendering instance)
        vkCmdFillBuffer(rcb(), lb, 0, lds_bytes, 0);
        tess_lds = {lb, 0, lds_bytes};
    }
    tess_idx = {ring.buffer, off, idx_bytes};
    if (!dev_lds) tess_lds = {ring.buffer, off + idx_bytes, lds_bytes};
    cur_tess_ls = 1;
    BoundShader& ls = bs_ls;
    get_shader(ShStage::VS, r, ls);
    cur_tess_ls = 0;
    if (!ls.e || !ls.e->tr.error.empty()) return false;
    uint32_t max_id = 0;
    for (uint32_t k = 0; k < count; ++k) max_id = std::max(max_id, ids[1 + k]);
    cur_vtx_records = uint64_t(max_id) + 1;  // the LS fetches by absolute vertex id (v0 = ids[gid]): bind records [0, max id] (no offset window)
    cur_vtx_first = 0;
    record_compute(ls, &r.sh[kShLsUser], (count + 63) / 64, 1, 1, !dev_lds);
    std::vector<uint16_t>& grid = tess_grid[level];
    if (grid.empty()) {  // N x N cells of two triangles over the (N+1)^2 grid vertices
        const uint32_t w = level + 1;
        for (uint32_t j = 0; j < level; ++j)
            for (uint32_t i = 0; i < level; ++i) {
                const uint16_t a = uint16_t(j * w + i), b = uint16_t(a + 1), c2 = uint16_t(a + w), d2 = uint16_t(c2 + 1);
                for (const uint16_t v : {a, b, c2, b, d2, c2}) grid.push_back(v);
            }
    }
    out = in;
    out.indexed = true;
    out.index_bytes = 2;
    out.count = uint32_t(grid.size());
    out.instances = patches;
    out.index_addr = reinterpret_cast<uint64_t>(grid.data());
    out.host_indices = true;
    return true;
}

// BB_SCALE_LOG (see Backend::census): "base:WxH:f<VkFormat><class>:<rt|ds|tex>"
std::string Backend::census_name(const Image* im) const {
    const auto rt = rts.find(im->base);
    const auto ds = dss.find(im->base);
    const char* kind = rt != rts.end() && rt->second.get() == im ? "rt" : ds != dss.end() && ds->second.get() == im ? "ds" : "tex";
    char b[96];
    std::snprintf(b, sizeof b, "%llx:%ux%u:f%d%c:%s%s", (unsigned long long)im->base, im->gw(), im->gh(), int(im->format), scale_class(im->gw(), im->gh()), kind, im->scale > 1 ? "*" : "");
    return b;
}

// One image binding of the op being recorded: use (L loaded only, S sampled, W storage write), T# size and class, the bound image.
void Backend::census_img(const BoundShader& bs, const Resource& r, const uint32_t* w, const Image* im) {
    const uint32_t tw = (w[2] & 0x3FFF) + 1, th = ((w[2] >> 14) & 0x3FFF) + 1;
    const char use = r.type == Resource::StorageImage ? 'W' : r.sampler == -1 ? 'L' : 'S', cls = scale_class(tw, th);
    char b[64];
    std::snprintf(b, sizeof b, " %c:T#%ux%u%c=", use, tw, th, cls);
    census_res += b + census_name(im);
    if (cls != 'S' || use == 'S') return;
    const char* st = bs.e->stage == ShStage::CS ? "CS" : bs.e->stage == ShStage::PS ? "PS" : "VS";
    ++census_sum[std::string(st) + (use == 'W' ? " writes" : " loads (image_load)") + " a screen-sized image: " + bs.e->log_name + " -> " + census_name(im)];
}

// Called per guest frame: arms the census frame once BB_SCALE_LOG seconds passed, prints its summary when it ended.
void Backend::census_frame() {
    static const double at = std::atof(std::getenv("BB_SCALE_LOG"));
    static const auto t0 = std::chrono::steady_clock::now();
    static bool done = false;
    if (done) return;
    if (census) {
        std::fprintf(stderr, "scale: summary (ops per line) of frame %llu\n", (unsigned long long)g_frame_counter.load() - 1);
        for (const auto& [k, n] : census_sum) std::fprintf(stderr, "scale: %6llu  %s\n", (unsigned long long)n, k.c_str());
        census = false, done = true;
        census_sum.clear();
        return;
    }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < at) return;
    census = true;
    std::fprintf(stderr, "scale: census of frame %llu (BB_SCALE_LOG)\n", (unsigned long long)g_frame_counter.load());
}

// BB_PIXEL=<hex 8-bit target base>[+<base>...]:<x>:<y>:<first draw>: after every draw/dispatch read one pixel of each target back (synchronously) and log whatever changes it.
void Backend::pixel_probe(const std::string& who) {
    struct P { std::vector<uint64_t> rt; uint32_t x = 0, y = 0; uint64_t from = 0; };
    static const P pp = [] {
        P v;
        unsigned long long from = 0; unsigned x = 0, y = 0;
        if (const char* e = std::getenv("BB_PIXEL")) {
            const char* c = e;
            for (char* end; (v.rt.push_back(std::strtoull(c, &end, 16)), end != c) && *end == '+'; c = end + 1) {}
            if (const char* col = std::strchr(e, ':'); col && std::sscanf(col, ":%u:%u:%llu", &x, &y, &from) >= 2) { v.x = x; v.y = y; v.from = from; }
            else v.rt.clear();
        }
        return v;
    }();
    static std::map<uint64_t, uint64_t> last;
    static int n = 0;
    static const auto t0 = std::chrono::steady_clock::now();
    static const char* after_e = std::getenv("BB_PIXEL_AFTER");  // debug: start probing after N wall seconds ("p<N>": on the pad script clock)
    static const bool after_pad = after_e && *after_e == 'p';
    static const long after_s = after_e ? std::atol(after_e + after_pad) : 0;
    if (draws < pp.from || n >= 80000 || (after_pad ? pad_script_clock() < double(after_s) : std::chrono::steady_clock::now() - t0 < std::chrono::seconds(after_s))) return;
    static bool said = false;
    for (const uint64_t rt : pp.rt) {
        auto it = rts.find(rt);
        auto dit = dss.find(rt);  // a depth buffer: its depth plane is read (float32)
        const bool is_ds = it == rts.end() && dit != dss.end();
        if (!said) std::fprintf(stderr, "pixel: probing 0x%llx (%u,%u) from draw %llu: target %s\n", (unsigned long long)rt, pp.x, pp.y, (unsigned long long)draws, it != rts.end() ? "found" : is_ds ? "found (depth)" : "not known (yet)");
        if ((it == rts.end() && !is_ds) || !(is_ds ? dit->second : it->second)->initialised) continue;
        Image* im = (is_ds ? dit->second : it->second).get();
        const bool px32 = is_ds ? im->format != VK_FORMAT_S8_UINT : im->format == VK_FORMAT_R8G8B8A8_UNORM || im->format == VK_FORMAT_R8G8B8A8_SRGB || im->format == VK_FORMAT_B8G8R8A8_UNORM || im->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || im->format == VK_FORMAT_R32_SFLOAT;
        const bool px64 = !is_ds && (im->format == VK_FORMAT_R16G16B16A16_SFLOAT || im->format == VK_FORMAT_R32G32_SFLOAT);
        if ((!px32 && !px64) || pp.x >= im->w || pp.y >= im->h) { std::fprintf(stderr, "pixel: unusable target format %d %ux%u\n", int(im->format), im->w, im->h); n = 80000; return; }
        size_t off = 0;
        if (!ring_alloc(16, 16, off)) return;
        barrier();
        VkBufferImageCopy cp{off, 0, 0, {VkImageAspectFlags(is_ds ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1}, {int32_t(pp.x), int32_t(pp.y), 0}, {1, 1, 1}};
        vkCmdCopyImageToBuffer(rcb(), im->image, VK_IMAGE_LAYOUT_GENERAL, ring.buffer, 1, &cp);
        barrier();
        const size_t roff = off;
        flush();
        uint64_t px = 0;
        std::memcpy(&px, ring.mapped + roff, px64 ? 8 : 4);
        ++n;
        uint64_t& l = last[rt];
        if (px != l) {
            std::fprintf(stderr, "pixel: #%llu: [%llx] %s: %08llx -> %08llx\n", (unsigned long long)draws, (unsigned long long)rt, who.c_str(), (unsigned long long)l, (unsigned long long)px);
            l = px;
        }
    }
    said = true;
}

void Backend::shutdown() {
    VkCtx& c = vk();
    if (c.device) vkDeviceWaitIdle(c.device);
}

// ---- hooks -------------------------------------------------------------------------------------------------------------------
struct Clk { std::chrono::steady_clock::time_point t = tnow(); uint64_t ns_since() const { return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(tnow() - t).count()); } };
uint64_t g_ns_draw = 0, g_ns_disp = 0, g_ns_flush = 0, g_ns_lock = 0;  // g_ns_lock: render-thread waits for the backend lock (inside draw/dispatch)
struct BeLock {  // the backend lock for draws/dispatches; a contended wait (fence watcher, present thread) is timed
    std::unique_lock<std::mutex> lk{g_be->mtx, std::try_to_lock};
    BeLock() { if (!lk.owns_lock()) { const Clk k; lk.lock(); g_ns_lock += k.ns_since(); } }
};
// BB_OPS_LOG=<first>,<count> (or t<seconds>,<count> / p<pad-clock seconds>,<count>): from the <first>-th draw (or that second) on, log <count> draws/dispatches with
// their shaders, targets and outcome (recorded / skip reason)
struct OpsLog {
    bool active = false;
    static inline std::map<std::string, uint64_t> before;  // (static: an MSVC map allocates its head node, per draw/dispatch otherwise)
    uint64_t rendered0 = 0, dispatched0 = 0;
    static bool enabled(const Backend& b, uint64_t counted) {
        static uint64_t first = 0, count = 0, left = 0;
        static char clock = 0;  // 't': wall seconds, 'p': pad script clock (BB_PAD_CLOCK=flips: the same game state in every run)
        static const auto t0 = std::chrono::steady_clock::now();
        static const bool on = [] {
            const char* e = std::getenv("BB_OPS_LOG");
            if (e && (*e == 't' || *e == 'p')) clock = *e++;
            return e && std::sscanf(e, "%llu,%llu", (unsigned long long*)&first, (unsigned long long*)&count) == 2;
        }();
        if (!on) return false;
        const bool due = clock == 't' ? std::chrono::steady_clock::now() - t0 >= std::chrono::seconds(first) : clock == 'p' ? pad_script_clock() >= double(first) : counted >= first;
        if (left == 0 && due && count) { left = count; count = 0; }
        if (left == 0) return false;
        --left;
        (void)b;
        return true;
    }
};
void hook_draw(const RegView& r, const DrawCmd& d) {
    Clk k;
    BeLock lk;
    OpsLog ol;
    ol.active = OpsLog::enabled(*g_be, g_be->draws + g_be->dispatches);
    if (ol.active) { ol.before = g_be->skip_by; ol.rendered0 = g_be->rendered; }
    g_be->chunk_tick();
    g_be->draw(r, d);
    if (ol.active) {
        std::string why = g_be->rendered != ol.rendered0 ? "recorded" : "skipped?";
        for (const auto& [kk, v] : g_be->skip_by) if (!ol.before.count(kk) || ol.before[kk] != v) why = "skip:" + kk;
        const uint64_t ps = (uint64_t(r.sh[kShPsLo + 1]) << 40) | (uint64_t(r.sh[kShPsLo]) << 8), vs = (uint64_t(r.sh[kShVsLo + 1]) << 40) | (uint64_t(r.sh[kShVsLo]) << 8);
        std::fprintf(stderr, "op: draw vs=%llx ps=%llx prim=%u n=%u inst=%u rt0=%llx info=%08x mask=%x zctl=%x -> %s\n", (unsigned long long)vs, (unsigned long long)ps, r.uconfig[kVgtPrimType] & 0x3F,
                     d.count, d.instances, (unsigned long long)r.context[kCbColor0Base] << 8, r.context[kCbColor0Base + 4], r.context[kCbTargetMask], r.context[kDbDepthControl], why.c_str());
    }
    g_ns_draw += k.ns_since();
}
void hook_dispatch(const RegView& r, uint32_t x, uint32_t y, uint32_t z) {
    Clk k;
    BeLock lk;
    OpsLog ol;
    ol.active = OpsLog::enabled(*g_be, g_be->draws + g_be->dispatches);
    if (ol.active) { ol.before = g_be->skip_by; ol.dispatched0 = g_be->dispatched; }
    g_be->chunk_tick();
    g_be->dispatch(r, x, y, z);
    if (ol.active) {
        std::string why = g_be->dispatched != ol.dispatched0 ? "recorded" : "skipped?";
        for (const auto& [kk, v] : g_be->skip_by) if (!ol.before.count(kk) || ol.before[kk] != v) why = "skip:" + kk;
        const uint64_t cs = (uint64_t(r.sh[kShCsLo + 1]) << 40) | (uint64_t(r.sh[kShCsLo]) << 8);
        std::fprintf(stderr, "op: dispatch cs=%llx groups=%ux%ux%u -> %s\n", (unsigned long long)cs, x, y, z, why.c_str());
    }
    g_ns_disp += k.ns_since();
}
// DISPATCH_INDIRECT whose arguments the GPU is still writing: the dispatch reads them on the GPU (vkCmdDispatchIndirect from the
// imported guest memory) instead of draining the GPU for a CPU read. false: the PM4 scan reads them on the CPU (after sync_read).
// Gnm helper shaders (image ops on the CPU) need the counts and take the CPU path. BB_GPU_INDIRECT=0: always the CPU path.
bool hook_dispatch_indirect(const RegView& r, uint64_t args) {
    static const bool on = !(std::getenv("BB_GPU_INDIRECT") && std::getenv("BB_GPU_INDIRECT")[0] == '0');
    if (!on) return false;
    Clk k;
    std::lock_guard<std::mutex> lk(g_be->mtx);
    Backend& b = *g_be;
    if ((args & 3) || !b.overlaps_dirty(args, args + 12)) return false;  // vkCmdDispatchIndirect offsets are multiples of 4
    for (uint32_t s = 0; s < b.nslots; ++s)
        for (const Writeback& w : b.wb_s[s]) { const uint64_t wl = reinterpret_cast<uintptr_t>(w.dst); if (wl < args + 12 && args < wl + w.size) return false; }
    const BoundShader& cs = b.bs_cs;  // (dispatch below looks it up again into the same scratch)
    b.get_shader(ShStage::CS, r, b.bs_cs);
    if (!cs.e || cs.e->kind || !cs.e->tr.error.empty()) return false;
    const uint64_t win = args & ~0xFFFFull, win_bytes = (args + 12 - win + 0xFFFF) & ~0xFFFFull;  // imports want >= 64 KiB, aligned
    VkBuffer hb = VK_NULL_HANDLE;
    VkDeviceSize hoff = 0;
    if (!hooks().mem_valid || !hooks().mem_valid(win, win_bytes) || !b.get_host_buf(win, win_bytes, hb, hoff)) return false;
    b.chunk_tick();
    b.cur_indirect = {hb, hoff + (args - win)};
    b.dispatch(r, 1, 1, 1);  // the counts come from the buffer
    b.cur_indirect = {};
    ++b.gpu_indirect;
    g_ns_disp += k.ns_since();
    return true;
}
// GDS <-> guest memory (DMA_DATA with a GDS end, EVENT_WRITE_EOS "store GDS data"): recorded in stream order between the dispatches
// that use the counters. The game saves its append counters to memory and restores them this way.
void hook_gds_copy(bool to_gds, uint64_t mem, uint32_t gds_off, uint32_t bytes) {
    std::lock_guard<std::mutex> lk(g_be->mtx);
    Backend& b = *g_be;
    VkBuffer hb = VK_NULL_HANDLE;
    VkDeviceSize hoff = 0;
    const uint64_t win = mem & ~0xFFFFull, win_bytes = (mem + bytes - win + 0xFFFF) & ~0xFFFFull;  // imports want >= 64 KiB, aligned
    if (!bytes || uint64_t(gds_off) + bytes > Backend::kGdsBytes || !hooks().mem_valid || !hooks().mem_valid(win, win_bytes) || !b.get_host_buf(win, win_bytes, hb, hoff)) {
        b.log_once("gds" + std::to_string(mem), "GDS copy with an unusable range skipped");
        return;
    }
    hoff += mem - win;
    b.cut(true);  // GDS / guest memory writes: a replay skips them
    b.barrier();
    const VkBufferCopy cp{to_gds ? hoff : b.gds_off() + gds_off, to_gds ? b.gds_off() + gds_off : hoff, bytes};
    b.ts_label("copy: GDS");
    vkCmdCopyBuffer(b.rcb(), to_gds ? hb : b.ring.buffer, to_gds ? b.ring.buffer : hb, 1, &cp);
    b.barrier();
    b.cut(false);
    if (!to_gds) { b.dirty_add(mem, mem + bytes); b.imp_written = true; gpu_watch("GDS copy", mem, bytes); }
}
// DMA_DATA memory -> memory whose source the GPU is still writing (pending in the recording or an in-flight slot): copied on the GPU
// in stream order, as the console's CP does, instead of draining the GPU so the PM4 scan can memmove (ROADMAP 55). Both ends go
// through host imports; a source that is a write-back (the GPU result still sits in the ring) or overlapping ends take the CPU path.
// BB_GPU_DMA=0: always the CPU path.
bool hook_mem_copy(uint64_t dst, uint64_t src, uint64_t bytes) {
    static const bool on = !(std::getenv("BB_GPU_DMA") && std::getenv("BB_GPU_DMA")[0] == '0');
    if (!on || !bytes) return false;
    std::lock_guard<std::mutex> lk(g_be->mtx);
    Backend& b = *g_be;
    const uint64_t se = src + bytes, de = dst + bytes;
    if (src < de && dst < se) return false;
    if (!b.overlaps_dirty(src, se)) return false;
    for (uint32_t s = 0; s < b.nslots; ++s)
        for (const Writeback& w : b.wb_s[s]) {
            const uint64_t wl = reinterpret_cast<uintptr_t>(w.dst);
            if (wl < se && src < wl + w.size) return false;
        }
    auto import = [&](uint64_t a, VkBuffer& buf, VkDeviceSize& off) {
        const uint64_t win = a & ~0xFFFFull, win_bytes = (a + bytes - win + 0xFFFF) & ~0xFFFFull;  // imports want >= 64 KiB, aligned
        if (!hooks().mem_valid || !hooks().mem_valid(win, win_bytes) || !b.get_host_buf(win, win_bytes, buf, off)) return false;
        off += a - win;
        return true;
    };
    VkBuffer sb = VK_NULL_HANDLE, db = VK_NULL_HANDLE;
    VkDeviceSize so = 0, dof = 0;
    if (!import(src, sb, so) || !import(dst, db, dof)) return false;
    b.cut(true);  // guest memory write: a replay skips it
    b.barrier();
    const VkBufferCopy cp{so, dof, bytes};
    b.ts_label("copy: DMA buffer");
    vkCmdCopyBuffer(b.rcb(), sb, db, 1, &cp);
    b.barrier();
    b.cut(false);
    b.dirty_add(dst, de);
    b.imp_written = true;
    ++b.gpu_dmas;
    gpu_watch("DMA_DATA copy (GPU)", dst, bytes);
    return true;
}
// Completion label from the PM4 stream: queued with the slot recording the work ahead of it, written by settle() once that work ran.
void hook_label(uint64_t addr, uint64_t value, uint32_t bytes) {
    std::lock_guard<std::mutex> lk(g_be->mtx);
    g_be->labels_s[g_be->slot].push_back({addr, value, bytes});
}
// CPU access of guest memory pending GPU work may write (PM4 ops executed at scan time; why: see gpu_hooks.h). The writing ops (5 WRITE_DATA,
// 6 DMA_DATA destination, 9 DUMP_CONST_RAM) write right after this call: textures there are checked again at their next use.
void hook_sync_read(uint64_t addr, uint64_t bytes, int why) {
    std::lock_guard<std::mutex> lk(g_be->mtx);
    g_be->settle_pending(addr, addr + bytes, why);
    if (why == 5 || why == 6 || why == 9) g_be->tex_touch(addr, addr + bytes);
}
// BB_GPU_STATS: how many bytes of each render target are non-zero (proof that anything non-black was drawn).
void report_rt_stats() {
    Backend& b = *g_be;
    static bool selftest_done = false;
    if (!selftest_done && !b.rts.empty() && b.rts.begin()->second->initialised) {  // validate the instrument itself: a cleared RT must read back non-zero
        selftest_done = true;
        Image* im = b.rts.begin()->second.get();
        VkClearColorValue white{{1.f, 1.f, 1.f, 1.f}};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.barrier();
        vkCmdClearColorImage(b.rcb(), im->image, VK_IMAGE_LAYOUT_GENERAL, &white, 1, &range);
        b.barrier();
        size_t off = 0;
        uint8_t* p = b.ring_alloc(size_t(im->w) * im->h * 4, 16, off);
        VkBufferImageCopy copy{};
        copy.bufferOffset = off;
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {im->w, im->h, 1};
        vkCmdCopyImageToBuffer(b.rcb(), im->image, VK_IMAGE_LAYOUT_GENERAL, b.ring.buffer, 1, &copy);
        b.barrier();
        const size_t ring_off = off;
        b.flush();
        size_t nz = 0;
        for (size_t i = 0; p && i < size_t(im->w) * im->h * 4; ++i) nz += b.ring.mapped[ring_off + i] != 0;
        std::fprintf(stderr, "gpu: readback self-test (clear to white): %zu/%zu non-zero bytes\n", nz, size_t(im->w) * im->h * 4);
        VkClearColorValue zero{};
        vkCmdClearColorImage(b.rcb(), im->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
        b.barrier();
    }
    static const std::vector<uint64_t> dump_only = [] {  // debug: BB_GPU_DUMP_ONLY=<hex>[,<hex>...] reports/dumps just these RTs
        std::vector<uint64_t> v;
        if (const char* e = std::getenv("BB_GPU_DUMP_ONLY")) for (const char* c = e; *c;) { char* end = nullptr; v.push_back(std::strtoull(c, &end, 16)); if (end == c) break; c = *end == ',' ? end + 1 : end; }
        return v;
    }();
    for (auto& [addr, im] : b.rts) {
        if (!im->initialised || (!dump_only.empty() && std::find(dump_only.begin(), dump_only.end(), addr) == dump_only.end())) continue;
        const uint32_t bytes_px = im->format == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 4;
        const size_t n = size_t(im->w) * im->h * bytes_px;
        size_t off = 0;
        uint8_t* p = b.ring_alloc(n, 16, off);
        if (!p) continue;
        VkBufferImageCopy copy{};
        copy.bufferOffset = off;
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {im->w, im->h, 1};
        b.barrier();
        vkCmdCopyImageToBuffer(b.rcb(), im->image, VK_IMAGE_LAYOUT_GENERAL, b.ring.buffer, 1, &copy);
        b.barrier();
        const size_t ring_off = off;
        b.flush();
        const uint8_t* q = b.ring.mapped + ring_off;
        size_t nz = 0;
        for (size_t i = 0; i < n; ++i) nz += q[i] != 0;
        std::fprintf(stderr, "gpu: RT 0x%llx %ux%u fmt %d: %zu/%zu non-zero bytes\n", (unsigned long long)addr, im->w, im->h, int(im->format), nz, n);
        if (const char* dir = std::getenv("BB_GPU_DUMP")) {  // raw dump for offline conversion: <dir>/rt_<addr>_<w>x<h>_<fmt>.raw (latest wins)
            char path[400];
            std::snprintf(path, sizeof path, "%s/rt_%llx_%ux%u_%d.raw", dir, (unsigned long long)addr, im->w, im->h, int(im->format));
            if (FILE* f = std::fopen(path, "wb")) { std::fwrite(q, 1, n, f); std::fclose(f); }
        }
    }
    if (const char* dir = std::getenv("BB_GPU_DUMP")) {  // depth planes: <dir>/ds_<addr>_<w>x<h>.raw (float32, latest wins)
        for (auto& [addr, im] : b.dss) {
            if (!im->initialised || im->format == VK_FORMAT_S8_UINT) continue;
            const size_t n = size_t(im->w) * im->h * 4;
            size_t off = 0;
            uint8_t* p = b.ring_alloc(n, 16, off);
            if (!p) continue;
            VkBufferImageCopy copy{off, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {im->w, im->h, 1}};
            b.barrier();
            vkCmdCopyImageToBuffer(b.rcb(), im->image, VK_IMAGE_LAYOUT_GENERAL, b.ring.buffer, 1, &copy);
            b.barrier();
            const size_t ring_off = off;
            b.flush();
            char path[400];
            std::snprintf(path, sizeof path, "%s/ds_%llx_%ux%u.raw", dir, (unsigned long long)addr, im->w, im->h);
            if (FILE* f = std::fopen(path, "wb")) { std::fwrite(b.ring.mapped + ring_off, 1, n, f); std::fclose(f); }
        }
    }
}

uint64_t g_submit_n = 0;
void hook_end_submit() {
    ++g_submit_n;
    std::lock_guard<std::mutex> lk(g_be->mtx);
    {
        Clk k;
        ww_new_submit();
        ++g_tex_epoch;
        // Completion labels must not run ahead of the GPU: settle() writes them once the slot's fence signalled. The fence watcher
        // (backend_init) does that as soon as the GPU is done, so the submission itself need not wait. BB_LABEL_WATCH=0: wait here instead.
        static const bool sync_labels = std::getenv("BB_LABEL_WATCH") && std::getenv("BB_LABEL_WATCH")[0] == '0';
        // Coalescing: guest submits that follow within BB_COALESCE_US (default 300 us, 0 = off) go into the same command buffer; the fence
        // watcher submits it when the time runs out, so completion labels are delayed by at most that. Each vkQueueSubmit costs GPU time.
        static const int64_t coalesce_us = std::getenv("BB_COALESCE_US") ? std::atoll(std::getenv("BB_COALESCE_US")) : 300;
        if (coalesce_us > 0 && !sync_labels) {
            if (!g_be->flush_pending) { g_be->flush_pending = true; g_be->flush_due = std::chrono::steady_clock::now() + std::chrono::microseconds(coalesce_us); }
        } else g_be->flush(sync_labels && !g_be->labels_s[g_be->slot].empty());
        if (!g_be->recording && !g_be->labels_s[g_be->slot].empty()) g_be->settle(g_be->slot);  // labels without recorded work
        g_ns_flush += k.ns_since();
    }
    static const char* stats = std::getenv("BB_GPU_STATS");  // "1": every 300 submits; "t<seconds>": once at that wall time; "p<seconds>": on the pad script clock
    static uint64_t submits = 0;
    static const auto stats_t0 = std::chrono::steady_clock::now();
    static bool stats_once = false;
    if (stats && (stats[0] == 't' || stats[0] == 'p')) {
        const double now = stats[0] == 'p' ? pad_script_clock() : std::chrono::duration<double>(std::chrono::steady_clock::now() - stats_t0).count();
        if (!stats_once && now >= std::atof(stats + 1)) { stats_once = true; report_rt_stats(); }
    } else if (stats && ++submits % 300 == 0) report_rt_stats();
    static uint64_t last = 0;
    if (g_be->verbose && g_be->draws - last >= 100) {
        last = g_be->draws;
        static const auto t0 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "gpu: [%.1fs] %llu submits, %llu draws, %llu dispatches, %llu skipped", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), (unsigned long long)g_submit_n,
                     (unsigned long long)g_be->draws, (unsigned long long)g_be->dispatches, (unsigned long long)g_be->skipped);
        for (const auto& [k, v] : g_be->skip_by) std::fprintf(stderr, " %s=%llu", k.c_str(), (unsigned long long)v);
        std::fprintf(stderr, " [recorded: %llu draws, %llu dispatches (%llu image ops, %llu cpu ops = %llu MB, %llu MB written back), %llu ds clears, %llu fast clears; copied %llu MB buffers (%llu copies, %llu ms), %llu MB textures, %llu flushes; ms: draw %llu dispatch %llu flush %llu]", (unsigned long long)g_be->rendered,
                     (unsigned long long)g_be->dispatched, (unsigned long long)g_be->image_ops, (unsigned long long)g_be->cpu_ops, (unsigned long long)(g_be->cpu_bytes >> 20), (unsigned long long)(g_be->wb_bytes >> 20), (unsigned long long)g_be->ds_clears, (unsigned long long)g_be->fast_clears, (unsigned long long)(g_be->buf_bytes >> 20), (unsigned long long)g_be->buf_copies, (unsigned long long)(g_be->buf_ns / 1000000), (unsigned long long)(g_be->tex_bytes >> 20), (unsigned long long)g_be->flushes,
                     (unsigned long long)(g_ns_draw / 1000000), (unsigned long long)(g_ns_disp / 1000000), (unsigned long long)(g_ns_flush / 1000000));
        std::fprintf(stderr, " [textures: %llu lookups, %llu uploads, %llu ms; write_descriptors %llu ms, buffer cases %llu ms; host import %llu ms, %llu imports, %llu MB; %llu zero binds; %llu unsupported-tiling lookups; %zu rts, %zu dss, %zu texs]", (unsigned long long)g_be->tex_calls, (unsigned long long)g_be->tex_uploads, (unsigned long long)(g_be->tex_ns / 1000000), (unsigned long long)(g_be->wd_ns / 1000000), (unsigned long long)(g_be->wd_buf_ns / 1000000), (unsigned long long)(g_be->host_ns / 1000000), (unsigned long long)g_be->host_imports, (unsigned long long)(g_be->host_import_bytes >> 20), (unsigned long long)g_be->zero_binds, (unsigned long long)g_be->tex_unsup, g_be->rts.size(), g_be->dss.size(), g_be->texs.size());
        std::fprintf(stderr, " [buffer copies by kind: constant %llu MB, vertex %llu MB, other %llu MB; descriptor sets: alloc %llu ms, update %llu ms; texture views %llu ms, samplers %llu ms, index upload %llu ms]", (unsigned long long)(g_be->cb_bytes >> 20), (unsigned long long)(g_be->vb_bytes >> 20), (unsigned long long)(g_be->ob_bytes >> 20), (unsigned long long)(g_be->ds_alloc_ns / 1000000), (unsigned long long)(g_be->ds_update_ns / 1000000), (unsigned long long)(g_be->view_ns / 1000000), (unsigned long long)(g_be->smp_ns / 1000000), (unsigned long long)(g_be->idx_up_ns / 1000000));
        std::fprintf(stderr, " [descriptor sets from the pool (not pushed): %llu]", (unsigned long long)g_be->ds_pool_sets);
        std::fprintf(stderr, " [draw() sections ms: shader lookup %llu, targets+depth %llu, pipeline %llu, index scan %llu, descriptors %llu, recording %llu; backend lock waits %llu ms]", (unsigned long long)(g_ns_shader / 1000000), (unsigned long long)(g_ns_early / 1000000), (unsigned long long)(g_ns_pipe / 1000000), (unsigned long long)(g_ns_idx / 1000000), (unsigned long long)(g_ns_desc / 1000000), (unsigned long long)(g_ns_rec / 1000000), (unsigned long long)(g_ns_lock / 1000000));
        std::fprintf(stderr, " [frames presented %llu; padded-target crops %llu, %llu MB]", (unsigned long long)g_frame_counter.load(), (unsigned long long)g_be->crop_copies, (unsigned long long)(g_be->crop_bytes >> 20));
        { std::fprintf(stderr, " [textures by tiling/format:"); for (const auto& [k, v] : g_be->tex_mix) std::fprintf(stderr, " %s=%llu;", k.c_str(), (unsigned long long)v); std::fprintf(stderr, "]"); }
        std::fprintf(stderr, " [vertex cache: %llu hits, %llu uploads, index cache: %llu hits, %llu uploads, %llu MB resident, %zu entries; hashed %llu MB; evicted %llu, swept %llu, pool allocation failures %llu;", (unsigned long long)g_be->vtx_hits, (unsigned long long)g_be->vtx_uploads, (unsigned long long)g_be->idx_hits, (unsigned long long)g_be->idx_uploads, (unsigned long long)(g_be->vtx_cache_bytes >> 20), g_be->vtx_cache.size(), (unsigned long long)(g_be->vtx_hash_bytes >> 20), (unsigned long long)g_be->vtx_evicted, (unsigned long long)g_be->vtx_swept, (unsigned long long)g_be->vtx_alloc_fails);
        std::fprintf(stderr, " texture hashes %llu MB, checks %llu, full %llu, write-watch checks %llu, epoch %llu, dynamic %llu; write-watch polls %llu (%llu MB); write-watch verify: %llu checked, %llu wrong, %llu written after the submit's poll;", (unsigned long long)(g_be->tex_hash_bytes >> 20), (unsigned long long)g_be->tex_checks, (unsigned long long)g_be->tex_full_hashes, (unsigned long long)g_be->tex_ww_checks, (unsigned long long)g_tex_epoch.load(), (unsigned long long)g_be->tex_dynamic, (unsigned long long)ww_polls(), (unsigned long long)(ww_poll_pages() >> 8), (unsigned long long)g_be->ww_checked, (unsigned long long)g_be->ww_mismatch, (unsigned long long)g_be->ww_mismatch_late);
        std::fprintf(stderr, " texture verify: %llu checked, %llu stale in epoch, %llu stale by sample, %llu stale by write watch; texture memo hits %llu (renewed by the write watch %llu); mipped textures %llu of %llu]", (unsigned long long)g_be->tex_verified, (unsigned long long)g_be->tex_stale_epoch, (unsigned long long)g_be->tex_stale_sample, (unsigned long long)g_be->tex_stale_ww, (unsigned long long)g_be->tex_memo_hits, (unsigned long long)g_be->tex_ww_renewals, (unsigned long long)g_be->tex_mipped, (unsigned long long)g_be->tex_texture_images);
        std::fprintf(stderr, " [flushes %llu, synchronous %llu (graveyard %llu, host imports %llu, imported writes %llu, forced %llu), waits for reads of GPU-written memory %llu in flight, %llu still recording; patch draws across slots %llu]", (unsigned long long)g_be->flushes, (unsigned long long)g_be->sync_n, (unsigned long long)g_be->sync_grave, (unsigned long long)g_be->sync_host, (unsigned long long)g_be->sync_imp, (unsigned long long)g_be->sync_wait, (unsigned long long)g_be->sync_overlap, (unsigned long long)g_be->sync_cur, (unsigned long long)g_be->tess_cross);
        std::fprintf(stderr, " [vertex cache time %llu ms; flush: submit %llu ms, waiting for the reused slot %llu ms, synchronous waits %llu ms; GPU time of command buffers %llu ms over %llu; barriers %llu]", (unsigned long long)(g_be->vtx_ns / 1000000), (unsigned long long)(g_be->ft_submit / 1000000), (unsigned long long)(g_be->ft_settle / 1000000), (unsigned long long)(g_be->ft_syncwait / 1000000), (unsigned long long)(g_be->gpu_cb_us / 1000), (unsigned long long)g_be->gpu_cbs, (unsigned long long)g_be->barriers);
        std::fprintf(stderr, " [vertex cache uploads %llu ms, %llu MB, %llu]", (unsigned long long)(g_be->vtx_up_ns / 1000000), (unsigned long long)(g_be->vtx_up_bytes >> 20), (unsigned long long)(g_be->vtx_uploads + g_be->idx_uploads));
        std::fprintf(stderr, " [barriers skipped: nothing recorded %llu, independent passes %llu, before LS passes %llu; padded targets sampled in place %llu]", (unsigned long long)g_be->barriers_skipped, (unsigned long long)g_be->pass_barriers_skipped, (unsigned long long)g_be->ls_barriers_skipped, (unsigned long long)g_be->pad_direct_binds);
        { std::fprintf(stderr, " [read flushes by caller:"); static const char* const nm[11] = {"textures", "buffers", "indices", "tess", "indirect args", "WRITE_DATA", "DMA dst", "DMA src", "LOAD_CONST_RAM", "DUMP_CONST_RAM", "other"};
          for (int k = 0; k < 11; ++k) std::fprintf(stderr, " %s %llu,", nm[k], (unsigned long long)g_be->sync_cur_by[k]);
          std::fprintf(stderr, " inside one write-back %llu; DMA copies on the GPU %llu; imports ordered by a barrier %llu; indirect dispatches on the GPU %llu]", (unsigned long long)g_be->sync_cur_wb, (unsigned long long)g_be->gpu_dmas, (unsigned long long)g_be->import_barriers, (unsigned long long)g_be->gpu_indirect); }
        if (g_be->prof) {
            std::fputc('\n', stderr);
            std::vector<std::pair<double, std::string>> top;
            for (const auto& [k, v] : g_be->gpu_ms) top.push_back({v.first, k + " x" + std::to_string(v.second)});
            std::sort(top.rbegin(), top.rend());
            double total = 0;
            for (const auto& tp : top) total += tp.first;
            std::fprintf(stderr, "gpu:   prof total %.0f ms over %zu pipelines\n", total, top.size());
            for (size_t i = 0; i < top.size() && i < 6; ++i) std::fprintf(stderr, "gpu:   prof %.0f ms %s\n", top[i].first, top[i].second.c_str());
        }
        std::fputc('\n', stderr);
    }
}

}  // namespace

bool backend_init() {
    g_be = new Backend;
    if (!g_be->init()) return false;
    Hooks& h = hooks();
    h.draw = hook_draw;
    h.dispatch = hook_dispatch;
    h.end_submit = hook_end_submit;
    h.gds_copy = hook_gds_copy;
    h.sync_read = hook_sync_read;
    h.mem_copy = hook_mem_copy;
    h.dispatch_indirect = hook_dispatch_indirect;
    h.label = hook_label;
    // Fence watcher: settles a submitted slot (write-backs, then completion labels) right after its fence signals, so guest code that
    // waits for a label does not depend on the next backend activity. Polls under the backend lock (a wait outside it would race the
    // fence reset when the slot is reused); 100 us between polls while work is in flight.
    g_be->watcher = std::thread([] {
        Backend& b = *g_be;
        VkDevice dev = vk().device;
        while (!b.stop_watch.load(std::memory_order_relaxed)) {
            bool busy = false;
            {
                // try_lock: the render thread holds the lock while it records; then it settles slots itself when it reuses them. Waiting
                // here would hand the lock back and forth (profile before: ~16 % of the render thread in thread-alert waits, ROADMAP 55)
                std::unique_lock<std::mutex> lk(b.mtx, std::try_to_lock);
                if (!lk.owns_lock()) { fine_sleep_us(50); continue; }
                if (b.flush_pending) {  // coalesced end-of-submit flush (hook_end_submit): due now?
                    busy = true;
                    if (std::chrono::steady_clock::now() >= b.flush_due) b.flush(false);
                }
                for (uint32_t k = 1; k < b.nslots; ++k) {  // oldest first (one queue: fences signal in submission order)
                    const uint32_t s = b.older(k);
                    if (!b.in_flight[s]) continue;
                    busy = true;
                    if (vkGetFenceStatus(dev, b.fences[s]) != VK_SUCCESS) break;
                    b.settle(s);
                }
            }
            fine_sleep_us(busy ? 100 : 500);
        }
    });
    return true;
}

void backend_shutdown() {
    if (!g_be) return;
    g_be->stop_watch = true;
    if (g_be->watcher.joinable()) g_be->watcher.join();
    g_be->shutdown();
}

void backend_new_frame() {
    ++g_tex_epoch;
    ++g_frame_counter;
    if (static const bool census = std::getenv("BB_SCALE_LOG") != nullptr; census && g_be) {  // (default off: one branch per frame)
        std::lock_guard<std::mutex> lk(g_be->mtx);
        g_be->census_frame();
    }
    // Debug BB_FRAME_LOG=1: one line per frame with what the backend did since the previous one (which frames run long, and why)
    static const bool flog = std::getenv("BB_FRAME_LOG") != nullptr;
    if (!flog || !g_be) return;
    std::lock_guard<std::mutex> lk(g_be->mtx);
    const Backend& b = *g_be;
    const uint64_t v[] = {b.draws, b.dispatches, g_ns_draw, g_ns_disp, g_ns_lock, b.ft_settle, b.ft_syncwait, b.tex_uploads, b.tex_bytes, b.vtx_uploads + b.idx_uploads,
                          b.gpipelines.size(), b.shaders.size(), b.flushes, b.sync_n, b.host_imports, b.tex_full_hashes, b.tex_checks, b.tex_ww_checks, ww_polls(), ww_poll_pages(),
                          b.tex_ns, b.vtx_ns, b.wd_ns, ww_poll_ns()};
    constexpr size_t n = sizeof v / sizeof v[0];
    static uint64_t prev[n] = {};
    static const auto t0 = std::chrono::steady_clock::now();
    uint64_t d[n];
    for (size_t i = 0; i < n; ++i) d[i] = v[i] - prev[i], prev[i] = v[i];
    std::fprintf(stderr, "frame %llu t=%.4f draws=%llu disp=%llu draw_ms=%.2f disp_ms=%.2f lock_ms=%.2f slot_ms=%.2f sync_ms=%.2f tex_up=%llu tex_kb=%llu vtx_up=%llu pipes=%llu shaders=%llu flushes=%llu syncs=%llu imports=%llu tex_full=%llu "
                 "tex_sample=%llu tex_ww=%llu ww_polls=%llu ww_kb=%llu tex_ms=%.2f vtx_ms=%.2f wd_ms=%.2f ww_ms=%.3f\n",
                 (unsigned long long)g_frame_counter.load(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), (unsigned long long)d[0], (unsigned long long)d[1],
                 double(d[2]) / 1e6, double(d[3]) / 1e6, double(d[4]) / 1e6, double(d[5]) / 1e6, double(d[6]) / 1e6, (unsigned long long)d[7], (unsigned long long)(d[8] >> 10), (unsigned long long)d[9], (unsigned long long)d[10],
                 (unsigned long long)d[11], (unsigned long long)d[12], (unsigned long long)d[13], (unsigned long long)d[14], (unsigned long long)d[15],
                 (unsigned long long)d[16], (unsigned long long)d[17], (unsigned long long)d[18], (unsigned long long)(d[19] * 4), double(d[20]) / 1e6, double(d[21]) / 1e6, double(d[22]) / 1e6, double(d[23]) / 1e6);
}

bool backend_blit_frame(uint64_t addr, uint32_t width, uint32_t height, VkCommandBuffer cb, VkImage dst, VkExtent2D dst_extent) {
    if (!g_be) return false;
    std::lock_guard<std::mutex> lk(g_be->mtx);
    if (g_be->flush_pending) g_be->flush(false);  // a coalesced submit holds this frame's last work: it must reach the queue before the blit
    auto it = g_be->rts.find(addr);
    if (g_be->verbose) {
        static uint64_t n = 0;
        if (++n <= 5 || n % 200 == 0) {
            std::fprintf(stderr, "gpu: present request #%llu for display buffer 0x%llx: %s (%zu render targets known:", (unsigned long long)n, (unsigned long long)addr, it == g_be->rts.end() ? "NO render target" : "found", g_be->rts.size());
            for (auto& [a, im] : g_be->rts) std::fprintf(stderr, " 0x%llx(%ux%u)", (unsigned long long)a, im->w, im->h);
            std::fprintf(stderr, ")\n");
        }
    }
    if (it == g_be->rts.end()) return false;
    Image* im = it->second.get();
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {int32_t(std::min(im->w, width * im->scale)), int32_t(std::min(im->h, height * im->scale)), 1};  // (guest size -> its pixels)
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {int32_t(dst_extent.width), int32_t(dst_extent.height), 1};
    vkCmdBlitImage(cb, im->image, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    return true;
}

uint32_t res_scale() {
    static const uint32_t s = [] {
        const char* e = std::getenv("BB_RES_SCALE");
        uint32_t v = e ? uint32_t(std::clamp(std::atoi(e), 1, 3)) : 1;
        if (v > 1 && vk().props.limits.maxPushConstantsSize < 148) {  // (user data + scale bits for VS/CS at 0 and PS at 80, see Backend::push_user)
            std::fprintf(stderr, "gpu: BB_RES_SCALE=%u needs 148 bytes of push constants, the device has %u: scale 1\n", v, vk().props.limits.maxPushConstantsSize);
            v = 1;
        }
        if (v > 1) std::fprintf(stderr, "gpu: internal resolution scale %u\n", v);
        return v;
    }();
    return s;
}

void backend_request_replay() { g_replay_req = true; }
bool backend_replay_supported() { return !std::getenv("BB_GPU_PROF") && !std::getenv("BB_GPU_TS"); }  // (see replay_on)

bool backend_replay_tick(uint64_t addr, uint32_t width, uint32_t height, const VkImage* dst, VkExtent2D dst_extent, uint32_t n, const float* alpha, ReplayWarp* warp,
                         uint64_t* ready, ReplayResult* res) {
    if (!g_be || !g_be->replay_on) return false;
    std::lock_guard<std::mutex> lk(g_be->mtx);
    return g_be->replay_tick(addr, width, height, dst, dst_extent, n, alpha, g_be->replay_warp ? warp : nullptr, ready, res);
}
VkSemaphore backend_replay_sem() { return g_be ? g_be->ready_sem : VK_NULL_HANDLE; }  // (created before the first `ready` value is handed out)

double backend_replay_gpu_ms() { return g_be ? g_be->replay_gpu_ms.load() : 0.0; }
double backend_replay_hold_ms() { return g_be ? g_be->replay_hold_ms.load() : 0.0; }

}  // namespace bb::gpu
