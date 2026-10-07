// SPDX-License-Identifier: GPL-3.0-or-later
// Presentation: the guest's flipped display buffer is captured into a small ring of snapshot images; the window shows the newest snapshot
// (BB_FPS=native: as soon as it arrives, the game's own pace; BB_FPS=30: on a 30 Hz clock). Above 30 (a rate, unlimited, refresh) the
// interpolation hybrid: up to rate/30 images per game frame re-rendered by the backend with camera and objects in between
// (backend_replay_tick), the remaining presents the nearest rendered image reprojected by depth (warp_one); BB_INTERP=0 repeats images
// instead. The game itself keeps running at its own pace: only the picture is resampled in time.
#include "gpu/present.h"
#include "gpu/backend.h"

#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <vk_mem_alloc.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "gpu/pad.h"
#include "gpu/spirv_builder.h"
#include "gpu/vk_context.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <volk.h>
#include <spirv/unified1/GLSL.std.450.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace bb::gpu {
namespace {

using Clock = std::chrono::steady_clock;

struct Snap {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    // reprojection source (BB_INTERP_WARP): the display pass's source before / after the HUD (scene / pre, see ReplayWarp) and the main
    // camera's depth (w * h floats)
    VkImage scene = VK_NULL_HANDLE, pre = VK_NULL_HANDLE;
    VmaAllocation scene_alloc = nullptr, pre_alloc = nullptr;
    VkImageView scene_view = VK_NULL_HANDLE, pre_view = VK_NULL_HANDLE;
    VkBuffer depth = VK_NULL_HANDLE;
    VmaAllocation depth_alloc = nullptr;
};

constexpr uint32_t kPackets = 4;  // replay packets in rotation (the hybrid, see State)
constexpr uint32_t kMaxImages = 8;  // per packet: the replays (alpha ascending) and last the tick's own picture (alpha 1)
struct Packet {
    Clock::time_point t{};  // window start
    double tick = 1.0 / 30;
    uint64_t frame = 0;
    uint32_t n = 0;
    float a[kMaxImages] = {};
    int idx[kMaxImages] = {};
    bool warp = false;  // the main camera (backend ReplayWarp): its block's dwords 8-71 in the previous / this tick, its viewport
    float cam0[64] = {}, cam1[64] = {}, vport[6] = {};
    uint64_t ready[kMaxImages] = {};  // (presenter queue) the backend's replay semaphore value per image (backend_replay_tick)
    bool cut = false;  // camera cut: every image is the tick's own picture, shown for the whole window (no alpha 0 = previous picture, no warp)
};

struct State {
    SDL_Window* window = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent{};
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    VkCommandPool pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> cmds;
    VkSemaphore image_ready = VK_NULL_HANDLE;
    std::vector<VkSemaphore> render_done;  // per swapchain image: the present may still hold the previous image's semaphore
    VkFence in_flight = VK_NULL_HANDLE;
    std::atomic<uint64_t> requested{0};
    uint64_t captured = 0;
    std::atomic<uint32_t> last_buffer{0};
    std::atomic<uint64_t> last_addr{0};
    std::atomic<uint32_t> last_w{1920}, last_h{1080};
    // frame-rate options: fps < 0 = native (present each game frame when it arrives; -2: at the display refresh, FIFO, with the hybrid),
    // 0 = unlimited, else the target rate
    int fps = -1;
    static constexpr uint32_t kMaxSnaps = 33;  // (kPackets * kMaxImages packet images + the reprojection target)
    Snap snaps[kMaxSnaps];
    uint32_t nsnaps = 3;
    uint32_t snap_w = 0, snap_h = 0;
    // BB_INTERP_WARP (default on with the hybrid): a present between rendered images shows the nearest one reprojected to its alpha
    // (depth-based, warp_one): snaps[nsnaps] is the target, warp_depth its depth buffer.
    bool warp = false;
    VkImage warp_depth = VK_NULL_HANDLE;
    VmaAllocation warp_depth_alloc = nullptr;
    VkImageView warp_depth_view = VK_NULL_HANDLE;
    VkDescriptorSetLayout warp_dsl = VK_NULL_HANDLE;
    VkPipelineLayout warp_layout = VK_NULL_HANDLE;
    VkPipeline warp_pipeline = VK_NULL_HANDLE, warp_pipeline_torn = VK_NULL_HANDLE;  // (make_warp_fs without / with the search)
    VkDescriptorPool warp_dpool = VK_NULL_HANDLE;
    VkSampler nearest = VK_NULL_HANDLE;
    uint32_t warp_cell = 2;         // BB_INTERP_WARP_CELL: grid cell in pixels
    VkQueryPool warp_q = VK_NULL_HANDLE;  // GPU time of the last warp (BB_PRESENT_LOG)
    std::vector<double> warp_ms;
    uint64_t warps = 0;
    // Hybrid (replay): per guest frame ("tick") a packet of snapshots from backend_replay_tick. The tick's display window is
    // [t + delay, t + delay + tick_s): a present at time p shows the packet image with the largest alpha <= (p - t - delay) / tick_s (alpha 0 =
    // the previous packet's own picture). The alphas are the present times that fall into the window (as many as the GPU budget allows,
    // evenly picked), so every image covers whole presents. kPackets packets rotate through the snapshots (the presenter uses the newest two).
    uint32_t rep_max = 0;           // images per packet at most (0: no replay); rep_budget: what the GPU keeps up with now
    std::atomic<uint32_t> rep_budget{0};
    uint32_t budget_ok_ticks = 0;
    bool pq = false;                // presenter on its own queue (vk().queue2): images are used once their replay semaphore value is reached
    uint64_t sem_seen = 0;          // (presenter thread) that semaphore's value at the last pick_image
    bool fixed_n = false;           // BB_INTERP_N: always that many (no GPU budget; measurements)
    Packet packets[kPackets];
    uint64_t pk_seq = 0;            // packets published; packets[(pk_seq - 1) % kPackets] is the newest (pk_pub)
    std::mutex pk_snaps, pk_pub;    // pk_snaps: snapshot images (render thread writes them, the presenter recreates them)
    double tick_s = 1.0 / 30;       // guest frame interval (EMA, render thread)
    Clock::time_point last_flip{};
    double delay_ms = -1;           // BB_INTERP_DELAY_MS (default: max(one present period, replay GPU time + 2 ms))
    std::atomic<int64_t> clk_ref{0}, clk_period{0};  // present clock (ns since the steady-clock epoch): a recent present time and the period
    int shown = -1;
    uint64_t distinct = 0;          // presents that showed another image than the previous one (BB_PRESENT_LOG)
    std::vector<double> intervals;  // present intervals since the last BB_PRESENT_LOG line (ms)
    int cur = -1;                   // snapshot index of the newest game frame (or the replay mode's image to show)
    uint32_t write_idx = 0;
    // present pass
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    uint64_t presents = 0;  // frames shown on screen (BB_PRESENT_LOG)
    // BB_UPSCALE=fsr (default; linear = the present pass stretches bilinearly): while the window is larger than the game picture, AMD FSR 1
    // replaces the present pass: EASU (snapshot -> `up`, window size) then RCAS (`up` -> swapchain image,
    // BB_FSR_SHARP = stops below full sharpness, default 0.2). Shaders: fsr1_spv.inc (tools/gen_fsr1_spv.py).
    bool fsr = true;
    float fsr_sharp = 0.2f;
    VkPipelineLayout fsr_layout = VK_NULL_HANDLE;
    VkPipeline easu = VK_NULL_HANDLE, rcas = VK_NULL_HANDLE;
    VkImage up = VK_NULL_HANDLE;
    VmaAllocation up_alloc = nullptr;
    VkImageView up_view = VK_NULL_HANDLE;
    VkExtent2D up_ext{};
    VkQueryPool out_q = VK_NULL_HANDLE;  // GPU time of the last present's final pass (blend or EASU + RCAS; BB_PRESENT_LOG)
    bool out_q_used = false, out_fsr = false;  // out_fsr: the last final pass was FSR
    std::vector<double> out_ms;
    // BB_SHOT_OUT=1: single shots (BB_SHOT_AT/_EVERY/_N outside replay mode) save the presented swapchain image (window size, after the
    // upscale) instead of the game snapshot. The copy is recorded into the next present; the file is written once its fence passed.
    bool shot_out = false, out_ok = false;  // out_ok: the swapchain images allow TRANSFER_SRC
    std::string out_path, out_pending;
    VkBuffer out_buf = VK_NULL_HANDLE;
    VmaAllocation out_alloc = nullptr;
    VmaAllocationInfo out_info{};
    VkExtent2D out_ext{};
} g;

bool vk_ok(VkResult r, const char* what) { return vk_check(r, what); }

// ---- present pass shaders ----------------------------------------------------------------------------------------------
// VS: full-screen triangle from the vertex index. FS: the snapshot (binding 0; the set layout is shared with FSR's EASU pass).
std::vector<uint32_t> make_vs() {
    SpvBuilder b;
    b.capability(spv::Capability::Shader);
    const auto F = b.t_f32(), U = b.t_u32(), I = b.t_i32(), V2 = b.t_vec(F, 2), V4 = b.t_vec(F, 4);
    const auto vi = b.global_var(b.t_ptr(spv::StorageClass::Input, I), spv::StorageClass::Input);
    b.decorate(vi, spv::Decoration::BuiltIn, {uint32_t(spv::BuiltIn::VertexIndex)});
    const auto pos = b.global_var(b.t_ptr(spv::StorageClass::Output, V4), spv::StorageClass::Output);
    b.decorate(pos, spv::Decoration::BuiltIn, {uint32_t(spv::BuiltIn::Position)});
    const auto uv = b.global_var(b.t_ptr(spv::StorageClass::Output, V2), spv::StorageClass::Output);
    b.decorate(uv, spv::Decoration::Location, {0});
    const auto fn = b.begin_function(b.t_void(), b.t_fn(b.t_void()));
    const auto idx = b.op(spv::Op::OpBitcast, U, {b.op(spv::Op::OpLoad, I, {vi})});
    const auto x = b.op(spv::Op::OpBitwiseAnd, U, {b.op(spv::Op::OpShiftLeftLogical, U, {idx, b.c_u32(1)}), b.c_u32(2)});
    const auto y = b.op(spv::Op::OpBitwiseAnd, U, {idx, b.c_u32(2)});
    const auto fx = b.op(spv::Op::OpConvertUToF, F, {x}), fy = b.op(spv::Op::OpConvertUToF, F, {y});
    const auto px = b.op(spv::Op::OpFSub, F, {b.op(spv::Op::OpFMul, F, {fx, b.c_f32(2.f)}), b.c_f32(1.f)});
    const auto py = b.op(spv::Op::OpFSub, F, {b.op(spv::Op::OpFMul, F, {fy, b.c_f32(2.f)}), b.c_f32(1.f)});
    b.op0(spv::Op::OpStore, {pos, b.op(spv::Op::OpCompositeConstruct, V4, {px, py, b.c_f32(0.f), b.c_f32(1.f)})});
    b.op0(spv::Op::OpStore, {uv, b.op(spv::Op::OpCompositeConstruct, V2, {fx, fy})});
    b.op0(spv::Op::OpReturn, {});
    b.end_function();
    b.entry_point(spv::ExecutionModel::Vertex, fn, "main", {vi, pos, uv});
    return b.finish();
}

std::vector<uint32_t> make_fs() {
    SpvBuilder b;
    b.capability(spv::Capability::Shader);
    const auto F = b.t_f32(), V2 = b.t_vec(F, 2), V4 = b.t_vec(F, 4);
    const auto img = b.t_image(F, spv::Dim::Dim2D, false, 1), simg = b.t_sampled_image(img);
    auto uc = [&](uint32_t binding, uint32_t type) {
        const auto v = b.global_var(b.t_ptr(spv::StorageClass::UniformConstant, type), spv::StorageClass::UniformConstant);
        b.decorate(v, spv::Decoration::DescriptorSet, {0});
        b.decorate(v, spv::Decoration::Binding, {binding});
        return v;
    };
    const auto t0 = uc(0, img), smp = uc(1, b.t_sampler());
    const auto in_uv = b.global_var(b.t_ptr(spv::StorageClass::Input, V2), spv::StorageClass::Input);
    b.decorate(in_uv, spv::Decoration::Location, {0});
    const auto out = b.global_var(b.t_ptr(spv::StorageClass::Output, V4), spv::StorageClass::Output);
    b.decorate(out, spv::Decoration::Location, {0});
    const auto fn = b.begin_function(b.t_void(), b.t_fn(b.t_void()));
    const auto uv = b.op(spv::Op::OpLoad, V2, {in_uv});
    const auto s = b.op(spv::Op::OpLoad, b.t_sampler(), {smp});
    b.op0(spv::Op::OpStore, {out, b.op(spv::Op::OpImageSampleImplicitLod, V4, {b.op(spv::Op::OpSampledImage, simg, {b.op(spv::Op::OpLoad, img, {t0}), s}), uv})});
    b.op0(spv::Op::OpReturn, {});
    b.end_function();
    b.entry_point(spv::ExecutionModel::Fragment, fn, "main", {in_uv, out, t0, smp});  // SPIR-V >= 1.4: every used global
    b.exec_mode(fn, spv::ExecutionMode::OriginUpperLeft);
    return b.finish();
}

VkShaderModule module_of(const std::vector<uint32_t>& code) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size() * 4;
    ci.pCode = code.data();
    VkShaderModule m = VK_NULL_HANDLE;
    vk_ok(vkCreateShaderModule(vk().device, &ci, nullptr, &m), "blend shader module");
    return m;
}

#include "gpu/fsr1_spv.inc"

// A full-screen triangle pipeline (make_vs) with fragment shader `fs` into one g.format attachment; consumes `fs`.
bool fullscreen_pipeline(VkShaderModule fs, VkPipelineLayout layout, VkPipeline& out, const char* what) {
    VkDevice dev = vk().device;
    VkShaderModule vs = module_of(make_vs());
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", nullptr}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbs.attachmentCount = 1;
    cbs.pAttachments = &cba;
    const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dsi.dynamicStateCount = 2;
    dsi.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &g.format;
    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.pNext = &rci;
    gp.stageCount = 2;
    gp.pStages = st;
    gp.pVertexInputState = &vi; gp.pInputAssemblyState = &ia; gp.pViewportState = &vp; gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms; gp.pColorBlendState = &cbs; gp.pDynamicState = &dsi;
    gp.layout = layout;
    const bool ok = vk_ok(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &out), what);
    vkDestroyShaderModule(dev, vs, nullptr);
    vkDestroyShaderModule(dev, fs, nullptr);
    return ok;
}

// FSR push constants (fsr1_spv.inc): EASU con0..con3; RCAS con0 only
struct FsrPc { uint32_t con[16] = {}; };
uint32_t f32_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
// FsrEasuCon of ffx_fsr1.h (MIT, see fsr1_spv.inc): the whole `in` image upscaled to `out`
void easu_con(FsrPc& pc, float in_w, float in_h, float out_w, float out_h) {
    const float c[16] = {in_w / out_w, in_h / out_h, 0.5f * in_w / out_w - 0.5f, 0.5f * in_h / out_h - 0.5f,
                         1.0f / in_w, 1.0f / in_h, 1.0f / in_w, -1.0f / in_h,
                         -1.0f / in_w, 2.0f / in_h, 1.0f / in_w, 2.0f / in_h,
                         0.0f, 4.0f / in_h, 0.0f, 0.0f};
    for (int i = 0; i < 16; ++i) pc.con[i] = f32_bits(c[i]);
}
// FsrRcasCon: sharpness in stops (0 = maximum); con[1] (packed half pair) is only read by the 16-bit variant
void rcas_con(FsrPc& pc, float stops) { pc.con[0] = f32_bits(std::exp2(-stops)); }

bool create_blend_pipeline() {
    VkDevice dev = vk().device;
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (!vk_ok(vkCreateSampler(dev, &sci, nullptr, &g.sampler), "blend sampler")) return false;
    const VkDescriptorSetLayoutBinding bind[2] = {{0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                                                  {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 2;
    dli.pBindings = bind;
    if (!vk_ok(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &g.dsl), "blend set layout")) return false;
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g.dsl;
    if (!vk_ok(vkCreatePipelineLayout(dev, &pli, nullptr, &g.layout), "present pipeline layout")) return false;
    const VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4}, {VK_DESCRIPTOR_TYPE_SAMPLER, 4}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 4;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = sizes;
    if (!vk_ok(vkCreateDescriptorPool(dev, &dpi, nullptr, &g.dpool), "blend descriptor pool")) return false;
    if (!fullscreen_pipeline(module_of(make_fs()), g.layout, g.pipeline, "present pipeline")) return false;
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 2;
    if (!vk_ok(vkCreateQueryPool(dev, &qi, nullptr, &g.out_q), "present query pool")) return false;
    if (!g.fsr) return true;
    const VkPushConstantRange fpr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(FsrPc)};
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &fpr;
    if (!vk_ok(vkCreatePipelineLayout(dev, &pli, nullptr, &g.fsr_layout), "FSR pipeline layout")) return false;
    return fullscreen_pipeline(module_of({std::begin(kFsrEasuFs), std::end(kFsrEasuFs)}), g.fsr_layout, g.easu, "FSR EASU pipeline") &&
           fullscreen_pipeline(module_of({std::begin(kFsrRcasFs), std::end(kFsrRcasFs)}), g.fsr_layout, g.rcas, "FSR RCAS pipeline");
}

// ---- reprojection (BB_INTERP_WARP) ----------------------------------------------------------------------------------------
// A forward warp drawn as a grid mesh: one quad (two triangles) per warp_cell x warp_cell pixels of the source image; each vertex reads the
// source depth at its corner and is transformed by M (source pixel + depth -> target clip space, see warp_matrices), the fragment samples
// the source picture at the vertex's source position. The depth test (target view depth) lets the nearest surface win where the warp folds.
// Where it tears (a disocclusion: the cell's corners differ in depth by more than the push constant's ratio) the cell's triangles stretch
// over the gap and the fragment shader searches the source pixel per fragment (make_warp_fs: foreground and background stay pixel-exact,
// the uncovered rest takes the background behind the edge). A gather for every pixel would cost the search everywhere; the mesh does the
// bulk at a fixed cost (W/cell * H/cell quads) and the search runs only on the few torn cells. Push constants: rows 0, 1, 3 of M, the
// source's inverse-depth row (1/view depth = r3 . (x, y, d, 1)), grid width in cells, cell size, source width and height, disocclusion ratio,
// pass (0: the cells that do not tear, 1: those that do, with the searching fragment shader; the other kind is collapsed).
struct WarpPc { float m0[4], m1[4], m3[4], r3[4]; uint32_t gw, cell, w, h; float disc; uint32_t pass; };  // the warp shaders' push constants
static_assert(sizeof(WarpPc) == 88);
// The push-constant block (WarpPc) and the source depth buffer (binding 0), shared by the warp shaders.
void warp_decls(SpvBuilder& b, uint32_t& pc, uint32_t& sb) {
    using SC = spv::StorageClass;
    const auto F = b.t_f32(), U = b.t_u32(), V4 = b.t_vec(F, 4);
    const auto pcs = b.t_struct({V4, V4, V4, V4, U, U, U, U, F, U});
    b.decorate(pcs, spv::Decoration::Block);
    const uint32_t offs[10] = {0, 16, 32, 48, 64, 68, 72, 76, 80, 84};
    for (uint32_t i = 0; i < 10; ++i) b.member_decorate(pcs, i, spv::Decoration::Offset, {offs[i]});
    pc = b.global_var(b.t_ptr(SC::PushConstant, pcs), SC::PushConstant);
    const auto arr = b.t_rtarray(F);
    b.decorate(arr, spv::Decoration::ArrayStride, {4});
    const auto sst = b.t_struct({arr});
    b.decorate(sst, spv::Decoration::Block);
    b.member_decorate(sst, 0, spv::Decoration::Offset, {0});
    b.member_decorate(sst, 0, spv::Decoration::NonWritable);
    sb = b.global_var(b.t_ptr(SC::StorageBuffer, sst), SC::StorageBuffer);
    b.decorate(sb, spv::Decoration::DescriptorSet, {0});
    b.decorate(sb, spv::Decoration::Binding, {0});
}
std::vector<uint32_t> make_warp_vs() {
    SpvBuilder b;
    b.capability(spv::Capability::Shader);
    using SC = spv::StorageClass;
    using O = spv::Op;
    const auto F = b.t_f32(), U = b.t_u32(), I = b.t_i32(), V2 = b.t_vec(F, 2), V4 = b.t_vec(F, 4), B = b.t_bool();
    uint32_t pc, sb;
    warp_decls(b, pc, sb);
    const auto vi = b.global_var(b.t_ptr(SC::Input, I), SC::Input);
    b.decorate(vi, spv::Decoration::BuiltIn, {uint32_t(spv::BuiltIn::VertexIndex)});
    const auto pos = b.global_var(b.t_ptr(SC::Output, V4), SC::Output);
    b.decorate(pos, spv::Decoration::BuiltIn, {uint32_t(spv::BuiltIn::Position)});
    const auto uv = b.global_var(b.t_ptr(SC::Output, V2), SC::Output);
    b.decorate(uv, spv::Decoration::Location, {0});
    const auto nf_out = b.global_var(b.t_ptr(SC::Output, V4), SC::Output);  // torn cells: the nearest and the farthest corner (source pixels)
    b.decorate(nf_out, spv::Decoration::Location, {1});
    b.decorate(nf_out, spv::Decoration::Flat);
    const auto fn = b.begin_function(b.t_void(), b.t_fn(b.t_void()));
    auto pcv = [&](uint32_t m, uint32_t t) { return b.op(O::OpLoad, t, {b.op(O::OpAccessChain, b.t_ptr(SC::PushConstant, t), {pc, b.c_u32(m)})}); };
    const auto m0 = pcv(0, V4), m1 = pcv(1, V4), m3 = pcv(2, V4), r3 = pcv(3, V4), gw = pcv(4, U), cell = pcv(5, U), w = pcv(6, U), h = pcv(7, U), disc = pcv(8, F);
    const auto torn_pass = b.op(O::OpINotEqual, B, {pcv(9, U), b.c_u32(0)});
    const auto vid = b.op(O::OpBitcast, U, {b.op(O::OpLoad, I, {vi})});
    const auto cid = b.op(O::OpUDiv, U, {vid, b.c_u32(6)}), k = b.op(O::OpUMod, U, {vid, b.c_u32(6)});
    const auto cx = b.op(O::OpUMod, U, {cid, gw}), cy = b.op(O::OpUDiv, U, {cid, gw});
    auto keq = [&](uint32_t v) { return b.op(O::OpIEqual, B, {k, b.c_u32(v)}); };
    auto lor = [&](uint32_t x, uint32_t y) { return b.op(O::OpLogicalOr, B, {x, y}); };
    const auto ox = lor(lor(keq(1), keq(3)), keq(4)), oy = lor(lor(keq(2), keq(4)), keq(5));  // triangles (0,0)(1,0)(0,1) and (1,0)(1,1)(0,1)
    const auto w1 = b.op(O::OpISub, U, {w, b.c_u32(1)}), h1 = b.op(O::OpISub, U, {h, b.c_u32(1)});
    const auto half = b.c_f32(0.5f), one = b.c_f32(1.0f);
    uint32_t v[4], iw[4];  // corners (0,0) (1,0) (0,1) (1,1): source pixel centre, depth, 1; inverse view depth
    for (uint32_t c = 0; c < 4; ++c) {
        const auto px = b.glsl(U, GLSLstd450UMin, {b.op(O::OpIMul, U, {b.op(O::OpIAdd, U, {cx, b.c_u32(c & 1)}), cell}), w1});
        const auto py = b.glsl(U, GLSLstd450UMin, {b.op(O::OpIMul, U, {b.op(O::OpIAdd, U, {cy, b.c_u32(c >> 1)}), cell}), h1});
        const auto at = b.op(O::OpIAdd, U, {b.op(O::OpIMul, U, {py, w}), px});
        const auto d = b.op(O::OpLoad, F, {b.op(O::OpAccessChain, b.t_ptr(SC::StorageBuffer, F), {sb, b.c_u32(0), at})});
        v[c] = b.op(O::OpCompositeConstruct, V4, {b.op(O::OpFAdd, F, {b.op(O::OpConvertUToF, F, {px}), half}), b.op(O::OpFAdd, F, {b.op(O::OpConvertUToF, F, {py}), half}), d, one});
        iw[c] = b.op(O::OpDot, F, {r3, v[c]});
    }
    auto sel = [&](uint32_t t, uint32_t cond, uint32_t x, uint32_t y) { return b.op(O::OpSelect, t, {cond, x, y}); };
    const auto vown = sel(V4, oy, sel(V4, ox, v[3], v[2]), sel(V4, ox, v[1], v[0]));
    const auto iwown = sel(F, oy, sel(F, ox, iw[3], iw[2]), sel(F, ox, iw[1], iw[0]));
    uint32_t vf = v[0], iwf = iw[0], vn = v[0], iwn = iw[0];  // the farthest / nearest corner (smallest / largest inverse depth)
    for (uint32_t c = 1; c < 4; ++c) {
        const auto lt = b.op(O::OpFOrdLessThan, B, {iw[c], iwf}), gt = b.op(O::OpFOrdGreaterThan, B, {iw[c], iwn});
        vf = sel(V4, lt, v[c], vf);
        iwf = sel(F, lt, iw[c], iwf);
        vn = sel(V4, gt, v[c], vn);
        iwn = sel(F, gt, iw[c], iwn);
    }
    // the ring of corners around the cell too: cells beside a disocclusion edge count as torn (the search pass then keeps the edge's
    // anti-aliased rim texels out of the background, see make_warp_fs)
    uint32_t rmin = iwf, rmax = iwn;
    const auto cell_i = b.op(O::OpBitcast, I, {cell}), w1_i = b.op(O::OpBitcast, I, {w1}), h1_i = b.op(O::OpBitcast, I, {h1});
    const auto cx_i = b.op(O::OpBitcast, I, {cx}), cy_i = b.op(O::OpBitcast, I, {cy});
    for (int dy = -1; dy <= 2; ++dy)
        for (int dx = -1; dx <= 2; ++dx) {
            if (dx >= 0 && dx <= 1 && dy >= 0 && dy <= 1) continue;
            const auto px = b.glsl(I, GLSLstd450SClamp, {b.op(O::OpIMul, I, {b.op(O::OpIAdd, I, {cx_i, b.c_i32(dx)}), cell_i}), b.c_i32(0), w1_i});
            const auto py = b.glsl(I, GLSLstd450SClamp, {b.op(O::OpIMul, I, {b.op(O::OpIAdd, I, {cy_i, b.c_i32(dy)}), cell_i}), b.c_i32(0), h1_i});
            const auto at = b.op(O::OpBitcast, U, {b.op(O::OpIAdd, I, {b.op(O::OpIMul, I, {py, b.op(O::OpBitcast, I, {w})}), px})});
            const auto d = b.op(O::OpLoad, F, {b.op(O::OpAccessChain, b.t_ptr(SC::StorageBuffer, F), {sb, b.c_u32(0), at})});
            const auto vr = b.op(O::OpCompositeConstruct, V4, {b.op(O::OpFAdd, F, {b.op(O::OpConvertSToF, F, {px}), half}), b.op(O::OpFAdd, F, {b.op(O::OpConvertSToF, F, {py}), half}), d, one});
            const auto iwr = b.op(O::OpDot, F, {r3, vr});
            rmin = b.glsl(F, GLSLstd450FMin, {rmin, iwr});
            rmax = b.glsl(F, GLSLstd450FMax, {rmax, iwr});
        }
    const auto torn = b.op(O::OpLogicalOr, B, {b.op(O::OpFOrdGreaterThan, B, {iwn, b.op(O::OpFMul, F, {iwf, disc})}),
                                               b.op(O::OpFOrdGreaterThan, B, {rmax, b.op(O::OpFMul, F, {rmin, disc})})});
    const auto clx = b.op(O::OpDot, F, {m0, vown}), cly = b.op(O::OpDot, F, {m1, vown}), clw = b.op(O::OpDot, F, {m3, vown});
    const auto zt = b.glsl(F, GLSLstd450FMax, {b.op(O::OpFDiv, F, {clw, b.glsl(F, GLSLstd450FMax, {iwown, b.c_f32(1e-30f)})}), b.c_f32(0.0f)});  // target view depth
    const auto z = b.glsl(F, GLSLstd450FClamp, {b.op(O::OpFSub, F, {one, b.op(O::OpFDiv, F, {one, b.op(O::OpFAdd, F, {one, zt})})}), b.c_f32(0.0f), one});
    const auto skip = b.op(O::OpLogicalNotEqual, B, {torn, torn_pass});  // the other pass's cell: collapsed to a point outside the view
    const auto off = b.op(O::OpCompositeConstruct, V4, {b.c_f32(2.0f), b.c_f32(2.0f), b.c_f32(0.0f), one});
    b.op0(O::OpStore, {pos, sel(V4, skip, off, b.op(O::OpCompositeConstruct, V4, {clx, cly, b.op(O::OpFMul, F, {z, clw}), clw}))});
    const auto size = b.op(O::OpCompositeConstruct, V2, {b.op(O::OpConvertUToF, F, {w}), b.op(O::OpConvertUToF, F, {h})});
    b.op0(O::OpStore, {uv, b.op(O::OpFDiv, V2, {b.op(O::OpVectorShuffle, V2, {vown, vown, 0, 1}), size})});
    b.op0(O::OpStore, {nf_out, b.op(O::OpVectorShuffle, V4, {vn, vf, 0, 1, 4, 5})});
    b.op0(O::OpReturn, {});
    b.end_function();
    b.entry_point(spv::ExecutionModel::Vertex, fn, "main", {vi, pos, uv, nf_out, pc, sb});
    return b.finish();
}

// FS (search = false): the source's final picture at the warped position (pass 0, early depth test). Torn cells (pass 1, search = true:
// their triangles stretch over a disocclusion) search per fragment for the source pixel that lands on it: from the cell's nearest corner,
// from the stretched position and from the farthest corner, three steps each of s += fragment - warp(texel(s)); a start whose texel
// lands within 0.75 px counts, the nearest one wins (colour and depth of that texel: silhouettes stay pixel-exact, no cell-sized
// staircase). A texel beside a much nearer one (the anti-aliased rim of a foreground edge, half its colour) counts only as a last resort
// and with the colour 2 texels further out, so the rim does not stay behind as a line when the foreground moves on. None (the uncovered
// gap): the background 1.5 px beyond the farthest corner at its depth, written per fragment.
// The HUD (screen-fixed, its own pixels' colours) comes from the display pass's source before / after the HUD's draws (scene / pre: the
// same colour space, they differ where the HUD is): at a target pixel the HUD covers, the source's final picture at that very pixel;
// where the source position lies under the HUD (the scene behind it is unknown) the same (the unwarped picture there, no HUD ghost).
std::vector<uint32_t> make_warp_fs(bool search) {
    SpvBuilder b;
    b.capability(spv::Capability::Shader);
    b.capability(spv::Capability::ImageQuery);
    using SC = spv::StorageClass;
    using O = spv::Op;
    const auto F = b.t_f32(), U = b.t_u32(), I = b.t_i32(), V2 = b.t_vec(F, 2), V4 = b.t_vec(F, 4), IV2 = b.t_vec(I, 2), B = b.t_bool();
    const auto img = b.t_image(F, spv::Dim::Dim2D, false, 1), simg = b.t_sampled_image(img);
    auto uc = [&](uint32_t binding, uint32_t type) {
        const auto var = b.global_var(b.t_ptr(SC::UniformConstant, type), SC::UniformConstant);
        b.decorate(var, spv::Decoration::DescriptorSet, {0});
        b.decorate(var, spv::Decoration::Binding, {binding});
        return var;
    };
    const auto tS = uc(1, img), tF = uc(2, img), smp = uc(3, b.t_sampler()), tP = uc(4, img);
    uint32_t pc, sb;
    warp_decls(b, pc, sb);
    const auto in_uv = b.global_var(b.t_ptr(SC::Input, V2), SC::Input);
    b.decorate(in_uv, spv::Decoration::Location, {0});
    const auto in_nf = b.global_var(b.t_ptr(SC::Input, V4), SC::Input);
    b.decorate(in_nf, spv::Decoration::Location, {1});
    b.decorate(in_nf, spv::Decoration::Flat);
    const auto fc = b.global_var(b.t_ptr(SC::Input, V4), SC::Input);
    b.decorate(fc, spv::Decoration::BuiltIn, {uint32_t(spv::BuiltIn::FragCoord)});
    const auto out = b.global_var(b.t_ptr(SC::Output, V4), SC::Output);
    b.decorate(out, spv::Decoration::Location, {0});
    uint32_t fdepth = 0;
    if (search) {
        fdepth = b.global_var(b.t_ptr(SC::Output, F), SC::Output);
        b.decorate(fdepth, spv::Decoration::BuiltIn, {uint32_t(spv::BuiltIn::FragDepth)});
    }
    const auto fn = b.begin_function(b.t_void(), b.t_fn(b.t_void()));
    auto sel = [&](uint32_t t, uint32_t cond, uint32_t x, uint32_t y) { return b.op(O::OpSelect, t, {cond, x, y}); };
    auto fetch = [&](uint32_t im, uint32_t at) { return b.op(O::OpImageFetch, V4, {im, at, uint32_t(spv::ImageOperandsMask::Lod), b.c_i32(0)}); };
    auto x_of = [&](uint32_t v2) { return b.op(O::OpCompositeExtract, F, {v2, 0}); };
    auto y_of = [&](uint32_t v2) { return b.op(O::OpCompositeExtract, F, {v2, 1}); };
    const auto s_img = b.op(O::OpLoad, img, {tS}), f_img = b.op(O::OpLoad, img, {tF}), p_img = b.op(O::OpLoad, img, {tP});
    const auto uv = b.op(O::OpLoad, V2, {in_uv});
    const auto fcv = b.op(O::OpLoad, V4, {fc});
    const auto fxy = b.op(O::OpVectorShuffle, V2, {fcv, fcv, 0, 1});
    const auto size_i = b.op(O::OpImageQuerySizeLod, IV2, {f_img, b.c_i32(0)});
    const auto size = b.op(O::OpConvertSToF, V2, {size_i});
    uint32_t c = 0, s = 0, depth = 0;
    if (!search) {
        c = b.op(O::OpImageSampleImplicitLod, V4, {b.op(O::OpSampledImage, simg, {f_img, b.op(O::OpLoad, b.t_sampler(), {smp})}), uv});
        s = b.op(O::OpFMul, V2, {uv, size});
    } else {
        const auto s_n = b.op(O::OpFMul, V2, {uv, size});
        auto pcv = [&](uint32_t m, uint32_t t) { return b.op(O::OpLoad, t, {b.op(O::OpAccessChain, b.t_ptr(SC::PushConstant, t), {pc, b.c_u32(m)})}); };
        const auto m0 = pcv(0, V4), m1 = pcv(1, V4), m3 = pcv(2, V4), r3 = pcv(3, V4), w = pcv(6, U);
        const auto nf = b.op(O::OpLoad, V4, {in_nf});
        const auto s_near = b.op(O::OpVectorShuffle, V2, {nf, nf, 0, 1}), s_far = b.op(O::OpVectorShuffle, V2, {nf, nf, 2, 3});
        const auto half2 = b.op(O::OpCompositeConstruct, V2, {b.c_f32(0.5f), b.c_f32(0.5f)});
        const auto zero2 = b.op(O::OpCompositeConstruct, V2, {b.c_f32(0.0f), b.c_f32(0.0f)});
        const auto max2 = b.op(O::OpFSub, V2, {size, b.op(O::OpCompositeConstruct, V2, {b.c_f32(1.0f), b.c_f32(1.0f)})});
        auto texel = [&](uint32_t at) { return b.glsl(V2, GLSLstd450FClamp, {b.glsl(V2, GLSLstd450Floor, {at}), zero2, max2}); };
        struct Warped { uint32_t t, clw, iw; };
        auto warp_texel = [&](uint32_t p) {  // texel centre p + 0.5 with its source depth -> target pixel position, clip w, inverse view depth
            const auto at = b.op(O::OpIAdd, U, {b.op(O::OpIMul, U, {b.op(O::OpConvertFToU, U, {y_of(p)}), w}), b.op(O::OpConvertFToU, U, {x_of(p)})});
            const auto d = b.op(O::OpLoad, F, {b.op(O::OpAccessChain, b.t_ptr(SC::StorageBuffer, F), {sb, b.c_u32(0), at})});
            const auto pc2 = b.op(O::OpFAdd, V2, {p, half2});
            const auto v = b.op(O::OpCompositeConstruct, V4, {x_of(pc2), y_of(pc2), d, b.c_f32(1.0f)});
            const auto clw = b.op(O::OpDot, F, {m3, v});
            const auto ndc = b.op(O::OpFDiv, V2, {b.op(O::OpCompositeConstruct, V2, {b.op(O::OpDot, F, {m0, v}), b.op(O::OpDot, F, {m1, v})}),
                                                   b.op(O::OpCompositeConstruct, V2, {clw, clw})});
            const auto t = b.op(O::OpFMul, V2, {b.op(O::OpFAdd, V2, {b.op(O::OpFMul, V2, {ndc, half2}), half2}), size});
            return Warped{t, clw, b.op(O::OpDot, F, {r3, v})};
        };
        const auto disc = pcv(8, F);
        auto iw_at = [&](uint32_t p) {  // inverse view depth of texel p
            const auto at = b.op(O::OpIAdd, U, {b.op(O::OpIMul, U, {b.op(O::OpConvertFToU, U, {y_of(p)}), w}), b.op(O::OpConvertFToU, U, {x_of(p)})});
            const auto d = b.op(O::OpLoad, F, {b.op(O::OpAccessChain, b.t_ptr(SC::StorageBuffer, F), {sb, b.c_u32(0), at})});
            const auto pc2 = b.op(O::OpFAdd, V2, {p, half2});
            return b.op(O::OpDot, F, {r3, b.op(O::OpCompositeConstruct, V4, {x_of(pc2), y_of(pc2), d, b.c_f32(1.0f)})});
        };
        auto found = b.c_bool(false), rim_found = b.c_bool(false);
        uint32_t best_iw = b.c_f32(-1e30f), best_clw = b.c_f32(1.0f), best_p = zero2;
        uint32_t rim_iw = b.c_f32(-1e30f), rim_clw = b.c_f32(1.0f), rim_cp = zero2;  // best rim texel: its depth, the colour 2 texels away from the foreground
        for (const uint32_t start : {s_near, s_n, s_far}) {
            uint32_t sp = start, p = 0;
            Warped wp{};
            for (int it = 0; it < 4; ++it) {
                p = texel(sp);
                wp = warp_texel(p);
                if (it < 3) sp = b.op(O::OpFAdd, V2, {b.op(O::OpFAdd, V2, {p, half2}), b.op(O::OpFSub, V2, {fxy, wp.t})});
            }
            // a texel beside a much nearer one is the anti-aliased rim of that foreground (its colour half foreground): it keeps its depth
            // but takes the colour 2 texels further away from the foreground
            auto rim = b.c_bool(false);
            uint32_t away = zero2;
            static constexpr float kNb[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (const auto& o : kNb) {
                const auto off = b.op(O::OpCompositeConstruct, V2, {b.c_f32(o[0]), b.c_f32(o[1])});
                const auto nearer = b.op(O::OpFOrdGreaterThan, B, {iw_at(texel(b.op(O::OpFAdd, V2, {p, off}))), b.op(O::OpFMul, F, {wp.iw, disc})});
                rim = b.op(O::OpLogicalOr, B, {rim, nearer});
                away = sel(V2, nearer, b.op(O::OpFSub, V2, {away, off}), away);
            }
            const auto e = b.op(O::OpFSub, V2, {fxy, wp.t});
            const auto lands = b.op(O::OpLogicalAnd, B, {b.op(O::OpFOrdLessThan, B, {b.op(O::OpDot, F, {e, e}), b.c_f32(0.5625f)}),
                                                         b.op(O::OpLogicalAnd, B, {b.op(O::OpFOrdGreaterThan, B, {wp.clw, b.c_f32(0.0f)}),
                                                                                    b.op(O::OpFOrdGreaterThan, B, {wp.iw, b.c_f32(0.0f)})})});
            const auto valid = b.op(O::OpLogicalAnd, B, {lands, b.op(O::OpLogicalNot, B, {rim})});
            const auto rim_valid = b.op(O::OpLogicalAnd, B, {lands, rim});
            const auto better = b.op(O::OpLogicalAnd, B, {valid, b.op(O::OpFOrdGreaterThan, B, {wp.iw, best_iw})});
            best_iw = sel(F, better, wp.iw, best_iw);
            best_clw = sel(F, better, wp.clw, best_clw);
            best_p = sel(V2, better, p, best_p);
            found = b.op(O::OpLogicalOr, B, {found, valid});
            const auto rim_better = b.op(O::OpLogicalAnd, B, {rim_valid, b.op(O::OpFOrdGreaterThan, B, {wp.iw, rim_iw})});
            rim_iw = sel(F, rim_better, wp.iw, rim_iw);
            rim_clw = sel(F, rim_better, wp.clw, rim_clw);
            rim_cp = sel(V2, rim_better, texel(b.op(O::OpFAdd, V2, {p, b.op(O::OpVectorTimesScalar, V2, {away, b.c_f32(2.0f)})})), rim_cp);
            rim_found = b.op(O::OpLogicalOr, B, {rim_found, rim_valid});
        }
        const auto dir = b.op(O::OpFSub, V2, {s_far, s_near});
        const auto len = b.op(O::OpFAdd, F, {b.glsl(F, GLSLstd450Length, {dir}), b.c_f32(1e-6f)});
        const auto bg_p = texel(b.op(O::OpFAdd, V2, {s_far, b.op(O::OpVectorTimesScalar, V2, {dir, b.op(O::OpFDiv, F, {b.c_f32(1.5f), len})})}));
        const auto wf = warp_texel(texel(s_far));
        const auto tiny = b.c_f32(1e-30f);
        const auto zt_found = b.op(O::OpFDiv, F, {best_clw, b.glsl(F, GLSLstd450FMax, {best_iw, tiny})});
        const auto zt_rim = b.op(O::OpFDiv, F, {rim_clw, b.glsl(F, GLSLstd450FMax, {rim_iw, tiny})});
        const auto zt_far = b.op(O::OpFMul, F, {b.op(O::OpFDiv, F, {wf.clw, b.glsl(F, GLSLstd450FMax, {wf.iw, tiny})}), b.c_f32(1.002f)});
        const auto zt = b.glsl(F, GLSLstd450FMax, {sel(F, found, zt_found, sel(F, rim_found, zt_rim, zt_far)), b.c_f32(0.0f)});
        const auto one = b.c_f32(1.0f);
        depth = b.glsl(F, GLSLstd450FClamp, {b.op(O::OpFSub, F, {one, b.op(O::OpFDiv, F, {one, b.op(O::OpFAdd, F, {one, zt})})}), b.c_f32(0.0f), one});
        const auto p_t = sel(V2, found, best_p, sel(V2, rim_found, rim_cp, bg_p));
        c = fetch(f_img, b.op(O::OpConvertFToS, IV2, {p_t}));
        s = b.op(O::OpFAdd, V2, {p_t, half2});
    }
    const auto q = b.op(O::OpConvertFToS, IV2, {fxy});
    const auto one2 = b.op(O::OpCompositeConstruct, IV2, {b.c_i32(1), b.c_i32(1)}), izero2 = b.op(O::OpCompositeConstruct, IV2, {b.c_i32(0), b.c_i32(0)});
    const auto qs = b.glsl(IV2, GLSLstd450SClamp, {b.op(O::OpConvertFToS, IV2, {s}), izero2, b.op(O::OpISub, IV2, {size_i, one2})});
    auto hud_at = [&](uint32_t at) {  // pre and scene differ (a colour channel by more than 1.5 / 255)
        const auto dl = b.glsl(V4, GLSLstd450FAbs, {b.op(O::OpFSub, V4, {fetch(p_img, at), fetch(s_img, at)})});
        auto comp = [&](uint32_t i) { return b.op(O::OpCompositeExtract, F, {dl, i}); };
        const auto m = b.glsl(F, GLSLstd450FMax, {b.glsl(F, GLSLstd450FMax, {comp(0), comp(1)}), comp(2)});
        return b.op(O::OpFOrdGreaterThan, B, {m, b.c_f32(1.5f / 255.0f)});
    };
    const auto hud = b.op(O::OpLogicalOr, B, {hud_at(q), hud_at(qs)});
    b.op0(O::OpStore, {out, b.op(O::OpSelect, V4, {hud, fetch(f_img, q), c})});
    if (search) b.op0(O::OpStore, {fdepth, depth});
    b.op0(O::OpReturn, {});
    b.end_function();
    std::vector<uint32_t> io{tS, tF, smp, tP, pc, sb, in_uv, in_nf, fc, out};
    if (search) io.push_back(fdepth);
    b.entry_point(spv::ExecutionModel::Fragment, fn, "main", io);
    b.exec_mode(fn, spv::ExecutionMode::OriginUpperLeft);
    if (search) b.exec_mode(fn, spv::ExecutionMode::DepthReplacing);
    return b.finish();
}

bool create_warp_pipeline() {
    VkDevice dev = vk().device;
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (!vk_ok(vkCreateSampler(dev, &sci, nullptr, &g.nearest), "warp sampler")) return false;
    const VkDescriptorSetLayoutBinding bind[5] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                                                  {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                                                  {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                                                  {3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                                                  {4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 5;
    dli.pBindings = bind;
    if (!vk_ok(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &g.warp_dsl), "warp set layout")) return false;
    const VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(WarpPc)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g.warp_dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (!vk_ok(vkCreatePipelineLayout(dev, &pli, nullptr, &g.warp_layout), "warp pipeline layout")) return false;
    const VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;  // (reset per warp)
    dpi.poolSizeCount = 3;
    dpi.pPoolSizes = sizes;
    if (!vk_ok(vkCreateDescriptorPool(dev, &dpi, nullptr, &g.warp_dpool), "warp descriptor pool")) return false;
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 2;
    if (!vk_ok(vkCreateQueryPool(dev, &qi, nullptr, &g.warp_q), "warp query pool")) return false;

    VkShaderModule vs = module_of(make_warp_vs()), fs[2] = {module_of(make_warp_fs(false)), module_of(make_warp_fs(true))};
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs[0], "main", nullptr}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbs.attachmentCount = 1;
    cbs.pAttachments = &cba;
    const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dsi.dynamicStateCount = 2;
    dsi.pDynamicStates = dyn;
    const VkFormat cf = VK_FORMAT_B8G8R8A8_UNORM;
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &cf;
    rci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.pNext = &rci;
    gp.stageCount = 2;
    gp.pStages = st;
    gp.pVertexInputState = &vi; gp.pInputAssemblyState = &ia; gp.pViewportState = &vp; gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms; gp.pDepthStencilState = &ds; gp.pColorBlendState = &cbs; gp.pDynamicState = &dsi;
    gp.layout = g.warp_layout;
    bool ok = vk_ok(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &g.warp_pipeline), "warp pipeline");
    st[1].module = fs[1];
    ok = ok && vk_ok(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &g.warp_pipeline_torn), "warp pipeline (torn cells)");
    vkDestroyShaderModule(dev, vs, nullptr);
    for (VkShaderModule m : fs) vkDestroyShaderModule(dev, m, nullptr);
    return ok;
}

// ---- snapshots -----------------------------------------------------------------------------------------------------------
void destroy_snaps() {
    VkDevice dev = vk().device;
    for (Snap& s : g.snaps) {
        if (s.view) vkDestroyImageView(dev, s.view, nullptr);
        if (s.image) vmaDestroyImage(vk().vma, s.image, s.alloc);
        if (s.scene_view) vkDestroyImageView(dev, s.scene_view, nullptr);
        if (s.scene) vmaDestroyImage(vk().vma, s.scene, s.scene_alloc);
        if (s.pre_view) vkDestroyImageView(dev, s.pre_view, nullptr);
        if (s.pre) vmaDestroyImage(vk().vma, s.pre, s.pre_alloc);
        if (s.depth) vmaDestroyBuffer(vk().vma, s.depth, s.depth_alloc);
        s = Snap{};
    }
    if (g.warp_depth_view) vkDestroyImageView(dev, g.warp_depth_view, nullptr);
    if (g.warp_depth) vmaDestroyImage(vk().vma, g.warp_depth, g.warp_depth_alloc);
    g.warp_depth = VK_NULL_HANDLE, g.warp_depth_view = VK_NULL_HANDLE;
    g.cur = -1;
}

bool create_image(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, VkImage& image, VmaAllocation& alloc, VkImageView& view) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = {w, h, 1};
    ici.mipLevels = ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (!vk_ok(vmaCreateImage(vk().vma, &ici, &aci, &image, &alloc, nullptr), "snapshot image")) return false;
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = {VkImageAspectFlags(fmt == VK_FORMAT_D32_SFLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
    return vk_ok(vkCreateImageView(vk().device, &vci, nullptr, &view), "snapshot view");
}

bool create_snaps(uint32_t w, uint32_t h) {
    destroy_snaps();
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;  // SRC: BB_SHOT_AT readback
    for (uint32_t i = 0; i < g.nsnaps + (g.warp ? 1 : 0); ++i) {
        Snap& s = g.snaps[i];
        if (!create_image(w, h, VK_FORMAT_B8G8R8A8_UNORM, usage | (i == g.nsnaps ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0), s.image, s.alloc, s.view)) return false;
        if (!g.warp || i == g.nsnaps) continue;
        if (!create_image(w, h, VK_FORMAT_B8G8R8A8_UNORM, usage, s.scene, s.scene_alloc, s.scene_view)) return false;
        if (!create_image(w, h, VK_FORMAT_B8G8R8A8_UNORM, usage, s.pre, s.pre_alloc, s.pre_view)) return false;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = VkDeviceSize(w) * h * 4;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (!vk_ok(vmaCreateBuffer(vk().vma, &bci, &aci, &s.depth, &s.depth_alloc, nullptr), "snapshot depth")) return false;
    }
    if (g.warp && !create_image(w, h, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, g.warp_depth, g.warp_depth_alloc, g.warp_depth_view)) return false;
    g.snap_w = w;
    g.snap_h = h;
    return true;
}

void layout_barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                    VkAccessFlags src_access, VkAccessFlags dst_access) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// ---- swapchain -----------------------------------------------------------------------------------------------------------
void destroy_swapchain() {
    if (!vk().device) return;
    vkDeviceWaitIdle(vk().device);
    if (!g.cmds.empty()) vkFreeCommandBuffers(vk().device, g.pool, uint32_t(g.cmds.size()), g.cmds.data());
    g.cmds.clear();
    for (VkImageView v : g.views) vkDestroyImageView(vk().device, v, nullptr);
    g.views.clear();
    if (g.swapchain) vkDestroySwapchainKHR(vk().device, g.swapchain, nullptr);
    g.swapchain = VK_NULL_HANDLE;
    g.images.clear();
}

bool create_swapchain() {
    VkSurfaceCapabilitiesKHR caps;
    if (!vk_ok(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk().phys, g.surface, &caps), "surface caps")) return false;
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(g.window, &w, &h);
    g.extent = caps.currentExtent.width != 0xFFFFFFFFu ? caps.currentExtent
                                                        : VkExtent2D{std::clamp<uint32_t>(uint32_t(w), caps.minImageExtent.width, caps.maxImageExtent.width),
                                                                     std::clamp<uint32_t>(uint32_t(h), caps.minImageExtent.height, caps.maxImageExtent.height)};
    if (g.extent.width == 0 || g.extent.height == 0) return false;  // minimised
    if (std::getenv("BB_PRESENT_LOG")) std::fprintf(stderr, "present: swapchain %ux%u\n", g.extent.width, g.extent.height);

    // FIFO (vsync) in native mode; otherwise MAILBOX (no tearing, not tied to the display refresh) when the driver has it.
    // (MAILBOX in native mode and acquiring with a fence instead of a semaphore were measured: no change of the render-thread waits.)
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    if (g.fps >= 0) {
        uint32_t n = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(vk().phys, g.surface, &n, nullptr);
        std::vector<VkPresentModeKHR> modes(n);
        vkGetPhysicalDeviceSurfacePresentModesKHR(vk().phys, g.surface, &n, modes.data());
        for (VkPresentModeKHR m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) mode = m;
        if (mode == VK_PRESENT_MODE_FIFO_KHR) for (VkPresentModeKHR m : modes) if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) mode = m;
    }
    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = g.surface;
    sci.minImageCount = std::min(caps.minImageCount + 1, caps.maxImageCount ? caps.maxImageCount : ~0u);
    sci.imageFormat = g.format;
    sci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    sci.imageExtent = g.extent;
    sci.imageArrayLayers = 1;
    g.out_ok = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;  // (BB_SHOT_OUT)
    sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | (g.out_ok ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = mode;
    sci.clipped = VK_TRUE;
    if (!vk_ok(vkCreateSwapchainKHR(vk().device, &sci, nullptr, &g.swapchain), "swapchain")) return false;
    uint32_t n = 0;
    vkGetSwapchainImagesKHR(vk().device, g.swapchain, &n, nullptr);
    g.images.resize(n);
    vkGetSwapchainImagesKHR(vk().device, g.swapchain, &n, g.images.data());
    g.views.resize(n);
    for (VkSemaphoreCreateInfo sem{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; g.render_done.size() < n;) {
        VkSemaphore s = VK_NULL_HANDLE;
        vkCreateSemaphore(vk().device, &sem, nullptr, &s);
        g.render_done.push_back(s);
    }
    for (uint32_t i = 0; i < n; ++i) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = g.images[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = g.format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (!vk_ok(vkCreateImageView(vk().device, &vci, nullptr, &g.views[i]), "swapchain view")) return false;
    }
    g.cmds.resize(n);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = n;
    return vk_ok(vkAllocateCommandBuffers(vk().device, &ai, g.cmds.data()), "command buffers");
}

// ---- reprojection: matrices and the pass -----------------------------------------------------------------------------------
struct WarpReq { int src = -1; float a_src = 0, a_t = 0; Packet p; };  // warp image `src` (rendered at alpha a_src) to alpha a_t of packet p
using M4 = std::array<double, 16>;  // row-major
M4 mul(const M4& a, const M4& b) {
    M4 r{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k) r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
    return r;
}
bool invert(M4 a, M4& r) {  // Gauss-Jordan with partial pivoting
    r = M4{};
    for (int i = 0; i < 4; ++i) r[i * 5] = 1;
    for (int c = 0; c < 4; ++c) {
        int p = c;
        for (int i = c + 1; i < 4; ++i) if (std::fabs(a[i * 4 + c]) > std::fabs(a[p * 4 + c])) p = i;
        if (std::fabs(a[p * 4 + c]) < 1e-30) return false;
        for (int j = 0; j < 4; ++j) std::swap(a[c * 4 + j], a[p * 4 + j]), std::swap(r[c * 4 + j], r[p * 4 + j]);
        const double d = 1 / a[c * 4 + c];
        for (int j = 0; j < 4; ++j) a[c * 4 + j] *= d, r[c * 4 + j] *= d;
        for (int i = 0; i < 4; ++i) {
            if (i == c) continue;
            const double f = a[i * 4 + c];
            for (int j = 0; j < 4; ++j) a[i * 4 + j] -= f * a[c * 4 + j], r[i * 4 + j] -= f * r[c * 4 + j];
        }
    }
    return true;
}
// The camera at alpha: the block's dwords interpolated as the backend does for the replays (Backend::replay_tick), so a rendered image's
// camera is exact. View-projection = P * V (row-major, column vectors): V = the view 3x4 (dwords 8-19), P = dwords 52-67 (D3D-style:
// depth = f/(f-n) - fn/((f-n) z), w = view z, forward +z). Dwords 20-35 hold (P * V)^-1 and 36-51 P^-1 (checked offline).
M4 camera_at(const Packet& p, float alpha) {
    auto dw = [&](int i) {  // block dword i
        const float x = p.cam0[i - 8], y = p.cam1[i - 8];
        uint32_t a, b;
        std::memcpy(&a, &x, 4), std::memcpy(&b, &y, 4);
        auto ok = [](uint32_t u) { const uint32_t e = u >> 23 & 0xFF; return (e && e != 0xFF) || !(u & 0x7FFFFFFF); };
        return double(alpha == 0 ? x : (a == b || !ok(a) || !ok(b) || std::fabs(y - x) > 0.5f * std::max({1.0f, std::fabs(x), std::fabs(y)})) ? y : x + (y - x) * alpha);
    };
    M4 V{}, P{};
    for (int j = 0; j < 12; ++j) V[j] = dw(8 + j);
    V[15] = 1;
    for (int j = 0; j < 16; ++j) P[j] = dw(52 + j);
    return mul(P, V);
}
// M = T * VP(a_t) * VP(a_src)^-1 * S: S maps a source pixel (x + 0.5, y + 0.5, depth) to the guest's NDC through its viewport, T the
// target NDC to Vulkan clip space of the snapshot; r3 = row 3 of VP(a_src)^-1 * S (1 / source view depth).
bool warp_constants(const WarpReq& wr, WarpPc& pc) {
    const float* v = wr.p.vport;
    if (v[0] == 0 || v[2] == 0 || v[4] == 0) return false;
    const double W = g.snap_w, H = g.snap_h;
    const M4 S{1 / v[0], 0, 0, -v[1] / v[0], 0, 1 / v[2], 0, -v[3] / v[2], 0, 0, 1 / v[4], -v[5] / v[4], 0, 0, 0, 1};
    const M4 T{2 * v[0] / W, 0, 0, 2 * v[1] / W - 1, 0, 2 * v[2] / H, 0, 2 * v[3] / H - 1, 0, 0, 1, 0, 0, 0, 0, 1};
    M4 inv;
    if (!invert(camera_at(wr.p, wr.a_src), inv)) return false;
    const M4 A = mul(inv, S), M = mul(T, mul(camera_at(wr.p, wr.a_t), A));
    for (int j = 0; j < 4; ++j) pc.m0[j] = float(M[j]), pc.m1[j] = float(M[4 + j]), pc.m3[j] = float(M[12 + j]), pc.r3[j] = float(A[12 + j]);
    pc.cell = g.warp_cell;
    pc.w = g.snap_w, pc.h = g.snap_h;
    pc.gw = (pc.w + pc.cell - 1) / pc.cell;
    pc.disc = 1.15f;  // a cell whose corners' view depths differ by more than 15 % is torn (a disocclusion edge)
    return true;
}

// Records the warp of wr.src into snaps[nsnaps] (then SHADER_READ_ONLY). false: nothing recorded.
bool warp_one(VkCommandBuffer cb, const WarpReq& wr) {
    if (wr.src < 0 || !g.warp_pipeline) return false;
    const Snap& s = g.snaps[wr.src];
    Snap& t = g.snaps[g.nsnaps];
    WarpPc pc;
    if (!s.scene || !s.pre || !s.depth || !t.image || !warp_constants(wr, pc)) return false;
    VkDevice dev = vk().device;
    uint64_t ts[2];  // the previous warp's GPU time (its command buffer has completed: present_one waited for it)
    if (g.warps && vkGetQueryPoolResults(dev, g.warp_q, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && ts[1] >= ts[0])
        g.warp_ms.push_back(double(ts[1] - ts[0]) * vk().props.limits.timestampPeriod * 1e-6);
    vkResetDescriptorPool(dev, g.warp_dpool, 0);
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = g.warp_dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &g.warp_dsl;
    if (vkAllocateDescriptorSets(dev, &dai, &set) != VK_SUCCESS) return false;
    const VkDescriptorBufferInfo bi{s.depth, 0, VK_WHOLE_SIZE};
    const VkDescriptorImageInfo i1{VK_NULL_HANDLE, s.scene_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorImageInfo i2{VK_NULL_HANDLE, s.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorImageInfo i4{VK_NULL_HANDLE, s.pre_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorImageInfo is{g.nearest, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    const VkWriteDescriptorSet wr5[5] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi, nullptr},
                                         {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &i1, nullptr, nullptr},
                                         {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 2, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &i2, nullptr, nullptr},
                                         {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 3, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &is, nullptr, nullptr},
                                         {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 4, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &i4, nullptr, nullptr}};
    vkUpdateDescriptorSets(dev, 5, wr5, 0, nullptr);

    vkCmdResetQueryPool(cb, g.warp_q, 0, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g.warp_q, 0);
    // background where the mesh leaves the screen uncovered (edges the camera turns towards): the source's final picture
    layout_barrier(cb, s.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    layout_barrier(cb, t.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   0, VK_ACCESS_TRANSFER_WRITE_BIT);
    const VkImageCopy cp{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {g.snap_w, g.snap_h, 1}};
    vkCmdCopyImage(cb, s.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    layout_barrier(cb, s.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT);
    layout_barrier(cb, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    VkImageMemoryBarrier db{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    db.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    db.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    db.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    db.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    db.srcQueueFamilyIndex = db.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    db.image = g.warp_depth;
    db.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    const VkPipelineStageFlags dstages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    vkCmdPipelineBarrier(cb, dstages, dstages, 0, 0, nullptr, 0, nullptr, 1, &db);

    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = t.view;
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    da.imageView = g.warp_depth_view;
    da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    da.clearValue.depthStencil = {1.0f, 0};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {g.snap_w, g.snap_h}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    ri.pDepthAttachment = &da;
    vkCmdBeginRendering(cb, &ri);
    const VkViewport vp{0, 0, float(g.snap_w), float(g.snap_h), 0, 1};
    const VkRect2D sc{{0, 0}, {g.snap_w, g.snap_h}};
    const uint32_t verts = pc.gw * ((pc.h + pc.cell - 1) / pc.cell) * 6;
    for (uint32_t pass = 0; pass < 2; ++pass) {  // the cells that do not tear (early depth test), then the torn ones (search, depth per fragment)
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pass ? g.warp_pipeline_torn : g.warp_pipeline);
        if (!pass) {
            vkCmdSetViewport(cb, 0, 1, &vp);
            vkCmdSetScissor(cb, 0, 1, &sc);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g.warp_layout, 0, 1, &set, 0, nullptr);
        }
        pc.pass = pass;
        vkCmdPushConstants(cb, g.warp_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, &pc);
        vkCmdDraw(cb, verts, 1, 0, 0);
    }
    vkCmdEndRendering(cb);
    layout_barrier(cb, t.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g.warp_q, 1);
    ++g.warps;
    return true;
}

// A binary PPM of a B8G8R8A8 image (tightly packed rows).
void write_ppm(const std::string& path, const uint8_t* px, uint32_t w, uint32_t h) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%u %u\n255\n", w, h);
    std::vector<uint8_t> row(size_t(w) * 3);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) { const uint8_t* q = px + (size_t(y) * w + x) * 4; row[3 * x] = q[2]; row[3 * x + 1] = q[1]; row[3 * x + 2] = q[0]; }  // BGRA -> RGB
        std::fwrite(row.data(), 1, row.size(), f);
    }
    std::fclose(f);
    std::fprintf(stderr, "present: wrote %s (%ux%u)\n", path.c_str(), w, h);
}

// The EASU target in swapchain size, (re)created when the window size changed (the previous present, the only user, has completed).
bool up_image() {
    if (g.up && g.up_ext.width == g.extent.width && g.up_ext.height == g.extent.height) return true;
    if (g.up_view) vkDestroyImageView(vk().device, g.up_view, nullptr);
    if (g.up) vmaDestroyImage(vk().vma, g.up, g.up_alloc);
    g.up = VK_NULL_HANDLE, g.up_view = VK_NULL_HANDLE;
    if (!create_image(g.extent.width, g.extent.height, g.format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, g.up, g.up_alloc, g.up_view)) return false;
    g.up_ext = g.extent;
    return true;
}

// BB_SHOT_OUT readback buffer in swapchain size (host visible).
bool out_buffer() {
    if (g.out_buf && g.out_ext.width == g.extent.width && g.out_ext.height == g.extent.height) return true;
    if (g.out_buf) vmaDestroyBuffer(vk().vma, g.out_buf, g.out_alloc);
    g.out_buf = VK_NULL_HANDLE;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = VkDeviceSize(g.extent.width) * g.extent.height * 4;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};
    aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    if (!vk_ok(vmaCreateBuffer(vk().vma, &bci, &aci, &g.out_buf, &g.out_alloc, &g.out_info), "output shot buffer")) return false;
    g.out_ext = g.extent;
    return true;
}

// Captures the flipped display buffer (if a new game frame arrived) and shows the newest snapshot on the swapchain. `show` >= 0 (replay
// mode): show that snapshot as is, no capture; with `warp` (warp->src >= 0) instead image warp->src reprojected (warp_one). `wait_v`: with
// the presenter queue (g.pq) the backend's replay semaphore value the shown / warped images need. Returns whether it warped.
bool present_one(bool new_frame, int show = -1, const WarpReq* warp = nullptr, uint64_t wait_v = 0) {
    if (g.swapchain == VK_NULL_HANDLE && !create_swapchain()) return false;
    if (g.snap_w != g.last_w * res_scale() || g.snap_h != g.last_h * res_scale()) {  // (snapshots in target pixels: BB_RES_SCALE)
        std::lock_guard<std::mutex> lk(g.pk_snaps);  // (replay mode: the render thread writes them)
        { std::lock_guard<std::mutex> q(vk().queue_mutex); vkDeviceWaitIdle(vk().device); }
        { std::lock_guard<std::mutex> p(g.pk_pub); g.pk_seq = 0; }
        if (!create_snaps(g.last_w * res_scale(), g.last_h * res_scale())) return false;
        new_frame = true;
        show = -1, warp = nullptr;
    }
    if (show >= 0) g.cur = show;
    vkWaitForFences(vk().device, 1, &g.in_flight, VK_TRUE, UINT64_MAX);
    // the previous present has completed: its final pass's GPU time, its BB_SHOT_OUT copy
    if (g.out_q_used) {
        uint64_t ts[2];
        if (vkGetQueryPoolResults(vk().device, g.out_q, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && ts[1] >= ts[0])
            g.out_ms.push_back(double(ts[1] - ts[0]) * vk().props.limits.timestampPeriod * 1e-6);
        g.out_q_used = false;
    }
    if (!g.out_pending.empty()) {
        write_ppm(g.out_pending, static_cast<const uint8_t*>(g.out_info.pMappedData), g.out_ext.width, g.out_ext.height);
        g.out_pending.clear();
    }
    uint32_t idx = 0;
    VkResult r = vkAcquireNextImageKHR(vk().device, g.swapchain, UINT64_MAX, g.image_ready, VK_NULL_HANDLE, &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) { destroy_swapchain(); return false; }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) return false;
    vkResetFences(vk().device, 1, &g.in_flight);

    VkCommandBuffer cb = g.cmds[idx];
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &bi);
    const bool warped = warp && warp->src >= 0 && warp_one(cb, *warp);
    if (warped) g.cur = int(g.nsnaps);

    // 1. capture: backend render target -> next snapshot (the backend's rendering must be visible to the blit)
    bool have_frame = g.cur >= 0;
    if (g.rep_max) new_frame = false;
    if (new_frame) {
        const uint32_t w = g.write_idx % 3;
        Snap& s = g.snaps[w];
        VkMemoryBarrier render_to_blit{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        render_to_blit.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        render_to_blit.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &render_to_blit, 0, nullptr, 0, nullptr);
        layout_barrier(cb, s.image, s.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        if (backend_blit_frame(g.last_addr, g.last_w, g.last_h, cb, s.image, {g.snap_w, g.snap_h})) {
            layout_barrier(cb, s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
            s.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            g.cur = int(w);
            ++g.write_idx;
            have_frame = true;
        } else {
            layout_barrier(cb, s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
            s.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    }

    // 2. the picture into the swapchain image: the present pass, or (upscaling with FSR) EASU into `up`, then RCAS into the swapchain image
    const bool fsr = have_frame && g.easu && (g.extent.width > g.snap_w || g.extent.height > g.snap_h) && up_image();
    VkDescriptorSet sets[2] = {};
    if (have_frame) {
        vkResetDescriptorPool(vk().device, g.dpool, 0);
        const VkDescriptorSetLayout dsls[2] = {g.dsl, g.dsl};
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = g.dpool;
        dai.descriptorSetCount = fsr ? 2 : 1;
        dai.pSetLayouts = dsls;
        have_frame = vkAllocateDescriptorSets(vk().device, &dai, sets) == VK_SUCCESS;
        auto write_set = [](VkDescriptorSet set, VkImageView v) {
            const VkDescriptorImageInfo i0{VK_NULL_HANDLE, v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            const VkDescriptorImageInfo is{g.sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
            VkWriteDescriptorSet wr[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &i0, nullptr, nullptr},
                                          {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &is, nullptr, nullptr}};
            vkUpdateDescriptorSets(vk().device, 2, wr, 0, nullptr);
        };
        if (have_frame) write_set(sets[0], g.snaps[g.cur].view);
        if (have_frame && fsr) write_set(sets[1], g.up_view);
    }
    const VkViewport vp{0, 0, float(g.extent.width), float(g.extent.height), 0, 1};
    const VkRect2D sc{{0, 0}, g.extent};
    vkCmdResetQueryPool(cb, g.out_q, 0, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g.out_q, 0);
    FsrPc pc;
    if (have_frame && fsr) {
        layout_barrier(cb, g.up, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo ua{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        ua.imageView = g.up_view;
        ua.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ua.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ua.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ui{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ui.renderArea = sc;
        ui.layerCount = 1;
        ui.colorAttachmentCount = 1;
        ui.pColorAttachments = &ua;
        vkCmdBeginRendering(cb, &ui);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g.easu);
        vkCmdSetViewport(cb, 0, 1, &vp);
        vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g.fsr_layout, 0, 1, &sets[0], 0, nullptr);
        easu_con(pc, float(g.snap_w), float(g.snap_h), float(g.extent.width), float(g.extent.height));
        vkCmdPushConstants(cb, g.fsr_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, &pc);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRendering(cb);
        layout_barrier(cb, g.up, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    // source stage = the semaphore's wait stage: the layout transition waits for the acquire (sync validation: WAR after vkAcquireNextImageKHR)
    layout_barrier(cb, g.images[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    att.imageView = g.views[idx];
    att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    const float t = float(g.presents % 120) / 120.0f;  // until a frame exists: a dim heartbeat colour shows the guest is alive
    att.clearValue.color = {{0.04f + 0.10f * t, 0.02f, 0.05f + 0.05f * (1.0f - t), 1.0f}};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = sc;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;
    vkCmdBeginRendering(cb, &ri);
    if (have_frame) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, fsr ? g.rcas : g.pipeline);
        vkCmdSetViewport(cb, 0, 1, &vp);
        vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, fsr ? g.fsr_layout : g.layout, 0, 1, &sets[fsr ? 1 : 0], 0, nullptr);
        if (fsr) {
            rcas_con(pc, g.fsr_sharp);
            vkCmdPushConstants(cb, g.fsr_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, &pc);
        }
        vkCmdDraw(cb, 3, 1, 0, 0);
    }
    vkCmdEndRendering(cb);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g.out_q, 1);
    g.out_q_used = have_frame, g.out_fsr = fsr;
    if (have_frame && !g.out_path.empty() && g.out_ok && out_buffer()) {  // BB_SHOT_OUT: the presented image, read back with this submit
        layout_barrier(cb, g.images[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        const VkBufferImageCopy cp{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {g.extent.width, g.extent.height, 1}};
        vkCmdCopyImageToBuffer(cb, g.images[idx], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g.out_buf, 1, &cp);
        layout_barrier(cb, g.images[idx], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_ACCESS_TRANSFER_READ_BIT, 0);
        g.out_pending = std::move(g.out_path);
        g.out_path.clear();
    } else
        layout_barrier(cb, g.images[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0);
    vkEndCommandBuffer(cb);

    // replay mode's snapshots go through the presenter queue (if any), which waits for the images' replay semaphore value; a capture of the
    // backend's render target needs the backend's queue order
    const bool q2 = g.pq && show >= 0;
    const VkPipelineStageFlags wait_stage[2] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    const VkSemaphore waits[2] = {g.image_ready, backend_replay_sem()};
    const uint64_t wait_vals[2] = {0, wait_v};
    VkTimelineSemaphoreSubmitInfo tsi{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, nullptr, 2, wait_vals, 0, nullptr};
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = q2 && waits[1] && wait_v ? 2 : 1;
    if (si.waitSemaphoreCount == 2) si.pNext = &tsi;
    si.pWaitSemaphores = waits;
    si.pWaitDstStageMask = wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &g.render_done[idx];
    VkQueue q = q2 ? vk().queue2 : vk().queue;
    std::mutex& qm = q2 ? vk().queue2_mutex : vk().queue_mutex;
    { std::lock_guard<std::mutex> lk(qm); vkQueueSubmit(q, 1, &si, g.in_flight); }
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &g.render_done[idx];
    pi.swapchainCount = 1;
    pi.pSwapchains = &g.swapchain;
    pi.pImageIndices = &idx;
    { std::lock_guard<std::mutex> lk(qm); r = vkQueuePresentKHR(q, &pi); }
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) destroy_swapchain();
    ++g.presents;
    return warped;
}

} // namespace

bool init(int width, int height, const char* title) {
    if (const char* f = std::getenv("BB_FPS")) {  // 30 / 60 / 120 / 144 / unlimited / native / refresh (every display refresh, FIFO; the hybrid)
        const std::string s = f;
        g.fps = s == "native" ? -1 : s == "refresh" ? -2 : s == "unlimited" ? 0 : std::max(1, std::atoi(f));
    }
    // Above 30 (a rate, refresh or unlimited) the hybrid unless BB_INTERP=0: up to ceil(rate / 30) images per game frame (BB_INTERP_N caps
    // it; unlimited: kMaxImages), all but the last replays (backend_replay_tick), the other presents reprojections. Native and 30: none.
    const bool replay = !(std::getenv("BB_INTERP") && std::getenv("BB_INTERP")[0] == '0') && (g.fps > 30 || g.fps == 0 || g.fps == -2) && backend_replay_supported();
    if (const char* d = std::getenv("BB_INTERP_DELAY_MS")) g.delay_ms = std::atof(d);
    // BB_BACKGROUND=1 (test runs while the user works/plays): a small window that never takes focus and no gamepad subsystem, so the run steals
    // neither keyboard nor controller (scripted input via BB_PAD_SCRIPT still works)
    const bool background = std::getenv("BB_BACKGROUND") != nullptr;
    if (!SDL_Init(SDL_INIT_VIDEO | (background ? 0u : SDL_INIT_GAMEPAD))) {
        std::fprintf(stderr, "gpu: SDL_Init failed: %s\n", SDL_GetError());
        return false;
    }
    const bool sized = std::getenv("BB_WINDOW") != nullptr;  // a background run may still ask for a bigger window (someone is watching)
    g.window = background ? SDL_CreateWindow(title, sized ? width : 480, sized ? height : 270, SDL_WINDOW_VULKAN | SDL_WINDOW_NOT_FOCUSABLE | SDL_WINDOW_RESIZABLE)
                          : SDL_CreateWindow(title, width, height, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!g.window) {
        std::fprintf(stderr, "gpu: SDL_CreateWindow failed: %s\n", SDL_GetError());
        return false;
    }
    if (replay) {
        double rate = g.fps > 0 ? g.fps : g.fps == 0 ? 30.0 * kMaxImages : 60;
        if (g.fps == -2)
            if (const SDL_DisplayMode* m = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(g.window)); m && m->refresh_rate > 0) rate = m->refresh_rate;
        uint32_t n = std::min<uint32_t>(kMaxImages, uint32_t(std::ceil(rate / 30 - 0.01)));
        if (const char* c = std::getenv("BB_INTERP_N")) n = std::min<uint32_t>(n, uint32_t(std::max(1, std::atoi(c)))), g.fixed_n = true;  // (no GPU budget)
        if (n >= 2) {
            g.rep_max = n, g.rep_budget = n;
            g.nsnaps = kPackets * n;
            g.clk_period = int64_t(1e9 / rate);
            backend_request_replay();
            g.warp = !(std::getenv("BB_INTERP_WARP") && std::getenv("BB_INTERP_WARP")[0] == '0');
            if (const char* c = std::getenv("BB_INTERP_WARP_CELL")) g.warp_cell = uint32_t(std::clamp(std::atoi(c), 1, 16));
        }
    }
    uint32_t ext_count = 0;
    const char* const* exts = SDL_Vulkan_GetInstanceExtensions(&ext_count);
    if (!vk_create_instance(exts, ext_count)) return false;
    if (!SDL_Vulkan_CreateSurface(g.window, vk().instance, nullptr, &g.surface)) {
        std::fprintf(stderr, "gpu: SDL_Vulkan_CreateSurface failed: %s\n", SDL_GetError());
        return false;
    }
    vk().want_queue2 = g.rep_max > 0;
    if (!vk_create_device(g.surface)) return false;
    g.pq = g.rep_max && vk().queue2;  // (a 2-queue family and not BB_PRESENT_QUEUE=0)

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = vk().family;
    if (!vk_ok(vkCreateCommandPool(vk().device, &pci, nullptr, &g.pool), "command pool")) return false;
    VkSemaphoreCreateInfo sem{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(vk().device, &sem, nullptr, &g.image_ready);
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(vk().device, &fci, nullptr, &g.in_flight);
    if (const char* u = std::getenv("BB_UPSCALE")) g.fsr = std::string(u) != "linear";  // (see State: fsr, fsr_sharp, shot_out)
    if (const char* s = std::getenv("BB_FSR_SHARP")) g.fsr_sharp = std::clamp(float(std::atof(s)), 0.0f, 4.0f);
    g.shot_out = std::getenv("BB_SHOT_OUT") && std::getenv("BB_SHOT_OUT")[0] == '1';
    if (!create_blend_pipeline()) return false;
    if (g.warp && !create_warp_pipeline()) g.warp = false;
    return create_swapchain();
}

// Seconds from a flip to the start of its display window: one present period, at least a replay's GPU time + 2 ms (BB_INTERP_DELAY_MS).
static double replay_delay(double period) {
    return g.delay_ms >= 0 ? g.delay_ms / 1000 : std::max(period, backend_replay_gpu_ms() / 1000 + 0.002);
}

// The hybrid: the guest flip ends the tick; the backend renders the tick's packet: one replay per alpha the present schedule asks for
// (within the GPU budget), then the tick's own picture.
static void replay_frame(uint64_t addr, uint32_t w, uint32_t h) {
    const auto now = Clock::now();
    if (g.last_flip != Clock::time_point{}) {
        const double dt = std::chrono::duration<double>(now - g.last_flip).count();
        if (dt < 0.1) g.tick_s += (dt - g.tick_s) * 0.1;  // (loading stalls are not the cadence)
    }
    g.last_flip = now;
    // GPU budget (images per tick, the tick's own included): the original costs about one replay, all must fit in ~85 % of a tick; one
    // image less whenever the render thread waited > 2 ms for a replay, one more after 30 ticks without.
    const double gr = backend_replay_gpu_ms();
    uint32_t cap = gr > 0.5 ? std::clamp<uint32_t>(uint32_t(0.85 * g.tick_s * 1000 / gr), 1, g.rep_max) : g.rep_max;
    static uint32_t probe = 0;  // (render thread)
    if (cap == 1 && ++probe % 30 == 0) cap = 2;  // no replay, no new measurement: one now and then (the load may have gone)
    if (backend_replay_hold_ms() > 2.0) g.rep_budget = std::max(1u, g.rep_budget.load() - 1), g.budget_ok_ticks = 0;
    else if (++g.budget_ok_ticks >= 30 && g.rep_budget < g.rep_max) ++g.rep_budget, g.budget_ok_ticks = 0;
    if (g.fixed_n) cap = g.rep_max, g.rep_budget = g.rep_max;
    const uint32_t reps = std::min(cap, g.rep_budget.load()) - 1;
    // alphas: the present times inside the window [now + d, now + d + tick) (0 = the previous picture, already there; ~1 = this tick's)
    const double period = double(g.clk_period.load()) * 1e-9, d = replay_delay(period);
    float cand[64];
    uint32_t nc = 0;
    if (g.fps == 0) {
        for (uint32_t j = 1; j <= reps; ++j) cand[nc++] = float(j) / float(reps + 1);
    } else {
        const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        const int64_t ws = now_ns + int64_t(d * 1e9), tick = int64_t(g.tick_s * 1e9), per = std::max<int64_t>(g.clk_period, 1000000);
        int64_t ref = g.clk_ref;
        if (!ref) ref = ws;
        for (int64_t t = ref + (ws > ref ? (ws - ref + per - 1) / per * per : 0); t < ws + tick && nc < 64; t += per)
            if (const float a = float(double(t - ws) / double(tick)); a > 0.02f && a < 0.98f) cand[nc++] = a;
    }
    std::lock_guard<std::mutex> lk(g.pk_snaps);
    if (g.snap_w != w * res_scale() || g.snap_h != h * res_scale()) return;  // the presenter (re)creates the snapshots first
    const uint64_t k = g.pk_seq;  // (only this thread advances it)
    Packet p;
    p.n = std::min(nc, reps) + 1;
    for (uint32_t j = 0; j + 1 < p.n; ++j) p.a[j] = nc <= reps ? cand[j] : cand[uint32_t((j + 0.5) * nc / reps)];  // too many: evenly picked
    p.a[p.n - 1] = 1.0f;
    VkImage img[kMaxImages], scene[kMaxImages], pre[kMaxImages];
    VkBuffer depth[kMaxImages];
    for (uint32_t i = 0; i < p.n; ++i) {
        const Snap& s = g.snaps[p.idx[i] = int((k % kPackets) * g.rep_max + i)];
        img[i] = s.image, scene[i] = s.scene, pre[i] = s.pre, depth[i] = s.depth;
    }
    ReplayWarp rw;
    rw.scene = scene, rw.pre = pre, rw.depth = depth;
    ReplayResult rr;
    if (!backend_replay_tick(addr, w, h, img, {g.snap_w, g.snap_h}, p.n, p.a, g.warp ? &rw : nullptr, g.pq ? p.ready : nullptr, &rr)) return;
    if ((p.warp = rw.ok)) {
        std::memcpy(p.cam0, rw.cam0, sizeof p.cam0), std::memcpy(p.cam1, rw.cam1, sizeof p.cam1), std::memcpy(p.vport, rw.vport, sizeof p.vport);
    }
    p.cut = rr.cut;
    if (!rr.replayed && !rr.cut && p.n > 1) {  // no in-between images (every one is the own picture): only the own one, at alpha 1
        p.idx[0] = p.idx[p.n - 1], p.ready[0] = p.ready[p.n - 1], p.a[0] = 1.0f;
        p.n = 1;
    }
    p.t = now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(d));  // (window start)
    p.frame = g.requested;
    p.tick = g.tick_s;
    std::lock_guard<std::mutex> pl(g.pk_pub);
    g.packets[k % kPackets] = p;
    g.pk_seq = k + 1;
}

void request_present(uint32_t buffer_index, uint64_t guest_addr, uint32_t width, uint32_t height) {
    pad_note_flip();
    backend_new_frame();
    g.last_w = width;
    g.last_h = height;
    g.last_buffer = buffer_index;
    g.last_addr = guest_addr;
    if (g.rep_max) replay_frame(guest_addr, width, height);
    ++g.requested;
}

// Debug: BB_SHOT_AT=<seconds>:<file.ppm> writes the newest captured game frame once (screen grabs fail as soon as another window covers ours).
// save_snaps: (snapshot, file) pairs read back in one submit.
static void save_snaps(const std::vector<std::pair<int, std::string>>& shots) {
    vkWaitForFences(vk().device, 1, &g.in_flight, VK_TRUE, UINT64_MAX);  // the last present may still be sampling the snapshot
    const uint32_t w = g.snap_w, h = g.snap_h;
    const VkDeviceSize bytes = VkDeviceSize(w) * h * 4;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes * shots.size();
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};
    aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VmaAllocationInfo info{};
    if (!vk_ok(vmaCreateBuffer(vk().vma, &bci, &aci, &buf, &alloc, &info), "shot buffer")) return;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(vk().device, &ai, &cb);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &bi);
    for (size_t k = 0; k < shots.size(); ++k) {
        const VkImage im = g.snaps[shots[k].first].image;
        layout_barrier(cb, im, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        const VkBufferImageCopy cp{bytes * k, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
        vkCmdCopyImageToBuffer(cb, im, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &cp);
        layout_barrier(cb, im, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    vkEndCommandBuffer(cb);
    VkFence fence = VK_NULL_HANDLE;
    const VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(vk().device, &fci, nullptr, &fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    { std::lock_guard<std::mutex> lk(vk().queue_mutex); vkQueueSubmit(vk().queue, 1, &si, fence); }
    vkWaitForFences(vk().device, 1, &fence, VK_TRUE, UINT64_MAX);
    for (size_t k = 0; k < shots.size(); ++k) write_ppm(shots[k].second, static_cast<const uint8_t*>(info.pMappedData) + bytes * k, w, h);
    vkDestroyFence(vk().device, fence, nullptr);
    vkFreeCommandBuffers(vk().device, g.pool, 1, &cb);
    vmaDestroyBuffer(vk().vma, buf, alloc);
}
void save_shot(const char* path) {
    if (g.shot_out) g.out_path = path;  // (the next present reads its swapchain image back, see present_one)
    else if (g.cur >= 0) save_snaps({{g.cur, path}});
}

// Replay mode: the image for a present at time `t` (the newest packet whose window has begun: its image with the largest alpha at or below
// t's alpha, give or take a quarter present; below its first: the previous packet's own picture) and when the next image is due. With
// `wr` (reprojection on): when no rendered image is within a quarter present of t's alpha, the nearest one warped to it (wr->src >= 0).
// With the presenter queue (g.pq) only images whose replay has finished count (the own picture is done first); `wait_v` = the replay
// semaphore value the chosen images need.
static int pick_image(Clock::time_point t, Clock::time_point& next_due, WarpReq* wr = nullptr, uint64_t* wait_v = nullptr) {
    std::lock_guard<std::mutex> lk(g.pk_pub);
    next_due = t + std::chrono::milliseconds(5);
    if (wr) wr->src = -1;
    if (g.pq)
        if (VkSemaphore s = backend_replay_sem()) vkGetSemaphoreCounterValue(vk().device, s, &g.sem_seen);
    auto ok = [](uint64_t v) { return !g.pq || v <= g.sem_seen; };
    auto pkt = [](uint64_t back) -> const Packet& { return g.packets[(g.pk_seq - 1 - back) % kPackets]; };
    auto at = [](const Packet& p, double a) { return p.t + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(a * p.tick)); };
    const double tol = 0.25 * double(g.clk_period.load()) * 1e-9;
    for (uint64_t back = 0; back < 2 && back < g.pk_seq; ++back) {
        const Packet& p = pkt(back);
        const double alpha = (std::chrono::duration<double>(t - p.t).count() + tol) / p.tick;
        if (alpha < 0 && back == 0 && g.pk_seq > 1) { next_due = at(p, 0); continue; }  // its window has not begun
        const bool has_prev = back + 1 < g.pk_seq;
        const int prev_own = has_prev ? pkt(back + 1).idx[pkt(back + 1).n - 1] : -1;
        const uint64_t prev_v = has_prev ? pkt(back + 1).ready[pkt(back + 1).n - 1] : 0;
        if (p.cut) {  // a camera cut: the tick's own picture for the whole window, nothing in between (the previous one until it is done)
            const bool done = ok(p.ready[p.n - 1]) || prev_own < 0;
            if (back == 0) next_due = at(p, 1.0);
            if (wait_v) *wait_v = done ? p.ready[p.n - 1] : prev_v;
            return done ? p.idx[p.n - 1] : prev_own;
        }
        int img = prev_own >= 0 ? prev_own : p.idx[0];  // alpha 0: the previous tick's picture
        uint64_t img_v = prev_own >= 0 ? prev_v : p.ready[0];
        uint32_t i = 0;
        for (; i < p.n && p.a[i] <= alpha; ++i)
            if (ok(p.ready[i])) img = p.idx[i], img_v = p.ready[i];
        if (back == 0) next_due = at(p, i < p.n ? p.a[i] : 1.0);
        uint64_t src_v = 0;
        if (wr && p.warp) {
            const float a_t = float(std::clamp(alpha - tol / p.tick, 0.0, 1.0));
            const bool prev_ok = prev_own >= 0 && pkt(back + 1).warp;  // (its scene, pre and depth were written)
            int src = prev_ok ? prev_own : -1;
            float a_src = 0;
            src_v = prev_ok ? prev_v : 0;
            for (uint32_t j = 0; j < p.n; ++j)
                if (ok(p.ready[j]) && (src < 0 || std::fabs(p.a[j] - a_t) < std::fabs(a_src - a_t))) src = p.idx[j], a_src = p.a[j], src_v = p.ready[j];
            if (src >= 0 && std::fabs(a_src - a_t) * p.tick > tol) wr->src = src, wr->a_src = a_src, wr->a_t = a_t, wr->p = p;
        }
        if (wait_v) *wait_v = std::max(img_v, src_v);
        return img;
    }
    return -1;
}

void run(const std::atomic<bool>& stop) {
    const bool log = std::getenv("BB_PRESENT_LOG") != nullptr;
    auto last_log = Clock::now();
    auto last_shown = Clock::now();  // (replay mode)
    uint64_t log_presents = 0, log_frames = 0, log_distinct = 0, log_warps = 0;
    auto next = Clock::now();
    bool quit = false;
    while (!quit && !stop) {
        static const auto t_start = Clock::now();
        static double shot_secs = -1;
        static std::string shot_path;
        static const bool shot_parsed = [] {
            if (const char* e = std::getenv("BB_SHOT_AT")) if (const char* c = std::strchr(e, ':')) { shot_secs = std::atof(e); shot_path = c + 1; }
            return true;
        }();
        (void)shot_parsed;
        // BB_SHOT_N=<n>: n consecutive game frames (<file>_<k>.ppm) instead of one, to see flicker between frames
        static int shot_left = 0, shot_k = 0;
        static uint64_t shot_seen = 0;
        static uint64_t shot_pk_frame = 0;  // replay mode: the packet of the last shot triple (its warps are saved as <file>_<k>_w<j>.ppm)
        static std::string shot_pk_base;
        static int shot_wj = 0;
        // BB_SHOT_EVERY=<seconds>: after the first shot, one more every interval (<file>_t<seconds>.ppm), to see when an image state changes
        static const double shot_every = std::getenv("BB_SHOT_EVERY") ? std::atof(std::getenv("BB_SHOT_EVERY")) : 0;
        // BB_SHOT_CLOCK=pad: the shot times count on the pad script's clock (with BB_PAD_CLOCK=flips: the same game state in every run)
        static const bool shot_pad = std::getenv("BB_SHOT_CLOCK") && std::string(std::getenv("BB_SHOT_CLOCK")) == "pad";
        if (shot_secs >= 0 && (shot_pad ? pad_script_clock() : std::chrono::duration<double>(Clock::now() - t_start).count()) >= shot_secs) {
            shot_left = std::getenv("BB_SHOT_N") ? std::max(1, std::atoi(std::getenv("BB_SHOT_N"))) : 0;
            if (!shot_left) save_shot(shot_every > 0 ? (shot_path + "_t" + std::to_string(int(shot_secs)) + ".ppm").c_str() : shot_path.c_str());
            shot_secs = shot_every > 0 ? shot_secs + shot_every : -1;
        }
        if (shot_left > 0 && g.rep_max) {  // replay mode: <file>_<k>_<i>.ppm: i = 0 game frame k-1, then frame k's packet by alpha (last: frame k)
            // The packet's images are read back one packet later (the readback stalls the presenter: its warps, saved as they are shown,
            // come first); by then the render thread writes another packet's snapshots.
            static uint64_t pend = 0;  // pk_seq when the pending shot's packet was the newest
            static int pend_k = 0;
            Packet a, b;
            uint64_t seq;
            { std::lock_guard<std::mutex> lk(g.pk_pub); seq = g.pk_seq; const uint64_t s = pend ? pend : seq; if (s >= 2) a = g.packets[(s - 2) % kPackets], b = g.packets[(s - 1) % kPackets]; }
            if (pend && seq >= pend + 1) {
                std::lock_guard<std::mutex> sl(g.pk_snaps);  // (the render thread records and submits a packet under it: none of frame k-1's now)
                { std::lock_guard<std::mutex> lk(g.pk_pub); seq = g.pk_seq; }
                if (seq == pend + 1) {  // (later the render thread may already be rewriting frame k-1's snapshots: that attempt is dropped)
                    std::fprintf(stderr, "present: shot %d frame %llu alphas", pend_k, (unsigned long long)(b.frame + 1));  // (flip count, as with BB_FLIP_LOG)
                    for (uint32_t i = 0; i < b.n; ++i) std::fprintf(stderr, " %.3f", double(b.a[i]));
                    std::fputc('\n', stderr);
                    std::vector<std::pair<int, std::string>> v{{a.idx[a.n - 1], shot_pk_base + "0.ppm"}};
                    for (uint32_t i = 0; i < b.n; ++i) v.push_back({b.idx[i], shot_pk_base + std::to_string(i + 1) + ".ppm"});
                    save_snaps(v);
                    --shot_left;
                }
                pend = 0, shot_seen = seq, shot_pk_frame = 0;
            } else if (!pend && seq >= 2 && (!shot_seen || seq >= shot_seen + 1) && b.frame == a.frame + 1) {
                pend = seq, pend_k = shot_k++;
                shot_pk_frame = b.frame, shot_pk_base = shot_path + "_" + std::to_string(pend_k) + "_", shot_wj = 0;
            }
        } else if (shot_left > 0 && g.captured != shot_seen && g.cur >= 0) {
            shot_seen = g.captured;
            std::fprintf(stderr, "present: shot %d frame %llu\n", shot_k, (unsigned long long)g.captured);
            save_shot((shot_path + "_" + std::to_string(shot_k++) + ".ppm").c_str());
            --shot_left;
        }
        SDL_Event e;
        while (SDL_PollEvent(&e))
        { pad_handle_event(e); if (e.type == SDL_EVENT_QUIT) quit = true;
          if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_F11 && !e.key.repeat) SDL_SetWindowFullscreen(g.window, !(SDL_GetWindowFlags(g.window) & SDL_WINDOW_FULLSCREEN)); }
        const bool fresh = g.requested != g.captured;
        if (g.rep_max) {  // replay mode: each present shows the image its time asks for (pick_image)
            if (fresh) { g.captured = g.requested; ++log_frames; }
            auto now = Clock::now();
            auto ns = [](Clock::time_point t) { return int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count()); };
            auto present = [&](Clock::time_point t) {
                Clock::time_point due;
                WarpReq wr;
                uint64_t wait_v = 0;
                const int img = pick_image(t, due, g.warp ? &wr : nullptr, &wait_v);
                if (last_shown != Clock::time_point{}) g.intervals.push_back(std::chrono::duration<double, std::milli>(now - last_shown).count());
                last_shown = now;
                const bool warped = present_one(false, img, &wr, wait_v);
                g.distinct += warped || img != g.shown;
                g.shown = warped ? -2 : img;
                if (warped && shot_pk_frame && wr.p.frame == shot_pk_frame && shot_wj < 16) {  // (replay-mode shots: the warps of the shot's packet)
                    std::fprintf(stderr, "present: warp shot %sw%d alpha %.3f src %.3f\n", shot_pk_base.c_str(), shot_wj, double(wr.a_t), double(wr.a_src));
                    save_snaps({{int(g.nsnaps), shot_pk_base + "w" + std::to_string(shot_wj++) + ".ppm"}});
                }
            };
            if (g.fps > 0) {  // a fixed rate (MAILBOX): present on the clock's ticks
                if (now < next) {
                    if (next - now > std::chrono::microseconds(1500)) std::this_thread::sleep_for(next - now - std::chrono::microseconds(1000));
                    while (Clock::now() < next) std::this_thread::yield();
                }
                const auto period = std::chrono::nanoseconds(1000000000ll / g.fps);
                now = Clock::now();
                g.clk_ref = ns(next);
                present(next);
                next = std::max(next + period, now - period);  // after a stall do not burst to catch up
            } else if (g.fps == -2) {  // every display refresh (FIFO: acquiring/presenting blocks): the clock follows the measured presents
                if (last_shown != Clock::time_point{}) {
                    const int64_t dt = ns(now) - ns(last_shown);
                    if (dt > 2000000 && dt < 50000000) g.clk_period = g.clk_period + (dt - g.clk_period) / 16;
                }
                g.clk_ref = ns(now);
                present(now);
            } else {  // unlimited: present when the next image is due (sleep, spin its last ms; else poll packets and events every 5 ms); with
                // reprojection every iteration (each present its own alpha)
                Clock::time_point due;
                const int img = pick_image(now, due);
                if (g.warp || img != g.shown || now - last_shown >= std::chrono::milliseconds(100)) {
                    present(now);
                } else {
                    const bool exact = due < now + std::chrono::milliseconds(5);
                    const auto until = exact ? due : now + std::chrono::milliseconds(5);
                    if (until - now > std::chrono::microseconds(1500)) std::this_thread::sleep_for(until - now - std::chrono::microseconds(1000));
                    while (exact && Clock::now() < until) std::this_thread::yield();
                }
            }
        } else if (g.fps < 0) {  // native: one presentation per game frame
            if (fresh) {
                g.captured = g.requested;
                ++log_frames;
                present_one(true);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        } else {
            bool new_frame = false;
            if (fresh) { g.captured = g.requested; new_frame = true; ++log_frames; }
            const auto now = Clock::now();
            if (g.fps > 0) {  // pace to the chosen rate: sleep most of the way, spin the last 0.5 ms
                if (now < next) {
                    if (next - now > std::chrono::microseconds(1500)) std::this_thread::sleep_for(next - now - std::chrono::microseconds(1000));
                    while (Clock::now() < next) std::this_thread::yield();
                }
                const auto period = std::chrono::nanoseconds(1000000000ll / g.fps);
                next = std::max(next + period, Clock::now() - period);  // after a stall do not burst to catch up
            }
            present_one(new_frame);
        }
        if (log) {
            const auto now = Clock::now();
            if (now - last_log >= std::chrono::seconds(5)) {
                const double dt = std::chrono::duration<double>(now - last_log).count();
                std::fprintf(stderr, "present: %.1f fps shown, %.1f game frames/s (BB_FPS=%d, interpolation %s)", double(g.presents - log_presents) / dt, double(log_frames) / dt, g.fps, g.rep_max ? "on" : "off");
                if (g.rep_max) {
                    std::fprintf(stderr, " replay up to %u images per frame, budget %u: %.1f distinct images/s", g.rep_max, g.rep_budget.load(), double(g.distinct - log_distinct) / dt);
                    if (g.intervals.size() > 10) {  // present intervals (pacing)
                        std::sort(g.intervals.begin(), g.intervals.end());
                        auto q = [](double f) { return g.intervals[size_t(f * double(g.intervals.size() - 1))]; };
                        std::fprintf(stderr, ", present interval p50 %.2f p95 %.2f p99 %.2f ms", q(0.5), q(0.95), q(0.99));
                    }
                    g.intervals.clear();
                    if (g.warp) {
                        std::fprintf(stderr, ", %.1f warps/s", double(g.warps - log_warps) / dt);
                        if (!g.warp_ms.empty()) {
                            std::sort(g.warp_ms.begin(), g.warp_ms.end());
                            std::fprintf(stderr, " (GPU median %.2f ms, p95 %.2f ms)", g.warp_ms[g.warp_ms.size() / 2], g.warp_ms[size_t(0.95 * double(g.warp_ms.size() - 1))]);
                        }
                        g.warp_ms.clear();
                        log_warps = g.warps;
                    }
                }
                if (!g.out_ms.empty()) {  // the final pass (blend, or FSR EASU + RCAS) at the swapchain size; min: the least disturbed by other GPU work
                    std::sort(g.out_ms.begin(), g.out_ms.end());
                    std::fprintf(stderr, ", %s %ux%u GPU min %.3f median %.3f p95 %.3f ms", g.out_fsr ? "FSR" : "blend", g.extent.width, g.extent.height, g.out_ms[0],
                                 g.out_ms[g.out_ms.size() / 2], g.out_ms[size_t(0.95 * double(g.out_ms.size() - 1))]);
                    g.out_ms.clear();
                }
                {  // device memory in use (VMA's view: all heaps' allocations; growth over a long run = a leak)
                    VmaBudget bud[VK_MAX_MEMORY_HEAPS] = {};
                    vmaGetHeapBudgets(vk().vma, bud);
                    uint64_t used = 0;
                    for (const VmaBudget& b : bud) used += b.statistics.allocationBytes;
                    std::fprintf(stderr, ", vram %llu MB", (unsigned long long)(used >> 20));
                }
                std::fputc('\n', stderr);
                last_log = now;
                log_presents = g.presents;
                log_distinct = g.distinct;
                log_frames = 0;
            }
        }
    }
}

void shutdown() {
    if (vk().device) {
        vkDeviceWaitIdle(vk().device);
        destroy_swapchain();
        destroy_snaps();
        if (g.pipeline) vkDestroyPipeline(vk().device, g.pipeline, nullptr);
        if (g.layout) vkDestroyPipelineLayout(vk().device, g.layout, nullptr);
        if (g.dsl) vkDestroyDescriptorSetLayout(vk().device, g.dsl, nullptr);
        if (g.dpool) vkDestroyDescriptorPool(vk().device, g.dpool, nullptr);
        if (g.sampler) vkDestroySampler(vk().device, g.sampler, nullptr);
        if (g.warp_pipeline) vkDestroyPipeline(vk().device, g.warp_pipeline, nullptr);
        if (g.warp_pipeline_torn) vkDestroyPipeline(vk().device, g.warp_pipeline_torn, nullptr);
        if (g.warp_layout) vkDestroyPipelineLayout(vk().device, g.warp_layout, nullptr);
        if (g.warp_dsl) vkDestroyDescriptorSetLayout(vk().device, g.warp_dsl, nullptr);
        if (g.warp_dpool) vkDestroyDescriptorPool(vk().device, g.warp_dpool, nullptr);
        if (g.nearest) vkDestroySampler(vk().device, g.nearest, nullptr);
        if (g.warp_q) vkDestroyQueryPool(vk().device, g.warp_q, nullptr);
        if (g.easu) vkDestroyPipeline(vk().device, g.easu, nullptr);
        if (g.rcas) vkDestroyPipeline(vk().device, g.rcas, nullptr);
        if (g.fsr_layout) vkDestroyPipelineLayout(vk().device, g.fsr_layout, nullptr);
        if (g.up_view) vkDestroyImageView(vk().device, g.up_view, nullptr);
        if (g.up) vmaDestroyImage(vk().vma, g.up, g.up_alloc);
        if (g.out_buf) vmaDestroyBuffer(vk().vma, g.out_buf, g.out_alloc);
        if (g.out_q) vkDestroyQueryPool(vk().device, g.out_q, nullptr);
    }
    if (g.surface) vkDestroySurfaceKHR(vk().instance, g.surface, nullptr);
    if (g.window) SDL_DestroyWindow(g.window);
    SDL_Quit();
}

} // namespace bb::gpu
