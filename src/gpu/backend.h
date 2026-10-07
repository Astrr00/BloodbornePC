// SPDX-License-Identifier: GPL-3.0-or-later
// Vulkan backend for the PS4 GPU: executes the draws/dispatches the PM4 interpreter reports (see gpu_hooks.h) by
// translating the GCN shaders to SPIR-V and mapping guest resources (V#/T#/S#, render targets) to Vulkan objects.
#pragma once
#include <volk.h>

#include <cstdint>

namespace bb::gpu {

// Creates backend state (requires vk_create_device) and installs the hooks consumed by the Gnm HLE layer.
bool backend_init();
void backend_shutdown();

// BB_RES_SCALE=<1..3> (default 1): internal resolution factor. Screen-sized render and depth targets live at guest size times this factor;
// guest registers, constants and memory stay in guest pixels (the backend scales at the Vulkan boundary). 1 when the device cannot
// (push constants < 148 bytes). backend_blit_frame / backend_replay_tick keep taking guest sizes; the presenter's images are this times larger.
uint32_t res_scale();

// Records a blit of the render target living at guest address `addr` into `dst` (swapchain image, currently in
// TRANSFER_DST layout). Returns false if no such render target exists (nothing drawn yet).
// Frame boundary (guest flip): textures are re-validated against guest memory at their next use (as after every guest submit).
void backend_new_frame();
bool backend_blit_frame(uint64_t addr, uint32_t width, uint32_t height, VkCommandBuffer cb, VkImage dst, VkExtent2D dst_extent);

// Interpolation hybrid replay (present.cpp, BB_FPS above 30). backend_request_replay: before backend_init, records each tick as replayable segments.
// backend_replay_tick (render thread, at the guest flip): dst[n-1] <- the tick's display target; dst[i] (i < n-1) <- the tick's work run
// again with camera and objects at alpha[i] (0 = the previous tick, 1 = this one). dst end up SHADER_READ_ONLY (one submit, queue order).
// With `warp`: per image also the HUD inputs: scene[i] = the display pass's source before the HUD's draws, pre[i] = that source after
// them (both before the display pass, e.g. its colour LUT; they differ exactly where the HUD is; same size as dst, SHADER_READ_ONLY), and
// the main camera's depth (depth[i]: width * height floats, the depth buffer's raw values), plus the main camera of both ticks (for
// reprojection, see present.cpp). false: no replay mode or no such render target.
struct ReplayWarp {
    const VkImage* scene = nullptr;
    const VkImage* pre = nullptr;
    const VkBuffer* depth = nullptr;
    bool ok = false;           // out: scene, pre, depth and camera are valid for every image
    float cam0[64], cam1[64];  // out: the main camera block's dwords 8-71 (view 3x4, ..., projection) in the previous tick and this one
    float vport[6];            // out: its viewport: x scale, x offset, y scale, y offset, z scale, z offset (window = scale * ndc + offset)
};
void backend_request_replay();
bool backend_replay_supported();  // false: the backend cannot replay (BB_GPU_PROF / BB_GPU_TS timestamps live in its command buffers)
// `ready` (optional, n values): per image the value backend_replay_sem() (a timeline semaphore) reaches once that image is written; with it
// the work is submitted in batches (the tick's own picture first), so a presenter on another queue can use each image as soon as it is done.
// `res` (optional, out): cut = the main camera jumped (or has no partner in the previous tick), replayed = the in-between images were
// rendered; without them (a cut, a dropped tick, no replayable work) every dst is the tick's own picture.
struct ReplayResult { bool cut = false, replayed = false; };
bool backend_replay_tick(uint64_t addr, uint32_t width, uint32_t height, const VkImage* dst, VkExtent2D dst_extent, uint32_t n, const float* alpha, ReplayWarp* warp,
                         uint64_t* ready = nullptr, ReplayResult* res = nullptr);
VkSemaphore backend_replay_sem();
double backend_replay_gpu_ms();   // GPU time of the last measured replay
double backend_replay_hold_ms();  // how long the last tick's recording waited for an earlier replay to finish

}  // namespace bb::gpu
