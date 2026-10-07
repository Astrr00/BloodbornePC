// SPDX-License-Identifier: GPL-3.0-or-later
// Shared Vulkan instance/device/queue + VMA allocator, used by the presenter and the GCN backend.
#pragma once
#include <volk.h>

#include <cstdint>
#include <mutex>
#include <vector>

struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;

namespace bb::gpu {

struct VkCtx {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    std::mutex queue_mutex;  // vkQueueSubmit/Present are externally synchronised
    // second queue of the family at a higher priority for the presenter: requested via want_queue2 before vk_create_device (the hybrid);
    // none with BB_PRESENT_QUEUE=0 or a single-queue family
    bool want_queue2 = false;
    VkQueue queue2 = VK_NULL_HANDLE;
    std::mutex queue2_mutex;
    VkPhysicalDeviceProperties props{};
    VmaAllocator vma = nullptr;
    bool has_surface = false;
    bool host_import = false;           // VK_EXT_external_memory_host: guest memory can back a VkBuffer without copying
    VkDeviceSize host_import_align = 0;  // minImportedHostPointerAlignment
    bool bary = false;                  // VK_KHR_fragment_shader_barycentric enabled (exact GCN-style interpolation)
    uint32_t max_push = 0;              // VK_KHR_push_descriptor enabled: maxPushDescriptors (0: not available or BB_PUSH_DESC=0)
    bool aniso = false;                 // samplerAnisotropy enabled (limit: props.limits.maxSamplerAnisotropy)
};

VkCtx& vk();

// Instance creation: `exts` are extra instance extensions (e.g. the surface extensions from SDL). Idempotent.
bool vk_create_instance(const char* const* exts, uint32_t ext_count);
// Device creation: surface may be VK_NULL_HANDLE for headless use. Requires the instance.
bool vk_create_device(VkSurfaceKHR surface);
void vk_destroy();

bool vk_check(VkResult r, const char* what);

}  // namespace bb::gpu
