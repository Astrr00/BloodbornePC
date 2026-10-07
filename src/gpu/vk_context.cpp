// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpu/vk_context.h"

#include <cstdio>
#include <cstring>
#include <vector>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace bb::gpu {

VkCtx& vk() {
    static VkCtx c;
    return c;
}

bool vk_check(VkResult r, const char* what) {
    if (r == VK_SUCCESS) return true;
    std::fprintf(stderr, "gpu: %s failed (VkResult %d)\n", what, int(r));
    return false;
}

bool vk_create_instance(const char* const* exts, uint32_t ext_count) {
    VkCtx& c = vk();
    if (c.instance) return true;
    if (!vk_check(volkInitialize(), "volkInitialize (no Vulkan loader?)")) return false;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "BloodbornePC";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = ext_count;
    ici.ppEnabledExtensionNames = exts;
    if (!vk_check(vkCreateInstance(&ici, nullptr, &c.instance), "vkCreateInstance")) return false;
    volkLoadInstance(c.instance);
    return true;
}

bool vk_create_device(VkSurfaceKHR surface) {
    VkCtx& c = vk();
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(c.instance, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(c.instance, &n, devs.data());
    for (VkPhysicalDevice d : devs) {  // first device with graphics (+ present) + compute; prefer discrete GPUs
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> q(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, q.data());
        for (uint32_t i = 0; i < qn; ++i) {
            VkBool32 present = surface == VK_NULL_HANDLE;
            if (surface) vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface, &present);
            if (!(q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) || !(q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) || !present) continue;
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(d, &props);
            if (!c.phys || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { c.phys = d; c.family = i; }
            break;
        }
    }
    if (!c.phys) { std::fputs("gpu: no Vulkan device with graphics+compute (+present) support\n", stderr); return false; }
    vkGetPhysicalDeviceProperties(c.phys, &c.props);
    std::fprintf(stderr, "gpu: using %s (Vulkan %u.%u)\n", c.props.deviceName, VK_VERSION_MAJOR(c.props.apiVersion), VK_VERSION_MINOR(c.props.apiVersion));
    if (c.props.apiVersion < VK_API_VERSION_1_3) { std::fputs("gpu: Vulkan 1.3 required (dynamic rendering)\n", stderr); return false; }

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.pNext = &f13;
    f12.timelineSemaphore = VK_TRUE;  // (core in 1.2: replay images signal the presenter's queue)
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    VkPhysicalDeviceFeatures& f = f2.features;
    VkPhysicalDeviceFeatures have;
    vkGetPhysicalDeviceFeatures(c.phys, &have);
    f.robustBufferAccess = have.robustBufferAccess;
    f.samplerAnisotropy = have.samplerAnisotropy;
    c.aniso = have.samplerAnisotropy == VK_TRUE;
    f.textureCompressionBC = have.textureCompressionBC;
    f.shaderStorageImageWriteWithoutFormat = have.shaderStorageImageWriteWithoutFormat;
    f.shaderStorageImageReadWithoutFormat = have.shaderStorageImageReadWithoutFormat;
    f.fragmentStoresAndAtomics = have.fragmentStoresAndAtomics;
    f.vertexPipelineStoresAndAtomics = have.vertexPipelineStoresAndAtomics;
    f.independentBlend = have.independentBlend;
    f.depthClamp = have.depthClamp;
    f.fillModeNonSolid = have.fillModeNonSolid;
    f.imageCubeArray = have.imageCubeArray;
    f.geometryShader = have.geometryShader;
    f.shaderInt64 = have.shaderInt64;
    f.shaderClipDistance = have.shaderClipDistance;  // POS1-3 exports (PA_CL_VS_OUT_CNTL clip/cull distances)
    f.shaderCullDistance = have.shaderCullDistance;

    // VK_KHR_fragment_shader_barycentric: PS inputs interpolated as P0 + i*(P1-P0) + j*(P2-P0) like GCN, which is exact for attributes
    // that are equal on all three vertices (material/instance ids converted to integers); plain smooth interpolation is not.
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    {
        VkPhysicalDeviceFeatures2 q2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        q2.pNext = &bary;
        vkGetPhysicalDeviceFeatures2(c.phys, &q2);
        c.bary = bary.fragmentShaderBarycentric == VK_TRUE;
        bary.pNext = f2.pNext;  // chain the queried struct (feature on) into the enabled features
        if (c.bary) f2.pNext = &bary;
    }
    // a second queue (if asked for and the family has one) at a higher priority for the presenter in the hybrid: its short submits need not
    // wait behind the replays
    uint32_t fam_queues = 1;
    {
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> q(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, q.data());
        if (c.family < qn) fam_queues = q[c.family].queueCount;
    }
    const bool two = c.want_queue2 && fam_queues >= 2 && !(std::getenv("BB_PRESENT_QUEUE") && std::getenv("BB_PRESENT_QUEUE")[0] == '0');
    const float prio[2] = {two ? 0.5f : 1.0f, 1.0f};
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = c.family;
    qci.queueCount = two ? 2 : 1;
    qci.pQueuePriorities = prio;
    std::vector<const char*> dev_exts;
    if (surface) dev_exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    if (c.bary) dev_exts.push_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
    {
        uint32_t next = 0;
        vkEnumerateDeviceExtensionProperties(c.phys, nullptr, &next, nullptr);
        std::vector<VkExtensionProperties> have_ext(next);
        vkEnumerateDeviceExtensionProperties(c.phys, nullptr, &next, have_ext.data());
        for (const auto& e : have_ext) {
            if (!std::strcmp(e.extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
                dev_exts.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
                VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
                VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                p2.pNext = &hp;
                vkGetPhysicalDeviceProperties2(c.phys, &p2);
                c.host_import = hp.minImportedHostPointerAlignment > 0 && !(std::getenv("BB_HOST_IMPORT") && std::getenv("BB_HOST_IMPORT")[0] == '0');  // BB_HOST_IMPORT=0: copy instead (diagnosis)
                c.host_import_align = hp.minImportedHostPointerAlignment;
            }
            // VK_KHR_push_descriptor: per-draw descriptors pushed into the command buffer instead of allocated, updated and bound
            if (!std::strcmp(e.extensionName, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) && !(std::getenv("BB_PUSH_DESC") && std::getenv("BB_PUSH_DESC")[0] == '0')) {  // BB_PUSH_DESC=0: pool path (diagnosis)
                dev_exts.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
                VkPhysicalDevicePushDescriptorPropertiesKHR pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
                VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                p2.pNext = &pp;
                vkGetPhysicalDeviceProperties2(c.phys, &p2);
                c.max_push = pp.maxPushDescriptors;
            }
        }
    }
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(dev_exts.size());
    dci.ppEnabledExtensionNames = dev_exts.data();
    if (!vk_check(vkCreateDevice(c.phys, &dci, nullptr, &c.device), "vkCreateDevice")) return false;
    volkLoadDevice(c.device);
    vkGetDeviceQueue(c.device, c.family, 0, &c.queue);
    if (two) vkGetDeviceQueue(c.device, c.family, 1, &c.queue2);

    VmaVulkanFunctions fn{};
    fn.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    fn.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo aci{};
    aci.physicalDevice = c.phys;
    aci.device = c.device;
    aci.instance = c.instance;
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    aci.pVulkanFunctions = &fn;
    if (!vk_check(vmaCreateAllocator(&aci, &c.vma), "vmaCreateAllocator")) return false;
    c.has_surface = surface != VK_NULL_HANDLE;
    return true;
}

void vk_destroy() {
    VkCtx& c = vk();
    if (c.device) vkDeviceWaitIdle(c.device);
    if (c.vma) vmaDestroyAllocator(c.vma);
    c.vma = nullptr;
    if (c.device) vkDestroyDevice(c.device, nullptr);
    c.device = VK_NULL_HANDLE;
    if (c.instance) vkDestroyInstance(c.instance, nullptr);
    c.instance = VK_NULL_HANDLE;
}

}  // namespace bb::gpu
