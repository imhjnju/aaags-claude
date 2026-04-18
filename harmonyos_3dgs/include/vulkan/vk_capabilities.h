// vk_capabilities.h -- Device capability snapshot populated at VulkanContext
// init time and cached for the life of the context. Used by pipeline-time
// validation (e.g. push-constant size limit) and SPIR-V selection (e.g.
// native atomic float vs CAS fallback path in SP-3 onward).

#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

struct VulkanDeviceCapabilities {
    // Required minimums per spec §3.1. VulkanContext::init() rejects a
    // device that does not satisfy these.
    uint32_t max_push_constants_size        = 0;   // >= 128
    uint32_t max_compute_workgroup_invocations = 0; // >= 256
    uint32_t max_compute_shared_memory_size  = 0;   // >= 16384
    uint32_t max_compute_workgroup_size[3]   = {0, 0, 0}; // >= {256,256,64}

    // Subgroup properties (recorded; enforcement is per-shader in SP-2+).
    uint32_t               subgroup_size              = 0;
    VkShaderStageFlags     subgroup_supported_stages  = 0;   // must contain COMPUTE
    VkSubgroupFeatureFlags subgroup_supported_ops     = 0;

    // Extension detection (affects SP-3 backward CAS fallback).
    bool has_shader_atomic_float = false;   // VK_EXT_shader_atomic_float

    // Vulkan API version the device reports.
    uint32_t api_version = 0;               // >= VK_API_VERSION_1_1
};
