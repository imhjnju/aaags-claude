#include "vulkan/vk_capabilities.h"
#include <cstring>
#include <vector>

namespace {

bool query_atomic_float_ext(VkPhysicalDevice dev) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> props(n);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, props.data());
    for (const auto& e : props) {
        if (std::strcmp(e.extensionName, "VK_EXT_shader_atomic_float") == 0)
            return true;
    }
    return false;
}

}  // namespace

VulkanDeviceCapabilities probe_capabilities(VkPhysicalDevice phys) {
    VulkanDeviceCapabilities caps;

    // Basic props (includes api_version, limits)
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);
    caps.api_version = props.apiVersion;
    caps.max_push_constants_size         = props.limits.maxPushConstantsSize;
    caps.max_compute_workgroup_invocations = props.limits.maxComputeWorkGroupInvocations;
    caps.max_compute_shared_memory_size   = props.limits.maxComputeSharedMemorySize;
    caps.max_compute_workgroup_size[0]    = props.limits.maxComputeWorkGroupSize[0];
    caps.max_compute_workgroup_size[1]    = props.limits.maxComputeWorkGroupSize[1];
    caps.max_compute_workgroup_size[2]    = props.limits.maxComputeWorkGroupSize[2];
    caps.timestamp_period                 = props.limits.timestampPeriod;

    // Subgroup props (Vulkan 1.1+ core; via pNext chain)
    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(phys, &props2);
    caps.subgroup_size             = subgroup.subgroupSize;
    caps.subgroup_supported_stages = subgroup.supportedStages;
    caps.subgroup_supported_ops    = subgroup.supportedOperations;

    if (props.apiVersion >= VK_API_VERSION_1_2) {
        VkPhysicalDeviceVulkan12Features f12{};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &f12;
        vkGetPhysicalDeviceFeatures2(phys, &f2);
        caps.has_shader_int64                      = f2.features.shaderInt64;
        caps.has_shader_int16                      = f2.features.shaderInt16;
        caps.has_buffer_device_address             = f12.bufferDeviceAddress;
        caps.has_vulkan_memory_model               = f12.vulkanMemoryModel;
        caps.has_vulkan_memory_model_device_scope  = f12.vulkanMemoryModelDeviceScope;
    }

    // Extensions
    caps.has_shader_atomic_float = query_atomic_float_ext(phys);

    return caps;
}
