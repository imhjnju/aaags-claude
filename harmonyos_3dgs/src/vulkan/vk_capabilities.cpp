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

    // Extensions
    caps.has_shader_atomic_float = query_atomic_float_ext(phys);

    return caps;
}
