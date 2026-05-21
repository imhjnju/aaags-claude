#include "vulkan/vk_context.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Enforce spec §3.1 required minimums. A device that fails any check is
// rejected during selection; the first check that fails is logged.
constexpr VkSubgroupFeatureFlags kRequiredSubgroupOps =
    VK_SUBGROUP_FEATURE_BASIC_BIT |
    VK_SUBGROUP_FEATURE_BALLOT_BIT |
    VK_SUBGROUP_FEATURE_SHUFFLE_BIT;

static bool meets_minimums(const VulkanDeviceCapabilities& c) {
    if (c.api_version < VK_API_VERSION_1_1) return false;
    if (c.max_push_constants_size < 128u) return false;
    if (c.max_compute_workgroup_invocations < 256u) return false;
    if (c.max_compute_shared_memory_size < 16384u) return false;
    if (c.max_compute_workgroup_size[0] < 256u) return false;
    if (c.max_compute_workgroup_size[1] < 256u) return false;
    if (c.max_compute_workgroup_size[2] < 64u) return false;
    if (!(c.subgroup_supported_stages & VK_SHADER_STAGE_COMPUTE_BIT)) return false;
    if ((c.subgroup_supported_ops & kRequiredSubgroupOps) != kRequiredSubgroupOps) return false;
    return true;
}

// Print the first failing minimum for diagnostics. Mirrors the order of
// meets_minimums() so "what failed" matches "what was checked".
static void log_minimums_failure(const char* device_name,
                                 const VulkanDeviceCapabilities& c) {
    const char* reason = "unknown";
    if (c.api_version < VK_API_VERSION_1_1)                        reason = "api_version < 1.1";
    else if (c.max_push_constants_size < 128u)                     reason = "max_push_constants_size < 128";
    else if (c.max_compute_workgroup_invocations < 256u)           reason = "max_compute_workgroup_invocations < 256";
    else if (c.max_compute_shared_memory_size < 16384u)            reason = "max_compute_shared_memory_size < 16384";
    else if (c.max_compute_workgroup_size[0] < 256u)               reason = "max_compute_workgroup_size[0] < 256";
    else if (c.max_compute_workgroup_size[1] < 256u)               reason = "max_compute_workgroup_size[1] < 256";
    else if (c.max_compute_workgroup_size[2] < 64u)                reason = "max_compute_workgroup_size[2] < 64";
    else if (!(c.subgroup_supported_stages & VK_SHADER_STAGE_COMPUTE_BIT))
        reason = "subgroup_supported_stages lacks COMPUTE";
    else if ((c.subgroup_supported_ops & kRequiredSubgroupOps) != kRequiredSubgroupOps)
        reason = "subgroup_supported_ops lacks BASIC/BALLOT/SHUFFLE";
    std::fprintf(stderr,
                 "[vk_context] rejecting device \"%s\": %s\n",
                 device_name ? device_name : "<unnamed>", reason);
}

namespace {

struct Candidate {
    VkPhysicalDevice dev;
    uint32_t         qf;
    std::string      name;
    VkPhysicalDeviceType type;
    VulkanDeviceCapabilities caps;
};

int device_type_rank(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 0;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return 2;
        default:                                     return 3;
    }
}

}  // namespace

bool VulkanContext::init() {
    if (instance_ != VK_NULL_HANDLE) return true;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "gs3d_vk";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS) {
        instance_ = VK_NULL_HANDLE;
        return false;
    }

    // Enumerate all physical devices
    uint32_t n_phys = 0;
    vkEnumeratePhysicalDevices(instance_, &n_phys, nullptr);
    if (n_phys == 0) {
        std::cerr << "VulkanContext: no physical devices\n";
        release();
        return false;
    }
    std::vector<VkPhysicalDevice> phys_list(n_phys);
    vkEnumeratePhysicalDevices(instance_, &n_phys, phys_list.data());

    // Env overrides
    const char* env_index = std::getenv("GS3D_VK_DEVICE");
    const char* env_name  = std::getenv("GS3D_VK_DEVICE_NAME");

    // Gather candidates passing minimums + having compute queue
    std::vector<Candidate> candidates;
    for (uint32_t i = 0; i < n_phys; ++i) {
        VkPhysicalDevice pd = phys_list[i];
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);

        // Find compute queue family
        uint32_t nqf = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(nqf);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfs.data());
        uint32_t chosen_qf = UINT32_MAX;
        for (uint32_t q = 0; q < nqf; ++q) {
            if (qfs[q].queueFlags & VK_QUEUE_COMPUTE_BIT) { chosen_qf = q; break; }
        }
        if (chosen_qf == UINT32_MAX) continue;

        VulkanDeviceCapabilities caps = probe_capabilities(pd);
        if (!meets_minimums(caps)) {
            log_minimums_failure(props.deviceName, caps);
            continue;
        }

        candidates.push_back({pd, chosen_qf, std::string(props.deviceName),
                              props.deviceType, caps});
    }
    if (candidates.empty()) {
        std::cerr << "VulkanContext: no device passes capability minimums\n";
        release();
        return false;
    }

    // Env override by index
    bool picked = false;
    if (env_index) {
        int want = std::atoi(env_index);
        if (want < 0 || static_cast<uint32_t>(want) >= n_phys) {
            std::cerr << "GS3D_VK_DEVICE=" << want << " out of range (0.."
                      << (n_phys-1) << ")\n";
            release();
            return false;
        }
        VkPhysicalDevice wanted = phys_list[want];
        for (auto& c : candidates) {
            if (c.dev == wanted) {
                phys_        = c.dev;
                compute_qf_  = c.qf;
                device_name_ = c.name;
                api_version_ = c.caps.api_version;
                caps_        = c.caps;
                picked = true;
                break;
            }
        }
        if (!picked) {
            std::cerr << "GS3D_VK_DEVICE index selected a device that fails minimums\n";
            release();
            return false;
        }
    }

    // Env override by name substring
    if (!picked && env_name) {
        std::string needle(env_name);
        for (auto& c : candidates) {
            if (c.name.find(needle) != std::string::npos) {
                phys_        = c.dev;
                compute_qf_  = c.qf;
                device_name_ = c.name;
                api_version_ = c.caps.api_version;
                caps_        = c.caps;
                picked = true;
                break;
            }
        }
        if (!picked) {
            std::cerr << "GS3D_VK_DEVICE_NAME=" << needle
                      << " matched no capable device\n";
            release();
            return false;
        }
    }

    // Auto-priority: DISCRETE > INTEGRATED > CPU > OTHER
    if (!picked) {
        std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                return device_type_rank(a.type) < device_type_rank(b.type);
            });
        phys_        = candidates[0].dev;
        compute_qf_  = candidates[0].qf;
        device_name_ = candidates[0].name;
        api_version_ = candidates[0].caps.api_version;
        caps_        = candidates[0].caps;
    }

    float qp = 1.0f;
    VkDeviceQueueCreateInfo dqci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dqci.queueFamilyIndex = compute_qf_;
    dqci.queueCount = 1;
    dqci.pQueuePriorities = &qp;

    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.bufferDeviceAddress = caps_.has_buffer_device_address ? VK_TRUE : VK_FALSE;
    f12.vulkanMemoryModel = caps_.has_vulkan_memory_model ? VK_TRUE : VK_FALSE;
    f12.vulkanMemoryModelDeviceScope = caps_.has_vulkan_memory_model_device_scope ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.features.shaderInt64 = caps_.has_shader_int64 ? VK_TRUE : VK_FALSE;
    f2.features.shaderInt16 = caps_.has_shader_int16 ? VK_TRUE : VK_FALSE;
    if (caps_.api_version >= VK_API_VERSION_1_2) {
        f2.pNext = &f12;
    }

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dqci;
    dci.pEnabledFeatures = nullptr;
    if (vkCreateDevice(phys_, &dci, nullptr, &device_) != VK_SUCCESS) {
        release();
        return false;
    }
    vkGetDeviceQueue(device_, compute_qf_, 0, &compute_queue_);

    {
        uint32_t nqf = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &nqf, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(nqf);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &nqf, qfs.data());
        if (compute_qf_ < nqf) caps_.timestamp_valid_bits = qfs[compute_qf_].timestampValidBits;
    }

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = compute_qf_;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device_, &cpci, nullptr, &command_pool_) !=
        VK_SUCCESS) {
        release();
        return false;
    }

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(device_, &fci, nullptr, &submit_fence_) != VK_SUCCESS) {
        release();
        return false;
    }
    return true;
}

void VulkanContext::release() {
    if (device_ != VK_NULL_HANDLE)
        vkDeviceWaitIdle(device_);   // ensure GPU is idle before releasing resources
    if (submit_fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_, submit_fence_, nullptr);
        submit_fence_ = VK_NULL_HANDLE;
    }
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
    }
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
    phys_ = VK_NULL_HANDLE;
    compute_queue_ = VK_NULL_HANDLE;
    compute_qf_ = 0;
    device_name_.clear();
    api_version_ = 0;
    caps_ = {};
}

VulkanContext::~VulkanContext() { release(); }

uint32_t VulkanContext::findHostVisibleMemType(uint32_t type_bits) const {
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(phys_, &mem);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & want) == want)
            return i;
    }
    throw std::runtime_error("no host-visible+coherent memory type");
}

uint32_t VulkanContext::findDeviceLocalMemType(uint32_t type_bits) const {
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(phys_, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags &
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            return i;
    }
    throw std::runtime_error("no device-local memory type");
}

VkCommandBuffer VulkanContext::allocatePrimary() {
    VkCommandBufferAllocateInfo cbai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = command_pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device_, &cbai, &cmd));
    return cmd;
}

void VulkanContext::freePrimary(VkCommandBuffer cmd) {
    if (cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
}

void VulkanContext::submitAndWait(VkCommandBuffer cmd) {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkResetFences(device_, 1, &submit_fence_));
    VK_CHECK(vkQueueSubmit(compute_queue_, 1, &si, submit_fence_));
    VK_CHECK(vkWaitForFences(device_, 1, &submit_fence_, VK_TRUE, UINT64_MAX));
}
