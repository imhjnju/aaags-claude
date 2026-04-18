// vk_context.h -- RAII wrapper around a Vulkan instance, device, compute
// queue, and command pool. Mirrors the style of opencl_context.h.

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vulkan/vk_capabilities.h"

VulkanDeviceCapabilities probe_capabilities(VkPhysicalDevice phys);

// ---------------------------------------------------------------------------
// VK_CHECK -- throw on any non-VK_SUCCESS result.
// ---------------------------------------------------------------------------
#define VK_CHECK(fn_call)                                                   \
    do {                                                                    \
        VkResult _vk_err = (fn_call);                                       \
        if (_vk_err != VK_SUCCESS)                                          \
            throw std::runtime_error(                                       \
                std::string("Vulkan error ") + std::to_string(_vk_err) +    \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));        \
    } while (0)

// ---------------------------------------------------------------------------
// VulkanContext -- owns instance, physical device, logical device, compute
// queue, and a command pool keyed to that queue family. init() picks the
// first physical device with a compute-capable queue family.
// ---------------------------------------------------------------------------
class VulkanContext {
public:
    /// Create instance, pick compute device, create logical device + command
    /// pool. Returns false if no Vulkan device with compute queue is present.
    bool init();

    /// Release all Vulkan resources. Safe to call multiple times.
    void release();

    ~VulkanContext();

    // -- Accessors ----------------------------------------------------------

    VkInstance        instance()           const { return instance_; }
    VkPhysicalDevice  physicalDevice()     const { return phys_; }
    VkDevice          device()             const { return device_; }
    VkQueue           computeQueue()       const { return compute_queue_; }
    uint32_t          computeQueueFamily() const { return compute_qf_; }
    VkCommandPool     commandPool()        const { return command_pool_; }

    const std::string& deviceName() const { return device_name_; }
    uint32_t apiVersion() const { return api_version_; }
    const VulkanDeviceCapabilities& capabilities() const { return caps_; }

    // -- Memory helpers -----------------------------------------------------

    /// Find a memory type matching type_bits with HOST_VISIBLE + HOST_COHERENT.
    /// Throws if no match.
    uint32_t findHostVisibleMemType(uint32_t type_bits) const;

    /// Find a memory type matching type_bits with DEVICE_LOCAL.
    /// Throws if no match.
    uint32_t findDeviceLocalMemType(uint32_t type_bits) const;

    // -- Command buffers ----------------------------------------------------

    /// Allocate a primary command buffer from the owned pool.
    VkCommandBuffer allocatePrimary();

    /// Free a primary command buffer back to the owned pool.
    void freePrimary(VkCommandBuffer cmd);

    /// Submit cmd on compute queue and vkQueueWaitIdle. Single-submit style.
    void submitAndWait(VkCommandBuffer cmd);

private:
    VkInstance       instance_      = VK_NULL_HANDLE;
    VkPhysicalDevice phys_          = VK_NULL_HANDLE;
    VkDevice         device_        = VK_NULL_HANDLE;
    VkQueue          compute_queue_ = VK_NULL_HANDLE;
    uint32_t         compute_qf_    = 0;
    VkCommandPool    command_pool_  = VK_NULL_HANDLE;
    std::string      device_name_;
    uint32_t         api_version_   = 0;
    VulkanDeviceCapabilities caps_{};
};
