#include "vulkan/vk_context.h"

#include <cstdio>
#include <vector>

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

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    if (n == 0) { release(); return false; }
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance_, &n, devs.data());

    bool picked = false;
    for (VkPhysicalDevice pd : devs) {
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qs.data());
        for (uint32_t i = 0; i < qn; ++i) {
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                phys_ = pd;
                compute_qf_ = i;
                picked = true;
                break;
            }
        }
        if (picked) break;
    }
    if (!picked) { release(); return false; }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys_, &props);
    device_name_ = props.deviceName;
    api_version_ = props.apiVersion;

    float qp = 1.0f;
    VkDeviceQueueCreateInfo dqci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dqci.queueFamilyIndex = compute_qf_;
    dqci.queueCount = 1;
    dqci.pQueuePriorities = &qp;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dqci;
    if (vkCreateDevice(phys_, &dci, nullptr, &device_) != VK_SUCCESS) {
        release();
        return false;
    }
    vkGetDeviceQueue(device_, compute_qf_, 0, &compute_queue_);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = compute_qf_;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device_, &cpci, nullptr, &command_pool_) !=
        VK_SUCCESS) {
        release();
        return false;
    }
    return true;
}

void VulkanContext::release() {
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
    VK_CHECK(vkQueueSubmit(compute_queue_, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(compute_queue_));
}
