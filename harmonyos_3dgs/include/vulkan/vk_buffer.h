// vk_buffer.h -- Host-visible coherent SSBO wrapper. Phase 1: no staging,
// direct map/unmap for upload/download. Device-local + staging buffers come
// in a later phase when we care about PCIe traffic during training.

#pragma once

#include "vulkan/vk_context.h"

#include <vulkan/vulkan.h>

#include <cstddef>

class VulkanBuffer {
public:
    /// Create a host-visible coherent storage buffer of `bytes` bytes.
    VulkanBuffer(VulkanContext& ctx, VkDeviceSize bytes,
                 VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ~VulkanBuffer();

    VulkanBuffer(const VulkanBuffer&)            = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    VkBuffer      handle() const { return buffer_; }
    VkDeviceSize  size()   const { return size_; }

    /// Copy `bytes` bytes from `src` to the buffer at `offset`.
    void upload(const void* src, std::size_t bytes, VkDeviceSize offset = 0);

    /// Copy `bytes` bytes from the buffer at `offset` to `dst`.
    void download(void* dst, std::size_t bytes, VkDeviceSize offset = 0) const;

    /// Fill `bytes` bytes of the buffer at `offset` with zero via a host
    /// memory map. Cheaper than uploading a zero-filled staging vector; used
    /// by the Phase 4 traced rasterize path to initialize cursor/wcur SSBOs
    /// before the shader ran.
    void zero_fill(std::size_t bytes, VkDeviceSize offset = 0);

private:
    VulkanContext& ctx_;
    VkBuffer       buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize   size_   = 0;
};
