#include "vulkan/vk_buffer.h"

#include <cstring>

VulkanBuffer::VulkanBuffer(VulkanContext& ctx, VkDeviceSize bytes,
                           VkBufferUsageFlags usage)
    : ctx_(ctx), size_(bytes) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(ctx_.device(), &bci, nullptr, &buffer_));

    VkMemoryRequirements mreq{};
    vkGetBufferMemoryRequirements(ctx_.device(), buffer_, &mreq);
    uint32_t mtype = ctx_.findHostVisibleMemType(mreq.memoryTypeBits);

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mreq.size;
    mai.memoryTypeIndex = mtype;
    VK_CHECK(vkAllocateMemory(ctx_.device(), &mai, nullptr, &memory_));
    VK_CHECK(vkBindBufferMemory(ctx_.device(), buffer_, memory_, 0));
}

VulkanBuffer::~VulkanBuffer() {
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx_.device(), buffer_, nullptr);
    }
    if (memory_ != VK_NULL_HANDLE)
        vkFreeMemory(ctx_.device(), memory_, nullptr);
}

void VulkanBuffer::upload(const void* src, std::size_t bytes,
                          VkDeviceSize offset) {
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(ctx_.device(), memory_, offset, bytes, 0, &mapped));
    std::memcpy(mapped, src, bytes);
    vkUnmapMemory(ctx_.device(), memory_);
}

void VulkanBuffer::download(void* dst, std::size_t bytes,
                            VkDeviceSize offset) const {
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(ctx_.device(), memory_, offset, bytes, 0, &mapped));
    std::memcpy(dst, mapped, bytes);
    vkUnmapMemory(ctx_.device(), memory_);
}

void VulkanBuffer::zero_fill(std::size_t bytes, VkDeviceSize offset) {
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(ctx_.device(), memory_, offset, bytes, 0, &mapped));
    std::memset(mapped, 0, bytes);
    vkUnmapMemory(ctx_.device(), memory_);
}
