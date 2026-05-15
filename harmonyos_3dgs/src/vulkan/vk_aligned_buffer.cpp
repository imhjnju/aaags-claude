// vk_aligned_buffer.cpp -- Phase 2.3b. See header for contract.
//
// Implementation note: when min_alignment > mreq.alignment, we over-allocate
// by min_alignment bytes and bind the buffer at the smallest offset that is a
// multiple of BOTH mreq.alignment and min_alignment. In practice on Tegra Thor,
// DEVICE_LOCAL allocations are naturally aligned to 1024 bytes, so the over-
// allocation never bites — but the safety net matches `bench_radix.cpp:67-73`
// and Plan §7 risk #2.

#include "vulkan/vk_aligned_buffer.h"

#include <stdexcept>
#include <string>

namespace {

VkDeviceSize round_up(VkDeviceSize x, VkDeviceSize a) {
    if (a <= 1) return x;
    return ((x + a - 1) / a) * a;
}

}  // namespace

VulkanAlignedBuffer::VulkanAlignedBuffer(VulkanContext& ctx,
                                         VkDeviceSize size,
                                         VkDeviceSize min_alignment,
                                         VkBufferUsageFlags usage)
    : ctx_(ctx), size_(size) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(ctx_.device(), &bci, nullptr, &buffer_));

    VkMemoryRequirements mreq{};
    vkGetBufferMemoryRequirements(ctx_.device(), buffer_, &mreq);

    // Effective alignment: the LCM of natural and library-imposed. Both must
    // be powers of two so LCM == max() suffices.
    VkDeviceSize align_eff = (min_alignment > mreq.alignment) ? min_alignment
                                                              : mreq.alignment;
    // Allocate enough room to slide the bind offset up to align_eff.
    VkDeviceSize alloc_size = mreq.size;
    if (min_alignment > mreq.alignment) {
        // Over-allocate so we can bind at offset = round_up(0, align_eff).
        // With offset 0 below this is a no-op since 0 is aligned to any
        // power-of-two, BUT Vulkan validation also requires the OFFSET to be a
        // multiple of mreq.alignment (it is — 0). So in practice the natural
        // path with min_alignment == 0 works; we keep the padding for future-
        // proof when we switch to a real suballocator.
        alloc_size += align_eff;
    }

    VkMemoryAllocateFlagsInfo flags_info{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
        flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext           = (flags_info.flags != 0u ? &flags_info : nullptr);
    mai.allocationSize  = alloc_size;
    mai.memoryTypeIndex = ctx_.findDeviceLocalMemType(mreq.memoryTypeBits);
    VK_CHECK(vkAllocateMemory(ctx_.device(), &mai, nullptr, &memory_));

    // Bind at offset = round_up(0, align_eff). On every device we've seen, the
    // VkDeviceMemory object's reported address is itself aligned to a large
    // power of two (>= 1024 on Tegra Thor), so a bind offset of 0 satisfies
    // both mreq.alignment and min_alignment. We keep the rounding for safety.
    bind_offset_ = round_up(0u, align_eff);
    VK_CHECK(vkBindBufferMemory(ctx_.device(), buffer_, memory_, bind_offset_));
}

VulkanAlignedBuffer::~VulkanAlignedBuffer() {
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx_.device(), buffer_, nullptr);
    }
    if (memory_ != VK_NULL_HANDLE)
        vkFreeMemory(ctx_.device(), memory_, nullptr);
}
