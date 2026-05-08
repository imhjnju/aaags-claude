// vk_aligned_buffer.h -- Phase 2.3b helper: device-local SSBO with
// SHADER_DEVICE_ADDRESS support, plus over-allocate-and-bind-at-aligned-offset
// to satisfy library-imposed alignment requirements (Fuchsia vk_radix_sort
// demands 256-byte alignment for its keyval and internal scratch buffers).
//
// Pattern mirrors `spike/fuchsia_radix_sort/bench_radix.cpp:51-89` and
// `tests/test_fuchsia_radix_wrapper.cpp::make_raw_buffer`.
//
// Why a separate type from VulkanBuffer:
//   - VulkanBuffer is host-visible-coherent (Phase-1 abstraction). Fuchsia
//     requires DEVICE_LOCAL + SHADER_DEVICE_ADDRESS_BIT.
//   - VulkanBuffer binds at offset 0 with mreq.size; it does NOT honor a
//     min_alignment stricter than the buffer's natural alignment.
//
// The "aligned offset" returned by handle_offset() is always 0 in practice on
// Tegra Thor (DEVICE_LOCAL natural alignment >= 256), but the helper keeps the
// safety net for weaker devices (Maleoon Bifrost8 — see Plan §7 risk #2).

#pragma once

#include "vulkan/vk_context.h"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>

class VulkanAlignedBuffer {
public:
    /// Create a DEVICE_LOCAL VkBuffer of `size` bytes with usage flags. If
    /// `min_alignment` exceeds the buffer's natural mreq.alignment, the
    /// allocation is over-allocated by `min_alignment` bytes and the buffer
    /// is bound at the aligned offset.
    /// `usage` must include VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT if the
    /// caller wants the Vulkan Memory Model device-address path.
    VulkanAlignedBuffer(VulkanContext& ctx,
                        VkDeviceSize size,
                        VkDeviceSize min_alignment,
                        VkBufferUsageFlags usage);

    ~VulkanAlignedBuffer();

    VulkanAlignedBuffer(const VulkanAlignedBuffer&)            = delete;
    VulkanAlignedBuffer& operator=(const VulkanAlignedBuffer&) = delete;

    VkBuffer handle() const { return buffer_; }
    /// Always 0 in practice (we bind at the aligned offset, so the buffer
    /// itself starts at offset 0 within its memory range). Provided for
    /// completeness and future-proofing if we ever switch to suballocation.
    VkDeviceSize handle_offset() const { return bind_offset_; }
    VkDeviceSize size() const { return size_; }

private:
    VulkanContext&  ctx_;
    VkBuffer        buffer_      = VK_NULL_HANDLE;
    VkDeviceMemory  memory_      = VK_NULL_HANDLE;
    VkDeviceSize    size_        = 0;
    VkDeviceSize    bind_offset_ = 0;
};
