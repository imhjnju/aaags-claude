// binning_download.h -- env-aware download helper for BinningOutput.
//
// Phase 2.3b finalization: under env-on (GS3D_USE_FUCHSIA_SORT=1) the sorter
// no longer downloads sorted keyvals / tile_ranges to the host — it leaves
// `bin.keyvals_sorted` and `bin.tile_ranges` null and publishes the GPU
// VkBuffer handles in `bin.keyvals_sorted_gpu` / `bin.tile_ranges_gpu`.
//
// The two test invariants we verify (monotonic sorted keyvals + tile_ranges
// internal consistency) MUST stay identical across env states. Only the read
// mechanism changes. This helper hides the carrier:
//   - if the host pointer is non-null  -> return it directly (env-off path).
//   - else                              -> stage-copy the GPU buffer into the
//                                          caller-provided host_buf and return
//                                          its data() pointer (env-on path).
//
// The helper is header-only (inline) so it doesn't need a separate .cpp in
// the test target's source list.

#pragma once

#include "types.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/vk_context.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace test_helpers {

namespace detail {

// Stage-copy `bytes` bytes from `src_gpu` into a freshly-allocated
// host-visible staging buffer, then download into `host_buf`. Submits a
// one-shot command buffer and waits for completion (single-frame-per-call
// pattern matches the sync rasterizer / sorter paths).
//
// `src_gpu` must have been created with VK_BUFFER_USAGE_TRANSFER_SRC_BIT.
inline void stage_download(VulkanContext& ctx,
                           VkBuffer src_gpu,
                           std::size_t bytes,
                           void* host_dst) {
    if (src_gpu == VK_NULL_HANDLE)
        throw std::runtime_error(
            "test_helpers::stage_download: src_gpu is VK_NULL_HANDLE");
    if (bytes == 0u || host_dst == nullptr) return;

    VulkanBuffer staging(ctx,
                         static_cast<VkDeviceSize>(bytes),
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    VkCommandBuffer cmd = ctx.allocatePrimary();
    if (cmd == VK_NULL_HANDLE)
        throw std::runtime_error(
            "test_helpers::stage_download: allocatePrimary failed");

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
        throw std::runtime_error(
            "test_helpers::stage_download: vkBeginCommandBuffer failed");

    // Barrier: previous compute-shader writes -> transfer read on src_gpu.
    {
        VkBufferMemoryBarrier mb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mb.buffer = src_gpu;
        mb.offset = 0;
        mb.size   = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &mb, 0, nullptr);
    }

    VkBufferCopy cp{0, 0, static_cast<VkDeviceSize>(bytes)};
    vkCmdCopyBuffer(cmd, src_gpu, staging.handle(), 1, &cp);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        throw std::runtime_error(
            "test_helpers::stage_download: vkEndCommandBuffer failed");

    ctx.submitAndWait(cmd);
    ctx.freePrimary(cmd);

    staging.download(host_dst, bytes);
}

}  // namespace detail

// Returns a host pointer to bin.keyvals_sorted[R].
// env-off: returns bin.keyvals_sorted directly (no copy).
// env-on : stage-downloads bin.keyvals_sorted_gpu into host_buf and returns
//          host_buf.data(). host_buf is resized to R.
inline const uint64_t* keyvals_sorted_host_view(
    VulkanContext& ctx,
    const BinningOutput& bin,
    std::vector<uint64_t>& host_buf) {
    if (bin.keyvals_sorted != nullptr) {
        return bin.keyvals_sorted;
    }
    if (bin.keyvals_sorted_gpu == nullptr) {
        throw std::runtime_error(
            "test_helpers::keyvals_sorted_host_view: both host and GPU "
            "carriers are null");
    }
    const std::size_t R = static_cast<std::size_t>(bin.total_pairs);
    host_buf.assign(R, 0u);
    detail::stage_download(ctx,
                           static_cast<VkBuffer>(bin.keyvals_sorted_gpu),
                           R * sizeof(uint64_t),
                           host_buf.data());
    return host_buf.data();
}

// Returns a host pointer to bin.tile_ranges[num_tiles*2] (flat start/end
// pairs, uint32 — matches the on-device std430 layout).
// env-off: returns bin.tile_ranges directly.
// env-on : stage-downloads bin.tile_ranges_gpu into host_buf, returns ptr.
inline const uint32_t* tile_ranges_host_view(
    VulkanContext& ctx,
    const BinningOutput& bin,
    std::vector<uint32_t>& host_buf) {
    if (bin.tile_ranges != nullptr) {
        return bin.tile_ranges;
    }
    if (bin.tile_ranges_gpu == nullptr) {
        throw std::runtime_error(
            "test_helpers::tile_ranges_host_view: both host and GPU carriers "
            "are null");
    }
    const std::size_t pairs = static_cast<std::size_t>(bin.num_tiles) * 2u;
    host_buf.assign(pairs, 0u);
    detail::stage_download(ctx,
                           static_cast<VkBuffer>(bin.tile_ranges_gpu),
                           pairs * sizeof(uint32_t),
                           host_buf.data());
    return host_buf.data();
}

}  // namespace test_helpers
