// rasterize_backward_pass.h — SP-3 T22: thin owner of the
// rasterize_backward.comp compute pipeline.
//
// One workgroup = one 16x16 tile. The shader traverses the sorted Gaussian
// list back-to-front and accumulates gradients via atomicAdd into the output
// SSBOs. dispatch_sync() dispatches ceil(W/16) × ceil(H/16) × 1 workgroups.
//
// Bindings match rasterize_backward_bind:: in backward_bindings.h:
//   0..7   read-only SSBOs  (tile_ranges, values_sorted, means2D,
//                             conic_opacity_packed, colors, T_final,
//                             n_contrib, dL_dpixels)
//   8..11  read-write SSBOs (dL_dmeans2D, dL_dconics, dL_dopacity, dL_dcolors)
//   12     UBO              (RasterizeBackwardUBO)
//
// The caller is responsible for zeroing the gradient SSBOs before calling
// bind_buffers() / dispatch_sync(). VulkanBuffer does not zero on allocation.
#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_shader.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/backward_bindings.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

class RasterizeBackwardPass {
public:
    /// All VkBuffers wired to bindings 0..12. Names mirror rasterize_backward_bind::.
    struct Buffers {
        VkBuffer tile_ranges;     // RO uint[num_tiles*2]
        VkBuffer values_sorted;   // RO uint[R]
        VkBuffer means2D;         // RO float[N*2]
        VkBuffer conic_opacity;   // RO float[N*4]  {a, b, c, opacity}
        VkBuffer colors;          // RO float[N*3]
        VkBuffer T_final;         // RO float[H*W]
        VkBuffer n_contrib;       // RO uint[H*W]
        VkBuffer dL_dpixels;      // RO float[H*W*3]  pixel-major
        VkBuffer dL_dmeans2D;     // RW float[N*2]    zero-filled by caller
        VkBuffer dL_dconics;      // RW float[N*3]    zero-filled by caller
        VkBuffer dL_dopacity;     // RW float[N]      zero-filled by caller
        VkBuffer dL_dcolors;      // RW float[N*3]    zero-filled by caller
    };

    explicit RasterizeBackwardPass(VulkanContext& ctx);
    ~RasterizeBackwardPass();  // out-of-line: unique_ptr to incomplete type

    RasterizeBackwardPass(const RasterizeBackwardPass&)            = delete;
    RasterizeBackwardPass& operator=(const RasterizeBackwardPass&) = delete;

    /// Wire the descriptor set to these buffers. Safe to call once per scene.
    /// ubo must point to an uploaded RasterizeBackwardUBO (32 bytes, std140).
    void bind_buffers(const Buffers& b, VkBuffer ubo);

    /// Synchronous dispatch: num_tiles_x × num_tiles_y × 1 workgroups.
    /// bind_buffers() must have been called first.
    void dispatch_sync(uint32_t num_tiles_x, uint32_t num_tiles_y);

    /// Record dispatch into cmd without submitting. bind_buffers() must be called first.
    void record(VkCommandBuffer cmd, uint32_t num_tiles_x, uint32_t num_tiles_y);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader>          shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    VkDescriptorSet                        descriptor_set_ = VK_NULL_HANDLE;
};
