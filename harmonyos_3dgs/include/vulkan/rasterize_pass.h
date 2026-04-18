// rasterize_pass.h -- SP-2 T17: thin owner of the rasterize.comp compute
// pipeline (spec §4.8.7). Wraps the shader module, the compute pipeline
// (with the 9-binding mixed SSBO+UBO layout), and one descriptor set.
//
// The rasterize shader uses a 16x16 workgroup where one workgroup = one
// 16x16 output tile. dispatch_sync() takes the tile grid dimensions and
// issues (num_tiles_x, num_tiles_y, 1) workgroups.
//
// Bindings 0..7 are SSBOs (5 read-only inputs, 3 write-only outputs).
// Binding 8 is the RasterizeUBO (background colour, 16 bytes std140).
//
// Like PreprocessPass / ScatterPass, RasterizePass allocates its descriptor
// set once in the constructor and updates bindings in place in bind_buffers()
// so the 4-set pool is never exhausted by per-frame rebinding.
#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_shader.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/preprocess_bindings.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

class RasterizePass {
public:
    /// All buffers wired to bindings 0..8. Names mirror rasterize_bind::.
    struct Buffers {
        VkBuffer values_sorted;         // RO uint[R]
        VkBuffer tile_ranges;           // RO uint[num_tiles*2] (flat pairs)
        VkBuffer means2D;               // RO float[N*2]
        VkBuffer conic_opacity_packed;  // RO float[N*4]  {a, b, c, opacity}
        VkBuffer rgb;                   // RO float[N*3]
        VkBuffer out_image;             // WO float[3*H*W]  CHW
        VkBuffer transmittance;         // WO float[H*W]
        VkBuffer n_contrib;             // WO uint[H*W]
        VkBuffer raster_ubo;            // UB  RasterizeUBO (16 bytes)
    };

    explicit RasterizePass(VulkanContext& ctx);
    ~RasterizePass() = default;

    RasterizePass(const RasterizePass&)            = delete;
    RasterizePass& operator=(const RasterizePass&) = delete;

    /// Rewire the descriptor set to these buffers. Safe to call per-frame —
    /// updates the pre-allocated set in place.
    void bind_buffers(const Buffers& b);

    /// Layer 1: synchronous dispatch. Grid = (num_tiles_x, num_tiles_y, 1).
    /// Each workgroup is 16x16 and renders one 16x16 tile.
    void dispatch_sync(uint32_t num_gaussians,
                       uint32_t image_width, uint32_t image_height,
                       uint32_t num_tiles_x, uint32_t num_tiles_y);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader>          shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    VkDescriptorSet                        descriptor_set_ = VK_NULL_HANDLE;
};
