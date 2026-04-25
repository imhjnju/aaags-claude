// rasterize_pass.h -- SP-2 T17: thin owner of the rasterize.comp compute
// pipeline (spec §4.8.7). Wraps the shader module, the compute pipeline
// (with 14 bindings: 12 SSBO + 2 UBO), and one descriptor set.
//
// The rasterize shader uses a 16x16 workgroup where one workgroup = one
// 16x16 output tile. dispatch_sync() takes the tile grid dimensions and
// issues (num_tiles_x, num_tiles_y, 1) workgroups.
//
// Bindings 0..7 are SSBOs (5 read-only inputs, 3 write-only outputs).
// Binding 8 is the RasterizeUBO (background colour, 16 bytes std140).
// Bindings 9..13 are eval_3D only (gauss2screen, opacities, cov3D_inv,
// mean_offset, RasterEval3DUBO).
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
    /// All buffers wired to bindings 0..13. Names mirror rasterize_bind::.
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
        VkBuffer gauss2screen;          // RO float[N*16]  (binding 9, eval_3D only)
        VkBuffer opacities_2d;          // RO float[N]     (binding 10, eval_3D only)
        VkBuffer cov3D_inv;             // RO float[N*6]   (binding 11, eval_3D only)
        VkBuffer mean_offset;           // RO float[N*3]   (binding 12, eval_3D only)
        VkBuffer raster_eval3d_ubo;     // UB  96 bytes    (binding 13, eval_3D only)
    };

    /// Cascade trace buffers — only consumed when the pass was constructed
    /// with `spec_trace_enabled=1`. Bindings 14..30 follow the layout in
    /// dev_notes/phase4_vk_cascade_port_plan.md §5.2 and must match
    /// rasterize_trace_bind:: in preprocess_bindings.h.
    struct TraceBuffers {
        VkBuffer trace_meta_ubo;    // UB 32 B TraceMetaUBO (binding 14)
        VkBuffer slot_lookup;       // SSBO int[num_tiles]           (15)
        VkBuffer tail_depths;       // SSBO float[K*512*16*64]       (16)
        VkBuffer tail_ids;          // SSBO int  [K*512*16*64]       (17)
        VkBuffer tail_wcur;         // SSBO uint [K]                 (18)
        VkBuffer mid_depths;        // SSBO float[K*1024*16*4*8]     (19)
        VkBuffer mid_ids;           // SSBO int  [K*1024*16*4*8]     (20)
        VkBuffer mid_wcur;          // SSBO uint [K]                 (21)
        VkBuffer head_ins_depth;    // SSBO float[K*256*4096]        (22)
        VkBuffer head_ins_alpha;    // SSBO float[K*256*4096]        (23)
        VkBuffer head_ins_gid;      // SSBO int  [K*256*4096]        (24)
        VkBuffer head_ins_cursor;   // SSBO uint [K*256]             (25)
        VkBuffer head_blend_depth;  // SSBO float[K*256*4096]        (26)
        VkBuffer head_blend_alpha;  // SSBO float[K*256*4096]        (27)
        VkBuffer head_blend_T;      // SSBO float[K*256*4096]        (28)
        VkBuffer head_blend_gid;    // SSBO int  [K*256*4096]        (29)
        VkBuffer head_blend_cursor; // SSBO uint [K*256]             (30)
    };

    /// Construct a pass with both specialization constants fixed at build-time.
    /// `spec_eval_3D`       : 0 = 2D conic path, 1 = eval_3D k-buffer path.
    /// `spec_trace_enabled` : 0 = trace OFF (14-binding DSL), 1 = trace ON
    ///                        (31-binding DSL; caller must also supply
    ///                        TraceBuffers via bind_trace_buffers()).
    explicit RasterizePass(VulkanContext& ctx,
                           uint32_t spec_eval_3D = 0u,
                           uint32_t spec_trace_enabled = 0u);
    ~RasterizePass() = default;

    RasterizePass(const RasterizePass&)            = delete;
    RasterizePass& operator=(const RasterizePass&) = delete;

    /// Rewire the descriptor set to these buffers. Safe to call per-frame —
    /// updates the pre-allocated set in place.
    void bind_buffers(const Buffers& b);

    /// Rewire the descriptor set's cascade-trace bindings (14..30). Only
    /// valid when constructed with spec_trace_enabled=1 — throws otherwise.
    void bind_trace_buffers(const TraceBuffers& tb);

    /// Whether this pass was built with cascade trace plumbing enabled.
    bool trace_enabled() const { return trace_enabled_; }

    /// Layer 1: synchronous dispatch. Grid = (num_tiles_x, num_tiles_y, 1).
    /// Each workgroup is 16x16 and renders one 16x16 tile.
    void dispatch_sync(uint32_t num_gaussians,
                       uint32_t image_width, uint32_t image_height,
                       uint32_t num_tiles_x, uint32_t num_tiles_y);

    /// Layer 2: record dispatch into an external command buffer.
    /// bind_buffers() must have been called. No internal barrier — the
    /// caller inserts a barrier BEFORE (producers → SSBO inputs) and AFTER
    /// (consumer reading out_image / T_final / n_contrib) if needed.
    void record(VkCommandBuffer cmd,
                uint32_t num_gaussians,
                uint32_t image_width, uint32_t image_height,
                uint32_t num_tiles_x, uint32_t num_tiles_y);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader>          shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    VkDescriptorSet                        descriptor_set_ = VK_NULL_HANDLE;
    bool                                   trace_enabled_ = false;
};
