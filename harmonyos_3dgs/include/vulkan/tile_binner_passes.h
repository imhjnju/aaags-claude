// tile_binner_passes.h -- SP-2 T10: thin owners of the two compute pipelines
// that together implement the tile-binning half of the Vulkan forward pass.
//
//   * PrefixScanPass wraps prefix_sum.comp (spec §4.8.2), orchestrating the
//     3-phase Blelloch scan (phase 0 local → phase 1 workgroup-sum scan →
//     phase 2 add-back) with compute-to-compute barriers between phases.
//   * ScatterPass   wraps scatter.comp   (spec §4.8.3), emitting one
//     (uint64 sort-key, uint32 Gaussian-index) pair per (Gaussian × tile)
//     overlap.
//
// Both classes mirror the PreprocessPass (T5) design:
//   - Allocate one descriptor set in the constructor via
//     allocate_empty_descriptor_set() to avoid exhausting the 4-set pool
//     if bind_buffers() is called per-frame.
//   - Expose a Layer-1 "sync dispatch" entry point for bring-up/tests, and a
//     Layer-2 "record into external cmd buffer" entry point for the chained
//     forward pipeline (preprocess → scan → scatter → sort → rasterize).
//
// Neither class owns the underlying SSBOs — callers pass raw VkBuffer handles,
// and those buffers must remain alive for the duration of the dispatch.

#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_shader.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/preprocess_bindings.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

// ---------------------------------------------------------------------------
// PrefixScanPass -- 3-phase Blelloch exclusive scan on a uint[] input array.
// ---------------------------------------------------------------------------
class PrefixScanPass {
public:
    explicit PrefixScanPass(VulkanContext& ctx);
    ~PrefixScanPass() = default;

    PrefixScanPass(const PrefixScanPass&)            = delete;
    PrefixScanPass& operator=(const PrefixScanPass&) = delete;

    /// Rewire the descriptor set to the given buffer handles. Safe to call
    /// per-frame — updates in place rather than allocating a new set.
    /// `workgroup_sums` must hold at least `ceil(num_elements / 256)` uints.
    /// This single-buffer overload supports up to 65536 elements.
    void bind_buffers(VkBuffer input_array,
                      VkBuffer output_array,
                      VkBuffer workgroup_sums);

    /// Extended bind for 2-level scan (N up to ~16M).
    /// `workgroup_sums`  must hold at least `ceil(N / 256)` uints.
    /// `workgroup_sums2` must hold at least `ceil(ceil(N/256) / 256)` uints.
    /// After calling this, scan_sync / record will use the 2-level path
    /// automatically when num_elements > 65536.
    void bind_buffers_2level(VkBuffer input_array,
                             VkBuffer output_array,
                             VkBuffer workgroup_sums,
                             VkBuffer workgroup_sums2);

    /// Layer 1: synchronous exclusive scan of `num_elements` uints.
    /// Allocates a transient command buffer, records all 3 phases with
    /// compute barriers between them, submits, and vkQueueWaitIdles.
    void scan_sync(uint32_t num_elements);

    /// Layer 2: record the 3-phase scan into an external command buffer.
    /// Caller owns begin/end/submit and any surrounding barriers (but the
    /// two barriers BETWEEN phases are inserted here — they are internal
    /// dependencies of the scan algorithm).
    void record(VkCommandBuffer cmd, uint32_t num_elements);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader>           shader_;
    std::unique_ptr<VulkanComputePipeline>  pipeline_;
    VkDescriptorSet                         descriptor_set_  = VK_NULL_HANDLE;
    // Second descriptor set for 2-level scan: bound to {wg_sums1, wg_sums1, wg_sums2}.
    VkDescriptorSet                         descriptor_set2_ = VK_NULL_HANDLE;
    bool                                    has_2level_      = false;

    // Dispatch one of phases 0/1/2 into `cmd`. `num_wgs` is the grid size in
    // x (1 for phase 1).
    void dispatch_phase(VkCommandBuffer cmd,
                        uint32_t num_elements,
                        uint32_t phase,
                        uint32_t num_wgs,
                        VkDescriptorSet ds = VK_NULL_HANDLE);
};

// ---------------------------------------------------------------------------
// ScatterPass -- duplicateWithKeys: one thread per Gaussian emits
// tiles_touched[i] (key, value) pairs starting at point_offsets[i].
// ---------------------------------------------------------------------------
class ScatterPass {
public:
    /// All bindings mirror scatter_bind:: (spec §4.8.3).
    struct Buffers {
        VkBuffer means2D;         // RO float[N*2]
        VkBuffer depths;          // RO float[N]
        VkBuffer radii;           // RO int[N]
        VkBuffer point_offsets;   // RO uint[N]  — exclusive prefix
        VkBuffer tiles_touched;   // RO int[N]
        VkBuffer keys_unsorted;   // WO uint64[R]
        VkBuffer values_unsorted; // WO uint[R]
        VkBuffer radius_f;        // RO float[N]  — float eigenvalue radius from preprocess
        VkBuffer cov3D_inv;       // RO float[N*6]   (binding 8, eval_3D only)
        VkBuffer mean_offset;     // RO float[N*3]   (binding 9, eval_3D only)
        VkBuffer scatter_ubo;     // UB ScatterUBO   (binding 10, eval_3D only)
    };

    explicit ScatterPass(VulkanContext& ctx);
    ~ScatterPass() = default;

    ScatterPass(const ScatterPass&)            = delete;
    ScatterPass& operator=(const ScatterPass&) = delete;

    void bind_buffers(const Buffers& b);

    /// Layer 1: sync dispatch of scatter.comp over `num_gaussians` threads.
    /// Tile grid parameters are required because scatter recomputes rect.
    void dispatch_sync(uint32_t num_gaussians,
                       uint32_t num_tiles_x,
                       uint32_t num_tiles_y);

    /// Layer 2: record into an external cmd buffer.
    void record(VkCommandBuffer cmd,
                uint32_t num_gaussians,
                uint32_t num_tiles_x,
                uint32_t num_tiles_y);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader>           shader_;
    std::unique_ptr<VulkanComputePipeline>  pipeline_;
    VkDescriptorSet                         descriptor_set_ = VK_NULL_HANDLE;
};
