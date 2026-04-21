// tile_binner_vulkan.h -- SP-2 T10/T19: Vulkan TileBinner adapter.
//
// Implements the TileBinner interface using PrefixScanPass + ScatterPass.
// Supports two usage patterns:
//
//   1) Layer-1 bin()          — sync, self-contained. Allocates / uploads /
//                                dispatches / downloads within one call.
//   2) Layer-2 prepare_record() + record() — chained forward pipeline (T19).
//      Inputs (tiles_touched / means2D / depths / radii) are EXTERNAL handles
//      already produced by PreprocessorVulkan::record(); this adapter allocates
//      its own point_offsets, wg_sums, keys_unsorted, values_unsorted buffers
//      and binds both passes. record() records the scan+scatter dispatches.
//
// BinningOutput fields populated by bin() (Layer-1 path):
//   total_pairs, keys_unsorted, values_unsorted, num_tiles.
// Layer-2 path does not produce a BinningOutput — callers read the handles
// via keys_unsorted_buf() / values_unsorted_buf().

#pragma once

#include "tile_binner.h"
#include "vulkan/vk_context.h"
#include "vulkan/tile_binner_passes.h"

#include <memory>

class VulkanBuffer;

class TileBinnerVulkan : public TileBinner {
public:
    explicit TileBinnerVulkan(VulkanContext& ctx);
    ~TileBinnerVulkan() override;

    TileBinnerVulkan(const TileBinnerVulkan&)            = delete;
    TileBinnerVulkan& operator=(const TileBinnerVulkan&) = delete;

    BinningOutput bin(const PreprocessOutput& preprocess,
                      int num_gaussians,
                      const Camera& camera,
                      const RenderConfig& config,
                      FrameAllocator& allocator) override;

    // -- Layer-2 record-mode API -------------------------------------------
    //
    // Allocate internal GPU buffers for the chained forward pipeline and bind
    // both passes (scan + scatter). External input handles come from
    // PreprocessorVulkan's buffer getters. R_max is an upper bound on the
    // number of (Gaussian, tile) pairs; the actual R is not known until after
    // scan runs, but the keys_unsorted / values_unsorted buffers must be
    // pre-allocated before we record the scatter. Callers that know R from
    // golden data can pass R exactly; otherwise N * num_tiles_x * num_tiles_y
    // is a safe over-estimate.
    //
    // Buffers persist until the next prepare_record() call or destruction.
    void prepare_record(uint32_t N, uint32_t R_max,
                        uint32_t num_tiles_x, uint32_t num_tiles_y,
                        VkBuffer tiles_touched,
                        VkBuffer means2D,
                        VkBuffer depths,
                        VkBuffer radii,
                        VkBuffer radius_f,
                        VkBuffer cov3D_inv,
                        VkBuffer mean_offset,
                        VkBuffer gauss2screen,
                        bool eval_3D,
                        const Camera& cam);

    // Record prefix-scan + scatter into cmd. prepare_record() must have been
    // called. Inserts an internal barrier between scan and scatter (scatter
    // reads scan's point_offsets output).
    void record(VkCommandBuffer cmd,
                uint32_t N, uint32_t num_tiles_x, uint32_t num_tiles_y);

    // Output handles valid after prepare_record(). Used by SorterVulkan::
    // prepare_record() to thread the chain. Return VK_NULL_HANDLE before
    // the first prepare_record() call.
    VkBuffer keys_unsorted_buf()   const;
    VkBuffer values_unsorted_buf() const;

private:
    VulkanContext& ctx_;
    std::unique_ptr<PrefixScanPass> scan_pass_;
    std::unique_ptr<ScatterPass>    scatter_pass_;

    // Layer-2 persistent buffers (owned by this adapter).
    std::unique_ptr<VulkanBuffer> r_po_buf_;      // point_offsets [N]
    std::unique_ptr<VulkanBuffer> r_ws_buf_;      // workgroup sums [ceil(N/256)]
    std::unique_ptr<VulkanBuffer> r_ws2_buf_;     // level-2 wg sums [ceil(ceil(N/256)/256)]
    std::unique_ptr<VulkanBuffer> r_keys_buf_;    // keys_unsorted [R_max] u64
    std::unique_ptr<VulkanBuffer> r_vals_buf_;    // values_unsorted [R_max] u32
    std::unique_ptr<VulkanBuffer> r_scatter_ubo_; // ScatterUBO (eval_3D inverse_vp + cam_pos + img_size)
    std::unique_ptr<VulkanBuffer> r_dummy4_;      // 4-byte dummy for unbound eval_3D SSBOs
    bool r_eval_3D_ = false;                      // stashed for record()

    // --- Layer-1 persistent buffers (bin() reuses across calls) ---
    // Scan buffers sized to max N seen; scatter buffers sized to max ACTUAL R
    // seen (NOT N*num_tiles worst-case). prepare_for_scatter() is called after
    // the scan gives us actual R, so we never over-allocate by 100-1000x.
    uint32_t bin_N_     = 0u;  // Gaussian count at last scan alloc
    uint64_t bin_R_max_ = 0u;  // actual scatter buffer capacity (grow-only)
    uint32_t bin_wg_    = 0u;  // workgroup count = ceil(N/256) at last alloc

    std::unique_ptr<VulkanBuffer> bin_tt_buf_;    // tiles_touched [N] i32
    std::unique_ptr<VulkanBuffer> bin_po_buf_;    // point_offsets [N] u32
    std::unique_ptr<VulkanBuffer> bin_ws_buf_;    // workgroup_sums [ceil(N/256)] u32
    std::unique_ptr<VulkanBuffer> bin_ws2_buf_;   // level-2 wg sums [ceil(ceil(N/256)/256)] u32
    std::unique_ptr<VulkanBuffer> bin_m2d_buf_;   // means2D [N*2] f32
    std::unique_ptr<VulkanBuffer> bin_dep_buf_;   // depths [N] f32
    std::unique_ptr<VulkanBuffer> bin_rad_buf_;   // radii [N] i32
    std::unique_ptr<VulkanBuffer> bin_rf_buf_;    // radius_f [N*2] f32 (extent_x, extent_y)
    std::unique_ptr<VulkanBuffer> bin_keys_buf_;  // keys_unsorted [R] u64
    std::unique_ptr<VulkanBuffer> bin_vals_buf_;  // values_unsorted [R] u32

    // Allocate/grow the 7 N-sized + 1 wg-sized scan buffers.
    void prepare_for_bin(uint32_t N, uint32_t num_wgs);
    // Allocate/grow scatter output buffers to at least R entries.
    // Called after scan gives us actual R — never over-allocates to N*num_tiles.
    void prepare_for_scatter(uint64_t R);
};
