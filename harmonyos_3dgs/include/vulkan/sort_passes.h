// sort_passes.h -- SP-2 T14: thin owners of the two radix-sort compute pipelines
// and the tile_range compute pipeline that together implement the sort half of
// the Vulkan forward pass.
//
//   * RadixSortPass wraps radix_sort_count.comp (spec §4.8.4) and
//     radix_sort_scatter.comp (spec §4.8.5), orchestrating 16 LSD passes of
//     (count → prefix-scan → scatter) ping-pong over uint64 keys + uint32
//     values. A private PrefixScanPass is used for the per-pass 16-element
//     scan of the count histogram.
//
//   * TileRangePass wraps tile_range.comp (spec §4.8.6), emitting one
//     (start, end) pair per tile from a sorted stream of uint64 keys.
//
// Both classes mirror the PrefixScanPass / ScatterPass (T10) design:
//   - Allocate one descriptor set in the constructor via
//     allocate_empty_descriptor_set() to avoid exhausting the 4-set pool
//     if bind_buffers() is called per-frame.
//   - Expose a Layer-1 "sync dispatch" entry point for bring-up/tests. (A
//     Layer-2 record() entry point is not provided here: SP-2 Phase 1 uses the
//     sync API end-to-end; the chained forward pipeline is T19 scope.)
//
// Neither class owns the underlying SSBOs — callers pass raw VkBuffer handles,
// and those buffers must remain alive for the duration of the dispatch.
//
// Single-workgroup constraint (SP-2 Phase 1): the radix shaders dispatch
// exactly one workgroup with local_size_x=256, so num_elements must be <= 256.
// This matches the tiny fixture (R = 103) and is enforced at the API boundary.

#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_shader.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/tile_binner_passes.h"  // PrefixScanPass (used internally)
#include "vulkan/preprocess_bindings.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

// ---------------------------------------------------------------------------
// RadixSortPass -- LSD 4-bit radix sort over uint64 keys + uint32 values.
//
// The sort runs 16 passes (current_bit = 0, 4, 8, ..., 60). Each pass is:
//   count  :  radix_sort_count.comp      — per-WG 16-bucket histogram
//   scan   :  prefix_sum.comp (exclusive) over the 16 histogram entries
//   scatter:  radix_sort_scatter.comp    — stable scatter using bucket offsets
// Two uint64/uint32 buffers A/B are ping-ponged between passes. Because 16
// passes is even, the sorted result ends up in the buffer passed as
// `keys_in_buf` / `values_in_buf` (the "A" side).
// ---------------------------------------------------------------------------
class RadixSortPass {
public:
    explicit RadixSortPass(VulkanContext& ctx);
    ~RadixSortPass() = default;

    RadixSortPass(const RadixSortPass&)            = delete;
    RadixSortPass& operator=(const RadixSortPass&) = delete;

    /// Layer 1: synchronous full-radix sort.
    ///
    /// On entry: `keys_in_buf` / `values_in_buf` hold the unsorted data (R
    /// elements). `keys_out_buf` / `values_out_buf` are scratch buffers of
    /// identical size used for ping-pong between passes.
    /// On exit: the sorted data lives in `keys_in_buf` / `values_in_buf`
    /// (16 passes = even number of swaps).
    ///
    /// `hist_count_buf` : uint[16] scratch (count-shader output)
    /// `hist_scan_buf`  : uint[16] scratch (scan-shader output, used as
    ///                    bucket_offsets by the scatter shader)
    /// `wg_sums_buf`    : uint[>=1] scratch for the inner prefix scan
    ///
    /// Constraints: 0 < R <= 256 (single-workgroup radix).
    void sort_sync(VkBuffer keys_in_buf, VkBuffer values_in_buf,
                   VkBuffer keys_out_buf, VkBuffer values_out_buf,
                   VkBuffer hist_count_buf, VkBuffer hist_scan_buf,
                   VkBuffer wg_sums_buf,
                   uint32_t num_elements);

private:
    VulkanContext& ctx_;

    std::unique_ptr<VulkanShader>          count_shader_;
    std::unique_ptr<VulkanShader>          scatter_shader_;
    std::unique_ptr<VulkanComputePipeline> count_pipeline_;
    std::unique_ptr<VulkanComputePipeline> scatter_pipeline_;
    std::unique_ptr<PrefixScanPass>        scan_pass_;
    VkDescriptorSet count_ds_   = VK_NULL_HANDLE;
    VkDescriptorSet scatter_ds_ = VK_NULL_HANDLE;
};

// ---------------------------------------------------------------------------
// TileRangePass -- emits tile_ranges[num_tiles*2] from sorted keys.
// ---------------------------------------------------------------------------
class TileRangePass {
public:
    explicit TileRangePass(VulkanContext& ctx);
    ~TileRangePass() = default;

    TileRangePass(const TileRangePass&)            = delete;
    TileRangePass& operator=(const TileRangePass&) = delete;

    /// Rewire the descriptor set. Safe to call per-frame.
    void bind_buffers(VkBuffer keys_sorted, VkBuffer tile_ranges);

    /// Layer 1: synchronous dispatch. Caller must zero-initialise
    /// `tile_ranges` before calling — tile_range.comp only writes tiles that
    /// contain at least one sorted key, and relies on the pre-zero to leave
    /// empty tiles as [0, 0).
    void dispatch_sync(uint32_t num_elements, uint32_t num_tiles);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader>          shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    VkDescriptorSet                        descriptor_set_ = VK_NULL_HANDLE;
};
