// sorter_vulkan.h -- SP-2 T14/T19: Sorter adapter driving RadixSortPass +
// TileRangePass.
//
// Supports two usage patterns:
//
//   1) Layer-1 sort()          — sync, self-contained. Allocates / uploads /
//                                 dispatches / downloads within one call.
//                                 Reads binning.keys_unsorted / values_unsorted.
//   2) Layer-2 prepare_record() + record() — chained forward pipeline (T19).
//      Input handles (keys_unsorted / values_unsorted) come from
//      TileBinnerVulkan's record-mode outputs; SorterVulkan allocates its own
//      ping-pong "B" buffers, histogram scratch, and tile_ranges buffer.

#pragma once

#include "sorter.h"
#include "vulkan/vk_context.h"
#include "vulkan/sort_passes.h"

#include <memory>

class VulkanBuffer;
class VulkanAlignedBuffer;
class RadixSortFuchsia;
class CanonicalSortPairPackPass;
class CanonicalSortPairExtractPass;

class SorterVulkan : public Sorter {
public:
    explicit SorterVulkan(VulkanContext& ctx);
    ~SorterVulkan() override;

    SorterVulkan(const SorterVulkan&)            = delete;
    SorterVulkan& operator=(const SorterVulkan&) = delete;

    /// Sort binning.keys_unsorted / values_unsorted (R elements) into
    /// binning.keys_sorted / values_sorted, then populate binning.tile_ranges
    /// from the sorted keys. All output arrays are allocated from `allocator`.
    void sort(BinningOutput& binning, FrameAllocator& allocator) override;

    void set_host_mirror_enabled(bool enabled) { host_mirror_enabled_ = enabled; }

    // -- Layer-2 record-mode API -------------------------------------------
    //
    // Allocate ping-pong "B" buffers, histogram scratch, and tile_ranges.
    // keys_unsorted / values_unsorted are EXTERNAL handles from the binner:
    // they serve as the "A" side of the 16-pass ping-pong radix sort. After
    // 16 (even) passes, the sorted data lives in the A side again, so
    // keys_sorted_buf() returns the same handle the caller passed in.
    //
    // tile_ranges is zero-initialised here (precondition of tile_range.comp).
    // Buffers persist until the next prepare_record() call or destruction.
    void prepare_record(uint32_t R, uint32_t num_tiles,
                        VkBuffer keys_unsorted,
                        VkBuffer values_unsorted);

    void prepare_record_fuchsia_sortpairs(uint32_t R, uint32_t num_tiles,
                                          VkBuffer keys_unsorted,
                                          VkBuffer values_unsorted);

    // Record radix sort + tile_range sweep into cmd. prepare_record() must
    // have been called. Inserts a compute barrier between the sort and the
    // tile_range sweep (tile_range.comp reads keys_sorted).
    void record(VkCommandBuffer cmd, uint32_t R, uint32_t num_tiles);

    void record_fuchsia_sortpairs(VkCommandBuffer cmd,
                                  uint32_t R,
                                  uint32_t num_tiles);

    // Output handles valid after prepare_record(). Caller reads these to
    // thread into RasterizerVulkan::prepare_record(). The project radix path
    // returns the external A-side handles; the recorded Fuchsia SortPairs path
    // returns owned canonical output buffers.
    VkBuffer keys_sorted_buf()   const;
    VkBuffer values_sorted_buf() const;
    VkBuffer tile_ranges_buf()   const;

    static bool fuchsia_sort_enabled();

private:
    VulkanContext& ctx_;
    std::unique_ptr<RadixSortPass> sort_pass_;
    std::unique_ptr<TileRangePass> range_pass_;
    std::unique_ptr<RadixSortFuchsia> fuchsia_pairs_;
    std::unique_ptr<CanonicalSortPairPackPass> pair_pack_pass_;
    std::unique_ptr<CanonicalSortPairExtractPass> pair_extract_pass_;

    // Layer-1 optimized canonical SortPairs buffers.
    std::unique_ptr<VulkanAlignedBuffer> f_records_even_;
    std::unique_ptr<VulkanAlignedBuffer> f_records_odd_;
    std::unique_ptr<VulkanAlignedBuffer> f_internal_scratch_;
    std::unique_ptr<VulkanBuffer> f_keys_sorted_;
    std::unique_ptr<VulkanBuffer> f_values_sorted_;
    std::unique_ptr<VulkanBuffer> f_ranges_;
    uint32_t f_record_capacity_ = 0;
    uint32_t f_output_capacity_ = 0;
    uint32_t f_range_capacity_ = 0;
    VkDeviceSize f_record_bytes_capacity_ = 0;
    VkDeviceSize f_internal_bytes_capacity_ = 0;
    bool host_mirror_enabled_ = true;

    bool sort_via_fuchsia_sortpairs(BinningOutput& binning, FrameAllocator& allocator);

    bool r_fuchsia_sortpairs_record_ = false;
    VkBuffer r_f_keys_input_ = VK_NULL_HANDLE;
    VkBuffer r_f_values_input_ = VK_NULL_HANDLE;

    // Layer-2 persistent buffers. "A" handles are external (not owned);
    // "B" + histograms + tile_ranges are owned by this adapter.
    VkBuffer r_keys_a_ = VK_NULL_HANDLE;
    VkBuffer r_vals_a_ = VK_NULL_HANDLE;
    std::unique_ptr<VulkanBuffer> r_keys_b_;
    std::unique_ptr<VulkanBuffer> r_vals_b_;
    std::unique_ptr<VulkanBuffer> r_hist_cnt_;
    std::unique_ptr<VulkanBuffer> r_hist_scn_;
    std::unique_ptr<VulkanBuffer> r_wg_sums_;
    std::unique_ptr<VulkanBuffer> r_wg_sums2_;
    std::unique_ptr<VulkanBuffer> r_ranges_;
};
