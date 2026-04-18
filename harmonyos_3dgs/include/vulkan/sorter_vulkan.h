// sorter_vulkan.h -- SP-2 T14: Sorter adapter that drives RadixSortPass +
// TileRangePass over the unsorted (key, value) pairs emitted by
// TileBinnerVulkan. Mirrors the shape of SorterCPU / SorterGPU as an
// implementor of the `Sorter` virtual interface.
//
// Phase-1 contract:
//   * Input:  binning.keys_unsorted[R], binning.values_unsorted[R],
//             binning.num_tiles (set by TileBinnerVulkan).
//   * Output: binning.keys_sorted[R], binning.values_sorted[R],
//             binning.tile_ranges[num_tiles*2] — all allocated from the
//             caller-provided FrameAllocator.
//
// Buffer lifecycle: fresh host-visible SSBOs are allocated per call for
// ping-pong and scratch; released at end-of-call. Matches the Phase-1 style
// of TileBinnerVulkan::bin() — no pooling yet.

#pragma once

#include "sorter.h"
#include "vulkan/vk_context.h"
#include "vulkan/sort_passes.h"

#include <memory>

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

private:
    VulkanContext& ctx_;
    std::unique_ptr<RadixSortPass> sort_pass_;
    std::unique_ptr<TileRangePass> range_pass_;
};
