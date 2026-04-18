// tile_binner_vulkan.h -- SP-2 T10: Vulkan TileBinner adapter.
//
// Implements the TileBinner interface using PrefixScanPass + ScatterPass.
// Phase 1 sync path: upload preprocess.tiles_touched + related arrays to
// Vulkan SSBOs, run exclusive prefix scan into point_offsets, scatter into
// (key, value) pairs, and download to host-visible BinningOutput fields.
//
// BinningOutput fields populated here:
//   total_pairs, keys_unsorted, values_unsorted, num_tiles
// Left null for downstream SorterVulkan (T14):
//   keys_sorted, values_sorted, tile_ranges

#pragma once

#include "tile_binner.h"
#include "vulkan/vk_context.h"
#include "vulkan/tile_binner_passes.h"

#include <memory>

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

private:
    VulkanContext& ctx_;
    std::unique_ptr<PrefixScanPass> scan_pass_;
    std::unique_ptr<ScatterPass>    scatter_pass_;
};
