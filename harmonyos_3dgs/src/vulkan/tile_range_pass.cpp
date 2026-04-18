// SP-2 T14: TileRangePass — wraps tile_range.comp (spec §4.8.6).
//
// Sweeps sorted keys and writes per-tile (start, end) ranges. The shader
// precondition (tile_ranges zero-initialised before dispatch) is the
// caller's responsibility — empty tiles are never written and must read back
// as [0, 0).

#include "vulkan/sort_passes.h"

// xxd-embedded SPIR-V (CMake build dir produces tile_range_spv.h).
#include "tile_range_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
// tile_range.comp uses local_size_x=256 — keep in one place to avoid drift
// between shader and host dispatch arithmetic.
constexpr uint32_t kTileRangeLocalSize = 256;
}  // namespace

TileRangePass::TileRangePass(VulkanContext& ctx)
    : ctx_(ctx) {
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(tile_range_spv),
        static_cast<std::size_t>(tile_range_spv_len));

    // 2 SSBOs: keys_sorted (RO), tile_ranges (RW). Both storage buffers.
    std::vector<VkDescriptorType> binding_types(2,
                                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_,
        *shader_,
        binding_types,
        /*push_constant_bytes=*/sizeof(TileRangePushConstants),
        /*max_descriptor_sets=*/4);

    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

void TileRangePass::bind_buffers(VkBuffer keys_sorted,
                                 VkBuffer tile_ranges) {
    pipeline_->update_ssbo(descriptor_set_, tile_range_bind::KEYS_SORTED, keys_sorted);
    pipeline_->update_ssbo(descriptor_set_, tile_range_bind::TILE_RANGES, tile_ranges);
}

void TileRangePass::dispatch_sync(uint32_t num_elements, uint32_t num_tiles) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "TileRangePass::dispatch_sync called before bind_buffers()");
    if (num_elements == 0u) return;  // nothing to sweep; ranges remain as-is.
    if (num_tiles == 0u)
        throw std::runtime_error("TileRangePass: num_tiles must be > 0 when num_elements > 0");

    TileRangePushConstants pc{};
    pc.num_elements = num_elements;
    pc.num_tiles    = num_tiles;
    pc._pad0        = 0u;
    pc._pad1        = 0u;

    const uint32_t groups =
        (num_elements + kTileRangeLocalSize - 1u) / kTileRangeLocalSize;
    pipeline_->dispatch_sync(descriptor_set_,
                             groups, 1u, 1u,
                             &pc, sizeof(pc));
}
