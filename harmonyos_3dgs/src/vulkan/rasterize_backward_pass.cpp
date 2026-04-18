// SP-3 T22: RasterizeBackwardPass — wraps rasterize_backward.comp.
//
// Bindings (mirroring rasterize_backward_bind::):
//   0..7  read-only SSBOs: tile_ranges, values_sorted, means2D,
//                           conic_opacity, colors, T_final, n_contrib, dL_dpixels
//   8..11 read-write SSBOs: dL_dmeans2D, dL_dconics, dL_dopacity, dL_dcolors
//   12    uniform buffer:   RasterizeBackwardUBO (32 bytes)
//
// Dispatch: one 16x16 workgroup per tile. Grid = (num_tiles_x, num_tiles_y, 1).

#include "vulkan/rasterize_backward_pass.h"

// xxd-embedded SPIR-V (CMake produces rasterize_backward_spv.h in build dir).
#include "rasterize_backward_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

RasterizeBackwardPass::RasterizeBackwardPass(VulkanContext& ctx)
    : ctx_(ctx) {
    // --- 1. Load SPIR-V from embedded bytes --------------------------------
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(rasterize_backward_spv),
        static_cast<std::size_t>(rasterize_backward_spv_len));

    // --- 2. Descriptor layout: 12 SSBOs (bindings 0..11) + 1 UBO (binding 12) ---
    // 13 bindings total; binding 12 is UNIFORM_BUFFER.
    std::vector<VkDescriptorType> binding_types(13,
                                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    binding_types[rasterize_backward_bind::BACKWARD_UBO] =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_,
        *shader_,
        binding_types,
        /*push_constant_bytes=*/0u,
        /*max_descriptor_sets=*/4u);

    // --- 3. Allocate the descriptor set once (update in-place per dispatch) ---
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

RasterizeBackwardPass::~RasterizeBackwardPass() = default;

void RasterizeBackwardPass::bind_buffers(const Buffers& b, VkBuffer ubo) {
    // SSBOs 0..11.
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_bind::TILE_RANGES,   b.tile_ranges);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_bind::VALUES_SORTED, b.values_sorted);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_bind::MEANS2D,       b.means2D);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_bind::CONIC_OPACITY, b.conic_opacity);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_bind::COLORS,        b.colors);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_bind::T_FINAL,       b.T_final);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_bind::N_CONTRIB,     b.n_contrib);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_bind::DL_DPIXELS,    b.dL_dpixels);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_bind::DL_DMEANS2D,   b.dL_dmeans2D);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_bind::DL_DCONICS,    b.dL_dconics);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_bind::DL_DOPACITY,   b.dL_dopacity);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_bind::DL_DCOLORS,    b.dL_dcolors);

    // UBO binding 12 (RasterizeBackwardUBO, 32 bytes).
    pipeline_->update_ubo(descriptor_set_,
                          rasterize_backward_bind::BACKWARD_UBO,
                          ubo,
                          sizeof(RasterizeBackwardUBO));
}

void RasterizeBackwardPass::dispatch_sync(uint32_t num_tiles_x,
                                          uint32_t num_tiles_y) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "RasterizeBackwardPass::dispatch_sync called before bind_buffers()");
    if (num_tiles_x == 0u || num_tiles_y == 0u) return;

    // No push constants — W/H/num_tiles_x/bg_color are in the UBO.
    pipeline_->dispatch_sync(descriptor_set_,
                             num_tiles_x, num_tiles_y, 1u,
                             nullptr, 0u);
}
