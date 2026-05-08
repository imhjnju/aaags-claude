#include "vulkan/rasterize_backward_eval3d_pass.h"

#include "rasterize_backward_eval3d_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

RasterizeBackwardEval3DPass::RasterizeBackwardEval3DPass(VulkanContext& ctx)
    : ctx_(ctx) {
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(rasterize_backward_eval3d_spv),
        static_cast<std::size_t>(rasterize_backward_eval3d_spv_len));

    std::vector<VkDescriptorType> binding_types(
        rasterize_backward_eval3d_bind::BINDING_COUNT,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    binding_types[rasterize_backward_eval3d_bind::BACKWARD_UBO] =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *shader_, binding_types, 0u, 4u);
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

RasterizeBackwardEval3DPass::~RasterizeBackwardEval3DPass() = default;

void RasterizeBackwardEval3DPass::bind_buffers(const Buffers& b, VkBuffer ubo) {
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_eval3d_bind::TILE_RANGES,      b.tile_ranges);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_eval3d_bind::VALUES_SORTED,    b.values_sorted);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_eval3d_bind::GAUSS2SCREEN,     b.gauss2screen);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_eval3d_bind::CONIC_OPACITY,    b.conic_opacity);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_eval3d_bind::COLORS,           b.colors);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_eval3d_bind::T_FINAL,          b.T_final);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_eval3d_bind::N_CONTRIB,        b.n_contrib);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_eval3d_bind::RENDERED_IMAGE,   b.rendered_image);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_eval3d_bind::DL_DPIXELS,       b.dL_dpixels);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_eval3d_bind::DL_DGAUSS2SCREEN, b.dL_dgauss2screen);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_eval3d_bind::DL_DOPACITY,      b.dL_dopacity);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_eval3d_bind::DL_DCOLORS,       b.dL_dcolors);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_eval3d_bind::REPLAY_ORDER_OFFSETS, b.replay_order_offsets);
    pipeline_->update_ssbo(descriptor_set_, rasterize_backward_eval3d_bind::REPLAY_ORDER_GIDS,    b.replay_order_gids);
    pipeline_->update_ubo(descriptor_set_,
                          rasterize_backward_eval3d_bind::BACKWARD_UBO,
                          ubo,
                          sizeof(RasterizeBackwardUBO));
}

void RasterizeBackwardEval3DPass::dispatch_sync(uint32_t num_tiles_x,
                                                uint32_t num_tiles_y) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "RasterizeBackwardEval3DPass::dispatch_sync called before bind_buffers()");
    if (num_tiles_x == 0u || num_tiles_y == 0u) return;
    pipeline_->dispatch_sync(descriptor_set_, num_tiles_x, num_tiles_y, 1u, nullptr, 0u);
}

void RasterizeBackwardEval3DPass::record(VkCommandBuffer cmd,
                                          uint32_t num_tiles_x,
                                          uint32_t num_tiles_y) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "RasterizeBackwardEval3DPass::record called before bind_buffers()");
    if (num_tiles_x == 0u || num_tiles_y == 0u) return;
    pipeline_->record(cmd, descriptor_set_, num_tiles_x, num_tiles_y, 1u, nullptr, 0u);
}
