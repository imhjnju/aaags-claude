#include "vulkan/rasterize_backward_eval3d_pass.h"

#include "rasterize_backward_eval3d_spv.h"
#include "rasterize_backward_eval3d_replay_spv.h"

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
    replay_shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(rasterize_backward_eval3d_replay_spv),
        static_cast<std::size_t>(rasterize_backward_eval3d_replay_spv_len));

    std::vector<VkDescriptorType> binding_types(
        rasterize_backward_eval3d_bind::BINDING_COUNT,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    binding_types[rasterize_backward_eval3d_bind::BACKWARD_UBO] =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *shader_, binding_types, 0u, 4u);
    replay_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *replay_shader_, binding_types, 0u, 4u);
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
    replay_descriptor_set_ = replay_pipeline_->allocate_empty_descriptor_set();
}

RasterizeBackwardEval3DPass::~RasterizeBackwardEval3DPass() = default;

void RasterizeBackwardEval3DPass::bind_buffers(const Buffers& b, VkBuffer ubo) {
    auto bind = [&](VulkanComputePipeline& pipeline, VkDescriptorSet set) {
        pipeline.update_ssbo(set, rasterize_backward_eval3d_bind::TILE_RANGES,      b.tile_ranges);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_bind::VALUES_SORTED,    b.values_sorted);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_bind::GAUSS2SCREEN,     b.gauss2screen);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_bind::CONIC_OPACITY,    b.conic_opacity);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_bind::COLORS,           b.colors);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_bind::T_FINAL,          b.T_final);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_bind::N_CONTRIB,        b.n_contrib);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_bind::RENDERED_IMAGE,   b.rendered_image);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_bind::DL_DPIXELS,       b.dL_dpixels);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_bind::DL_DGAUSS2SCREEN, b.dL_dgauss2screen);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_bind::DL_DOPACITY,      b.dL_dopacity);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_bind::DL_DCOLORS,       b.dL_dcolors);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_bind::REPLAY_ORDER_OFFSETS, b.replay_order_offsets);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_bind::REPLAY_ORDER_GIDS,    b.replay_order_gids);
        pipeline.update_ubo(set,
                            rasterize_backward_eval3d_bind::BACKWARD_UBO,
                            ubo,
                            sizeof(RasterizeBackwardUBO));
    };
    bind(*pipeline_, descriptor_set_);
    bind(*replay_pipeline_, replay_descriptor_set_);
}

void RasterizeBackwardEval3DPass::dispatch_sync(uint32_t num_tiles_x,
                                                uint32_t num_tiles_y,
                                                bool use_replay_order) {
    VkDescriptorSet set = use_replay_order ? replay_descriptor_set_ : descriptor_set_;
    VulkanComputePipeline* pipeline = use_replay_order ? replay_pipeline_.get() : pipeline_.get();
    if (set == VK_NULL_HANDLE)
        throw std::runtime_error(
            "RasterizeBackwardEval3DPass::dispatch_sync called before bind_buffers()");
    if (num_tiles_x == 0u || num_tiles_y == 0u) return;
    pipeline->dispatch_sync(set, num_tiles_x, num_tiles_y, 1u, nullptr, 0u);
}

void RasterizeBackwardEval3DPass::record(VkCommandBuffer cmd,
                                          uint32_t num_tiles_x,
                                          uint32_t num_tiles_y,
                                          bool use_replay_order) {
    VkDescriptorSet set = use_replay_order ? replay_descriptor_set_ : descriptor_set_;
    VulkanComputePipeline* pipeline = use_replay_order ? replay_pipeline_.get() : pipeline_.get();
    if (set == VK_NULL_HANDLE)
        throw std::runtime_error(
            "RasterizeBackwardEval3DPass::record called before bind_buffers()");
    if (num_tiles_x == 0u || num_tiles_y == 0u) return;
    pipeline->record(cmd, set, num_tiles_x, num_tiles_y, 1u, nullptr, 0u);
}
