#include "vulkan/rasterize_backward_eval3d_pass.h"

#include "rasterize_backward_eval3d_spv.h"
#include "rasterize_backward_eval3d_replay_spv.h"
#include "rasterize_backward_eval3d_replay_fused_spv.h"
#include "rasterize_backward_eval3d_replay_fused_tangent_spv.h"
#include "rasterize_backward_eval3d_replay_fused_tangent_subgroup_spv.h"
#include "rasterize_backward_eval3d_replay_fused_hot_spv.h"
#include "rasterize_backward_eval3d_hot_geom_reduce_spv.h"
#include "rasterize_backward_eval3d_tangent_rot_reduce_spv.h"

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
    fused_replay_shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(rasterize_backward_eval3d_replay_fused_spv),
        static_cast<std::size_t>(rasterize_backward_eval3d_replay_fused_spv_len));
    fused_replay_tangent_shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(rasterize_backward_eval3d_replay_fused_tangent_spv),
        static_cast<std::size_t>(rasterize_backward_eval3d_replay_fused_tangent_spv_len));
    fused_replay_tangent_subgroup_shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(rasterize_backward_eval3d_replay_fused_tangent_subgroup_spv),
        static_cast<std::size_t>(rasterize_backward_eval3d_replay_fused_tangent_subgroup_spv_len));
    fused_replay_tangent_hot_shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(rasterize_backward_eval3d_replay_fused_hot_spv),
        static_cast<std::size_t>(rasterize_backward_eval3d_replay_fused_hot_spv_len));
    hot_geom_reduce_shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(rasterize_backward_eval3d_hot_geom_reduce_spv),
        static_cast<std::size_t>(rasterize_backward_eval3d_hot_geom_reduce_spv_len));
    tangent_reduce_shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(rasterize_backward_eval3d_tangent_rot_reduce_spv),
        static_cast<std::size_t>(rasterize_backward_eval3d_tangent_rot_reduce_spv_len));

    std::vector<VkDescriptorType> binding_types(
        rasterize_backward_eval3d_bind::BINDING_COUNT,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    binding_types[rasterize_backward_eval3d_bind::BACKWARD_UBO] =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    std::vector<VkDescriptorType> fused_binding_types(
        rasterize_backward_eval3d_fused_bind::BINDING_COUNT,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    fused_binding_types[rasterize_backward_eval3d_fused_bind::FUSED_UBO] =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    std::vector<VkDescriptorType> fused_tangent_binding_types(
        rasterize_backward_eval3d_fused_tangent_bind::BINDING_COUNT,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    fused_tangent_binding_types[rasterize_backward_eval3d_fused_tangent_bind::FUSED_UBO] =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    std::vector<VkDescriptorType> fused_tangent_hot_binding_types(
        rasterize_backward_eval3d_fused_tangent_hot_bind::BINDING_COUNT,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    fused_tangent_hot_binding_types[rasterize_backward_eval3d_fused_tangent_hot_bind::FUSED_UBO] =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    fused_tangent_hot_binding_types[rasterize_backward_eval3d_fused_tangent_hot_bind::HOT_UBO] =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    std::vector<VkDescriptorType> hot_reduce_binding_types(
        rasterize_backward_eval3d_hot_geom_reduce_bind::BINDING_COUNT,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    hot_reduce_binding_types[rasterize_backward_eval3d_hot_geom_reduce_bind::HOT_UBO] =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    std::vector<VkDescriptorType> tangent_reduce_binding_types(
        rasterize_backward_eval3d_tangent_reduce_bind::BINDING_COUNT,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *shader_, binding_types, 0u, 4u);
    replay_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *replay_shader_, binding_types, 0u, 4u);
    fused_replay_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *fused_replay_shader_, fused_binding_types, 0u, 4u);
    fused_replay_tangent_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *fused_replay_tangent_shader_, fused_tangent_binding_types, 0u, 4u);
    fused_replay_tangent_subgroup_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *fused_replay_tangent_subgroup_shader_, fused_tangent_binding_types, 0u, 4u);
    fused_replay_tangent_hot_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *fused_replay_tangent_hot_shader_, fused_tangent_hot_binding_types, 0u, 4u);
    hot_geom_reduce_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *hot_geom_reduce_shader_, hot_reduce_binding_types, 0u, 4u);
    tangent_reduce_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *tangent_reduce_shader_, tangent_reduce_binding_types,
        sizeof(RasterizeBackwardEval3DTangentReducePC), 4u);
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
    replay_descriptor_set_ = replay_pipeline_->allocate_empty_descriptor_set();
    fused_replay_descriptor_set_ = fused_replay_pipeline_->allocate_empty_descriptor_set();
    fused_replay_tangent_descriptor_set_ = fused_replay_tangent_pipeline_->allocate_empty_descriptor_set();
    fused_replay_tangent_subgroup_descriptor_set_ = fused_replay_tangent_subgroup_pipeline_->allocate_empty_descriptor_set();
    fused_replay_tangent_hot_descriptor_set_ = fused_replay_tangent_hot_pipeline_->allocate_empty_descriptor_set();
    hot_geom_reduce_descriptor_set_ = hot_geom_reduce_pipeline_->allocate_empty_descriptor_set();
    tangent_reduce_descriptor_set_ = tangent_reduce_pipeline_->allocate_empty_descriptor_set();
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

void RasterizeBackwardEval3DPass::bind_fused_replay_buffers(const FusedReplayBuffers& b,
                                                              VkBuffer fused_ubo) {
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::GAUSS2SCREEN, b.gauss2screen);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::CONIC_OPACITY, b.conic_opacity);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::COLORS, b.colors);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::T_FINAL, b.T_final);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::N_CONTRIB, b.n_contrib);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::DL_DPIXELS, b.dL_dpixels);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::DL_DOPACITY, b.dL_dopacity);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::DL_DCOLORS, b.dL_dcolors);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::REPLAY_ORDER_OFFSETS, b.replay_order_offsets);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::REPLAY_ORDER_GIDS, b.replay_order_gids);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::POSITIONS, b.positions);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::SCALES, b.scales);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::ROTATIONS, b.rotations);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::RAW_ROTATIONS, b.raw_rotations);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::FILTER_3D, b.filter_3D);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::D_MEANS3D, b.d_means3D);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::D_SCALES, b.d_scales);
    fused_replay_pipeline_->update_ssbo(fused_replay_descriptor_set_, rasterize_backward_eval3d_fused_bind::D_ROTATIONS, b.d_rotations);
    fused_replay_pipeline_->update_ubo(fused_replay_descriptor_set_,
                                       rasterize_backward_eval3d_fused_bind::FUSED_UBO,
                                       fused_ubo,
                                       sizeof(RasterizeBackwardEval3DFusedUBO));
}

void RasterizeBackwardEval3DPass::bind_fused_replay_tangent_buffers(const FusedReplayTangentBuffers& b,
                                                                      VkBuffer fused_ubo) {
    auto bind_fused = [&](VulkanComputePipeline& pipeline, VkDescriptorSet set) {
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::GAUSS2SCREEN, b.gauss2screen);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::CONIC_OPACITY, b.conic_opacity);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::COLORS, b.colors);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::T_FINAL, b.T_final);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::N_CONTRIB, b.n_contrib);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::DL_DPIXELS, b.dL_dpixels);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::DL_DOPACITY, b.dL_dopacity);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::DL_DCOLORS, b.dL_dcolors);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::REPLAY_ORDER_OFFSETS, b.replay_order_offsets);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::REPLAY_ORDER_GIDS, b.replay_order_gids);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::POSITIONS, b.positions);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::SCALES, b.scales);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::ROTATIONS, b.rotations);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::RAW_ROTATIONS, b.raw_rotations);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::FILTER_3D, b.filter_3D);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::D_MEANS3D, b.d_means3D);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::D_SCALES, b.d_scales);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_bind::D_ROT_TANGENT, b.d_rot_tangent);
        pipeline.update_ubo(set,
                            rasterize_backward_eval3d_fused_tangent_bind::FUSED_UBO,
                            fused_ubo,
                            sizeof(RasterizeBackwardEval3DFusedUBO));
    };
    bind_fused(*fused_replay_tangent_pipeline_, fused_replay_tangent_descriptor_set_);
    bind_fused(*fused_replay_tangent_subgroup_pipeline_, fused_replay_tangent_subgroup_descriptor_set_);

    tangent_reduce_pipeline_->update_ssbo(tangent_reduce_descriptor_set_, rasterize_backward_eval3d_tangent_reduce_bind::ROTATIONS, b.rotations);
    tangent_reduce_pipeline_->update_ssbo(tangent_reduce_descriptor_set_, rasterize_backward_eval3d_tangent_reduce_bind::D_ROT_TANGENT, b.d_rot_tangent);
    tangent_reduce_pipeline_->update_ssbo(tangent_reduce_descriptor_set_, rasterize_backward_eval3d_tangent_reduce_bind::D_ROTATIONS, b.d_rotations);
}

void RasterizeBackwardEval3DPass::bind_fused_replay_tangent_hot_buffers(
    const FusedReplayTangentHotBuffers& b,
    VkBuffer fused_ubo,
    VkBuffer hot_ubo) {
    auto bind_hot = [&](VulkanComputePipeline& pipeline, VkDescriptorSet set) {
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::GAUSS2SCREEN, b.base.gauss2screen);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::CONIC_OPACITY, b.base.conic_opacity);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::COLORS, b.base.colors);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::T_FINAL, b.base.T_final);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::N_CONTRIB, b.base.n_contrib);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::DL_DPIXELS, b.base.dL_dpixels);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::DL_DOPACITY, b.base.dL_dopacity);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::DL_DCOLORS, b.base.dL_dcolors);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::REPLAY_ORDER_OFFSETS, b.base.replay_order_offsets);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::REPLAY_ORDER_GIDS, b.base.replay_order_gids);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::POSITIONS, b.base.positions);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::SCALES, b.base.scales);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::ROTATIONS, b.base.rotations);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::RAW_ROTATIONS, b.base.raw_rotations);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::FILTER_3D, b.base.filter_3D);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::D_MEANS3D, b.base.d_means3D);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::D_SCALES, b.base.d_scales);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::D_ROT_TANGENT, b.base.d_rot_tangent);
        pipeline.update_ubo(set,
                            rasterize_backward_eval3d_fused_tangent_hot_bind::FUSED_UBO,
                            fused_ubo,
                            sizeof(RasterizeBackwardEval3DFusedUBO));
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::HOT_LOOKUP, b.hot_lookup);
        pipeline.update_ssbo(set, rasterize_backward_eval3d_fused_tangent_hot_bind::HOT_GEOM_SCRATCH, b.hot_geom_scratch);
        pipeline.update_ubo(set,
                            rasterize_backward_eval3d_fused_tangent_hot_bind::HOT_UBO,
                            hot_ubo,
                            sizeof(RasterizeBackwardEval3DHotUBO));
    };
    bind_hot(*fused_replay_tangent_hot_pipeline_, fused_replay_tangent_hot_descriptor_set_);

    hot_geom_reduce_pipeline_->update_ssbo(hot_geom_reduce_descriptor_set_, rasterize_backward_eval3d_hot_geom_reduce_bind::HOT_GIDS, b.hot_gids);
    hot_geom_reduce_pipeline_->update_ssbo(hot_geom_reduce_descriptor_set_, rasterize_backward_eval3d_hot_geom_reduce_bind::HOT_GEOM_SCRATCH, b.hot_geom_scratch);
    hot_geom_reduce_pipeline_->update_ssbo(hot_geom_reduce_descriptor_set_, rasterize_backward_eval3d_hot_geom_reduce_bind::D_MEANS3D, b.base.d_means3D);
    hot_geom_reduce_pipeline_->update_ssbo(hot_geom_reduce_descriptor_set_, rasterize_backward_eval3d_hot_geom_reduce_bind::D_SCALES, b.base.d_scales);
    hot_geom_reduce_pipeline_->update_ssbo(hot_geom_reduce_descriptor_set_, rasterize_backward_eval3d_hot_geom_reduce_bind::D_ROT_TANGENT, b.base.d_rot_tangent);
    hot_geom_reduce_pipeline_->update_ubo(hot_geom_reduce_descriptor_set_,
                                          rasterize_backward_eval3d_hot_geom_reduce_bind::HOT_UBO,
                                          hot_ubo,
                                          sizeof(RasterizeBackwardEval3DHotUBO));

    tangent_reduce_pipeline_->update_ssbo(tangent_reduce_descriptor_set_, rasterize_backward_eval3d_tangent_reduce_bind::ROTATIONS, b.base.rotations);
    tangent_reduce_pipeline_->update_ssbo(tangent_reduce_descriptor_set_, rasterize_backward_eval3d_tangent_reduce_bind::D_ROT_TANGENT, b.base.d_rot_tangent);
    tangent_reduce_pipeline_->update_ssbo(tangent_reduce_descriptor_set_, rasterize_backward_eval3d_tangent_reduce_bind::D_ROTATIONS, b.base.d_rotations);
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

void RasterizeBackwardEval3DPass::record_fused_replay(VkCommandBuffer cmd,
                                                       uint32_t num_tiles_x,
                                                       uint32_t num_tiles_y) {
    if (fused_replay_descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "RasterizeBackwardEval3DPass::record_fused_replay called before bind_fused_replay_buffers()");
    if (num_tiles_x == 0u || num_tiles_y == 0u) return;
    fused_replay_pipeline_->record(cmd, fused_replay_descriptor_set_, num_tiles_x, num_tiles_y, 1u, nullptr, 0u);
}

void RasterizeBackwardEval3DPass::record_fused_replay_tangent(VkCommandBuffer cmd,
                                                               uint32_t num_tiles_x,
                                                               uint32_t num_tiles_y,
                                                               uint32_t num_gaussians) {
    if (fused_replay_tangent_descriptor_set_ == VK_NULL_HANDLE || tangent_reduce_descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "RasterizeBackwardEval3DPass::record_fused_replay_tangent called before bind_fused_replay_tangent_buffers()");
    if (num_tiles_x == 0u || num_tiles_y == 0u || num_gaussians == 0u) return;

    fused_replay_tangent_pipeline_->record(cmd, fused_replay_tangent_descriptor_set_, num_tiles_x, num_tiles_y, 1u, nullptr, 0u);
    insert_compute_barrier(cmd);

    RasterizeBackwardEval3DTangentReducePC pc{};
    pc.N = num_gaussians;
    const uint32_t groups = (num_gaussians + 255u) / 256u;
    tangent_reduce_pipeline_->record(cmd, tangent_reduce_descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));
}

void RasterizeBackwardEval3DPass::record_fused_replay_tangent_subgroup(VkCommandBuffer cmd,
                                                                        uint32_t num_tiles_x,
                                                                        uint32_t num_tiles_y,
                                                                        uint32_t num_gaussians) {
    if (fused_replay_tangent_subgroup_descriptor_set_ == VK_NULL_HANDLE || tangent_reduce_descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "RasterizeBackwardEval3DPass::record_fused_replay_tangent_subgroup called before bind_fused_replay_tangent_buffers()");
    if (num_tiles_x == 0u || num_tiles_y == 0u || num_gaussians == 0u) return;

    fused_replay_tangent_subgroup_pipeline_->record(cmd, fused_replay_tangent_subgroup_descriptor_set_, num_tiles_x, num_tiles_y, 1u, nullptr, 0u);
    insert_compute_barrier(cmd);

    RasterizeBackwardEval3DTangentReducePC pc{};
    pc.N = num_gaussians;
    const uint32_t groups = (num_gaussians + 255u) / 256u;
    tangent_reduce_pipeline_->record(cmd, tangent_reduce_descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));
}

void RasterizeBackwardEval3DPass::record_fused_replay_tangent_hot(VkCommandBuffer cmd,
                                                                   uint32_t num_tiles_x,
                                                                   uint32_t num_tiles_y,
                                                                   uint32_t num_gaussians,
                                                                   uint32_t hot_count) {
    if (fused_replay_tangent_hot_descriptor_set_ == VK_NULL_HANDLE ||
        hot_geom_reduce_descriptor_set_ == VK_NULL_HANDLE ||
        tangent_reduce_descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "RasterizeBackwardEval3DPass::record_fused_replay_tangent_hot called before bind_fused_replay_tangent_hot_buffers()");
    if (num_tiles_x == 0u || num_tiles_y == 0u || num_gaussians == 0u) return;

    fused_replay_tangent_hot_pipeline_->record(cmd, fused_replay_tangent_hot_descriptor_set_, num_tiles_x, num_tiles_y, 1u, nullptr, 0u);
    insert_compute_barrier(cmd);

    if (hot_count > 0u) {
        const uint32_t hot_groups = (hot_count * 9u + 255u) / 256u;
        hot_geom_reduce_pipeline_->record(cmd, hot_geom_reduce_descriptor_set_, hot_groups, 1u, 1u, nullptr, 0u);
        insert_compute_barrier(cmd);
    }

    RasterizeBackwardEval3DTangentReducePC pc{};
    pc.N = num_gaussians;
    const uint32_t groups = (num_gaussians + 255u) / 256u;
    tangent_reduce_pipeline_->record(cmd, tangent_reduce_descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));
}
