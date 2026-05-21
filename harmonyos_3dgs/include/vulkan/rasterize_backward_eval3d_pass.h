#pragma once

#include "vulkan/backward_bindings.h"
#include "vulkan/vk_context.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/vk_shader.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

class RasterizeBackwardEval3DPass {
public:
    struct Buffers {
        VkBuffer tile_ranges;
        VkBuffer values_sorted;
        VkBuffer gauss2screen;
        VkBuffer conic_opacity;
        VkBuffer colors;
        VkBuffer T_final;
        VkBuffer n_contrib;
        VkBuffer rendered_image; // Compatibility slot; shader no longer reads it.
        VkBuffer dL_dpixels;
        VkBuffer dL_dgauss2screen;
        VkBuffer dL_dopacity;
        VkBuffer dL_dcolors;
        VkBuffer replay_order_offsets;
        VkBuffer replay_order_gids;
    };

    struct FusedReplayBuffers {
        VkBuffer gauss2screen;
        VkBuffer conic_opacity;
        VkBuffer colors;
        VkBuffer T_final;
        VkBuffer n_contrib;
        VkBuffer dL_dpixels;
        VkBuffer dL_dopacity;
        VkBuffer dL_dcolors;
        VkBuffer replay_order_offsets;
        VkBuffer replay_order_gids;
        VkBuffer positions;
        VkBuffer scales;
        VkBuffer rotations;
        VkBuffer raw_rotations;
        VkBuffer filter_3D;
        VkBuffer d_means3D;
        VkBuffer d_scales;
        VkBuffer d_rotations;
    };

    struct FusedReplayTangentBuffers {
        VkBuffer gauss2screen;
        VkBuffer conic_opacity;
        VkBuffer colors;
        VkBuffer T_final;
        VkBuffer n_contrib;
        VkBuffer dL_dpixels;
        VkBuffer dL_dopacity;
        VkBuffer dL_dcolors;
        VkBuffer replay_order_offsets;
        VkBuffer replay_order_gids;
        VkBuffer positions;
        VkBuffer scales;
        VkBuffer rotations;
        VkBuffer raw_rotations;
        VkBuffer filter_3D;
        VkBuffer d_means3D;
        VkBuffer d_scales;
        VkBuffer d_rot_tangent;
        VkBuffer d_rotations;
    };

    struct FusedReplayTangentHotBuffers {
        FusedReplayTangentBuffers base;
        VkBuffer hot_lookup;
        VkBuffer hot_gids;
        VkBuffer hot_geom_scratch;
    };

    explicit RasterizeBackwardEval3DPass(VulkanContext& ctx);
    ~RasterizeBackwardEval3DPass();

    RasterizeBackwardEval3DPass(const RasterizeBackwardEval3DPass&) = delete;
    RasterizeBackwardEval3DPass& operator=(const RasterizeBackwardEval3DPass&) = delete;

    void bind_buffers(const Buffers& b, VkBuffer ubo);
    void bind_fused_replay_buffers(const FusedReplayBuffers& b, VkBuffer fused_ubo);
    void bind_fused_replay_tangent_buffers(const FusedReplayTangentBuffers& b, VkBuffer fused_ubo);
    void bind_fused_replay_tangent_hot_buffers(const FusedReplayTangentHotBuffers& b, VkBuffer fused_ubo, VkBuffer hot_ubo);
    void dispatch_sync(uint32_t num_tiles_x, uint32_t num_tiles_y, bool use_replay_order);
    void record(VkCommandBuffer cmd, uint32_t num_tiles_x, uint32_t num_tiles_y, bool use_replay_order);
    void record_fused_replay(VkCommandBuffer cmd, uint32_t num_tiles_x, uint32_t num_tiles_y);
    void record_fused_replay_tangent(VkCommandBuffer cmd, uint32_t num_tiles_x, uint32_t num_tiles_y, uint32_t num_gaussians);
    void record_fused_replay_tangent_subgroup(VkCommandBuffer cmd, uint32_t num_tiles_x, uint32_t num_tiles_y, uint32_t num_gaussians);
    void record_fused_replay_tangent_hot(VkCommandBuffer cmd, uint32_t num_tiles_x, uint32_t num_tiles_y, uint32_t num_gaussians, uint32_t hot_count);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader>          shader_;
    std::unique_ptr<VulkanShader>          replay_shader_;
    std::unique_ptr<VulkanShader>          fused_replay_shader_;
    std::unique_ptr<VulkanShader>          fused_replay_tangent_shader_;
    std::unique_ptr<VulkanShader>          fused_replay_tangent_subgroup_shader_;
    std::unique_ptr<VulkanShader>          fused_replay_tangent_hot_shader_;
    std::unique_ptr<VulkanShader>          hot_geom_reduce_shader_;
    std::unique_ptr<VulkanShader>          tangent_reduce_shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    std::unique_ptr<VulkanComputePipeline> replay_pipeline_;
    std::unique_ptr<VulkanComputePipeline> fused_replay_pipeline_;
    std::unique_ptr<VulkanComputePipeline> fused_replay_tangent_pipeline_;
    std::unique_ptr<VulkanComputePipeline> fused_replay_tangent_subgroup_pipeline_;
    std::unique_ptr<VulkanComputePipeline> fused_replay_tangent_hot_pipeline_;
    std::unique_ptr<VulkanComputePipeline> hot_geom_reduce_pipeline_;
    std::unique_ptr<VulkanComputePipeline> tangent_reduce_pipeline_;
    VkDescriptorSet                        descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet                        replay_descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet                        fused_replay_descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet                        fused_replay_tangent_descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet                        fused_replay_tangent_subgroup_descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet                        fused_replay_tangent_hot_descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet                        hot_geom_reduce_descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet                        tangent_reduce_descriptor_set_ = VK_NULL_HANDLE;
};
