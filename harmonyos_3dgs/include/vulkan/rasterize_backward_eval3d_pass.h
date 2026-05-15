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

    explicit RasterizeBackwardEval3DPass(VulkanContext& ctx);
    ~RasterizeBackwardEval3DPass();

    RasterizeBackwardEval3DPass(const RasterizeBackwardEval3DPass&) = delete;
    RasterizeBackwardEval3DPass& operator=(const RasterizeBackwardEval3DPass&) = delete;

    void bind_buffers(const Buffers& b, VkBuffer ubo);
    void dispatch_sync(uint32_t num_tiles_x, uint32_t num_tiles_y, bool use_replay_order);
    void record(VkCommandBuffer cmd, uint32_t num_tiles_x, uint32_t num_tiles_y, bool use_replay_order);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader>          shader_;
    std::unique_ptr<VulkanShader>          replay_shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    std::unique_ptr<VulkanComputePipeline> replay_pipeline_;
    VkDescriptorSet                        descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet                        replay_descriptor_set_ = VK_NULL_HANDLE;
};
