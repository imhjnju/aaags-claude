#pragma once

#include "vulkan/backward_bindings.h"
#include "vulkan/vk_context.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/vk_shader.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

class PreprocessBackwardEval3DPass {
public:
    struct Buffers {
        VkBuffer positions;
        VkBuffer radii;
        VkBuffer sh_coeffs;
        VkBuffer scales;
        VkBuffer rotations;
        VkBuffer d_rgb;
        VkBuffer d_opacity;
        VkBuffer d_gauss2screen;
        VkBuffer d_means3D;
        VkBuffer d_sh;
        VkBuffer d_scales;
        VkBuffer d_rotations;
        VkBuffer opacities;
        VkBuffer d_raw_opacities;
        VkBuffer raw_rotations;
        VkBuffer filter_3D;
    };

    explicit PreprocessBackwardEval3DPass(VulkanContext& ctx,
                                          uint32_t spec_proper_ewa = 0u);
    ~PreprocessBackwardEval3DPass();

    PreprocessBackwardEval3DPass(const PreprocessBackwardEval3DPass&) = delete;
    PreprocessBackwardEval3DPass& operator=(const PreprocessBackwardEval3DPass&) = delete;

    void bind_buffers(const Buffers& b, VkBuffer ubo);
    void dispatch_sync(uint32_t num_gaussians, bool skip_geometry = false);
    void record(VkCommandBuffer cmd, uint32_t num_gaussians, bool skip_geometry = false);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader> shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    std::unique_ptr<VulkanComputePipeline> skip_geometry_pipeline_;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet skip_geometry_descriptor_set_ = VK_NULL_HANDLE;
};
