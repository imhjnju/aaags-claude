#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/vk_shader.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

namespace regularization_bind {
constexpr uint32_t ACTIVE_OPACITIES = 0;
constexpr uint32_t ACTIVE_SCALES = 1;
constexpr uint32_t D_RAW_OPACITIES = 2;
constexpr uint32_t D_RAW_SCALES = 3;
}

struct RegularizationPushConstants {
    uint32_t N = 0;
    float opacity_reg = 0.0f;
    float scale_reg = 0.0f;
    float inv_N = 0.0f;
};
static_assert(sizeof(RegularizationPushConstants) == 16, "RegularizationPushConstants must be 16 bytes");

class RegularizationPass {
public:
    explicit RegularizationPass(VulkanContext& ctx);
    ~RegularizationPass() = default;

    RegularizationPass(const RegularizationPass&) = delete;
    RegularizationPass& operator=(const RegularizationPass&) = delete;

    void bind_buffers(VkBuffer active_opacities,
                      VkBuffer active_scales,
                      VkBuffer d_raw_opacities,
                      VkBuffer d_raw_scales);
    void record(VkCommandBuffer cmd,
                uint32_t N,
                float opacity_reg,
                float scale_reg);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader> shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
};
