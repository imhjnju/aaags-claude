#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/vk_shader.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

namespace position_noise_bind {
constexpr uint32_t RAW_POSITIONS = 0;
constexpr uint32_t RAW_SCALES = 1;
constexpr uint32_t RAW_ROTATIONS = 2;
constexpr uint32_t RAW_OPACITIES = 3;
}

struct PositionNoisePushConstants {
    uint32_t N = 0;
    uint32_t step = 0;
    float noise_lr = 0.0f;
    float pos_lr = 0.0f;
};
static_assert(sizeof(PositionNoisePushConstants) == 16, "PositionNoisePushConstants must be 16 bytes");

// Uses deterministic shader-local normal samples; preserves noise distribution,
// opacity gating, and covariance scaling, not sample parity with CPU std::mt19937.
class PositionNoisePass {
public:
    explicit PositionNoisePass(VulkanContext& ctx);
    ~PositionNoisePass() = default;

    PositionNoisePass(const PositionNoisePass&) = delete;
    PositionNoisePass& operator=(const PositionNoisePass&) = delete;

    void bind_buffers(VkBuffer raw_positions,
                      VkBuffer raw_scales,
                      VkBuffer raw_rotations,
                      VkBuffer raw_opacities);
    void record(VkCommandBuffer cmd, uint32_t N, uint32_t step, float noise_lr, float pos_lr);
    void dispatch_sync(uint32_t N, uint32_t step, float noise_lr, float pos_lr);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader> shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    bool buffers_bound_ = false;
};
